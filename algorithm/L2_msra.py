"""L2: MSRA-MARL (Hu et al., "Real-time coordination of human couriers and drones
for on-demand food-delivery platforms: A multi-stage risk-aware multi-agent
reinforcement learning framework", Transportation Research Part C 181, 2025),
adapted to launchpad drone dispatch.  Label: ``MSRA-Matching+IPPO``.

What MSRA-MARL is (paper Sections 3.2, 4):
  1. Two heterogeneous agent types, **couriers** (drop-off choice, IDDQN) and
     **drones** (relocation target l in L, IPPO with a shared policy and a local
     critic V(s_m); per-drone reward = bonus for the order it then serves on
     time minus its operating cost).
  2. **Order assignment is not learned** -- it is a dynamic bipartite matching
     solved with ``scipy.optimize.linear_sum_assignment`` every decision step.
     For drones at a launchpad (Eq. 22): utility
        U(m, o_l) = R_o + lambda_p * 1[on time] - C_m + V_pi_z(s_{m,t+1}),
     drone and order must be at the same launchpad; the drone delivery time is
     computed exactly ("precisely computed via drone routing").
  3. **Graph Coordinator** -- a spatiotemporal launchpad graph (node = launchpad,
     features = current / anticipated orders and drones) encoded by a two-layer
     mean-aggregation GNN, concatenated with an agent MLP, with two heads:
     P(deadline violated) (BCE) and delivery time (MSE); trained supervised
     online from completed orders. Its risk estimate enters the matching as a
     HARD constraint Risk <= G (G = 0.5) and the courier reward.
  4. **Multi-stage training**: pretraining with Risk := 0 and the coordinator
     frozen, then supervised coordinator training, then joint fine-tuning.

Mapping to this simulator (differences we do NOT hide):
  * There are no couriers: orders appear at launchpads. The courier agent, the
    courier matching (Eq. 21) and the courier-side risk reward (Eq. 48) have no
    counterpart and are dropped. What remains is exactly the drone side.
  * The drone RL action. In MSRA a drone that has just delivered chooses the
    launchpad it relocates to; here that choice is the **return launchpad l**
    committed at dispatch. So pi_z(l | s_m, o) is the learned policy, a pointer
    over the feasible return pads for the matched order (paper: independent
    PPO, shared policy, local critic) -- structure **IPPO**.
  * Batteries do not exist in MSRA. A dispatch needs one here, so the battery is
    a fixed rule: the lowest-charge battery that can fly (o, l) (the R3 rule).
  * Order->drone assignment per launchpad is the paper's bipartite matching:
    U(o) = v_o + lambda_p * 1[ETA <= deadline] - C_m * flight_min + gamma * V(s'),
    with s' the drone's state after choosing l* = pi_z(. | s_m, o); identical
    drones at one pad make the assignment the top-k orders by U. Idle drones
    that are not matched stay (fleet relocation is disabled for every method).
  * Graph Coordinator kept as specified (pad graph, 2-layer mean GNN + agent MLP,
    BCE + lambda_g MSE, online supervised from completed orders) and used as the
    matching hard constraint Risk(o, l*) <= G. Our lateness is deterministic at
    dispatch, so it learns the ETA-vs-deadline boundary; the mechanism is the
    paper's, its information content is honestly smaller here.
  * Multi-stage: the first ``gc_warmup_episodes`` episodes run with Risk := 0
    while the coordinator trains (paper stages 2-3), then the constraint is on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from algorithm.F2.menu import FactorMenu
from algorithm.dispatch.presets import MSRA_MATCHING
from algorithm.F2.policy import F2Policy
from algorithm.dispatch.presets import IPPO
from algorithm.F2.policy import _JointPlanDecision
from algorithm.L3_mapdp import PointerMenu, pointer_menu_from_factors, ORDER_TOKEN_DIM, PAD_TOKEN_DIM, ROUTE_EDGE_DIM
from algorithm.common.torch_learner_base import BaseTorchMAPPOLearner
from simulator.model.enums import OrderStatus


# --------------------------------------------------------------------------- #
# Networks: relocation (return-pad) policy pi_z, local critic, Graph Coordinator
# --------------------------------------------------------------------------- #
class _RelocationActor(nn.Module):
    """pi_z(l | s_m, o): score each feasible return pad for a (drone, order)."""

    def __init__(self, context_dim: int, hidden_dim: int):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(context_dim + ORDER_TOKEN_DIM + hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.k = nn.Sequential(nn.Linear(PAD_TOKEN_DIM + ROUTE_EDGE_DIM, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, context, order_tok, graph_emb, pads, edge, mask):
        # context [R,C], order_tok [R,9], graph_emb [R,H], pads [R,n_l,12], edge [R,n_l,4], mask [R,n_l]
        q = self.q(torch.cat([context, order_tok, graph_emb], -1)).unsqueeze(1).expand(-1, pads.shape[1], -1)
        k = self.k(torch.cat([pads, edge], -1))
        logits = self.score(torch.cat([q, k], -1)).squeeze(-1)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class _LocalCritic(nn.Module):
    """V(s_m): the drone's own observation (context, matched order, graph embedding)."""

    def __init__(self, context_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(context_dim + ORDER_TOKEN_DIM + hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, context, order_tok, graph_emb):
        return self.net(torch.cat([context, order_tok, graph_emb], -1)).squeeze(-1)


class GraphCoordinator(nn.Module):
    """Paper Section 4.2: two-layer mean-aggregation GNN over the launchpad graph
    (fully connected) + agent MLP -> [risk logit, delivery-time regression]."""

    def __init__(self, node_dim: int, agent_dim: int, hidden_dim: int):
        super().__init__()
        self.B1 = nn.Linear(node_dim, hidden_dim); self.W1 = nn.Linear(node_dim, hidden_dim, bias=False)
        self.B2 = nn.Linear(hidden_dim, hidden_dim); self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.agent = nn.Linear(agent_dim, hidden_dim)
        self.combine = nn.Linear(2 * hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 1)
        self.reg = nn.Linear(hidden_dim, 1)

    def encode_graph(self, nodes, node_mask):
        # nodes [R, P, d], node_mask [R, P]; fully connected -> neighbour mean = masked mean over the others
        m = node_mask.unsqueeze(-1).float()
        cnt = m.sum(1, keepdim=True).clamp_min(1.0)
        neigh1 = ((nodes * m).sum(1, keepdim=True) - nodes * m) / (cnt - 1).clamp_min(1.0)
        h1 = torch.relu(self.B1(nodes) + self.W1(neigh1)) * m
        neigh2 = ((h1).sum(1, keepdim=True) - h1) / (cnt - 1).clamp_min(1.0)
        h2 = torch.relu(self.B2(h1) + self.W2(neigh2)) * m
        return h2                                                          # [R, P, H]

    def forward(self, nodes, node_mask, ego_index, agent_feats):
        # ego_index [R]: the node of the launchpad the drone is at (h^2_{v_{n,t}} in Eq. 39)
        h2 = self.encode_graph(nodes, node_mask)
        r = torch.arange(h2.shape[0], device=h2.device)
        h = torch.relu(self.combine(torch.cat([torch.relu(self.agent(agent_feats)), h2[r, ego_index]], -1)))
        return self.cls(h).squeeze(-1), self.reg(h).squeeze(-1)

    def graph_embedding(self, nodes, node_mask):
        h2 = self.encode_graph(nodes, node_mask)
        m = node_mask.unsqueeze(-1).float()
        return (h2 * m).sum(1) / m.sum(1).clamp_min(1.0)                   # [R, H]


# --------------------------------------------------------------------------- #
# Per-drone sample / learner
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MSRASample:
    context: np.ndarray        # [C] drone / pad context
    order_tok: np.ndarray      # [9]
    pads: np.ndarray           # [n_l, 12]
    edge: np.ndarray           # [n_l, 4]
    pad_mask: np.ndarray       # [n_l]
    nodes: np.ndarray          # [P, C] launchpad graph (pad contexts)
    action: int                # chosen pad index
    log_prob: float
    value: float


class TorchMSRAIPPOLearner(BaseTorchMAPPOLearner):
    """IPPO relocation policy + local critic + supervised Graph Coordinator."""

    actor_context_label = "MSRA drone context"

    def __init__(self, *, candidate_dim: int, context_dim: int, gc_lambda_reg: float = 0.2, gc_lr: float = 1e-3, gc_batch: int = 256, gc_capacity: int = 5000, **kwargs):
        self.context_dim = int(context_dim)
        super().__init__(candidate_dim=candidate_dim, actor_context_dim=self.context_dim, **kwargs)
        H = self.hidden_dim
        self.actor = _RelocationActor(self.context_dim, H).to(self.device)
        self.critic = _LocalCritic(self.context_dim, H).to(self.device)
        self.coordinator = GraphCoordinator(self.context_dim, self.context_dim + ORDER_TOKEN_DIM + PAD_TOKEN_DIM + ROUTE_EDGE_DIM, H).to(self.device)
        self._initialize_optimizers()
        self.gc_optimizer = torch.optim.Adam(self.coordinator.parameters(), lr=float(gc_lr))
        self.gc_lambda_reg = float(gc_lambda_reg); self.gc_batch = int(gc_batch); self.gc_capacity = int(gc_capacity)
        self._gc_buffer: list[tuple] = []   # (nodes, ego, agent_feats, late_label, time_label)
        self._rollout: list[dict[str, Any]] = []

    # ---- batching helpers ----
    def _pad_batch(self, pads_list, edge_list, mask_list):
        R = len(pads_list); n_l = max(p.shape[0] for p in pads_list)
        pads = np.zeros((R, n_l, PAD_TOKEN_DIM), np.float32); edge = np.zeros((R, n_l, ROUTE_EDGE_DIM), np.float32); mask = np.zeros((R, n_l), bool)
        for r, (p, e, m) in enumerate(zip(pads_list, edge_list, mask_list)):
            pads[r, : p.shape[0]] = p; edge[r, : e.shape[0]] = e; mask[r, : m.shape[0]] = m
        t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=self.device)
        return t(pads, torch.float32), t(edge, torch.float32), t(mask, torch.bool)

    def _nodes_batch(self, nodes_list):
        R = len(nodes_list); P = max((len(n) for n in nodes_list), default=1) or 1
        nodes = np.zeros((R, P, self.context_dim), np.float32); mask = np.zeros((R, P), bool)
        for r, n in enumerate(nodes_list):
            if len(n): nodes[r, : len(n)] = n; mask[r, : len(n)] = True
        return torch.as_tensor(nodes, device=self.device), torch.as_tensor(mask, device=self.device)

    # ---- one pad, one step: score every order's best return pad ----
    @torch.no_grad()
    def plan_pad(self, pm: PointerMenu, context: np.ndarray, nodes: np.ndarray, ego: int, *, training: bool):
        """For every order at the pad: l* ~ pi_z(. | s, o), log pi, V(s), risk P(late | g, s, o, l*).
        Returns dict of arrays over orders (n_o)."""
        n_o = pm.n_o
        if n_o == 0:
            return None
        ctx = torch.as_tensor(np.repeat(context[None], n_o, 0), dtype=torch.float32, device=self.device)
        order_tok = torch.as_tensor(pm.orders, dtype=torch.float32, device=self.device)
        pads, edge, mask = self._pad_batch([pm.pads] * n_o, list(pm.edge), list(pm.ol_ok))
        nodes_t, nmask_t = self._nodes_batch([nodes] * n_o)
        g = self.coordinator.graph_embedding(nodes_t, nmask_t)
        logits = self.actor(ctx, order_tok, g, pads, edge, mask)
        dist = torch.distributions.Categorical(logits=logits)
        l = dist.sample() if (training or self.evaluation_action_mode == "stochastic") else logits.argmax(-1)
        has_route = mask.any(-1)
        value = self.critic(ctx, order_tok, g)
        r = torch.arange(n_o, device=self.device)
        agent_feats = torch.cat([ctx, order_tok, pads[r, l], edge[r, l]], -1)
        risk_logit, _ = self.coordinator(nodes_t, nmask_t, torch.full((n_o,), int(ego), device=self.device), agent_feats)
        return {
            "l": l.cpu().numpy(), "log_prob": dist.log_prob(l).cpu().numpy(), "value": value.cpu().numpy(),
            "risk": torch.sigmoid(risk_logit).cpu().numpy(), "has_route": has_route.cpu().numpy(),
        }

    # ---- rollout / PPO ----
    def _sample_rollout_fields(self, sample: MSRASample) -> dict[str, object]:
        return {"sample": sample}

    def update(self, epochs: int | None = None) -> dict[str, float]:
        gc_loss = self._update_coordinator()
        if not self._rollout:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "rollout_size": 0.0, "gc_loss": gc_loss}
        n = len(self._rollout)
        adv = np.asarray([row["advantage"] for row in self._rollout], np.float32)
        if n > 1 and adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        a_l, c_l, ents = [], [], []
        for _ in range(self.ppo_epochs if epochs is None else max(1, int(epochs))):
            perm = self._rng.permutation(n)
            for s0 in range(0, n, self.batch_size):
                idx = perm[s0: s0 + self.batch_size]; rows = [self._rollout[int(i)] for i in idx]
                S = [r["sample"] for r in rows]
                ctx = torch.as_tensor(np.stack([s.context for s in S]), dtype=torch.float32, device=self.device)
                otk = torch.as_tensor(np.stack([s.order_tok for s in S]), dtype=torch.float32, device=self.device)
                pads, edge, mask = self._pad_batch([s.pads for s in S], [s.edge for s in S], [s.pad_mask for s in S])
                nodes_t, nmask_t = self._nodes_batch([s.nodes for s in S])
                g = self.coordinator.graph_embedding(nodes_t, nmask_t).detach()
                logits = self.actor(ctx, otk, g, pads, edge, mask)
                dist = torch.distributions.Categorical(logits=logits)
                act = torch.as_tensor([r["action"] for r in rows], dtype=torch.long, device=self.device)
                logp = dist.log_prob(act); old = torch.as_tensor([r["old_log_prob"] for r in rows], dtype=torch.float32, device=self.device)
                A = torch.as_tensor(adv[idx], dtype=torch.float32, device=self.device)
                ratio = torch.exp(logp - old)
                obj = torch.minimum(ratio * A, torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * A).mean()
                ent = dist.entropy().mean()
                actor_loss = -obj - self.entropy_coefficient * ent
                self.actor_optimizer.zero_grad(); actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm); self.actor_optimizer.step()
                ret = torch.as_tensor([r["return"] for r in rows], dtype=torch.float32, device=self.device)
                critic_loss = self.value_coefficient * nn.functional.mse_loss(self.critic(ctx, otk, g), ret)
                self.critic_optimizer.zero_grad(); critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm); self.critic_optimizer.step()
                a_l.append(float(actor_loss)); c_l.append(float(critic_loss)); ents.append(float(ent))
        self._rollout = []
        return {"actor_loss": float(np.mean(a_l)), "critic_loss": float(np.mean(c_l)), "entropy": float(np.mean(ents)), "rollout_size": float(n), "gc_loss": gc_loss}

    # ---- Graph Coordinator: supervised (BCE + lambda_g * MSE) ----
    def record_outcome(self, nodes, ego, agent_feats, late: bool, delivery_min: float) -> None:
        self._gc_buffer.append((nodes, int(ego), agent_feats, float(late), float(delivery_min)))
        if len(self._gc_buffer) > self.gc_capacity:
            del self._gc_buffer[: len(self._gc_buffer) - self.gc_capacity]

    def _update_coordinator(self) -> float:
        if len(self._gc_buffer) < 32:
            return 0.0
        idx = self._rng.choice(len(self._gc_buffer), size=min(self.gc_batch, len(self._gc_buffer)), replace=False)
        batch = [self._gc_buffer[int(i)] for i in idx]
        nodes_t, nmask_t = self._nodes_batch([b[0] for b in batch])
        ego = torch.as_tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        feats = torch.as_tensor(np.stack([b[2] for b in batch]), dtype=torch.float32, device=self.device)
        y_cls = torch.as_tensor([b[3] for b in batch], dtype=torch.float32, device=self.device)
        y_reg = torch.as_tensor([b[4] for b in batch], dtype=torch.float32, device=self.device) / 60.0
        logit, reg = self.coordinator(nodes_t, nmask_t, ego, feats)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, y_cls) + self.gc_lambda_reg * nn.functional.mse_loss(reg, y_reg)
        self.gc_optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.coordinator.parameters(), self.max_grad_norm); self.gc_optimizer.step()
        return float(loss)

    # ---- checkpoint ----
    def save(self, path: str) -> None:
        from pathlib import Path
        out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self._checkpoint_metadata())
        payload.update({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "coordinator": self.coordinator.state_dict(),
                        "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(), "gc_optimizer": self.gc_optimizer.state_dict()})
        torch.save(payload, out)

    def load(self, path: str) -> None:
        from pathlib import Path
        payload = torch.load(Path(path), map_location=self.device)
        self._validate_checkpoint(payload)
        self.actor.load_state_dict(payload["actor"]); self.critic.load_state_dict(payload["critic"]); self.coordinator.load_state_dict(payload["coordinator"])
        for k, opt in (("actor_optimizer", self.actor_optimizer), ("critic_optimizer", self.critic_optimizer), ("gc_optimizer", self.gc_optimizer)):
            if k in payload: opt.load_state_dict(payload[k])

    def _checkpoint_metadata(self):
        return {"candidate_dim": self.candidate_dim, "context_dim": self.context_dim, "max_candidates": self.max_candidates,
                "feature_schema": self.feature_schema, "hidden_dim": self.hidden_dim, "transition_mode": "msra_matching_ippo"}

    def _validate_checkpoint(self, payload):
        if payload.get("feature_schema") != self.feature_schema:
            raise ValueError("MSRA checkpoint feature schema is incompatible")
        if payload.get("transition_mode") != "msra_matching_ippo":
            raise ValueError("checkpoint is not an L2_MSRA checkpoint")

    def _actor_forward(self, batch): raise NotImplementedError
    def _critic_forward(self, batch): raise NotImplementedError
    def _rollout_array_fields(self): return {}


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
class MSRAOrderBatteryPolicy(F2Policy):
    """L2: MSRA-Matching + IPPO."""

    # -- agent structure IPPO: independent per-drone trajectories, own-order rewards, local critic --
    agent_structure = IPPO

    def _training_trajectory_groups(self) -> list[list[_JointPlanDecision]]:
        trajectories: dict[str, list[_JointPlanDecision]] = {}
        for decision in self._episode_decisions:
            key = decision.drone_id or f"launchpad:{decision.launchpad_id}"
            trajectories.setdefault(str(key), []).append(decision)
        return list(trajectories.values())

    def _trajectory_reward_events(self, trajectory, reward_events):
        order_ids = {
            str(decision.order_id)
            for decision in trajectory
            if decision.kind == "dispatch" and decision.order_id is not None
        }
        if not order_ids:
            return []
        return [
            event for event in (reward_events or [])
            if event.get("order_id") is not None and str(event.get("order_id")) in order_ids
        ]

    candidate_layout = "olb"
    dispatch_action_prefix = "msra_matching_dispatch"
    joint_plan_config_key = "order_battery_msra"
    joint_plan_feature_schema = "msra_matching_ippo_v1"
    learner_class = TorchMSRAIPPOLearner
    action_expansion = MSRA_MATCHING

    def __init__(self, config: dict, *, seed: int = 0):
        cfg = config.get("drone_decision", {}).get(self.joint_plan_config_key, {})
        self.lambda_p = float(cfg.get("lambda_p", 3.0))            # on-time bonus weight in U  (paper: 100 HKD with 30 HKD base; scaled to our ~17 value)
        self.cost_per_min = float(cfg.get("drone_cost_per_min", 0.0))
        self.risk_threshold = float(cfg.get("risk_threshold_G", 0.5))
        self.gc_warmup_episodes = int(cfg.get("gc_warmup_episodes", 30))
        self._episodes_done = 0
        self._open: dict[str, tuple] = {}   # order_id -> (nodes, ego, agent_feats) awaiting outcome
        super().__init__(config, seed=seed)

    def _learner_kwargs(self, cfg: dict, *, seed: int) -> dict:
        kwargs = super()._learner_kwargs(cfg, seed=seed)
        for knob in ("interaction_rank", "critic_type", "actor_type"):  # F2 actor/critic knobs; MSRA has its own nets
            kwargs.pop(knob, None)
        kwargs["gc_lambda_reg"] = float(cfg.get("gc_lambda_reg", 0.2))
        kwargs["gc_lr"] = float(cfg.get("gc_lr", 1e-3))
        return kwargs

    def _menu_input(self, candidates):
        menu = self.menu.factor_menu
        if menu is None or len(menu) != len(candidates):
            raise RuntimeError("MSRA requires the factored olb menu")
        pm = pointer_menu_from_factors(menu, candidates)
        self.menu.feature_rows += pm.factor_rows
        return pm

    @staticmethod
    def _rule_battery(pm: PointerMenu, menu: FactorMenu, o: int, l: int) -> int:
        """Flat candidate index of the lowest-SoC battery that can fly (o, l)."""
        flats = pm.triplet_to_flat[o, l]
        ok = np.nonzero(flats >= 0)[0]
        socs = pm.bats[ok, 0]
        return int(flats[ok[int(np.argmin(socs))]])

    def plan(self, state, used_orders, used_batteries):
        del used_orders, used_batteries
        self._begin_plan(state)
        current_time = int(state.get("time", 0))
        horizon = max(1.0, float(self.config.get("simulation", {}).get("horizon_seconds", 1)))
        self._record_completed(state)
        fleet = self._collect_fleet_menus(state)            # per available drone: (drone_id, pad_id, pad, candidates, context, PointerMenu)
        nodes = self.menu.global_state(state)                    # launchpad graph node features [P, C]
        pad_ids = [str(pid) for pid, pad in state.get("launchpads", {}).items() if self.menu.available_drone_count(state, pad) > 0]
        ego_of = {pid: i for i, pid in enumerate(pad_ids)}
        risk_on = self._episodes_done >= self.gc_warmup_episodes
        actions = []
        by_pad: dict[str, list] = {}
        for item in fleet:
            by_pad.setdefault(item[1], []).append(item)
        for pad_id, drones in by_pad.items():
            # all drones at a pad share the pad menu; use the first drone's (identical drone type)
            drone_id0, _pid, launchpad, candidates, context, pm = drones[0]
            if pm.n_o == 0 or pm.n_l == 0:
                continue
            out = self.learner.plan_pad(pm, context, nodes, ego_of.get(pad_id, 0), training=self.training)
            if out is None:
                continue
            # ---- bipartite matching utility per order (Eq. 22), drones identical -> top-k ----
            util = np.full(pm.n_o, -np.inf)
            for o in range(pm.n_o):
                if not out["has_route"][o]:
                    continue
                l = int(out["l"][o])
                flat = self._rule_battery(pm, self.menu.factor_menu, o, l)
                cand = candidates[flat]
                order = state["orders"][cand.order_id]
                # exact ETA: loaded leg time = dist / cruise speed (route edge holds return time; loaded time from order feature 5 = dist/5000)
                dist_m = float(pm.orders[o, 5]) * 5000.0
                eta = current_time + dist_m / float(self.config.get("vehicles", {}).get("uavs", {}).get("speed_mps", 13.8889))
                on_time = eta <= float(order.get("deadline", current_time))
                flight_min = (dist_m / 13.8889 + float(pm.edge[o, l, 1]) * horizon) / 60.0
                risk = float(out["risk"][o])
                if risk_on and risk > self.risk_threshold:
                    continue                                              # hard constraint (21f analogue)
                util[o] = float(order.get("value", 0.0)) + self.lambda_p * float(on_time) - self.cost_per_min * flight_min + self.learner.gamma * float(out["value"][o])
            order_rank = [o for o in np.argsort(-util) if np.isfinite(util[o])]
            for (drone_id, _pid, launchpad, candidates, context, pm_d), o in zip(drones, order_rank):
                l = int(out["l"][o])
                flat = self._rule_battery(pm, self.menu.factor_menu, o, l)
                cand = candidates[flat]
                agent_feats = np.concatenate([context, pm.orders[o], pm.pads[l], pm.edge[o, l]]).astype(np.float32)
                if self.training:
                    sample = MSRASample(context=context, order_tok=pm.orders[o], pads=pm.pads, edge=pm.edge[o], pad_mask=pm.ol_ok[o],
                                        nodes=nodes, action=l, log_prob=float(out["log_prob"][o]), value=float(out["value"][o]))
                    self._episode_decisions.append(_JointPlanDecision(current_time, sample, "dispatch", pad_id, drone_id, cand.order_id))
                    self._open[cand.order_id] = (nodes, ego_of.get(pad_id, 0), agent_feats, current_time)
                self._metrics.record_group(order_ids=[c.order_id for c in candidates if c.kind == "dispatch"],
                                           battery_ids=[c.battery_id for c in candidates if c.kind == "dispatch"],
                                           selected_count=1, available_drones=1, selected_kind="dispatch")
                actions.append({
                    "action_id": f"{self.dispatch_action_prefix}_{current_time}_{drone_id}_{cand.order_id}_{cand.battery_id}_{cand.return_launchpad_id}",
                    "action_type": "order_battery_dispatch", "order_id": cand.order_id, "battery_id": cand.battery_id,
                    "target_launchpad_id": cand.return_launchpad_id, "drone_id": drone_id, "feasibility": True,
                })
        return actions

    def _record_completed(self, state) -> None:
        """Feed the Graph Coordinator with outcomes of dispatched orders (late flag, delivery time)."""
        if not self._open:
            return
        orders = state.get("orders", {})
        for oid in list(self._open):
            order = orders.get(oid)
            if order is None or order.get("status") != OrderStatus.DELIVERED.value:
                continue
            nodes, ego, feats, t0 = self._open.pop(oid)
            delivered = float(order.get("delivery_time", t0)); deadline = float(order.get("deadline", delivered))
            self.learner.record_outcome(nodes, ego, feats, late=delivered > deadline, delivery_min=(delivered - t0) / 60.0)

    def finish_episode(self, *args, **kwargs):
        result = super().finish_episode(*args, **kwargs)
        self._episodes_done += 1
        return result

    def clear_episode(self) -> None:
        super().clear_episode()
        self._open = {}
