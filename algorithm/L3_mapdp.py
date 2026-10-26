"""L3: MAPDP (Zong, Zheng, Li, Zhang, Zheng, Xu -- "MAPDP: Cooperative Multi-Agent
Reinforcement Learning to Solve Pickup and Delivery Problems", AAAI 2022),
adapted to launchpad drone dispatch.  Label: ``MAPDP-Pointer+MAPPO``.

What MAPDP is (the pieces we keep):
  1. **Paired context encoder** -- a transformer encoder over problem entities in
     which each pickup node is embedded together with its paired delivery node
     (and vice versa), so the graph embedding is pair-aware.
  2. **Cooperative multi-agent pointer decoders** -- every vehicle owns a decoder
     that, at each decoding step, forms a query from [graph embedding, its own
     state, a communication summary of the other vehicles], applies a multi-head
     glimpse over the entity embeddings and a single-head, tanh-clipped (C=10)
     compatibility to point at the next node; all vehicles decode in parallel
     and conflicts on the same node are resolved by rule.
  3. **Joint reward, centralised critic** -- one team reward per step and a
     critic on the global state.

Mapping to this simulator (differences we do NOT hide):
  * pickup = the order's origin launchpad, delivery = its destination; the paired
    context embedding of an order is [order attributes | pickup-pad state |
    destination coordinates] -- exactly the (o) block of our olb layout.
  * one "node choice" of MAPDP becomes the three pointer steps a drone needs to
    commit a dispatch here: order -> return launchpad -> battery, each an
    attention pointer over that entity type, autoregressive within the drone's
    step (the return-pad query is conditioned on the chosen order, the battery
    query on the chosen (order, pad)); feasibility masks come from the
    simulator exactly as for every other method (a drone may also point at a
    NO-DISPATCH token = wait).
  * MAPDP's static instance is encoded once; our menu changes every step, so
    the encoder runs once per launchpad per step (tokens are the pad's orders,
    return pads and batteries -- O(|O|+|L|+|B|), never the triplet product).
  * MAPDP trains with cooperative A2C; we train every learned method with PPO,
    so the adapter uses the repository's synchronous-step PPO (AS1 structure).
Agent structure = MAPPO (synchronous joint step, rule conflict resolution,
GlobalStateCritic shared with AS1/AS3); action expansion = MAPDP pointer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from algorithm.F2.menu import BATTERY_DIM, FactorMenu
from algorithm.dispatch.presets import MAPDP_POINTER
from algorithm.AS1_mappo import MAPPOStructure
from algorithm.AS1_mappo import TorchSynchronousDroneMAPPO
from algorithm.F2.critic import GlobalStateCritic, global_state_batch

ORDER_TOKEN_DIM = 9      # order7 + dest_xy
PAD_TOKEN_DIM = 12       # target11 (dist dest->pad zeroed: order specific) + pad_xy ... see PointerMenu
BATTERY_TOKEN_DIM = BATTERY_DIM
ROUTE_EDGE_DIM = 4       # dist_return/5000, time_return/horizon, E_task/1500, E_return/1500 for an (o,l) pair
CLIP_C = 10.0


# --------------------------------------------------------------------------- #
# Pointer menu: entity tokens + feasibility structure of one drone's decision
# --------------------------------------------------------------------------- #
class PointerMenu:
    """Entity view of a pad menu for the pointer decoder.

    orders   [n_o, 9]        order7 | dest_xy                 (paired-context: pickup pad state is in the context)
    pads     [n_l, 12]       target features w/o the order-specific distance | pad_xy
    bats     [n_b, 6]
    edge     [n_o, n_l, 4]   (o,l) route scalars (0 where the pair is not a route)
    ol_ok    [n_o, n_l]      the (o,l) route exists (some battery can fly it)
    olb_ok   [n_o, n_l, n_b] feasible triplet
    flat_index[(o,l,b)] -> index into the candidate list; stop_index / defer_index[o]
    """

    __slots__ = ("orders", "pads", "bats", "edge", "ol_ok", "olb_ok", "triplet_to_flat", "defer_flat", "stop_flat", "n_o", "n_l", "n_b")

    def __init__(self, orders, pads, bats, edge, ol_ok, olb_ok, triplet_to_flat, defer_flat, stop_flat):
        self.orders = orders; self.pads = pads; self.bats = bats; self.edge = edge
        self.ol_ok = ol_ok; self.olb_ok = olb_ok
        self.triplet_to_flat = triplet_to_flat   # np.int64 [n_o, n_l, n_b], -1 where infeasible
        self.defer_flat = defer_flat             # np.int64 [n_o], -1 if the order has no defer entry
        self.stop_flat = int(stop_flat)          # -1 if no stop entry
        self.n_o, self.n_l, self.n_b = int(len(orders)), int(len(pads)), int(len(bats))

    @property
    def factor_rows(self) -> int:
        return self.n_o + self.n_l + self.n_b

    def no_dispatch_flat(self) -> int:
        if self.stop_flat >= 0:
            return self.stop_flat
        valid = self.defer_flat[self.defer_flat >= 0]
        return int(valid[0]) if len(valid) else 0


def pointer_menu_from_factors(menu: FactorMenu, candidates: list) -> PointerMenu:
    """Build the entity view from the factored menu + the flat candidate list."""
    order_ids: list[str] = []
    pad_ids: list[str] = []
    bat_ids: list[str] = []
    o_index: dict[str, int] = {}
    l_index: dict[str, int] = {}
    b_index: dict[str, int] = {}
    for c in candidates:
        if c.order_id is not None and c.order_id not in o_index:
            o_index[c.order_id] = len(order_ids); order_ids.append(c.order_id)
        if c.kind == "dispatch":
            if c.return_launchpad_id not in l_index:
                l_index[c.return_launchpad_id] = len(pad_ids); pad_ids.append(c.return_launchpad_id)
            if c.battery_id not in b_index:
                b_index[c.battery_id] = len(bat_ids); bat_ids.append(c.battery_id)
    n_o, n_l, n_b = len(order_ids), len(pad_ids), len(bat_ids)
    orders = np.zeros((n_o, ORDER_TOKEN_DIM), dtype=np.float32)
    pads = np.zeros((max(n_l, 1), PAD_TOKEN_DIM), dtype=np.float32)
    bats = np.zeros((max(n_b, 1), BATTERY_TOKEN_DIM), dtype=np.float32)
    edge = np.zeros((n_o, max(n_l, 1), ROUTE_EDGE_DIM), dtype=np.float32)
    ol_ok = np.zeros((n_o, max(n_l, 1)), dtype=bool)
    olb_ok = np.zeros((n_o, max(n_l, 1), max(n_b, 1)), dtype=bool)
    triplet_to_flat = -np.ones((n_o, max(n_l, 1), max(n_b, 1)), dtype=np.int64)
    defer_flat = -np.ones((max(n_o, 1),), dtype=np.int64)
    stop_flat = -1
    seen_o = set(); seen_l = set(); seen_b = set()
    for flat, c in enumerate(candidates):
        if c.kind == "stop":
            stop_flat = flat; continue
        oi = o_index[c.order_id]
        row = menu.ol_raw[menu.ol_idx[flat]]
        if oi not in seen_o:
            orders[oi] = row[0:9]; seen_o.add(oi)
        if c.kind == "defer":
            defer_flat[oi] = flat; continue
        li = l_index[c.return_launchpad_id]; bi = b_index[c.battery_id]
        if li not in seen_l:
            # target11 = [dist(dest,pad)/5000, backlog, docked, reserved, incoming, outgoing, projected, backlog-proj, charged, charging, is_origin]
            # drop the order-specific distance (index 0) -> 10 pad-only features + pad_xy
            pads[li, 0:10] = row[10:20]; pads[li, 10:12] = row[20:22]; seen_l.add(li)
        if bi not in seen_b:
            bats[bi] = menu.b_raw[menu.b_idx[flat], 0:BATTERY_DIM]; seen_b.add(bi)
        if not ol_ok[oi, li]:
            e_task, e_ret, d_ret, t_ret = row[22], row[23], row[24], row[25]
            edge[oi, li] = (d_ret, t_ret, e_task / 1500.0, e_ret / 1500.0)
            ol_ok[oi, li] = True
        olb_ok[oi, li, bi] = True
        triplet_to_flat[oi, li, bi] = flat
    return PointerMenu(orders, pads, bats, edge, ol_ok, olb_ok, triplet_to_flat, defer_flat, stop_flat)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class _MHA(nn.Module):
    """Explicit multi-head attention (no fused kernels; the stock TransformerEncoder
    backward fails on this GPU). q [R, nq, H], kv [R, nk, H], kv_mask [R, nk] True=allowed."""

    def __init__(self, hidden_dim: int, heads: int):
        super().__init__()
        assert hidden_dim % heads == 0
        self.h, self.d = heads, hidden_dim // heads
        self.q = nn.Linear(hidden_dim, hidden_dim); self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim); self.o = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, kv, kv_mask):
        R, nq, _ = q.shape; nk = kv.shape[1]
        Q = self.q(q).view(R, nq, self.h, self.d).transpose(1, 2)          # [R, h, nq, d]
        K = self.k(kv).view(R, nk, self.h, self.d).transpose(1, 2)
        V = self.v(kv).view(R, nk, self.h, self.d).transpose(1, 2)
        att = (Q @ K.transpose(-1, -2)) * (self.d ** -0.5)                 # [R, h, nq, nk]
        att = att.masked_fill(~kv_mask[:, None, None, :], torch.finfo(att.dtype).min)
        att = torch.softmax(att, dim=-1)
        out = (att @ V).transpose(1, 2).reshape(R, nq, self.h * self.d)
        return self.o(out)


class _LayerNorm(nn.Module):
    """Explicit LayerNorm (nn.LayerNorm's fused backward fails on this GPU/torch build)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)); self.bias = nn.Parameter(torch.zeros(dim)); self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True); var = ((x - mu) ** 2).mean(-1, keepdim=True)
        return (x - mu) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class _EncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int):
        super().__init__()
        self.n1 = _LayerNorm(hidden_dim); self.att = _MHA(hidden_dim, heads)
        self.n2 = _LayerNorm(hidden_dim)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.ReLU(), nn.Linear(2 * hidden_dim, hidden_dim))

    def forward(self, h, mask):
        x = self.n1(h); h = h + self.att(x, x, mask)
        return h + self.ff(self.n2(h))


class _EntityEncoder(nn.Module):
    """Transformer encoder over the pad's entity tokens (orders | pads | batteries).

    Type-specific input projections (the paired-context embedding for orders),
    then ``layers`` pre-norm self-attention blocks over the concatenated set.
    """

    def __init__(self, hidden_dim: int, heads: int, layers: int):
        super().__init__()
        self.order_in = nn.Linear(ORDER_TOKEN_DIM, hidden_dim)
        self.pad_in = nn.Linear(PAD_TOKEN_DIM, hidden_dim)
        self.bat_in = nn.Linear(BATTERY_TOKEN_DIM, hidden_dim)
        self.type_embed = nn.Parameter(torch.zeros(3, hidden_dim))
        self.blocks = nn.ModuleList([_EncoderBlock(hidden_dim, heads) for _ in range(layers)])

    def forward(self, orders, o_mask, pads, l_mask, bats, b_mask):
        # orders [R, n_o, 9], pads [R, n_l, 12], bats [R, n_b, 6]; masks True = real token
        h = torch.cat([
            self.order_in(orders) + self.type_embed[0],
            self.pad_in(pads) + self.type_embed[1],
            self.bat_in(bats) + self.type_embed[2],
        ], dim=1)                                                          # [R, n_o+n_l+n_b, H]
        mask = torch.cat([o_mask, l_mask, b_mask], dim=1)                  # [R, T]
        safe = mask.clone(); safe[:, 0] = True                             # never a fully masked row
        for block in self.blocks:
            h = block(h, safe)
        n_o, n_l = orders.shape[1], pads.shape[1]
        return h[:, :n_o], h[:, n_o:n_o + n_l], h[:, n_o + n_l:], h, mask


class _PointerStep(nn.Module):
    """MAPDP decoding step: multi-head glimpse then single-head clipped compatibility."""

    def __init__(self, hidden_dim: int, heads: int):
        super().__init__()
        self.glimpse = _MHA(hidden_dim, heads)
        self.q_out = nn.Linear(hidden_dim, hidden_dim)
        self.k_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5

    def forward(self, query, keys, key_mask):
        # query [R, H], keys [R, n, H], key_mask [R, n] (True = allowed)
        q = query.unsqueeze(1)
        safe = key_mask.clone(); safe[:, 0] |= ~key_mask.any(1)            # glimpse needs >= 1 key
        g = self.glimpse(q, keys, safe)                                    # [R, 1, H]
        q2 = self.q_out(q + g)
        compat = (q2 @ self.k_out(keys).transpose(1, 2)).squeeze(1) * self.scale   # [R, n]
        logits = CLIP_C * torch.tanh(compat)
        return logits.masked_fill(~key_mask, torch.finfo(logits.dtype).min)


class MAPDPPointerActor(nn.Module):
    """Per-drone cooperative pointer decoder over the encoded pad entities.

    query_0 = W[ graph_mean | own context | mean of other drones' contexts (communication) ]
    step 1 (order | NO-DISPATCH token), step 2 (return pad | chosen order, route edge),
    step 3 (battery | chosen order, pad).  Returns the three masked logit sets so
    the caller can sample / evaluate log-probs autoregressively.
    """

    def __init__(self, context_dim: int, hidden_dim: int, heads: int, layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = _EntityEncoder(hidden_dim, heads, layers)
        self.ctx_in = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.ReLU())
        self.comm_in = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.ReLU())
        self.q0 = nn.Linear(3 * hidden_dim, hidden_dim)
        # learnable NO-DISPATCH key; random init so its compatibility is not pinned at 0 (k_out has no bias)
        self.no_dispatch = nn.Parameter(torch.randn(hidden_dim) * hidden_dim ** -0.5)
        self.edge_in = nn.Linear(ROUTE_EDGE_DIM, hidden_dim)
        self.q_pad = nn.Linear(2 * hidden_dim, hidden_dim)
        self.q_bat = nn.Linear(3 * hidden_dim, hidden_dim)
        self.step_order = _PointerStep(hidden_dim, heads)
        self.step_pad = _PointerStep(hidden_dim, heads)
        self.step_bat = _PointerStep(hidden_dim, heads)

    def encode(self, b: dict[str, torch.Tensor]):
        h_o, h_l, h_b, h_all, mask_all = self.encoder(b["orders"], b["o_mask"], b["pads"], b["l_mask"], b["bats"], b["b_mask"])
        m = mask_all.unsqueeze(-1).float()
        graph = (h_all * m).sum(1) / m.sum(1).clamp_min(1.0)               # [R, H]
        q0 = self.q0(torch.cat([graph, self.ctx_in(b["context"]), self.comm_in(b["comm"])], -1))
        return h_o, h_l, h_b, q0

    def order_logits(self, q0, h_o, o_ok):
        # NO-DISPATCH token at index 0 (padding-independent), orders at 1..n_o
        keys = torch.cat([self.no_dispatch.expand(h_o.shape[0], 1, -1), h_o], 1)
        allowed = torch.cat([torch.ones_like(o_ok[:, :1]), o_ok], 1)
        return self.step_order(q0, keys, allowed)                          # [R, 1 + n_o]

    def pad_logits(self, q0, h_o, h_l, edge, ol_ok, o_choice):
        # o_choice [R] indices into orders (valid rows only)
        r = torch.arange(h_o.shape[0], device=h_o.device)
        ho = h_o[r, o_choice]                                              # [R, H]
        keys = h_l + self.edge_in(edge[r, o_choice])                        # [R, n_l, H]  route-aware keys
        q = self.q_pad(torch.cat([q0, ho], -1))
        return self.step_pad(q, keys, ol_ok[r, o_choice])                  # [R, n_l]

    def bat_logits(self, q0, h_o, h_l, h_b, olb_ok, o_choice, l_choice):
        r = torch.arange(h_o.shape[0], device=h_o.device)
        q = self.q_bat(torch.cat([q0, h_o[r, o_choice], h_l[r, l_choice]], -1))
        return self.step_bat(q, h_b, olb_ok[r, o_choice, l_choice])        # [R, n_b]


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
def stack_pointer_menus(menus: list[PointerMenu], contexts: np.ndarray, comm: np.ndarray, device) -> dict[str, torch.Tensor]:
    R = len(menus)
    n_o = max(m.n_o for m in menus) or 1
    n_l = max(m.pads.shape[0] for m in menus)
    n_b = max(m.bats.shape[0] for m in menus)
    orders = np.zeros((R, n_o, ORDER_TOKEN_DIM), np.float32); o_mask = np.zeros((R, n_o), bool)
    pads = np.zeros((R, n_l, PAD_TOKEN_DIM), np.float32); l_mask = np.zeros((R, n_l), bool)
    bats = np.zeros((R, n_b, BATTERY_TOKEN_DIM), np.float32); b_mask = np.zeros((R, n_b), bool)
    edge = np.zeros((R, n_o, n_l, ROUTE_EDGE_DIM), np.float32)
    ol_ok = np.zeros((R, n_o, n_l), bool); olb_ok = np.zeros((R, n_o, n_l, n_b), bool)
    o_ok = np.zeros((R, n_o), bool)
    for r, m in enumerate(menus):
        if m.n_o:
            orders[r, : m.n_o] = m.orders; o_mask[r, : m.n_o] = True
            o_ok[r, : m.n_o] = m.ol_ok.any(axis=1)
            edge[r, : m.n_o, : m.pads.shape[0]] = m.edge
            ol_ok[r, : m.n_o, : m.pads.shape[0]] = m.ol_ok
            olb_ok[r, : m.n_o, : m.pads.shape[0], : m.bats.shape[0]] = m.olb_ok
        pads[r, : m.pads.shape[0]] = m.pads; l_mask[r, : m.n_l] = True
        bats[r, : m.bats.shape[0]] = m.bats; b_mask[r, : m.n_b] = True
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    return {
        "orders": t(orders, torch.float32), "o_mask": t(o_mask, torch.bool), "o_ok": t(o_ok, torch.bool),
        "pads": t(pads, torch.float32), "l_mask": t(l_mask, torch.bool),
        "bats": t(bats, torch.float32), "b_mask": t(b_mask, torch.bool),
        "edge": t(edge, torch.float32), "ol_ok": t(ol_ok, torch.bool), "olb_ok": t(olb_ok, torch.bool),
        "context": t(contexts, torch.float32), "comm": t(comm, torch.float32),
    }


def _communication(contexts: np.ndarray) -> np.ndarray:
    """MAPDP communication summary: mean of the OTHER agents' contexts (zero when alone)."""
    A = len(contexts)
    if A <= 1:
        return np.zeros_like(contexts)
    total = contexts.sum(axis=0, keepdims=True)
    return ((total - contexts) / (A - 1)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Learner
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MAPDPJointSample:
    menus: list                 # PointerMenu per agent
    agent_contexts: np.ndarray  # [A, C]
    comm: np.ndarray            # [A, C]
    choices: np.ndarray         # [A, 3] (o, l, b); o == 0 means NO-DISPATCH (l, b = -1), else order index + 1
    agent_log_probs: np.ndarray # [A]
    log_prob: float
    value: float
    global_state: np.ndarray | None = None

    # compatibility with the AS1 learner's store_joint_episode / _batch_rows contract
    @property
    def candidate_mask(self):
        return np.ones((len(self.menus), 1), dtype=bool)


class TorchMAPDPPointerMAPPO(TorchSynchronousDroneMAPPO):
    """Synchronous-step PPO with the MAPDP pointer actor; GlobalStateCritic."""

    actor_context_label = "MAPDP drone context"
    supports_factor_menu = True  # we take the FactorMenu and build the entity view from it

    def __init__(self, *, candidate_dim: int, context_dim: int, attention_heads: int = 4, encoder_layers: int = 2,
                 update_cost_budget: int = 10_000_000, **kwargs):
        kwargs.pop("actor_type", None); kwargs.pop("interaction_rank", None)
        self.context_dim = int(context_dim)
        self.attention_heads = int(attention_heads); self.encoder_layers = int(encoder_layers)
        # bound on the padded work per pointer call (PPO update and act_parallel; memory, not maths)
        self.update_cost_budget = int(update_cost_budget)
        # BaseTorchMAPPOLearner builds a generic actor/critic; replace both below
        super(TorchSynchronousDroneMAPPO, self).__init__(candidate_dim=candidate_dim, actor_context_dim=self.context_dim, **kwargs)
        self.actor_type = "mapdp_pointer"
        self.interaction_rank = 0
        self.actor = MAPDPPointerActor(self.context_dim, self.hidden_dim, self.attention_heads, self.encoder_layers).to(self.device)
        self.critic = GlobalStateCritic(self.context_dim, self.hidden_dim).to(self.device)
        self._initialize_optimizers()
        self._joint_rollout: list[dict[str, Any]] = []

    # ---- forward over a batch of agents: sample (choices=None) or evaluate ----
    def _joint_forward(self, b: dict[str, torch.Tensor], choices: torch.Tensor | None, *, greedy: bool):
        h_o, h_l, h_b, q0 = self.actor.encode(b)
        R = h_o.shape[0]; n_o = h_o.shape[1]
        lo = self.actor.order_logits(q0, h_o, b["o_ok"])                    # [R, 1+n_o]; slot 0 = NO-DISPATCH
        do = torch.distributions.Categorical(logits=lo)
        o = choices[:, 0] if choices is not None else (lo.argmax(-1) if greedy else do.sample())
        logp = do.log_prob(o); ent = do.entropy()
        dispatch = o > 0
        o_safe = torch.where(dispatch, o - 1, torch.zeros_like(o))         # order row index
        lp = self.actor.pad_logits(q0, h_o, h_l, b["edge"], b["ol_ok"], o_safe)
        # rows that did not dispatch: give them a dummy allowed key 0 so Categorical is valid
        lp = torch.where(dispatch.unsqueeze(1), lp, torch.zeros_like(lp))
        dl = torch.distributions.Categorical(logits=lp)
        l = choices[:, 1].clamp_min(0) if choices is not None else (lp.argmax(-1) if greedy else dl.sample())
        lb = self.actor.bat_logits(q0, h_o, h_l, h_b, b["olb_ok"], o_safe, l)
        lb = torch.where(dispatch.unsqueeze(1), lb, torch.zeros_like(lb))
        db = torch.distributions.Categorical(logits=lb)
        bb = choices[:, 2].clamp_min(0) if choices is not None else (lb.argmax(-1) if greedy else db.sample())
        d = dispatch.float()
        logp = logp + d * (dl.log_prob(l) + db.log_prob(bb))
        # per-decision entropy: mean over the pointer steps taken (1 for NO-DISPATCH, 3 for a dispatch)
        ent = (ent + d * (dl.entropy() + db.entropy())) / (1.0 + 2.0 * d)
        out_choices = torch.stack([o, torch.where(dispatch, l, torch.full_like(l, -1)), torch.where(dispatch, bb, torch.full_like(bb, -1))], -1)  # o: 0 = no-dispatch, else order+1
        return out_choices, logp, ent

    def act_parallel(self, candidate_features, agent_contexts, global_state=None, *, training: bool):
        menus = list(candidate_features)
        contexts = np.asarray(agent_contexts, dtype=np.float32).reshape(len(menus), self.context_dim)
        comm = _communication(contexts)
        greedy = not (training or self.evaluation_action_mode == "stochastic")
        # agents of one step go through the pointer in memory-bounded chunks (the
        # entity attention is quadratic in the token count; a meituan backlog can
        # put hundreds of order tokens on every pad)
        pairs = [(m, contexts[a], comm[a], None, None, a) for a, m in enumerate(menus)]
        choices_parts, logp_parts = [], []
        with torch.no_grad():
            for chunk in self._pair_chunks(pairs):
                b = stack_pointer_menus([q[0] for q in chunk], np.stack([q[1] for q in chunk]), np.stack([q[2] for q in chunk]), self.device)
                c, lp, _ = self._joint_forward(b, None, greedy=greedy)
                choices_parts.append(c); logp_parts.append(lp)
            choices = torch.cat(choices_parts, 0); logp = torch.cat(logp_parts, 0)
            if not training or global_state is None or len(global_state) == 0:
                value_t = torch.zeros((1,), device=self.device)
            else:
                value_t = self.critic(*global_state_batch([np.asarray(global_state, dtype=np.float32)], self.context_dim, self.device))
        choices_np = choices.cpu().numpy().astype(np.int64)
        flat = np.zeros(len(menus), dtype=np.int64)
        for a, m in enumerate(menus):
            o, l, bb = choices_np[a]
            flat[a] = m.no_dispatch_flat() if o == 0 else int(m.triplet_to_flat[o - 1, l, bb])
            if flat[a] < 0:
                raise RuntimeError("MAPDP pointer selected an infeasible triplet")
        agent_lp = logp.cpu().numpy().astype(np.float32)
        return flat, agent_lp, MAPDPJointSample(
            menus=menus, agent_contexts=contexts, comm=comm, choices=choices_np,
            agent_log_probs=agent_lp, log_prob=float(agent_lp.sum()), value=float(value_t.item()),
            global_state=global_state,
        )

    def update(self, epochs: int | None = None) -> dict[str, float]:
        if not self._joint_rollout:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "rollout_size": 0.0}
        rollout_size = len(self._joint_rollout)
        advantages = np.asarray([row["advantage"] for row in self._joint_rollout], dtype=np.float32)
        if len(advantages) > 1 and float(advantages.std()) > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_losses, critic_losses, entropies = [], [], []
        update_epochs = self.ppo_epochs if epochs is None else max(1, int(epochs))
        for _ in range(update_epochs):
            indices = self._rng.permutation(rollout_size)
            for start in range(0, rollout_size, self.batch_size):
                idx = indices[start: start + self.batch_size]
                rows = [self._joint_rollout[int(i)] for i in idx]
                # flatten (transition, agent) pairs; each agent is one pointer decode
                pairs = []   # (menu, context, comm, choice, old_log_prob, transition index)
                for t, row in enumerate(rows):
                    s = row["sample"]
                    for a, m in enumerate(s.menus):
                        pairs.append((m, s.agent_contexts[a], s.comm[a], s.choices[a], s.agent_log_probs[a], t))
                owner_all = torch.as_tensor([p[5] for p in pairs], dtype=torch.long, device=self.device)
                counts = torch.bincount(owner_all, minlength=len(rows)).clamp_min(1).float()
                adv_rows = torch.as_tensor(advantages[idx], dtype=torch.float32, device=self.device)
                # the objective is separable per agent (mean over agents within a transition,
                # then over transitions, as AS1), so the pairs go through the pointer in
                # memory-bounded chunks and the gradients accumulate; identical to one pass
                self.actor_optimizer.zero_grad()
                obj_sum = torch.zeros((), device=self.device); ent_sum = torch.zeros(len(rows), device=self.device)
                for chunk in self._pair_chunks(pairs):
                    b = stack_pointer_menus([p[0] for p in chunk], np.stack([p[1] for p in chunk]), np.stack([p[2] for p in chunk]), self.device)
                    _, logp, ent = self._joint_forward(b, torch.as_tensor(np.stack([p[3] for p in chunk]), dtype=torch.long, device=self.device), greedy=False)
                    old = torch.as_tensor(np.asarray([p[4] for p in chunk]), dtype=torch.float32, device=self.device)
                    owner_t = torch.as_tensor([p[5] for p in chunk], dtype=torch.long, device=self.device)
                    w = 1.0 / (counts[owner_t] * len(rows))          # per-pair weight of the double mean
                    adv = adv_rows[owner_t]
                    ratio = torch.exp(logp - old)
                    objective = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv)
                    chunk_loss = -(w * objective).sum() - self.entropy_coefficient * (w * ent).sum()
                    chunk_loss.backward()
                    obj_sum = obj_sum + (w * objective).detach().sum()
                    ent_sum = ent_sum.index_add(0, owner_t, ent.detach())
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm); self.actor_optimizer.step()
                ent_t = ent_sum / counts
                actor_loss = -obj_sum - self.entropy_coefficient * ent_t.mean()
                value = self.critic(*global_state_batch(
                    [row["sample"].global_state if row["sample"].global_state is not None else np.zeros((0, self.context_dim), np.float32) for row in rows],
                    self.context_dim, self.device))
                ret = torch.as_tensor([row["return"] for row in rows], dtype=torch.float32, device=self.device)
                critic_loss = self.value_coefficient * nn.functional.mse_loss(value, ret)
                self.critic_optimizer.zero_grad(); critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm); self.critic_optimizer.step()
                actor_losses.append(float(actor_loss.item())); critic_losses.append(float(critic_loss.item())); entropies.append(float(ent_t.mean().item()))
        self._joint_rollout = []
        return {"actor_loss": float(np.mean(actor_losses)), "critic_loss": float(np.mean(critic_losses)),
                "entropy": float(np.mean(entropies)), "rollout_size": float(rollout_size)}

    def _pair_chunks(self, pairs):
        """Split (transition, agent) pairs into chunks whose padded work stays bounded:
        sum over the chunk of n_o*n_l*n_b + 4*heads*layers*(n_o+n_l+n_b)^2 <= update_cost_budget
        (used by the PPO update and by act_parallel)."""
        budget = float(self.update_cost_budget)
        chunk, load = [], 0.0
        for p in pairs:
            m = p[0]
            n_o, n_l, n_b = max(1, m.n_o), max(1, m.pads.shape[0]), max(1, m.bats.shape[0])
            # grid tensors + attention maps (heads x layers x saved activations) per pair
            cost = float(n_o * n_l * n_b) + 4.0 * self.attention_heads * self.encoder_layers * float(n_o + n_l + n_b) ** 2
            if chunk and load + cost > budget:
                yield chunk
                chunk, load = [], 0.0
            chunk.append(p); load += cost
        if chunk:
            yield chunk

    def _checkpoint_metadata(self):
        return {
            "candidate_dim": self.candidate_dim, "context_dim": self.context_dim, "max_candidates": self.max_candidates,
            "feature_schema": self.feature_schema, "hidden_dim": self.hidden_dim,
            "attention_heads": self.attention_heads, "encoder_layers": self.encoder_layers,
            "transition_mode": "mapdp_pointer_synchronous_mappo",
        }

    def _validate_checkpoint(self, payload):
        if payload.get("feature_schema") != self.feature_schema:
            raise ValueError("MAPDP checkpoint feature schema is incompatible")
        if payload.get("transition_mode") != "mapdp_pointer_synchronous_mappo":
            raise ValueError("checkpoint is not an L3_MAPDP checkpoint")


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
class MAPDPOrderBatteryPolicy(MAPPOStructure):
    """L3: MAPDP-Pointer + MAPPO."""

    candidate_layout = "olb"          # factored menu -> entity tokens
    dispatch_action_prefix = "mapdp_pointer_dispatch"
    joint_plan_config_key = "order_battery_mapdp_pointer"
    joint_plan_feature_schema = "mapdp_pointer_entities_v1"
    learner_class = TorchMAPDPPointerMAPPO
    action_expansion = MAPDP_POINTER

    def _learner_kwargs(self, cfg: dict, *, seed: int) -> dict:
        kwargs = super()._learner_kwargs(cfg, seed=seed)
        kwargs["attention_heads"] = int(cfg.get("attention_heads", 4))
        kwargs["encoder_layers"] = int(cfg.get("encoder_layers", 2))
        return kwargs

    def _menu_input(self, candidates):
        menu = self.menu.factor_menu
        if menu is None or len(menu) != len(candidates):
            raise RuntimeError("MAPDP pointer requires the factored olb menu")
        pm = pointer_menu_from_factors(menu, candidates)
        self.menu.feature_rows += pm.factor_rows
        return pm
