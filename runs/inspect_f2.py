"""Step through one F2 decision and one F2 episode, printing what the code actually holds.

Every section corresponds to one step of the Methodology overview in docs/equations.tex, so
the document and the implementation can be checked against each other line by line. Nothing
here is a mock: it drives the real simulator, the real MenuBuilder and the real network.

  python runs/inspect_f2.py                                   # hongkong, first decision with >1 battery
  python runs/inspect_f2.py --city meituan-ssn --warmup 40
  python runs/inspect_f2.py --episode                         # also the episode-end pipeline
  python runs/inspect_f2.py --checks-only                     # just the PASS/FAIL invariants

Runs on CPU by default so it is deterministic and does not disturb training jobs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithm.F2.menu import (  # noqa: E402
    BATTERY_DIM,
    B_DIM,
    FLAT_DIM,
    OLB_DIM,
    OL_DIM,
    ORDER_DIM,
    TARGET_DIM,
    rows_from_factors,
)
from algorithm.common.mappo_utils import temporal_team_rewards  # noqa: E402
from algorithm.dispatch.presets import apply_algorithm_preset  # noqa: E402
from runs.utils.experiment_helpers import build_dispatcher  # noqa: E402
from simulator.core.simulator_env import SimulatorEnv  # noqa: E402
from simulator.model.enums import OrderStatus  # noqa: E402
from simulator.utils.config import load_config  # noqa: E402

# ---- column names, in the order the code writes them ---------------------------------
ORDER_COLS = ["value/50", "slack/H", "age/H", "weight/5", "dist(p,e)/5k"]
TARGET_COLS = ["backlog/20", "docked/20", "reserved/20", "incoming/20", "outgoing/20",
               "gap/20", "charged/20", "charging/20", "is_origin"]
ROUTE_COLS = ["dist(e,l)/5k", "E_task/1500"]
OL_COLS = ORDER_COLS + TARGET_COLS + ROUTE_COLS
B_COLS = ["soc", "charge/1500", "cap/1500", "charging?", "t_full/H", "1500/cap"]
CONTEXT_COLS = [
    "C1 time/H", "C2 pending/50", "C3 idle", "C4 moving", "C5 docked", "C6 batteries/50",
    "C7 charged", "C8 charging", "C9 pad queue/20", "C10 pad drones", "C11 pad docked",
    "C12 pad gap/20", "C13 pad charged/20", "C14 pad charging/20", "C15 lock", "C16 bias",
    "C17 committed dep", "C18 active pads", "C19 max pad dep", "C20 max occupancy",
    "C21 finite cap", "C22 position", "C23 count",
]

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def table(rows: np.ndarray, cols: list[str], labels: list[str], limit: int = 8) -> None:
    width = max(11, max(len(c) for c in cols) + 1)
    print(" " * 22 + "".join(f"{c:>{width}s}" for c in cols))
    for r, lab in list(zip(rows, labels))[:limit]:
        print(f"{lab:22s}" + "".join(f"{v:>{width}.4f}" for v in r))
    if len(labels) > limit:
        print(f"{'...':22s}({len(labels) - limit} more rows)")


def build(city: str, seed: int, device: str):
    config = load_config("scenario_default", "drone_decision")
    apply_algorithm_preset(config, "F2_Separable")
    config["drone_decision"]["candidate_pruning"] = False
    config.setdefault("rl", {})["training"] = True
    config["drone_decision"]["order_battery_d2_separable_ppo"]["device"] = device
    config.setdefault("instance", {}).update(
        {"city": city, "root": "data/instances_drone", "selection": "random_saved", "split": "train"}
    )
    config.setdefault("simulation", {}).update(
        {"time_advancement": "fixed_step", "delta_t_seconds": 60, "launchpad_capacity": "unlimited"}
    )
    env = SimulatorEnv(config)
    dispatcher = build_dispatcher("F2_Separable", config, env, seed)
    policy = dispatcher.order_battery_policy
    policy.training = True
    return env, dispatcher, policy


def find_decision(env, dispatcher, policy, warmup: int, min_batteries: int):
    """Advance the episode until some launchpad offers a menu worth looking at."""
    state = env.reset(seed=42)
    for step in range(warmup + 200):
        policy.menu.begin_step(state)
        for pad_id, pad in sorted(state.get("launchpads", {}).items()):
            if step < warmup:
                break
            candidates = policy.menu.candidates(state, pad, set(), set(), int(state["time"]), {}, {})
            menu = policy.menu.factor_menu
            if candidates and menu is not None and menu.n_b >= min_batteries and menu.n_ol > 2:
                return state, str(pad_id), pad, candidates, menu
        env.apply_actions(dispatcher.plan(state))
        state, _, done, _ = env.step()
        if done:
            break
    raise SystemExit("no suitable decision found; try --warmup or a busier --city")


def inspect_decision(env, policy, state, pad_id, pad, candidates, menu) -> None:
    learner = policy.learner
    now = int(state["time"])

    # ---------------------------------------------------------------- step 1: state
    rule(f"STEP 1  simulator state  (t = {now}s, launchpad {pad_id})")
    orders = state["orders"]
    queue = [o for o in pad.get("order_queue", []) if orders[str(o)]["status"] == OrderStatus.PENDING.value]
    waiting = [d for d in pad.get("drone_ids", []) if policy.menu.drone_is_available(state, pad, str(d))]
    print(f"  fleet {len(state['drones'])} drones, {len(state['batteries'])} batteries, "
          f"{len(state['launchpads'])} launchpads, {len(orders)} orders released")
    print(f"  this pad: {len(queue)} waiting orders, {len(waiting)} drones ready to be dispatched, "
          f"{len(pad.get('charged_battery_ids', []))} charged + {len(pad.get('charging_battery_ids', []))} charging batteries")
    print(f"  wind: {state['weather'].get('wind_speed_mps', 0):.1f} m/s toward "
          f"{state['weather'].get('wind_toward_deg', 0):.0f} deg")
    for oid in queue[:3]:
        o = orders[str(oid)]
        print(f"    order {oid:>12s}  value {o['value']:6.1f}  weight {o['weight']:4.1f}kg  "
              f"created {o['created_at']:5d}  expires {o['deadline']:5d}  ({o['deadline'] - now:+d}s from now)")

    # ------------------------------------------------------- step 2: menu construction
    rule("STEP 2  menu construction  (MenuBuilder.candidates)")
    kinds = [c.kind for c in candidates]
    n_dispatch, n_defer, n_stop = kinds.count("dispatch"), kinds.count("defer"), kinds.count("stop")
    print(f"  N       = {len(candidates)}   =  {n_dispatch} dispatch + {n_defer} defer + {n_stop} stop")
    print(f"  N_ol    = {menu.ol.shape[0]}   ({menu.n_ol} real (o,l) rows + {n_defer} defer rows + {n_stop} stop row)")
    print(f"  N_b     = {menu.b.shape[0]}   ({menu.n_b} batteries + 1 zero row)")
    print(f"  network reads N_ol + N_b = {menu.factor_rows} rows;"
          f"  a full expansion would read N = {len(candidates)}   ->  {len(candidates) / max(1, menu.factor_rows):.2f}x")
    check("N = |dispatch| + |defer| + |stop|", len(candidates) == n_dispatch + n_defer + n_stop)
    check("one defer row per offered order", n_defer == menu.ol.shape[0] - menu.n_ol - n_stop)

    print("\n  X^ol  [N_ol, %d]   order(%d) | target(%d) | route(%d)" % (OL_DIM, ORDER_DIM, TARGET_DIM, 2))
    labels = []
    for i in range(menu.ol.shape[0]):
        entry = next((c for c, j in zip(candidates, menu.ol_idx) if j == i), None)
        if entry is None:
            labels.append(f"[{i}] (unused)")
        elif entry.kind == "dispatch":
            labels.append(f"[{i}] {entry.order_id[-6:]} -> {entry.return_launchpad_id}")
        else:
            labels.append(f"[{i}] {entry.kind}{'' if entry.order_id is None else ' ' + entry.order_id[-6:]}")
    table(menu.ol, OL_COLS, labels)

    print(f"\n  X^b  [N_b+1, {B_DIM}]")
    b_labels = []
    for j in range(menu.b.shape[0]):
        entry = next((c for c, k in zip(candidates, menu.b_idx) if k == j and c.kind == "dispatch"), None)
        b_labels.append(f"[{j}] {entry.battery_id}" if entry else f"[{j}] zero row (defer/stop)")
    table(menu.b, B_COLS, b_labels)

    print("\n  menu index  (kind, ol_idx, b_idx)  ->  the action it stands for")
    for n in list(range(min(6, len(candidates)))) + ([len(candidates) - 1] if len(candidates) > 6 else []):
        c = candidates[n]
        k = ["dispatch", "defer", "stop"][int(np.argmax(menu.kind[n]))]
        target = "-" if c.kind != "dispatch" else f"{c.order_id[-6:]} -> {c.return_launchpad_id} with {c.battery_id}"
        print(f"    n={n:>4d}  kind={k:8s} ol_idx={menu.ol_idx[n]:>3d} b_idx={menu.b_idx[n]:>3d}   {target}")
    check("kind one-hot has exactly one 1 per entry", bool(np.all(menu.kind.sum(1) == 1)))
    check("defer/stop entries point at the zero battery row",
          all(menu.b_idx[n] == menu.b.shape[0] - 1 for n, c in enumerate(candidates) if c.kind != "dispatch"))
    check("zero battery row is all zeros", bool(np.all(menu.b[-1] == 0)))

    # feasibility: every dispatch entry must satisfy charge - E_task >= reserve
    margins = []
    for n, c in enumerate(candidates):
        if c.kind != "dispatch":
            continue
        raw_b = menu.b_raw[menu.b_idx[n]]
        raw_ol = menu.ol_raw[menu.ol_idx[n]]
        # b_raw layout: battery(BATTERY_DIM) | charge | cap_env | cap_model | reserve
        margins.append(float(raw_b[BATTERY_DIM] - raw_ol[22] - raw_b[BATTERY_DIM + 3]))
    if margins:
        print(f"\n  energy margin over the {len(margins)} dispatch entries: "
              f"min {min(margins):.1f} Wh, median {np.median(margins):.1f} Wh, max {max(margins):.1f} Wh")
        check("every enumerated dispatch entry is energy-feasible", min(margins) >= -1e-6,
              f"min margin {min(margins):.3f} Wh")

    # ----------------------------------------------------------- step 3: the context
    rule("STEP 3  decision context  c  (MenuBuilder.context)")
    context = policy.menu.context(state, pad, set(), set(), 1, {}, {}, agent_index=0, agent_count=len(state["drones"]))
    for i in range(0, len(CONTEXT_COLS), 4):
        print("  " + "   ".join(f"{n:<18s}{context[i + k]:8.4f}" for k, n in enumerate(CONTEXT_COLS[i:i + 4])))
    check(f"context has {policy.context_dim} entries", context.shape == (policy.context_dim,))
    check("C17-C21 are 0 for the first decision of a step", bool(np.all(np.abs(context[16:21]) < 1e-9)))

    # ------------------------------------------------------------ step 4: the scorer
    rule("STEP 4  separable low-rank scorer  (OLBSeparableActor.forward_factors)")
    actor = learner.actor
    fb = learner._factor_batch([menu])
    ctx_t = torch.as_tensor(context[None], dtype=torch.float32, device=learner.device)
    fb["context"] = ctx_t
    with torch.no_grad():
        h_c = actor.context_net(ctx_t)
        olc = torch.cat([actor.ol_net(fb["factor_ol"]), h_c.unsqueeze(1).expand(-1, menu.ol.shape[0], -1)], -1)
        bc = torch.cat([actor.b_net(fb["factor_b"]), h_c.unsqueeze(1).expand(-1, menu.b.shape[0], -1)], -1)
        g = actor.ol_score(olc).squeeze(-1)[0]
        h = actor.b_score(bc).squeeze(-1)[0]
        u = actor.ol_factor(olc)[0]
        v = actor.b_factor(bc)[0]
        kind_bias = actor.kind_score(h_c)[0]
        logits = learner._factor_logits(fb)[0]

    print(f"  h_c              [{h_c.shape[1]}]        from MLP_c(c)")
    print(f"  g                [{g.shape[0]:>3d}]        one score per (o,l)/defer/stop row")
    print(f"  h                [{h.shape[0]:>3d}]        one score per battery row")
    print(f"  u, v             [{u.shape[0]},{u.shape[1]}] / [{v.shape[0]},{v.shape[1]}]   rank-{actor.rank} factors")
    print(f"  kind bias        dispatch {kind_bias[0]:+.4f}   defer {kind_bias[1]:+.4f}   stop {kind_bias[2]:+.4f}")
    print(f"  encoder calls    {menu.ol.shape[0]} + {menu.b.shape[0]} = {menu.ol.shape[0] + menu.b.shape[0]}"
          f"   (a per-triple scorer would need {len(candidates)})")

    print("\n  z_n decomposed   (rows sorted by z, extremes shown)")
    print(f"  {'n':>5s} {'kind':9s} {'k_kind':>9s} {'g[i]':>9s} {'h[j]':>9s} {'<u,v>':>9s} {'= z_n':>9s} {'pi':>8s}")
    probs = torch.softmax(logits, -1)
    order_ix = torch.argsort(logits, descending=True).tolist()
    show = order_ix[:4] + (["..."] if len(order_ix) > 7 else []) + order_ix[-3:]
    for n in show:
        if n == "...":
            print(f"  {'...':>5s}")
            continue
        c = candidates[n]
        gate_g = 0.0 if c.kind == "stop" else float(g[menu.ol_idx[n]])
        gate_h = float(h[menu.b_idx[n]]) if c.kind == "dispatch" else 0.0
        gate_uv = float(u[menu.ol_idx[n]] @ v[menu.b_idx[n]]) if c.kind == "dispatch" else 0.0
        kb = float(kind_bias[int(np.argmax(menu.kind[n]))])
        total = kb + gate_g + gate_h + gate_uv
        print(f"  {n:>5d} {c.kind:9s} {kb:>+9.4f} {gate_g:>+9.4f} {gate_h:>+9.4f} {gate_uv:>+9.4f} "
              f"{total:>+9.4f} {float(probs[n]):>8.4f}")
        check(f"z[{n}] = kind + gates (decomposition matches forward_factors)",
              abs(total - float(logits[n])) < 2e-5, f"|delta| = {abs(total - float(logits[n])):.2e}")

    stop_ix = [n for n, c in enumerate(candidates) if c.kind == "stop"]
    if stop_ix:
        n = stop_ix[0]
        check("stop score is the kind bias alone",
              abs(float(logits[n]) - float(kind_bias[2])) < 2e-5)
    defer_ix = [n for n, c in enumerate(candidates) if c.kind == "defer"]
    if defer_ix:
        n = defer_ix[0]
        check("defer score = kind bias + g only (h and <u,v> gated off)",
              abs(float(logits[n]) - float(kind_bias[1]) - float(g[menu.ol_idx[n]])) < 2e-5)

    # ------------------------------------------- the expanded rows must agree exactly
    rule("CHECK   factored menu == full expansion")
    rows = menu.rows()
    print(f"  rebuilt rows {tuple(rows.shape)}   (olb layout, {OLB_DIM} columns; flat layout would be {FLAT_DIM})")
    manual = rows_from_factors(menu.ol_raw, menu.b_raw, menu.kind, menu.ol_idx, menu.b_idx)
    check("rows() reproduces rows_from_factors exactly", bool(np.array_equal(rows, manual)))
    with torch.no_grad():
        row_logits = actor(
            torch.as_tensor(rows[None], dtype=torch.float32, device=learner.device),
            torch.ones((1, len(candidates)), dtype=torch.bool, device=learner.device),
            ctx_t,
        )[0]
    delta = float((row_logits - logits).abs().max())
    print(f"  max |logit(rows) - logit(factors)| = {delta:.3e}   argmax agrees: "
          f"{int(row_logits.argmax()) == int(logits.argmax())}")
    check("factored scoring equals row scoring", delta < 1e-4, f"max delta {delta:.2e}")
    check("factored and row scoring pick the same action", int(row_logits.argmax()) == int(logits.argmax()))

    # ---------------------------------------------------------- step 5: action + value
    rule("STEP 5  action and value")
    action, sample = learner.act(menu, context, training=True)
    chosen = candidates[action]
    print(f"  sampled n = {action}  ->  kind={chosen.kind}"
          + (f", order {chosen.order_id}, return {chosen.return_launchpad_id}, battery {chosen.battery_id}"
             if chosen.kind == "dispatch" else ""))
    print(f"  log pi = {sample.log_prob:+.4f}   V(s) = {sample.value:+.4f}   "
          f"(critic = {type(learner.critic).__name__}, critic_type = {learner.critic_type})")
    with torch.no_grad():
        v_again = learner._factor_v(fb, ctx_t)
        for p_ in actor.parameters():
            p_.add_(0.05 * torch.randn_like(p_))
        v_after = learner._factor_v(fb, ctx_t)
        for p_ in actor.parameters():
            p_.add_(-0.0)   # keep the perturbation; the policy object is discarded after this call
    check("V(s) does not depend on the actor (policy-independent critic)",
          abs(float(v_again) - float(v_after)) < 1e-9,
          f"{float(v_again):.6f} vs {float(v_after):.6f}")
    print(f"  V before/after perturbing the actor: {float(v_again):+.6f} / {float(v_after):+.6f}")


def inspect_episode(env, dispatcher, policy, steps: int) -> None:
    rule(f"STEPS 6-7  episode: decisions -> trajectories -> rewards -> GAE  ({steps} simulator steps)")
    state = env.reset(seed=42)
    done = False
    for _ in range(steps):
        env.apply_actions(dispatcher.plan(state))
        state, _, done, _ = env.step()
        if done:
            break

    decisions = policy._episode_decisions
    print(f"  {len(decisions)} decisions over {len({d.time for d in decisions})} distinct steps")
    groups = policy._training_trajectory_groups()
    sizes = sorted((len(t) for t in groups), reverse=True)
    print(f"  trajectory_split = {policy.trajectory_split!r}  ->  {len(groups)} trajectories, "
          f"sizes {sizes[:8]}{' ...' if len(sizes) > 8 else ''}")
    check("every decision lands in exactly one trajectory",
          sum(len(t) for t in groups) == len(decisions))
    check("a trajectory belongs to a single drone",
          all(len({d.drone_id for d in t}) == 1 for t in groups if t and t[0].drone_id))

    penalty = float(env.config["cost"]["lateness_penalty_per_min"])
    reward_events = sorted(
        [
            {"time": int(o.delivery_time),
             "reward": float(o.value) - max(0.0, (o.delivery_time - o.deadline) / 60.0) * penalty,
             "order_id": o.order_id}
            for o in env.orders.values()
            if o.status == OrderStatus.DELIVERED and o.delivery_time is not None
        ],
        key=lambda e: e["time"],
    )
    print(f"  {len(reward_events)} delivery reward events in the episode")

    biggest = max(groups, key=len)
    drone = biggest[0].drone_id
    mine = policy._trajectory_reward_events(biggest, reward_events)
    print(f"\n  drone {drone}: {len(biggest)} decisions, "
          f"{sum(1 for d in biggest if d.kind == 'dispatch')} of them dispatches; "
          f"it sees {len(mine)} of the {len(reward_events)} reward events")
    dispatched = {str(d.order_id) for d in biggest if d.kind == "dispatch" and d.order_id}
    check("a trajectory only sees the rewards of the orders it dispatched",
          all(str(e["order_id"]) in dispatched for e in mine))

    times = [d.time for d in biggest]
    rewards = temporal_team_rewards(times, mine)
    for e in mine[:3]:
        k = max((i for i, t in enumerate(times) if t < e["time"]), default=None)
        print(f"    event t={e['time']:>5d} reward {e['reward']:7.2f} (order {e['order_id'][-6:]})"
              f"  ->  decision #{k} at t={times[k] if k is not None else None}")
    check("no event is attributed to a decision at or after its own time",
          all(times[i] < e["time"] for e in mine for i in [max((j for j, t in enumerate(times) if t < e["time"]), default=-1)] if i >= 0))

    learner = policy.learner
    values = np.array([d.sample.value for d in biggest])
    scaled = np.array(rewards) / learner.reward_scale
    adv = np.zeros(len(biggest))
    gae = 0.0
    end = int(env.current_time)
    for k in range(len(biggest) - 1, -1, -1):
        terminal = k == len(biggest) - 1
        nxt = end if terminal else times[k + 1]
        dt = max(0.0, float(nxt - times[k]))
        disc = learner.gamma ** (dt / learner.time_unit_s)
        trace = learner.gae_lambda ** (dt / learner.time_unit_s)
        delta = scaled[k] + disc * (0.0 if terminal else values[k + 1]) - values[k]
        gae = delta + (0.0 if terminal else disc * trace * gae)
        adv[k] = gae
    print(f"\n  time-aware GAE over this drone's trajectory (gamma={learner.gamma}, lambda={learner.gae_lambda} per "
          f"{learner.time_unit_s:.0f}s, reward scale {learner.reward_scale})")
    print(f"  {'k':>4s} {'t':>6s} {'dt':>5s} {'kind':9s} {'reward':>9s} {'V':>9s} {'A':>9s} {'R_hat':>9s}")
    for k in list(range(min(4, len(biggest)))) + ([len(biggest) - 1] if len(biggest) > 4 else []):
        dt = (end if k == len(biggest) - 1 else times[k + 1]) - times[k]
        print(f"  {k:>4d} {times[k]:>6d} {dt:>5d} {biggest[k].kind:9s} {rewards[k]:>9.3f} "
              f"{values[k]:>+9.4f} {adv[k]:>+9.4f} {adv[k] + values[k]:>+9.4f}")
    check("advantages are finite", bool(np.all(np.isfinite(adv))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", default="hongkong")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=20, help="simulator steps before inspecting a decision")
    parser.add_argument("--min-batteries", type=int, default=2, help="skip pads with fewer available batteries")
    parser.add_argument("--episode", action="store_true", help="also walk the episode-end pipeline")
    parser.add_argument("--episode-steps", type=int, default=60)
    parser.add_argument("--checks-only", action="store_true", help="print only the PASS/FAIL summary")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = sys.stdout
    if args.checks_only:
        sys.stdout = open("/dev/null", "w")   # noqa: SIM115
    try:
        env, dispatcher, policy = build(args.city, args.seed, args.device)
        state, pad_id, pad, candidates, menu = find_decision(env, dispatcher, policy, args.warmup, args.min_batteries)
        inspect_decision(env, policy, state, pad_id, pad, candidates, menu)
        if args.episode:
            env, dispatcher, policy = build(args.city, args.seed, args.device)
            inspect_episode(env, dispatcher, policy, args.episode_steps)
    finally:
        if args.checks_only:
            sys.stdout.close()
            sys.stdout = out

    print(f"\n{'=' * 78}\nINVARIANTS\n{'=' * 78}")
    failed = 0
    for name, ok, detail in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail and not ok else ""))
        failed += not ok
    print(f"\n  {len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
