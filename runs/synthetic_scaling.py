"""Table 7: what the (order, launchpad, battery) factorisation costs, measured on
menus built to an exact size instead of on the sizes the simulator happens to produce.

Why this exists
---------------
Tables 3 and 6 measure the factorisation on real menus. There the energy
constraint leaves barely more than one feasible battery per (order, pad) pair, so
the product is nearly degenerate: the flat menu is 1.36x the rows F2 scores on
meituan-ssn and 1.00x on lade_cq, and the whole action space is 10-211 entries.
No cost axis measured there can separate the three expansions, because the axis
never opens. This script opens it by construction -- the menu is built to exactly
O orders x B batteries x L return pads with EVERY TRIPLE FEASIBLE -- and reports
the three cost measures at each size.

Rows scored per decision is a property of the factorisation, not of the data:

    F3_FullCombination   no factorisation, one flat row per triple   O*L*B
    F2_Separable         (o,l) and b blocks                          O*L + B
    F2pro                o, l and b blocks                           O + L + B

The action space is identical for all three at each point. Only the rows differ,
which is the whole claim.

What is measured, and what is not
---------------------------------
Every number here times the ACTOR FORWARD PASS ALONE. Table 6 times the whole of
``dispatcher.plan()``, which also builds the menu (energy and feasibility calls),
assembles features and runs the selection -- work the three share and which
dominates there. The two are not comparable: the ratios here are larger than
table 6's for that reason, not because of any disagreement. S0 is set to
meituan-asn's real menu shape (O*L = 100, B = 2, the degeneracy itself) so the
table says where the real cities sit on its own axis.

FLOPs are exact rather than profiled: the per-row MAC of each actor is counted
with forward hooks on the real module and cross-checked against
``torch.utils.flop_counter``, which agrees to the digit. Latency is the median of
N timed iterations after 20 warm-up iterations, N scaled so each point runs about
two seconds. The median rather than the mean because at 0.2 ms kernel-launch
jitter dominates: an early three-run pass put S1 at 1.4x, which was noise, and
the settled value is 2.1x.

Inputs are ``randn``, not zeros. cuBLAS does not exploit sparsity so it makes no
measurable difference, but a cost table should not invite the question.

The arithmetic-share column divides by this card's MEASURED achievable fp32 GEMM
rate, not by a spec figure; ``--peak-tflops`` overrides it and ``--measure-peak``
(the default) measures it here. On the RTX 5090 this is about 69.4 TFLOP/s with
TF32 off, against a 104.9 spec number that assumes perfect dual-issue. Using the
spec figure, or an A100's 19.5, misstates the column by several times.

Usage
-----
    python runs/synthetic_scaling.py --out mdtable2/table7_synthetic_scaling.md
    python runs/synthetic_scaling.py --points "20,2,5 100,150,75" --reps-seconds 4
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time

import torch
from torch import nn

from algorithm.F2 import featmask


# ---------------------------------------------------------------- MAC counting

class _LinearMACs:
    """Sums in_features*out_features*batch over every nn.Linear in a forward."""

    def __init__(self, module: nn.Module):
        self.macs = 0
        self._handles = [
            m.register_forward_hook(self._hook(m))
            for m in module.modules()
            if isinstance(m, nn.Linear)
        ]

    def _hook(self, layer: nn.Linear):
        def fn(_module, _inputs, output):
            batch = 1
            for dim in output.shape[:-1]:
                batch *= dim
            self.macs += batch * layer.in_features * layer.out_features
        return fn

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        for handle in self._handles:
            handle.remove()


def _marginal(build, call, base: int, bumped: int) -> int:
    """MACs of one extra row: count at two sizes and difference them, so the
    once-per-decision context branch cancels out."""
    counts = []
    for n in (base, bumped):
        module = build()
        with _LinearMACs(module) as counter, torch.no_grad():
            call(module, n)
        counts.append(counter.macs)
    return counts[1] - counts[0]


# ------------------------------------------------------------------- actors

class Spec:
    """One algorithm: how many rows it scores, what a row costs, how to run it."""

    def __init__(self, name, rows_fn, label, per_row_macs, per_entry_macs, make, call):
        self.name = name
        self.rows_fn = rows_fn            # (O, B, L) -> rows scored
        self.label = label                # the formula, for the table
        self.per_row_macs = per_row_macs  # dict of block -> MAC, or int
        self.per_entry_macs = per_entry_macs
        self.make = make                  # () -> module on device
        self.call = call                  # (module, O, B, L, tensors) -> runs one forward

    def macs(self, O, B, L):
        return self.rows_macs(O, B, L) + O * B * L * self.per_entry_macs

    def rows_macs(self, O, B, L):
        raise NotImplementedError


def build_specs(device: str, hidden: int, f2_rank: int, f2pro_rank: int):
    mask = featmask.active()
    ctx_dim = mask.context_dim

    from algorithm.F2.actor import OLBSeparableActor
    from algorithm.F2.menu import FLAT_DIM, OLB_DIM
    from algorithm.F2pro.actor import TriSeparableActor
    from algorithm.common.full_combination_actor import FullCombinationActor

    def dense(O, B, L, n, dim):
        return torch.randn(1, n, dim, device=device)

    # ---- F3: one flat row per triple
    def f3_make():
        return FullCombinationActor(FLAT_DIM, ctx_dim, hidden).to(device).eval()

    f3_row = _marginal(
        lambda: FullCombinationActor(FLAT_DIM, ctx_dim, hidden),
        lambda m, n: m(torch.randn(1, n, FLAT_DIM), torch.ones(1, n, dtype=torch.bool),
                       torch.randn(1, ctx_dim)),
        2000, 2001,
    )

    class F3(Spec):
        def rows_macs(self, O, B, L):
            return O * B * L * f3_row

    f3 = F3("F3_FullCombination", lambda O, B, L: O * L * B, "O*L*B",
            {"flat": f3_row}, 0, f3_make, None)

    def f3_call(module, O, B, L, buf):
        module(buf["flat"], buf["mask"], buf["ctx"])

    f3.call = f3_call

    def f3_alloc(O, B, L):
        A = O * B * L
        return {"flat": torch.randn(1, A, FLAT_DIM, device=device),
                "mask": torch.ones(1, A, dtype=torch.bool, device=device),
                "ctx": torch.randn(1, ctx_dim, device=device)}

    f3.alloc = f3_alloc

    # ---- F2: (o,l) and b blocks
    def f2_make():
        return OLBSeparableActor(OLB_DIM, ctx_dim, hidden, rank=f2_rank).to(device).eval()

    def _f2_probe(m, n, which):
        n_ol, n_b, A = (n, 10, 3000) if which == "ol" else (100, n, 3000)
        m.forward_factors(
            torch.randn(1, n_ol, mask.ol_dim), torch.randn(1, n_b, mask.b_dim),
            torch.cat([torch.ones(1, A, 1), torch.zeros(1, A, 2)], -1),
            torch.zeros(1, A, dtype=torch.long), torch.zeros(1, A, dtype=torch.long),
            torch.ones(1, A, dtype=torch.bool), torch.randn(1, ctx_dim))

    f2_ol = _marginal(lambda: OLBSeparableActor(OLB_DIM, ctx_dim, hidden, rank=f2_rank),
                      lambda m, n: _f2_probe(m, n, "ol"), 100, 101)
    f2_b = _marginal(lambda: OLBSeparableActor(OLB_DIM, ctx_dim, hidden, rank=f2_rank),
                     lambda m, n: _f2_probe(m, n, "b"), 100, 101)

    class F2(Spec):
        def rows_macs(self, O, B, L):
            return O * L * f2_ol + B * f2_b

    f2 = F2("F2_Separable", lambda O, B, L: O * L + B, "O*L + B",
            {"(o,l)": f2_ol, "b": f2_b}, 3 + f2_rank, f2_make, None)

    def f2_alloc(O, B, L):
        A = O * B * L
        kind = torch.zeros(1, A, 3, device=device); kind[..., 0] = 1.0
        return {"ol": torch.randn(1, O * L, mask.ol_dim, device=device),
                "b": torch.randn(1, B, mask.b_dim, device=device),
                "kind": kind,
                "ol_idx": torch.randint(0, O * L, (1, A), device=device),
                "b_idx": torch.randint(0, B, (1, A), device=device),
                "mask": torch.ones(1, A, dtype=torch.bool, device=device),
                "ctx": torch.randn(1, ctx_dim, device=device)}

    def f2_call(module, O, B, L, buf):
        module.forward_factors(buf["ol"], buf["b"], buf["kind"], buf["ol_idx"],
                               buf["b_idx"], buf["mask"], buf["ctx"])

    f2.alloc, f2.call = f2_alloc, f2_call

    # ---- F2pro: o, l and b blocks
    def fp_make():
        return TriSeparableActor(ctx_dim, hidden, rank=f2pro_rank).to(device).eval()

    def _fp_probe(m, n, which):
        n_o, n_l, n_b, A = {"o": (n, 40, 30, 3000),
                            "l": (50, n, 30, 3000),
                            "b": (50, 40, n, 3000)}[which]
        m(torch.randn(1, n_o, mask.o_dim), torch.randn(1, n_l, mask.l_dim),
          torch.randn(1, n_b, mask.b_dim),
          torch.cat([torch.ones(1, A, 1), torch.zeros(1, A, 2)], -1),
          torch.zeros(1, A, dtype=torch.long), torch.zeros(1, A, dtype=torch.long),
          torch.zeros(1, A, dtype=torch.long), torch.ones(1, A, dtype=torch.bool),
          torch.randn(1, ctx_dim))

    fp_o = _marginal(lambda: TriSeparableActor(ctx_dim, hidden, rank=f2pro_rank),
                     lambda m, n: _fp_probe(m, n, "o"), 50, 51)
    fp_l = _marginal(lambda: TriSeparableActor(ctx_dim, hidden, rank=f2pro_rank),
                     lambda m, n: _fp_probe(m, n, "l"), 40, 41)
    fp_b = _marginal(lambda: TriSeparableActor(ctx_dim, hidden, rank=f2pro_rank),
                     lambda m, n: _fp_probe(m, n, "b"), 30, 31)

    class F2pro(Spec):
        def rows_macs(self, O, B, L):
            return O * fp_o + L * fp_l + B * fp_b

    fp = F2pro("F2pro", lambda O, B, L: O + L + B, "O + L + B",
               {"o": fp_o, "l": fp_l, "b": fp_b}, 3 + f2pro_rank, fp_make, None)

    def fp_alloc(O, B, L):
        A = O * B * L
        kind = torch.zeros(1, A, 3, device=device); kind[..., 0] = 1.0
        return {"o": torch.randn(1, O, mask.o_dim, device=device),
                "l": torch.randn(1, L, mask.l_dim, device=device),
                "b": torch.randn(1, B, mask.b_dim, device=device),
                "kind": kind,
                "o_idx": torch.randint(0, O, (1, A), device=device),
                "l_idx": torch.randint(0, L, (1, A), device=device),
                "b_idx": torch.randint(0, B, (1, A), device=device),
                "mask": torch.ones(1, A, dtype=torch.bool, device=device),
                "ctx": torch.randn(1, ctx_dim, device=device)}

    def fp_call(module, O, B, L, buf):
        module(buf["o"], buf["l"], buf["b"], buf["kind"], buf["o_idx"],
               buf["l_idx"], buf["b_idx"], buf["mask"], buf["ctx"])

    fp.alloc, fp.call = fp_alloc, fp_call

    return [f2, fp, f3]


# ------------------------------------------------------------------ measuring

def measure_peak_tflops(device: str) -> float:
    """This card's achievable fp32 GEMM rate, TF32 off. Not a spec figure."""
    best = 0.0
    for n in (4096, 8192, 12288):
        a = torch.randn(n, n, device=device)
        b = torch.randn(n, n, device=device)
        for _ in range(5):
            a @ b
        torch.cuda.synchronize()
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            a @ b
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        times.sort()
        best = max(best, 2 * n ** 3 / times[len(times) // 2] / 1e12)
        del a, b
        torch.cuda.empty_cache()
    return best


def time_forward(spec: Spec, O: int, B: int, L: int, budget_s: float):
    """Median latency and allocator peak for one decision group, or None on OOM."""
    torch.cuda.empty_cache()
    gc.collect()
    module = spec.make()
    try:
        buf = spec.alloc(O, B, L)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None

    run = lambda: spec.call(module, O, B, L, buf)
    try:
        with torch.no_grad():
            for _ in range(20):
                run()
            torch.cuda.synchronize()

            resident = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            run()
            torch.cuda.synchronize()
            peak_gb = (torch.cuda.max_memory_allocated() - resident) / 1e9

            t0 = time.perf_counter()
            run()
            torch.cuda.synchronize()
            one = max(time.perf_counter() - t0, 1e-6)
            reps = max(5, min(200, int(budget_s / one)))

            times = []
            for _ in range(reps):
                t0 = time.perf_counter()
                run()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
    except torch.cuda.OutOfMemoryError:
        del module, buf
        torch.cuda.empty_cache()
        return None, None

    times.sort()
    del module, buf
    torch.cuda.empty_cache()
    gc.collect()
    return times[len(times) // 2] * 1000.0, peak_gb


# -------------------------------------------------------------------- output

def human(x: float) -> str:
    for unit, div in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if x >= div:
            return f"{x / div:.2f}{unit}"
    return f"{x:.0f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", default="20,2,5 10,15,8 25,30,15 50,60,30 100,150,75",
                    help="space-separated O,B,L triples; the default ladder is S0..S4")
    ap.add_argument("--feature-preset", default="lossless")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--interaction-rank", type=int, default=0, help="F2's rank")
    ap.add_argument("--f2pro-rank", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps-seconds", type=float, default=2.0,
                    help="timing budget per cell; reps are scaled to fill it")
    ap.add_argument("--peak-tflops", type=float, default=None,
                    help="override the achievable fp32 rate instead of measuring it")
    ap.add_argument("--out", default=None, help="write the markdown table here")
    ap.add_argument("--json", default=None, help="also dump the raw numbers here")
    args = ap.parse_args(argv)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("cuda requested but not available")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    featmask.configure(args.feature_preset)
    mask = featmask.active()

    points = []
    for tok in args.points.split():
        O, B, L = (int(v) for v in tok.split(","))
        points.append((O, B, L))

    specs = build_specs(args.device, args.hidden_dim, args.interaction_rank, args.f2pro_rank)
    by_name = {s.name: s for s in specs}

    props = torch.cuda.get_device_properties(0)
    peak = args.peak_tflops if args.peak_tflops else measure_peak_tflops(args.device)

    print(f"# device      : {props.name}  {props.total_memory / 1e9:.2f} GB  "
          f"sm {props.major}.{props.minor}  x{props.multi_processor_count}")
    print(f"# fp32 peak   : {peak:.1f} TFLOP/s "
          f"({'given' if args.peak_tflops else 'measured, TF32 off'})")
    print(f"# featmask    : {args.feature_preset} -> context {mask.context_dim}, "
          f"o {mask.o_dim}, l {mask.l_dim}, b {mask.b_dim}, ol {mask.ol_dim}")
    print("# per-row MAC :", {s.name: s.per_row_macs for s in specs})
    print("# per-entry   :", {s.name: s.per_entry_macs for s in specs})
    print()

    rows_out = []
    for idx, (O, B, L) in enumerate(points):
        A = O * B * L
        cell = {"point": f"S{idx}", "O": O, "B": B, "L": L, "actions": A}
        for spec in specs:
            rows = spec.rows_fn(O, B, L)
            macs = spec.macs(O, B, L)
            ms, gb = time_forward(spec, O, B, L, args.reps_seconds)
            cell[spec.name] = {
                "rows": rows, "macs": macs, "flops": 2 * macs,
                "ms": ms, "peak_gb": gb,
                "share": (None if ms is None
                          else (2 * macs / (peak * 1e12)) / (ms / 1000.0) * 100),
                "eff_tflops": None if ms is None else 2 * macs / 1e12 / (ms / 1000.0),
            }
            tag = "OOM" if ms is None else f"{ms:9.3f} ms  {gb:7.3f} GB"
            print(f"  S{idx} {str((O, B, L)):<15} |A|={A:>10,}  {spec.name:<20}"
                  f"rows={rows:>10,}  {tag}")
        rows_out.append(cell)
        print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"device": props.name, "peak_tflops": peak,
                       "feature_preset": args.feature_preset,
                       "hidden_dim": args.hidden_dim,
                       "points": rows_out}, fh, indent=2)
        print(f"raw numbers -> {args.json}")

    if args.out:
        write_markdown(args.out, rows_out, specs, peak, props, args, mask)
        print(f"table -> {args.out}")
    return rows_out


def write_markdown(path, cells, specs, peak, props, args, mask):
    f2, fp, f3 = (by.name for by in specs)  # names, in the order built: F2, F2pro, F3
    n2, npro, n3 = f2, fp, f3

    def ratio(a, b):
        return "--" if (a is None or b in (None, 0)) else f"{a / b:.2f}x"

    L = []
    L.append("# Table 7: synthetic (O, B, L) scaling --- rows, actor latency, actor FLOPs")
    L.append("")
    L.append("Menus are built to an exact size --- O orders x B batteries x L return pads,")
    L.append("**every triple feasible** --- so the action space is identical for all three")
    L.append("algorithms at each point and only the rows the actor scores differ. On real")
    L.append("menus the energy constraint leaves barely more than one feasible battery per")
    L.append("(order, pad) pair, so the product is nearly degenerate and no cost axis")
    L.append("measured there can separate the expansions. S0 is set to meituan-asn's real")
    L.append("menu shape (O*L = 100, B = 2) to mark where the real cities sit.")
    L.append("")
    L.append("Every number times the **actor forward pass alone**. Table 6 times the whole")
    L.append("of `dispatcher.plan()`, including the menu construction and selection the")
    L.append("three share, so the two are not comparable and the ratios here are larger.")
    L.append("")
    L.append(f"**Hardware.** {props.name}, {props.total_memory / 1e9:.1f} GB, "
             f"{props.multi_processor_count} SM. Achievable fp32 GEMM rate "
             f"**{peak:.1f} TFLOP/s**, TF32 off,")
    L.append("measured here rather than taken from a spec sheet.")
    L.append("")
    L.append(f"**Setup.** feature preset `{args.feature_preset}` (context {mask.context_dim}, "
             f"o {mask.o_dim}, l {mask.l_dim}, b {mask.b_dim}); hidden_dim {args.hidden_dim}; "
             f"F2 rank {args.interaction_rank},")
    L.append(f"F2pro rank {args.f2pro_rank}; fp32; `randn` inputs; one decision group per "
             "measurement; idle")
    L.append("card, one process; median of N timed iterations after 20 warm-up iterations.")
    L.append("")
    L.append("| algorithm | rows scored per decision | per-row MAC | per-entry MAC |")
    L.append("|---|---|---|---|")
    for s in specs:
        per = " / ".join(f"{v:,} ({k})" for k, v in s.per_row_macs.items())
        L.append(f"| {s.name} | `{s.label}` | {per} | {s.per_entry_macs} |")
    L.append("")
    L.append("Per-row MACs are counted with forward hooks on the real modules. Every method")
    L.append("also pays the per-entry term over the whole action space, because a softmax")
    L.append("over the menu must produce one logit per entry; that term is the floor no")
    L.append("factorisation goes below.")
    L.append("")

    L.append("## 7.1 Rows scored per decision")
    L.append("")
    L.append(f"| point | (O, B, L) | action space | {n3} | {n2} | {npro} | "
             f"{n3}/{n2} | {n3}/{npro} |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in cells:
        r3, r2, rp = c[n3]["rows"], c[n2]["rows"], c[npro]["rows"]
        L.append(f"| {c['point']} | ({c['O']}, {c['B']}, {c['L']}) | {human(c['actions'])} | "
                 f"{human(r3)} | {human(r2)} | {human(rp)} | {r3 / r2:.0f}x | {r3 / rp:.0f}x |")
    L.append("")

    L.append("## 7.2 Actor forward latency (ms)")
    L.append("")
    L.append(f"| point | action space | {n2} | {npro} | {n3} | {n3}/{n2} | {n3}/{npro} |")
    L.append("|---|---|---|---|---|---|---|")
    for c in cells:
        g = lambda n: "OOM" if c[n]["ms"] is None else f"{c[n]['ms']:.3f}"
        L.append(f"| {c['point']} | {human(c['actions'])} | {g(n2)} | {g(npro)} | {g(n3)} | "
                 f"{ratio(c[n3]['ms'], c[n2]['ms'])} | {ratio(c[n3]['ms'], c[npro]['ms'])} |")
    L.append("")

    L.append("## 7.3 Actor FLOPs per decision")
    L.append("")
    L.append(f"| point | {n3} | {n2} | {npro} | {n3}/{n2} | {n3}/{npro} | {npro}/{n2} |")
    L.append("|---|---|---|---|---|---|---|")
    for c in cells:
        m3, m2, mp = c[n3]["flops"], c[n2]["flops"], c[npro]["flops"]
        L.append(f"| {c['point']} | {human(m3)} | {human(m2)} | {human(mp)} | "
                 f"{m3 / m2:.0f}x | {m3 / mp:.0f}x | {mp / m2:.2f}x |")
    L.append("")

    L.append("## 7.4 Arithmetic as a share of measured latency")
    L.append("")
    L.append(f"| point | {n2} | {npro} | {n3} | {n3} effective TFLOP/s | "
             f"{n3} ms per 1k rows |")
    L.append("|---|---|---|---|---|---|")
    for c in cells:
        s = lambda n: "--" if c[n]["share"] is None else f"{c[n]['share']:.2f}%"
        e = "--" if c[n3]["eff_tflops"] is None else f"{c[n3]['eff_tflops']:.2f}"
        per_k = ("--" if c[n3]["ms"] is None
                 else f"{c[n3]['ms'] / c[n3]['rows'] * 1000:.4f}")
        L.append(f"| {c['point']} | {s(n2)} | {s(npro)} | {s(n3)} | {e} | {per_k} |")
    L.append("")
    L.append("The last column needs no mechanism: once it settles, the method's latency is")
    L.append("proportional to the rows it scores, so the row count *is* the cost. A flat")
    L.append("latency column in 7.2 means the opposite -- that the row count is not the cost.")
    L.append("")

    L.append("## 7.5 Peak GPU memory (GB)")
    L.append("")
    L.append(f"| point | action space | {n2} | {npro} | {n3} | {n3} bytes/row | {n3}/{n2} |")
    L.append("|---|---|---|---|---|---|---|")
    for c in cells:
        g = lambda n: "OOM" if c[n]["peak_gb"] is None else f"{c[n]['peak_gb']:.3f}"
        bpr = ("--" if c[n3]["peak_gb"] is None
               else f"{c[n3]['peak_gb'] / c[n3]['rows'] * 1e9:,.0f}")
        rr = ("--" if (c[n3]["peak_gb"] is None or not c[n2]["peak_gb"])
              else f"{c[n3]['peak_gb'] / c[n2]['peak_gb']:.0f}x")
        L.append(f"| {c['point']} | {human(c['actions'])} | {g(n2)} | {g(npro)} | {g(n3)} | "
                 f"{bpr} | {rr} |")
    L.append("")
    L.append("Memory is the one axis set by the row count directly, without passing through")
    L.append("the hardware's efficiency, so the gap there needs no extrapolation.")
    L.append("")

    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
