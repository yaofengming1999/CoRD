"""F0OneDecision: the (order, battery) | return pad split, as one decision.

F0 showed what it costs to make the dispatch twice -- a dispatch agent that picks
(order, battery) and a separate agent that picks the pad (-1.8% to -7.1% on-time
GMV, table 3). This keeps F0's grouping of the action, since feasibility couples
the order and the battery, but takes the decision once: one actor scores the
factored menu whose first block is one row per (order, battery) pair and whose
second is one row per return pad.

Why the split matters, and what this measures. The energy margin is

    charge_b - 0.2C - E_deliver(o) - E_return(o, l).

The proposed method groups (o,l), which puts all of E_task into the first block
and leaves charge in the second, so the margin is a row effect plus a column
effect: an additive score represents it exactly, and rank 0 suffices (table 4).
Grouping (o,b) instead puts charge and E_deliver together, but then nothing holds
E_return(o, l) -- the pad block knows only the pad. The coupling has to come from
the bilinear term.

Drones here fly straight lines (air_router uses the haversine distance), so the
geometry is reachable in principle: squared distance is

    ||x_dest||^2 + ||x_pad||^2 - 2<x_dest, x_pad>,

both blocks carry coordinates, the two norms are inside the additive parts, and
the cross term is rank 2 in the plane. But the energy follows the distance, not
its square, and a square root of a bilinear form is not of the form
g + h + <u, v>. So the prediction is a ladder: rank 0 cannot see the return leg
at all, rank 2 buys the geometry up to that square root, and no finite rank
recovers what the (o,l) grouping has for free.

The candidates are not changed. The menu is built exactly as the proposed method
builds it -- same feasibility test, same energy calls, same entries -- and only
then regrouped, so a difference between the two is the split and nothing else.
"""

from __future__ import annotations

from algorithm.F2.policy import F2Policy


class F0OneDecisionOrderBatteryPolicy(F2Policy):
    """One decision over (order, battery) | return pad."""

    factor_split = "ob_l"
    actor_type = "obl"
    joint_plan_feature_schema = "joint_dispatch_ob_l_separable_v1"
