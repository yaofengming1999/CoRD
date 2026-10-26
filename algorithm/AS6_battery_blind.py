"""AS6: the proposed method with no battery state and a random battery.

Table 4 reads the score of one decision as a matrix over the (order, return pad)
rows and the battery columns,

    Z = g 1' + 1 h' + U V',

and finds r=0 enough: rank two, a row effect plus a column effect. This drops the
column effect too. The battery block of the actor's input is blanked, so every
entry sharing an (order, return pad) ties and the actor is choosing the pair
alone; the battery is then drawn uniformly from the ones that could have served
that pair. The score matrix is rank one.

What the comparison isolates. Feasibility is untouched -- the menu is still built
from the real charges, so a drawn battery can always fly the task, and the two
rows face the same action set. What AS6 gives up is only the preference among
feasible batteries: how much margin to leave, and whether to spend a nearly full
battery on a short errand. If AS6 matches r=0, that preference is worth nothing
beyond feasibility, and the battery is not a decision at all but a constraint.
"""

from __future__ import annotations

from algorithm.F2.policy import F2Policy


class AS6BatteryBlindOrderBatteryPolicy(F2Policy):
    """AS6 = F2 with the battery blanked from the input and drawn at random."""

    battery_blind = True
