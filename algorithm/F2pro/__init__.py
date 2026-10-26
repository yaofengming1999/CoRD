"""F2pro: three separate blocks, with the interaction only between order and pad.

The proposed method (algorithm/F2) scores a decision as a factored menu of
(order, return pad) rows against battery rows, and table 4 shows it needs no
bilinear coupling between the two: with one battery capacity the margin

    charge_b - 0.2C - E_task(o, l)

is a row effect plus a column effect, so an additive score is exact.

That says the battery couples to the rest additively. It says nothing about the
order and the pad, which the proposed method keeps fused into one block -- and
that fusion is what costs rows: N_o*N_l + N_b, with a long tail when many orders
meet many pads. Splitting them gives N_o + N_l + N_b, which on the cities
measured cuts the worst-case rows of a decision by 11-15x and removes the tail
(meituan-asn: 2596 -> 169) while the mean barely moves.

The price is that E_task(o, l) is no longer a feature of any block: it is a
property of the pair. It has to come from the interaction term instead, and since
drones fly straight lines the return leg is a Euclidean distance -- the square
root of a bilinear form, whose cross term is rank 2 in the plane but whose square
root no finite rank reproduces. So F2pro measures what a precomputed pair feature
is worth, and the expectation is that rank 2 buys the geometry and still falls
short of having E_task outright.

Reuse is confined to candidate generation: F2pro takes the candidate list the F2
menu builder produces, so both face exactly the same actions and feasibility, and
a difference between them is the scoring structure alone. Everything on the
scoring side -- the menu container, the actor, the critic, the learner -- is in
this package.
"""
