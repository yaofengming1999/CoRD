"""Dispatch algorithms.

F2/            the proposed method (one place): policy.py (decision loop, trajectories),
               features.py (MenuBuilder: candidates, factored (o,l)|b menu, context, global
               state), menu.py (FactorMenu / layouts), actor.py, critic.py, learner.py (PPO)
F2_separable.py, F3_full_combination.py, F1_staged_olb.py       action-expansion axis (F* = F2 + one swap)
AS1_mappo.py, AS2_single_drone_ppo.py, AS3_global_drones_ppo.py agent-structure axis (AS* = F2 + one swap;
               AS1/AS3 also host the MAPPO / GlobalDronesPPO structure classes that L3 / L4 build on)
L1_rbg.py .. L5_hrl4ag.py                                      literature baselines
R_rule_based.py                                                rule baselines R1-R5
common/        shared plumbing: torch PPO/GAE scaffolding, full-combination actor, metrics, geometry
dispatch/      DroneDecisionDispatcher, preset registry, rule return policies
"""
