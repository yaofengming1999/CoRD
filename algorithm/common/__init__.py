"""Shared plumbing that is not specific to one method.

torch_learner_base.py   BaseTorchMAPPOLearner: PPO/GAE lifecycle, rollout storage, optimisers, save/load
full_combination_actor.py  FullCombinationActor: scores expanded flat triplet rows (F3, flat-actor variants)
decision_metrics.py     per-episode decision counters (D1 metrics)
geometry_utils.py       haversine distance, urgency, safe ratios
mappo_utils.py          temporal team rewards, unique-agents-per-time
"""
