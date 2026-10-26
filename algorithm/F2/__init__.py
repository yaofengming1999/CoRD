"""F2 -- <OL+B+LowRank>+SerializedMARL, the proposed method, self-contained.

policy.py    F2Policy: per-step drone-ordered decisions, per-drone GAE trajectories, learner hand-off
features.py  MenuBuilder: feasible (o,l,b) menu in factored form, decision context, global state
menu.py      FactorMenu and the olb / flat candidate layouts
actor.py     OLBSeparableActor: logit = k + g(s,o,l) + h(s,b) + <u(s,o,l), v(s,b)>
critic.py    FactorVCritic (default), FactorQCritic, CandidateQCritic, GlobalStateAgentCritic, GlobalStateCritic
learner.py   F2Learner: PPO over ragged menus with the actor / critic switches
"""

from algorithm.F2.policy import F2Policy
from algorithm.F2.learner import F2Learner
from algorithm.F2.features import MenuBuilder
from algorithm.F2.menu import FactorMenu

__all__ = ["F2Policy", "F2Learner", "MenuBuilder", "FactorMenu"]
