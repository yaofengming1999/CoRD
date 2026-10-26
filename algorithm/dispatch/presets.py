from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmPreset:
    order_battery_policy: str
    drone_return_policy: str
    rebalance_enabled: bool
    trainable: bool
    description: str
    # Learned methods are named <action_expansion>+<agent_structure>; both axes
    # are recorded here so tables/tags can be derived instead of hand-written.
    action_expansion: str = ""
    agent_structure: str = ""

    @property
    def label(self) -> str:
        if self.action_expansion and self.agent_structure:
            return f"{self.action_expansion}+{self.agent_structure}"
        return ""


# Action-expansion axis (how one dispatch decision is scored/decoded):
OLB_LOWRANK = "<OL+B+LowRank>"          # olb layout, (o,l)|b separable actor (proposed)
FULL_COMBINATION = "FullCombination"    # flat layout, full-capacity actor over the triplet
AUTOREGRESSIVE_OLB = "AutoregressiveOLB"  # staged o -> l -> b decode
INDEPENDENT_OLB = "IndependentOLB"      # F1b: o, l, b drawn independently, no conditioning
TWO_STAGE_OB = "TwoStage(o,b)|l"        # F0: (o,b) and l decided by two separate agents
MOE_POINTER = "MoE-Pointer (KDD'26)"
MAPDP_POINTER = "MAPDP-Pointer (AAAI'22)"   # entity encoder + cooperative 3-step pointer decode
MSRA_MATCHING = "MSRA-Matching (TRC'25)"    # bipartite order matching on U = R + lambda*ontime - C + V, GC risk constraint

# Agent-structure axis (who is the agent, how the stream is cut for PPO):
SERIALIZED_MARL = "SerializedMARL"      # drones decide in sequence, one GAE trajectory per drone (proposed)
SINGLE_DRONE_PPO = "SingleDronePPO"     # drones decide in sequence, ONE trajectory per episode
MAPPO = "MAPPO"                         # synchronous joint step, central critic
GLOBAL_DRONES_PPO = "GlobalDronesPPO"   # one joint transition per physical step, global critic
IPPO = "IPPO"                           # independent learners: synchronous, per-drone reward + trajectory, local critic


ACTIVE_ALGORITHM_NAMES = (
    # R*: rule-based baselines
    "R1_urgentLower_nearest",
    "R2_random_random",
    "R3_fifoLower_backlogHeu",
    "R4_fifoFull_backlogHeu",
    "R5_hungarianGmv_backlogHeu",
    # L*: learned baselines from the literature
    "L1_RBG",
    "L2_MSRA",
    "L3_MAPDP",
    "L4_MoEPointer",        # MoE-Pointer (KDD'26)+GlobalDronesPPO
    "L5_HRL4AG",
    # AS*: architecture/credit-assignment variants of our own stack
    "AS1_MAPPO",
    "AS2_SingleDronePPO", 
    "AS3_GlobalDronesPPO",
    "AS4_SyncOwnReward",   # AS1 + own-order reward (credit under simultaneous decode)
    "AS6_BatteryBlind",    # no battery state, random battery (rank one)
    "F0_OneDecision",      # (order, battery) | return pad: the other factorisation
    "F2pro",               # o | l | b, interaction on (o,l) only
    "AS5_TeamReward",      # F2 + team reward (reward attribution, cut held fixed)
    # F*: feature-side reform variants of our own stack
    "F3_FullCombination",  # SeqBatR: full triplet expansion (control) follow AS2_SingleDronePPO
    "F2_Separable",        # main proposal: (o,l)|b separable follow AS2_SingleDronePPO
    "F1_StagedOLB",        # staged o -> l -> b decode follow AS2_SingleDronePPO
    "F1b_IndependentOLB",  # o, l, b drawn independently (no conditioning at all)
    "F0_TwoStage",         # dispatch agent (o,b) + separate return agent (l)
)


ALGORITHM_PRESETS: dict[str, AlgorithmPreset] = {
    "R1_urgentLower_nearest": AlgorithmPreset(
        "heuristic", "nearest", False, False,
        "Greedy order-battery dispatch with nearest feasible return.",
    ),
    "R2_random_random": AlgorithmPreset(
        "random", "random", False, False,
        "Random feasible order-battery dispatch and return.",
    ),
    "R3_fifoLower_backlogHeu": AlgorithmPreset(
        "lowest_feasible", "heuristic", False, False,
        "FIFO orders use the lowest-charge feasible battery; heuristic return.",
    ),
    "R4_fifoFull_backlogHeu": AlgorithmPreset(
        "full_charge_fifo", "heuristic", False, False,
        "FIFO orders use only fully charged batteries; heuristic return.",
    ),
    "R5_hungarianGmv_backlogHeu": AlgorithmPreset(
        "hungarian_ontime_gmv", "heuristic", False, False,
        "Hungarian order-battery matching maximising predicted on-time GMV; heuristic return.",
    ),
    "AS2_SingleDronePPO": AlgorithmPreset(
        "drone_trajectory_joint_plan_ppo", "preplanned", False, True,
        "Drone-ordered sequential decisions, ONE PPO trajectory per episode (team reward); "
        "proposed (o,l)|b input. Differs from F2 only in the trajectory cut.",
        OLB_LOWRANK, SINGLE_DRONE_PPO,
    ),
    "F3_FullCombination": AlgorithmPreset(
        "drone_ordered_counterfactual_ppo", "preplanned", False, True,
        "Full flat (order, return, battery) triplet menu (flat) scored by a full-capacity "
        "actor; SerializedMARL structure. Action-expansion control for F2.",
        FULL_COMBINATION, SERIALIZED_MARL,
    ),
    "L4_MoEPointer": AlgorithmPreset(
        "moe_pointer", "preplanned", False, True,
        "MoE-Pointer (KDD'26): causal MoE-pointer decoder over battery then return pad, "
        "one joint sequence / value per step (GlobalDronesPPO structure).",
        MOE_POINTER, GLOBAL_DRONES_PPO,
    ),
    # Literature baseline implemented with the two-stage simulator interface.
    "L1_RBG": AlgorithmPreset(
        "rbg_svm_auction", "heuristic", False, True,
        "L1 RBG (AAMAS'25): per-battery online SVM bidding, least-confident auction.",
    ),
    "L2_MSRA": AlgorithmPreset(
        "msra", "preplanned", False, True,
        "MSRA-MARL (TRC'25) adapter, drone side: bipartite order matching per launchpad on "
        "U = value + lambda_p*ontime - cost + gamma*V, IPPO return-pad policy with local critic, "
        "Graph Coordinator (pad-graph GNN, BCE+MSE) as matching hard constraint Risk <= G.",
        MSRA_MATCHING, IPPO,
    ),
    "L3_MAPDP": AlgorithmPreset(
        "mapdp", "preplanned", False, True,
        "MAPDP (AAAI'22) adapter: paired-context entity encoder, cooperative parallel "
        "pointer decoders (order -> return pad -> battery), joint reward, global-state critic.",
        MAPDP_POINTER, MAPPO,
    ),
    "L5_HRL4AG": AlgorithmPreset(
        "hrl4ag", "preplanned", False, True,
        "L5 HRL4AG (KDD'26) adapter: UAV worker with the ground-fleet manager branch fixed off.",
    ),
    "AS1_MAPPO": AlgorithmPreset(
        "drone_synchronous_mappo", "preplanned", False, True,
        "Synchronous per-drone actors, one joint action per physical time, central critic; "
        "proposed (o,l)|b input.",
        OLB_LOWRANK, MAPPO,
    ),
    "AS3_GlobalDronesPPO": AlgorithmPreset(
        "global_ppo", "preplanned", False, True,
        "One joint action/reward/transition per physical time with an agent-invariant global "
        "set critic; proposed (o,l)|b input.",
        OLB_LOWRANK, GLOBAL_DRONES_PPO,
    ),
    "F1_StagedOLB": AlgorithmPreset(
        "staged_order_launchpad_battery_ppo", "preplanned", False, True,
        "Autoregressive order -> return-launchpad -> battery decode with cheap stage-1 "
        "feasibility (never forms the flat triplet product); SerializedMARL structure.",
        AUTOREGRESSIVE_OLB, SERIALIZED_MARL,
    ),
    "F1b_IndependentOLB": AlgorithmPreset(
        "independent_order_launchpad_battery_ppo", "preplanned", False, True,
        "F1 with the sequential conditioning removed: order, return-launchpad and "
        "battery are drawn by three heads that all read the same state context and "
        "whose menus are built against fixed representatives, so no head sees any "
        "other's choice. The triple's joint feasibility is checked only after all "
        "three have drawn; an infeasible triple dispatches nothing. Prices the "
        "conditioning alone -- features, PPO math and trajectory cut are F1's.",
        INDEPENDENT_OLB, SERIALIZED_MARL,
    ),
    "F0_TwoStage": AlgorithmPreset(
        "f0_dispatch_only_ppo", "f0_learned_ppo", False, True,
        "Two learned agents that never share a menu: dispatch picks (order, battery) "
        "with the return axis collapsed to the energy-minimal pad, and a separate "
        "return agent picks the pad once the drone is at the endpoint. Joint-action "
        "control for F2 (same scorer, same SerializedMARL credit assignment).",
        TWO_STAGE_OB, SERIALIZED_MARL,
    ),
    "F2pro": AlgorithmPreset(
        "f2pro_ppo", "preplanned", False, True,
        "Three separate blocks -- orders, return pads, batteries -- additive, with a "
        "rank-r interaction on the (order, pad) pair only, since that is the pair the "
        "return energy couples. Rows per decision fall from N_o*N_l + N_b to "
        "N_o + N_l + N_b, which removes the tail (2596 -> 169 on meituan-asn).",
        OLB_LOWRANK, SERIALIZED_MARL,
    ),
    "F0_OneDecision": AlgorithmPreset(
        "ob_l_one_decision_ppo", "preplanned", False, True,
        "F0's grouping taken in one decision: a factored menu of (order, battery) "
        "rows against return-pad rows. The return energy straddles the two blocks, "
        "so unlike the proposed (o,l) | b split it is not additively separable.",
        OLB_LOWRANK, SERIALIZED_MARL,
    ),
    "AS6_BatteryBlind": AlgorithmPreset(
        "d2_separable_battery_blind_ppo", "preplanned", False, True,
        "The proposed method with the battery blanked from the actor's input and "
        "drawn at random among the feasible ones: the score matrix drops from rank "
        "two to rank one. Prices the battery preference, feasibility held fixed.",
        OLB_LOWRANK, SERIALIZED_MARL,
    ),
    "AS4_SyncOwnReward": AlgorithmPreset(
        "drone_synchronous_mappo_own_reward", "preplanned", False, True,
        "AS1 with the OWN-ORDER reward: same simultaneous decode and rule conflict "
        "resolution, but one GAE trajectory per drone over the steps it acted in, and "
        "V(s, drone) in place of the team V(s). Isolates the credit under simultaneity.",
        OLB_LOWRANK, MAPPO,
    ),
    "AS5_TeamReward": AlgorithmPreset(
        "d2_separable_team_reward_ppo", "preplanned", False, True,
        "F2 with the TEAM reward on per-drone trajectories: same decode, same cut, "
        "only the reward stream each trajectory carries changes. Isolates the reward "
        "attribution, which AS2 cannot (AS2 moves the cut as well).",
        OLB_LOWRANK, SERIALIZED_MARL,
    ),
    "F2_Separable": AlgorithmPreset(
        "d2_separable_ppo", "preplanned", False, True,
        "PROPOSED: (o,l)|b separable actor on the olb layout, SerializedMARL structure "
        "(one GAE trajectory per drone). The yaml defaults are interaction_rank 0 and "
        "critic_type factor_v; an earlier version of this string said rank 8 and a "
        "candidate-Q critic, which the configuration has never set.",
        OLB_LOWRANK, SERIALIZED_MARL,
    ),
}

# Literature names above are now runnable adapters; this tuple remains for
# future literature stubs only.
PLANNED_ALGORITHM_NAMES = ()

CUSTOM_ALGORITHM = "drone_decision"
ALGORITHM_CHOICES = (CUSTOM_ALGORITHM, *ACTIVE_ALGORITHM_NAMES, *PLANNED_ALGORITHM_NAMES)


def apply_algorithm_preset(config: dict, algorithm: str) -> AlgorithmPreset | None:
    if algorithm == CUSTOM_ALGORITHM:
        config.setdefault("_runtime", {})["algorithm"] = algorithm
        return None
    try:
        preset = ALGORITHM_PRESETS[algorithm]
    except KeyError as exc:
        raise ValueError(f"Unknown algorithm preset: {algorithm}") from exc

    decision = config.setdefault("drone_decision", {})
    decision["order_battery_policy"] = preset.order_battery_policy
    decision["drone_return_policy"] = preset.drone_return_policy
    decision.setdefault("rebalance", {})["enabled"] = preset.rebalance_enabled
    runtime = config.setdefault("_runtime", {})
    runtime["algorithm"] = algorithm
    runtime["algorithm_preset"] = {
        "order_battery_policy": preset.order_battery_policy,
        "drone_return_policy": preset.drone_return_policy,
        "rebalance_enabled": preset.rebalance_enabled,
        "trainable": preset.trainable,
        "description": preset.description,
    }
    return preset
