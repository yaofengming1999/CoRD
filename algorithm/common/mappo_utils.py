from __future__ import annotations

from bisect import bisect_left


def mean_unique_agents_per_time(events: list[tuple[int, str]]) -> float:
    """Average active agents over physical times with at least one decision."""
    agents_by_time: dict[int, set[str]] = {}
    for decision_time, agent_id in events:
        agents_by_time.setdefault(int(decision_time), set()).add(str(agent_id))
    if not agents_by_time:
        return 0.0
    return sum(len(agent_ids) for agent_ids in agents_by_time.values()) / len(
        agents_by_time
    )


def temporal_team_rewards(
    decision_times: list[int],
    reward_events: list[dict] | None,
) -> list[float]:
    """Attach each realized team reward to the latest preceding decision.

    GAE propagates that reward to earlier decisions. Decisions made at the
    reward timestamp observe the event and are therefore causally later, so
    the event is assigned strictly before that timestamp. Zero-time GAE links
    still carry it through the preceding decision round without duplication.
    """
    rewards = [0.0] * len(decision_times)
    if not decision_times:
        return rewards
    for event in reward_events or []:
        event_time = int(event.get("time", 0))
        index = bisect_left(decision_times, event_time) - 1
        if index >= 0:
            rewards[index] += float(event.get("reward", 0.0) or 0.0)
    return rewards
