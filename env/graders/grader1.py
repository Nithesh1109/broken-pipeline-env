"""Grader 1: score the Easy task (null detection/fixing)."""

from typing import Any, Dict

from env.models import GraderResult


def grade(state: Dict[str, Any], history: list) -> GraderResult:
    """Deterministically score a completed task1_easy episode.

    Scoring
    -------
    - Base score is the proportion of required null-columns fixed.
    - Small efficiency bonus if completed within 4 steps.
    - Score capped at 1.0.
    """
    target: list = state.get("target_columns", [])
    fixed: list = state.get("fixed_columns", [])

    if not target:
        return GraderResult(score=1.0, details={"msg": "No issues to fix"})

    proportion_fixed = len(set(fixed) & set(target)) / len(target)

    steps_used = len(history)
    efficiency_bonus = 0.1 if steps_used <= 4 and proportion_fixed == 1.0 else 0.0

    score = min(1.0, round(proportion_fixed + efficiency_bonus, 4))
    return GraderResult(
        score=score,
        details={
            "target_columns": target,
            "fixed_columns": fixed,
            "proportion_fixed": proportion_fixed,
            "steps_used": steps_used,
            "efficiency_bonus": efficiency_bonus,
        },
    )
