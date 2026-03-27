"""Grader 1 – Scores a Task 1 (Easy) episode deterministically.

Scoring criteria
----------------
inspection_done  : 0.40  – agent issued INSPECT at least once
issues_found     : 0.30  – agent discovered the null columns
validated        : 0.30  – agent issued VALIDATE after inspecting
"""

from __future__ import annotations

from typing import Any, Dict

from env.models import GraderResult


def grade(state: Dict[str, Any]) -> GraderResult:
    """Score a completed (or partial) Task 1 episode.

    Parameters
    ----------
    state:
        The dict returned by ``OpenEnvEnvironment.state()``.

    Returns
    -------
    GraderResult
    """
    breakdown: Dict[str, float] = {
        "inspection_done": 0.0,
        "issues_found": 0.0,
        "validated": 0.0,
    }

    actions = state.get("actions_taken", [])
    issues = state.get("issues_found", [])
    done = state.get("done", False)

    # 1. Inspection credit
    if "INSPECT" in actions:
        breakdown["inspection_done"] = 0.40

    # 2. Issues discovered
    if issues:
        breakdown["issues_found"] = 0.30

    # 3. Validation credit
    if "VALIDATE" in actions and done:
        breakdown["validated"] = 0.30

    score = round(sum(breakdown.values()), 4)

    explanation_parts = []
    if breakdown["inspection_done"]:
        explanation_parts.append("data was inspected")
    if breakdown["issues_found"]:
        explanation_parts.append("null columns were identified")
    if breakdown["validated"]:
        explanation_parts.append("validation completed successfully")
    if not explanation_parts:
        explanation_parts.append("no meaningful actions taken")

    explanation = "Task 1: " + "; ".join(explanation_parts) + f". Score: {score}."

    return GraderResult(
        task_id="task1",
        score=score,
        breakdown=breakdown,
        explanation=explanation,
    )
