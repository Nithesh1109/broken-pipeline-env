"""Grader 3 – Scores a Task 3 (Hard) episode deterministically.

Scoring criteria
----------------
inspection_done     : 0.20  – agent issued INSPECT
all_issues_logged   : 0.20  – at least 3 distinct issues found
fix_phase1          : 0.20  – first FIX (type errors) applied
fix_phase2          : 0.20  – second FIX (nulls/duplicates) applied
validated           : 0.20  – VALIDATE succeeded (done=True)
"""

from __future__ import annotations

from typing import Any, Dict

from env.models import GraderResult


def grade(state: Dict[str, Any]) -> GraderResult:
    """Score a completed (or partial) Task 3 episode."""
    breakdown: Dict[str, float] = {
        "inspection_done": 0.0,
        "all_issues_logged": 0.0,
        "fix_phase1": 0.0,
        "fix_phase2": 0.0,
        "validated": 0.0,
    }

    actions = state.get("actions_taken", [])
    issues = state.get("issues_found", [])
    done = state.get("done", False)

    fix_count = actions.count("FIX")

    if "INSPECT" in actions:
        breakdown["inspection_done"] = 0.20

    if len(issues) >= 3:
        breakdown["all_issues_logged"] = 0.20

    if fix_count >= 1:
        breakdown["fix_phase1"] = 0.20

    if fix_count >= 2:
        breakdown["fix_phase2"] = 0.20

    if "VALIDATE" in actions and done:
        breakdown["validated"] = 0.20

    score = round(sum(breakdown.values()), 4)

    explanation_parts = []
    if breakdown["inspection_done"]:
        explanation_parts.append("full inspection done")
    if breakdown["all_issues_logged"]:
        explanation_parts.append("all issue categories discovered")
    if breakdown["fix_phase1"]:
        explanation_parts.append("type errors fixed")
    if breakdown["fix_phase2"]:
        explanation_parts.append("nulls and duplicates removed")
    if breakdown["validated"]:
        explanation_parts.append("pipeline validated clean")
    if not explanation_parts:
        explanation_parts.append("no meaningful actions taken")

    explanation = "Task 3: " + "; ".join(explanation_parts) + f". Score: {score}."

    return GraderResult(
        task_id="task3",
        score=score,
        breakdown=breakdown,
        explanation=explanation,
    )
