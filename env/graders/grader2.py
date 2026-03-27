"""Grader 2 – Scores a Task 2 (Medium) episode deterministically.

Scoring criteria
----------------
inspection_done  : 0.25  – agent issued INSPECT
issues_found     : 0.25  – issues list is non-empty
fix_applied      : 0.25  – agent issued FIX
validated        : 0.25  – agent issued VALIDATE after fixing
"""

from __future__ import annotations

from typing import Any, Dict

from env.models import GraderResult


def grade(state: Dict[str, Any]) -> GraderResult:
    """Score a completed (or partial) Task 2 episode."""
    breakdown: Dict[str, float] = {
        "inspection_done": 0.0,
        "issues_found": 0.0,
        "fix_applied": 0.0,
        "validated": 0.0,
    }

    actions = state.get("actions_taken", [])
    issues = state.get("issues_found", [])
    done = state.get("done", False)

    if "INSPECT" in actions:
        breakdown["inspection_done"] = 0.25

    if issues:
        breakdown["issues_found"] = 0.25

    if "FIX" in actions:
        breakdown["fix_applied"] = 0.25

    if "VALIDATE" in actions and done:
        breakdown["validated"] = 0.25

    score = round(sum(breakdown.values()), 4)

    explanation_parts = []
    if breakdown["inspection_done"]:
        explanation_parts.append("data inspected")
    if breakdown["issues_found"]:
        explanation_parts.append("schema issues identified")
    if breakdown["fix_applied"]:
        explanation_parts.append("fix attempted")
    if breakdown["validated"]:
        explanation_parts.append("schema validated successfully")
    if not explanation_parts:
        explanation_parts.append("no meaningful actions taken")

    explanation = "Task 2: " + "; ".join(explanation_parts) + f". Score: {score}."

    return GraderResult(
        task_id="task2",
        score=score,
        breakdown=breakdown,
        explanation=explanation,
    )
