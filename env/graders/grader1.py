from __future__ import annotations

from env.models import DataAction, DataObservation, GraderResult


def grade(action: DataAction, observation: DataObservation, ground_truth: dict) -> GraderResult:
    expected = max(1, int(ground_truth.get("total_issues", 1)))
    found = int(observation.metrics.get("total_issues", 0))
    detection_score = min(found / expected, 1.0)

    action_bonus = 0.0
    if action.findings:
        action_bonus += 0.1
    if action.remediations:
        action_bonus += 0.1

    score = min(1.0, detection_score + action_bonus)
    passed = score >= 0.7

    feedback = [
        f"Detected {found}/{expected} expected issues.",
        "Included findings." if action.findings else "No findings submitted.",
        "Included remediations." if action.remediations else "No remediations submitted.",
    ]

    return GraderResult(task_id="task1", score=score, passed=passed, feedback=feedback)