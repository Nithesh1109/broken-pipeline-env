from __future__ import annotations

from env.models import DataAction, DataObservation, GraderResult


def grade(action: DataAction, observation: DataObservation, ground_truth: dict) -> GraderResult:
    severity = str(observation.metrics.get("severity", "low"))
    expected_severity = str(ground_truth.get("severity", "high"))
    severity_score = 1.0 if severity == expected_severity else 0.5 if severity in {"medium", "high"} else 0.0

    expected_actions = [a.lower() for a in ground_truth.get("expected_actions", [])]
    proposed = " ".join(action.remediations).lower()
    action_hits = sum(1 for item in expected_actions if item in proposed)
    action_score = action_hits / max(1, len(expected_actions))

    score = min(1.0, 0.5 * severity_score + 0.5 * action_score)
    passed = score >= 0.7

    feedback = [
        f"Incident severity observed: {severity} (expected {expected_severity}).",
        f"Matched expected action steps: {action_hits}/{max(1, len(expected_actions))}.",
    ]
    return GraderResult(task_id="task3", score=score, passed=passed, feedback=feedback)