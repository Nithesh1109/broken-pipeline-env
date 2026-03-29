from __future__ import annotations

from env.models import DataAction, DataObservation, GraderResult


def grade(action: DataAction, observation: DataObservation, ground_truth: dict) -> GraderResult:
    drift_count = int(observation.metrics.get("drift_count", 0))
    expected_drift = int(ground_truth.get("drift_count", 1))
    detection_score = min(drift_count / max(1, expected_drift), 1.0)

    rem_text = " ".join(action.remediations).lower()
    remediation_checks = ["rename", "department", "cast", "salary", "drop", "region"]
    remediation_hits = sum(1 for token in remediation_checks if token in rem_text)
    remediation_score = min(remediation_hits / 3, 1.0)

    score = min(1.0, 0.6 * detection_score + 0.4 * remediation_score)
    passed = score >= 0.7

    feedback = [
        f"Observed drift count: {drift_count}.",
        f"Remediation keyword hits: {remediation_hits}.",
    ]
    return GraderResult(task_id="task2", score=score, passed=passed, feedback=feedback)