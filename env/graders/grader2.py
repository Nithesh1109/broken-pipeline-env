"""
Grader 2 – Schema Task Grader.

Produces a structured grade report for a SchemaTask episode.
"""

from typing import Any, Dict, List


class SchemaGrader:
    """Grade the result of a Task-2 (schema validation) episode."""

    def grade(
        self,
        expected_violations: List[Dict[str, Any]],
        reported_violations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare the expected violations with those reported by the agent.

        Each violation is a dict with keys: record_index, field, violation_type.
        Returns precision, recall, F1, and a letter grade.
        """

        def _key(v: Dict[str, Any]):
            return (
                v.get("record_index"),
                v.get("field"),
                v.get("violation_type"),
            )

        expected_keys = {_key(v) for v in expected_violations}
        reported_keys = {_key(v) for v in reported_violations}

        tp = reported_keys & expected_keys
        fp = reported_keys - expected_keys
        fn = expected_keys - reported_keys

        precision = len(tp) / len(reported_keys) if reported_keys else 0.0
        recall = len(tp) / len(expected_keys) if expected_keys else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": len(tp),
            "false_positives": len(fp),
            "false_negatives": len(fn),
            "grade": self._letter_grade(f1),
        }

    @staticmethod
    def _letter_grade(score: float) -> str:
        if score >= 0.9:
            return "A"
        if score >= 0.75:
            return "B"
        if score >= 0.6:
            return "C"
        if score >= 0.4:
            return "D"
        return "F"
