"""
Grader 1 – Audit Task Grader.

Produces a structured grade report for an AuditTask episode.
"""

from typing import Any, Dict, List


class AuditGrader:
    """Grade the result of a Task-1 (audit) episode."""

    def grade(
        self,
        expected_anomalies: List[int],
        identified_anomalies: List[int],
    ) -> Dict[str, Any]:
        """
        Compare the expected anomaly indices with those identified by the agent.

        Returns a dict containing precision, recall, F1, and a letter grade.
        """
        expected = set(expected_anomalies)
        identified = set(identified_anomalies)

        tp = identified & expected
        fp = identified - expected
        fn = expected - identified

        precision = len(tp) / len(identified) if identified else 0.0
        recall = len(tp) / len(expected) if expected else 1.0
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
