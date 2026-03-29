"""
Grader 3 – Incident Task Grader.

Produces a structured grade report for an IncidentTask episode.
"""

from typing import Any, Dict, List


class IncidentGrader:
    """Grade the result of a Task-3 (incident response) episode."""

    def grade(
        self,
        root_cause: str,
        accepted_causes: List[str],
        mitigation_keywords: List[str],
        agent_diagnosis: str,
        agent_mitigation: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate root-cause identification and mitigation quality.

        Returns scores for each dimension and an overall letter grade.
        """
        cause_correct = agent_diagnosis in accepted_causes
        cause_score = 1.0 if cause_correct else 0.0

        mitigation_text = " ".join(agent_mitigation).lower()
        matched = [kw for kw in mitigation_keywords if kw.lower() in mitigation_text]
        mitigation_score = (
            len(matched) / len(mitigation_keywords)
            if mitigation_keywords
            else (1.0 if agent_mitigation else 0.0)
        )

        overall = round(0.6 * cause_score + 0.4 * mitigation_score, 4)

        return {
            "cause_correct": cause_correct,
            "cause_score": cause_score,
            "mitigation_score": round(mitigation_score, 4),
            "matched_keywords": matched,
            "overall_score": overall,
            "grade": self._letter_grade(overall),
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
