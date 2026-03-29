"""
Task 3 – Incident Response.

The agent is given a pipeline incident report (symptoms, timeline,
system metrics) and must identify the root cause from a set of
candidates as well as propose mitigation steps.
"""

from typing import Any, Dict, List, Tuple

from env.models import Action, Observation


class IncidentTask:
    """Evaluate an agent's ability to diagnose and respond to a pipeline incident."""

    max_steps: int = 10

    def __init__(self, scenario: Dict[str, Any]):
        self._scenario = scenario
        self._report: Dict[str, Any] = scenario["incident_report"]
        self._root_cause: str = scenario["root_cause"]
        self._accepted_causes: List[str] = scenario.get(
            "accepted_causes", [self._root_cause]
        )
        self._mitigation_keywords: List[str] = scenario.get(
            "mitigation_keywords", []
        )
        self._diagnosis: str = ""
        self._mitigation: List[str] = []
        self._solved = False

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def get_observation(self, step: int) -> Observation:
        return Observation(
            task_id="task3_incident",
            step=step,
            data={"incident_report": self._report},
            errors=[],
            metadata={"root_cause_candidates": self._accepted_causes},
        )

    def evaluate_action(
        self, action: Action, step: int
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Expected action:
          action_type = "diagnose"
          payload     = {
              "root_cause": <str>,
              "mitigation_steps": [<str>, ...]
          }
        """
        if action.action_type != "diagnose":
            return -0.1, False, {"error": "Unknown action type."}

        diagnosis: str = action.payload.get("root_cause", "")
        mitigation: List[str] = action.payload.get("mitigation_steps", [])

        cause_correct = diagnosis in self._accepted_causes
        cause_reward = 0.6 if cause_correct else 0.0

        mitigation_text = " ".join(mitigation).lower()
        matched_keywords = [
            kw
            for kw in self._mitigation_keywords
            if kw.lower() in mitigation_text
        ]
        mitigation_score = (
            len(matched_keywords) / len(self._mitigation_keywords)
            if self._mitigation_keywords
            else (1.0 if mitigation else 0.0)
        )
        mitigation_reward = round(0.4 * mitigation_score, 4)

        reward = round(cause_reward + mitigation_reward, 4)
        self._diagnosis = diagnosis
        self._mitigation = mitigation
        self._solved = cause_correct and mitigation_score >= 0.5

        done = self._solved
        info = {
            "cause_correct": cause_correct,
            "mitigation_score": mitigation_score,
            "matched_keywords": matched_keywords,
        }
        return reward, done, info

    def is_solved(self) -> bool:
        return self._solved

    def get_details(self) -> Dict[str, Any]:
        return {
            "root_cause": self._root_cause,
            "agent_diagnosis": self._diagnosis,
            "mitigation_steps": self._mitigation,
        }
