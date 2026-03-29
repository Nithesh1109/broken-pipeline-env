"""
Task 1 – Audit Log Analysis.

The agent is presented with a list of pipeline audit-log entries, some of
which contain injected anomalies (e.g. unexpected status codes, missing
fields, out-of-range values).  The agent must identify every anomalous
entry by its index.
"""

from typing import Any, Dict, List, Tuple

from env.models import Action, Observation


class AuditTask:
    """Evaluate an agent's ability to audit pipeline log entries."""

    max_steps: int = 10

    def __init__(self, scenario: Dict[str, Any]):
        self._scenario = scenario
        self._logs: List[Dict[str, Any]] = scenario["logs"]
        self._anomaly_indices: List[int] = scenario["anomaly_indices"]
        self._identified: List[int] = []
        self._solved = False

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def get_observation(self, step: int) -> Observation:
        return Observation(
            task_id="task1_audit",
            step=step,
            data={"logs": self._logs},
            errors=[],
            metadata={
                "total_entries": len(self._logs),
                "identified_so_far": list(self._identified),
            },
        )

    def evaluate_action(
        self, action: Action, step: int
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Expected action:
          action_type = "identify_anomalies"
          payload     = {"indices": [<int>, ...]}
        """
        if action.action_type != "identify_anomalies":
            return -0.1, False, {"error": "Unknown action type."}

        submitted: List[int] = action.payload.get("indices", [])
        correct = set(self._anomaly_indices)
        submitted_set = set(submitted)

        true_positives = submitted_set & correct
        false_positives = submitted_set - correct
        false_negatives = correct - submitted_set

        precision = (
            len(true_positives) / len(submitted_set) if submitted_set else 0.0
        )
        recall = (
            len(true_positives) / len(correct) if correct else 1.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        self._identified = list(submitted_set)
        self._solved = len(false_positives) == 0 and len(false_negatives) == 0

        reward = round(f1, 4)
        done = self._solved
        info = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": list(false_positives),
            "false_negatives": list(false_negatives),
        }
        return reward, done, info

    def is_solved(self) -> bool:
        return self._solved

    def get_details(self) -> Dict[str, Any]:
        return {
            "anomaly_indices": self._anomaly_indices,
            "identified": self._identified,
        }
