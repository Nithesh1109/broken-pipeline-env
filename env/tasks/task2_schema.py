"""
Task 2 – Schema Validation.

The agent receives a list of records and a target schema definition.
Some records deliberately violate the schema (wrong types, missing
required fields, values outside allowed ranges).  The agent must
return the list of (record_index, field_name, violation_type) tuples
that describe every violation.
"""

from typing import Any, Dict, List, Tuple

from env.models import Action, Observation


class SchemaTask:
    """Evaluate an agent's ability to detect schema violations."""

    max_steps: int = 10

    def __init__(self, scenario: Dict[str, Any]):
        self._scenario = scenario
        self._records: List[Dict[str, Any]] = scenario["records"]
        self._schema: Dict[str, Any] = scenario["schema"]
        self._violations: List[Dict[str, Any]] = scenario["violations"]
        self._reported: List[Dict[str, Any]] = []
        self._solved = False

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def get_observation(self, step: int) -> Observation:
        return Observation(
            task_id="task2_schema",
            step=step,
            data={"records": self._records, "schema": self._schema},
            errors=[],
            metadata={"reported_so_far": list(self._reported)},
        )

    def evaluate_action(
        self, action: Action, step: int
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Expected action:
          action_type = "report_violations"
          payload     = {
              "violations": [
                  {"record_index": <int>, "field": <str>, "violation_type": <str>},
                  ...
              ]
          }
        """
        if action.action_type != "report_violations":
            return -0.1, False, {"error": "Unknown action type."}

        submitted: List[Dict[str, Any]] = action.payload.get("violations", [])

        def _key(v: Dict[str, Any]) -> Tuple:
            return (v.get("record_index"), v.get("field"), v.get("violation_type"))

        expected_keys = {_key(v) for v in self._violations}
        submitted_keys = {_key(v) for v in submitted}

        tp = submitted_keys & expected_keys
        fp = submitted_keys - expected_keys
        fn = expected_keys - submitted_keys

        precision = len(tp) / len(submitted_keys) if submitted_keys else 0.0
        recall = len(tp) / len(expected_keys) if expected_keys else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        self._reported = submitted
        self._solved = len(fp) == 0 and len(fn) == 0

        reward = round(f1, 4)
        done = self._solved
        info = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": len(fp),
            "false_negatives": len(fn),
        }
        return reward, done, info

    def is_solved(self) -> bool:
        return self._solved

    def get_details(self) -> Dict[str, Any]:
        return {
            "expected_violations": self._violations,
            "reported_violations": self._reported,
        }
