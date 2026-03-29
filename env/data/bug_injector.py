"""
Bug Injector.

Utility for deliberately introducing bugs into pipeline data structures
so that the environment can present challenging scenarios to agents.
"""

import random
from typing import Any, Dict, List


class BugInjector:
    """Inject bugs into scenario data for the broken-pipeline tasks."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Audit log bugs
    # ------------------------------------------------------------------

    def inject_audit_bugs(
        self,
        logs: List[Dict[str, Any]],
        num_bugs: int = 3,
    ) -> List[int]:
        """
        Inject *num_bugs* anomalies into *logs* (in-place).

        Returns the sorted list of affected indices.
        """
        indices = self._rng.sample(range(len(logs)), min(num_bugs, len(logs)))
        for idx in indices:
            bug_type = self._rng.choice(
                ["error_status", "negative_duration", "zero_records", "missing_field"]
            )
            if bug_type == "error_status":
                logs[idx]["status"] = "ERROR"
            elif bug_type == "negative_duration":
                logs[idx]["duration_ms"] = -abs(logs[idx].get("duration_ms", 100))
            elif bug_type == "zero_records":
                logs[idx]["records_processed"] = 0
            else:
                logs[idx].pop("stage", None)
        return sorted(indices)

    # ------------------------------------------------------------------
    # Record / schema bugs
    # ------------------------------------------------------------------

    def inject_schema_bugs(
        self,
        records: List[Dict[str, Any]],
        schema: Dict[str, Any],
        num_bugs: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Inject *num_bugs* schema violations into *records* (in-place).

        Returns a list of violation descriptors:
          {"record_index": int, "field": str, "violation_type": str}
        """
        fields = list(schema.get("fields", {}).keys())
        if not fields:
            return []

        indices = self._rng.sample(range(len(records)), min(num_bugs, len(records)))
        violations: List[Dict[str, Any]] = []

        for idx in indices:
            field = self._rng.choice(fields)
            field_def = schema["fields"][field]
            bug_type = self._rng.choice(["wrong_type", "out_of_range", "missing_required"])

            if bug_type == "wrong_type" and field_def.get("type") == "int":
                records[idx][field] = "invalid"
                violations.append(
                    {"record_index": idx, "field": field, "violation_type": "wrong_type"}
                )
            elif bug_type == "out_of_range" and "max" in field_def:
                records[idx][field] = field_def["max"] + 9999
                violations.append(
                    {"record_index": idx, "field": field, "violation_type": "out_of_range"}
                )
            elif bug_type == "missing_required" and field_def.get("required"):
                records[idx].pop(field, None)
                violations.append(
                    {
                        "record_index": idx,
                        "field": field,
                        "violation_type": "missing_required",
                    }
                )
            else:
                records[idx][field] = None
                violations.append(
                    {"record_index": idx, "field": field, "violation_type": "null_value"}
                )

        return violations

    # ------------------------------------------------------------------
    # Incident bugs
    # ------------------------------------------------------------------

    def inject_incident_bug(
        self, report: Dict[str, Any], bug_type: str = "memory_leak"
    ) -> Dict[str, Any]:
        """
        Augment an incident *report* dict with symptoms consistent with
        *bug_type*.  Returns the modified report.
        """
        symptom_map: Dict[str, str] = {
            "memory_leak": "Memory usage increasing steadily over time.",
            "disk_full": "Write operations failing with 'No space left on device'.",
            "network_partition": "Intermittent connection timeouts between nodes.",
            "deadlock": "Multiple transactions waiting indefinitely for locks.",
            "config_error": "Service failing to start after recent configuration change.",
        }
        symptom = symptom_map.get(bug_type, "Unknown anomaly detected.")
        report.setdefault("symptoms", []).append(symptom)
        report["injected_bug"] = bug_type
        return report
