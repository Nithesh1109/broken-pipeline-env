"""
Scenario Generator.

Creates synthetic scenarios for the three broken-pipeline tasks.
All scenarios are pure-Python dicts that can be serialised to JSON.
"""

import random
from typing import Any, Dict, List


class ScenarioGenerator:
    """Generate synthetic task scenarios with configurable bug injection."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Task 1 – Audit Log Scenarios
    # ------------------------------------------------------------------

    def generate_audit_scenario(
        self,
        num_logs: int = 20,
        num_anomalies: int = 4,
    ) -> Dict[str, Any]:
        """Return a dict compatible with AuditTask."""
        statuses = ["SUCCESS", "SUCCESS", "SUCCESS", "WARNING"]
        logs: List[Dict[str, Any]] = []
        for i in range(num_logs):
            logs.append(
                {
                    "index": i,
                    "timestamp": f"2024-01-01T{i % 24:02d}:00:00Z",
                    "stage": f"stage_{i % 5}",
                    "status": self._rng.choice(statuses),
                    "duration_ms": self._rng.randint(100, 500),
                    "records_processed": self._rng.randint(1000, 5000),
                }
            )

        anomaly_indices = self._rng.sample(range(num_logs), min(num_anomalies, num_logs))
        for idx in anomaly_indices:
            anomaly_type = self._rng.choice(["bad_status", "negative_duration", "zero_records"])
            if anomaly_type == "bad_status":
                logs[idx]["status"] = "ERROR"
            elif anomaly_type == "negative_duration":
                logs[idx]["duration_ms"] = -1
            else:
                logs[idx]["records_processed"] = 0

        return {
            "task_id": "task1_audit",
            "logs": logs,
            "anomaly_indices": sorted(anomaly_indices),
        }

    # ------------------------------------------------------------------
    # Task 2 – Schema Validation Scenarios
    # ------------------------------------------------------------------

    def generate_schema_scenario(
        self,
        num_records: int = 15,
        num_violations: int = 3,
    ) -> Dict[str, Any]:
        """Return a dict compatible with SchemaTask."""
        schema = {
            "fields": {
                "id": {"type": "int", "required": True},
                "name": {"type": "str", "required": True},
                "age": {"type": "int", "required": True, "min": 0, "max": 120},
                "score": {"type": "float", "required": False},
            }
        }

        records: List[Dict[str, Any]] = []
        for i in range(num_records):
            records.append(
                {
                    "id": i,
                    "name": f"user_{i}",
                    "age": self._rng.randint(18, 65),
                    "score": round(self._rng.uniform(0.0, 100.0), 2),
                }
            )

        violation_indices = self._rng.sample(
            range(num_records), min(num_violations, num_records)
        )
        violations: List[Dict[str, Any]] = []
        violation_types = ["wrong_type", "out_of_range", "missing_required"]
        for idx in violation_indices:
            vtype = self._rng.choice(violation_types)
            if vtype == "wrong_type":
                records[idx]["age"] = "not_a_number"
                violations.append(
                    {"record_index": idx, "field": "age", "violation_type": "wrong_type"}
                )
            elif vtype == "out_of_range":
                records[idx]["age"] = 200
                violations.append(
                    {"record_index": idx, "field": "age", "violation_type": "out_of_range"}
                )
            else:
                del records[idx]["name"]
                violations.append(
                    {
                        "record_index": idx,
                        "field": "name",
                        "violation_type": "missing_required",
                    }
                )

        return {
            "task_id": "task2_schema",
            "schema": schema,
            "records": records,
            "violations": violations,
        }

    # ------------------------------------------------------------------
    # Task 3 – Incident Response Scenarios
    # ------------------------------------------------------------------

    def generate_incident_scenario(self) -> Dict[str, Any]:
        """Return a dict compatible with IncidentTask."""
        causes = [
            "memory_leak",
            "disk_full",
            "network_partition",
            "deadlock",
            "config_error",
        ]
        cause = self._rng.choice(causes)
        mitigations = {
            "memory_leak": {
                "keywords": ["restart", "memory", "heap", "profiler"],
                "steps": [
                    "Restart the affected service to free heap memory.",
                    "Attach a memory profiler to identify the leak source.",
                    "Review recent code changes for unclosed resources.",
                ],
            },
            "disk_full": {
                "keywords": ["disk", "cleanup", "archive", "storage"],
                "steps": [
                    "Delete or archive old log files to free disk space.",
                    "Add additional disk storage.",
                    "Configure log rotation to prevent recurrence.",
                ],
            },
            "network_partition": {
                "keywords": ["network", "partition", "reconnect", "timeout"],
                "steps": [
                    "Check network connectivity between pipeline nodes.",
                    "Increase timeout thresholds.",
                    "Implement retry logic with exponential back-off.",
                ],
            },
            "deadlock": {
                "keywords": ["deadlock", "lock", "transaction", "rollback"],
                "steps": [
                    "Identify and kill blocked transactions.",
                    "Review locking order in database queries.",
                    "Implement deadlock detection and automatic rollback.",
                ],
            },
            "config_error": {
                "keywords": ["config", "configuration", "rollback", "validate"],
                "steps": [
                    "Roll back to the last known-good configuration.",
                    "Validate configuration files before deployment.",
                    "Add configuration schema checks to CI pipeline.",
                ],
            },
        }

        m = mitigations[cause]
        return {
            "task_id": "task3_incident",
            "incident_report": {
                "summary": f"Pipeline failure detected. Root cause: [{cause}] (hidden).",
                "symptoms": [
                    "Job execution stalled for > 10 minutes.",
                    "Alert fired: pipeline SLA breach.",
                ],
                "timeline": [
                    {"time": "T-30m", "event": "Normal operations."},
                    {"time": "T-10m", "event": "Latency spike observed."},
                    {"time": "T-0m", "event": "Pipeline halted."},
                ],
                "metrics": {
                    "cpu_usage": "35%",
                    "memory_usage": "87%",
                    "disk_usage": "52%",
                },
            },
            "root_cause": cause,
            "accepted_causes": [cause],
            "mitigation_keywords": m["keywords"],
            "suggested_mitigation": m["steps"],
        }
