from __future__ import annotations

from collections import Counter
from datetime import date

from env.models import DataObservation


def run_task(records: list[dict]) -> DataObservation:
    id_counts = Counter(row.get("employee_id") for row in records)
    duplicate_employee_ids = sum(count - 1 for count in id_counts.values() if count > 1)

    today = date.today().isoformat()
    future_start_dates = sum(1 for row in records if str(row.get("start_date", "")) > today)
    missing_departments = sum(1 for row in records if row.get("department") in (None, ""))
    zero_or_negative_salary = sum(
        1
        for row in records
        if isinstance(row.get("salary"), (int, float)) and float(row["salary"]) <= 0
    )

    incident_count = (
        duplicate_employee_ids + future_start_dates + missing_departments + zero_or_negative_salary
    )
    severity = "high" if incident_count >= 3 else "medium" if incident_count > 0 else "low"

    action_plan = [
        "isolate bad records",
        "rollback schema change",
        "backfill critical fields",
        "add monitoring alerts",
    ]

    return DataObservation(
        task_id="task3",
        status="critical" if severity == "high" else "warning" if severity == "medium" else "ok",
        message="Incident triage complete",
        metrics={
            "incident_count": incident_count,
            "duplicate_employee_ids": duplicate_employee_ids,
            "future_start_dates": future_start_dates,
            "missing_departments": missing_departments,
            "zero_or_negative_salary": zero_or_negative_salary,
            "severity": severity,
        },
        payload={
            "action_plan": action_plan,
            "pager": "data-platform-oncall",
        },
    )