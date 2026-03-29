from __future__ import annotations

from collections import Counter

from env.models import DataObservation


def run_task(records: list[dict]) -> DataObservation:
    id_counts = Counter(row.get("employee_id") for row in records)
    duplicate_ids = sum(count - 1 for count in id_counts.values() if count > 1)

    missing_salary = sum(
        1 for row in records if ("salary" not in row or row.get("salary") in (None, ""))
    )
    negative_salary = sum(
        1
        for row in records
        if isinstance(row.get("salary"), (int, float)) and float(row.get("salary", 0)) < 0
    )

    total_issues = duplicate_ids + missing_salary + negative_salary
    status = "ok" if total_issues == 0 else "warning"

    return DataObservation(
        task_id="task1",
        status=status,
        message="Data quality audit complete",
        metrics={
            "record_count": len(records),
            "duplicate_ids": duplicate_ids,
            "missing_salary": missing_salary,
            "negative_salary": negative_salary,
            "total_issues": total_issues,
        },
        payload={
            "checklist": ["duplicates", "missing salary", "negative salary"],
        },
    )