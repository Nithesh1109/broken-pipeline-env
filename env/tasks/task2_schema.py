from __future__ import annotations

from env.models import DataObservation


EXPECTED_COLUMNS = {"employee_id", "name", "department", "salary", "start_date"}


def run_task(records: list[dict]) -> DataObservation:
    salary_as_string = sum(1 for row in records if isinstance(row.get("salary"), str))
    team_used = sum(1 for row in records if "team" in row and "department" not in row)

    columns_seen: set[str] = set()
    for row in records:
        columns_seen.update(row.keys())

    unexpected_columns = sorted(col for col in columns_seen if col not in EXPECTED_COLUMNS)

    remediated: list[dict] = []
    for row in records:
        fixed = dict(row)
        if "team" in fixed and "department" not in fixed:
            fixed["department"] = fixed.pop("team")
        if isinstance(fixed.get("salary"), str):
            try:
                fixed["salary"] = int(fixed["salary"])
            except ValueError:
                fixed["salary"] = None
        for extra_key in list(fixed.keys()):
            if extra_key not in EXPECTED_COLUMNS:
                fixed.pop(extra_key)
        remediated.append(fixed)

    drift_count = salary_as_string + team_used + max(0, len(unexpected_columns) - 1) * len(records)

    return DataObservation(
        task_id="task2",
        status="warning" if drift_count > 0 else "ok",
        message="Schema drift analysis complete",
        metrics={
            "record_count": len(records),
            "salary_as_string": salary_as_string,
            "team_used_instead_of_department": team_used,
            "drift_count": drift_count,
        },
        payload={
            "unexpected_columns": unexpected_columns,
            "remediated_preview": remediated[:3],
        },
    )