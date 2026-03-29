from __future__ import annotations

from copy import deepcopy


def inject_task1_bugs(records: list[dict]) -> tuple[list[dict], dict]:
    broken = deepcopy(records)

    # Duplicate IDs.
    broken.extend([deepcopy(broken[0]), deepcopy(broken[1]), deepcopy(broken[2])])

    # Missing salary values.
    broken[5]["salary"] = None
    broken[8]["salary"] = None

    # Impossible salary.
    broken[11]["salary"] = -100

    ground_truth = {
        "duplicate_ids": 3,
        "missing_salary": 2,
        "negative_salary": 1,
        "total_issues": 6,
    }
    return broken, ground_truth


def inject_task2_bugs(records: list[dict]) -> tuple[list[dict], dict]:
    broken = deepcopy(records)

    # Drift to `team` and mixed salary typing.
    for idx, row in enumerate(broken):
        row["team"] = row.pop("department")
        if idx < 12:
            row["salary"] = str(row["salary"])

    # Introduce extra column unexpected by consumer contract.
    for row in broken[:8]:
        row["region"] = "us-east"

    ground_truth = {
        "expected_columns": ["employee_id", "name", "department", "salary", "start_date"],
        "salary_should_be_int": True,
        "drift_count": 20,
    }
    return broken, ground_truth


def inject_task3_bugs(records: list[dict]) -> tuple[list[dict], dict]:
    broken = deepcopy(records)

    # Trigger multi-symptom incident.
    broken[2]["employee_id"] = broken[1]["employee_id"]
    broken[4]["start_date"] = "2099-01-01"
    broken[6]["department"] = None
    broken[7]["department"] = None
    broken[9]["salary"] = 0

    ground_truth = {
        "severity": "high",
        "expected_actions": [
            "isolate bad records",
            "rollback schema change",
            "backfill critical fields",
            "add monitoring alerts",
        ],
    }
    return broken, ground_truth