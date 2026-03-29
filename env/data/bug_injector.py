import json
from pathlib import Path

import pandas as pd

from env.models import DetectedIssue


# Canonical matching function — used by ALL graders
def matches_ground_truth(detected: DetectedIssue, truth: dict) -> bool:
    """
    Returns True if detected issue matches a ground truth entry.
    Compares issue_type AND column only. Severity is irrelevant.
    """
    return detected.issue_type == truth["type"] and detected.column == truth.get("column")


def load_scenario(scenario_path: str) -> list[dict]:
    """Load bug spec from JSON scenario file."""
    path = Path(scenario_path)
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"Scenario file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in scenario file {path}: {exc}") from exc
    # Support both list format (task1) and dict format (task2/task3)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "bugs" in data:
        return data["bugs"]
    raise ValueError(f"Unrecognized scenario format in {path}")


def inject_bugs(df: pd.DataFrame, bug_spec: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """
    Inject bugs into df in strict order. Returns (corrupted_df, ground_truth_list).
    Injection order is MANDATORY:
      1. null_injection
      2. type_corruption
      3. out_of_range
      4. format_inconsistency
      5. schema_drift
      6. pii_leak
      7. duplicate_rows  ← ALWAYS LAST
    """
    INJECTION_ORDER = [
        "null_injection",
        "type_corruption",
        "out_of_range",
        "format_inconsistency",
        "schema_drift",
        "pii_leak",
        "duplicate_rows",
    ]

    corrupted = df.copy()
    ground_truth: list[dict] = []

    for bug_type in INJECTION_ORDER:
        bugs_of_type = [b for b in bug_spec if b["type"] == bug_type]
        for bug in bugs_of_type:
            affected_rows: list[int] = []

            if bug_type == "null_injection":
                col = bug["column"]
                rows = bug.get("rows", [])
                if rows == "ALL":
                    corrupted[col] = None
                    affected_rows = list(range(len(corrupted)))
                else:
                    corrupted.loc[rows, col] = None
                    affected_rows = rows

            elif bug_type == "type_corruption":
                col = bug["column"]
                row = bug["row"]
                if col not in corrupted.columns and col == "rev_amt" and "revenue_amount" in corrupted.columns:
                    col = "revenue_amount"
                # CRITICAL: cast to object BEFORE assignment to suppress FutureWarning
                corrupted[col] = corrupted[col].astype(object)
                corrupted.loc[row, col] = bug["value"]
                affected_rows = [row]

            elif bug_type == "out_of_range":
                col = bug["column"]
                row = bug["row"]
                corrupted.loc[row, col] = bug["value"]
                affected_rows = [row]

            elif bug_type == "format_inconsistency":
                col = bug.get("column", "phone")
                row = bug["row"]
                existing = str(corrupted.loc[row, col])
                # valid: "98XXXXXXXX" → corrupted: "+91-98-XXXXXXXX"
                if existing.startswith("98") and len(existing) == 10:
                    corrupted.loc[row, col] = f"+91-{existing[:2]}-{existing[2:]}"
                else:
                    corrupted.loc[row, col] = f"+91-{existing}"
                affected_rows = [row]

            elif bug_type == "schema_drift":
                old_col = bug["old_col"]
                new_col = bug["new_col"]
                if old_col in corrupted.columns:
                    corrupted.rename(columns={old_col: new_col}, inplace=True)
                affected_rows = []

            elif bug_type == "pii_leak":
                # SSN already present — flag in ground truth only
                affected_rows = list(range(len(corrupted)))

            elif bug_type == "duplicate_rows":
                # ALWAYS LAST — pd.concat, never df.append (deprecated)
                indices = bug.get("indices", [])
                if indices:
                    extra = corrupted.iloc[indices].copy()
                    corrupted = pd.concat([corrupted, extra], ignore_index=True)
                affected_rows = indices

            ground_truth.append(
                {
                    "bug_id": bug["bug_id"],
                    "type": bug_type,
                    "column": bug.get("column"),
                    "description": bug.get("description", ""),
                    "severity": bug.get("severity", "medium"),
                    "affected_rows": affected_rows,
                }
            )

    return corrupted, ground_truth


if __name__ == "__main__":
    from env.data.generator import generate_employee_dataset

    scenarios = [
        "env/data/scenarios/task1_scenario.json",
        "env/data/scenarios/task2_scenario.json",
        "env/data/scenarios/task3_scenario.json",
    ]
    df_clean = generate_employee_dataset(seed=42)
    for path in scenarios:
        spec = load_scenario(path)
        corrupted, gt = inject_bugs(df_clean.copy(), spec)
        print(f"{path}: {len(gt)} bugs, corrupted shape={corrupted.shape}")