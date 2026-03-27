"""Task 2 (Medium): Fix schema / type problems in the dataset."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from env.data.generator import generate_dataset
from env.data.bug_injector import inject_nulls, inject_wrong_types
from env.models import ActionType

TASK_ID = "task2_medium"
TASK_NAME = "Fix Schema Problems"
DIFFICULTY = "medium"

_TYPE_COLUMNS = ["age", "score"]


def build_initial_state(seed: int = 42) -> Dict[str, Any]:
    """Return the initial task state with injected type bugs (and some nulls)."""
    clean = generate_dataset(seed=seed)
    buggy = inject_nulls(clean, columns=["score"], frac=0.08, seed=seed)
    buggy = inject_wrong_types(buggy, columns=_TYPE_COLUMNS, seed=seed)
    type_errors: Dict[str, str] = {}
    for col in _TYPE_COLUMNS:
        mask = pd.to_numeric(buggy[col], errors="coerce").isna() & buggy[col].notna()
        if mask.any():
            type_errors[col] = str(buggy[col].dtype)
    return {
        "task_id": TASK_ID,
        "df": buggy,
        "type_errors": type_errors,
        "fixed_columns": [],
        "target_columns": list(type_errors.keys()),
    }


def evaluate_action(
    state: Dict[str, Any],
    action_type: ActionType,
    target_column: Optional[str],
    value: Any,
) -> Tuple[float, Dict[str, Any], bool]:
    """Apply *action* to *state*.

    Returns
    -------
    reward : float
    info   : dict
    done   : bool
    """
    target_columns: List[str] = state["target_columns"]
    fixed: List[str] = state["fixed_columns"]

    if action_type == ActionType.NOOP:
        return -0.05, {"msg": "NOOP — no change"}, False

    if action_type == ActionType.FIX_TYPE:
        if target_column and target_column in target_columns and target_column not in fixed:
            df: pd.DataFrame = state["df"]
            df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
            median_val = df[target_column].median()
            df[target_column] = df[target_column].fillna(median_val)
            if target_column == "age":
                df[target_column] = df[target_column].astype(int)
            fixed.append(target_column)
            state["fixed_columns"] = fixed
            done = set(fixed) >= set(target_columns)
            return 0.2, {"msg": f"Fixed type in '{target_column}'"}, done
        return -0.1, {"msg": f"Invalid FIX_TYPE target: '{target_column}'"}, False

    if action_type == ActionType.FIX_SCHEMA:
        if target_column and target_column in target_columns and target_column not in fixed:
            df = state["df"]
            df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
            df[target_column] = df[target_column].fillna(df[target_column].median())
            fixed.append(target_column)
            state["fixed_columns"] = fixed
            done = set(fixed) >= set(target_columns)
            return 0.2, {"msg": f"Fixed schema for '{target_column}'"}, done
        return -0.1, {"msg": f"Invalid FIX_SCHEMA target: '{target_column}'"}, False

    return -0.1, {"msg": f"Action '{action_type}' is not applicable to this task"}, False
