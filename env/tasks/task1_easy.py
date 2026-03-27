"""Task 1 (Easy): Detect and fix null / missing values in the dataset."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from env.data.generator import generate_dataset
from env.data.bug_injector import inject_nulls
from env.models import ActionType

TASK_ID = "task1_easy"
TASK_NAME = "Detect Null Values"
DIFFICULTY = "easy"

# Columns that may contain nulls in this task
_NULL_COLUMNS = ["age", "score", "category"]


def build_initial_state(seed: int = 42) -> Dict[str, Any]:
    """Return the initial task state with injected null bugs."""
    clean = generate_dataset(seed=seed)
    buggy = inject_nulls(clean, columns=_NULL_COLUMNS, frac=0.12, seed=seed)
    null_counts: Dict[str, int] = {
        col: int(buggy[col].isnull().sum())
        for col in _NULL_COLUMNS
        if buggy[col].isnull().any()
    }
    return {
        "task_id": TASK_ID,
        "df": buggy,
        "null_counts": null_counts,
        "fixed_columns": [],
        "target_columns": list(null_counts.keys()),
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

    if action_type == ActionType.FIX_NULL:
        if target_column and target_column in target_columns and target_column not in fixed:
            df: pd.DataFrame = state["df"]
            fill_value = value if value is not None else df[target_column].median() if pd.api.types.is_numeric_dtype(df[target_column]) else "unknown"
            df[target_column] = df[target_column].fillna(fill_value)
            fixed.append(target_column)
            state["fixed_columns"] = fixed
            done = set(fixed) >= set(target_columns)
            return 0.2, {"msg": f"Fixed nulls in '{target_column}'"}, done
        return -0.1, {"msg": f"Invalid FIX_NULL target: '{target_column}'"}, False

    return -0.1, {"msg": f"Action '{action_type}' is not applicable to this task"}, False
