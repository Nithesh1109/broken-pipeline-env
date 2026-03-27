"""Task 3 (Hard): Full pipeline debugging — nulls, type errors, and duplicates."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from env.data.generator import generate_dataset
from env.data.bug_injector import inject_all_bugs
from env.models import ActionType

TASK_ID = "task3_hard"
TASK_NAME = "Full Pipeline Debugging"
DIFFICULTY = "hard"


def _audit_state(state: Dict[str, Any]) -> None:
    """Recompute remaining issues from the current DataFrame."""
    df: pd.DataFrame = state["df"]
    null_cols = [c for c in df.columns if df[c].isnull().any()]
    type_error_cols = [
        c for c in ["age", "score"]
        if c in df.columns
        and (pd.to_numeric(df[c], errors="coerce").isna() & df[c].notna()).any()
    ]
    has_duplicates = df.duplicated().any()
    state["null_columns"] = null_cols
    state["type_error_columns"] = type_error_cols
    state["has_duplicates"] = bool(has_duplicates)


def build_initial_state(seed: int = 42) -> Dict[str, Any]:
    """Return the initial task state with all bugs injected."""
    clean = generate_dataset(seed=seed)
    buggy = inject_all_bugs(clean, seed=seed)
    state: Dict[str, Any] = {
        "task_id": TASK_ID,
        "df": buggy,
        "null_columns": [],
        "type_error_columns": [],
        "has_duplicates": False,
        "actions_taken": [],
    }
    _audit_state(state)
    return state


def _is_done(state: Dict[str, Any]) -> bool:
    return (
        not state["null_columns"]
        and not state["type_error_columns"]
        and not state["has_duplicates"]
    )


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
    if action_type == ActionType.NOOP:
        return -0.05, {"msg": "NOOP — no change"}, False

    df: pd.DataFrame = state["df"]

    if action_type == ActionType.FIX_NULL:
        if target_column and target_column in state["null_columns"]:
            fill_value = (
                df[target_column].median()
                if pd.api.types.is_numeric_dtype(df[target_column])
                else "unknown"
            )
            df[target_column] = df[target_column].fillna(fill_value)
            _audit_state(state)
            state["actions_taken"].append(action_type)
            return 0.2, {"msg": f"Fixed nulls in '{target_column}'"}, _is_done(state)
        return -0.1, {"msg": f"FIX_NULL: no nulls in '{target_column}'"}, False

    if action_type in (ActionType.FIX_TYPE, ActionType.FIX_SCHEMA):
        if target_column and target_column in state["type_error_columns"]:
            df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
            df[target_column] = df[target_column].fillna(df[target_column].median())
            _audit_state(state)
            state["actions_taken"].append(action_type)
            return 0.2, {"msg": f"Fixed type/schema in '{target_column}'"}, _is_done(state)
        return -0.1, {"msg": f"FIX_TYPE: no type errors in '{target_column}'"}, False

    if action_type == ActionType.FIX_DUPLICATE:
        if state["has_duplicates"]:
            state["df"] = df.drop_duplicates().reset_index(drop=True)
            _audit_state(state)
            state["actions_taken"].append(action_type)
            return 0.2, {"msg": "Removed duplicate rows"}, _is_done(state)
        return -0.1, {"msg": "No duplicates to fix"}, False

    if action_type == ActionType.DEBUG_PIPELINE:
        issues = (
            state["null_columns"]
            + state["type_error_columns"]
            + (["duplicates"] if state["has_duplicates"] else [])
        )
        return 0.2, {"msg": "Pipeline debug scan complete", "issues": issues}, False

    return -0.1, {"msg": f"Unknown action '{action_type}'"}, False
