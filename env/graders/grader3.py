"""Grader 3: score the Hard task (full pipeline debugging)."""

from typing import Any, Dict

import pandas as pd

from env.models import GraderResult

_ISSUE_WEIGHT = 1.0 / 3.0  # three issue categories


def grade(state: Dict[str, Any], history: list) -> GraderResult:
    """Deterministically score a completed task3_hard episode.

    Scoring
    -------
    Three equally-weighted sub-scores:
      1. Null resolution   : 1.0 if no remaining nulls, else fraction resolved.
      2. Type resolution   : 1.0 if no remaining type errors, else fraction resolved.
      3. Duplicate removal : 1.0 if no duplicates remain, else 0.0.
    """
    df: pd.DataFrame = state.get("df", pd.DataFrame())
    if df.empty:
        return GraderResult(score=0.0, details={"msg": "No data"})

    # --- null sub-score ---
    total_cells = df.size
    null_cells = int(df.isnull().sum().sum())
    null_score = round(max(0.0, 1.0 - null_cells / total_cells), 4) if total_cells else 1.0

    # --- type sub-score ---
    type_cols = [c for c in ["age", "score"] if c in df.columns]
    if type_cols:
        total_invalid = sum(
            int((pd.to_numeric(df[c], errors="coerce").isna() & df[c].notna()).sum())
            for c in type_cols
        )
        type_score = round(max(0.0, 1.0 - total_invalid / len(df)), 4)
    else:
        type_score = 1.0

    # --- duplicate sub-score ---
    dup_score = 0.0 if df.duplicated().any() else 1.0

    score = round((null_score + type_score + dup_score) * _ISSUE_WEIGHT, 4)
    return GraderResult(
        score=score,
        details={
            "null_score": null_score,
            "type_score": type_score,
            "duplicate_score": dup_score,
            "steps_used": len(history),
        },
    )
