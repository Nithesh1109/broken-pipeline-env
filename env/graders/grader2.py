"""Grader 2: score the Medium task (schema / type fixing)."""

from typing import Any, Dict

import pandas as pd

from env.models import GraderResult


def grade(state: Dict[str, Any], history: list) -> GraderResult:
    """Deterministically score a completed task2_medium episode.

    Scoring
    -------
    - Each target column that is now fully numeric contributes equally.
    - Penalty proportional to remaining invalid values.
    """
    target: list = state.get("target_columns", [])
    df: pd.DataFrame = state.get("df", pd.DataFrame())

    if not target or df.empty:
        return GraderResult(score=1.0, details={"msg": "No issues to fix"})

    col_scores = {}
    for col in target:
        if col not in df.columns:
            col_scores[col] = 0.0
            continue
        invalid = (pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()).sum()
        total = len(df)
        col_scores[col] = round(max(0.0, 1.0 - invalid / total), 4)

    score = round(sum(col_scores.values()) / len(target), 4)
    return GraderResult(
        score=score,
        details={
            "target_columns": target,
            "column_scores": col_scores,
            "steps_used": len(history),
        },
    )
