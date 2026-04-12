from __future__ import annotations

import pandas as pd

from env.models import GraderResult
from env.tasks.task2_schema import Task2SchemaEnv

EXPECTED_COLUMNS = {
    "employee_id",
    "name",
    "age",
    "salary",
    "department",
    "phone",
    "ssn",
    "hire_date",
    "consent_flag",
}
CRITICAL_COLUMNS = {"employee_id", "salary", "consent_flag"}


def _rows_passing(env: Task2SchemaEnv) -> int:
    df = env.df
    if df is None or len(df) == 0:
        return 0
    present = set(df.columns)
    missing_crit = CRITICAL_COLUMNS - present
    if missing_crit:
        return 0
    present_crit = list(CRITICAL_COLUMNS & present)
    return int(df[present_crit].notna().all(axis=1).sum())


def _column_recovery(env: Task2SchemaEnv) -> float:
    """Fraction of expected columns present after agent fixes."""
    if env.df is None:
        return 0.0
    present = set(env.df.columns)
    return round(len(EXPECTED_COLUMNS & present) / len(EXPECTED_COLUMNS), 4)


def _type_correctness(env: Task2SchemaEnv) -> float:
    """Check critical numeric columns have correct dtype."""
    if env.df is None:
        return 0.0

    checks = [
        ("age", pd.api.types.is_numeric_dtype),
        ("salary", pd.api.types.is_numeric_dtype),
    ]
    score = 0.0
    total = 0
    for col, check_fn in checks:
        if col in env.df.columns:
            total += 1
            if check_fn(env.df[col]):
                score += 1.0
    return round(score / total, 4) if total > 0 else 0.0


def grade_task2(env: Task2SchemaEnv) -> GraderResult:
    """
    Task 2: Schema Drift Remediation scorer.
    Formula:
      fix_score     = len(fixed_bug_ids) / TOTAL_BUGS
      blast_penalty = -0.10 * blast_events
      score         = clamp(fix_score + blast_penalty, 0.0001, 0.9999)
    """
    total = len(env.df) if env.df is not None else 0
    if total == 0:
        return GraderResult(score=0.0001, breakdown={}, explanation="Empty DataFrame.")

    total_bugs = getattr(env, "TOTAL_BUGS", 5)
    
    fix_score = len(env.fixed_bug_ids) / total_bugs
    blast_penalty = round(-0.10 * env.blast_events, 4)
    score = round(max(0.0001, min(0.9999, fix_score + blast_penalty)), 4)

    return GraderResult(
        score=score,
        breakdown={
            "bugs_fixed": round(len(env.fixed_bug_ids) / total_bugs, 4),
            "fix_score": fix_score,
            "blast_radius_penalty": blast_penalty,
            "blast_events": float(env.blast_events),
            "rows_passing": float(_rows_passing(env)),
            "column_recovery": _column_recovery(env),
            "type_correctness": _type_correctness(env),
        },
        explanation=(
            f"fix_score={fix_score}, "
            f"blast_penalty={blast_penalty} -> {score}"
        ),
    )