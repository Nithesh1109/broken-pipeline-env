from __future__ import annotations

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
    mask = df[list(CRITICAL_COLUMNS & present)].notna().all(axis=1)
    return int(mask.sum())


def grade_task2(env: Task2SchemaEnv) -> GraderResult:
    """
    Task 2: Schema Drift Remediation scorer.
    Formula: base = rows_passing / total_rows
             blast_penalty = -0.10 * blast_events
             score = clamp(base + blast_penalty, 0.0, 1.0)
    """
    total = len(env.df) if env.df is not None else 0
    if total == 0:
        return GraderResult(score=0.0, breakdown={}, explanation="Empty DataFrame.")

    passing = _rows_passing(env)
    base = passing / total
    blast_penalty = -0.10 * env.blast_events
    score = round(max(0.0, min(1.0, base + blast_penalty)), 4)

    present_ratio = len(set(env.df.columns) & EXPECTED_COLUMNS) / len(EXPECTED_COLUMNS)

    return GraderResult(
        score=score,
        breakdown={
            "schema_compliance": round(base, 4),
            "blast_radius_penalty": round(blast_penalty, 4),
            "bugs_fixed": round(len(env.fixed_bug_ids) / env.TOTAL_BUGS, 4),
            "column_coverage": round(present_ratio, 4),
            "rows_passing": float(passing),
            "total_rows": float(total),
        },
        explanation=(
            f"{passing}/{total} rows pass schema validation. "
            f"Blast events: {env.blast_events} (penalty={blast_penalty:.2f}). "
            f"Column coverage: {present_ratio:.2f}. Score={score}"
        ),
    )