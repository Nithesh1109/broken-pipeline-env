from env.models import GraderResult
from env.tasks.task2_schema import Task2SchemaEnv


def grade_task2(env: Task2SchemaEnv) -> GraderResult:
    if len(env.df) == 0:
        return GraderResult(score=0.0, breakdown={}, explanation="Empty dataframe.")

    expected_cols = {
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
    present_cols = set(env.df.columns)
    schema_ok = len(expected_cols & present_cols) / len(expected_cols)

    rows_passing = int(schema_ok * len(env.df))
    base_score = rows_passing / len(env.df)

    blast_penalty = -0.10 * env.blast_events
    final_score = round(max(0.0, min(1.0, base_score + blast_penalty)), 4)

    return GraderResult(
        score=final_score,
        breakdown={
            "schema_compliance": round(base_score, 4),
            "blast_radius_penalty": round(blast_penalty, 4),
            "bugs_fixed": round(len(env.fixed_bug_ids) / env.TOTAL_BUGS, 4),
        },
        explanation=(
            f"{rows_passing}/{len(env.df)} rows pass schema. "
            f"Blast events: {env.blast_events}. "
            f"Final: {final_score}"
        ),
    )