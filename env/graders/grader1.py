from env.models import GraderResult
from env.tasks.task1_audit import Task1AuditEnv


def grade_task1(env: Task1AuditEnv) -> GraderResult:
    total = env.TOTAL_BUGS
    id_score = len(env.identified_bug_ids) / total
    fix_score = len(env.fixed_bug_ids) / total
    score = round(0.4 * id_score + 0.6 * fix_score, 4)
    score = round(max(0.0, min(1.0, score)), 4)
    return GraderResult(
        score=score,
        breakdown={
            "identification": round(id_score, 4),
            "remediation": round(fix_score, 4),
        },
        explanation=(
            f"Identified {len(env.identified_bug_ids)}/{total} bugs, "
            f"fixed {len(env.fixed_bug_ids)}/{total} bugs. "
            f"Score = 0.4*{id_score:.2f} + 0.6*{fix_score:.2f} = {score}"
        ),
    )