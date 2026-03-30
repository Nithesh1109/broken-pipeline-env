from __future__ import annotations

from env.models import GraderResult
from env.tasks.task1_audit import Task1AuditEnv


def grade_task1(env: Task1AuditEnv) -> GraderResult:
    """
    Task 1: Data Quality Audit scorer.
    Formula: score = 0.4 * id_score + 0.6 * fix_score
    Both numerators are set sizes — no double counting.
    """
    total = env.TOTAL_BUGS
    if total == 0:
        return GraderResult(score=0.0, breakdown={}, explanation="No bugs in scenario.")

    num_id = len(env.identified_bug_ids)
    num_fix = len(env.fixed_bug_ids)

    id_score = num_id / total
    fix_score = num_fix / total
    raw = 0.4 * id_score + 0.6 * fix_score
    score = round(max(0.0, min(1.0, raw)), 4)

    return GraderResult(
        score=score,
        breakdown={
            "identification": round(id_score, 4),
            "remediation": round(fix_score, 4),
            "bugs_identified": float(num_id),
            "bugs_fixed": float(num_fix),
            "total_bugs": float(total),
        },
        explanation=(
            f"Identified {num_id}/{total} bugs (weight 0.4), "
            f"fixed {num_fix}/{total} bugs (weight 0.6). "
            f"Score = 0.4x{id_score:.3f} + 0.6x{fix_score:.3f} = {score}"
        ),
    )