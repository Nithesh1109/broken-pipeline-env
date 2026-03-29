from env.models import GraderResult
from env.tasks.task3_incident import Task3IncidentEnv


WEIGHTS = {"diagnosis": 0.25, "fix": 0.35, "pii_sweep": 0.20, "validation": 0.20}


def grade_task3(env: Task3IncidentEnv) -> GraderResult:
    sub_scores = {
        "diagnosis": 1.0 if env.diagnosis_correct else 0.0,
        "fix": 1.0 if env.fix_applied else 0.0,
        "pii_sweep": 1.0 if env.pii_masked else 0.0,
        "validation": 1.0 if env.validation_passed else 0.0,
    }
    raw_total = sum(WEIGHTS[k] * v for k, v in sub_scores.items())
    pii_penalty = -0.20 if not env.pii_masked else 0.0
    final_score = round(max(0.0, min(1.0, raw_total + pii_penalty)), 4)

    return GraderResult(
        score=final_score,
        breakdown={
            **{k: round(v, 4) for k, v in sub_scores.items()},
            "pii_compliance_penalty": round(pii_penalty, 4),
            "downstream_health": round(env.downstream_health, 4),
        },
        explanation=(
            f"Diagnosis:{sub_scores['diagnosis']} Fix:{sub_scores['fix']} "
            f"PII:{sub_scores['pii_sweep']} Validation:{sub_scores['validation']} "
            f"PII_penalty:{pii_penalty} -> Final:{final_score}"
        ),
    )