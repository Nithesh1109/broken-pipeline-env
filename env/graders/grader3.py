from __future__ import annotations

from env.models import GraderResult
from env.tasks.task3_incident import Task3IncidentEnv

WEIGHTS = {
    "diagnosis": 0.25,
    "fix": 0.35,
    "pii_sweep": 0.20,
    "validation": 0.20,
}

DIAGNOSIS_KEYWORDS = [
    "stage 3",
    "join stage",
    "schema drift",
    "ssn",
    "pii",
    "type mismatch",
    "revenue",
    "aggregation",
    "rev_amt",
    "corruption",
    "join failure",
    "type error",
]


def _reasoning_bonus(env: Task3IncidentEnv) -> float:
    """
    Award up to +0.15 for correct diagnostic language in justifications.
    Scans entire AER history — rewards agents that diagnosed correctly
    even if they applied the wrong fix action.
    Max: 3 keywords × 0.05 = 0.15
    """
    if not getattr(env, "aer_history", None):
        return 0.0
    combined = " ".join(r.justification.lower() for r in env.aer_history)
    hits = sum(1 for kw in DIAGNOSIS_KEYWORDS if kw in combined)
    return round(min(0.05 * hits, 0.15), 4)


def grade_task3(env: Task3IncidentEnv) -> GraderResult:
    """
    Task 3: Full Incident Response scorer.

    Weighted sub-scores:
      diagnosis  × 0.25
      fix        × 0.35
      pii_sweep  × 0.20
      validation × 0.20

    PII penalty: -0.20 if pii_masked is False (compliance violation).
    Reasoning bonus: up to +0.15 for correct diagnostic keywords in AER.

    Final: clamp(weighted_sum + pii_penalty + reasoning_bonus, 0.0, 1.0)
    """
    sub = {
        "diagnosis": 1.0 if env.diagnosis_correct else 0.0,
        "fix": 1.0 if env.fix_applied else 0.0,
        "pii_sweep": 1.0 if env.pii_masked else 0.0,
        "validation": 1.0 if env.validation_passed else 0.0,
    }
    weighted = sum(WEIGHTS[k] * v for k, v in sub.items())
    pii_penalty = -0.20 if not env.pii_masked else 0.0
    reasoning_bon = _reasoning_bonus(env)

    score = round(max(0.0, min(1.0, weighted + pii_penalty + reasoning_bon)), 4)

    return GraderResult(
        score=score,
        breakdown={
            **{k: round(v, 4) for k, v in sub.items()},
            "pii_compliance_penalty": round(pii_penalty, 4),
            "reasoning_bonus": round(reasoning_bon, 4),
            "signals_unlocked": float(len(getattr(env, "signals_unlocked", set()))),
            "downstream_health": round(env.downstream_health, 4),
        },
        explanation=(
            f"D:{sub['diagnosis']} F:{sub['fix']} "
            f"P:{sub['pii_sweep']} V:{sub['validation']} | "
            f"PII_penalty:{pii_penalty} Reasoning_bonus:{reasoning_bon} "
            f"-> {score}"
        ),
    )