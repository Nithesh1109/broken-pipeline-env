"""
inference.py - DataPipelineEnv LLM Agent Loop (Member 2)

Architecture:
  Observe -> Hypothesize -> Tool Call -> Update Belief -> Fix

Memory: rolling window of last ROLLING_WINDOW message pairs.
Self-correction: up to MAX_PARSE_RETRIES before NOOP fallback.
Belief state: in-memory dict tracking candidate root causes.
Context compaction: at step 6, inject compressed incident summary.
PII sanitizer: redact SSN patterns from reasoning traces before logging.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Optional

import requests as http
from openai import OpenAI


# -- Environment variables -- raise immediately if missing ------------------
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

if not API_BASE_URL:
    raise EnvironmentError("API_BASE_URL is not set.")
if not MODEL_NAME:
    raise EnvironmentError("MODEL_NAME is not set.")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN is not set.")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")

client = OpenAI(api_key=HF_TOKEN, base_url=LLM_BASE_URL)


# -- Constants -------------------------------------------------------------
MAX_STEPS = 8
MAX_PARSE_RETRIES = 2
ROLLING_WINDOW = 6
COMPACTION_STEP = 6
MAX_RUNTIME_SECS = 19 * 60
HTTP_TIMEOUT = 30

VALID_ACTION_TYPES = {
    "INSPECT",
    "RENAME_COLUMN",
    "CAST_TYPE",
    "FILL_DEFAULT",
    "DROP_COLUMN",
    "VALIDATE",
    "MASK_PII",
    "NOOP",
}

FALLBACK_ACTION = {
    "action_type": "NOOP",
    "target_column": None,
    "transformation": None,
    "justification": "Fallback NOOP - could not parse valid action after retries.",
    "identified_issues": None,
}

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")

_EPISODE_START: float = 0.0


# -- System prompt ---------------------------------------------------------
SYSTEM_PROMPT = """You are a senior data engineer on call.
A production data pipeline is broken. Investigate systematically.

STRATEGY:
  Step 1-2: INSPECT to gather evidence (use target_column to unlock facets:
            \"logs\", \"metrics\", \"dag\", \"pii\", or a specific column name)
  Step 3-5: Apply targeted fixes based on what you found
  Step 6+:  VALIDATE to confirm fixes, then confirm pipeline health

OUTPUT FORMAT - reply ONLY with valid JSON, no markdown fences, no explanation:
{
  \"action_type\": \"INSPECT\"|\"RENAME_COLUMN\"|\"CAST_TYPE\"|\"FILL_DEFAULT\"|
                 \"DROP_COLUMN\"|\"VALIDATE\"|\"MASK_PII\"|\"NOOP\",
  \"target_column\": \"column_name\" or null,
  \"transformation\": \"cast_to_int\"|\"cast_to_float\"|\"fill_median\"|
                    \"fill_zero\"|\"drop_duplicates\" or null,
  \"justification\": \"One sentence: what you observed and why this action.\",
  \"identified_issues\": [
    {
      \"issue_type\": \"null_injection\"|\"type_corruption\"|\"out_of_range\"|
                    \"format_inconsistency\"|\"schema_drift\"|\"pii_leak\"|
                    \"duplicate_rows\",
      \"column\": \"column_name\" or null,
      \"description\": \"what you found\",
      \"severity\": \"low\"|\"medium\"|\"high\"|\"critical\"
    }
  ] or null
}

RULES:
- Always include justification explaining your reasoning
- On INSPECT, populate identified_issues with everything you observe
- If you see SSN data anywhere, immediately use MASK_PII on \"ssn\"
- Never DROP_COLUMN without checking dependencies first
- After all fixes, use VALIDATE to confirm"""


# -- Utility functions -----------------------------------------------------

def _sanitize_pii(text: str) -> str:
    """Redact SSN and email patterns from reasoning traces before logging."""
    text = _SSN_RE.sub("[SSN-REDACTED]", text)
    text = _EMAIL_RE.sub("[EMAIL-REDACTED]", text)
    return text


def _check_runtime():
    """Raise SystemExit if total elapsed time exceeds limit."""
    if time.time() - _EPISODE_START > MAX_RUNTIME_SECS:
        print(f"\n[TIMEOUT] Exceeded {MAX_RUNTIME_SECS//60}min limit. Stopping.")
        sys.exit(1)


def _observation_to_prompt(obs: dict, belief_state: dict, step_num: int) -> str:
    """
    Convert DataObservation dict to LLM prompt string.
    Includes belief state (candidate root causes) for hypothesis tracking.
    """
    lines = [
        f"=== STEP {step_num + 1}/{MAX_STEPS} ===",
        f"Pipeline stage  : {obs.get('pipeline_stage', 'UNKNOWN')}",
        f"Steps remaining : {obs.get('time_remaining', 0)}",
        f"Downstream health: {obs.get('downstream_health', 0):.2f}",
    ]

    visible = obs.get("visible_signals") or {}
    if visible.get("alert"):
        a = visible["alert"]
        lines.append(f"\n[ALERT] severity={a.get('severity')} risk={a.get('risk_score',0):.2f}")
        lines.append(f"  {a.get('message','')}")
    if visible.get("logs"):
        lg = visible["logs"]
        lines.append(f"\n[LOGS] status={lg.get('last_run_status')}")
        for err in (lg.get("recent_errors") or [])[:3]:
            lines.append(f"  ERROR: {err}")
    if visible.get("metrics"):
        m = visible["metrics"]
        lines.append(
            f"\n[METRICS] rows={m.get('row_count')} avg={m.get('historical_avg')} "
            f"null_ratio={m.get('null_ratio',0):.3f}"
        )
    if visible.get("compliance"):
        c = visible["compliance"]
        lines.append(
            f"\n[COMPLIANCE] pii_detected={c.get('pii_detected')} "
            f"risky_cols={c.get('risky_columns')}"
        )

    lines.append(f"\nSchema:\n{json.dumps(obs.get('schema', {}), indent=2)}")
    lines.append(f"\nDataset preview (first 5 rows):\n{json.dumps(obs.get('dataset_preview', [])[:5], indent=2)}")

    if obs.get("validation_report"):
        lines.append(f"\nOpen issues:\n{json.dumps(obs['validation_report'], indent=2)}")

    if belief_state.get("candidates"):
        lines.append(f"\n[BELIEF STATE] Candidate root causes: {belief_state['candidates']}")
    if belief_state.get("confirmed"):
        lines.append(f"[BELIEF STATE] Confirmed: {belief_state['confirmed']}")

    lines.append("\nWhat is your next action?")
    return "\n".join(lines)


def _compaction_summary(belief_state: dict, step_errors: list[str]) -> str:
    """
    At step COMPACTION_STEP, inject a compressed incident summary.
    Replaces verbose older observations with a concise fact sheet.
    Saves ~40% token cost on remaining steps.
    """
    candidates = belief_state.get("candidates", ["unknown"])
    confirmed = belief_state.get("confirmed", [])
    fixes_done = belief_state.get("fixes_done", [])

    return (
        f"[INCIDENT FACT SHEET - Step {COMPACTION_STEP} Summary]\n"
        f"Candidate root causes identified: {candidates}\n"
        f"Confirmed failures: {confirmed}\n"
        f"Fixes applied so far: {fixes_done}\n"
        f"Recent errors: {step_errors[-3:] if step_errors else ['none']}\n"
        f"--- Continue investigation or validate if all issues resolved ---"
    )


def _truncate_messages(messages: list[dict], system_msg: dict) -> list[dict]:
    """
    Keep only last ROLLING_WINDOW user+assistant pairs.
    System message always preserved at index 0.
    """
    non_system = [m for m in messages if m["role"] != "system"]
    if len(non_system) > ROLLING_WINDOW * 2:
        non_system = non_system[-(ROLLING_WINDOW * 2) :]
    return [system_msg] + non_system


def _parse_json_from_text(text: str) -> Optional[dict]:
    """
    Extract JSON from LLM output. Handles:
    - clean JSON
    - ```json ... ``` fences
    - ``` ... ``` fences
    - JSON embedded in prose
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _validate_action(action_dict: dict) -> bool:
    """Validate parsed action has required fields and valid action_type."""
    if not isinstance(action_dict, dict):
        return False
    if action_dict.get("action_type") not in VALID_ACTION_TYPES:
        return False
    if not action_dict.get("justification"):
        return False
    return True


def _update_belief_state(belief: dict, action: dict, result: dict) -> dict:
    """
    Update belief state from action and step result.
    Tracks candidate root causes and confirmed fixes.
    """
    action_type = action.get("action_type", "NOOP")
    target = action.get("target_column")
    justif = _sanitize_pii(action.get("justification", ""))
    reward = result.get("reward", 0.0)
    info = result.get("info", {})

    identified = info.get("identified", [])
    fixed = info.get("fixed", [])

    keywords = [
        "stage 3",
        "schema drift",
        "type mismatch",
        "ssn",
        "pii",
        "revenue",
        "aggregation",
        "null",
        "duplicate",
        "join",
    ]
    found = [kw for kw in keywords if kw in justif.lower()]
    if found:
        belief.setdefault("candidates", [])
        for item in found:
            if item not in belief["candidates"]:
                belief["candidates"].append(item)

    if reward > 0.1 and action_type not in ("INSPECT", "NOOP"):
        belief.setdefault("confirmed", [])
        if target and target not in belief["confirmed"]:
            belief["confirmed"].append(f"{action_type}:{target}")

    belief["fixes_done"] = list(fixed)
    belief["issues_identified"] = list(identified)

    if info.get("signals_unlocked"):
        belief["signals_unlocked"] = info["signals_unlocked"]

    return belief


# -- Main episode loop -----------------------------------------------------
def run_episode(task_id: int) -> float:
    """
    Run one full episode for a task.

    Loop: Observe -> Hypothesize -> Tool Call -> Update Belief -> Fix
    Memory: rolling window of ROLLING_WINDOW message pairs
    Self-correction: up to MAX_PARSE_RETRIES before NOOP
    Compaction: inject summary at COMPACTION_STEP
    """
    _check_runtime()

    resp = http.post(f"{API_BASE_URL}/reset", params={"task_id": task_id}, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    obs = resp.json()

    if "info" in obs:
        obs["visible_signals"] = obs["info"].get("visible_signals", {})

    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    messages = [system_msg]
    belief_state: dict = {"candidates": [], "confirmed": [], "fixes_done": []}
    step_errors: list[str] = []

    for step_num in range(MAX_STEPS):
        _check_runtime()

        if step_num == COMPACTION_STEP:
            summary = _compaction_summary(belief_state, step_errors)
            non_system = [m for m in messages if m["role"] != "system"]
            last_two = non_system[-2:] if len(non_system) >= 2 else non_system
            messages = [system_msg, {"role": "user", "content": summary}] + last_two
            print(f"  [COMPACTION] Injected incident summary at step {step_num+1}")

        user_content = _observation_to_prompt(obs, belief_state, step_num)
        messages.append({"role": "user", "content": user_content})

        messages = _truncate_messages(messages, system_msg)

        action = None
        last_error = ""

        for attempt in range(MAX_PARSE_RETRIES + 1):
            if attempt > 0:
                correction_msg = (
                    f"Your previous response was invalid: {last_error}\n"
                    f"Please reply with ONLY valid JSON matching the schema. "
                    f"No markdown, no explanation."
                )
                messages.append({"role": "user", "content": correction_msg})

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=512,
                )
                raw = response.choices[0].message.content or ""
            except Exception as exc:
                last_error = f"LLM call failed: {exc}"
                step_errors.append(last_error)
                print(f"  [LLM ERROR] {exc}")
                raw = ""

            parsed = _parse_json_from_text(raw) if raw else None

            if parsed and _validate_action(parsed):
                action = parsed
                messages.append({"role": "assistant", "content": raw})
                break
            else:
                last_error = (
                    f"Invalid action_type '{parsed.get('action_type') if parsed else 'none'}' "
                    f"or missing justification"
                )

        if action is None:
            action = FALLBACK_ACTION
            messages.append({"role": "assistant", "content": json.dumps(FALLBACK_ACTION)})
            print(f"  [FALLBACK] Using NOOP after {MAX_PARSE_RETRIES} retries")

        try:
            step_resp = http.post(
                f"{API_BASE_URL}/step",
                json=action,
                params={"task_id": task_id},
                timeout=HTTP_TIMEOUT,
            )
            step_resp.raise_for_status()
            result = step_resp.json()
        except Exception as exc:
            print(f"  [STEP ERROR] {exc}")
            step_errors.append(str(exc))
            break

        obs = result.get("observation", obs)
        done = result.get("done", False)
        reward = float(result.get("reward", 0.0))

        step_info = result.get("info", {})
        obs["visible_signals"] = step_info.get("visible_signals", {})

        belief_state = _update_belief_state(belief_state, action, result)

        justif_short = _sanitize_pii(action.get("justification", ""))[:60]
        print(
            f"  Step {step_num+1:02d} | "
            f"{action['action_type']:15s} | "
            f"target={str(action.get('target_column',''))[:12]:12s} | "
            f"reward={reward:+.3f} | "
            f"health={obs.get('downstream_health',0):.2f} | "
            f"done={done} | "
            f"'{justif_short}...'"
        )

        if done:
            print(f"  [DONE] Episode completed at step {step_num+1}")
            break

    try:
        grade_resp = http.get(f"{API_BASE_URL}/grader", params={"task_id": task_id}, timeout=HTTP_TIMEOUT)
        grade_resp.raise_for_status()
        grade = grade_resp.json()
    except Exception as exc:
        print(f"  [GRADER ERROR] {exc}")
        return 0.0

    score = float(grade.get("score", 0.0))
    breakdown = grade.get("breakdown", {})
    explanation = grade.get("explanation", "")

    print(f"\n  GRADER BREAKDOWN: {json.dumps(breakdown, indent=4)}")
    print(f"  EXPLANATION: {explanation}")
    print(
        f"  BELIEF STATE: candidates={belief_state.get('candidates')} "
        f"confirmed={belief_state.get('confirmed')}"
    )

    return score


# -- Entry point -----------------------------------------------------------
if __name__ == "__main__":
    _EPISODE_START = time.time()

    scores: dict[int, float] = {}

    for task_id in [1, 2, 3]:
        print(f"\n{'='*65}")
        print(f"  TASK {task_id}")
        print(f"{'='*65}")
        try:
            scores[task_id] = run_episode(task_id)
        except Exception as exc:
            print(f"  [TASK {task_id} FAILED] {exc}")
            scores[task_id] = 0.0

        elapsed = time.time() - _EPISODE_START
        print(f"  -> Score: {scores[task_id]:.4f}  (total elapsed: {elapsed:.0f}s / {MAX_RUNTIME_SECS}s)")

    avg = sum(scores.values()) / 3
    print(f"\n{'='*65}")
    print("  FINAL SCORES")
    print(f"  Task 1: {scores.get(1,0):.4f}")
    print(f"  Task 2: {scores.get(2,0):.4f}")
    print(f"  Task 3: {scores.get(3,0):.4f}")
    print(f"  Average: {avg:.4f}")
    print(f"{'='*65}")
