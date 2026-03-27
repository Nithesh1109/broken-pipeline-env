"""Agent inference loop using the OpenAI client against an OpenEnv server.

Environment variables
---------------------
API_BASE_URL  : Base URL of the OpenAI-compatible endpoint (required)
MODEL_NAME    : Model identifier to use (required)
OPENAI_API_KEY / HF_TOKEN : API key / HuggingFace token (required)

Usage
-----
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export OPENAI_API_KEY=sk-...
    python inference.py
"""

import json
import os
import sys

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
# Support both OPENAI_API_KEY (standard) and HF_TOKEN (HuggingFace TGI)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "MISSING")

ENV_SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:8000")
MAX_STEPS = 8

TASK_ID = os.environ.get("TASK_ID", "task3_hard")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOOP_ACTION = {"action_type": "NOOP", "target_column": None, "value": None}

SYSTEM_PROMPT = """You are a data pipeline debugging agent.
You will receive the current state of a dataset and must choose an action to fix issues.

Available actions:
- NOOP: do nothing
- FIX_NULL: fix missing values (requires target_column)
- FIX_TYPE: fix type errors (requires target_column)
- FIX_DUPLICATE: remove duplicate rows
- FIX_SCHEMA: fix schema problems (requires target_column)
- DEBUG_PIPELINE: scan for all pipeline issues

Respond ONLY with a JSON object like:
{"action_type": "FIX_NULL", "target_column": "age", "value": null}
"""


def call_env(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{ENV_SERVER_URL}{path}"
    with httpx.Client(timeout=30) as client:
        if method == "GET":
            resp = client.get(url)
        else:
            resp = client.post(url, json=body or {})
    resp.raise_for_status()
    return resp.json()


def build_user_message(obs: dict) -> str:
    issues = obs.get("issues_found", [])
    snap = obs.get("data_snapshot", {})
    return (
        f"Step {obs['step_count']}/{MAX_STEPS}.\n"
        f"Issues detected: {issues}\n"
        f"Null counts: {snap.get('null_counts', {})}\n"
        f"Duplicate rows: {snap.get('duplicate_rows', 0)}\n"
        f"Shape: {snap.get('shape', [])}\n"
        f"Choose your next action as JSON."
    )


def parse_action(text: str) -> dict:
    """Extract JSON action from model response; fall back to NOOP."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        print(f"  [warn] Could not parse action from: {text!r} — using NOOP")
        return NOOP_ACTION


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------


def run_agent(task_id: str = TASK_ID) -> None:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

    print(f"\n=== OpenEnv Agent  task={task_id}  model={MODEL_NAME} ===\n")

    # Reset environment
    obs = call_env("POST", "/reset", {"task_id": task_id, "seed": 42})
    print(f"Initial issues: {obs.get('issues_found', [])}\n")

    total_reward = 0.0
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step_num in range(1, MAX_STEPS + 1):
        if obs.get("done"):
            print("Environment signalled done.")
            break

        user_msg = build_user_message(obs)
        messages.append({"role": "user", "content": user_msg})

        # Query the model
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=128,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [error] LLM call failed: {exc}  — using NOOP")
            raw = json.dumps(NOOP_ACTION)

        messages.append({"role": "assistant", "content": raw})
        action = parse_action(raw)
        print(f"Step {step_num}: action={action}")

        # Send action to environment
        result = call_env("POST", "/step", action)
        reward = result.get("reward", 0.0)
        total_reward += reward
        obs = result.get("observation", obs)
        info = result.get("info", {})
        print(f"         reward={reward:+.2f}  info={info.get('msg', '')}  issues={obs.get('issues_found', [])}")

        if result.get("done"):
            print("\nEpisode complete.")
            break

    # Grade the episode
    grader_result = call_env("POST", "/grader", {})
    final_score = grader_result.get("score", 0.0)
    details = grader_result.get("details", {})

    print(f"\n=== Results ===")
    print(f"Total reward : {total_reward:+.4f}")
    print(f"Final score  : {final_score:.4f}")
    print(f"Details      : {details}")


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else TASK_ID
    run_agent(task)
