"""inference.py – Agent loop using the OpenAI client against the running server.

Environment variables
---------------------
API_BASE_URL   – Base URL of the running broken-pipeline-env server
                 (default: http://localhost:8000)
MODEL_NAME     – OpenAI-compatible model name (default: gpt-4o)
HF_TOKEN       – Optional Hugging Face token (not used by OpenAI client but
                 read for completeness / custom deployments)

Usage
-----
    export API_BASE_URL="http://localhost:8000"
    export MODEL_NAME="gpt-4o"
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")  # Reserved for custom deployments
FALLBACK_ACTION: str = "NOOP"
MAX_STEPS: int = 8

# Task to run during inference (can be overridden via env var)
TASK_ID: str = os.environ.get("TASK_ID", "task1")

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

openai_client = OpenAI()  # Reads OPENAI_API_KEY from environment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_env(method: str, path: str, payload: dict | None = None) -> dict:
    """Send an HTTP request to the environment server and return parsed JSON."""
    url = f"{API_BASE_URL}{path}"
    with httpx.Client(timeout=30) as client:
        if method == "GET":
            resp = client.get(url)
        else:
            resp = client.post(url, json=payload or {})
    resp.raise_for_status()
    return resp.json()


def _choose_action(observation: dict) -> str:
    """Ask the LLM to choose the next action based on the observation.

    Returns one of: INSPECT, FIX, VALIDATE, NOOP
    """
    system_prompt = (
        "You are an expert data-pipeline debugger. "
        "Your goal is to solve the task described in the observation. "
        "Available actions: INSPECT, FIX, VALIDATE, NOOP. "
        "Reply with EXACTLY one action keyword and nothing else."
    )
    user_content = (
        f"Task: {observation.get('description', '')}\n"
        f"Step: {observation.get('step', 0)} / {observation.get('max_steps', MAX_STEPS)}\n"
        f"Issues found so far: {observation.get('issues_found', [])}\n"
        f"Last hint: {observation.get('hint', 'none')}\n\n"
        "What is your next action?"
    )

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=10,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip().upper()
        # Accept only valid actions
        valid = {"INSPECT", "FIX", "VALIDATE", "NOOP"}
        action = raw if raw in valid else FALLBACK_ACTION
    except Exception as exc:  # Network / API errors
        print(f"[WARN] LLM call failed ({exc}). Using fallback: {FALLBACK_ACTION}")
        action = FALLBACK_ACTION

    return action


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent(task_id: str = TASK_ID) -> None:
    print(f"\n=== broken-pipeline-env inference ===")
    print(f"Task   : {task_id}")
    print(f"Model  : {MODEL_NAME}")
    print(f"Server : {API_BASE_URL}\n")

    # 1. Reset the environment
    obs = _call_env("POST", "/reset", {"task_id": task_id})
    print(f"[RESET] {obs.get('description', '')}\n")

    cumulative_reward = 0.0

    # 2. Agent loop
    for step_num in range(1, MAX_STEPS + 1):
        action = _choose_action(obs)
        print(f"[STEP {step_num}] Action chosen: {action}")

        result = _call_env("POST", "/step", {"action": action})
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        hint = result.get("info", {}).get("hint", "")
        cumulative_reward += reward

        print(f"         Reward: {reward:+.2f}  |  Cumulative: {cumulative_reward:+.2f}")
        if hint:
            print(f"         Hint  : {hint}")

        obs = result.get("observation", obs)

        if done:
            print(f"\n[DONE] Episode finished at step {step_num}.")
            break
    else:
        print(f"\n[MAX STEPS] Reached maximum of {MAX_STEPS} steps.")

    # 3. Grader
    grader_result = _call_env("GET", "/grader")
    print(f"\n=== Grader Result ===")
    print(json.dumps(grader_result, indent=2))


if __name__ == "__main__":
    task_arg = sys.argv[1] if len(sys.argv) > 1 else TASK_ID
    run_agent(task_id=task_arg)
