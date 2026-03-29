import os
import json
import sys
import time

import requests
from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

if not API_BASE_URL:
	raise EnvironmentError("API_BASE_URL environment variable is not set.")
if not MODEL_NAME:
	raise EnvironmentError("MODEL_NAME environment variable is not set.")
if not HF_TOKEN:
	raise EnvironmentError("HF_TOKEN environment variable is not set.")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")

client = OpenAI(api_key=HF_TOKEN, base_url=LLM_BASE_URL)

MAX_STEPS = 8
MAX_RUNTIME_SECS = 19 * 60
EPISODE_START = None

FALLBACK_ACTION = {
	"action_type": "NOOP",
	"target_column": None,
	"transformation": None,
	"justification": "Fallback NOOP — LLM output was not valid JSON.",
	"identified_issues": None,
}

SYSTEM_PROMPT = """
You are a senior data engineer on call. A production data pipeline is broken.
Inspect the dataset, identify all issues, fix the highest severity ones first,
then validate. If you see SSN data in analytics output, mask it immediately.

Reply ONLY with valid JSON — no markdown, no explanation, no code fences:
{
  "action_type": "INSPECT" | "RENAME_COLUMN" | "CAST_TYPE" | "FILL_DEFAULT" |
				 "DROP_COLUMN" | "VALIDATE" | "MASK_PII" | "NOOP",
  "target_column": "column_name" or null,
  "transformation": "cast_to_int" | "cast_to_float" | "fill_median" | "fill_zero" |
					"drop_duplicates" | null,
  "justification": "One sentence explaining your reasoning.",
  "identified_issues": [...] or null
}
"""


def check_runtime() -> None:
	"""Raise if total elapsed time exceeds MAX_RUNTIME_SECS."""
	elapsed = time.time() - EPISODE_START
	if elapsed > MAX_RUNTIME_SECS:
		print(f"\n[TIMEOUT] Exceeded {MAX_RUNTIME_SECS/60:.0f} min limit. Stopping.")
		sys.exit(1)


def observation_to_prompt(obs: dict) -> str:
	return (
		f"Pipeline stage: {obs['pipeline_stage']}\n"
		f"Steps remaining: {obs['time_remaining']}\n"
		f"Downstream health: {obs['downstream_health']:.2f}\n\n"
		f"Column schema:\n{json.dumps(obs.get('schema', {}), indent=2)}\n\n"
		f"Dataset preview:\n{json.dumps(obs['dataset_preview'][:10], indent=2)}\n\n"
		f"Current issues:\n{json.dumps(obs['validation_report'], indent=2)}\n\n"
		f"What is your next action?"
	)


def parse_action(raw: str) -> dict:
	try:
		text = raw.strip()
		if "```" in text:
			parts = text.split("```")
			for part in parts:
				part = part.strip()
				if part.startswith("json"):
					part = part[4:].strip()
				try:
					return json.loads(part)
				except Exception:
					continue
		action = json.loads(text)
		valid_types = {
			"INSPECT",
			"RENAME_COLUMN",
			"CAST_TYPE",
			"FILL_DEFAULT",
			"DROP_COLUMN",
			"VALIDATE",
			"MASK_PII",
			"NOOP",
		}
		if action.get("action_type") not in valid_types:
			return FALLBACK_ACTION
		if "justification" not in action:
			return FALLBACK_ACTION
		return action
	except Exception:
		return FALLBACK_ACTION


def run_episode(task_id: int) -> float:
	check_runtime()

	obs = requests.post(f"{API_BASE_URL}/reset", params={"task_id": task_id}, timeout=30).json()

	messages = [{"role": "system", "content": SYSTEM_PROMPT}]

	for step_num in range(MAX_STEPS):
		check_runtime()

		user_msg = observation_to_prompt(obs)
		messages.append({"role": "user", "content": user_msg})

		try:
			response = client.chat.completions.create(
				model=MODEL_NAME,
				messages=messages,
				temperature=0.2,
				max_tokens=512,
			)
			raw = response.choices[0].message.content
		except Exception as exc:
			print(f"  [LLM ERROR] {exc} — applying fallback")
			raw = json.dumps(FALLBACK_ACTION)

		action = parse_action(raw)
		messages.append({"role": "assistant", "content": raw})

		result = requests.post(
			f"{API_BASE_URL}/step",
			json=action,
			params={"task_id": task_id},
			timeout=30,
		).json()

		obs = result["observation"]
		done = result["done"]

		print(
			f"  Step {step_num+1:02d} | "
			f"{action['action_type']:15s} | "
			f"reward={result['reward']:+.3f} | "
			f"health={obs['downstream_health']:.2f} | "
			f"done={done}"
		)

		if done:
			break

	grade = requests.get(f"{API_BASE_URL}/grader", params={"task_id": task_id}, timeout=30).json()

	return float(grade["score"])


if __name__ == "__main__":
	EPISODE_START = time.time()
	scores: dict[int, float] = {}

	for task_id in [1, 2, 3]:
		print(f"\n{'='*60}")
		print(f"  TASK {task_id}")
		print(f"{'='*60}")
		scores[task_id] = run_episode(task_id)
		elapsed = time.time() - EPISODE_START
		print(f"  -> Score: {scores[task_id]:.4f}  (elapsed: {elapsed:.0f}s)")

	avg = sum(scores.values()) / 3
	print(f"\n{'='*60}")
	print(f"  T1={scores[1]:.4f}  T2={scores[2]:.4f}  T3={scores[3]:.4f}")
	print(f"  AVERAGE: {avg:.4f}")
	print(f"{'='*60}")
