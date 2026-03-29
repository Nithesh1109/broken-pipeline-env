# broken-pipeline-env

An agent evaluation environment for debugging data pipelines with increasing difficulty.

## Tasks

- `task1` (easy): data quality audit
- `task2` (medium): schema drift remediation
- `task3` (hard): full incident response

## Action Space

All actions are posted to `POST /act/{task_id}` as `DataAction`:

- `task_id` (string): one of `task1`, `task2`, `task3`
- `findings` (list of strings): what the agent found
- `remediations` (list of strings): fixes or mitigations the agent proposes
- `notes` (string, optional): concise task notes
- `metadata` (object): optional task-specific metadata

## Observation Space

Observations are returned by `GET /observe/{task_id}` as `DataObservation`:

- `task_id` (string)
- `status` (string): `ok`, `warning`, or `critical`
- `message` (string): human-readable summary
- `metrics` (object): numeric/task metrics
- `payload` (object): task-specific structured details

## Grading

Each action is graded with `POST /act/{task_id}` and returns `GraderResult`:

- `score` in `[0.0, 1.0]`
- `passed` threshold defaults to `>= 0.7`
- task-specific feedback

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn inference:app --reload --port 8000
```

## Run Tests

```bash
pytest -q
```

## Baseline Scores

- Minimal/no-op agent: ~`0.20` to `0.35`
- Rule-based agent with explicit checks: ~`0.60` to `0.78`
- Strong agent with grounded remediations: ~`0.80` to `1.00`