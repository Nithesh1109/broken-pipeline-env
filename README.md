# broken-pipeline-env

An **OpenEnv-style** API environment for training and evaluating LLM agents on
data-pipeline debugging tasks.  It exposes a simple HTTP interface that mirrors
the classic `reset / step / state / grader` loop so any agent can interact with
it without a browser or UI.

---

## Project Structure

```
/inference.py          – Agent loop using the OpenAI client
/openenv.yaml          – Environment configuration
/Dockerfile            – Container build file
/requirements.txt      – Python dependencies

/env/
    __init__.py
    core.py            – OpenEnvEnvironment class (reset / step / state)
    models.py          – Pydantic data models
    server.py          – FastAPI application

    tasks/
        task1_easy.py      – Detect missing values (easy)
        task2_medium.py    – Fix schema/format problems (medium)
        task3_hard.py      – Full pipeline debug (hard)

    graders/
        grader1.py
        grader2.py
        grader3.py

    data/
        generator.py       – Synthetic dataset factory
        bug_injector.py    – Injects null / type / duplicate bugs

/tests/
    test_env.py        – Pytest suite
```

---

## Quick Start

### Run with Docker

```bash
docker build -t broken-pipeline-env .
docker run -p 8000:8000 broken-pipeline-env
```

### Run locally

```bash
pip install -r requirements.txt
uvicorn env.server:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

| Method | Path      | Description                          |
|--------|-----------|--------------------------------------|
| GET    | /ping     | Health check                         |
| GET    | /tasks    | List available tasks                 |
| POST   | /reset    | Reset environment (body: task_id)    |
| POST   | /step     | Take an action (body: ActionType)    |
| GET    | /state    | Get current environment state        |
| GET    | /grader   | Score the current episode            |

### Example

```bash
# Reset with task 1
curl -X POST http://localhost:8000/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "task1"}'

# Take a step
curl -X POST http://localhost:8000/step \
     -H "Content-Type: application/json" \
     -d '{"action": "INSPECT"}'

# Score
curl http://localhost:8000/grader
```

---

## Inference Script

```bash
export API_BASE_URL="http://localhost:8000"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="<your-hf-token-if-needed>"

python inference.py
```

The script runs an 8-step agent loop using the OpenAI chat API, prints the
reward at every step, and calls the grader at the end.

---

## Running Tests

```bash
pytest tests/
```

---

## Reward Structure

| Event          | Reward  |
|----------------|---------|
| Correct action | +0.20   |
| Wrong action   | −0.10   |
| NOOP           | −0.05   |

Episodes terminate after **8 steps** or when the task is solved.
