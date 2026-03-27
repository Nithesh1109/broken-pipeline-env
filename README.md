# broken-pipeline-env

An **OpenEnv**-compatible environment for training and evaluating LLM agents on data pipeline debugging tasks.

## Overview

This project implements a standardised OpenEnv interface exposing three tasks of increasing difficulty:

| Task | Difficulty | Goal |
|------|-----------|------|
| `task1_easy` | Easy | Detect null / missing values |
| `task2_medium` | Medium | Fix schema / type problems |
| `task3_hard` | Hard | Full pipeline debugging (nulls + types + duplicates) |

## Project Structure

```
/inference.py          # Agent loop using OpenAI client
/openenv.yaml          # Environment configuration
/Dockerfile            # Container definition
/requirements.txt      # Python dependencies

/env/
    __init__.py
    core.py            # Environment class (reset / step / state)
    models.py          # Pydantic models
    server.py          # FastAPI server
    tasks/
        task1_easy.py
        task2_medium.py
        task3_hard.py
    graders/
        grader1.py
        grader2.py
        grader3.py
    data/
        generator.py   # Synthetic dataset generator
        bug_injector.py

/tests/
    test_env.py
```

## Quick Start

### Run the server

```bash
pip install -r requirements.txt
uvicorn env.server:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker build -t broken-pipeline-env .
docker run -p 8000:8000 broken-pipeline-env
```

### Run the agent (requires OpenAI-compatible endpoint)

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...
python inference.py
```

### Run tests

```bash
pytest tests/ -v
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/ping` | Health check |
| GET | `/tasks` | List available tasks |
| POST | `/reset` | Reset environment to a new episode |
| POST | `/step` | Take an action, receive observation + reward |
| GET | `/state` | Get current environment state |
| POST | `/grader` | Score agent performance deterministically |

## Reward Structure

- **Correct action**: `+0.2`
- **Wrong action**: `-0.1`
- **NOOP**: `-0.05`
- **Max steps per episode**: `8`
