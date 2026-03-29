from fastapi.testclient import TestClient

from env.server import app


client = TestClient(app)


def _sample_action(task_id: str) -> dict:
	if task_id == "task1":
		return {
			"task_id": "task1",
			"findings": ["duplicate IDs", "missing salary", "negative salary"],
			"remediations": ["deduplicate records", "fill missing salary from source"],
			"notes": "audit complete",
			"metadata": {},
		}
	if task_id == "task2":
		return {
			"task_id": "task2",
			"findings": ["team column replacing department", "salary typed as string"],
			"remediations": [
				"rename team to department",
				"cast salary to int",
				"drop region",
			],
			"notes": "schema drift repaired",
			"metadata": {},
		}
	return {
		"task_id": "task3",
		"findings": ["incident severity high"],
		"remediations": [
			"isolate bad records",
			"rollback schema change",
			"backfill critical fields",
			"add monitoring alerts",
		],
		"notes": "incident handled",
		"metadata": {},
	}


def test_health_endpoint():
	response = client.get("/health")
	assert response.status_code == 200
	assert response.json()["status"] == "ok"


def test_task_listing():
	response = client.get("/tasks")
	assert response.status_code == 200
	assert response.json()["tasks"] == ["task1", "task2", "task3"]


def test_task1_observe_and_grade():
	assert client.post("/reset/task1").status_code == 200
	obs = client.get("/observe/task1")
	assert obs.status_code == 200
	assert obs.json()["task_id"] == "task1"
	result = client.post("/act/task1", json=_sample_action("task1"))
	assert result.status_code == 200
	payload = result.json()
	assert 0.0 <= payload["score"] <= 1.0


def test_task2_observe_and_grade():
	assert client.post("/reset/task2").status_code == 200
	obs = client.get("/observe/task2")
	assert obs.status_code == 200
	assert obs.json()["task_id"] == "task2"
	result = client.post("/act/task2", json=_sample_action("task2"))
	assert result.status_code == 200
	payload = result.json()
	assert 0.0 <= payload["score"] <= 1.0


def test_task3_observe_and_grade():
	assert client.post("/reset/task3").status_code == 200
	obs = client.get("/observe/task3")
	assert obs.status_code == 200
	assert obs.json()["task_id"] == "task3"
	result = client.post("/act/task3", json=_sample_action("task3"))
	assert result.status_code == 200
	payload = result.json()
	assert 0.0 <= payload["score"] <= 1.0
