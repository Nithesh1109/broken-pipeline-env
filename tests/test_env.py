"""
Tests for the broken-pipeline-env package.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.core import BrokenPipelineEnv
from env.models import Action, EpisodeResult, Observation, StepResult
from env.graders.grader1 import AuditGrader
from env.graders.grader2 import SchemaGrader
from env.graders.grader3 import IncidentGrader
from env.data.generator import ScenarioGenerator
from env.data.bug_injector import BugInjector


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestObservation:
    def test_to_dict(self):
        obs = Observation(task_id="t1", step=0, data={"key": "val"})
        d = obs.to_dict()
        assert d["task_id"] == "t1"
        assert d["step"] == 0
        assert d["data"] == {"key": "val"}
        assert d["errors"] == []

    def test_to_dict_with_errors(self):
        obs = Observation(task_id="t1", step=1, data={}, errors=["e1", "e2"])
        d = obs.to_dict()
        assert d["errors"] == ["e1", "e2"]


class TestAction:
    def test_to_dict(self):
        a = Action(action_type="identify_anomalies", payload={"indices": [1, 2]})
        d = a.to_dict()
        assert d["action_type"] == "identify_anomalies"
        assert d["payload"]["indices"] == [1, 2]


class TestStepResult:
    def test_to_dict(self):
        obs = Observation(task_id="t1", step=1, data={})
        sr = StepResult(observation=obs, reward=0.5, done=False)
        d = sr.to_dict()
        assert d["reward"] == 0.5
        assert d["done"] is False


# ---------------------------------------------------------------------------
# ScenarioGenerator tests
# ---------------------------------------------------------------------------

class TestScenarioGenerator:
    def setup_method(self):
        self.gen = ScenarioGenerator(seed=0)

    def test_audit_scenario_structure(self):
        s = self.gen.generate_audit_scenario(num_logs=10, num_anomalies=2)
        assert "logs" in s
        assert "anomaly_indices" in s
        assert len(s["logs"]) == 10
        assert len(s["anomaly_indices"]) == 2

    def test_audit_anomaly_indices_valid(self):
        s = self.gen.generate_audit_scenario(num_logs=10, num_anomalies=3)
        for idx in s["anomaly_indices"]:
            assert 0 <= idx < 10

    def test_schema_scenario_structure(self):
        s = self.gen.generate_schema_scenario(num_records=10, num_violations=2)
        assert "records" in s
        assert "schema" in s
        assert "violations" in s
        assert len(s["records"]) == 10
        assert len(s["violations"]) == 2

    def test_incident_scenario_structure(self):
        s = self.gen.generate_incident_scenario()
        assert "incident_report" in s
        assert "root_cause" in s
        assert "accepted_causes" in s
        assert "mitigation_keywords" in s


# ---------------------------------------------------------------------------
# BugInjector tests
# ---------------------------------------------------------------------------

class TestBugInjector:
    def setup_method(self):
        self.injector = BugInjector(seed=1)

    def test_inject_audit_bugs_returns_correct_count(self):
        logs = [
            {"index": i, "status": "SUCCESS", "duration_ms": 100, "records_processed": 500}
            for i in range(10)
        ]
        indices = self.injector.inject_audit_bugs(logs, num_bugs=3)
        assert len(indices) == 3
        assert all(0 <= i < 10 for i in indices)

    def test_inject_schema_bugs_returns_violations(self):
        schema = {
            "fields": {
                "age": {"type": "int", "required": True, "min": 0, "max": 120}
            }
        }
        records = [{"age": 25} for _ in range(5)]
        violations = self.injector.inject_schema_bugs(records, schema, num_bugs=2)
        assert len(violations) == 2
        for v in violations:
            assert "record_index" in v
            assert "field" in v
            assert "violation_type" in v

    def test_inject_incident_bug(self):
        report = {"symptoms": []}
        modified = self.injector.inject_incident_bug(report, bug_type="disk_full")
        assert "disk" in modified["symptoms"][-1].lower() or modified["injected_bug"] == "disk_full"
        assert modified["injected_bug"] == "disk_full"


# ---------------------------------------------------------------------------
# Task 1 – AuditTask / BrokenPipelineEnv
# ---------------------------------------------------------------------------

class TestAuditTask:
    def test_reset_returns_observation(self):
        env = BrokenPipelineEnv("task1_audit")
        obs = env.reset()
        assert isinstance(obs, Observation)
        assert obs.task_id == "task1_audit"
        assert obs.step == 0
        assert "logs" in obs.data

    def test_correct_identification_gives_full_reward(self):
        env = BrokenPipelineEnv("task1_audit")
        env.reset()
        anomaly_indices = env._task._anomaly_indices
        action = Action(
            action_type="identify_anomalies",
            payload={"indices": anomaly_indices},
        )
        result = env.step(action)
        assert isinstance(result, StepResult)
        assert result.reward == pytest.approx(1.0, abs=1e-4)
        assert result.done is True

    def test_empty_identification_gives_zero_reward(self):
        env = BrokenPipelineEnv("task1_audit")
        env.reset()
        action = Action(action_type="identify_anomalies", payload={"indices": []})
        result = env.step(action)
        assert result.reward == pytest.approx(0.0, abs=1e-4)

    def test_unknown_action_gives_penalty(self):
        env = BrokenPipelineEnv("task1_audit")
        env.reset()
        action = Action(action_type="bad_action", payload={})
        result = env.step(action)
        assert result.reward < 0

    def test_episode_result_after_solve(self):
        env = BrokenPipelineEnv("task1_audit")
        env.reset()
        action = Action(
            action_type="identify_anomalies",
            payload={"indices": env._task._anomaly_indices},
        )
        env.step(action)
        result = env.get_episode_result()
        assert isinstance(result, EpisodeResult)
        assert result.success is True
        assert result.task_id == "task1_audit"

    def test_step_before_reset_raises(self):
        env = BrokenPipelineEnv("task1_audit")
        with pytest.raises(RuntimeError):
            env.step(Action(action_type="identify_anomalies", payload={"indices": []}))

    def test_step_after_done_raises(self):
        env = BrokenPipelineEnv("task1_audit")
        env.reset()
        action = Action(
            action_type="identify_anomalies",
            payload={"indices": env._task._anomaly_indices},
        )
        env.step(action)
        with pytest.raises(RuntimeError):
            env.step(action)


# ---------------------------------------------------------------------------
# Task 2 – SchemaTask
# ---------------------------------------------------------------------------

class TestSchemaTask:
    def test_reset_returns_observation(self):
        env = BrokenPipelineEnv("task2_schema")
        obs = env.reset()
        assert obs.task_id == "task2_schema"
        assert "records" in obs.data
        assert "schema" in obs.data

    def test_perfect_report_gives_full_reward(self):
        env = BrokenPipelineEnv("task2_schema")
        env.reset()
        violations = env._task._violations
        action = Action(
            action_type="report_violations",
            payload={"violations": violations},
        )
        result = env.step(action)
        assert result.reward == pytest.approx(1.0, abs=1e-4)
        assert result.done is True

    def test_empty_report_gives_zero_reward(self):
        env = BrokenPipelineEnv("task2_schema")
        env.reset()
        action = Action(action_type="report_violations", payload={"violations": []})
        result = env.step(action)
        assert result.reward == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Task 3 – IncidentTask
# ---------------------------------------------------------------------------

class TestIncidentTask:
    def test_reset_returns_observation(self):
        env = BrokenPipelineEnv("task3_incident")
        obs = env.reset()
        assert obs.task_id == "task3_incident"
        assert "incident_report" in obs.data

    def test_correct_cause_and_mitigation_gives_high_reward(self):
        env = BrokenPipelineEnv("task3_incident")
        env.reset()
        root_cause = env._task._root_cause
        keywords = env._task._mitigation_keywords
        action = Action(
            action_type="diagnose",
            payload={
                "root_cause": root_cause,
                "mitigation_steps": keywords,
            },
        )
        result = env.step(action)
        assert result.reward > 0.6

    def test_wrong_cause_reduces_reward(self):
        env = BrokenPipelineEnv("task3_incident")
        env.reset()
        action = Action(
            action_type="diagnose",
            payload={"root_cause": "totally_wrong_cause", "mitigation_steps": []},
        )
        result = env.step(action)
        assert result.reward < 0.6


# ---------------------------------------------------------------------------
# Invalid task_id
# ---------------------------------------------------------------------------

class TestInvalidTaskId:
    def test_unknown_task_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown task_id"):
            BrokenPipelineEnv("nonexistent_task")


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

class TestAuditGrader:
    def setup_method(self):
        self.grader = AuditGrader()

    def test_perfect_score(self):
        report = self.grader.grade([1, 2, 3], [1, 2, 3])
        assert report["f1"] == pytest.approx(1.0, abs=1e-4)
        assert report["grade"] == "A"

    def test_zero_score(self):
        report = self.grader.grade([1, 2, 3], [])
        assert report["f1"] == pytest.approx(0.0, abs=1e-4)
        assert report["grade"] == "F"

    def test_partial_score(self):
        report = self.grader.grade([1, 2, 3, 4], [1, 2])
        assert 0 < report["f1"] < 1.0


class TestSchemaGrader:
    def setup_method(self):
        self.grader = SchemaGrader()

    def test_perfect_score(self):
        v = [{"record_index": 0, "field": "age", "violation_type": "wrong_type"}]
        report = self.grader.grade(v, v)
        assert report["f1"] == pytest.approx(1.0, abs=1e-4)

    def test_empty_report(self):
        v = [{"record_index": 0, "field": "age", "violation_type": "wrong_type"}]
        report = self.grader.grade(v, [])
        assert report["f1"] == pytest.approx(0.0, abs=1e-4)


class TestIncidentGrader:
    def setup_method(self):
        self.grader = IncidentGrader()

    def test_perfect_score(self):
        report = self.grader.grade(
            root_cause="deadlock",
            accepted_causes=["deadlock"],
            mitigation_keywords=["lock", "transaction", "rollback"],
            agent_diagnosis="deadlock",
            agent_mitigation=["Resolve lock contention", "Rollback transaction"],
        )
        assert report["cause_correct"] is True
        assert report["overall_score"] > 0.6

    def test_wrong_cause(self):
        report = self.grader.grade(
            root_cause="deadlock",
            accepted_causes=["deadlock"],
            mitigation_keywords=["lock"],
            agent_diagnosis="memory_leak",
            agent_mitigation=["restart service"],
        )
        assert report["cause_correct"] is False
        assert report["cause_score"] == 0.0
