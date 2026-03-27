"""tests/test_env.py – Pytest suite for the OpenEnv environment.

Tests
-----
1. test_reset_returns_observation        – reset() returns valid Observation
2. test_step_loop_terminates             – episode ends within max_steps
3. test_grader_deterministic             – repeated grader calls return same score
4. test_all_tasks_reset                  – all three task IDs can be reset
5. test_noop_penalised                   – NOOP gives a negative reward
6. test_step_after_done_raises           – step() raises after episode ends
7. test_task1_optimal_sequence           – INSPECT→VALIDATE yields positive reward and done
8. test_task2_optimal_sequence           – INSPECT→FIX→VALIDATE yields done
9. test_task3_optimal_sequence           – INSPECT→FIX→FIX→VALIDATE yields done
"""

from __future__ import annotations

import pytest

from env.core import OpenEnvEnvironment, MAX_STEPS
from env.models import Observation, StepResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env() -> OpenEnvEnvironment:
    return OpenEnvEnvironment()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation(self, env: OpenEnvEnvironment) -> None:
        obs = env.reset("task1")
        assert isinstance(obs, Observation)
        assert obs.task_id == "task1"
        assert obs.step == 0
        assert obs.max_steps == MAX_STEPS
        assert len(obs.description) > 0

    def test_all_tasks_reset(self, env: OpenEnvEnvironment) -> None:
        for task_id in ("task1", "task2", "task3"):
            obs = env.reset(task_id)
            assert obs.task_id == task_id

    def test_reset_invalid_task_raises(self, env: OpenEnvEnvironment) -> None:
        with pytest.raises(ValueError, match="Unknown task"):
            env.reset("task_unknown")


class TestStep:
    def test_step_returns_step_result(self, env: OpenEnvEnvironment) -> None:
        env.reset("task1")
        result = env.step("INSPECT")
        assert isinstance(result, StepResult)
        assert result.observation.step == 1

    def test_noop_penalised(self, env: OpenEnvEnvironment) -> None:
        env.reset("task1")
        result = env.step("NOOP")
        assert result.reward < 0

    def test_step_loop_terminates(self, env: OpenEnvEnvironment) -> None:
        """Episode must end within MAX_STEPS regardless of actions."""
        env.reset("task1")
        done = False
        for _ in range(MAX_STEPS + 5):  # extra steps to verify termination
            if done:
                break
            result = env.step("NOOP")
            done = result.done
        assert done, "Episode did not terminate within MAX_STEPS"

    def test_step_after_done_raises(self, env: OpenEnvEnvironment) -> None:
        env.reset("task1")
        # Force episode to end by exhausting all steps
        for _ in range(MAX_STEPS):
            env.step("NOOP")
        with pytest.raises(RuntimeError):
            env.step("NOOP")

    def test_invalid_action_falls_back_to_noop(self, env: OpenEnvEnvironment) -> None:
        env.reset("task1")
        result = env.step("TOTALLY_INVALID")
        # Should not raise; treated as NOOP
        assert result.reward == pytest.approx(-0.05)


class TestTask1Sequence:
    def test_optimal_sequence(self, env: OpenEnvEnvironment) -> None:
        env.reset("task1")
        r1 = env.step("INSPECT")
        assert r1.reward == pytest.approx(0.2)

        r2 = env.step("VALIDATE")
        assert r2.reward == pytest.approx(0.2)
        assert r2.done is True


class TestTask2Sequence:
    def test_optimal_sequence(self, env: OpenEnvEnvironment) -> None:
        env.reset("task2")
        env.step("INSPECT")
        env.step("FIX")
        r3 = env.step("VALIDATE")
        assert r3.done is True


class TestTask3Sequence:
    def test_optimal_sequence(self, env: OpenEnvEnvironment) -> None:
        env.reset("task3")
        env.step("INSPECT")
        env.step("FIX")   # phase 1
        env.step("FIX")   # phase 2
        r4 = env.step("VALIDATE")
        assert r4.done is True


class TestGrader:
    def test_grader_deterministic(self, env: OpenEnvEnvironment) -> None:
        """Calling grader twice on the same state returns identical results."""
        env.reset("task1")
        env.step("INSPECT")
        env.step("VALIDATE")

        result1 = env.grade()
        result2 = env.grade()

        assert result1["score"] == result2["score"]
        assert result1["breakdown"] == result2["breakdown"]

    def test_grader_before_reset_raises(self) -> None:
        fresh_env = OpenEnvEnvironment()
        with pytest.raises(RuntimeError):
            fresh_env.grade()

    def test_grader_score_range(self, env: OpenEnvEnvironment) -> None:
        for task_id in ("task1", "task2", "task3"):
            env.reset(task_id)
            result = env.grade()
            assert 0.0 <= result["score"] <= 1.0

    def test_grader_perfect_score_task1(self, env: OpenEnvEnvironment) -> None:
        env.reset("task1")
        env.step("INSPECT")
        env.step("VALIDATE")
        result = env.grade()
        assert result["score"] == pytest.approx(1.0)

    def test_grader_perfect_score_task2(self, env: OpenEnvEnvironment) -> None:
        env.reset("task2")
        env.step("INSPECT")
        env.step("FIX")
        env.step("VALIDATE")
        result = env.grade()
        assert result["score"] == pytest.approx(1.0)

    def test_grader_perfect_score_task3(self, env: OpenEnvEnvironment) -> None:
        env.reset("task3")
        env.step("INSPECT")
        env.step("FIX")
        env.step("FIX")
        env.step("VALIDATE")
        result = env.grade()
        assert result["score"] == pytest.approx(1.0)
