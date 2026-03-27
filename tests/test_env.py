"""Tests for the OpenEnv broken-pipeline-env project."""

import pytest

from env.core import Environment, MAX_STEPS
from env.models import ActionType, DataAction, GraderResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env(task_id: str = "task1_easy", seed: int = 42) -> Environment:
    env = Environment()
    env.reset(task_id=task_id, seed=seed)
    return env


# ---------------------------------------------------------------------------
# test_reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_returns_observation(self):
        env = Environment()
        obs = env.reset(task_id="task1_easy", seed=42)
        assert obs.step_count == 0
        assert not obs.done
        assert isinstance(obs.data_snapshot, dict)

    def test_reset_task1_has_null_issues(self):
        env = Environment()
        obs = env.reset(task_id="task1_easy", seed=42)
        null_issues = [i for i in obs.issues_found if i.startswith("null:")]
        assert len(null_issues) > 0, "task1_easy should detect null issues"

    def test_reset_task2_has_type_issues(self):
        env = Environment()
        obs = env.reset(task_id="task2_medium", seed=42)
        type_issues = [i for i in obs.issues_found if i.startswith("type_error:")]
        assert len(type_issues) > 0, "task2_medium should detect type errors"

    def test_reset_task3_has_multiple_issues(self):
        env = Environment()
        obs = env.reset(task_id="task3_hard", seed=42)
        assert len(obs.issues_found) > 0, "task3_hard should have multiple issues"

    def test_reset_unknown_task_raises(self):
        env = Environment()
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="task_unknown")

    def test_reset_clears_previous_state(self):
        env = make_env("task1_easy")
        env.step(DataAction(action_type=ActionType.NOOP))
        env.reset(task_id="task2_medium", seed=42)
        assert env._step_count == 0
        assert env._task_id == "task2_medium"

    def test_reset_all_tasks(self):
        for task_id in ["task1_easy", "task2_medium", "task3_hard"]:
            env = Environment()
            obs = env.reset(task_id=task_id, seed=42)
            assert obs.step_count == 0
            assert not obs.done


# ---------------------------------------------------------------------------
# test_step_loop_ends
# ---------------------------------------------------------------------------


class TestStepLoop:
    def test_step_loop_ends_at_max_steps(self):
        env = make_env("task1_easy")
        done = False
        step_count = 0
        while not done and step_count < MAX_STEPS + 5:
            result = env.step(DataAction(action_type=ActionType.NOOP))
            done = result.done
            step_count += 1
        assert done, "Episode should end after MAX_STEPS"
        assert step_count <= MAX_STEPS

    def test_step_after_done_returns_zero_reward(self):
        env = make_env("task1_easy")
        for _ in range(MAX_STEPS):
            env.step(DataAction(action_type=ActionType.NOOP))
        result = env.step(DataAction(action_type=ActionType.NOOP))
        assert result.done
        assert result.reward == 0.0

    def test_noop_gives_small_negative_reward(self):
        env = make_env("task1_easy")
        result = env.step(DataAction(action_type=ActionType.NOOP))
        assert result.reward == pytest.approx(-0.05)

    def test_correct_action_gives_positive_reward_task1(self):
        env = Environment()
        env.reset(task_id="task1_easy", seed=42)
        target_col = env._state["target_columns"][0]
        result = env.step(DataAction(action_type=ActionType.FIX_NULL, target_column=target_col))
        assert result.reward == pytest.approx(0.2)

    def test_wrong_action_gives_negative_reward(self):
        env = make_env("task1_easy")
        result = env.step(
            DataAction(action_type=ActionType.FIX_NULL, target_column="nonexistent_column")
        )
        assert result.reward == pytest.approx(-0.1)

    def test_fix_null_reduces_issues(self):
        env = Environment()
        env.reset(task_id="task1_easy", seed=42)
        target_col = env._state["target_columns"][0]
        before = len(env._state.get("fixed_columns", []))
        env.step(DataAction(action_type=ActionType.FIX_NULL, target_column=target_col))
        after = len(env._state.get("fixed_columns", []))
        assert after > before

    def test_step_increments_step_count(self):
        env = make_env("task1_easy")
        env.step(DataAction(action_type=ActionType.NOOP))
        assert env._step_count == 1
        env.step(DataAction(action_type=ActionType.NOOP))
        assert env._step_count == 2

    def test_step_without_reset_raises(self):
        env = Environment()
        with pytest.raises(RuntimeError, match="reset"):
            env.step(DataAction(action_type=ActionType.NOOP))

    def test_task3_fix_duplicate_action(self):
        env = Environment()
        env.reset(task_id="task3_hard", seed=42)
        result = env.step(DataAction(action_type=ActionType.FIX_DUPLICATE))
        assert result.reward == pytest.approx(0.2)

    def test_task3_debug_pipeline_action(self):
        env = Environment()
        env.reset(task_id="task3_hard", seed=42)
        result = env.step(DataAction(action_type=ActionType.DEBUG_PIPELINE))
        assert result.reward == pytest.approx(0.2)
        assert "issues" in result.info

    def test_task2_fix_type_action(self):
        env = Environment()
        env.reset(task_id="task2_medium", seed=42)
        target_col = env._state["target_columns"][0]
        result = env.step(DataAction(action_type=ActionType.FIX_TYPE, target_column=target_col))
        assert result.reward == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# test_grader_deterministic
# ---------------------------------------------------------------------------


class TestGraderDeterministic:
    def _run_fixed_episode(self, task_id: str) -> GraderResult:
        env = Environment()
        env.reset(task_id=task_id, seed=42)
        for _ in range(MAX_STEPS):
            env.step(DataAction(action_type=ActionType.NOOP))
        return env.grade()

    def test_grader_task1_deterministic(self):
        result1 = self._run_fixed_episode("task1_easy")
        result2 = self._run_fixed_episode("task1_easy")
        assert result1.score == result2.score

    def test_grader_task2_deterministic(self):
        result1 = self._run_fixed_episode("task2_medium")
        result2 = self._run_fixed_episode("task2_medium")
        assert result1.score == result2.score

    def test_grader_task3_deterministic(self):
        result1 = self._run_fixed_episode("task3_hard")
        result2 = self._run_fixed_episode("task3_hard")
        assert result1.score == result2.score

    def test_grader_score_between_0_and_1(self):
        for task_id in ["task1_easy", "task2_medium", "task3_hard"]:
            result = self._run_fixed_episode(task_id)
            assert 0.0 <= result.score <= 1.0, f"{task_id}: score={result.score} out of range"

    def test_grader_score_improves_with_correct_actions_task1(self):
        env = Environment()
        env.reset(task_id="task1_easy", seed=42)
        baseline = env.grade().score

        # Apply correct actions
        for col in list(env._state["target_columns"]):
            env.step(DataAction(action_type=ActionType.FIX_NULL, target_column=col))
        improved = env.grade().score
        assert improved >= baseline

    def test_grader_returns_grader_result(self):
        env = make_env("task1_easy")
        result = env.grade()
        assert isinstance(result, GraderResult)
        assert hasattr(result, "score")
        assert hasattr(result, "details")

    def test_grader_no_episode_returns_zero(self):
        env = Environment()
        result = env.grade()
        assert result.score == 0.0

    def test_grader_details_is_dict(self):
        env = make_env("task2_medium")
        result = env.grade()
        assert isinstance(result.details, dict)


# ---------------------------------------------------------------------------
# test_state
# ---------------------------------------------------------------------------


class TestState:
    def test_state_before_reset(self):
        env = Environment()
        s = env.state()
        assert s["status"] == "not_started"

    def test_state_after_reset(self):
        env = make_env("task1_easy")
        s = env.state()
        assert s["task_id"] == "task1_easy"
        assert s["step_count"] == 0
        assert "data_snapshot" in s

    def test_list_tasks_returns_all(self):
        env = Environment()
        tasks = env.list_tasks()
        task_ids = [t["task_id"] for t in tasks]
        assert "task1_easy" in task_ids
        assert "task2_medium" in task_ids
        assert "task3_hard" in task_ids

    def test_list_tasks_has_required_fields(self):
        env = Environment()
        for task in env.list_tasks():
            assert "task_id" in task
            assert "name" in task
            assert "difficulty" in task
            assert "max_steps" in task
