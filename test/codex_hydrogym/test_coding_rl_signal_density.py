"""CPU-only contracts for coding-agent PPO signal selection helpers."""

import importlib
import sys
import types

# coding_rl imports MLflow only for runtime tracking; these helper contracts do not use it.
sys.modules.setdefault("mlflow", types.ModuleType("mlflow"))

experiment = importlib.import_module("codex_hydrogym.coding_rl.experiment")


def _evaluation(status, reward, full_repair=False, unsafe=False):
    return {
        "status": status,
        "reward": reward,
        "full_repair": full_repair,
        "unsafe": unsafe,
    }


def test_difficulty_screening_selects_only_band_and_is_deterministic():
    tasks = tuple(task for task in experiment.repair_tasks() if task.split == "train")[:3]
    solve_rates = {tasks[0].task_id: 0.0, tasks[1].task_id: 0.25, tasks[2].task_id: 0.75}

    first = experiment.select_tasks_in_solve_rate_band(tasks, solve_rates, minimum=0.25, maximum=0.75)
    second = experiment.select_tasks_in_solve_rate_band(tasks, solve_rates, minimum=0.25, maximum=0.75)

    assert [task.task_id for task in first] == [tasks[1].task_id, tasks[2].task_id]
    assert first == second
    assert all(0.25 <= solve_rates[task.task_id] <= 0.75 for task in first)


def test_base_solve_rate_estimation_reuses_evaluator_deterministically(monkeypatch, tmp_path):
    tasks = tuple(task for task in experiment.repair_tasks() if task.split == "train")[:2]
    calls = []

    def fake_evaluate_policy(**kwargs):
        calls.append(kwargs)
        return [
            {"task_id": tasks[0].task_id, "full_repair": kwargs["condition"] != "base_screen_1"},
            {"task_id": tasks[1].task_id, "full_repair": False},
        ]

    monkeypatch.setattr(experiment, "evaluate_policy", fake_evaluate_policy)
    rates = experiment.estimate_base_solve_rates(
        trainer=object(),
        tokenizer=object(),
        tasks=tasks,
        model_id="base",
        snapshot_root=tmp_path,
        corpus_digest="digest",
        generation_batch_size=2,
        trials=2,
    )

    assert rates == {tasks[0].task_id: 0.5, tasks[1].task_id: 0.0}
    assert [call["condition"] for call in calls] == ["base_screen_0", "base_screen_1"]
    assert all(call["do_sample"] is True and call["trace_records"] is False for call in calls)


def test_signal_density_distinguishes_dead_and_live_updates():
    dead = experiment.signal_density_metrics([_evaluation("executed", -0.5), _evaluation("executed", -0.5)])
    live = experiment.signal_density_metrics(
        [_evaluation("executed", -0.5), _evaluation("executed", 1.25, full_repair=True)]
    )

    assert dead == {
        "batch_distinct_reward_values": 1.0,
        "batch_has_success_and_failure": 0.0,
        "batch_is_dead": 1.0,
    }
    assert live == {
        "batch_distinct_reward_values": 2.0,
        "batch_has_success_and_failure": 1.0,
        "batch_is_dead": 0.0,
    }


def test_infrastructure_mask_excludes_only_timeouts():
    evaluations = [
        _evaluation("execution_timeout", -0.75),
        _evaluation("execution_error", -0.75),
        _evaluation("invalid_patch_envelope", -1.0),
        _evaluation("forbidden_ast_node", -1.25, unsafe=True),
        _evaluation("executed", 1.25, full_repair=True),
    ]

    retained = experiment.mask_infrastructure_failures(evaluations, enabled=True)

    assert [value["status"] for value in retained] == [
        "execution_error",
        "invalid_patch_envelope",
        "forbidden_ast_node",
        "executed",
    ]


def test_all_opt_in_flags_off_preserve_current_helper_behavior():
    evaluations = [
        _evaluation("execution_timeout", -0.75),
        _evaluation("invalid_patch_envelope", -1.0),
    ]

    # The masking switch is off by default, so every original policy reward is retained.
    assert experiment.mask_infrastructure_failures(evaluations, enabled=False) == evaluations


def test_masked_step_never_calls_trainer_with_underfilled_batch():
    class FakeTrainer:
        def __init__(self):
            self.calls = []
        def tensor(self, value):
            return value
        def step(self, queries, responses, scores):
            self.calls.append(len(scores))
            return {}

    trainer = FakeTrainer()
    evaluations = [_evaluation("execution_timeout", -0.75), _evaluation("executed", 1.0)]
    stats, retained, skipped = experiment.masked_ppo_step(
        trainer=trainer, queries=["q0", "q1"], responses=["r0", "r1"],
        evaluations=evaluations, batch_size=2,
    )
    assert stats == {} and len(retained) == 1 and skipped == 1
    assert trainer.calls == []
    stats, retained, skipped = experiment.masked_ppo_step(
        trainer=trainer, queries=["q0", "q1"], responses=["r0", "r1"],
        evaluations=[_evaluation("executed", 1.0), _evaluation("executed", 0.5)], batch_size=2,
    )
    assert stats == {} and len(retained) == 2 and skipped == 0
    assert trainer.calls == [2]


def test_unserializable_policy_outputs_are_not_masked_as_infrastructure(tmp_path):
    task = next(task for task in experiment.repair_tasks() if task.split == "train")
    for expression in ("float('nan')", "set()"):
        result = experiment.evaluate_response(
            task=task, response=f"PATCH: {expression}", condition="test", sequence=0,
            snapshot_root=tmp_path / expression.replace("'", ""),
        )
        assert result["status"] == "execution_error"
        assert experiment.mask_infrastructure_failures([result], enabled=True) == [result]
