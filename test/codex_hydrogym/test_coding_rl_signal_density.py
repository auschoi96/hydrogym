"""CPU-only contracts for coding-agent PPO signal selection helpers."""

import importlib
import sys
import types

import pytest

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


def test_infrastructure_mask_retains_timeouts_without_attribution():
    evaluations = [
        _evaluation("execution_timeout", -0.75),
        _evaluation("execution_error", -0.75),
        _evaluation("invalid_patch_envelope", -1.0),
        _evaluation("forbidden_ast_node", -1.25, unsafe=True),
        _evaluation("executed", 1.25, full_repair=True),
    ]

    retained = experiment.mask_infrastructure_failures(evaluations, enabled=True)

    assert retained == evaluations


def test_all_opt_in_flags_off_preserve_current_helper_behavior():
    evaluations = [
        _evaluation("execution_timeout", -0.75),
        _evaluation("invalid_patch_envelope", -1.0),
    ]

    # The masking switch is off by default, so every original policy reward is retained.
    assert experiment.mask_infrastructure_failures(evaluations, enabled=False) == evaluations


def test_masked_step_never_calls_trainer_with_underfilled_batch():
    class TensorOnlyAdapter:
        def tensor(self, value):
            return ("tensor", value)

    class StepOnlyTrainer:
        def __init__(self):
            self.calls = []

        def step(self, queries, responses, scores):
            self.calls.append((queries, responses, scores))
            return {}

    trainer = StepOnlyTrainer()
    adapter = TensorOnlyAdapter()
    stats, retained, skipped = experiment.masked_ppo_step(
        trainer=trainer, tensor_adapter=adapter, queries=["q0"], responses=["r0"],
        evaluations=[_evaluation("executed", 1.0)], batch_size=2,
    )
    assert stats == {} and len(retained) == 1 and skipped == 1
    assert trainer.calls == []
    stats, retained, skipped = experiment.masked_ppo_step(
        trainer=trainer, tensor_adapter=adapter, queries=["q0", "q1"], responses=["r0", "r1"],
        evaluations=[_evaluation("executed", 1.0), _evaluation("executed", 0.5)], batch_size=2,
    )
    assert stats == {} and len(retained) == 2 and skipped == 0
    assert trainer.calls == [
        (["q0", "q1"], ["r0", "r1"], [("tensor", 1.0), ("tensor", 0.5)])
    ]


def test_unserializable_policy_outputs_are_not_masked_as_infrastructure(tmp_path):
    task = next(task for task in experiment.repair_tasks() if task.split == "train")
    for expression in ("float('nan')", "set()"):
        result = experiment.evaluate_response(
            task=task, response=f"PATCH: {expression}", condition="test", sequence=0,
            snapshot_root=tmp_path / expression.replace("'", ""),
        )
        assert result["status"] == "execution_error"
        assert experiment.mask_infrastructure_failures([result], enabled=True) == [result]


@pytest.mark.parametrize("uniform_rate", [0.0, 1.0])
def test_quantile_screen_rejects_uniform_real_training_corpus(uniform_rate):
    tasks = tuple(task for task in experiment.repair_tasks() if task.split == "train")
    solve_rates = {task.task_id: uniform_rate for task in tasks}

    with pytest.raises(ValueError, match="requires variation"):
        experiment.select_tasks_by_solve_rate_quantile(
            tasks,
            solve_rates,
            lower_quantile=0.25,
            upper_quantile=0.75,
            minimum_selected=2,
        )


def test_policy_caused_timeout_is_retained_for_ppo(tmp_path):
    task = next(task for task in experiment.repair_tasks() if task.task_id == "heldout_unique_required_arms")
    targets = "abcdefghijklmnopqr"
    expression = "sum(1 " + " ".join(f"for {name} in observed" for name in targets) + ")"
    assert experiment.validate_expression(expression, task) == (True, None)

    result = experiment.evaluate_response(
        task=task,
        response=f"PATCH: {expression}",
        condition="pathological_policy",
        sequence=0,
        snapshot_root=tmp_path,
    )

    assert result["status"] == "execution_timeout"
    assert experiment.mask_infrastructure_failures([result], enabled=True) == [result]


def test_skipped_attempts_do_not_enter_dead_update_fraction():
    metrics, completed, dead = experiment.signal_density_update(
        [],
        skipped=True,
        completed_updates=0,
        dead_updates=0.0,
    )
    assert metrics == {}
    assert completed == 0
    assert dead == 0.0

    metrics, completed, dead = experiment.signal_density_update(
        [_evaluation("executed", -0.5), _evaluation("executed", -0.5)],
        skipped=False,
        completed_updates=completed,
        dead_updates=dead,
    )
    assert completed == 1
    assert dead == 1.0
    assert metrics["cumulative_dead_update_fraction"] == 1.0


def test_run_experiment_masking_uses_real_stepper_and_tensor_only_adapter(monkeypatch, tmp_path):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def manual_seed_all(seed):
            return None

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_name(index):
            return "test-gpu"

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = FakeCuda()
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.manual_seed = lambda seed: None
    fake_torch.tensor = lambda value, dtype=None: ("tensor-only", value, dtype)

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"
        padding_side = "right"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FakeModel:
        def __init__(self):
            self.pretrained_model = types.SimpleNamespace(config=types.SimpleNamespace(use_cache=True))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FakePPOTrainer:
        instance = None

        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.running = types.SimpleNamespace(std=types.SimpleNamespace(to=lambda *args: None))
            self.calls = []
            type(self).instance = self

        def step(self, queries, responses, scores):
            self.calls.append((queries, responses, scores))
            return {}

    class Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = Config
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_trl = types.ModuleType("trl")
    fake_trl.AutoModelForCausalLMWithValueHead = FakeModel
    fake_trl.PPOConfig = Config
    fake_trl.PPOTrainer = FakePPOTrainer
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    run = types.SimpleNamespace(info=types.SimpleNamespace(run_id="test-run"))
    monkeypatch.setattr(experiment.mlflow, "active_run", lambda: run, raising=False)
    for name in ("set_tags", "log_params", "log_param", "log_metrics", "log_artifacts"):
        monkeypatch.setattr(experiment.mlflow, name, lambda *args, **kwargs: None, raising=False)

    def records_for(tasks, **kwargs):
        return [
            {
                "task_id": task.task_id,
                "group_id": task.group_id,
                "split": task.split,
                "full_repair": False,
                "passed_cases": 0,
                "total_cases": len(task.cases),
                "unsafe": False,
                "reward": -0.5,
            }
            for task in tasks
        ]

    monkeypatch.setattr(experiment, "evaluate_policy", lambda tasks, **kwargs: records_for(tasks))
    monkeypatch.setattr(
        experiment,
        "_generate",
        lambda trainer, tokenizer, tasks, **kwargs: (
            [f"query-{index}" for index in range(len(tasks))],
            [f"response-{index}" for index in range(len(tasks))],
            ["PATCH: False"] * len(tasks),
        ),
    )
    monkeypatch.setattr(
        experiment,
        "evaluate_response",
        lambda task, **kwargs: {
            "status": "executed",
            "reward": -0.5,
            "full_repair": False,
            "passed_cases": 0,
            "total_cases": len(task.cases),
            "unsafe": False,
        },
    )
    monkeypatch.setattr(experiment, "_save_adapter", lambda *args, **kwargs: {"files": []})
    monkeypatch.setenv("CODEX_HYDROGYM_CODING_PPO_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("CODE_SOURCE_PATH", str(experiment.Path(experiment.__file__).parents[2]))

    summary = experiment.run_experiment(
        {
            "ppo_updates": 1,
            "batch_size": 2,
            "mini_batch_size": 1,
            "enable_infrastructure_failure_masking": True,
        }
    )

    assert summary["training"]["updates_skipped_infrastructure"] == 0
    assert FakePPOTrainer.instance.calls == [
        (
            ["query-0", "query-1"],
            ["response-0", "response-1"],
            [("tensor-only", -0.5, "float32"), ("tensor-only", -0.5, "float32")],
        )
    ]
