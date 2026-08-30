"""Tamper-evident source pin for the frozen policy-quality metric.

Three independent layers, all executed here (and in CI via
.github/workflows/pytest.yml, which did not run pytest at all before round 2):

1. the reviewed digest below is an independent copy of the pin module's
   constant, and this file fails if the two diverge;
2. ``assert_frozen_metric_source`` hashes the *imported* module's resolved
   path, so shadowing modules and tampered installed wheels are caught at
   import time and at the training entrypoint, not only in tests;
3. the repo checkout file itself is re-hashed here.

Changing the metric source, the pin, or this file is a visible protocol change
that must be reviewed independently of candidate reward experiments.  An agent
that edits all three in one commit still passes automation; only human diff
review catches that, which is why the guard is described as tamper-*evident*,
not tamper-*proof*.
"""

import ast
import hashlib
import sys
import types
from pathlib import Path

import pytest

from codex_hydrogym.training import frozen_metric_pin
from codex_hydrogym.training.frozen_control_metric import SeedOutcome  # noqa: F401  (import runs the guard)
from codex_hydrogym.training.frozen_metric_pin import assert_frozen_metric_source, imported_metric_path

# Independent copy of the reviewed digest.  Tampering must now edit the metric
# module, the pin module, AND this file in one commit -- three unrelated-looking
# diffs -- to keep automation green.
REVIEWED_SOURCE_SHA256 = "3ef70d7f3fe99fc923d44e243ea81ec898a8f1d6fb3fec9b5e3ec4c4a911d9ce"
SOURCE = Path(__file__).resolve().parents[2] / "codex_hydrogym/training/frozen_control_metric.py"
RUNNER = Path(__file__).resolve().parents[2] / "codex_hydrogym/training/runner.py"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_frozen_source(path):
    actual = _sha256(path)
    if actual != REVIEWED_SOURCE_SHA256:
        raise AssertionError(
            "frozen control metric source digest changed; this is a protocol change: "
            f"expected {REVIEWED_SOURCE_SHA256}, got {actual}"
        )


def test_frozen_metric_source_matches_reviewed_sha256():
    _assert_frozen_source(SOURCE)


def test_guard_fails_when_metric_source_is_mutated(tmp_path):
    mutated = tmp_path / SOURCE.name
    mutated.write_bytes(SOURCE.read_bytes() + b"# candidate reward agent mutation\n")

    with pytest.raises(AssertionError, match="protocol change"):
        _assert_frozen_source(mutated)


def test_pin_digest_matches_independent_reviewed_copy():
    assert frozen_metric_pin.EXPECTED_SOURCE_SHA256 == REVIEWED_SOURCE_SHA256


def test_import_check_hashes_the_imported_module_path():
    import codex_hydrogym.training.frozen_control_metric as metric_module

    assert imported_metric_path() == Path(metric_module.__file__).resolve()
    # In this checkout the imported module IS the reviewed file.
    assert _sha256(imported_metric_path()) == REVIEWED_SOURCE_SHA256


def test_import_check_rejects_shadowing_module(tmp_path):
    shadow = tmp_path / "frozen_control_metric.py"
    shadow.write_bytes(SOURCE.read_bytes() + b"\n# shadowing tamper\n")
    fake = types.ModuleType(frozen_metric_pin.METRIC_MODULE_NAME)
    fake.__file__ = str(shadow)
    sentinel = sys.modules.get(frozen_metric_pin.METRIC_MODULE_NAME)
    sys.modules[frozen_metric_pin.METRIC_MODULE_NAME] = fake
    try:
        with pytest.raises(RuntimeError, match="digest mismatch"):
            assert_frozen_metric_source()
    finally:
        if sentinel is None:
            sys.modules.pop(frozen_metric_pin.METRIC_MODULE_NAME, None)
        else:
            sys.modules[frozen_metric_pin.METRIC_MODULE_NAME] = sentinel


def test_training_entrypoint_calls_the_frozen_source_guard_first():
    """Structural check that survives the runner's heavy jax/flax import tree."""
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assert_frozen_metric_source"
    ]
    assert calls, "run_training must call assert_frozen_metric_source"
    run_training = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_training")
    statements = run_training.body
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        statements = statements[1:]  # skip the docstring
    first_statement = statements[0]
    assert isinstance(first_statement, ast.Expr)
    assert isinstance(first_statement.value, ast.Call)
    assert getattr(first_statement.value.func, "id", None) == "assert_frozen_metric_source", (
        "the guard must run before any other training work in run_training"
    )


def test_entrypoint_guard_aborts_before_any_training_setup(tmp_path, monkeypatch):
    pytest.importorskip("jax", reason="runner imports the full jax/flax training stack")
    from codex_hydrogym.training import runner

    def synthetic_tamper():
        raise RuntimeError("synthetic frozen-source tamper")

    monkeypatch.setattr(runner, "assert_frozen_metric_source", synthetic_tamper)
    output = tmp_path / "codex_hydrogym_output"
    with pytest.raises(RuntimeError, match="synthetic frozen-source tamper"):
        runner.run_training(runner.smoke_config(), output)
    assert not output.exists(), "no training setup may happen after a failed frozen-source check"


def test_imported_metric_module_exposes_the_pinned_seed_contract():
    # Guards against accidental edits to the frozen constants consumed by compare_candidates.
    from codex_hydrogym.training.frozen_control_metric import (
        DIAGNOSTIC_DEVELOPMENT_SEEDS,
        FROZEN_SEED_COUNT,
        FROZEN_T_CRITICAL_95,
        HISTORICAL_GATE_SEEDS,
    )

    assert FROZEN_SEED_COUNT == 4
    assert FROZEN_T_CRITICAL_95 == 3.182446305284263  # df=3, four clusters (settled fact)
    assert DIAGNOSTIC_DEVELOPMENT_SEEDS == (401, 503, 607, 709)
    assert HISTORICAL_GATE_SEEDS == frozenset({7, 101, 211, 307})
    assert SeedOutcome.__dataclass_params__.frozen
