"""Tamper-evident pin for the frozen control evaluation metric.

This tiny module holds the reviewed SHA-256 of
``codex_hydrogym/training/frozen_control_metric.py`` and the single check
function used at THREE enforcement points:

1. import time of the metric module itself (its last statement), so importing a
   shadowing module or a tampered installed wheel hard-fails immediately;
2. the training entrypoint ``codex_hydrogym/training/runner.py::run_training``,
   so a tampered tree hard-fails the loop before any training work;
3. ``test/codex_hydrogym/test_frozen_control_metric_guard.py``, which keeps its
   OWN independent copy of the digest, runs in CI via
   ``.github/workflows/pytest.yml``, and re-hashes the repo checkout file.

The check hashes the IMPORTED module's resolved ``__file__`` -- never a path
guessed from a test file's location -- so a shadowing ``frozen_control_metric.py``
earlier on ``sys.path`` and a divergent installed-wheel copy are caught.

Honest boundary: all pins live inside this repository.  An agent that edits the
metric, this pin, and the guard test in one commit still passes automation.
The guard converts tampering into a visible, reviewable, multi-file diff; it
does not make tampering impossible, and only human diff review catches the
same-commit case.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

METRIC_MODULE_NAME = "codex_hydrogym.training.frozen_control_metric"

# SHA-256 of the reviewed frozen_control_metric.py source (round 2 review).
EXPECTED_SOURCE_SHA256 = "3ef70d7f3fe99fc923d44e243ea81ec898a8f1d6fb3fec9b5e3ec4c4a911d9ce"


def imported_metric_path() -> Path:
    """Resolve the source path of the IMPORTED metric module (not a guessed path)."""
    module = sys.modules.get(METRIC_MODULE_NAME)
    source = getattr(module, "__file__", None) if module is not None else None
    if not source:
        import importlib

        source = importlib.import_module(METRIC_MODULE_NAME).__file__
    return Path(str(source)).resolve()


def assert_frozen_metric_source() -> Path:
    """Raise ``RuntimeError`` unless the imported metric source matches the pin."""
    path = imported_metric_path()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "frozen control metric source digest mismatch: the imported module is not the reviewed "
            "source (shadowing module, tampered installed wheel, or edited working tree?). "
            f"expected {EXPECTED_SOURCE_SHA256}, got {actual} at {path}"
        )
    return path
