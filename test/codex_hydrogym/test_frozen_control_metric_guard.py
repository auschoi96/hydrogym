"""Tamper-evident source pin for the frozen policy-quality metric.

Changing either this digest or the metric source is a visible protocol change
that must be reviewed independently of candidate reward experiments.
"""

import hashlib
from pathlib import Path

import pytest

EXPECTED_SOURCE_SHA256 = "0151021f67b19fa60034ebdff0a237751bd2774ceb3f52fadd2bea0378341c83"
SOURCE = Path(__file__).resolve().parents[2] / "codex_hydrogym/training/frozen_control_metric.py"


def _assert_frozen_source(path):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EXPECTED_SOURCE_SHA256:
        raise AssertionError(
            "frozen control metric source digest changed; this is a protocol change: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {actual}"
        )


def test_frozen_metric_source_matches_reviewed_sha256():
    _assert_frozen_source(SOURCE)


def test_guard_fails_when_metric_source_is_mutated(tmp_path):
    mutated = tmp_path / SOURCE.name
    mutated.write_bytes(SOURCE.read_bytes() + b"# candidate reward agent mutation\n")

    with pytest.raises(AssertionError, match="protocol change"):
        _assert_frozen_source(mutated)
