"""Contracts for the prospectively fixed ten-seed replication study."""

from dataclasses import asdict
import hashlib
import json
import math
from statistics import fmean

import pytest

from codex_hydrogym.gate0 import ensemble_replication as replication
from codex_hydrogym.gate0.protocol import REQUIRED_NUMERICAL_GATES


def _sha256(*parts: object) -> str:
    material = ":".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _analysis_condition_payload(spec, label, zero, feedback):
    traces = []
    for seed_index, seed in enumerate(spec.seeds):
        for phase_index, phase in enumerate(spec.phase_turns):
            case = {
                "split": "development",
                "phase_index": phase_index,
                "phase_turns": phase,
                "seed_index": seed_index,
                "seed": seed,
            }
            for arm, values in (("zero", zero), ("signed_feedback", feedback)):
                traces.append(
                    {
                        "arm": arm,
                        "case": case,
                        "numerical_gates": dict.fromkeys(REQUIRED_NUMERICAL_GATES, True),
                        "windows": [
                            {"window_index": index, "mean_tke": values[label][index]}
                            for index in range(spec.scoring_windows)
                        ],
                    }
                )
    return {"traces": traces}


def _completed_condition_payload(spec, config, condition, implementation_digest):
    tke = {
        "base": {"zero": 2.0, "signed_feedback": 1.0},
        "temporal": {"zero": 2.01, "signed_feedback": 1.005},
        "spatial": {"zero": 2.02, "signed_feedback": 1.01},
    }
    effort = {"zero": 0.0, "signed_feedback": 0.2}
    traces = []
    for case in config.cases("development"):
        initial_digest = _sha256(condition.label, case.case_id, "initial")
        control_start_digest = _sha256(condition.label, case.case_id, "control-start")
        for arm in ("zero", "signed_feedback"):
            scored_start_digest = _sha256(condition.label, case.case_id, arm, "scored-start")
            expected_start_digest = scored_start_digest
            windows = []
            all_tke = []
            all_effort = []
            for index in range(spec.scoring_windows):
                end_digest = _sha256(condition.label, case.case_id, arm, "end", index)
                interval_tke = [tke[condition.label][arm]] * spec.intervals_per_window
                interval_effort = [effort[arm]] * spec.intervals_per_window
                windows.append(
                    {
                        "window_index": index,
                        "interval_count": spec.intervals_per_window,
                        "mean_tke": fmean(interval_tke),
                        "rms_l2_effort": math.sqrt(fmean(value**2 for value in interval_effort)),
                        "interval_mean_tke": interval_tke,
                        "interval_action_l2": interval_effort,
                        "start_state_digest": expected_start_digest,
                        "end_state_digest": end_digest,
                    }
                )
                expected_start_digest = end_digest
                all_tke.extend(interval_tke)
                all_effort.extend(interval_effort)
            traces.append(
                {
                    "condition": asdict(condition),
                    "arm": arm,
                    "case": asdict(case),
                    "initial_state_digest": initial_digest,
                    "control_start_digest": control_start_digest,
                    "scored_start_digest": scored_start_digest,
                    "mean_tke": fmean(all_tke),
                    "rms_l2_effort": math.sqrt(fmean(value**2 for value in all_effort)),
                    "numerical_gates": dict.fromkeys(REQUIRED_NUMERICAL_GATES, True),
                    "controller_input_history_digest": _sha256(condition.label, case.case_id, arm, "inputs"),
                    "action_history_digest": _sha256(condition.label, case.case_id, arm, "actions"),
                    "state_history_digest": _sha256(condition.label, case.case_id, arm, "states"),
                    "windows": windows,
                }
            )
    return replication._artifact(
        {
            "status": "completed",
            "study_fingerprint": spec.fingerprint,
            "implementation_digest": implementation_digest,
            "condition": asdict(condition),
            "case_ids": tuple(case.case_id for case in config.cases("development")),
            "traces": traces,
        }
    )


def test_spec_fixes_ten_hash_derived_seeds_and_preserves_sealed_cases():
    spec = replication.EnsembleReplicationSpec()

    assert spec.seeds == (
        1100085772,
        619716833,
        1680869979,
        270788329,
        1326527252,
        625393611,
        901546380,
        1422036434,
        373522063,
        1374108181,
    )
    assert spec.seeds == replication.derive_replication_seeds()
    assert set(spec.seeds).isdisjoint(
        {
            *replication.PRIOR_GATE_SEEDS,
            *replication.PRIOR_DIAGNOSTIC_SEEDS,
            *replication.RESERVED_SEEDS,
        }
    )
    assert spec.reserved_seeds == (907, 1009)
    assert spec.reserved_phase_turns == (0.1875, 0.6875)
    assert spec.expected_trajectory_count == 120
    assert spec.expected_paired_window_block_count == 120
    assert "results must never be pooled or appended" in spec.claim_boundary
    assert "stop early" in spec.sampling_plan


def test_spec_preserves_physics_horizons_and_old_margins():
    spec = replication.EnsembleReplicationSpec()
    config = replication.study_gate_config(spec)

    assert config.protocol_id == replication.STUDY_ID
    assert config.development_seeds == spec.seeds
    assert config.heldout_seeds == spec.reserved_seeds
    assert config.development_phase_turns == spec.phase_turns
    assert config.heldout_phase_turns == spec.reserved_phase_turns
    assert config.reynolds_number == 100.0
    assert config.precision == "float64"
    assert config.grid_size == (64, 64)
    assert config.dt == 0.002
    assert config.scored_intervals == 200
    assert config.maximum_spectral_tail_fraction == 0.05
    assert config.maximum_temporal_arm_tke_relative_difference == 0.02
    assert config.maximum_temporal_effect_difference == 0.02
    assert config.maximum_spatial_arm_tke_relative_difference == 0.05
    assert config.maximum_spatial_effect_difference == 0.03


def test_spec_rejects_seed_substitution_or_opening_reserved_cases():
    with pytest.raises(ValueError, match="exact ten hash-derived"):
        replication.EnsembleReplicationSpec(seeds=tuple(range(10)))
    with pytest.raises(ValueError, match="remain sealed"):
        replication.EnsembleReplicationSpec(reserved_seeds=(907, 1013))


def test_analysis_uses_ten_clusters_and_the_unchanged_screening_rule():
    spec = replication.EnsembleReplicationSpec()
    zero = {
        "base": (2.0, 2.0),
        "temporal": (2.01, 2.01),
        "spatial": (2.02, 2.02),
    }
    feedback = {
        "base": (1.0, 1.0),
        "temporal": (1.005, 1.005),
        "spatial": (1.01, 1.01),
    }
    payloads = {
        label: _analysis_condition_payload(spec, label, zero, feedback) for label in ("base", "temporal", "spatial")
    }

    analysis = replication.analyze_replication_conditions(spec, payloads)

    assert len(analysis["condition_metrics"]["base"]["seed_cluster_relative_effects"]) == 10
    assert all(analysis["screening"].values())
    assert analysis["supports_designing_full_gate"] is True


def test_run_requires_a_separately_frozen_protocol(tmp_path, capsys):
    output = tmp_path / "replication"

    with pytest.raises(RuntimeError, match="freeze and review"):
        replication.main(["--stage", "run", "--output-dir", str(output)])
    assert replication.main(["--stage", "freeze", "--output-dir", str(output)]) == 0
    assert replication.main(["--stage", "freeze", "--output-dir", str(output)]) == 0

    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    command_output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert protocol["status"] == "frozen_before_execution"
    assert protocol["execution_plan"]["expected_trajectory_count"] == 120
    assert protocol["execution_plan"]["interim_decision_looks_allowed"] is False
    assert protocol["predecessor_evidence"]["prior_observations_in_replication_analysis"] == 0
    assert protocol["implementation_files"][replication._CONFIG_SOURCE]
    assert protocol["implementation_files"]["codex_hydrogym/gate0/ensemble_diagnostic.py"]
    assert not list(output.glob("condition_*.json"))
    assert command_output["supports_designing_full_gate"] is None
    assert command_output["rl_training_performed"] is False
    assert command_output["heldout_gate_performed"] is False


def test_completed_condition_artifacts_resume_without_simulation(tmp_path, monkeypatch, capsys):
    output = tmp_path / "replication"
    spec = replication.EnsembleReplicationSpec()
    config = replication.study_gate_config(spec)
    _, implementation_digest = replication._study_implementation_manifest()
    assert replication.main(["--stage", "freeze", "--output-dir", str(output)]) == 0
    for condition in spec.conditions:
        replication._write_immutable(
            output / f"condition_{condition.label}.json",
            _completed_condition_payload(spec, config, condition, implementation_digest),
        )

    def _unexpected_simulation(*_args, **_kwargs):
        raise AssertionError("completed conditions must not be simulated again")

    monkeypatch.setattr(replication, "_run_condition", _unexpected_simulation)
    assert replication.main(["--stage", "run", "--output-dir", str(output)]) == 0
    assert replication.main(["--stage", "run", "--output-dir", str(output)]) == 0

    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    command_output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["fixed_seed_cluster_count"] == 10
    assert result["trajectory_count"] == 120
    assert result["prior_observations_in_analysis"] == 0
    assert result["analysis"]["supports_designing_full_gate"] is True
    assert command_output["supports_designing_full_gate"] is True


def test_condition_validation_rejects_unpaired_initial_states(tmp_path):
    spec = replication.EnsembleReplicationSpec()
    config = replication.study_gate_config(spec)
    _, implementation_digest = replication._study_implementation_manifest()
    condition = spec.conditions[0]
    payload = _completed_condition_payload(spec, config, condition, implementation_digest)
    payload["traces"][1]["initial_state_digest"] = _sha256("mismatched")
    body = {key: value for key, value in payload.items() if key != "artifact_digest"}
    path = tmp_path / "condition_base.json"
    replication._write_immutable(path, replication._artifact(body))

    with pytest.raises(RuntimeError, match="share the initial state"):
        replication._validated_replication_condition(
            path,
            spec,
            config,
            condition,
            implementation_digest,
        )
