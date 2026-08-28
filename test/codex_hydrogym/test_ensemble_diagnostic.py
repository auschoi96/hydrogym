"""Contracts for the fresh-seed, development-only convergence diagnostic."""

import json

import pytest

from codex_hydrogym.gate0.ensemble_diagnostic import (
    CLAIM_BOUNDARY,
    EnsembleDiagnosticSpec,
    analyze_conditions,
    main as diagnostic_main,
    study_gate_config,
)


def _condition_payload(spec, label, zero, feedback):
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
                        "numerical_gates": {"finite": True},
                        "windows": [
                            {"window_index": index, "mean_tke": values[label][index]}
                            for index in range(spec.scoring_windows)
                        ],
                    }
                )
    return {"traces": traces}


def test_spec_uses_only_fresh_development_seeds_and_preserves_old_limits():
    spec = EnsembleDiagnosticSpec()
    config = study_gate_config(spec)

    assert set(spec.seeds).isdisjoint({7, 101, 211, 307})
    assert set(spec.seeds).isdisjoint(spec.reserved_seeds)
    assert config.reynolds_number == 100.0
    assert config.precision == "float64"
    assert config.scored_intervals == 200
    assert config.maximum_spectral_tail_fraction == 0.05
    assert spec.conditions[1].arm_relative_difference_limit == 0.02
    assert spec.conditions[1].effect_difference_limit == 0.02
    assert spec.conditions[2].arm_relative_difference_limit == 0.05
    assert spec.conditions[2].effect_difference_limit == 0.03
    assert "No held-out gate" in CLAIM_BOUNDARY


def test_analysis_clusters_effects_by_seed_and_supports_a_stable_screen():
    spec = EnsembleDiagnosticSpec()
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
        label: _condition_payload(spec, label, zero, feedback)
        for label in ("base", "temporal", "spatial")
    }

    analysis = analyze_conditions(spec, payloads)

    assert analysis["condition_metrics"]["base"]["aggregate_relative_effect"] == 0.5
    assert set(
        analysis["condition_metrics"]["base"]["seed_cluster_relative_effects"]
    ) == set(spec.seeds)
    assert all(analysis["screening"].values())
    assert analysis["supports_designing_full_gate"] is True


def test_analysis_fails_closed_when_numerics_or_old_margins_fail():
    spec = EnsembleDiagnosticSpec()
    zero = {
        "base": (2.0, 2.0),
        "temporal": (1.7, 1.7),
        "spatial": (1.8, 1.8),
    }
    feedback = {
        "base": (1.0, 1.0),
        "temporal": (1.2, 1.2),
        "spatial": (1.2, 1.2),
    }
    payloads = {
        label: _condition_payload(spec, label, zero, feedback)
        for label in ("base", "temporal", "spatial")
    }
    payloads["spatial"]["traces"][0]["numerical_gates"]["finite"] = False

    analysis = analyze_conditions(spec, payloads)

    assert analysis["screening"]["numerical_validity"] is False
    assert analysis["screening"]["temporal_arm_point_convergence"] is False
    assert analysis["supports_designing_full_gate"] is False


def test_freeze_is_immutable_and_never_claims_a_gate_or_rl_run(tmp_path, capsys):
    output = tmp_path / "ensemble-diagnostic"

    assert diagnostic_main(["--stage", "freeze", "--output-dir", str(output)]) == 0
    assert diagnostic_main(["--stage", "freeze", "--output-dir", str(output)]) == 0

    payload = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    command_output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["status"] == "frozen_before_execution"
    assert payload["spec"]["claim_boundary"] == CLAIM_BOUNDARY
    assert command_output["rl_training_performed"] is False
    assert command_output["heldout_gate_performed"] is False
    assert command_output["supports_designing_full_gate"] is None


def test_spec_rejects_historical_or_underpowered_seed_substitution():
    with pytest.raises(ValueError, match="four distinct"):
        EnsembleDiagnosticSpec(seeds=(401, 503, 607))
    with pytest.raises(ValueError, match="fresh"):
        EnsembleDiagnosticSpec(seeds=(7, 401, 503, 607))
