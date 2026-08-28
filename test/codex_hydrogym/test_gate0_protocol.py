"""Deterministic contracts for the pre-RL Gate 0 evaluator."""

from dataclasses import asdict, replace
import hashlib
import json
import math

import pytest

from codex_hydrogym.gate0 import cli as gate0_cli_module
from codex_hydrogym.gate0.cli import (
    _load_lock,
    _report_payload,
    _validated_artifact_digest,
    main as gate0_cli_main,
)
from codex_hydrogym.gate0.protocol import (
    REQUIRED_CONVERGENCE_GATES,
    REQUIRED_PRIMARY_GATES,
    EpisodeStep,
    FrozenSignedController,
    Gate0ConvergenceAttestation,
    Gate0Config,
    Gate0DevelopmentSearchError,
    Gate0RefinementProvenance,
    lock_development_controls,
    oracle_action,
    run_gate0,
    run_gate0_convergence,
)
from codex_hydrogym.gate0.re100_v2 import (
    RE100_V2_PROTOCOL_ID,
    _v2_implementation_manifest,
    main as re100_v2_main,
    re100_v2_config,
)


class _ToyEpisode:
    """Cheap deterministic plant with the same signed quadrature semantics."""

    def __init__(self, case):
        self.case = case
        self.uncontrolled_reset_prelude_intervals = 0
        self.observation = (math.cos(case.phase_radians), math.sin(case.phase_radians))
        self._actions = []
        self.state_digest = self._digest()
        self._tkes = []

    def _digest(self):
        payload = {"case": self.case.case_id, "actions": self._actions}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def advance(self, action):
        self._actions.append(action)
        residual_x = 0.5 * self.observation[0] + action[0]
        residual_y = 0.5 * self.observation[1] + action[1]
        tke = 1.0 + residual_x**2 + residual_y**2
        self._tkes.append(tke)
        self.state_digest = self._digest()
        return EpisodeStep(self.observation, tke, self.state_digest)

    def numerical_gates(self):
        return {
            "finite_state_and_metrics": True,
            "nonnegative_tke": True,
            "reward_tke_identity": True,
            "zero_mean_vorticity": True,
            "incompressible_velocity": True,
            "spectral_tail_controlled": True,
            "cfl_controlled": True,
        }


class _RecordingFactory:
    def __init__(self):
        self.calls = []

    def __call__(self, case):
        self.calls.append(case)
        return _ToyEpisode(case)


class _ScaledToyEpisode(_ToyEpisode):
    def __init__(self, case, scale):
        super().__init__(case)
        self.scale = scale

    def advance(self, action):
        step = super().advance(action)
        return EpisodeStep(step.observation, self.scale * step.mean_tke, step.state_digest)


class _ScaledRecordingFactory(_RecordingFactory):
    def __init__(self, scale=1.0, *, execution_grid_size=None, execution_dt=None):
        super().__init__()
        self.scale = scale
        self.execution_grid_size = execution_grid_size
        self.execution_dt = execution_dt

    def __call__(self, case):
        self.calls.append(case)
        return _ScaledToyEpisode(case, self.scale)


class _InvalidNumericalToyEpisode(_ToyEpisode):
    def numerical_gates(self):
        gates = super().numerical_gates()
        gates["cfl_controlled"] = False
        return gates


class _ToyHydroGymFactory(_RecordingFactory):
    def __init__(self, config, **execution_overrides):
        super().__init__()
        self.execution_grid_size = execution_overrides.get("execution_grid_size", config.grid_size)
        self.execution_dt = execution_overrides.get("execution_dt", config.dt)


class _InvalidToyHydroGymFactory(_ToyHydroGymFactory):
    def __call__(self, case):
        self.calls.append(case)
        return _InvalidNumericalToyEpisode(case)


class _SourceInvalidRefinementFactory(_ScaledRecordingFactory):
    def __init__(self, convergence_seed, **kwargs):
        super().__init__(**kwargs)
        self.convergence_seed = convergence_seed

    def __call__(self, case):
        self.calls.append(case)
        if case.seed != self.convergence_seed:
            return _InvalidNumericalToyEpisode(case)
        return _ScaledToyEpisode(case, self.scale)


class _ExplodingToyHydroGymFactory(_ToyHydroGymFactory):
    def __call__(self, case):
        raise RuntimeError(f"synthetic execution failure for {case.case_id}")


class _NonconvergentToyHydroGymFactory(_ToyHydroGymFactory):
    def __init__(self, config, **execution_overrides):
        super().__init__(config, **execution_overrides)
        self.scale = 1.1 if self.execution_dt == config.temporal_refinement_dt else 1.0

    def __call__(self, case):
        self.calls.append(case)
        return _ScaledToyEpisode(case, self.scale)


def _config():
    return replace(
        Gate0Config(),
        precision="float32",
        grid_size=(16, 16),
        action_time=0.1,
        save_time=0.05,
        uncontrolled_burn_in_intervals=1,
        controller_warmup_intervals=1,
        scored_intervals=3,
        heldout_seeds=(101, 211),
        feedback_gain_candidates=(0.25, 0.5, 1.0),
        minimum_relative_tke_reduction=0.01,
        minimum_absolute_tke_reduction=0.001,
        minimum_seed_win_fraction=0.5,
    )


def _attestation(report, config, *, gates=None, metrics=None):
    case_ids = tuple(
        case.case_id for case in config.cases("heldout") if case.seed == config.convergence_seed
    )
    return Gate0ConvergenceAttestation(
        protocol_fingerprint=report.protocol_fingerprint,
        development_lock_digest=report.development_lock_digest,
        primary_report_digest=hashlib.sha256(b"frozen-primary-report").hexdigest(),
        convergence_seed=config.convergence_seed,
        temporal_provenance=Gate0RefinementProvenance(
            label="temporal_refinement",
            grid_size=config.grid_size,
            dt=config.temporal_refinement_dt,
            case_ids=case_ids,
        ),
        spatial_provenance=Gate0RefinementProvenance(
            label="spatial_refinement",
            grid_size=config.spatial_refinement_grid_size,
            dt=config.dt,
            case_ids=case_ids,
        ),
        refinement_evidence_digests={"temporal": "1" * 64, "spatial": "2" * 64},
        gates={name: True for name in REQUIRED_CONVERGENCE_GATES} if gates is None else gates,
        metrics={"maximum_effect_difference": 0.01} if metrics is None else metrics,
    )


def test_oracle_uses_both_quadratures_with_rotation_invariant_authority():
    config = _config()
    diagonal = config.cases("heldout")[0]

    action = oracle_action(diagonal, config.radial_action_bound)

    assert action[0] != 0.0 and action[1] != 0.0
    assert action[2:] == (0.0, 0.0)
    assert math.hypot(action[0], action[1]) == pytest.approx(config.radial_action_bound)


def test_signed_controller_is_pure_radially_clipped_and_forced_mode_only():
    controller = FrozenSignedController(gain=10.0, radial_bound=0.5)

    action = controller((3.0, 4.0))

    assert action[2:] == (0.0, 0.0)
    assert math.hypot(action[0], action[1]) == pytest.approx(0.5)
    with pytest.raises(TypeError):
        controller((1.0, 0.0), phase=0.0)


def test_development_lock_never_opens_a_heldout_case():
    config = _config()
    factory = _RecordingFactory()

    lock = lock_development_controls(config, factory)

    assert {case.split for case in factory.calls} == {"development"}
    assert lock.protocol_fingerprint == config.fingerprint
    assert set(lock.development_case_ids) == {case.case_id for case in config.cases("development")}
    assert lock.fixed_action in config.constant_candidates
    assert lock.feedback_gain in config.feedback_gain_candidates


def test_heldout_gate_preserves_global_observation_action_marginals_and_effort():
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())
    report = run_gate0(config, lock, _RecordingFactory())

    assert report.gates["matched_initial_states"] is True
    assert report.gates["phase_and_seed_derangement"] is True
    assert report.gates["source_observation_trajectories_differ"] is True
    assert report.gates["observation_marginal_exact"] is True
    assert report.gates["action_marginal_exact"] is True
    assert report.gates["rotation_invariant_effort_exact"] is True
    assert report.gates["oracle_materially_beats_zero"] is True
    assert report.gates["feedback_materially_beats_deranged"] is True
    assert report.primary_passed is True
    assert report.passed is False
    assert report.convergence_attestation_digest is None
    assert report.rl_training_performed is False
    assert "per-reset phase" in report.claim_boundary

    shuffled = [trace for trace in report.traces if trace.arm == "observation_deranged"]
    by_id = {case.case_id: case for case in config.cases("heldout")}
    for trace in shuffled:
        source = by_id[trace.source_case_id]
        assert source.phase_index != trace.case.phase_index
        assert source.seed_index != trace.case.seed_index


@pytest.mark.parametrize(
    "gates",
    [
        pytest.param(None, id="none"),
        pytest.param({}, id="empty"),
        pytest.param(
            {name: True for name in REQUIRED_PRIMARY_GATES if name != "numerical_validity"},
            id="missing",
        ),
        pytest.param(
            {**{name: True for name in REQUIRED_PRIMARY_GATES}, "unregistered": True},
            id="extra",
        ),
        pytest.param(
            {**{name: True for name in REQUIRED_PRIMARY_GATES}, "numerical_validity": False},
            id="false",
        ),
        pytest.param(
            {**{name: True for name in REQUIRED_PRIMARY_GATES}, "numerical_validity": 1},
            id="non-boolean",
        ),
    ],
)
def test_primary_gate_fails_closed_for_nonexact_gate_mappings(gates):
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())
    report = run_gate0(config, lock, _RecordingFactory())

    candidate = replace(report, gates=gates)

    assert candidate.primary_passed is False
    assert candidate.passed is False


def test_primary_report_defensively_copies_gate_mappings():
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())
    report = run_gate0(config, lock, _RecordingFactory())
    source = dict(report.gates)

    copied = replace(report, gates=source)
    source["numerical_validity"] = False

    assert copied.gates["numerical_validity"] is True
    assert copied.primary_passed is True


def test_matching_convergence_attestation_completes_an_otherwise_passing_report():
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())
    report = run_gate0(config, lock, _RecordingFactory())
    attestation = _attestation(report, config)

    completed = report.with_convergence(
        attestation,
        primary_report_digest=attestation.primary_report_digest,
    )

    assert report.primary_passed is True
    assert report.passed is False
    assert completed.convergence_attestation_digest == attestation.digest
    assert completed.convergence_gates == attestation.gates
    assert completed.passed is True


def test_actual_convergence_runner_scores_both_refinements_and_binds_primary_artifact():
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())
    report = run_gate0(config, lock, _RecordingFactory())
    primary_report_digest = _report_payload(report)["artifact_digest"]
    factories = []

    def build_factory(_grid_size, _dt):
        factory = _ScaledRecordingFactory(
            execution_grid_size=_grid_size,
            execution_dt=_dt,
        )
        factories.append(factory)
        return factory

    convergence_run = run_gate0_convergence(
        config,
        lock,
        report,
        primary_report_digest,
        build_factory,
    )
    attestation = convergence_run.attestation
    completed = report.with_convergence(
        attestation,
        primary_report_digest=primary_report_digest,
    )

    assert set(attestation.gates) == REQUIRED_CONVERGENCE_GATES
    assert all(attestation.gates.values())
    assert attestation.primary_report_digest == primary_report_digest
    assert attestation.temporal_provenance.grid_size == config.grid_size
    assert attestation.temporal_provenance.dt == config.temporal_refinement_dt
    assert attestation.spatial_provenance.grid_size == config.spatial_refinement_grid_size
    assert attestation.spatial_provenance.dt == config.dt
    assert any(
        case_id.endswith("_211") for case_id in attestation.temporal_provenance.case_ids
    )
    assert len(factories) == 2
    assert convergence_run.temporal_evidence.digest == (
        attestation.refinement_evidence_digests["temporal"]
    )
    assert completed.passed is True


def test_convergence_runner_records_failed_refinement_gates_without_claiming_success():
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())
    report = run_gate0(config, lock, _RecordingFactory())
    primary_report_digest = _report_payload(report)["artifact_digest"]

    def build_factory(grid_size, dt):
        if grid_size == config.grid_size and dt == config.temporal_refinement_dt:
            return _ScaledRecordingFactory(
                scale=1.1,
                execution_grid_size=grid_size,
                execution_dt=dt,
            )
        return _ScaledRecordingFactory(
            execution_grid_size=grid_size,
            execution_dt=dt,
        )

    convergence_run = run_gate0_convergence(
        config,
        lock,
        report,
        primary_report_digest,
        build_factory,
    )
    attestation = convergence_run.attestation
    completed = report.with_convergence(
        attestation,
        primary_report_digest=primary_report_digest,
    )

    assert attestation.gates["temporal_arm_tke_convergence"] is False
    assert attestation.metrics["temporal_maximum_arm_tke_relative_difference"] > (
        config.maximum_temporal_arm_tke_relative_difference
    )
    assert attestation.gates["spatial_arm_tke_convergence"] is True
    assert completed.passed is False


def test_convergence_runner_counts_derangement_source_numerical_failures():
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())
    report = run_gate0(config, lock, _RecordingFactory())
    primary_report_digest = _report_payload(report)["artifact_digest"]

    convergence_run = run_gate0_convergence(
        config,
        lock,
        report,
        primary_report_digest,
        lambda grid_size, dt: _SourceInvalidRefinementFactory(
            config.convergence_seed,
            execution_grid_size=grid_size,
            execution_dt=dt,
        ),
    )

    assert convergence_run.attestation.gates["temporal_numerical_validity"] is False
    assert convergence_run.attestation.gates["spatial_numerical_validity"] is False
    assert any(
        summary.case_id.endswith("_211")
        and summary.numerical_gates["cfl_controlled"] is False
        for summary in convergence_run.temporal_evidence.traces
    )


def test_convergence_runner_rejects_factory_resolution_claim_mismatch_before_execution():
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())
    report = run_gate0(config, lock, _RecordingFactory())
    primary_report_digest = _report_payload(report)["artifact_digest"]
    factory = _ScaledRecordingFactory(
        execution_grid_size=config.grid_size,
        execution_dt=config.dt,
    )

    with pytest.raises(ValueError, match="requested dt"):
        run_gate0_convergence(
            config,
            lock,
            report,
            primary_report_digest,
            lambda _grid_size, _dt: factory,
        )

    assert factory.calls == []


def test_convergence_runner_rejects_unpassed_or_mismatched_primary_evidence_before_factories_run():
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())
    report = run_gate0(config, lock, _RecordingFactory())
    primary_report_digest = _report_payload(report)["artifact_digest"]
    factory_calls = []

    def build_factory(grid_size, dt):
        factory_calls.append((grid_size, dt))
        return _RecordingFactory()

    failed_report = replace(
        report,
        gates={**report.gates, "numerical_validity": False},
    )
    with pytest.raises(ValueError, match="must pass"):
        run_gate0_convergence(
            config,
            lock,
            failed_report,
            primary_report_digest,
            build_factory,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        run_gate0_convergence(config, lock, report, "not-a-digest", build_factory)

    assert factory_calls == []


def test_report_requires_the_exact_primary_artifact_digest_when_attaching_convergence():
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())
    report = run_gate0(config, lock, _RecordingFactory())
    attestation = _attestation(report, config)

    with pytest.raises(ValueError, match="artifact digest"):
        report.with_convergence(attestation, primary_report_digest="0" * 64)


@pytest.mark.parametrize(
    ("digest", "gates"),
    [
        pytest.param(None, {name: True for name in REQUIRED_CONVERGENCE_GATES}, id="no-digest"),
        pytest.param("", {name: True for name in REQUIRED_CONVERGENCE_GATES}, id="empty-digest"),
        pytest.param("0" * 64, None, id="no-gates"),
        pytest.param("0" * 64, {}, id="empty-gates"),
        pytest.param(
            "0" * 64,
            {
                name: True
                for name in REQUIRED_CONVERGENCE_GATES
                if name != "temporal_numerical_validity"
            },
            id="missing-gate",
        ),
        pytest.param(
            "0" * 64,
            {**{name: True for name in REQUIRED_CONVERGENCE_GATES}, "unregistered_gate": True},
            id="extra-gate",
        ),
        pytest.param(
            "0" * 64,
            {
                **{name: True for name in REQUIRED_CONVERGENCE_GATES},
                "spatial_effect_convergence": False,
            },
            id="false-gate",
        ),
    ],
)
def test_final_gate_fails_closed_for_incomplete_convergence_evidence(digest, gates):
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())
    report = run_gate0(config, lock, _RecordingFactory())

    candidate = replace(
        report,
        convergence_attestation_digest=digest,
        convergence_gates=gates,
    )

    assert candidate.primary_passed is True
    assert candidate.passed is False


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("protocol_fingerprint", "protocol fingerprint"),
        ("development_lock_digest", "development lock"),
    ],
)
def test_report_rejects_convergence_attestation_for_different_evidence(field, message):
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())
    report = run_gate0(config, lock, _RecordingFactory())
    mismatched = replace(_attestation(report, config), **{field: "0" * 64})

    with pytest.raises(ValueError, match=message):
        report.with_convergence(
            mismatched,
            primary_report_digest=mismatched.primary_report_digest,
        )


def test_convergence_digest_is_mapping_order_independent():
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())
    report = run_gate0(config, lock, _RecordingFactory())
    attestation = _attestation(
        report,
        config,
        metrics={"temporal_effect_difference": 0.01, "spatial_effect_difference": 0.02},
    )
    reordered = replace(
        attestation,
        gates=dict(reversed(tuple(attestation.gates.items()))),
        metrics=dict(reversed(tuple(attestation.metrics.items()))),
    )

    assert reordered.as_dict() == attestation.as_dict()
    assert reordered.digest == attestation.digest


def test_convergence_attestation_defensively_copies_source_mappings():
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())
    report = run_gate0(config, lock, _RecordingFactory())
    gates = {name: True for name in REQUIRED_CONVERGENCE_GATES}
    metrics = {"effect_difference": 0.01}
    attestation = _attestation(report, config, gates=gates, metrics=metrics)
    original_digest = attestation.digest

    gates["temporal_numerical_validity"] = False
    metrics["effect_difference"] = math.inf

    assert attestation.gates["temporal_numerical_validity"] is True
    assert attestation.metrics["effect_difference"] == 0.01
    assert attestation.digest == original_digest
    with pytest.raises(TypeError):
        attestation.gates["temporal_numerical_validity"] = False


def test_convergence_attestation_rejects_invalid_digest_duplicate_cases_and_nonfinite_metrics():
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())
    report = run_gate0(config, lock, _RecordingFactory())
    attestation = _attestation(report, config)

    with pytest.raises(ValueError, match="SHA-256"):
        replace(attestation, primary_report_digest="not-a-digest")
    with pytest.raises(ValueError, match="duplicates"):
        replace(
            attestation.temporal_provenance,
            case_ids=(
                attestation.temporal_provenance.case_ids[0],
                attestation.temporal_provenance.case_ids[0],
            ),
        )
    with pytest.raises(ValueError, match="finite"):
        replace(attestation, metrics={"effect_difference": math.nan})


def test_protocol_fails_closed_on_lock_mismatch_or_unversioned_sensor():
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())

    with pytest.raises(ValueError, match="development lock"):
        run_gate0(replace(config, scored_intervals=4), lock, _RecordingFactory())
    with pytest.raises(ValueError, match="signed forced-mode"):
        replace(config, observation_contract_version="magnitude_grid")


def test_default_protocol_freezes_developed_horizons_and_refinements():
    config = Gate0Config()

    assert config.uncontrolled_burn_in_intervals * config.action_time == 100.0
    assert config.controller_warmup_intervals * config.action_time == 50.0
    assert config.scored_intervals * config.action_time == 100.0
    assert config.temporal_refinement_dt == 0.001
    assert config.spatial_refinement_grid_size == (64, 64)


def test_re100_v2_changes_only_the_explicitly_approved_scientific_claim():
    original = Gate0Config().as_dict()
    re100 = re100_v2_config().as_dict()
    intended_changes = {
        "protocol_id",
        "reynolds_number",
        "grid_size",
        "spatial_refinement_grid_size",
    }

    assert {key for key in original if original[key] != re100[key]} == intended_changes
    assert re100["protocol_id"] == RE100_V2_PROTOCOL_ID
    assert re100["reynolds_number"] == 100.0
    assert re100["grid_size"] == (64, 64)
    assert re100["spatial_refinement_grid_size"] == (96, 96)
    assert re100["maximum_spectral_tail_fraction"] == 0.05
    assert re100["temporal_refinement_dt"] == 0.001


def test_re100_v2_freeze_is_immutable_and_binds_its_config_source(tmp_path):
    output = tmp_path / "gate0-re100-v2"

    assert re100_v2_main(["--stage", "freeze", "--output-dir", str(output)]) == 0
    assert re100_v2_main(["--stage", "freeze", "--output-dir", str(output)]) == 0

    payload = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    implementation_files, implementation_digest = _v2_implementation_manifest()
    config = re100_v2_config()
    assert payload["protocol_fingerprint"] == config.fingerprint
    assert payload["implementation_digest"] == implementation_digest
    assert payload["implementation_files"] == implementation_files
    assert "codex_hydrogym/gate0/re100_v2.py" in implementation_files
    assert payload["config"]["protocol_id"] == RE100_V2_PROTOCOL_ID
    assert payload["config"]["reynolds_number"] == 100.0
    assert payload["config"]["grid_size"] == [64, 64]
    assert payload["config"]["spatial_refinement_grid_size"] == [96, 96]
    assert payload["config"]["maximum_spectral_tail_fraction"] == 0.05
    assert payload["config"]["temporal_refinement_dt"] == 0.001


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"grid_size": (16, 18)}, "square grid"),
        ({"action_time": 0.11}, "action_time/save_time"),
        ({"save_time": 0.025}, "save_time/dt"),
        ({"convergence_seed": 999}, "convergence_seed"),
        (
            {"heldout_seeds": (-1, 211, 307), "convergence_seed": -1},
            "nonnegative",
        ),
        ({"spatial_refinement_grid_size": (48, 48)}, "finer"),
    ],
)
def test_protocol_rejects_ambiguous_numerics(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(Gate0Config(), **changes)


def test_freeze_cli_writes_one_immutable_nonclaiming_protocol(tmp_path):
    output = tmp_path / "gate0"

    assert gate0_cli_main(["--stage", "freeze", "--output-dir", str(output)]) == 0
    assert gate0_cli_main(["--stage", "freeze", "--output-dir", str(output)]) == 0

    payload = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert payload["status"] == "frozen_before_execution"
    assert payload["protocol_fingerprint"] == Gate0Config().fingerprint
    assert "proves no learned improvement" in payload["claim_boundary"]


def test_convergence_stage_requires_existing_primary_evidence(tmp_path, monkeypatch):
    output = tmp_path / "gate0"
    monkeypatch.setattr(gate0_cli_module, "Gate0Config", _config)

    with pytest.raises(RuntimeError, match="existing primary-report"):
        gate0_cli_main(["--stage", "convergence", "--output-dir", str(output)])

    assert not (output / "development_lock.json").exists()
    assert not (output / "primary_report.json").exists()


def test_development_lock_artifact_round_trips_with_search_provenance(tmp_path):
    config = _config()
    lock = lock_development_controls(config, _RecordingFactory())
    path = tmp_path / "development_lock.json"
    path.write_text(
        json.dumps({"lock": asdict(lock), "lock_digest": lock.digest}),
        encoding="utf-8",
    )

    loaded = _load_lock(path, config)

    assert loaded == lock
    assert loaded.search_scores
    assert loaded.digest == lock.digest


def test_convergence_cli_persists_exact_cross_artifact_digest_binding(tmp_path, monkeypatch):
    config = _config()
    output = tmp_path / "gate0"
    monkeypatch.setattr(gate0_cli_module, "Gate0Config", lambda: config)
    monkeypatch.setattr(gate0_cli_module, "HydroGymEpisodeFactory", _ToyHydroGymFactory)

    assert gate0_cli_main(["--stage", "primary", "--output-dir", str(output)]) == 0

    def unexpected_primary_rerun(*_args, **_kwargs):
        raise AssertionError("primary evidence must not be rerun")

    monkeypatch.setattr(gate0_cli_module, "run_gate0", unexpected_primary_rerun)
    assert gate0_cli_main(["--stage", "convergence", "--output-dir", str(output)]) == 0

    primary = json.loads((output / "primary_report.json").read_text(encoding="utf-8"))
    convergence = json.loads(
        (output / "convergence_attestation.json").read_text(encoding="utf-8")
    )
    final = json.loads((output / "final_report.json").read_text(encoding="utf-8"))

    assert _validated_artifact_digest(primary) == primary["artifact_digest"]
    assert _validated_artifact_digest(convergence) == convergence["artifact_digest"]
    assert convergence["attestation"]["primary_report_digest"] == primary["artifact_digest"]
    assert convergence["attestation_digest"] == final["convergence_attestation_digest"]
    assert final["final_gate0_passed"] is True
    for label in ("temporal", "spatial"):
        evidence = convergence["refinement_evidence"][label]
        assert evidence["evidence_digest"] == convergence["attestation"][
            "refinement_evidence_digests"
        ][label]
        assert evidence["traces"]

    def unexpected_convergence_rerun(*_args, **_kwargs):
        raise AssertionError("convergence evidence must not be rerun")

    monkeypatch.setattr(
        gate0_cli_module,
        "run_gate0_convergence",
        unexpected_convergence_rerun,
    )
    monkeypatch.setattr(gate0_cli_module, "HydroGymEpisodeFactory", unexpected_convergence_rerun)
    assert gate0_cli_main(["--stage", "convergence", "--output-dir", str(output)]) == 0

    tampered = dict(primary)
    tampered["primary_passed"] = False
    with pytest.raises(RuntimeError, match="digest validation"):
        _validated_artifact_digest(tampered)


def test_convergence_cli_rejects_tampered_primary_before_any_rerun(tmp_path, monkeypatch):
    config = _config()
    output = tmp_path / "gate0"
    monkeypatch.setattr(gate0_cli_module, "Gate0Config", lambda: config)
    monkeypatch.setattr(gate0_cli_module, "HydroGymEpisodeFactory", _ToyHydroGymFactory)
    assert gate0_cli_main(["--stage", "primary", "--output-dir", str(output)]) == 0

    primary_path = output / "primary_report.json"
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    primary["primary_passed"] = False
    primary_path.write_text(json.dumps(primary), encoding="utf-8")

    def unexpected_execution(*_args, **_kwargs):
        raise AssertionError("tampered evidence must fail before simulation")

    monkeypatch.setattr(gate0_cli_module, "run_gate0", unexpected_execution)
    monkeypatch.setattr(gate0_cli_module, "run_gate0_convergence", unexpected_execution)
    monkeypatch.setattr(gate0_cli_module, "HydroGymEpisodeFactory", unexpected_execution)
    with pytest.raises(RuntimeError, match="artifact digest"):
        gate0_cli_main(["--stage", "convergence", "--output-dir", str(output)])


def test_failed_convergence_cli_persists_auditable_evidence_and_final_failure(tmp_path, monkeypatch):
    config = _config()
    output = tmp_path / "gate0"
    monkeypatch.setattr(gate0_cli_module, "Gate0Config", lambda: config)
    monkeypatch.setattr(
        gate0_cli_module,
        "HydroGymEpisodeFactory",
        _NonconvergentToyHydroGymFactory,
    )

    assert gate0_cli_main(["--stage", "all", "--output-dir", str(output)]) == 2

    convergence = json.loads(
        (output / "convergence_attestation.json").read_text(encoding="utf-8")
    )
    final = json.loads((output / "final_report.json").read_text(encoding="utf-8"))
    assert convergence["attestation"]["gates"]["temporal_arm_tke_convergence"] is False
    assert convergence["refinement_evidence"]["temporal"]["traces"]
    assert final["final_gate0_passed"] is False


def test_failed_development_search_persists_scores_without_writing_a_lock(tmp_path, monkeypatch):
    config = _config()
    output = tmp_path / "gate0"

    with pytest.raises(Gate0DevelopmentSearchError) as captured:
        lock_development_controls(config, _InvalidToyHydroGymFactory(config))
    assert captured.value.search_scores
    assert all(not score.numerical_valid for score in captured.value.search_scores)
    assert captured.value.failures
    assert all(
        failure.failed_numerical_gates == ("cfl_controlled",)
        for failure in captured.value.failures
    )

    monkeypatch.setattr(gate0_cli_module, "Gate0Config", lambda: config)
    monkeypatch.setattr(
        gate0_cli_module,
        "HydroGymEpisodeFactory",
        _InvalidToyHydroGymFactory,
    )

    assert gate0_cli_main(["--stage", "lock", "--output-dir", str(output)]) == 2

    failure_path = output / "development_search_failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert _validated_artifact_digest(failure) == failure["artifact_digest"]
    assert failure["reason"] == "development_candidate_numerical_gate_failure"
    assert len(failure["search_scores"]) == len(captured.value.search_scores)
    assert all(not score["numerical_valid"] for score in failure["search_scores"])
    assert all(item["failed_numerical_gates"] == ["cfl_controlled"] for item in failure["failures"])
    assert not (output / "development_lock.json").exists()

    def unexpected_search_rerun(*_args, **_kwargs):
        raise AssertionError("durable failed search must not be rerun")

    monkeypatch.setattr(gate0_cli_module, "HydroGymEpisodeFactory", unexpected_search_rerun)
    assert gate0_cli_main(["--stage", "lock", "--output-dir", str(output)]) == 2

    (output / "development_lock.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="contradictory"):
        gate0_cli_main(["--stage", "lock", "--output-dir", str(output)])


def test_development_execution_error_persists_partial_diagnostics(tmp_path, monkeypatch):
    config = _config()
    output = tmp_path / "gate0"
    monkeypatch.setattr(gate0_cli_module, "Gate0Config", lambda: config)
    monkeypatch.setattr(
        gate0_cli_module,
        "HydroGymEpisodeFactory",
        _ExplodingToyHydroGymFactory,
    )

    assert gate0_cli_main(["--stage", "lock", "--output-dir", str(output)]) == 2

    failure = json.loads(
        (output / "development_search_failure.json").read_text(encoding="utf-8")
    )
    assert failure["reason"] == "development_candidate_execution_error"
    assert failure["search_scores"] == []
    assert failure["failures"][0]["case_id"].startswith("development_")
    assert failure["failures"][0]["error_type"] == "RuntimeError"
    assert "synthetic execution failure" in failure["failures"][0]["error_message"]
    assert not (output / "development_lock.json").exists()
