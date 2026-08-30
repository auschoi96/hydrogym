"""Frozen, deterministic policy-quality metric for Kolmogorov control.

This module is outside the candidate-reward path.  Its bytes are pinned by
``codex_hydrogym/training/frozen_metric_pin.py`` and, independently, by
``test/codex_hydrogym/test_frozen_control_metric_guard.py``.  A reward-writing
agent must not edit this file, the pin, or the guard.  At import time this
module re-derives the SHA-256 of its *imported* source file (resolved from the
module object, never from a test file's neighbour, so shadowing modules and
tampered installed wheels are caught) and raises ``RuntimeError`` on mismatch.
``codex_hydrogym/training/runner.py::run_training`` re-asserts the same pin
before any training work starts, and ``.github/workflows/pytest.yml`` runs the
guard tests in CI.  Tampering therefore hard-fails the loop instead of only
failing a test that nothing executes.

Honest guard boundary: every pin lives inside this repository, so an agent that
edits this file, the pin, and the guard in one commit still passes every
automated check.  The guard does not make tampering impossible; it converts
tampering into a visible, reviewable, multi-file diff whose only plausible
motivation is changing the frozen metric, and it hard-fails every
divergent-copy attack that does not also rewrite the pins: shadowing modules
earlier on ``sys.path``, tampered installed wheels, and edited working-tree
files.

Wiring status (round 2, stated plainly): this is an UNWIRED COMPONENT.  Nothing
in the training or evaluation loop calls :func:`compare_candidates` yet, no
candidate policy has been evaluated with it, and **no ``C*`` has been
preregistered** -- no preregistration artifact exists in this repository
(``PREREGISTERED_ARTIFACT_PATH`` is ``None``).  The aggregation producer
(:func:`seed_outcome_from_intervals`) and the write-once preregistration
artifact (:func:`write_preregistration` / :func:`load_preregistration`) are
shipped and unit-tested against synthetic interval records, but the rollout run
that would measure a real ``C*`` at the frozen evaluation configuration
(Re=100, 64x64 grid, dt=0.002, 2x100-interval scoring windows, seeds
401/503/607/709) has NOT been executed.  Inventing a ``C*`` without that run
would be exactly the self-deception this metric exists to prevent.  Follow-up:
run the frozen-config rollouts for the zero-action baseline and at least two
trained candidates, preregister the matched target from that evidence BEFORE
any candidate comparison, and call ``compare_candidates`` from the evaluation
path.

For seed ``s``, at a preregistered matched mean action-L1 cost ``C*``, the
control-quality score is normalized TKE reduction::

    q_s = (TKE_uncontrolled,s - TKE_candidate,s) / TKE_uncontrolled,s

Every candidate/seed is checked against ``abs(C_s - C*) <= tolerance * C*``.
A violation no longer aborts the whole report: the candidate is recorded as
cost-rejected with per-seed reasons while the remaining candidates are still
reported, so one out-of-band seed cannot hide anyone's result.  Only
cost-admitted candidates enter the reward-vs-metric correlation and pairwise
hacking flags.  Honest calibration note: the +/-5% default band is a design
default, not a calibrated one.  No real-rollout measurement exists of per-seed
mean action-L1 spread at this configuration, so the fraction of real candidates
that survives the band is UNMEASURED.  Per-seed ``mean_control_l1`` and
per-candidate ``max_relative_cost_deviation`` are reported so the band's
tightness can be audited once real rollouts exist.

The score never consumes the candidate reward. It and its ranking are thus
invariant to every positive affine reward transformation ``R' = a R + b``,
``a > 0``. Pearson reward-vs-metric correlation is also invariant to that
transformation and is reported only as a hacking diagnostic. Pairwise flags
identify cost-admitted candidates whose mean candidate reward is higher while
their frozen metric is lower.

``q_s`` is bounded above by 1 but unbounded below: a diverged rollout
(``q_s = -1000``) can drag the four-point student-t interval.  Seeds whose
``q_s`` falls below ``normalized_tke_reduction_floor`` (default 0.0, i.e.
worse than the uncontrolled baseline) are flagged per seed and per candidate
and the per-seed spread is surfaced in ``q_s_spread``; flagged seeds are NOT
silently dropped, and the interval is still computed over all four clusters.

The identical-window requirement is a checked contract, not an assumption:
every :class:`SeedOutcome` carries its evaluation-window structure
(``window_count`` x ``window_intervals``) and :func:`compare_candidates`
refuses candidates whose window structures differ.  The producer additionally
requires both arms to expose exactly the same segmented interval counts, so
trajectory-mean TKE, control L1, and reward are averaged over identical
windows for both arms.  Temporal alignment of the windows themselves (same
phases, same dt) remains the caller's responsibility; it cannot be verified
from bare aggregates and is stated here rather than checked.

HydroGym already computes trajectory-mean TKE in
``hydrogym/jax/envs/kolmogorov.py:562-567`` from the TKE implementation at
``hydrogym/jax/utils/utils.py:180-199``. It exposes that quantity and action
L1 at ``hydrogym/jax/envs/kolmogorov.py:646-649``. Training validation already
requires both fields at ``codex_hydrogym/training/validation.py:150-151``.

Limitations: the metric measures TKE suppression only at the frozen action-L1
operating point. It cannot compare unmatched control budgets, certify
stability outside measured rollouts, detect unmeasured actuator power or
state-constraint violations, establish causality, or generalize beyond the
sampled seeds. Action L1 is control authority, not physical actuator energy;
matching mean action L1 does not preclude a candidate concentrating control in
bursts inside scoring windows, which is why the arms must be measured over
identical windows. The uncontrolled denominator must be positive. Exactly four
independent seed clusters are required because the reused frozen 95% t critical
is for four clusters; phases/windows must be reduced within seed before calling
this API. Callers are warned when the seed set is not the frozen ensemble
diagnostic's 401/503/607/709, reuses the historical Gate 0 v1/v2 seeds
7/101/211/307 that the diagnostic forbids, or overlaps its reserved seeds.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import correlation, fmean, stdev
from typing import Any

from codex_hydrogym.gate0.ensemble_diagnostic import EnsembleDiagnosticSpec, _mean_ci
from codex_hydrogym.training.frozen_metric_pin import assert_frozen_metric_source

FROZEN_SEED_COUNT = 4

# Round 1 instantiated EnsembleDiagnosticSpec() here, which ran the diagnostic's
# entire __post_init__ (study identity, schema version, unrelated frozen artifact
# digests) at import time.  The defaults are now read from the dataclass fields
# directly, so an unrelated frozen-artifact change cannot break this module's
# import; the four-cluster count contract is independently re-enforced below.


def _field_default(spec_type: type, field_name: str) -> Any:
    for field in fields(spec_type):
        if field.name == field_name:
            return field.default
    raise AttributeError(f"{spec_type.__name__} has no field {field_name!r}")


FROZEN_T_CRITICAL_95 = float(_field_default(EnsembleDiagnosticSpec, "seed_cluster_t_critical_95"))
DIAGNOSTIC_DEVELOPMENT_SEEDS = tuple(int(seed) for seed in _field_default(EnsembleDiagnosticSpec, "seeds"))
DIAGNOSTIC_RESERVED_SEEDS = tuple(int(seed) for seed in _field_default(EnsembleDiagnosticSpec, "reserved_seeds"))
# Historical Gate 0 v1/v2 seeds that ensemble_diagnostic.__post_init__ forbids reusing;
# hardcoded here (with citation) because they are a guard invariant, not a field.
HISTORICAL_GATE_SEEDS = frozenset({7, 101, 211, 307})

# No preregistered matched cost C* exists in this repository as of round 2.  The
# write-once preregistration machinery below is the only sanctioned way to record
# one; a follow-up must run the real frozen-config rollouts described in the
# module docstring and then flip this constant together with a new artifact.
PREREGISTERED_ARTIFACT_PATH: str | None = None

PREREGISTRATION_SCHEMA_VERSION = "codex_hydrogym.training.frozen_control_metric.preregistration.v1"
PREREGISTRATION_STATUS = "preregistered_before_training"


def _finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True)
class SeedOutcome:
    """Seed-cluster aggregates from matched uncontrolled/candidate rollouts.

    ``window_count`` x ``window_intervals`` is the evaluation-window structure
    over which both arms were reduced; the identical-window contract requires
    every outcome entering one comparison to carry the same structure.
    """

    seed: int
    uncontrolled_mean_tke: float
    candidate_mean_tke: float
    mean_control_l1: float
    mean_candidate_reward: float
    window_count: int
    window_intervals: int

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for field_name in (
            "uncontrolled_mean_tke",
            "candidate_mean_tke",
            "mean_control_l1",
            "mean_candidate_reward",
        ):
            object.__setattr__(self, field_name, _finite(getattr(self, field_name), field_name))
        for field_name in ("window_count", "window_intervals"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.uncontrolled_mean_tke <= 0.0:
            raise ValueError("uncontrolled_mean_tke must be positive")
        if self.candidate_mean_tke < 0.0:
            raise ValueError("candidate_mean_tke must be nonnegative")
        if self.mean_control_l1 < 0.0:
            raise ValueError("mean_control_l1 must be nonnegative")


@dataclass(frozen=True)
class CandidateOutcomes:
    """All independent seed clusters for one reward-trained policy."""

    candidate: str
    outcomes: tuple[SeedOutcome, ...]

    def __post_init__(self) -> None:
        if not self.candidate.strip():
            raise ValueError("candidate must be nonempty")
        if len(self.outcomes) != FROZEN_SEED_COUNT:
            raise ValueError(f"comparison requires exactly {FROZEN_SEED_COUNT} independent seed clusters")
        seeds = [outcome.seed for outcome in self.outcomes]
        if len(set(seeds)) != len(seeds):
            raise ValueError("candidate outcomes require distinct seeds")


def _interval_value(record: Mapping[str, Any], key: str, label: str, *, nonnegative: bool = False) -> float:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label} interval records must be mappings (the env info dict)")
    if key not in record:
        raise ValueError(f"{label} interval record is missing {key!r}; the env info dict exposes it")
    value = _finite(record[key], f"{label} {key}")
    if nonnegative and value < 0.0:
        raise ValueError(f"{label} {key} must be nonnegative")
    return value


def seed_outcome_from_intervals(
    *,
    seed: int,
    uncontrolled_intervals: Sequence[Mapping[str, Any]],
    candidate_intervals: Sequence[Mapping[str, Any]],
    window_count: int,
    window_intervals: int,
) -> SeedOutcome:
    """Aggregate per-interval env info records into one :class:`SeedOutcome`.

    Both arms must be reduced over IDENTICAL evaluation windows: exactly
    ``window_count`` windows of exactly ``window_intervals`` env steps each,
    sampled at the same dt and phases, with the uncontrolled arm driven by the
    zero action.  This is the checked half of the contract (equal segmented
    counts per arm); temporal alignment of the windows themselves remains the
    caller's responsibility.  ``mean_tke`` is the env's trajectory-mean TKE
    (``hydrogym/jax/envs/kolmogorov.py`` info key), ``control_l1`` its action
    L1, and ``reward_total`` its scalar reward; all three candidate quantities
    are averaged over the same scored intervals.
    """
    if isinstance(window_count, bool) or not isinstance(window_count, int) or window_count <= 0:
        raise ValueError("window_count must be a positive integer")
    if isinstance(window_intervals, bool) or not isinstance(window_intervals, int) or window_intervals <= 0:
        raise ValueError("window_intervals must be a positive integer")
    scored_intervals = window_count * window_intervals
    if len(uncontrolled_intervals) != scored_intervals or len(candidate_intervals) != scored_intervals:
        raise ValueError(
            "identical evaluation windows require exactly "
            f"{scored_intervals} scored intervals per arm "
            f"({window_count} windows x {window_intervals} intervals); got "
            f"uncontrolled={len(uncontrolled_intervals)}, candidate={len(candidate_intervals)}"
        )
    uncontrolled_tke = []
    candidate_tke = []
    candidate_l1 = []
    candidate_reward = []
    for index in range(scored_intervals):
        uncontrolled_tke.append(_interval_value(uncontrolled_intervals[index], "mean_tke", "uncontrolled"))
        _interval_value(uncontrolled_intervals[index], "control_l1", "uncontrolled", nonnegative=True)
        candidate_tke.append(_interval_value(candidate_intervals[index], "mean_tke", "candidate"))
        candidate_l1.append(_interval_value(candidate_intervals[index], "control_l1", "candidate", nonnegative=True))
        candidate_reward.append(_interval_value(candidate_intervals[index], "reward_total", "candidate"))
    return SeedOutcome(
        seed=seed,
        uncontrolled_mean_tke=fmean(uncontrolled_tke),
        candidate_mean_tke=fmean(candidate_tke),
        mean_control_l1=fmean(candidate_l1),
        mean_candidate_reward=fmean(candidate_reward),
        window_count=window_count,
        window_intervals=window_intervals,
    )


@dataclass(frozen=True)
class MetricPreregistration:
    """Write-once record of the matched cost target fixed before training.

    The artifact's claim of being written before training is enforced by review
    of git history, not cryptographically; the schema makes that claim explicit
    and loadable so the evaluation path can bind a comparison to it.
    """

    schema_version: str
    status: str
    target_mean_control_l1: float
    relative_cost_tolerance: float
    window_count: int
    window_intervals: int
    seeds: tuple[int, ...]
    derivation: str


def _validate_preregistration_fields(
    target_mean_control_l1: float,
    relative_cost_tolerance: float,
    window_count: int,
    window_intervals: int,
    seeds: Sequence[int],
    derivation: str,
) -> None:
    if _finite(target_mean_control_l1, "target_mean_control_l1") <= 0.0:
        raise ValueError("target_mean_control_l1 must be positive")
    tolerance = _finite(relative_cost_tolerance, "relative_cost_tolerance")
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("relative_cost_tolerance must be in [0, 1)")
    if isinstance(window_count, bool) or not isinstance(window_count, int) or window_count <= 0:
        raise ValueError("window_count must be a positive integer")
    if isinstance(window_intervals, bool) or not isinstance(window_intervals, int) or window_intervals <= 0:
        raise ValueError("window_intervals must be a positive integer")
    if len(seeds) != FROZEN_SEED_COUNT or len(set(seeds)) != FROZEN_SEED_COUNT:
        raise ValueError(f"preregistration requires exactly {FROZEN_SEED_COUNT} distinct seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("preregistration seeds must be integers")
    if not isinstance(derivation, str) or not derivation.strip():
        raise ValueError("derivation must describe how the target cost was chosen")


def write_preregistration(
    path: str | Path,
    *,
    target_mean_control_l1: float,
    relative_cost_tolerance: float,
    window_count: int,
    window_intervals: int,
    seeds: Sequence[int],
    derivation: str,
) -> MetricPreregistration:
    """Record the preregistered matched cost target; refuses to overwrite."""
    _validate_preregistration_fields(
        target_mean_control_l1,
        relative_cost_tolerance,
        window_count,
        window_intervals,
        seeds,
        derivation,
    )
    preregistration = MetricPreregistration(
        schema_version=PREREGISTRATION_SCHEMA_VERSION,
        status=PREREGISTRATION_STATUS,
        target_mean_control_l1=float(target_mean_control_l1),
        relative_cost_tolerance=float(relative_cost_tolerance),
        window_count=window_count,
        window_intervals=window_intervals,
        seeds=tuple(int(seed) for seed in seeds),
        derivation=derivation,
    )
    artifact = Path(path)
    if artifact.exists():
        raise FileExistsError(f"preregistration is write-once; refusing to overwrite: {artifact}")
    payload = {
        "schema_version": preregistration.schema_version,
        "status": preregistration.status,
        "target_mean_control_l1": preregistration.target_mean_control_l1,
        "relative_cost_tolerance": preregistration.relative_cost_tolerance,
        "window_count": preregistration.window_count,
        "window_intervals": preregistration.window_intervals,
        "seeds": list(preregistration.seeds),
        "derivation": preregistration.derivation,
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return preregistration


def load_preregistration(path: str | Path) -> MetricPreregistration:
    """Load and fully validate a preregistration artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "status",
        "target_mean_control_l1",
        "relative_cost_tolerance",
        "window_count",
        "window_intervals",
        "seeds",
        "derivation",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("preregistration artifact schema mismatch")
    if payload["schema_version"] != PREREGISTRATION_SCHEMA_VERSION:
        raise ValueError("preregistration artifact schema_version mismatch")
    if payload["status"] != PREREGISTRATION_STATUS:
        raise ValueError("preregistration artifact status mismatch")
    seeds = payload["seeds"]
    if not isinstance(seeds, list):
        raise ValueError("preregistration seeds must be a list")
    _validate_preregistration_fields(
        payload["target_mean_control_l1"],
        payload["relative_cost_tolerance"],
        payload["window_count"],
        payload["window_intervals"],
        seeds,
        payload["derivation"],
    )
    return MetricPreregistration(
        schema_version=payload["schema_version"],
        status=payload["status"],
        target_mean_control_l1=float(payload["target_mean_control_l1"]),
        relative_cost_tolerance=float(payload["relative_cost_tolerance"]),
        window_count=payload["window_count"],
        window_intervals=payload["window_intervals"],
        seeds=tuple(int(seed) for seed in seeds),
        derivation=payload["derivation"],
    )


def _seed_identity_warnings(seed_set: frozenset[int]) -> list[str]:
    messages: list[str] = []
    if seed_set != set(DIAGNOSTIC_DEVELOPMENT_SEEDS):
        messages.append(
            f"seed set {sorted(seed_set)} is not the frozen ensemble-diagnostic seed set "
            f"{list(DIAGNOSTIC_DEVELOPMENT_SEEDS)}"
        )
    if seed_set & HISTORICAL_GATE_SEEDS:
        messages.append(
            f"seed set {sorted(seed_set)} reuses historical Gate 0 v1/v2 seeds "
            f"{sorted(HISTORICAL_GATE_SEEDS)} that ensemble_diagnostic explicitly forbids"
        )
    if seed_set & set(DIAGNOSTIC_RESERVED_SEEDS):
        messages.append(
            f"seed set {sorted(seed_set)} overlaps the ensemble diagnostic's reserved seeds "
            f"{list(DIAGNOSTIC_RESERVED_SEEDS)}"
        )
    return messages


def compare_candidates(
    candidates: Sequence[CandidateOutcomes],
    *,
    matched_mean_control_l1: float,
    relative_cost_tolerance: float = 0.05,
    normalized_tke_reduction_floor: float = 0.0,
    preregistration: MetricPreregistration | None = None,
) -> dict[str, object]:
    """Compare policies on frozen TKE reduction and expose reward divergence.

    Cost mismatches mark candidates cost-rejected (with reasons) instead of
    aborting the report; structural contract violations (seed counts, seed-set
    identity across candidates, window-structure identity, preregistration
    binding) still raise.
    """
    target_cost = _finite(matched_mean_control_l1, "matched_mean_control_l1")
    tolerance = _finite(relative_cost_tolerance, "relative_cost_tolerance")
    floor = _finite(normalized_tke_reduction_floor, "normalized_tke_reduction_floor")
    if target_cost <= 0.0:
        raise ValueError("matched_mean_control_l1 must be positive")
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("relative_cost_tolerance must be in [0, 1)")
    if not floor < 1.0:
        raise ValueError("normalized_tke_reduction_floor must be below 1 (q_s never exceeds 1)")
    if len(candidates) < 2:
        raise ValueError("reward-vs-metric diagnostics require at least two candidates")
    names = [candidate.candidate for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be distinct")

    expected_seeds = {outcome.seed for outcome in candidates[0].outcomes}
    window_structures = {
        (outcome.window_count, outcome.window_intervals) for candidate in candidates for outcome in candidate.outcomes
    }
    if len(window_structures) != 1:
        raise ValueError(
            "all candidates must be reduced over identical evaluation windows; got window structures "
            f"{sorted(window_structures)}"
        )
    window_count, window_intervals = next(iter(window_structures))
    if preregistration is not None:
        if preregistration.target_mean_control_l1 != target_cost:
            raise ValueError(
                "preregistered target_mean_control_l1 "
                f"{preregistration.target_mean_control_l1} does not match the comparison target {target_cost}"
            )
        if preregistration.relative_cost_tolerance != tolerance:
            raise ValueError("preregistered relative_cost_tolerance does not match the comparison tolerance")
        if (preregistration.window_count, preregistration.window_intervals) != (window_count, window_intervals):
            raise ValueError("preregistered evaluation-window structure does not match the candidates")
        if set(preregistration.seeds) != expected_seeds:
            raise ValueError("preregistered seeds do not match the candidate seed set")

    allowed_difference = tolerance * target_cost
    identity_warnings = _seed_identity_warnings(frozenset(expected_seeds))
    for message in identity_warnings:
        warnings.warn(message, stacklevel=2)
    results: list[dict[str, object]] = []
    for candidate in candidates:
        if {outcome.seed for outcome in candidate.outcomes} != expected_seeds:
            raise ValueError("all candidates must use the identical seed clusters")
        ordered = sorted(candidate.outcomes, key=lambda outcome: outcome.seed)
        cost_rejection_reasons: list[str] = []
        seeds_below_floor: list[int] = []
        per_seed = []
        q_values = []
        max_cost_deviation = 0.0
        for outcome in ordered:
            q_s = (outcome.uncontrolled_mean_tke - outcome.candidate_mean_tke) / outcome.uncontrolled_mean_tke
            cost_deviation = abs(outcome.mean_control_l1 - target_cost)
            max_cost_deviation = max(max_cost_deviation, cost_deviation)
            below_floor = q_s < floor
            if cost_deviation > allowed_difference:
                cost_rejection_reasons.append(
                    f"seed {outcome.seed} has unmatched mean_control_l1: "
                    f"{outcome.mean_control_l1} vs {target_cost} "
                    f"(allowed +/- {allowed_difference})"
                )
            if below_floor:
                seeds_below_floor.append(outcome.seed)
            q_values.append(q_s)
            per_seed.append(
                {
                    "seed": outcome.seed,
                    "normalized_tke_reduction": q_s,
                    "mean_control_l1": outcome.mean_control_l1,
                    "candidate_reward": outcome.mean_candidate_reward,
                    "below_floor": below_floor,
                }
            )
        results.append(
            {
                "candidate": candidate.candidate,
                "per_seed": per_seed,
                "clustered_interval_95": _mean_ci(q_values, FROZEN_T_CRITICAL_95),
                "mean_candidate_reward": fmean(outcome.mean_candidate_reward for outcome in ordered),
                "cost_admitted": not cost_rejection_reasons,
                "cost_rejection_reasons": cost_rejection_reasons,
                "max_relative_cost_deviation": max_cost_deviation / target_cost,
                "seeds_below_floor": seeds_below_floor,
                "q_s_spread": {"min": min(q_values), "max": max(q_values), "stdev": stdev(q_values)},
            }
        )

    admitted = [result for result in results if result["cost_admitted"]]
    rewards = [float(result["mean_candidate_reward"]) for result in admitted]
    metrics = [float(result["clustered_interval_95"]["mean"]) for result in admitted]
    reward_metric_correlation = None
    if len(rewards) >= 2 and len(set(rewards)) > 1 and len(set(metrics)) > 1:
        reward_metric_correlation = correlation(rewards, metrics)

    flagged = sorted(
        str(result["candidate"])
        for result in admitted
        if any(
            float(result["mean_candidate_reward"]) > float(other["mean_candidate_reward"])
            and float(result["clustered_interval_95"]["mean"]) < float(other["clustered_interval_95"]["mean"])
            for other in admitted
        )
    )
    return {
        "metric": "normalized_tke_reduction_at_matched_action_l1",
        "matched_mean_control_l1": target_cost,
        "relative_cost_tolerance": tolerance,
        "normalized_tke_reduction_floor": floor,
        "window_count": window_count,
        "window_intervals": window_intervals,
        "seed_count": FROZEN_SEED_COUNT,
        "preregistration": (
            {
                "schema_version": preregistration.schema_version,
                "status": preregistration.status,
                "target_mean_control_l1": preregistration.target_mean_control_l1,
            }
            if preregistration is not None
            else None
        ),
        "seed_identity_warnings": identity_warnings,
        "candidates": results,
        "cost_admitted_candidates": [str(result["candidate"]) for result in admitted],
        "cost_rejected_candidates": {
            str(result["candidate"]): list(result["cost_rejection_reasons"])
            for result in results
            if not result["cost_admitted"]
        },
        "reward_metric_correlation": reward_metric_correlation,
        "reward_metric_diverged": reward_metric_correlation is not None and reward_metric_correlation < 0.0,
        "reward_hacking_candidates": flagged,
    }


# Tamper evidence (import-time): hash the IMPORTED module's resolved source file
# against the pinned digest.  This runs on every import of this module -- in the
# training loop, in tests, and in CI -- so a shadowing module, a tampered
# installed wheel, or an edited working-tree file hard-fails here instead of
# only failing a test a human might never run.
assert_frozen_metric_source()
