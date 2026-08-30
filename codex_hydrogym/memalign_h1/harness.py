"""H1 harness: held-out judge-agreement comparison for MemAlign alignment.

Aligns the base judge on the locked TRAIN fold only (reusing
``codex_hydrogym.genai.optimization.align_critic_quality_judge``, whose locked-manifest
and per-trace-label checks fail loudly on any fold leakage), then compares base vs
aligned judge agreement against HELD-OUT HUMAN labels.

Statistics are the repository's own, not newly invented:

- MAE per dimension (the agreed label schema currently has one numeric dimension,
  ``critic_quality``, but the analysis is defined over D dimensions);
- the decision interval is the group-clustered 95% interval from
  ``codex_hydrogym.gate0.ensemble_diagnostic._mean_ci`` with the frozen df=3
  t-critical (4 group clusters);
- a fixed-seed group-clustered bootstrap interval is reported as a secondary
  sensitivity statistic only; it never carries the decision.

Unfavorable results are a legitimate outcome: if the aligned judge agrees no better
with held-out human labels, H1 fails or is inconclusive.  Nothing here licenses any
claim about RL or fluid performance -- the PPO reward is deterministic
(coding_rl/experiment.py:1110-1114) and MemAlign is not in that path.
"""

from __future__ import annotations

import random
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

from codex_hydrogym import CRITIC_QUALITY_ASSESSMENT_NAME
from codex_hydrogym.gate0.ensemble_diagnostic import _mean_ci
from codex_hydrogym.genai.optimization import align_critic_quality_judge
from codex_hydrogym.memalign_h1 import (
    FROZEN_BOOTSTRAP_REPLICATES,
    FROZEN_BOOTSTRAP_SEED,
    FROZEN_HELDOUT_GROUP_COUNT,
    FROZEN_T_CRITICAL_95,
    PROTOCOL_ID,
)

HARNESS_SCHEMA_VERSION = "codex_hydrogym.memalign_h1.harness.v1"
PRIMARY_DIMENSION = "value"

DECISION_PASS = "PASS"
DECISION_FAIL = "FAIL"
DECISION_INCONCLUSIVE = "INCONCLUSIVE"


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    numeric = float(value)
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return numeric


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute error between equally sized score vectors."""
    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("actual and predicted must be equally sized and non-empty")
    return fmean(
        abs(_finite_number(p, name="predicted") - _finite_number(a, name="actual"))
        for a, p in zip(actual, predicted, strict=True)
    )


def group_clustered_delta_mae_ci(
    *,
    per_group_delta_mae: Mapping[str, float],
    t_critical: float = FROZEN_T_CRITICAL_95,
) -> dict[str, float]:
    """Group-clustered interval on delta-MAE, reusing gate0's ``_mean_ci``.

    One delta-MAE per group cluster; the interval is ``mean +/- t_critical * SE`` over
    the cluster values with the frozen df=3 t-critical, so the held-out fold must
    contain exactly four group clusters.
    """
    if not isinstance(per_group_delta_mae, Mapping) or not per_group_delta_mae:
        raise ValueError("per_group_delta_mae must be a non-empty mapping")
    groups = sorted(per_group_delta_mae)
    if len(groups) != FROZEN_HELDOUT_GROUP_COUNT:
        raise ValueError(
            "the held-out fold must contain exactly "
            f"{FROZEN_HELDOUT_GROUP_COUNT} group clusters for the frozen df=3 t-critical; "
            f"found {len(groups)}"
        )
    if not isinstance(t_critical, (int, float)) or isinstance(t_critical, bool):
        raise ValueError("t_critical must be numeric")
    return _mean_ci(tuple(per_group_delta_mae[group] for group in groups), float(t_critical))


def group_clustered_bootstrap_delta_mae_ci(
    *,
    per_group_delta_mae: Mapping[str, float],
    seed: int = FROZEN_BOOTSTRAP_SEED,
    replicates: int = FROZEN_BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    """Secondary fixed-seed percentile bootstrap over the group clusters.

    Resamples whole groups with replacement.  Reported for sensitivity only; the
    decision uses ``group_clustered_delta_mae_ci``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or not 100 <= replicates <= 100_000:
        raise ValueError("replicates must be an integer in [100, 100000]")
    groups = sorted(per_group_delta_mae)
    values = tuple(per_group_delta_mae[group] for group in groups)
    if len(values) < 2:
        raise ValueError("the bootstrap requires at least two group clusters")
    rng = random.Random(seed)
    bootstrapped_means = [
        fmean(values[rng.randrange(len(values))] for _ in range(len(values))) for _ in range(replicates)
    ]
    order = sorted(bootstrapped_means)
    lower_index = int(0.025 * replicates)
    upper_index = int(0.975 * replicates) - 1
    return {
        "seed": seed,
        "replicates": replicates,
        "mean": fmean(order),
        "lower": order[lower_index],
        "upper": order[upper_index],
    }


def decide_h1(*, delta_mae_interval: Mapping[str, float]) -> str:
    """Decision rule: PASS only when the held-out delta-MAE interval is wholly favorable.

    delta-MAE is ``MAE(aligned) - MAE(base)`` per group, so favorable means negative
    (the aligned judge agrees better with held-out HUMAN labels).  A wholly negative
    interval passes; a wholly positive interval fails (the aligned judge regressed);
    anything straddling zero is inconclusive.
    """
    lower = _finite_number(delta_mae_interval.get("lower"), name="interval.lower")
    upper = _finite_number(delta_mae_interval.get("upper"), name="interval.upper")
    if upper < 0.0:
        return DECISION_PASS
    if lower > 0.0:
        return DECISION_FAIL
    return DECISION_INCONCLUSIVE


def heldout_agreement_metrics(*, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-dimension MAE plus the group-clustered intervals on delta-MAE.

    Each row is one held-out trace: ``group_id``, ``dimension`` (defaults to
    ``value``), ``base_score``, ``aligned_score``, ``human_score``.  Per-dimension MAE
    is reported for the base and aligned judges, including regressions; the decision
    interval is computed per dimension on each group's mean delta-MAE.
    """
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("at least one held-out agreement row is required")
    dimensions: dict[str, list[dict[str, Any]]] = {}
    for row in materialized:
        dimension = str(row.get("dimension", PRIMARY_DIMENSION))
        dimensions.setdefault(dimension, []).append(
            {
                "group_id": str(row["group_id"]),
                "base_score": _finite_number(row["base_score"], name="base_score"),
                "aligned_score": _finite_number(row["aligned_score"], name="aligned_score"),
                "human_score": _finite_number(row["human_score"], name="human_score"),
            }
        )

    per_dimension: dict[str, dict[str, Any]] = {}
    intervals: dict[str, dict[str, Any]] = {}
    bootstrap_intervals: dict[str, dict[str, Any]] = {}
    for dimension, dimension_rows in dimensions.items():
        base_mae = mae(
            [row["human_score"] for row in dimension_rows],
            [row["base_score"] for row in dimension_rows],
        )
        aligned_mae = mae(
            [row["human_score"] for row in dimension_rows],
            [row["aligned_score"] for row in dimension_rows],
        )
        per_group_delta: dict[str, float] = {}
        group_rows: dict[str, list[dict[str, float]]] = {}
        for row in dimension_rows:
            group_rows.setdefault(row["group_id"], []).append(row)
        for group_id, rows_in_group in group_rows.items():
            group_base_mae = mae(
                [row["human_score"] for row in rows_in_group],
                [row["base_score"] for row in rows_in_group],
            )
            group_aligned_mae = mae(
                [row["human_score"] for row in rows_in_group],
                [row["aligned_score"] for row in rows_in_group],
            )
            per_group_delta[group_id] = group_aligned_mae - group_base_mae
        per_dimension[dimension] = {
            "base_mae": base_mae,
            "aligned_mae": aligned_mae,
            "delta_mae": aligned_mae - base_mae,
            "heldout_traces": len(dimension_rows),
            "heldout_groups": len(per_group_delta),
            "per_group_delta_mae": per_group_delta,
        }
        intervals[dimension] = group_clustered_delta_mae_ci(per_group_delta_mae=per_group_delta)
        bootstrap_intervals[dimension] = group_clustered_bootstrap_delta_mae_ci(per_group_delta_mae=per_group_delta)

    decision = decide_h1(delta_mae_interval=intervals[PRIMARY_DIMENSION])
    return {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "primary_dimension": PRIMARY_DIMENSION,
        "judge_name": CRITIC_QUALITY_ASSESSMENT_NAME,
        "delta_mae_direction": "aligned_mae - base_mae; negative favors the aligned judge",
        "per_dimension": per_dimension,
        "group_clustered_ci_95": {
            dimension: {
                "mean": interval["mean"],
                "standard_error": interval["standard_error"],
                "lower": interval["lower"],
                "upper": interval["upper"],
                "t_critical": FROZEN_T_CRITICAL_95,
                "group_clusters": FROZEN_HELDOUT_GROUP_COUNT,
            }
            for dimension, interval in intervals.items()
        },
        "bootstrap_ci_95_secondary": bootstrap_intervals,
        "decision": decision,
        "decision_rule": "PASS only when the held-out delta-MAE 95% interval is wholly "
        "negative (upper < 0); FAIL when wholly positive; else INCONCLUSIVE",
    }


def evaluate_h1(
    *,
    train_traces: Sequence[Any],
    train_bundle_ids: Sequence[str],
    heldout_bundle_ids: Sequence[str],
    base_judge: Any,
    reflection_lm: str,
    embedding_model: str,
    heldout_rows: Sequence[Mapping[str, Any]],
    retrieval_k: int = 5,
    required_arms: Sequence[str] = ("codex", "claude"),
    optimizer_factory: Callable[..., Any] | None = None,
    align_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Align on the locked TRAIN fold, then measure agreement on held-out labels.

    Held-out bundle ids and held-out HUMAN labels never enter the alignment call: the
    imported ``align_critic_quality_judge`` raises on any trace outside the locked
    train manifest, and each train trace must already carry exactly one adjudicated
    ``critic_quality`` HUMAN label (a machine-written assessment is rejected there).
    The returned report carries the held-out agreement metrics and the H1 decision.
    """
    align = align_fn if align_fn is not None else align_critic_quality_judge
    aligned_judge = align(
        train_traces=tuple(train_traces),
        train_bundle_ids=tuple(train_bundle_ids),
        heldout_bundle_ids=tuple(heldout_bundle_ids),
        base_judge=base_judge,
        reflection_lm=reflection_lm,
        embedding_model=embedding_model,
        retrieval_k=retrieval_k,
        required_arms=tuple(required_arms),
        optimizer_factory=optimizer_factory,
    )

    agreement = heldout_agreement_metrics(rows=heldout_rows)
    agreement["aligned_judge_name"] = getattr(aligned_judge, "name", CRITIC_QUALITY_ASSESSMENT_NAME)
    agreement["train_bundle_count"] = len(set(train_bundle_ids))
    agreement["heldout_bundle_count"] = len(set(heldout_bundle_ids))
    agreement["heldout_label_count"] = len(heldout_rows)
    return agreement
