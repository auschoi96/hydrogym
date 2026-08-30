"""Preregistered operating-characteristic simulation for the MemAlign H1 endpoint."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from typing import Any, Sequence

import numpy as np
from scipy.stats import nct, t

from codex_hydrogym.memalign_h1 import FROZEN_T_CRITICAL_95
from codex_hydrogym.memalign_h1.harness import (
    DECISION_DEGENERATE,
    DECISION_FAIL,
    DECISION_INCONCLUSIVE,
    DECISION_PASS,
)

POWER_SIMULATION_SCHEMA_VERSION = "codex_hydrogym.memalign_h1.power.v1"
DEFAULT_EFFECTS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
DEFAULT_GROUP_SDS = (0.25, 0.5, 0.75)
DEFAULT_GROUP_SIZES = (2, 3, 5, 8)
DEFAULT_REPLICATES = 100_000
DEFAULT_SEED = 7021


def t_critical_95(group_count: int) -> float:
    """Return the correct two-sided 95% critical value for ``group_count - 1`` df."""
    if isinstance(group_count, bool) or not isinstance(group_count, int) or group_count < 2:
        raise ValueError("group_count must be an integer >= 2")
    if group_count == 4:
        return FROZEN_T_CRITICAL_95
    return float(t.ppf(0.975, df=group_count - 1))


def exact_normal_probabilities(*, true_improvement: float, group_sd: float, group_count: int) -> dict[str, float]:
    """Exact probabilities under independent normal group-level delta-MAEs."""
    if true_improvement < 0 or group_sd <= 0:
        raise ValueError("true_improvement must be >= 0 and group_sd must be > 0")
    df = group_count - 1
    critical = t_critical_95(group_count)
    noncentrality = true_improvement * sqrt(group_count) / group_sd
    passed = float(nct.sf(critical, df, noncentrality))
    failed = float(nct.cdf(-critical, df, noncentrality))
    return {
        DECISION_PASS: passed,
        DECISION_FAIL: failed,
        DECISION_INCONCLUSIVE: 1.0 - passed - failed,
        DECISION_DEGENERATE: 0.0,
    }


def simulate_probabilities(
    *,
    true_improvement: float,
    group_sd: float,
    group_count: int = 4,
    group_sizes: Sequence[int] = DEFAULT_GROUP_SIZES,
    distribution: str = "normal",
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Apply the production rule to simulated groups with preregistered unequal sizes.

    Group sizes are recorded but deliberately do not weight the endpoint: the
    production statistic gives each independent group-level delta one vote.
    ``scaled_t5`` is a heavy-tailed sensitivity distribution standardized to the
    requested group SD.
    """
    if true_improvement < 0 or group_sd <= 0:
        raise ValueError("true_improvement must be >= 0 and group_sd must be > 0")
    if len(group_sizes) != group_count or any(isinstance(size, bool) or size < 1 for size in group_sizes):
        raise ValueError("group_sizes must contain one positive integer per group")
    if distribution not in {"normal", "scaled_t5"}:
        raise ValueError("distribution must be normal or scaled_t5")
    if replicates < 100:
        raise ValueError("replicates must be >= 100")

    rng = np.random.default_rng(seed)
    if distribution == "normal":
        standardized = rng.normal(size=(replicates, group_count))
    else:
        standardized = rng.standard_t(df=5, size=(replicates, group_count)) * sqrt(3.0 / 5.0)
    samples = -true_improvement + group_sd * standardized
    means = samples.mean(axis=1)
    standard_errors = samples.std(axis=1, ddof=1) / sqrt(group_count)
    critical = t_critical_95(group_count)
    lower = means - critical * standard_errors
    upper = means + critical * standard_errors
    degenerate = standard_errors <= 0.0
    passed = (upper < 0.0) & ~degenerate
    failed = (lower > 0.0) & ~degenerate
    inconclusive = ~(passed | failed | degenerate)
    return {
        "true_improvement": true_improvement,
        "group_sd": group_sd,
        "group_count": group_count,
        "group_sizes": list(group_sizes),
        "distribution": distribution,
        "t_critical_95": critical,
        "replicates": replicates,
        "probabilities": {
            DECISION_PASS: float(passed.mean()),
            DECISION_FAIL: float(failed.mean()),
            DECISION_INCONCLUSIVE: float(inconclusive.mean()),
            DECISION_DEGENERATE: float(degenerate.mean()),
        },
    }


def minimum_effect_for_power(*, probability: float, group_sd: float, group_count: int = 4) -> float:
    """Smallest normal-model improvement attaining requested PASS probability."""
    if not 0.0 < probability < 1.0 or group_sd <= 0:
        raise ValueError("probability must be in (0, 1) and group_sd must be > 0")
    low, high = 0.0, 8.0 * group_sd
    for _ in range(100):
        midpoint = (low + high) / 2.0
        power = exact_normal_probabilities(
            true_improvement=midpoint, group_sd=group_sd, group_count=group_count
        )[DECISION_PASS]
        if power >= probability:
            high = midpoint
        else:
            low = midpoint
    return high


def groups_for_power(*, probability: float, true_improvement: float, group_sd: float) -> int:
    """Smallest group count attaining power, recomputing t for each matching df."""
    for group_count in range(2, 10_001):
        power = exact_normal_probabilities(
            true_improvement=true_improvement,
            group_sd=group_sd,
            group_count=group_count,
        )[DECISION_PASS]
        if power >= probability:
            return group_count
    raise ValueError("requested power was not attained by 10000 groups")


def power_report(*, replicates: int = DEFAULT_REPLICATES) -> dict[str, Any]:
    """Return the frozen sweep and decision-relevant conditional thresholds."""
    sweep = [
        simulate_probabilities(
            true_improvement=effect,
            group_sd=group_sd,
            distribution=distribution,
            replicates=replicates,
            seed=DEFAULT_SEED,
        )
        for distribution in ("normal", "scaled_t5")
        for group_sd in DEFAULT_GROUP_SDS
        for effect in DEFAULT_EFFECTS
    ]
    thresholds = {
        str(group_sd): {
            "power_50": minimum_effect_for_power(probability=0.5, group_sd=group_sd),
            "power_80": minimum_effect_for_power(probability=0.8, group_sd=group_sd),
        }
        for group_sd in DEFAULT_GROUP_SDS
    }
    required_groups = groups_for_power(probability=0.8, true_improvement=0.25, group_sd=0.5)
    return {
        "schema_version": POWER_SIMULATION_SCHEMA_VERSION,
        "seed": DEFAULT_SEED,
        "interpretation": "Negative delta-MAE favors alignment; groups, not rows, are independent.",
        "assumptions": {
            "plausible_true_improvement": 0.25,
            "plausible_group_sd": 0.5,
            "unequal_group_sizes": list(DEFAULT_GROUP_SIZES),
        },
        "four_group_thresholds": thresholds,
        "groups_for_80_percent_power_at_plausible_effect": required_groups,
        "heldout_labels_at_two_arms_per_group": 2 * required_groups,
        "sweep": sweep,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    rendered = json.dumps(power_report(replicates=args.replicates), indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
