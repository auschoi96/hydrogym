"""Held-out metrics for critic alignment, never fluid-performance metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math


CRITIC_ARMS = ("codex", "claude")


@dataclass(frozen=True)
class CriticScore:
    """One held-out critic score paired with one adjudicated human label."""

    bundle_id: str
    arm: str
    predicted: float
    gold: int
    fold: str = "test"

    def __post_init__(self) -> None:
        if not isinstance(self.bundle_id, str) or not self.bundle_id.strip():
            raise ValueError("bundle_id must be non-empty")
        if self.arm not in CRITIC_ARMS:
            raise ValueError(f"arm must be one of {CRITIC_ARMS}")
        if self.fold != "test":
            raise ValueError("critic alignment metrics require the held-out test fold")
        if isinstance(self.predicted, bool) or not isinstance(self.predicted, (int, float)):
            raise TypeError("predicted must be numeric")
        if not math.isfinite(float(self.predicted)) or not 1.0 <= float(self.predicted) <= 5.0:
            raise ValueError("predicted must be finite and in [1, 5]")
        if isinstance(self.gold, bool) or not isinstance(self.gold, int) or not 1 <= self.gold <= 5:
            raise ValueError("gold must be an integer in [1, 5]")


@dataclass(frozen=True)
class CriticAlignmentMetrics:
    """Summary of judge agreement with held-out human ``critic_quality`` labels."""

    score_count: int
    bundle_count: int
    mean_absolute_error: float
    spearman_correlation: float | None
    preference_correct: int
    preference_total: int

    @property
    def preference_agreement(self) -> float:
        return self.preference_correct / self.preference_total


def _average_ranks(values: list[float]) -> list[float]:
    """Assign one-based average ranks so Spearman ties are deterministic."""
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return None
    numerator = sum(a * b for a, b in zip(left_centered, right_centered, strict=True))
    return numerator / denominator


def _preference(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def compute_critic_alignment_metrics(scores: Iterable[CriticScore]) -> CriticAlignmentMetrics:
    """Compare judge scores with gold labels on complete held-out Codex/Claude pairs.

    Preference agreement is an exact three-way comparison: Codex preferred,
    Claude preferred, or tied. Human ties only agree with predicted ties.
    """
    materialized = list(scores)
    if not materialized:
        raise ValueError("at least one held-out critic-score pair is required")

    by_bundle: dict[str, dict[str, CriticScore]] = {}
    for score in materialized:
        if not isinstance(score, CriticScore):
            raise TypeError("scores must contain CriticScore instances")
        bundle = by_bundle.setdefault(score.bundle_id, {})
        if score.arm in bundle:
            raise ValueError("each bundle may contain only one score per critic arm")
        bundle[score.arm] = score
    for arms in by_bundle.values():
        if set(arms) != set(CRITIC_ARMS):
            raise ValueError("each held-out bundle must contain both Codex and Claude scores")

    ordered = [
        by_bundle[bundle_id][arm]
        for bundle_id in sorted(by_bundle)
        for arm in CRITIC_ARMS
    ]
    predicted = [float(score.predicted) for score in ordered]
    gold = [float(score.gold) for score in ordered]
    mae = sum(abs(actual - expected) for actual, expected in zip(predicted, gold, strict=True)) / len(ordered)
    spearman = _pearson(_average_ranks(predicted), _average_ranks(gold))

    preference_correct = 0
    for arms in by_bundle.values():
        codex = arms["codex"]
        claude = arms["claude"]
        predicted_preference = _preference(float(codex.predicted) - float(claude.predicted))
        gold_preference = _preference(float(codex.gold - claude.gold))
        preference_correct += predicted_preference == gold_preference

    return CriticAlignmentMetrics(
        score_count=len(ordered),
        bundle_count=len(by_bundle),
        mean_absolute_error=mae,
        spearman_correlation=spearman,
        preference_correct=preference_correct,
        preference_total=len(by_bundle),
    )
