"""Held-out critic-alignment metrics and claim boundaries."""

import pytest

from codex_hydrogym.genai.metrics import CriticScore, compute_critic_alignment_metrics


def _score(bundle_id, arm, predicted, gold, *, fold="test"):
    return CriticScore(
        bundle_id=bundle_id,
        arm=arm,
        predicted=predicted,
        gold=gold,
        fold=fold,
    )


def test_perfect_heldout_alignment_has_zero_mae_and_full_pair_agreement():
    metrics = compute_critic_alignment_metrics(
        [
            _score("bundle-a", "codex", 5, 5),
            _score("bundle-a", "claude", 2, 2),
            _score("bundle-b", "codex", 3, 3),
            _score("bundle-b", "claude", 3, 3),
        ]
    )

    assert metrics.score_count == 4
    assert metrics.bundle_count == 2
    assert metrics.mean_absolute_error == 0.0
    assert metrics.spearman_correlation == pytest.approx(1.0)
    assert metrics.preference_correct == metrics.preference_total == 2
    assert metrics.preference_agreement == 1.0


def test_spearman_uses_average_tie_ranks_and_reports_constant_predictions():
    tied = compute_critic_alignment_metrics(
        [
            _score("bundle-a", "codex", 5, 5),
            _score("bundle-a", "claude", 4, 4),
            _score("bundle-b", "codex", 2, 2),
            _score("bundle-b", "claude", 2, 2),
        ]
    )
    constant = compute_critic_alignment_metrics(
        [
            _score("bundle-a", "codex", 3, 5),
            _score("bundle-a", "claude", 3, 4),
        ]
    )

    assert tied.spearman_correlation == pytest.approx(1.0)
    assert constant.spearman_correlation is None
    assert constant.preference_agreement == 0.0


def test_metrics_reject_training_rows_duplicates_and_incomplete_pairs():
    with pytest.raises(ValueError, match="held-out test fold"):
        _score("bundle-a", "codex", 4, 4, fold="train")

    with pytest.raises(ValueError, match="only one score"):
        compute_critic_alignment_metrics(
            [
                _score("bundle-a", "codex", 4, 4),
                _score("bundle-a", "codex", 3, 3),
                _score("bundle-a", "claude", 2, 2),
            ]
        )

    with pytest.raises(ValueError, match="both Codex and Claude"):
        compute_critic_alignment_metrics([_score("bundle-a", "codex", 4, 4)])
