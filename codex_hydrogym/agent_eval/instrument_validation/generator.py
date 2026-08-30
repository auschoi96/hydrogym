"""Synthetic reward-candidate generator with known, ordered quality tiers.

Ground truth is constructed here, so it is known exactly. A candidate is a
bounded reward-spec text for the HydroGym cylinder-wake control task, and its
``tier`` is the quality level assigned by the rubric below. The ordering
``0 < 1 < 2 < 3 < 4`` (clearly bad to clearly good) is a rubric claim, not a
measured outcome: tier ``k`` dominates tier ``k - 1`` on expected held-out
control quality (mean-TKE reduction per unit of bounded actuation, with
smoothness of the action signal).

Each tier emits one mutually exclusive signature phrase set, so the judge can
read quality off the candidate text deterministically. Phrase selection and
flavor values inside the template are driven by a per-candidate RNG seed, so
two arms drawn from the same tier differ only by random seed while remaining
in the same quality band by construction.

This is instrument validation. No CFD, no RL training, and no claim about
coding agents improving control.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

QUALITY_TIERS: tuple[int, ...] = (0, 1, 2, 3, 4)

TIER_BASE_SCORE: dict[int, float] = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}

TIER_SIGNATURES: dict[int, tuple[str, ...]] = {
    0: ("constant terminal reward",),
    1: ("negligible flow-feedback weight",),
    2: ("mean kinetic-energy penalty without an effort bound",),
    3: ("bounded control_l1_weight",),
    4: ("bounded action_delta_l2_weight", "clamped bounded weights"),
}

SCENARIO_LINES: tuple[str, ...] = (
    "Re=100 cylinder-wake (blowing/suction) control scenario",
    "Re=100 Kolmogorov forced-mode control scenario",
    "two-dimensional cylinder wake at Re=100",
)

_TIER_TEMPLATES: dict[int, tuple[str, ...]] = {
    0: (
        "Reward spec: constant terminal reward of 1.0 per episode; the actuator "
        "runs at a fixed amplitude and flow state is never read.",
        "Reward spec: award constant terminal reward of 1.0 each episode; "
        "actuation is held at full amplitude; no flow observation is used.",
    ),
    1: (
        "Reward spec: negligible flow-feedback weight of 0.0001 on mean kinetic "
        "energy, with a large penalty on control effort and no bound on "
        "action-to-action changes.",
        "Reward spec: mean kinetic energy enters with a negligible "
        "flow-feedback weight of 0.0001; control effort dominates the reward "
        "and actuation changes are unbounded.",
    ),
    2: (
        "Reward spec: mean kinetic-energy penalty without an effort bound; the "
        "reward pushes velocity fluctuations toward zero at any actuation cost.",
        "Reward spec: only the mean kinetic-energy penalty without an effort "
        "bound is present, so actuation may saturate while chasing low TKE.",
    ),
    3: (
        "Reward spec: mean kinetic-energy penalty combined with a bounded "
        "control_l1_weight in the bounded range; actuation effort is priced but "
        "action-to-action changes are unconstrained.",
        "Reward spec: bounded control_l1_weight is included next to the mean "
        "kinetic-energy penalty; control effort is priced, changes between "
        "actions are not.",
    ),
    4: (
        "Reward spec: mean kinetic-energy penalty with bounded effort pricing, "
        "a bounded action_delta_l2_weight, and all weights clamped bounded "
        "weights within the bounded range; effort and action smoothness are "
        "both priced.",
        "Reward spec: bounded action_delta_l2_weight and clamped bounded "
        "weights complement the mean kinetic-energy penalty, so control effort "
        "and action smoothness are both priced within the bounded range.",
    ),
}

_FLAVOR_LINES: tuple[str, ...] = (
    "Training hint: evaluate on held-out episodes only.",
    "Training hint: keep the reward dense per control interval.",
)

_OTHER_TIER_STEMS: tuple[str, ...] = tuple(stem for tier in QUALITY_TIERS for stem in TIER_SIGNATURES[tier])


@dataclass(frozen=True)
class RewardCandidate:
    """One synthetic reward-spec text with its constructed quality tier."""

    candidate_id: str
    tier: int
    group_id: str
    seed: int
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if self.tier not in QUALITY_TIERS:
            raise ValueError(f"tier must be one of {QUALITY_TIERS}")
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("group_id must be non-empty")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be non-empty")


class RewardCandidateGenerator:
    """Generate one arm of reward candidates at a known quality tier.

    A generated arm is ``groups`` group clusters, one candidate per cluster,
    all at ``tier``. Group ids are stable across arms so the paired-delta
    harness can pair cluster ``g`` of arm A with cluster ``g`` of arm B.
    """

    def __init__(self, *, tier: int, groups: int, seed: int) -> None:
        if tier not in QUALITY_TIERS:
            raise ValueError(f"tier must be one of {QUALITY_TIERS}")
        if not isinstance(groups, int) or groups < 2:
            raise ValueError("groups must be an int of at least two clusters")
        if not isinstance(seed, int):
            raise TypeError("seed must be an int")
        self.tier = tier
        self.groups = groups
        self.seed = seed
        self._rng = random.Random(seed)

    def _render(self, phrase_seed: int) -> str:
        """Render from the seed stored on the resulting candidate."""
        rng = random.Random(phrase_seed)
        scenario = rng.choice(SCENARIO_LINES)
        template = rng.choice(_TIER_TEMPLATES[self.tier])
        flavor = rng.choice(_FLAVOR_LINES)
        return f"{template} {scenario} {flavor}"

    def generate(self) -> tuple[RewardCandidate, ...]:
        candidates = []
        for cluster_index in range(self.groups):
            phrase_seed = self._rng.randrange(0, 2**31 - 1)
            text = self._render(phrase_seed)
            if not _contains_only_own_stems(text, self.tier):
                raise AssertionError("generated text leaked a foreign tier signature")
            candidates.append(
                RewardCandidate(
                    candidate_id=f"tier{self.tier}-seed{self.seed}-g{cluster_index:02d}",
                    tier=self.tier,
                    group_id=f"cluster-{cluster_index:02d}",
                    seed=phrase_seed,
                    text=text,
                )
            )
        if len({candidate.group_id for candidate in candidates}) != self.groups:
            raise AssertionError("generated arm must have exactly one candidate per cluster")
        return tuple(candidates)


def _contains_only_own_stems(text: str, tier: int) -> bool:
    """Verify the text carries exactly its own tier signatures and no other."""
    own = set(TIER_SIGNATURES[tier])
    if not all(stem in text for stem in own):
        return False
    foreign = [stem for stem in _OTHER_TIER_STEMS if stem not in own and stem in text]
    return not foreign
