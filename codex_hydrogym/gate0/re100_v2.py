"""Immutable runner for the explicitly relabeled moderate-Re Gate 0 v2.

This protocol changes the scientific claim from the failed Re=200 v1 gate to
Re=100 and uses the resolution pair supported by the development-only design
probe.  It deliberately reuses the v1 evaluator and preserves all causal,
effort, numerical, and convergence thresholds.  It performs no RL.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Sequence

from codex_hydrogym.gate0.cli import (
    _default_output,
    _implementation_manifest,
    _run_stage,
)
from codex_hydrogym.gate0.protocol import Gate0Config


RE100_V2_PROTOCOL_ID = "offset_phase_fp64_re100_gate0_v2"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_SOURCE = "codex_hydrogym/gate0/re100_v2.py"


def _digest(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def re100_v2_config() -> Gate0Config:
    """Return the frozen Re=100 protocol while retaining every v1 proof limit."""
    return replace(
        Gate0Config(),
        protocol_id=RE100_V2_PROTOCOL_ID,
        reynolds_number=100.0,
        grid_size=(64, 64),
        spatial_refinement_grid_size=(96, 96),
    )


def _v2_implementation_manifest() -> tuple[dict[str, str], str]:
    implementation_files, _ = _implementation_manifest()
    implementation_files[_CONFIG_SOURCE] = hashlib.sha256(
        (_REPOSITORY_ROOT / _CONFIG_SOURCE).read_bytes()
    ).hexdigest()
    return implementation_files, _digest(implementation_files)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("freeze", "lock", "primary", "convergence", "all"),
        default="freeze",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    config = re100_v2_config()
    implementation_files, implementation_digest = _v2_implementation_manifest()
    output = args.output_dir or _default_output(config, implementation_digest)
    succeeded = _run_stage(
        args.stage,
        output,
        config,
        implementation_files,
        implementation_digest,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "protocol_fingerprint": config.fingerprint,
                "implementation_digest": implementation_digest,
                "protocol_id": config.protocol_id,
                "reynolds_number": config.reynolds_number,
                "requested_stage": args.stage,
                "stage_succeeded": succeeded,
                "final_claim_requires_convergence": True,
                "rl_training_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0 if succeeded else 2


if __name__ == "__main__":
    raise SystemExit(main())
