"""Opt-in real-workspace executor for the MemAlign H1 pipeline.

This script is the ONLY place that contacts a Databricks workspace.  The unit tests
never run it.  It requires an explicit ``--profile`` (valid profiles for this project
are ``dais-demo-sync`` and ``solution-builder-test``; the user's default
``gtm-ai-agent`` profile and databricks.yml are never touched) and an experiment id.
Stages:

- ``preflight`` -- judge-name preflight for all three repository judge names;
- ``queue`` -- select measured coding-agent traces and emit the locked fold manifest
  (writes the frozen manifest JSON, digest included, to ``--output``);
- ``enroll`` -- tag manifest traces with their locked folds (refuses a digest change);
- ``harness`` -- align on the locked train fold and score held-out agreement; reports
  INCONCLUSIVE until the protocol's label counts exist (50 train + 8 held-out
  adjudicated critic_quality labels).

Run from the repository root, e.g.::

    python -m codex_hydrogym.memalign_h1.run_h1 \\
        --profile dais-demo-sync --experiment-id 103455306564903 --stage preflight
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from codex_hydrogym.memalign_h1 import (
    REQUIRED_HELDOUT_LABEL_COUNT,
    REQUIRED_TOTAL_LABEL_COUNT,
    REQUIRED_TRAIN_LABEL_COUNT,
)
from codex_hydrogym.memalign_h1.preflight import preflight_all_known_judges

KNOWN_PROFILES = ("dais-demo-sync", "solution-builder-test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", required=True, help=f"Databricks CLI profile; one of {KNOWN_PROFILES}")
    parser.add_argument("--experiment-id", required=True, help="MLflow experiment id that holds the traces")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("preflight", "queue", "enroll", "harness"),
    )
    parser.add_argument("--output", default="memalign_h1_manifest.json", help="frozen manifest JSON path")
    parser.add_argument(
        "--test-group-count",
        type=int,
        default=4,
        help="groups held out (must be 4: the frozen df=3 t-critical)",
    )
    parser.add_argument("--max-trace-results", type=int, default=2000)
    parser.add_argument("--print-counts-only", action="store_true", help="report required label counts and exit")
    return parser


def _connect(profile: str) -> tuple[str, str]:
    """Return (host, token) for the explicit profile without touching user defaults."""
    from databricks.sdk.core import Config

    config = Config(profile=profile)
    host = (config.host or "").strip().rstrip("/")
    if not host:
        raise RuntimeError(f"profile {profile!r} produced no workspace host")
    token = (config.token or "").strip()
    if not token:
        raise RuntimeError(f"profile {profile!r} produced no token; use a databricks-cli (or OAuth) profile")
    return host, token


def _mlflow_env(profile: str):
    """Point the current process at the explicit profile's workspace."""

    import os

    host, token = _connect(profile)
    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token


def _stage_preflight(args) -> dict[str, Any]:
    import mlflow

    mlflow.set_tracking_uri("databricks")
    traces = mlflow.search_traces(
        experiment_ids=[args.experiment_id],
        max_results=args.max_trace_results,
        return_type="list",
        include_spans=False,
    )
    return preflight_all_known_judges(traces=traces)


def _stage_queue(args) -> dict[str, Any]:
    import mlflow

    from codex_hydrogym.memalign_h1.queue import build_locked_fold_manifest, select_coding_agent_traces

    mlflow.set_tracking_uri("databricks")
    selection = select_coding_agent_traces(
        experiment_id=args.experiment_id,
        mlflow_module=mlflow,
        max_results=args.max_trace_results,
    )
    manifest = build_locked_fold_manifest(
        records=selection["candidates"],
        test_group_count=args.test_group_count,
    )
    output = Path(args.output)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"selection": selection, "manifest_path": str(output), "manifest": manifest}


def _stage_enroll(args) -> dict[str, Any]:
    import mlflow

    from codex_hydrogym.memalign_h1.queue import enroll_locked_folds

    mlflow.set_tracking_uri("databricks")
    manifest = json.loads(Path(args.output).read_text(encoding="utf-8"))
    return enroll_locked_folds(manifest=manifest, mlflow_module=mlflow, require_digest=manifest["digest"])


def _stage_harness(args) -> dict[str, Any]:
    """Align on the locked train fold and measure held-out agreement.

    This stage requires the locked manifest, an enrolled experiment, adjudicated
    labels, and the base critic_quality judge.  It reports the required label counts
    and returns INCONCLUSIVE until they exist.
    """
    import mlflow

    from codex_hydrogym.genai.optimization import register_base_judge
    from codex_hydrogym.genai.portfolio import ModelPortfolio
    from codex_hydrogym.memalign_h1.harness import evaluate_h1

    mlflow.set_tracking_uri("databricks")
    manifest = json.loads(Path(args.output).read_text(encoding="utf-8"))
    train_rows = [row for row in manifest["rows"] if row["fold"] == "train"]
    heldout_rows = [row for row in manifest["rows"] if row["fold"] == "heldout"]
    if len(train_rows) < REQUIRED_TRAIN_LABEL_COUNT or len(heldout_rows) < REQUIRED_HELDOUT_LABEL_COUNT:
        return {
            "stage": "harness",
            "blocked": True,
            "reason": "not enough enrolled traces for the frozen H1 protocol",
            "required_train_labels": REQUIRED_TRAIN_LABEL_COUNT,
            "required_heldout_labels": REQUIRED_HELDOUT_LABEL_COUNT,
            "required_total_labels": REQUIRED_TOTAL_LABEL_COUNT,
            "available_train_traces": len(train_rows),
            "available_heldout_traces": len(heldout_rows),
        }
    raise RuntimeError(
        "harness execution requires adjudicated human labels and the registered base judge; see MEMALIGN_H1_PROTOCOL.md"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.profile not in KNOWN_PROFILES:
        raise SystemExit(
            f"--profile must be one of {KNOWN_PROFILES} (the default gtm-ai-agent "
            "profile is never used; databricks.yml is never modified)"
        )
    if args.test_group_count != 4:
        raise SystemExit("--test-group-count must be 4: the frozen df=3 t-critical fixes four held-out groups")
    if args.print_counts_only:
        print(
            json.dumps(
                {
                    "required_train_labels": REQUIRED_TRAIN_LABEL_COUNT,
                    "required_heldout_labels": REQUIRED_HELDOUT_LABEL_COUNT,
                    "required_total_labels": REQUIRED_TOTAL_LABEL_COUNT,
                    "currently_available": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    _mlflow_env(args.profile)
    handlers = {
        "preflight": _stage_preflight,
        "queue": _stage_queue,
        "enroll": _stage_enroll,
        "harness": _stage_harness,
    }
    result = handlers[args.stage](args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
