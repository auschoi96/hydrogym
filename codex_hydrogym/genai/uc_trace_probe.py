"""Emit one synthetic, clearly marked trace through the real harness tracing path."""

from __future__ import annotations

import argparse
import asyncio
import json

import mlflow

from codex_hydrogym.genai.contracts import AGENT_FEEDBACK_SCHEMA_VERSION, AgentFeedback, EvidenceArm, RunBundle
from codex_hydrogym.genai.harnesses import RawHarnessResponse, feedback_id_for_bundle
from codex_hydrogym.genai.tracing import analyze_run_bundle, configure_uc_trace_experiment

PROBE_ID = "uc_otel_throwaway_probe_20260829"
PROBE_EXPERIMENT = "/Shared/codex_hydrogym_uctrace_probe_v3"
PROBE_TABLE_PREFIX = "codex_hydrogym_uctrace_probe_20260829_v3"
_PROBE_DIGEST = "0" * 64


def _arm(arm_id: str, observations: bool) -> EvidenceArm:
    return EvidenceArm(
        arm_id=arm_id,
        run_id=f"{PROBE_ID}_{arm_id}",
        evidence_kind="synthetic_contract",
        artifact_ref=f"synthetic://{PROBE_ID}/{arm_id}",
        artifact_sha256=_PROBE_DIGEST,
        context_fingerprint=_PROBE_DIGEST,
        controller_kind="feedback" if observations else "constant_open_loop",
        uses_observations=observations,
        mean_tke=1.0,
        control_effort=0.1,
        physics_gates={"probe_only": True},
        metrics={"probe_only": 1.0},
    )


BUNDLE = RunBundle(
    bundle_id=PROBE_ID,
    group_id=PROBE_ID,
    task_contract_version="uc_otel_probe.v1",
    task={"probe_only": True, "claim_allowed": False},
    training={"probe_only": True},
    candidate=_arm("candidate", True),
    comparators=(_arm("constant", False),),
    diagnostics=("throwaway UC OTel delivery probe; not scientific evidence",),
    artifact_refs=(f"synthetic://{PROBE_ID}/candidate",),
)


class _StaticProbeHarness:
    arm = "codex"
    adapter_id = "codex_sdk"
    model = "uc-otel-probe-no-model-call"

    async def generate(self, _prompt: str) -> RawHarnessResponse:
        feedback = AgentFeedback(
            schema_version=AGENT_FEEDBACK_SCHEMA_VERSION,
            feedback_id=feedback_id_for_bundle(BUNDLE),
            decision="collect_evidence",
            diagnosis="Throwaway OpenTelemetry export probe.",
            evidence=("synthetic probe only",),
            falsification_test="Query the Unity Catalog spans table by this probe bundle ID.",
            claim_boundary="This is transport verification only, not fluid-control evidence.",
            estimated_cost="none",
            reward_spec=None,
        )
        return RawHarnessResponse(
            arm=self.arm,
            adapter_id=self.adapter_id,
            model=self.model,
            text=json.dumps(feedback.as_dict()),
            metadata={"probe_id": PROBE_ID, "no_model_call": True},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="dais-demo")
    args = parser.parse_args()
    mlflow.set_tracking_uri(f"databricks://{args.profile}")
    experiment = configure_uc_trace_experiment(
        experiment_name=PROBE_EXPERIMENT,
        table_prefix=PROBE_TABLE_PREFIX,
    )
    analysis = asyncio.run(analyze_run_bundle(bundle=BUNDLE, harness=_StaticProbeHarness()))
    print(
        json.dumps(
            {
                "experiment_id": experiment.experiment_id,
                "experiment_name": experiment.name,
                "probe_id": PROBE_ID,
                "trace_id": str(analysis.trace_id),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
