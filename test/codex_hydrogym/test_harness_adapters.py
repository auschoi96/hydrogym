"""Paired coding-agent adapters and native MLflow tracing."""

import asyncio
import inspect
import json
from types import SimpleNamespace

import mlflow
import pytest

from codex_hydrogym.genai.contracts import (
    AGENT_FEEDBACK_SCHEMA_VERSION,
    REQUIRED_PHYSICS_GATES,
    EvidenceArm,
    RunBundle,
)
from codex_hydrogym.genai.harnesses import (
    AGENT_FEEDBACK_JSON_SCHEMA,
    AGENT_FEEDBACK_TRANSPORT_SCHEMA,
    CLAUDE_DENIED_TOOLS,
    CODEX_ALLOWED_ITEM_TYPES,
    DIRECT_AGENT_FEEDBACK_RESPONSE_FORMAT,
    SHARED_SYSTEM_INSTRUCTIONS,
    ClaudeHarness,
    CodexHarness,
    DirectGatewayHarness,
    HarnessExecutionError,
    RawHarnessResponse,
    feedback_id_for_bundle,
    render_feedback_prompt,
)
from codex_hydrogym.genai.tracing import analyze_run_bundle


def _arm(arm_id, *, tke, effort, observations):
    return EvidenceArm(
        arm_id=arm_id,
        run_id=f"run_{arm_id}",
        evidence_kind="measured",
        artifact_ref=f"runs:/run_{arm_id}/evidence.json",
        artifact_sha256=("a" if observations else "b") * 64,
        context_fingerprint="c" * 64,
        controller_kind="feedback" if observations else "constant_open_loop",
        uses_observations=observations,
        mean_tke=tke,
        control_effort=effort,
        physics_gates=dict.fromkeys(REQUIRED_PHYSICS_GATES, True),
        metrics={},
    )


def _bundle():
    return RunBundle(
        bundle_id="bundle_adapter_test",
        group_id="group_adapter_test",
        task_contract_version="gate0.v1",
        task={"objective": "beat the open-loop frontier"},
        training={"budget_locked": True},
        candidate=_arm("feedback_candidate", tke=0.7, effort=0.4, observations=True),
        comparators=(_arm("constant_frontier", tke=1.0, effort=0.4, observations=False),),
        diagnostics=(),
        artifact_refs=("runs:/run_feedback_candidate/evidence.json",),
    )


def _feedback_dict(bundle):
    return {
        "schema_version": AGENT_FEEDBACK_SCHEMA_VERSION,
        "feedback_id": feedback_id_for_bundle(bundle),
        "decision": "collect_evidence",
        "diagnosis": "The causal observation-shuffle comparison is still missing.",
        "evidence": ["candidate mean_tke=0.7; constant mean_tke=1.0 at effort=0.4"],
        "falsification_test": "Shuffle observations while preserving the action distribution.",
        "claim_boundary": "This critique is not evidence of fluid improvement.",
        "estimated_cost": "cpu_gate",
        "reward_spec": None,
    }


def test_released_sdk_surfaces_match_the_adapter_contract(tmp_path):
    openai_codex = pytest.importorskip("openai_codex")
    claude_agent_sdk = pytest.importorskip("claude_agent_sdk")

    assert openai_codex.__version__ == "0.147.0"
    assert claude_agent_sdk.__version__ == "0.2.142"
    assert openai_codex.Sandbox.read_only.value == "read-only"
    assert openai_codex.ApprovalMode.deny_all.value == "deny_all"
    assert {"approval_mode", "output_schema"} <= set(inspect.signature(openai_codex.AsyncThread.run).parameters)

    codex_config = openai_codex.CodexConfig(
        cwd=str(tmp_path),
        env={"CODEX_HOME": str(tmp_path / "codex-home")},
    )
    assert codex_config.cwd == str(tmp_path)
    assert codex_config.env == {"CODEX_HOME": str(tmp_path / "codex-home")}

    claude_options = claude_agent_sdk.ClaudeAgentOptions(
        tools=[],
        allowed_tools=[],
        disallowed_tools=["*"],
        permission_mode="dontAsk",
        strict_mcp_config=True,
        setting_sources=[],
        skills=[],
        plugins=[],
        agents={},
        output_format={"type": "json_schema", "schema": AGENT_FEEDBACK_TRANSPORT_SCHEMA},
    )
    assert claude_options.tools == []
    assert claude_options.permission_mode == "dontAsk"
    assert claude_options.output_format["schema"] == AGENT_FEEDBACK_TRANSPORT_SCHEMA


def test_transport_schema_uses_databricks_supported_subset_and_local_schema_remains_strict():
    forbidden = {
        "pattern",
        "anyOf",
        "oneOf",
        "allOf",
        "prefixItems",
        "$ref",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    }

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert forbidden.isdisjoint(keys(AGENT_FEEDBACK_TRANSPORT_SCHEMA))
    assert {"pattern", "anyOf", "allOf"} <= keys(AGENT_FEEDBACK_JSON_SCHEMA)


class _CodexConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _CodexThread:
    def __init__(self, state):
        self.state = state
        self.id = "thread-1"

    async def run(self, prompt, **kwargs):
        self.state.run = (prompt, kwargs)
        items = [
            SimpleNamespace(root=SimpleNamespace(type=item_type))
            for item_type in self.state.item_types
        ]
        return SimpleNamespace(
            final_response=self.state.response,
            items=items,
            status="completed",
            error=None,
            duration_ms=12,
            usage={"input_tokens": 10},
        )


class _AsyncCodex:
    state = None

    def __init__(self, config):
        self.state = type(self).state
        self.state.config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def thread_start(self, **kwargs):
        self.state.thread_start = kwargs
        return _CodexThread(self.state)


def _codex_sdk(response, *, item_types=("reasoning", "agentMessage")):
    state = SimpleNamespace(response=response, item_types=item_types)
    _AsyncCodex.state = state
    sdk = SimpleNamespace(
        __version__="0.test",
        AsyncCodex=_AsyncCodex,
        CodexConfig=_CodexConfig,
        Sandbox=SimpleNamespace(read_only="read-only"),
        ApprovalMode=SimpleNamespace(deny_all="deny-all"),
    )
    return sdk, state


class _ClaudeOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _ResultMessage:
    def __init__(self, structured_output, *, subtype="success", is_error=False):
        self.subtype = subtype
        self.is_error = is_error
        self.structured_output = structured_output
        self.permission_denials = []
        self.deferred_tool_use = None
        self.terminal_reason = "completed"
        self.stop_reason = "end_turn"
        self.errors = None
        self.api_error_status = None
        self.duration_ms = 12
        self.duration_api_ms = 10
        self.num_turns = 1
        self.total_cost_usd = 0.01
        self.usage = {"input_tokens": 10}
        self.model_usage = {}
        self.session_id = "session-1"


class _ClaudeClient:
    state = None

    def __init__(self, *, options):
        self.state = type(self).state
        self.state.options = options

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def query(self, prompt):
        self.state.prompt = prompt

    async def receive_response(self):
        for message in self.state.messages:
            yield message
        yield self.state.final_message or _ResultMessage(self.state.structured_output)


def _claude_sdk(structured_output, *, final_message=None, messages=()):
    state = SimpleNamespace(
        structured_output=structured_output,
        final_message=final_message,
        messages=tuple(messages),
    )
    _ClaudeClient.state = state
    sdk = SimpleNamespace(
        __version__="0.test",
        ClaudeAgentOptions=_ClaudeOptions,
        ClaudeSDKClient=_ClaudeClient,
        ResultMessage=_ResultMessage,
    )
    return sdk, state


def test_adapters_receive_identical_prompt_schema_and_locked_permissions(tmp_path):
    bundle = _bundle()
    prompt = render_feedback_prompt(bundle)
    response = _feedback_dict(bundle)
    codex_sdk, codex_state = _codex_sdk(json.dumps(response))
    claude_sdk, claude_state = _claude_sdk(response)
    evidence_root = tmp_path / "evidence"
    codex_home = tmp_path / "codex-home"
    claude_config = tmp_path / "claude-config"
    evidence_root.mkdir()
    codex_home.mkdir()
    claude_config.mkdir()

    codex = CodexHarness(
        model="gpt-test",
        workspace_root=evidence_root,
        codex_home=codex_home,
        sdk_module=codex_sdk,
    )
    claude = ClaudeHarness(
        model="claude-test",
        workspace_root=evidence_root,
        claude_config_dir=claude_config,
        sdk_module=claude_sdk,
    )
    codex_result, claude_result = asyncio.run(_run_both(codex, claude, prompt))

    assert codex_state.run[0] == claude_state.prompt == prompt
    assert codex_state.run[1]["output_schema"] == AGENT_FEEDBACK_TRANSPORT_SCHEMA
    assert codex_state.thread_start["approval_mode"] == "deny-all"
    assert codex_state.thread_start["base_instructions"] == SHARED_SYSTEM_INSTRUCTIONS
    assert codex_state.thread_start["ephemeral"] is True
    assert claude_state.options.kwargs["system_prompt"] == SHARED_SYSTEM_INSTRUCTIONS
    assert claude_state.options.kwargs["tools"] == []
    assert claude_state.options.kwargs["disallowed_tools"] == list(CLAUDE_DENIED_TOOLS)
    assert claude_state.options.kwargs["permission_mode"] == "dontAsk"
    assert claude_state.options.kwargs["setting_sources"] == []
    assert claude_state.options.kwargs["strict_mcp_config"] is True
    assert claude_state.options.kwargs["output_format"]["schema"] == AGENT_FEEDBACK_TRANSPORT_SCHEMA
    assert json.loads(codex_result.text) == json.loads(claude_result.text) == response
    assert codex_result.metadata["sdk_version"] == claude_result.metadata["sdk_version"] == "0.test"
    assert set(codex_result.metadata["item_types"]) <= CODEX_ALLOWED_ITEM_TYPES
    assert codex_result.metadata["tool_activity_count"] == 0
    assert claude_result.metadata["tool_activity_count"] == 0


@pytest.mark.parametrize(
    "item_type",
    [
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "dynamicToolCall",
        "collabAgentToolCall",
        "subAgentActivity",
        "webSearch",
        "imageView",
        "sleep",
        "imageGeneration",
        "unknownFutureItem",
    ],
)
def test_codex_fails_closed_on_every_non_response_item(tmp_path, item_type):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    sdk, _state = _codex_sdk("{}", item_types=("agentMessage", item_type))
    harness = CodexHarness(
        model="gpt-test",
        workspace_root=tmp_path,
        codex_home=codex_home,
        sdk_module=sdk,
    )

    with pytest.raises(HarnessExecutionError, match="forbidden non-response activity"):
        asyncio.run(harness.generate("analyze this bundle"))


@pytest.mark.parametrize("item_types", [(), ("reasoning",)])
def test_codex_requires_auditable_agent_response_item(tmp_path, item_types):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    sdk, _state = _codex_sdk("{}", item_types=item_types)
    harness = CodexHarness(
        model="gpt-test",
        workspace_root=tmp_path,
        codex_home=codex_home,
        sdk_module=sdk,
    )

    with pytest.raises(HarnessExecutionError, match="no auditable response items|no agent response item"):
        asyncio.run(harness.generate("analyze this bundle"))


def test_codex_allows_non_actionable_input_plan_and_reasoning_items(tmp_path):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    sdk, _state = _codex_sdk(
        "{}",
        item_types=("userMessage", "plan", "reasoning", "agentMessage"),
    )
    harness = CodexHarness(
        model="gpt-test",
        workspace_root=tmp_path,
        codex_home=codex_home,
        sdk_module=sdk,
    )

    response = asyncio.run(harness.generate("analyze this bundle"))

    assert response.metadata["status"] == "completed"
    assert response.metadata["item_type_counts"] == {
        "agentMessage": 1,
        "plan": 1,
        "reasoning": 1,
        "userMessage": 1,
    }


def test_claude_failure_exposes_released_api_diagnostics(tmp_path):
    claude_config = tmp_path / "claude-config"
    claude_config.mkdir()
    final = _ResultMessage(None, subtype="error_during_execution", is_error=True)
    final.terminal_reason = "api_error"
    final.stop_reason = "refusal"
    final.errors = ["upstream request failed"]
    final.api_error_status = 503
    sdk, _state = _claude_sdk(None, final_message=final)
    harness = ClaudeHarness(
        model="claude-test",
        workspace_root=tmp_path,
        claude_config_dir=claude_config,
        sdk_module=sdk,
    )

    with pytest.raises(HarnessExecutionError) as raised:
        asyncio.run(harness.generate("analyze this bundle"))

    message = str(raised.value)
    assert '"api_error_status":503' in message
    assert '"errors":["upstream request failed"]' in message
    assert '"stop_reason":"refusal"' in message
    assert '"terminal_reason":"api_error"' in message


def test_claude_allows_only_its_matched_structured_output_transport(tmp_path):
    real_sdk = pytest.importorskip("claude_agent_sdk")
    claude_config = tmp_path / "claude-config"
    claude_config.mkdir()
    response = _feedback_dict(_bundle())
    tool_id = "structured-output-1"
    messages = (
        real_sdk.AssistantMessage(
            content=[real_sdk.ToolUseBlock(id=tool_id, name="StructuredOutput", input=response)],
            model="claude-test",
        ),
        real_sdk.UserMessage(
            content=[real_sdk.ToolResultBlock(tool_use_id=tool_id, content="accepted")],
        ),
    )
    sdk, _state = _claude_sdk(response, messages=messages)
    harness = ClaudeHarness(
        model="claude-test",
        workspace_root=tmp_path,
        claude_config_dir=claude_config,
        sdk_module=sdk,
    )

    result = asyncio.run(harness.generate("analyze this bundle"))

    assert result.metadata["schema_transport_activity_count"] == 2
    assert result.metadata["tool_activity_count"] == 0


def test_claude_rejects_external_tool_activity(tmp_path):
    real_sdk = pytest.importorskip("claude_agent_sdk")
    claude_config = tmp_path / "claude-config"
    claude_config.mkdir()
    response = _feedback_dict(_bundle())
    message = real_sdk.AssistantMessage(
        content=[real_sdk.ToolUseBlock(id="tool-1", name="Bash", input={"command": "pwd"})],
        model="claude-test",
    )
    sdk, _state = _claude_sdk(response, messages=(message,))
    harness = ClaudeHarness(
        model="claude-test",
        workspace_root=tmp_path,
        claude_config_dir=claude_config,
        sdk_module=sdk,
    )

    with pytest.raises(HarnessExecutionError, match="ToolUseBlock:Bash"):
        asyncio.run(harness.generate("analyze this bundle"))


async def _run_both(codex, claude, prompt):
    return await asyncio.gather(codex.generate(prompt), claude.generate(prompt))


class _StaticHarness:
    arm = "codex"
    adapter_id = "codex_sdk"
    model = "gpt-test"

    def __init__(self, response):
        self.response = response

    async def generate(self, _prompt):
        return RawHarnessResponse(
            arm=self.arm,
            adapter_id=self.adapter_id,
            model=self.model,
            text=json.dumps(self.response),
            metadata={},
        )


class _CrossWiredHarness(_StaticHarness):
    async def generate(self, _prompt):
        return RawHarnessResponse(
            arm="claude",
            adapter_id="claude_agent_sdk",
            model="wrong-model",
            text=json.dumps(self.response),
            metadata={},
        )


class _AdapterCrossWiredHarness(_StaticHarness):
    async def generate(self, _prompt):
        return RawHarnessResponse(
            arm=self.arm,
            adapter_id="codex_direct",
            model=self.model,
            text=json.dumps(self.response),
            metadata={},
        )


def test_trace_rejects_cross_wired_harness_provenance(tmp_path):
    bundle = _bundle()
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("harness-cross-wire-test")
    try:
        with pytest.raises(HarnessExecutionError, match="returned arm 'claude', expected 'codex'"):
            asyncio.run(
                analyze_run_bundle(
                    bundle=bundle,
                    harness=_CrossWiredHarness(_feedback_dict(bundle)),
                    mlflow_module=mlflow,
                )
            )
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)


def test_trace_rejects_cross_wired_adapter_provenance(tmp_path):
    bundle = _bundle()
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("harness-adapter-cross-wire-test")
    try:
        with pytest.raises(HarnessExecutionError, match="returned adapter_id 'codex_direct'"):
            asyncio.run(
                analyze_run_bundle(
                    bundle=bundle,
                    harness=_AdapterCrossWiredHarness(_feedback_dict(bundle)),
                    mlflow_module=mlflow,
                )
            )
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)


def test_native_mlflow_trace_has_agent_task_and_guardrail_spans(tmp_path):
    bundle = _bundle()
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = mlflow.set_experiment("harness-trace-test")
    try:
        result = asyncio.run(
            analyze_run_bundle(
                bundle=bundle,
                harness=_StaticHarness(_feedback_dict(bundle)),
                mlflow_module=mlflow,
            )
        )
        traces = mlflow.search_traces(
            locations=[experiment.experiment_id],
            return_type="list",
        )
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)

    assert result.trace_id
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert {(span.name, span.span_type) for span in spans} == {
        ("hydrogym_feedback_agent", "AGENT"),
        ("harness_call", "TASK"),
        ("contract_validation", "GUARDRAIL"),
    }
    root = next(span for span in spans if span.name == "hydrogym_feedback_agent")
    children = [span for span in spans if span.name != root.name]
    assert all(span.parent_id == root.span_id for span in children)
    assert traces[0].info.tags["codex_hydrogym.harness_arm"] == "codex"
    assert traces[0].info.tags["codex_hydrogym.harness_adapter_id"] == "codex_sdk"
    assert traces[0].info.tags["codex_hydrogym.evidence_kind"] == "measured"


def test_native_mlflow_trace_records_contract_validation_errors(tmp_path):
    bundle = _bundle()
    malformed = _feedback_dict(bundle)
    malformed["feedback_id"] = "analysis_wrong_bundle"
    old_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = mlflow.set_experiment("harness-error-trace-test")
    try:
        with pytest.raises(ValueError, match="feedback_id must equal"):
            asyncio.run(
                analyze_run_bundle(
                    bundle=bundle,
                    harness=_StaticHarness(malformed),
                    mlflow_module=mlflow,
                )
            )
        traces = mlflow.search_traces(
            locations=[experiment.experiment_id],
            return_type="list",
        )
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)

    assert len(traces) == 1
    assert traces[0].info.status.value == "ERROR"
    spans = {span.name: span for span in traces[0].data.spans}
    assert set(spans) == {"hydrogym_feedback_agent", "harness_call", "contract_validation"}
    assert spans["contract_validation"].status.status_code.name == "ERROR"
    assert "feedback_id must equal" in (spans["contract_validation"].status.description or "")


class _Gateway:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_direct_gateway_harness_uses_same_prompt_and_strict_transport_schema(tmp_path):
    bundle = _bundle()
    prompt = render_feedback_prompt(bundle)
    gateway = _Gateway(
        SimpleNamespace(
            text=json.dumps(_feedback_dict(bundle)),
            model="provider-reported-model",
            request_id="request-1",
            finish_reason="stop",
            usage={"input_tokens": 20, "output_tokens": 40},
        )
    )
    harness = DirectGatewayHarness(
        arm="codex",
        model="system.ai.gpt-test",
        gateway=gateway,
        accepted_reported_models={"provider-reported-model"},
    )

    result = asyncio.run(harness.generate(prompt))

    assert result.arm == "codex"
    assert result.adapter_id == "codex_direct"
    assert result.model == "system.ai.gpt-test"
    assert gateway.kwargs["messages"] == [
        {"role": "system", "content": SHARED_SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": prompt},
    ]
    assert gateway.kwargs["response_format"] == DIRECT_AGENT_FEEDBACK_RESPONSE_FORMAT
    assert gateway.kwargs["request_tags"]["adapter_id"] == "codex_direct"
    assert result.metadata["reported_model"] == "provider-reported-model"
    assert result.metadata["finish_reason"] == "stop"
    assert result.metadata["tool_activity_count"] == 0


def test_direct_gateway_harness_accepts_exact_configured_model_by_default():
    bundle = _bundle()
    configured_model = "system.ai.gpt-test"
    gateway = _Gateway(
        SimpleNamespace(
            text=json.dumps(_feedback_dict(bundle)),
            model=configured_model,
            request_id="request-exact",
            finish_reason="stop",
            usage={},
        )
    )
    harness = DirectGatewayHarness(
        arm="codex",
        model=configured_model,
        gateway=gateway,
    )

    result = asyncio.run(harness.generate(render_feedback_prompt(bundle)))

    assert result.model == configured_model
    assert result.metadata["reported_model"] == configured_model
    assert harness.accepted_reported_models == frozenset({configured_model})


@pytest.mark.parametrize("reported_model", [None, "", " ", "system.ai.gpt_test", "gpt-test"])
def test_direct_gateway_harness_rejects_missing_or_unexpected_reported_model(reported_model):
    bundle = _bundle()
    gateway = _Gateway(
        SimpleNamespace(
            text=json.dumps(_feedback_dict(bundle)),
            model=reported_model,
            request_id="request-mismatch",
            finish_reason="stop",
            usage={},
        )
    )
    harness = DirectGatewayHarness(
        arm="codex",
        model="system.ai.gpt-test",
        gateway=gateway,
    )

    with pytest.raises(HarnessExecutionError, match="model identifier"):
        asyncio.run(harness.generate(render_feedback_prompt(bundle)))
