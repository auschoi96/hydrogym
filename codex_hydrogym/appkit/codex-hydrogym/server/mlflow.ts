import type { WorkspaceClient } from '@databricks/appkit';
import { z } from 'zod';
import type {
  FeedbackResponse,
  ReviewAssessment,
  ReviewQueueResponse,
  ReviewTrace,
  TrainingRun,
  TrainingRunsResponse,
} from '../shared/contracts.js';

export const PROJECT_LABEL = 'codex_hydrogym';
export const FEEDBACK_NAME = 'critic_quality' as const;
const REVIEW_STATE_TAG = `${PROJECT_LABEL}.critic_review_state`;
const BUNDLE_ID_TAG = `${PROJECT_LABEL}.bundle_id`;
const GROUP_ID_TAG = `${PROJECT_LABEL}.group_id`;
const HARNESS_ARM_TAG = `${PROJECT_LABEL}.harness_arm`;
const EVIDENCE_KIND_TAG = `${PROJECT_LABEL}.evidence_kind`;
const EVIDENCE_DIGEST_TAG = `${PROJECT_LABEL}.evidence_digest`;
const CRITIC_FOLD_TAG = `${PROJECT_LABEL}.critic_fold`;
const SHA256 = /^[0-9a-f]{64}$/;

type HttpMethod = 'GET' | 'POST' | 'PATCH';

export interface JsonRequester {
  request(path: string, method: HttpMethod, payload?: unknown): Promise<unknown>;
}

const assessmentSchema = z
  .object({
    assessment_id: z.string().optional(),
    assessment_name: z.string().optional(),
    feedback: z.object({ value: z.unknown().optional() }).passthrough().optional(),
    rationale: z.string().optional(),
    source: z
      .object({
        source_type: z.string().optional(),
        source_id: z.string().optional(),
      })
      .passthrough()
      .optional(),
    create_time: z.string().optional(),
    valid: z.boolean().optional(),
  })
  .passthrough();

const traceSchema = z
  .object({
    trace_id: z.string(),
    request_preview: z.string().optional(),
    response_preview: z.string().optional(),
    request_time: z.union([z.string(), z.number()]).optional(),
    timestamp_ms: z.number().optional(),
    tags: z.record(z.string(), z.string()).optional(),
    assessments: z.array(assessmentSchema).optional(),
  })
  .passthrough();

const traceSearchSchema = z
  .object({
    traces: z.array(traceSchema).optional(),
    trace_infos: z.array(traceSchema).optional(),
  })
  .passthrough();

const runSchema = z
  .object({
    info: z
      .object({
        run_id: z.string(),
        status: z.string().optional(),
        start_time: z.number().optional(),
        artifact_uri: z.string().optional(),
      })
      .passthrough(),
    data: z
      .object({
        metrics: z
          .array(
            z
              .object({
                key: z.string(),
                value: z.number(),
              })
              .passthrough()
          )
          .optional(),
        tags: z
          .array(
            z
              .object({
                key: z.string(),
                value: z.string(),
              })
              .passthrough()
          )
          .optional(),
      })
      .passthrough()
      .optional(),
  })
  .passthrough();

const runSearchSchema = z.object({ runs: z.array(runSchema).optional() }).passthrough();

function safeJson(value: string | undefined): unknown {
  if (value === undefined) return null;
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return value;
  }
}

function timestampMs(value: string | number | undefined, legacy: number | undefined): number | null {
  if (legacy !== undefined) return legacy;
  if (typeof value === 'number') return value;
  if (typeof value === 'string') {
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function mapAssessment(raw: z.infer<typeof assessmentSchema>): ReviewAssessment {
  return {
    assessmentId: raw.assessment_id ?? null,
    name: raw.assessment_name ?? 'unknown',
    value: raw.feedback?.value ?? null,
    rationale: raw.rationale ?? null,
    sourceType: raw.source?.source_type ?? 'SOURCE_TYPE_UNSPECIFIED',
    sourceId: raw.source?.source_id ?? null,
    createdAt: raw.create_time ?? null,
    valid: raw.valid ?? true,
  };
}

function mapTrace(raw: z.infer<typeof traceSchema>): ReviewTrace {
  const tags = raw.tags ?? {};
  const harnessArm = tags[HARNESS_ARM_TAG];
  const criticFold = tags[CRITIC_FOLD_TAG];
  if ((harnessArm !== 'codex' && harnessArm !== 'claude') || (criticFold !== 'train' && criticFold !== 'test')) {
    throw new Error('review trace lost its validated harness arm or critic fold');
  }
  return {
    traceId: raw.trace_id,
    request: safeJson(raw.request_preview),
    response: safeJson(raw.response_preview),
    timestampMs: timestampMs(raw.request_time, raw.timestamp_ms),
    reviewState: tags[REVIEW_STATE_TAG] ?? 'unknown',
    bundleId: tags[BUNDLE_ID_TAG] ?? '',
    groupId: tags[GROUP_ID_TAG] ?? '',
    harnessArm,
    criticFold,
    evidenceDigest: tags[EVIDENCE_DIGEST_TAG] ?? '',
    assessments: (raw.assessments ?? []).map(mapAssessment),
  };
}

function isMeasuredReviewTrace(raw: z.infer<typeof traceSchema>): boolean {
  const tags = raw.tags ?? {};
  return (
    tags[EVIDENCE_KIND_TAG] === 'measured' &&
    (tags[CRITIC_FOLD_TAG] === 'train' || tags[CRITIC_FOLD_TAG] === 'test') &&
    (tags[HARNESS_ARM_TAG] === 'codex' || tags[HARNESS_ARM_TAG] === 'claude') &&
    Boolean(tags[BUNDLE_ID_TAG]?.trim()) &&
    Boolean(tags[GROUP_ID_TAG]?.trim()) &&
    SHA256.test(tags[EVIDENCE_DIGEST_TAG] ?? '')
  );
}

function traceSearchPayload(experimentId: string, state: string, maxResults: number) {
  return {
    locations: [
      {
        type: 'MLFLOW_EXPERIMENT',
        mlflow_experiment: { experiment_id: experimentId },
      },
    ],
    filter: `tags.\`${REVIEW_STATE_TAG}\` = '${state}'`,
    max_results: maxResults,
    order_by: ['attributes.timestamp_ms DESC'],
  };
}

export function workspaceRequester(client: WorkspaceClient): JsonRequester {
  return {
    async request(path, method, payload) {
      return client.apiClient.request({
        path,
        method,
        headers: new Headers({ 'Content-Type': 'application/json' }),
        raw: false,
        payload,
      });
    },
  };
}

export async function searchReviewTraces(
  requester: JsonRequester,
  experimentId: string,
  maxResults = 50
): Promise<ReviewQueueResponse> {
  const states = ['pending_adjudication', 'adjudicated'] as const;
  const responses = await Promise.all(
    states.map((state) =>
      requester.request('/api/3.0/mlflow/traces/search', 'POST', traceSearchPayload(experimentId, state, maxResults))
    )
  );

  const byId = new Map<string, ReviewTrace>();
  for (const response of responses) {
    const parsed = traceSearchSchema.parse(response);
    for (const trace of parsed.traces ?? parsed.trace_infos ?? []) {
      if (!isMeasuredReviewTrace(trace)) continue;
      const mapped = mapTrace(trace);
      byId.set(mapped.traceId, mapped);
    }
  }

  const traces = [...byId.values()]
    .sort((left, right) => (right.timestampMs ?? 0) - (left.timestampMs ?? 0))
    .slice(0, maxResults);
  const humanLabels = traces.reduce(
    (count, trace) =>
      count +
      trace.assessments.filter(
        (assessment) => assessment.valid && assessment.name === FEEDBACK_NAME && assessment.sourceType === 'HUMAN'
      ).length,
    0
  );

  return {
    experimentId,
    traces,
    summary: {
      total: traces.length,
      pending: traces.filter((trace) => trace.reviewState === 'pending_adjudication').length,
      labeledTraces: traces.filter((trace) =>
        trace.assessments.some(
          (assessment) => assessment.valid && assessment.name === FEEDBACK_NAME && assessment.sourceType === 'HUMAN'
        )
      ).length,
      humanLabels,
    },
    fetchedAt: new Date().toISOString(),
  };
}

export async function createHumanFeedback(
  requester: JsonRequester,
  traceId: string,
  score: number,
  rationale: string,
  reviewer: string
): Promise<FeedbackResponse> {
  const now = new Date().toISOString();
  const assessment = {
    assessment_name: FEEDBACK_NAME,
    trace_id: traceId,
    source: { source_type: 'HUMAN', source_id: reviewer },
    create_time: now,
    last_update_time: now,
    feedback: { value: score },
    rationale,
    metadata: { project: PROJECT_LABEL, adjudicator: reviewer, label_role: 'consensus' },
    valid: true,
  };
  const encodedTraceId = encodeURIComponent(traceId);

  await requester.request(`/api/3.0/mlflow/traces/${encodedTraceId}/assessments`, 'POST', { assessment });

  let reviewStateUpdated = true;
  try {
    await requester.request(`/api/2.0/mlflow/traces/${encodedTraceId}/tags`, 'PATCH', {
      key: REVIEW_STATE_TAG,
      value: 'adjudicated',
    });
  } catch {
    reviewStateUpdated = false;
  }

  return {
    traceId,
    reviewer,
    assessmentName: FEEDBACK_NAME,
    assessmentCreated: true,
    reviewStateUpdated,
  };
}

function mapRun(raw: z.infer<typeof runSchema>): TrainingRun {
  const metrics = new Map((raw.data?.metrics ?? []).map((entry) => [entry.key, entry.value]));
  const tags = new Map((raw.data?.tags ?? []).map((entry) => [entry.key, entry.value]));
  const completedUpdates = tags.get(`${PROJECT_LABEL}.completed_updates`);
  const trainingPhysicsMetric = metrics.get('physics/all_passed');
  const heldoutPhysicsMetric = metrics.get('heldout/physics_all_passed');
  const alignmentStage = tags.get(`${PROJECT_LABEL}.alignment_stage`);

  return {
    runId: raw.info.run_id,
    runName: tags.get('mlflow.runName') ?? null,
    status: raw.info.status ?? 'UNKNOWN',
    startedAt: raw.info.start_time ? new Date(raw.info.start_time).toISOString() : null,
    artifactUri: raw.info.artifact_uri ?? null,
    trainingMeanTke: metrics.get('train/mean_tke') ?? null,
    trainingControlL1: metrics.get('train/control_l1') ?? null,
    trainingPhysicsPassed: trainingPhysicsMetric === undefined ? null : trainingPhysicsMetric === 1,
    heldoutMeanTke: metrics.get('heldout/mean_tke') ?? null,
    heldoutControlL1: metrics.get('heldout/control_l1') ?? null,
    heldoutRewardTotal: metrics.get('heldout/reward_total') ?? null,
    heldoutPhysicsPassed: heldoutPhysicsMetric === undefined ? null : heldoutPhysicsMetric === 1,
    heldoutEvidenceDigest: tags.get(`${PROJECT_LABEL}.heldout_evidence_digest`) ?? null,
    completedUpdates:
      completedUpdates === undefined || !Number.isFinite(Number(completedUpdates)) ? null : Number(completedUpdates),
    registeredModel: tags.get(`${PROJECT_LABEL}.registered_model_name`) ?? null,
    modelVersion: tags.get(`${PROJECT_LABEL}.registered_model_version`) ?? null,
    modelAlias: tags.get(`${PROJECT_LABEL}.model_alias`) ?? null,
    alignmentStage: alignmentStage === 'baseline' || alignmentStage === 'aligned' ? alignmentStage : null,
    promptUri: tags.get(`${PROJECT_LABEL}.prompt_uri`) ?? null,
    evaluationContextFingerprint: tags.get(`${PROJECT_LABEL}.evaluation_context_fingerprint`) ?? null,
    frozenTrainingFingerprint: tags.get(`${PROJECT_LABEL}.frozen_training_fingerprint`) ?? null,
    rewardApprovalDigest: tags.get(`${PROJECT_LABEL}.reward_approval_digest`) ?? null,
    rewardCompiledDigest: tags.get(`${PROJECT_LABEL}.reward_compiled_digest`) ?? null,
  };
}

export async function searchTrainingRuns(
  requester: JsonRequester,
  experimentId: string,
  maxResults = 20
): Promise<TrainingRunsResponse> {
  const response = await requester.request('/api/2.0/mlflow/runs/search', 'POST', {
    experiment_ids: [experimentId],
    filter: `tags.\`${PROJECT_LABEL}.training_backend\` = 'jax_ppo'`,
    run_view_type: 'ACTIVE_ONLY',
    max_results: maxResults,
    order_by: ['attributes.start_time DESC'],
  });
  const parsed = runSearchSchema.parse(response);

  return {
    experimentId,
    runs: (parsed.runs ?? []).map(mapRun),
    fetchedAt: new Date().toISOString(),
  };
}
