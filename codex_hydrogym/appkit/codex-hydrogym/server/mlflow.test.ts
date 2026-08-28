import { describe, expect, it } from 'vitest';
import { createHumanFeedback, searchReviewTraces, searchTrainingRuns, type JsonRequester } from './mlflow.js';

class QueueRequester implements JsonRequester {
  readonly calls: Array<{ path: string; method: string; payload: unknown }> = [];

  constructor(private readonly responses: unknown[]) {}

  request(path: string, method: 'GET' | 'POST' | 'PATCH', payload?: unknown) {
    this.calls.push({ path, method, payload });
    return Promise.resolve(this.responses.shift() ?? {});
  }
}

describe('MLflow adapter', () => {
  it('merges review states, parses previews, and counts valid human labels', async () => {
    const requester = new QueueRequester([
      {
        traces: [
          {
            trace_id: 'tr-pending',
            request_preview: '{"scenario":"Re=200"}',
            response_preview: '{"reward_alpha":1}',
            request_time: '2026-08-22T20:00:00Z',
            tags: {
              'codex_hydrogym.critic_review_state': 'pending_adjudication',
              'codex_hydrogym.bundle_id': 'bundle-pending',
              'codex_hydrogym.group_id': 'group-pending',
              'codex_hydrogym.harness_arm': 'codex',
              'codex_hydrogym.evidence_kind': 'measured',
              'codex_hydrogym.evidence_digest': 'a'.repeat(64),
              'codex_hydrogym.critic_fold': 'train',
            },
            assessments: [],
          },
          {
            trace_id: 'tr-synthetic',
            request_preview: '{"scenario":"synthetic"}',
            response_preview: '{"accepted":true}',
            request_time: '2026-08-22T22:00:00Z',
            tags: {
              'codex_hydrogym.critic_review_state': 'pending_adjudication',
              'codex_hydrogym.bundle_id': 'bundle-synthetic',
              'codex_hydrogym.group_id': 'group-synthetic',
              'codex_hydrogym.harness_arm': 'codex',
              'codex_hydrogym.evidence_kind': 'synthetic_contract',
              'codex_hydrogym.evidence_digest': 'b'.repeat(64),
              'codex_hydrogym.critic_fold': 'train',
            },
            assessments: [],
          },
        ],
      },
      {
        traces: [
          {
            trace_id: 'tr-labeled',
            request_preview: 'plain text',
            response_preview: '{"accepted":true}',
            request_time: '2026-08-22T21:00:00Z',
            tags: {
              'codex_hydrogym.critic_review_state': 'adjudicated',
              'codex_hydrogym.bundle_id': 'bundle-labeled',
              'codex_hydrogym.group_id': 'group-labeled',
              'codex_hydrogym.harness_arm': 'claude',
              'codex_hydrogym.evidence_kind': 'measured',
              'codex_hydrogym.evidence_digest': 'c'.repeat(64),
              'codex_hydrogym.critic_fold': 'test',
            },
            assessments: [
              {
                assessment_id: 'a-1',
                assessment_name: 'critic_quality',
                feedback: { value: 5 },
                source: { source_type: 'HUMAN', source_id: 'expert@databricks.com' },
                valid: true,
              },
            ],
          },
        ],
      },
    ]);

    const result = await searchReviewTraces(requester, '123');

    expect(result.traces.map((trace) => trace.traceId)).toEqual(['tr-labeled', 'tr-pending']);
    expect(result.traces[1]?.request).toEqual({ scenario: 'Re=200' });
    expect(result.traces[0]).toMatchObject({
      bundleId: 'bundle-labeled',
      groupId: 'group-labeled',
      harnessArm: 'claude',
      criticFold: 'test',
      evidenceDigest: 'c'.repeat(64),
    });
    expect(result.summary).toEqual({ total: 2, pending: 1, labeledTraces: 1, humanLabels: 1 });
  });

  it('creates an attributable MLflow assessment before changing review state', async () => {
    const requester = new QueueRequester([{ assessment: { assessment_id: 'a-1' } }, {}]);

    const result = await createHumanFeedback(
      requester,
      'tr-123',
      4,
      'Bounded control effort and plausible TKE target.',
      'expert@databricks.com'
    );

    expect(result).toMatchObject({ assessmentCreated: true, reviewStateUpdated: true });
    expect(requester.calls[0]).toMatchObject({
      path: '/api/3.0/mlflow/traces/tr-123/assessments',
      method: 'POST',
    });
    expect(requester.calls[0]?.payload).toMatchObject({
      assessment: {
        assessment_name: 'critic_quality',
        feedback: { value: 4 },
        source: { source_type: 'HUMAN', source_id: 'expert@databricks.com' },
        metadata: { adjudicator: 'expert@databricks.com', label_role: 'consensus' },
      },
    });
    expect(requester.calls[1]).toMatchObject({
      path: '/api/2.0/mlflow/traces/tr-123/tags',
      method: 'PATCH',
      payload: {
        key: 'codex_hydrogym.critic_review_state',
        value: 'adjudicated',
      },
    });
  });

  it('keeps training diagnostics separate from held-out evidence and maps provenance', async () => {
    const requester = new QueueRequester([
      {
        runs: [
          {
            info: {
              run_id: 'run-1',
              status: 'FINISHED',
              start_time: 1_777_000_000_000,
              artifact_uri: 'dbfs:/codex_hydrogym/run-1',
            },
            data: {
              metrics: [
                { key: 'train/mean_tke', value: 0.42 },
                { key: 'train/control_l1', value: 0.08 },
                { key: 'physics/all_passed', value: 1 },
                { key: 'heldout/mean_tke', value: 0.37 },
                { key: 'heldout/control_l1', value: 0.06 },
                { key: 'heldout/reward_total', value: -0.43 },
                { key: 'heldout/physics_all_passed', value: 1 },
              ],
              tags: [
                { key: 'mlflow.runName', value: 'codex_hydrogym_h100' },
                { key: 'codex_hydrogym.completed_updates', value: '32' },
                { key: 'codex_hydrogym.registered_model_version', value: '7' },
                { key: 'codex_hydrogym.heldout_evidence_digest', value: 'a'.repeat(64) },
                { key: 'codex_hydrogym.frozen_training_fingerprint', value: 'b'.repeat(64) },
                { key: 'codex_hydrogym.reward_approval_digest', value: 'c'.repeat(64) },
                { key: 'codex_hydrogym.reward_compiled_digest', value: 'd'.repeat(64) },
              ],
            },
          },
        ],
      },
    ]);

    const result = await searchTrainingRuns(requester, '123');

    expect(result.runs[0]).toMatchObject({
      runId: 'run-1',
      runName: 'codex_hydrogym_h100',
      trainingMeanTke: 0.42,
      trainingControlL1: 0.08,
      trainingPhysicsPassed: true,
      heldoutMeanTke: 0.37,
      heldoutControlL1: 0.06,
      heldoutRewardTotal: -0.43,
      heldoutPhysicsPassed: true,
      heldoutEvidenceDigest: 'a'.repeat(64),
      frozenTrainingFingerprint: 'b'.repeat(64),
      rewardApprovalDigest: 'c'.repeat(64),
      rewardCompiledDigest: 'd'.repeat(64),
      completedUpdates: 32,
      modelVersion: '7',
    });
  });

  it('never maps train metrics into held-out fields', async () => {
    const requester = new QueueRequester([
      {
        runs: [
          {
            info: { run_id: 'training-only', status: 'FINISHED' },
            data: {
              metrics: [
                { key: 'train/mean_tke', value: 0.42 },
                { key: 'train/control_l1', value: 0.08 },
              ],
              tags: [],
            },
          },
        ],
      },
    ]);

    const result = await searchTrainingRuns(requester, '123');

    expect(result.runs[0]).toMatchObject({
      trainingMeanTke: 0.42,
      trainingControlL1: 0.08,
      heldoutMeanTke: null,
      heldoutControlL1: null,
      heldoutRewardTotal: null,
      heldoutPhysicsPassed: null,
      heldoutEvidenceDigest: null,
    });
  });
});
