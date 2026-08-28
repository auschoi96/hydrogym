import { describe, expect, it } from 'vitest';
import type { TrainingRun } from '@shared/contracts';
import { compareLatestRuns } from './comparison';

const CONTEXT = 'a'.repeat(64);
const FROZEN = 'b'.repeat(64);

function run(
  stage: 'baseline' | 'aligned',
  heldoutMeanTke: number | null,
  overrides: Partial<TrainingRun> = {}
): TrainingRun {
  return {
    runId: stage,
    runName: stage,
    status: 'FINISHED',
    startedAt: '2026-08-22T00:00:00Z',
    artifactUri: null,
    trainingMeanTke: 99,
    trainingControlL1: 99,
    trainingPhysicsPassed: true,
    heldoutMeanTke,
    heldoutControlL1: 0.1,
    heldoutRewardTotal: -1,
    heldoutPhysicsPassed: true,
    heldoutEvidenceDigest: 'c'.repeat(64),
    completedUpdates: 32,
    registeredModel: null,
    modelVersion: null,
    modelAlias: null,
    alignmentStage: stage,
    promptUri: null,
    evaluationContextFingerprint: CONTEXT,
    frozenTrainingFingerprint: FROZEN,
    rewardApprovalDigest: stage === 'aligned' ? 'd'.repeat(64) : null,
    rewardCompiledDigest: stage === 'aligned' ? 'e'.repeat(64) : null,
    ...overrides,
  };
}

describe('PPO comparison', () => {
  it('compares approved, physics-valid held-out evidence with frozen training parity', () => {
    const result = compareLatestRuns([run('aligned', 0.8), run('baseline', 1)]);

    expect(result.comparable).toBe(true);
    expect(result.baseline?.runId).toBe('baseline');
    expect(result.candidate?.runId).toBe('aligned');
    expect(result.tkeImprovement).toBeCloseTo(0.2);
  });

  it('does not substitute train metrics when held-out metrics are absent', () => {
    const result = compareLatestRuns([run('aligned', null), run('baseline', null)]);

    expect(result.comparable).toBe(false);
    expect(result.tkeImprovement).toBeNull();
    expect(result.baseline).toBeNull();
    expect(result.candidate).toBeNull();
  });

  it.each([
    ['context mismatch', { evaluationContextFingerprint: 'f'.repeat(64) }],
    ['training mismatch', { frozenTrainingFingerprint: 'f'.repeat(64) }],
    ['missing held-out digest', { heldoutEvidenceDigest: null }],
    ['failed physics', { heldoutPhysicsPassed: false }],
    ['missing approval', { rewardApprovalDigest: null }],
    ['missing compiled reward', { rewardCompiledDigest: null }],
  ])('fails closed on %s', (_label, candidateOverrides) => {
    const result = compareLatestRuns([run('aligned', 0.8, candidateOverrides), run('baseline', 1)]);

    expect(result).toEqual({
      baseline: null,
      candidate: null,
      comparable: false,
      tkeImprovement: null,
    });
  });
});
