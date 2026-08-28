import type { TrainingRun } from '@shared/contracts';

export interface RunComparison {
  baseline: TrainingRun | null;
  candidate: TrainingRun | null;
  comparable: boolean;
  tkeImprovement: number | null;
}

const SHA256 = /^[0-9a-f]{64}$/;

function isFiniteAtLeast(value: number | null, minimum: number): value is number {
  return value !== null && Number.isFinite(value) && value >= minimum;
}

function hasHeldoutProvenance(run: TrainingRun): boolean {
  return (
    run.status === 'FINISHED' &&
    isFiniteAtLeast(run.heldoutMeanTke, Number.EPSILON) &&
    isFiniteAtLeast(run.heldoutControlL1, 0) &&
    run.heldoutPhysicsPassed === true &&
    run.evaluationContextFingerprint !== null &&
    SHA256.test(run.evaluationContextFingerprint) &&
    run.frozenTrainingFingerprint !== null &&
    SHA256.test(run.frozenTrainingFingerprint) &&
    run.heldoutEvidenceDigest !== null &&
    SHA256.test(run.heldoutEvidenceDigest)
  );
}

function isApprovedCandidate(run: TrainingRun): boolean {
  return (
    run.alignmentStage === 'aligned' &&
    run.rewardApprovalDigest !== null &&
    SHA256.test(run.rewardApprovalDigest) &&
    run.rewardCompiledDigest !== null &&
    SHA256.test(run.rewardCompiledDigest)
  );
}

/**
 * Compare only dedicated `heldout/*` evidence. Training curves, alignment tags,
 * and matching evaluation contexts are insufficient on their own.
 *
 * Input order is newest first, matching the MLflow search endpoint.
 */
export function compareLatestRuns(runs: TrainingRun[]): RunComparison {
  const baselines = runs.filter((run) => run.alignmentStage === 'baseline' && hasHeldoutProvenance(run));
  const candidates = runs.filter((run) => isApprovedCandidate(run) && hasHeldoutProvenance(run));

  for (const candidate of candidates) {
    const baseline = baselines.find(
      (run) =>
        run.evaluationContextFingerprint === candidate.evaluationContextFingerprint &&
        run.frozenTrainingFingerprint === candidate.frozenTrainingFingerprint
    );
    if (!baseline || baseline.heldoutMeanTke === null || candidate.heldoutMeanTke === null) continue;
    return {
      baseline,
      candidate,
      comparable: true,
      tkeImprovement: (baseline.heldoutMeanTke - candidate.heldoutMeanTke) / baseline.heldoutMeanTke,
    };
  }

  return { baseline: null, candidate: null, comparable: false, tkeImprovement: null };
}
