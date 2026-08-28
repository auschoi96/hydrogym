export interface ReviewerIdentity {
  id: string | null;
  source: 'databricks_proxy' | 'local_override' | 'unavailable';
}

export interface AppMeta {
  projectLabel: 'codex_hydrogym';
  experimentId: string | null;
  experimentUrl: string | null;
  reviewer: ReviewerIdentity;
  executionIdentity: 'app_service_principal';
  evidencePolicy: {
    gate0Status: 'blocked_no_passing_artifact';
    directCriticStatus: 'synthetic_transport_only';
    criticLabelStatus: 'blocked_no_measured_native_traces';
    reviewWriteEnabled: false;
    memalignStatus: 'blocked_no_labels';
    gepaStatus: 'outside_mvp';
    ppoLaunchEnabled: false;
  };
  generatedAt: string;
}

export interface ReviewAssessment {
  assessmentId: string | null;
  name: string;
  value: unknown;
  rationale: string | null;
  sourceType: string;
  sourceId: string | null;
  createdAt: string | null;
  valid: boolean;
}

export interface ReviewTrace {
  traceId: string;
  request: unknown;
  response: unknown;
  timestampMs: number | null;
  reviewState: string;
  bundleId: string;
  groupId: string;
  harnessArm: 'codex' | 'claude';
  criticFold: 'train' | 'test';
  evidenceDigest: string;
  assessments: ReviewAssessment[];
}

export interface ReviewQueueSummary {
  total: number;
  pending: number;
  labeledTraces: number;
  humanLabels: number;
}

export interface ReviewQueueResponse {
  experimentId: string;
  traces: ReviewTrace[];
  summary: ReviewQueueSummary;
  fetchedAt: string;
}

export interface FeedbackRequest {
  traceId: string;
  score: number;
  rationale: string;
}

export interface FeedbackResponse {
  traceId: string;
  reviewer: string;
  assessmentName: 'critic_quality';
  assessmentCreated: boolean;
  reviewStateUpdated: boolean;
}

export interface TrainingRun {
  runId: string;
  runName: string | null;
  status: string;
  startedAt: string | null;
  artifactUri: string | null;
  trainingMeanTke: number | null;
  trainingControlL1: number | null;
  trainingPhysicsPassed: boolean | null;
  heldoutMeanTke: number | null;
  heldoutControlL1: number | null;
  heldoutRewardTotal: number | null;
  heldoutPhysicsPassed: boolean | null;
  heldoutEvidenceDigest: string | null;
  completedUpdates: number | null;
  registeredModel: string | null;
  modelVersion: string | null;
  modelAlias: string | null;
  alignmentStage: 'baseline' | 'aligned' | null;
  promptUri: string | null;
  evaluationContextFingerprint: string | null;
  frozenTrainingFingerprint: string | null;
  rewardApprovalDigest: string | null;
  rewardCompiledDigest: string | null;
}

export interface TrainingRunsResponse {
  experimentId: string;
  runs: TrainingRun[];
  fetchedAt: string;
}

export interface ApiErrorResponse {
  error: {
    code: string;
    message: string;
  };
}
