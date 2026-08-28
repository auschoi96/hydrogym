import type { LucideIcon } from 'lucide-react';
import { CheckCircle2, ExternalLink, Gauge, MessageSquareText, ShieldAlert, ShieldCheck, Waves } from 'lucide-react';
import {
  Alert,
  AlertDescription,
  AlertTitle,
  Badge,
  BarChart,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@databricks/appkit-ui/react';
import type { ReviewQueueResponse, TrainingRunsResponse } from '@shared/contracts';
import { DataProvenance, EmptyPanel, PanelError, PanelLoading } from '@/components/DataStates';
import { FlowField } from '@/components/FlowField';
import { MetricStrip } from '@/components/MetricStrip';
import { PageHeader } from '@/components/PageHeader';
import { useApiResource } from '@/lib/api';
import { useAppContext } from '@/lib/app-context';
import { compareLatestRuns } from '@/lib/comparison';
import { formatMetric, formatTimestamp } from '@/lib/format';

export const DEMO_HEADING = 'Prove the fluid task before training a controller';

interface EvidenceGateProps {
  step: string;
  title: string;
  status: string;
  description: string;
  detail: string;
  icon: LucideIcon;
  passed?: boolean;
}

function EvidenceGate({ step, title, status, description, detail, icon: Icon, passed = false }: EvidenceGateProps) {
  return (
    <Card className="workflow-card">
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <Badge variant="secondary">{step}</Badge>
          <Icon className="h-5 w-5 text-primary" aria-hidden="true" />
        </div>
        <CardTitle className="pt-3 text-lg">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="text-muted-foreground">Evidence state</span>
          <Badge variant="outline" className={passed ? 'status-success' : 'status-warning'}>
            {status}
          </Badge>
        </div>
        <p className="text-xs leading-5 text-muted-foreground">{detail}</p>
      </CardContent>
    </Card>
  );
}

export function DemoPage() {
  const { meta, metaLoading, metaError, refreshMeta } = useAppContext();
  const reviews = useApiResource<ReviewQueueResponse>('/api/codex-hydrogym/reviews');
  const runs = useApiResource<TrainingRunsResponse>('/api/codex-hydrogym/runs');
  const trainingRuns = runs.data?.runs ?? [];
  const labelCount = reviews.data?.summary.humanLabels ?? 0;
  const comparison = compareLatestRuns(trainingRuns);
  const comparisonData = [
    ...(comparison.baseline?.heldoutMeanTke !== null && comparison.baseline
      ? [{ stage: 'Locked baseline', meanTke: comparison.baseline.heldoutMeanTke }]
      : []),
    ...(comparison.candidate?.heldoutMeanTke !== null && comparison.candidate
      ? [{ stage: 'Approved candidate', meanTke: comparison.candidate.heldoutMeanTke }]
      : []),
  ];
  const policy = meta?.evidencePolicy;
  const freshness = formatTimestamp(runs.data?.fetchedAt ?? reviews.data?.fetchedAt ?? null);

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="HYDROGYM EVIDENCE DEMO"
        title={DEMO_HEADING}
        description="Visualize a genuine JAX flow trajectory and follow the cheapest falsifiable path from numerical validity to critic calibration to one bounded, human-approved controller trial."
        badge="CODEX_HYDROGYM"
        actions={
          meta?.experimentUrl ? (
            <Button asChild variant="outline">
              <a href={meta.experimentUrl} target="_blank" rel="noreferrer">
                Open MLflow
                <ExternalLink className="h-4 w-4" />
              </a>
            </Button>
          ) : null
        }
      />

      <MetricStrip
        freshness={freshness}
        items={[
          {
            label: 'Scientific Gate 0',
            value: policy?.gate0Status === 'blocked_no_passing_artifact' ? 'Blocked' : 'Checking',
            detail: 'v2 failed; the fresh-seed screen missed both equivalence-CI margins.',
            status: 'warning',
          },
          {
            label: 'Direct critics',
            value: policy?.directCriticStatus === 'synthetic_transport_only' ? '2 transports' : 'Checking',
            detail: 'GPT and Claude passed one synthetic schema contract only.',
            status: policy?.directCriticStatus === 'synthetic_transport_only' ? 'success' : 'warning',
          },
          {
            label: 'Critic-quality labels',
            value: reviews.loading ? '…' : String(labelCount),
            detail: 'Adjudicated 1–5 HUMAN labels on measured critiques.',
            status: labelCount > 0 ? 'success' : 'warning',
          },
          {
            label: 'PPO training runs',
            value: runs.loading ? '…' : String(trainingRuns.length),
            detail: 'Training diagnostics only; held-out proof is counted separately.',
            status: 'default',
          },
        ]}
      />

      {metaLoading ? <PanelLoading rows={1} /> : null}
      {metaError ? <PanelError title="App evidence policy unavailable" error={metaError} retry={refreshMeta} /> : null}
      {reviews.error ? (
        <PanelError title="Critic-review summary unavailable" error={reviews.error} retry={reviews.refresh} />
      ) : null}
      {runs.error ? <PanelError title="PPO evidence unavailable" error={runs.error} retry={runs.refresh} /> : null}
      <DataProvenance
        experimentId={meta?.experimentId ?? null}
        snapshots={[
          { label: 'Evidence policy', fetchedAt: meta?.generatedAt ?? null },
          { label: 'Controller runs', fetchedAt: runs.data?.fetchedAt ?? null },
          { label: 'Critic reviews', fetchedAt: reviews.data?.fetchedAt ?? null },
        ]}
      />

      <section className="overview-grid">
        <FlowField />
        <Card className="h100-card">
          <CardHeader>
            <div className="flex items-center justify-between gap-3">
              <Badge variant="secondary">TRAINING LOCKED</Badge>
              <Gauge className="h-5 w-5 text-primary" aria-hidden="true" />
            </div>
            <CardTitle className="pt-3">Why controller training remains locked</CardTitle>
            <CardDescription>
              Gate 0 has now failed twice: Re=200 v1 failed numerical resolution, while Re=100 v2 passed its causal
              primary comparison but failed the frozen temporal and spatial convergence contract. A later
              development-only ensemble screen improved the point estimates but did not establish effect equivalence.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <dl className="spec-list">
              <div>
                <dt>Re=100 primary gates</dt>
                <dd>20 / 20 passed</dd>
              </div>
              <div>
                <dt>Feedback vs zero TKE</dt>
                <dd>1.1356 vs 1.9099</dd>
              </div>
              <div>
                <dt>Temporal max arm change</dt>
                <dd>8.07% &gt; 2%</dd>
              </div>
              <div>
                <dt>Spatial max arm change</dt>
                <dd>5.38% &gt; 5%</dd>
              </div>
              <div>
                <dt>Fresh-seed point checks</dt>
                <dd>4 / 4 passed</dd>
              </div>
              <div>
                <dt>Fresh-seed equivalence CIs</dt>
                <dd>0 / 2 passed</dd>
              </div>
              <div>
                <dt>H100 user-code runs</dt>
                <dd>0</dd>
              </div>
              <div>
                <dt>GEPA in MVP</dt>
                <dd>No</dd>
              </div>
            </dl>
            <Alert>
              <ShieldAlert className="h-4 w-4" />
              <AlertTitle>A primary pass is not a final pass</AlertTitle>
              <AlertDescription>
                Both refinements remained numerically valid and preserved the causal pass decisions, but arm values,
                relative effects, and exact ordering did not meet the preregistered convergence limits. No PPO run is
                authorized.
              </AlertDescription>
            </Alert>
            <Alert>
              <ShieldAlert className="h-4 w-4" />
              <AlertTitle>Stable point estimates are not equivalence evidence</AlertTitle>
              <AlertDescription>
                Feedback beat zero across all 48 paired seed/phase/window blocks, but the temporal 90% effect-difference
                CI reached +2.52 percentage points against a ±2-point margin and the spatial CI reached +3.59 points
                against a ±3-point margin. The frozen development screen therefore returned false.
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      </section>

      <section>
        <div className="section-heading">
          <div>
            <p className="eyebrow">PREREGISTERED DECISION PATH</p>
            <h2 className="text-2xl font-semibold">Each stage must earn the next expense</h2>
          </div>
          <Badge variant="outline">No autonomous promotion</Badge>
        </div>
        <div className="workflow-grid">
          <EvidenceGate
            step="01"
            title="Pass CPU Gate 0"
            status="Blocked"
            description="Establish numerical validity, controllability, observability, and causal value before RL."
            detail="Re=100 v2 failed final convergence. The fresh-seed diagnostic passed point checks but missed both equivalence-CI margins, so a new design remains a separate decision."
            icon={Waves}
          />
          <EvidenceGate
            step="02"
            title="Collect measured RunBundles"
            status="Waiting on Gate 0"
            description="Package comparable fluid evidence with exact artifact and context hashes."
            detail="Synthetic P0 data cannot enter the train or held-out critic-quality folds."
            icon={ShieldCheck}
          />
          <EvidenceGate
            step="03"
            title="Calibrate reward review"
            status="No labels"
            description="Reuse one coding-agent draft across base-reviewer and MemAlign-reviewer revision arms."
            detail="Experts label reviewer quality on disjoint groups. MemAlign may improve that review loop; it cannot prove fluid improvement."
            icon={MessageSquareText}
          />
          <EvidenceGate
            step="04"
            title="Run one bounded trial"
            status="Not authorized"
            description="Compile a human-approved RewardSpec into deterministic HydroGym reward code, then hold PPO fixed."
            detail="Codex is the first coding-agent harness. Claude and GEPA remain later cost/quality benchmarks, not prerequisites."
            icon={CheckCircle2}
          />
        </div>
      </section>

      <section className="evidence-grid">
        <Card>
          <CardHeader>
            <CardTitle>Critic calibration evidence</CardTitle>
            <CardDescription>Transport, quality, and fluid performance are separate hypotheses.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="summary-row">
              <span>Direct GPT strict transport</span>
              <strong>Passed</strong>
            </div>
            <div className="summary-row">
              <span>Direct Claude strict transport</span>
              <strong>Passed</strong>
            </div>
            <div className="summary-row">
              <span>Measured critic corpus</span>
              <strong>None</strong>
            </div>
            <div className="summary-row">
              <span>Native remote source traces</span>
              <strong>None</strong>
            </div>
            <div className="summary-row">
              <span>Registered revision prompt</span>
              <strong>Version 1</strong>
            </div>
            <div className="summary-row">
              <span>MemAlign held-out result</span>
              <strong>Not run</strong>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Expert adjudication</CardTitle>
            <CardDescription>The queue opens only for measured, trace-native critic outputs.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="comparison-result">
              <span>Consensus labels</span>
              <strong>{labelCount}</strong>
            </div>
            <p className="text-sm leading-6 text-muted-foreground">
              Experts score one composite <code>critic_quality</code> field covering physics diagnosis, statistics,
              provenance, cost awareness, and claim discipline.
            </p>
            <Button asChild variant="outline" className="w-full">
              <a href="/review">Open evidence queue</a>
            </Button>
          </CardContent>
        </Card>
      </section>

      <section className="evidence-grid">
        <Card>
          <CardHeader>
            <CardTitle>Held-out mean TKE — locked baseline vs approved candidate</CardTitle>
            <CardDescription>
              Lower is better · same fingerprint and effort accounting required · MLflow runs only.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {runs.loading && !runs.data ? <PanelLoading rows={3} /> : null}
            {!runs.loading && comparisonData.length === 0 ? (
              <EmptyPanel
                title="No comparable controller evidence"
                description="This stays empty until Gate 0 passes and a human approves a bounded trial."
              />
            ) : null}
            {comparisonData.length > 0 ? (
              <BarChart
                data={comparisonData}
                xKey="stage"
                yKey="meanTke"
                colorPalette="categorical"
                height={320}
                ariaLabel="Held-out mean turbulent kinetic energy for baseline and approved candidate policies"
              />
            ) : null}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Fluid result</CardTitle>
            <CardDescription>Only comparable deterministic evidence can populate these fields.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="comparison-result">
              <span>Mean TKE change</span>
              <strong>
                {comparison.tkeImprovement === null ? 'Pending' : `${(comparison.tkeImprovement * 100).toFixed(1)}%`}
              </strong>
            </div>
            <div className="summary-row">
              <span>Comparable context</span>
              <strong>{comparison.comparable ? 'Yes' : 'Not yet'}</strong>
            </div>
            <div className="summary-row">
              <span>Baseline control L1</span>
              <strong>{formatMetric(comparison.baseline?.heldoutControlL1 ?? null)}</strong>
            </div>
            <div className="summary-row">
              <span>Candidate control L1</span>
              <strong>{formatMetric(comparison.candidate?.heldoutControlL1 ?? null)}</strong>
            </div>
          </CardContent>
        </Card>
      </section>

      {trainingRuns.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>Latest MLflow training diagnostics</CardTitle>
            <CardDescription>
              These are train/* summaries only. They never populate the held-out comparison above.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Run tag</TableHead>
                  <TableHead>Training mean TKE</TableHead>
                  <TableHead>Training control L1</TableHead>
                  <TableHead>Training physics</TableHead>
                  <TableHead>Model</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {trainingRuns.slice(0, 6).map((run) => (
                  <TableRow key={run.runId}>
                    <TableCell className="font-medium">
                      {run.alignmentStage === 'aligned'
                        ? 'Candidate tag'
                        : run.alignmentStage === 'baseline'
                          ? 'Baseline tag'
                          : 'Unlabeled'}
                    </TableCell>
                    <TableCell>{formatMetric(run.trainingMeanTke)}</TableCell>
                    <TableCell>{formatMetric(run.trainingControlL1)}</TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={run.trainingPhysicsPassed ? 'status-success' : 'status-warning'}
                      >
                        {run.trainingPhysicsPassed
                          ? 'Passed'
                          : run.trainingPhysicsPassed === false
                            ? 'Failed'
                            : 'Pending'}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      {run.modelVersion
                        ? `v${run.modelVersion}${run.modelAlias ? ` · ${run.modelAlias}` : ''}`
                        : 'Not registered'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ) : null}

      <Alert>
        <ShieldCheck className="h-4 w-4" />
        <AlertTitle>Truth boundary</AlertTitle>
        <AlertDescription>
          The animation is a genuine uncontrolled HydroGym reference trajectory. The repository and current live service
          prove only infrastructure plus one synthetic direct-critic transport check. They do not show RL, MemAlign
          benefit, autonomous improvement, or fluid improvement.
        </AlertDescription>
      </Alert>
    </div>
  );
}
