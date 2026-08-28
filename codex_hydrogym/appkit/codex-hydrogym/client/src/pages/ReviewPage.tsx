import { useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import {
  Alert,
  AlertDescription,
  AlertTitle,
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Label,
  ScrollArea,
  Separator,
  Slider,
  Textarea,
} from '@databricks/appkit-ui/react';
import { CheckCircle2, Clock3, MessageSquareText, RefreshCw, Send, ShieldCheck, UserCheck } from 'lucide-react';
import type { FeedbackResponse, ReviewQueueResponse, ReviewTrace } from '@shared/contracts';
import { DataProvenance, EmptyPanel, PanelError, PanelLoading } from '@/components/DataStates';
import { PageHeader } from '@/components/PageHeader';
import { fetchJson, useApiResource } from '@/lib/api';
import { useAppContext } from '@/lib/app-context';
import { compactId, formatTimestamp, prettyJson } from '@/lib/format';

function TraceList({
  traces,
  selected,
  onSelect,
}: {
  traces: ReviewTrace[];
  selected: string;
  onSelect: (traceId: string) => void;
}) {
  return (
    <div className="trace-list" aria-label="MLflow review traces">
      {traces.map((trace) => {
        const humanLabels = trace.assessments.filter(
          (assessment) => assessment.valid && assessment.sourceType === 'HUMAN' && assessment.name === 'critic_quality'
        ).length;
        return (
          <button
            key={trace.traceId}
            type="button"
            className={`trace-list-item ${selected === trace.traceId ? 'trace-list-item-active' : ''}`}
            onClick={() => onSelect(trace.traceId)}
          >
            <div className="flex items-start justify-between gap-2">
              <span className="font-mono text-xs font-medium text-foreground">{compactId(trace.traceId, 14)}</span>
              {humanLabels > 0 ? (
                <Badge variant="outline" className="status-success">
                  Labeled
                </Badge>
              ) : (
                <Badge variant="outline" className="status-warning">
                  Pending
                </Badge>
              )}
            </div>
            <div className="mt-2 flex items-center gap-1.5 text-xs text-muted-foreground">
              <Clock3 className="h-3.5 w-3.5" />
              {formatTimestamp(trace.timestampMs)}
              <span>·</span>
              {trace.harnessArm} · {trace.criticFold}
              <span>·</span>
              {humanLabels} human label{humanLabels === 1 ? '' : 's'}
            </div>
          </button>
        );
      })}
    </div>
  );
}

export function ReviewPage() {
  const { meta, metaLoading, metaError, refreshMeta } = useAppContext();
  const queue = useApiResource<ReviewQueueResponse>('/api/codex-hydrogym/reviews');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [score, setScore] = useState(3);
  const [rationale, setRationale] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submissionError, setSubmissionError] = useState<Error | null>(null);
  const [submitted, setSubmitted] = useState<FeedbackResponse | null>(null);
  const traces = queue.data?.traces ?? [];
  const selected = traces.find((trace) => trace.traceId === selectedId) ?? traces[0] ?? null;
  const consensusAssessment = useMemo(
    () =>
      selected?.assessments.find(
        (assessment) => assessment.valid && assessment.name === 'critic_quality' && assessment.sourceType === 'HUMAN'
      ) ?? null,
    [selected]
  );

  const chooseTrace = (traceId: string) => {
    setSelectedId(traceId);
    setScore(3);
    setRationale('');
    setSubmitted(null);
    setSubmissionError(null);
  };

  const submitFeedback = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selected || !meta?.reviewer.id || !meta.evidencePolicy.reviewWriteEnabled) return;
    setSubmitting(true);
    setSubmissionError(null);
    setSubmitted(null);
    try {
      const result = await fetchJson<FeedbackResponse>('/api/codex-hydrogym/feedback', {
        method: 'POST',
        body: JSON.stringify({ traceId: selected.traceId, score, rationale }),
      });
      setSubmitted(result);
      setRationale('');
      queue.refresh();
    } catch (error) {
      setSubmissionError(error instanceof Error ? error : new Error(String(error)));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="EXPERT ADJUDICATION"
        title="Calibrate critique quality with trace-native evidence"
        description="Score one composite critic_quality target for measured GPT and Claude critiques. The attributable consensus label may train or evaluate MemAlign; it is never fluid-performance evidence."
        badge="CONSENSUS WRITE-BACK"
        actions={
          <Button variant="outline" onClick={queue.refresh} disabled={queue.loading}>
            <RefreshCw className={`h-4 w-4 ${queue.loading ? 'animate-spin' : ''}`} />
            Refresh queue
          </Button>
        }
      />

      {metaLoading ? <PanelLoading rows={1} /> : null}
      {metaError ? <PanelError title="Reviewer identity unavailable" error={metaError} retry={refreshMeta} /> : null}
      {meta ? (
        <Alert>
          <UserCheck className="h-4 w-4" />
          <AlertTitle>
            {meta.reviewer.id ? `Signed in as ${meta.reviewer.id}` : 'Reviewer identity unavailable'}
          </AlertTitle>
          <AlertDescription>
            Databricks forwards your identity for attribution. The attached app service principal performs the MLflow
            write; it does not impersonate you.
          </AlertDescription>
        </Alert>
      ) : null}
      {meta && !meta.evidencePolicy.reviewWriteEnabled ? (
        <Alert>
          <ShieldCheck className="h-4 w-4" />
          <AlertTitle>Review write-back is evidence-locked</AlertTitle>
          <AlertDescription>
            Search results expose previews. Expert labels remain disabled until the server retrieves each full native
            trace and verifies its RunBundle and evidence digest.
          </AlertDescription>
        </Alert>
      ) : null}

      {queue.loading && !queue.data ? <PanelLoading rows={6} /> : null}
      {queue.error ? (
        <PanelError title="Could not load MLflow traces" error={queue.error} retry={queue.refresh} />
      ) : null}
      <DataProvenance
        experimentId={meta?.experimentId ?? null}
        snapshots={[
          { label: 'Reviewer identity', fetchedAt: meta?.generatedAt ?? null },
          { label: 'Review queue', fetchedAt: queue.data?.fetchedAt ?? null },
        ]}
      />
      {!queue.loading && !queue.error && traces.length === 0 ? (
        <EmptyPanel
          title="No measured critic traces are eligible yet"
          description="The queue stays closed until Gate 0 passes and measured RunBundles produce readable native MLflow traces with a locked group split."
        />
      ) : null}

      {selected ? (
        <section className="review-layout">
          <Card className="review-queue-card">
            <CardHeader>
              <div className="flex items-center justify-between gap-3">
                <div>
                  <CardTitle>Review queue</CardTitle>
                  <CardDescription>
                    {queue.data?.summary.pending ?? 0} pending · {queue.data?.summary.humanLabels ?? 0} labels
                  </CardDescription>
                </div>
                <Badge variant="secondary">{traces.length}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[680px] pr-3">
                <TraceList traces={traces} selected={selected.traceId} onSelect={chooseTrace} />
              </ScrollArea>
            </CardContent>
          </Card>

          <div className="space-y-4">
            <Card>
              <CardHeader>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <CardTitle className="font-mono text-base">{selected.traceId}</CardTitle>
                    <CardDescription>
                      {formatTimestamp(selected.timestampMs)} · {selected.harnessArm} · {selected.criticFold} fold ·
                      bundle {compactId(selected.bundleId, 12)} · group {compactId(selected.groupId, 12)} · state{' '}
                      {selected.reviewState}
                    </CardDescription>
                  </div>
                  <Badge variant="outline">MLflow trace</Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="evidence-pair">
                  <section>
                    <p className="evidence-label">RunBundle / critic input</p>
                    <ScrollArea className="evidence-code-scroll">
                      <pre className="evidence-code">{prettyJson(selected.request)}</pre>
                    </ScrollArea>
                  </section>
                  <section>
                    <p className="evidence-label">Structured experiment critique</p>
                    <ScrollArea className="evidence-code-scroll">
                      <pre className="evidence-code">{prettyJson(selected.response)}</pre>
                    </ScrollArea>
                  </section>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageSquareText className="h-5 w-5 text-primary" />
                  Critic quality consensus
                </CardTitle>
                <CardDescription>
                  Assess physics diagnosis, statistical validity, provenance, cost awareness, and claim discipline.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {consensusAssessment ? (
                  <Alert>
                    <CheckCircle2 className="h-4 w-4 text-[color:var(--success)]" />
                    <AlertTitle>This trace already has its consensus label</AlertTitle>
                    <AlertDescription>
                      {consensusAssessment.rationale ?? 'The assessment is recorded in MLflow.'}
                    </AlertDescription>
                  </Alert>
                ) : (
                  <form
                    className="space-y-5"
                    onSubmit={(event) => {
                      void submitFeedback(event);
                    }}
                  >
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <Label htmlFor="critic-quality-score">Critic-quality score</Label>
                        <span className="score-value" aria-live="polite">
                          {score} / 5
                        </span>
                      </div>
                      <Slider
                        id="critic-quality-score"
                        min={1}
                        max={5}
                        step={1}
                        value={[score]}
                        onValueChange={([value]) => setScore(value ?? 3)}
                      />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Unreliable critique</span>
                        <span>Excellent critique</span>
                      </div>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <Label htmlFor="review-rationale">Expert rationale</Label>
                      <Textarea
                        id="review-rationale"
                        value={rationale}
                        onChange={(event) => setRationale(event.target.value)}
                        maxLength={4_000}
                        rows={6}
                        placeholder="Explain the decisive strength or failure in the critique. This rationale becomes attributable MemAlign evidence."
                      />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Required · attributable · stored on the trace</span>
                        <span>{rationale.length} / 4,000</span>
                      </div>
                    </div>
                    {submissionError ? <PanelError title="Feedback was not saved" error={submissionError} /> : null}
                    {submitted ? (
                      <Alert>
                        <ShieldCheck className="h-4 w-4" />
                        <AlertTitle>Assessment saved to MLflow</AlertTitle>
                        <AlertDescription>
                          {submitted.reviewStateUpdated
                            ? 'The trace is now marked adjudicated.'
                            : 'The assessment was saved, but the review-state tag needs reconciliation.'}
                        </AlertDescription>
                      </Alert>
                    ) : null}
                    <Button
                      type="submit"
                      disabled={
                        !meta?.reviewer.id ||
                        !meta.evidencePolicy.reviewWriteEnabled ||
                        rationale.trim().length === 0 ||
                        submitting
                      }
                    >
                      <Send className="h-4 w-4" />
                      {submitting ? 'Saving to MLflow…' : 'Submit attributable feedback'}
                    </Button>
                  </form>
                )}
              </CardContent>
            </Card>
          </div>
        </section>
      ) : null}
    </div>
  );
}
