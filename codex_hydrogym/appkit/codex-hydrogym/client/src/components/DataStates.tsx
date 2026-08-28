import type { ReactNode } from 'react';
import {
  Alert,
  AlertDescription,
  AlertTitle,
  Badge,
  Button,
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
  Skeleton,
} from '@databricks/appkit-ui/react';
import { AlertTriangle, DatabaseZap, RefreshCw } from 'lucide-react';
import type { SnapshotFreshness } from '@/lib/freshness';
import { snapshotFreshness } from '@/lib/freshness';
import { formatTimestamp } from '@/lib/format';

export interface EvidenceSnapshot {
  label: string;
  fetchedAt: string | null;
}

export function PanelLoading({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-3" aria-label="Loading data">
      <Skeleton className="h-6 w-2/5" />
      {Array.from({ length: rows }, (_, index) => (
        <Skeleton key={`skeleton-${index}`} className="h-14 w-full" />
      ))}
    </div>
  );
}

export function PanelError({ title, error, retry }: { title: string; error: Error; retry?: () => void }) {
  return (
    <Alert variant="destructive">
      <AlertTriangle className="h-4 w-4" />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription className="space-y-3">
        <p>{error.message}</p>
        {retry ? (
          <Button variant="outline" size="sm" onClick={retry}>
            <RefreshCw className="h-4 w-4" />
            Retry
          </Button>
        ) : null}
      </AlertDescription>
    </Alert>
  );
}

export function EmptyPanel({ title, description, action }: { title: string; description: string; action?: ReactNode }) {
  return (
    <Empty className="min-h-56 border border-dashed">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <DatabaseZap />
        </EmptyMedia>
        <EmptyTitle>{title}</EmptyTitle>
        <EmptyDescription>{description}</EmptyDescription>
      </EmptyHeader>
      {action ? <EmptyContent>{action}</EmptyContent> : null}
    </Empty>
  );
}

export function DataProvenance({
  experimentId,
  snapshots,
}: {
  experimentId: string | null;
  snapshots: EvidenceSnapshot[];
}) {
  const states = snapshots.map((snapshot) => snapshotFreshness(snapshot.fetchedAt));
  const state: SnapshotFreshness = states.includes('stale')
    ? 'stale'
    : states.includes('unknown')
      ? 'unknown'
      : 'current';
  const title =
    state === 'stale'
      ? 'MLflow API snapshot may be stale'
      : state === 'unknown'
        ? 'MLflow API snapshot is incomplete'
        : 'MLflow API snapshot is current';

  return (
    <Alert>
      <DatabaseZap className="h-4 w-4" />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription className="space-y-2">
        <p>
          Experiment {experimentId ?? 'not attached'} · read by the app service principal. These timestamps describe API
          fetches, not the age of the underlying runs.
        </p>
        <div className="flex flex-wrap gap-2">
          {snapshots.map((snapshot, index) => {
            const snapshotState = states[index] ?? 'unknown';
            return (
              <Badge
                key={snapshot.label}
                variant="outline"
                className={snapshotState === 'current' ? 'status-success' : 'status-warning'}
              >
                {snapshot.label}: {formatTimestamp(snapshot.fetchedAt)} · {snapshotState}
              </Badge>
            );
          })}
        </div>
      </AlertDescription>
    </Alert>
  );
}
