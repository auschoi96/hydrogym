import { Badge } from '@databricks/appkit-ui/react';

export interface MetricItem {
  label: string;
  value: string;
  detail: string;
  status?: 'default' | 'success' | 'warning';
}

export function MetricStrip({ items, freshness }: { items: MetricItem[]; freshness: string }) {
  return (
    <section className="metric-strip" aria-label="Demo status metrics">
      <div className="metric-grid">
        {items.map((item) => (
          <div key={item.label} className="metric-item">
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{item.label}</p>
              {item.status && item.status !== 'default' ? (
                <Badge variant="outline" className={item.status === 'success' ? 'status-success' : 'status-warning'}>
                  {item.status === 'success' ? 'Ready' : 'Pending'}
                </Badge>
              ) : null}
            </div>
            <p className="mt-2 text-2xl font-semibold tracking-tight text-foreground">{item.value}</p>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">{item.detail}</p>
          </div>
        ))}
      </div>
      <p className="metric-source">Source: attached MLflow experiment · {freshness}</p>
    </section>
  );
}
