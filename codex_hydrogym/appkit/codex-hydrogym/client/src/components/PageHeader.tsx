import type { ReactNode } from 'react';
import { Badge } from '@databricks/appkit-ui/react';

interface PageHeaderProps {
  eyebrow: string;
  title: string;
  description: string;
  badge?: string;
  actions?: ReactNode;
}

export function PageHeader({ eyebrow, title, description, badge, actions }: PageHeaderProps) {
  return (
    <header className="page-header">
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <p className="eyebrow">{eyebrow}</p>
          {badge ? <Badge variant="outline">{badge}</Badge> : null}
        </div>
        <h1 className="text-3xl font-semibold tracking-tight text-foreground md:text-4xl">{title}</h1>
        <p className="max-w-3xl text-sm leading-6 text-muted-foreground md:text-base">{description}</p>
      </div>
      {actions ? <div className="shrink-0">{actions}</div> : null}
    </header>
  );
}
