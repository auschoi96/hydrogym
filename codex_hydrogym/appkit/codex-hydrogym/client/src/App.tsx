import { useState } from 'react';
import type { LucideIcon } from 'lucide-react';
import { FlaskConical, Menu, MessageSquareText, ShieldCheck, UserRound, Waves } from 'lucide-react';
import { createBrowserRouter, Navigate, NavLink, Outlet, RouterProvider } from 'react-router';
import {
  Badge,
  Button,
  Separator,
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@databricks/appkit-ui/react';
import type { AppMeta } from '@shared/contracts';
import { DemoPage } from '@/pages/DemoPage';
import { ReviewPage } from '@/pages/ReviewPage';
import { useApiResource } from '@/lib/api';
import type { AppOutletContext } from '@/lib/app-context';

interface NavigationItem {
  to: string;
  label: string;
  description: string;
  icon: LucideIcon;
  end?: boolean;
}

const NAVIGATION: NavigationItem[] = [
  {
    to: '/',
    label: 'HydroGym demo',
    description: 'Flow, runs & evidence',
    icon: Waves,
    end: true,
  },
  {
    to: '/review',
    label: 'Human feedback',
    description: 'Trace-native expert labels',
    icon: MessageSquareText,
  },
];

function Brand() {
  return (
    <div className="brand-lockup">
      <div className="brand-mark" aria-hidden="true">
        <FlaskConical className="h-5 w-5" />
      </div>
      <div>
        <p className="brand-name">HydroGym Control</p>
        <p className="brand-label">codex_hydrogym</p>
      </div>
    </div>
  );
}

function Navigation({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="shell-navigation" aria-label="Primary navigation">
      <p className="nav-section-label">Workspace</p>
      {NAVIGATION.map(({ to, label, description, icon: Icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          onClick={onNavigate}
          className={({ isActive }) => `shell-nav-link ${isActive ? 'shell-nav-link-active' : ''}`}
        >
          <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
          <span className="min-w-0">
            <span className="block text-sm font-medium">{label}</span>
            <span className="block truncate text-xs opacity-70">{description}</span>
          </span>
        </NavLink>
      ))}
    </nav>
  );
}

function AppRail({ experimentId }: { experimentId: string | null }) {
  return (
    <aside className="desktop-rail">
      <Brand />
      <Separator className="my-5" />
      <Navigation />
      <div className="rail-footer">
        <div className="rail-boundary">
          <ShieldCheck className="h-4 w-4 text-primary" aria-hidden="true" />
          <div>
            <p className="text-xs font-medium text-foreground">Physics-gated demo</p>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">LLMs advise. Deterministic gates promote.</p>
          </div>
        </div>
        <p className="rail-experiment">
          MLflow experiment
          <span>{experimentId ?? 'connecting…'}</span>
        </p>
      </div>
    </aside>
  );
}

function MobileNavigation() {
  const [open, setOpen] = useState(false);

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Open navigation">
          <Menu className="h-5 w-5" />
        </Button>
      </SheetTrigger>
      <SheetContent side="left" className="mobile-sheet">
        <SheetHeader className="text-left">
          <SheetTitle>
            <Brand />
          </SheetTitle>
          <SheetDescription>Evidence-gated fluid control on Databricks.</SheetDescription>
        </SheetHeader>
        <Separator className="my-4" />
        <Navigation onNavigate={() => setOpen(false)} />
      </SheetContent>
    </Sheet>
  );
}

function Layout() {
  const metaResource = useApiResource<AppMeta>('/api/codex-hydrogym/meta');
  const context: AppOutletContext = {
    meta: metaResource.data,
    metaLoading: metaResource.loading,
    metaError: metaResource.error,
    refreshMeta: metaResource.refresh,
  };
  const status = metaResource.error ? 'Unavailable' : metaResource.loading ? 'Connecting' : 'Live';

  return (
    <div className="app-shell">
      <AppRail experimentId={metaResource.data?.experimentId ?? null} />
      <div className="shell-workspace">
        <header className="shell-topbar">
          <div className="mobile-brand">
            <MobileNavigation />
            <Brand />
          </div>
          <div className="topbar-context">
            <p className="text-sm font-medium text-foreground">HydroGym evidence demo</p>
            <p className="text-xs text-muted-foreground">JAX flow · Gate 0 · paired critics · MLflow</p>
          </div>
          <div className="topbar-identity">
            <Badge variant="outline" className={metaResource.error ? 'status-warning' : 'status-success'}>
              <span className="status-dot" aria-hidden="true" />
              {status}
            </Badge>
            <div className="identity-chip">
              <UserRound className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
              <span>{metaResource.data?.reviewer.id ?? 'Databricks reviewer'}</span>
            </div>
          </div>
        </header>
        <main className="shell-content">
          <Outlet context={context} />
        </main>
      </div>
    </div>
  );
}

const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: '/', element: <DemoPage /> },
      { path: '/review', element: <ReviewPage /> },
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
