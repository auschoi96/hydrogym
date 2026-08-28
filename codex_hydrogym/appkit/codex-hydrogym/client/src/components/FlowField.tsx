import { useEffect, useMemo, useState } from 'react';
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  HeatmapChart,
  Slider,
} from '@databricks/appkit-ui/react';
import { Pause, Play, RotateCcw, Waves } from 'lucide-react';
import { useApiResource } from '@/lib/api';
import { frameToHeatmap, referenceFlowSchema } from '@/lib/flow';
import { PanelError, PanelLoading } from './DataStates';

export function FlowField() {
  const resource = useApiResource<unknown>('/codex_hydrogym_reference_flow.json');
  const parsed = useMemo(
    () => (resource.data === null ? null : referenceFlowSchema.safeParse(resource.data)),
    [resource.data]
  );
  const flow = parsed?.success ? parsed.data : null;
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!playing || !flow) return undefined;
    const timer = window.setInterval(() => {
      setFrameIndex((current) => (current + 1) % flow.frames.length);
    }, 700);
    return () => window.clearInterval(timer);
  }, [flow, playing]);

  const heatmapData = useMemo(
    () => (flow ? frameToHeatmap(flow, Math.min(frameIndex, flow.frames.length - 1)) : []),
    [flow, frameIndex]
  );

  if (resource.loading) return <PanelLoading rows={5} />;
  if (resource.error)
    return <PanelError title="Reference flow unavailable" error={resource.error} retry={resource.refresh} />;
  if (parsed && !parsed.success) {
    return (
      <PanelError
        title="Reference flow contract failed"
        error={new Error(parsed.error.message)}
        retry={resource.refresh}
      />
    );
  }
  if (!flow) return null;

  const activeFrame = flow.frames[Math.min(frameIndex, flow.frames.length - 1)];
  const maximum = flow.maximumAbsVorticity;
  const frameValues = activeFrame?.values ?? [];
  const frameStats = frameValues.reduce(
    (stats, value) => ({
      minimum: Math.min(stats.minimum, value),
      maximum: Math.max(stats.maximum, value),
      squaredSum: stats.squaredSum + value * value,
    }),
    { minimum: Number.POSITIVE_INFINITY, maximum: Number.NEGATIVE_INFINITY, squaredSum: 0 }
  );
  const rmsVorticity = frameValues.length > 0 ? Math.sqrt(frameStats.squaredSum / frameValues.length) : 0;

  return (
    <Card className="flow-card overflow-hidden">
      <CardHeader className="flow-card-header">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <CardTitle className="flex items-center gap-2 text-xl">
              <Waves className="h-5 w-5 text-primary" />
              Kolmogorov vorticity field
            </CardTitle>
            <CardDescription>
              Uncontrolled reference trajectory from HydroGym’s JAX pseudo-spectral solver.
            </CardDescription>
          </div>
          <Badge variant="outline" className="status-warning">
            Reference · not PPO evidence
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 p-0">
        <div className="px-2 sm:px-4">
          <HeatmapChart
            data={heatmapData}
            xKey="x"
            yAxisKey="y"
            yKey="vorticity"
            min={-maximum}
            max={maximum}
            colorPalette="diverging"
            height={460}
            ariaLabel="Vorticity heatmap for the uncontrolled HydroGym Kolmogorov reference simulation"
            options={{
              animationDuration: 500,
              grid: { left: 54, right: 76, top: 18, bottom: 52 },
              xAxis: { name: 'x grid index', nameLocation: 'middle', nameGap: 32, axisLabel: { interval: 3 } },
              yAxis: { name: 'y grid index', nameLocation: 'middle', nameGap: 38, axisLabel: { interval: 3 } },
            }}
          />
        </div>
        <div className="flow-controls">
          <Button
            variant="outline"
            size="icon"
            aria-label={playing ? 'Pause reference flow' : 'Play reference flow'}
            onClick={() => setPlaying((value) => !value)}
          >
            {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>
          <Slider
            aria-label="Reference flow frame"
            min={0}
            max={flow.frames.length - 1}
            step={1}
            value={[frameIndex]}
            onValueChange={([value]) => setFrameIndex(value ?? 0)}
            className="min-w-36 flex-1"
          />
          <Button
            variant="ghost"
            size="icon"
            aria-label="Reset reference flow"
            onClick={() => {
              setFrameIndex(0);
              setPlaying(false);
            }}
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
          <div className="min-w-28 text-right">
            <p className="text-sm font-medium text-foreground">t = {activeFrame?.time.toFixed(2) ?? '0.00'}</p>
            <p className="text-xs text-muted-foreground">solver time</p>
          </div>
        </div>
        <div className="flow-stat-grid" aria-label="Active frame vorticity statistics">
          <div className="flow-stat">
            <span>Minimum vorticity</span>
            <strong>{Number.isFinite(frameStats.minimum) ? frameStats.minimum.toFixed(4) : '—'}</strong>
          </div>
          <div className="flow-stat">
            <span>Maximum vorticity</span>
            <strong>{Number.isFinite(frameStats.maximum) ? frameStats.maximum.toFixed(4) : '—'}</strong>
          </div>
          <div className="flow-stat">
            <span>RMS vorticity</span>
            <strong>{rmsVorticity.toFixed(4)}</strong>
          </div>
        </div>
        <div className="flow-provenance">
          <span>Re = {flow.reynoldsNumber}</span>
          <span>forcing k = {flow.forcingWavenumber}</span>
          <span>{flow.gridSize.join(' × ')} grid</span>
          <span>{flow.actionDimension} actuators · zero action</span>
          <span>seed {flow.seed}</span>
        </div>
      </CardContent>
    </Card>
  );
}
