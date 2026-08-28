import { createApp, getExecutionContext, server } from '@databricks/appkit';
import type { AppMeta } from '../shared/contracts.js';
import { reviewerIdentity } from './identity.js';
import { PROJECT_LABEL, searchReviewTraces, searchTrainingRuns, workspaceRequester } from './mlflow.js';

function requiredExperimentId(): string {
  const experimentId = process.env.MLFLOW_EXPERIMENT_ID?.trim();
  if (!experimentId) {
    throw new Error('MLFLOW_EXPERIMENT_ID is not configured');
  }
  return experimentId;
}

function experimentUrl(experimentId: string | null): string | null {
  const host = process.env.DATABRICKS_HOST?.replace(/\/$/, '');
  const workspaceId = process.env.DATABRICKS_WORKSPACE_ID?.trim();
  if (!host || !experimentId) return null;
  const workspaceQuery = workspaceId ? `?o=${encodeURIComponent(workspaceId)}` : '';
  return `${host}/ml/experiments/${encodeURIComponent(experimentId)}${workspaceQuery}`;
}

function logRouteError(route: string, error: unknown) {
  const resolved = error instanceof Error ? error : new Error(String(error));
  console.error(
    JSON.stringify({
      project_label: PROJECT_LABEL,
      component: 'app_api',
      route,
      error_type: resolved.name,
      message: resolved.message,
    })
  );
  return resolved;
}

await createApp({
  plugins: [server()],
  onPluginsReady(appkit) {
    appkit.server.extend((app) => {
      app.get('/api/codex-hydrogym/meta', (req, res) => {
        const experimentId = process.env.MLFLOW_EXPERIMENT_ID?.trim() || null;
        const meta: AppMeta = {
          projectLabel: PROJECT_LABEL,
          experimentId,
          experimentUrl: experimentUrl(experimentId),
          reviewer: reviewerIdentity(req.headers),
          executionIdentity: 'app_service_principal',
          evidencePolicy: {
            gate0Status: 'blocked_no_passing_artifact',
            directCriticStatus: 'synthetic_transport_only',
            criticLabelStatus: 'blocked_no_measured_native_traces',
            reviewWriteEnabled: false,
            memalignStatus: 'blocked_no_labels',
            gepaStatus: 'outside_mvp',
            ppoLaunchEnabled: false,
          },
          generatedAt: new Date().toISOString(),
        };
        res.json(meta);
      });

      app.get('/api/codex-hydrogym/reviews', async (_req, res) => {
        try {
          const requester = workspaceRequester(getExecutionContext().client);
          res.json(await searchReviewTraces(requester, requiredExperimentId()));
        } catch (error) {
          const resolved = logRouteError('GET /reviews', error);
          res.status(502).json({ error: { code: 'MLFLOW_READ_FAILED', message: resolved.message } });
        }
      });

      app.get('/api/codex-hydrogym/runs', async (_req, res) => {
        try {
          const requester = workspaceRequester(getExecutionContext().client);
          res.json(await searchTrainingRuns(requester, requiredExperimentId()));
        } catch (error) {
          const resolved = logRouteError('GET /runs', error);
          res.status(502).json({ error: { code: 'MLFLOW_READ_FAILED', message: resolved.message } });
        }
      });

      app.post('/api/codex-hydrogym/feedback', (_req, res) => {
        res.status(409).json({
          error: {
            code: 'REVIEW_WRITE_BLOCKED',
            message: 'Review write-back is locked until full native traces and evidence digests are verified.',
          },
        });
      });
    });
  },
});
