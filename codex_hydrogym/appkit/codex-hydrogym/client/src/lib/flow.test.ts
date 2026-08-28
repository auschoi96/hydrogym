import { describe, expect, it } from 'vitest';
import { frameToHeatmap, referenceFlowSchema } from './flow';

describe('reference flow contract', () => {
  it('maps row-major solver output into AppKit heatmap points', () => {
    const flow = referenceFlowSchema.parse({
      formatVersion: 1,
      projectLabel: 'codex_hydrogym',
      datasetKind: 'reference_simulation',
      evidenceStatus: 'not_ppo_training',
      source: 'HydroGym JAX',
      description: 'test',
      generatedAt: '2026-08-22T00:00:00Z',
      seed: 1,
      reynoldsNumber: 200,
      forcingWavenumber: 4,
      gridSize: [2, 2],
      observationGrid: [1, 1],
      actionDimension: 4,
      action: [0, 0, 0, 0],
      dt: 0.01,
      actionTime: 0.2,
      saveTime: 0.02,
      maximumAbsVorticity: 2,
      frames: [{ time: 0, values: [1, 2, 3, 4] }],
    });

    expect(frameToHeatmap(flow, 0)).toEqual([
      { x: '00', y: '00', vorticity: 1 },
      { x: '00', y: '01', vorticity: 2 },
      { x: '01', y: '00', vorticity: 3 },
      { x: '01', y: '01', vorticity: 4 },
    ]);
  });

  it('rejects a frame with the wrong grid cardinality', () => {
    const flow = referenceFlowSchema.parse({
      formatVersion: 1,
      projectLabel: 'codex_hydrogym',
      datasetKind: 'reference_simulation',
      evidenceStatus: 'not_ppo_training',
      source: 'HydroGym JAX',
      description: 'test',
      generatedAt: '2026-08-22T00:00:00Z',
      seed: 1,
      reynoldsNumber: 200,
      forcingWavenumber: 4,
      gridSize: [2, 2],
      observationGrid: [1, 1],
      actionDimension: 4,
      action: [0, 0, 0, 0],
      dt: 0.01,
      actionTime: 0.2,
      saveTime: 0.02,
      maximumAbsVorticity: 2,
      frames: [{ time: 0, values: [1, 2, 3] }],
    });

    expect(() => frameToHeatmap(flow, 0)).toThrow('expected 4');
  });
});
