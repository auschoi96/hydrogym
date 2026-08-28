import { z } from 'zod';

const referenceFrameSchema = z.object({
  time: z.number(),
  values: z.array(z.number()),
});

export const referenceFlowSchema = z.object({
  formatVersion: z.literal(1),
  projectLabel: z.literal('codex_hydrogym'),
  datasetKind: z.literal('reference_simulation'),
  evidenceStatus: z.literal('not_ppo_training'),
  source: z.string(),
  description: z.string(),
  generatedAt: z.string(),
  seed: z.number().int(),
  reynoldsNumber: z.number(),
  forcingWavenumber: z.number().int(),
  gridSize: z.tuple([z.number().int().positive(), z.number().int().positive()]),
  observationGrid: z.tuple([z.number().int().positive(), z.number().int().positive()]),
  actionDimension: z.number().int().positive(),
  action: z.array(z.number()),
  dt: z.number().positive(),
  actionTime: z.number().positive(),
  saveTime: z.number().positive(),
  maximumAbsVorticity: z.number().positive(),
  frames: z.array(referenceFrameSchema).min(1),
});

export type ReferenceFlow = z.infer<typeof referenceFlowSchema>;

export interface HeatmapPoint extends Record<string, unknown> {
  x: string;
  y: string;
  vorticity: number;
}

export function frameToHeatmap(flow: ReferenceFlow, frameIndex: number): HeatmapPoint[] {
  const [nx, ny] = flow.gridSize;
  const frame = flow.frames[frameIndex];
  if (!frame) throw new Error(`Reference frame ${frameIndex} does not exist.`);
  if (frame.values.length !== nx * ny) {
    throw new Error(`Reference frame has ${frame.values.length} values; expected ${nx * ny}.`);
  }

  return frame.values.map((vorticity, index) => ({
    x: String(Math.floor(index / ny)).padStart(2, '0'),
    y: String(index % ny).padStart(2, '0'),
    vorticity,
  }));
}
