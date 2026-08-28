export interface PhysicsGate {
  name: string;
  criterion: string;
  reason: string;
}

export const PHYSICS_GATES: PhysicsGate[] = [
  { name: 'Finite state & metrics', criterion: 'non-finite count = 0', reason: 'Reject numerical blow-ups.' },
  {
    name: 'Reward identity',
    criterion: '|total − TKE − control| ≤ 1e−5',
    reason: 'Guarantee reward accounting is exact.',
  },
  { name: 'Nonnegative TKE', criterion: 'minimum TKE ≥ −1e−7', reason: 'Preserve physical energy semantics.' },
  { name: 'Bounded control', criterion: 'L1 effort ≤ 2.00001', reason: 'Enforce the four clipped actuators.' },
  {
    name: 'Zero-mean vorticity',
    criterion: 'zero-mode ratio ≤ 1e−5',
    reason: 'Protect the periodic spectral formulation.',
  },
  {
    name: 'Incompressibility',
    criterion: 'divergence ratio ≤ 1e−5',
    reason: 'Reject non-solenoidal velocity fields.',
  },
  {
    name: 'Spectral tail',
    criterion: 'tail power fraction ≤ 0.05',
    reason: 'Check two-thirds de-aliasing remains controlled.',
  },
  { name: 'CFL', criterion: 'advective CFL ≤ 1', reason: 'Reject unstable final-state integration.' },
  { name: 'Update budget', criterion: '0 ≤ updates ≤ configured total', reason: 'Validate checkpoint progress.' },
];
