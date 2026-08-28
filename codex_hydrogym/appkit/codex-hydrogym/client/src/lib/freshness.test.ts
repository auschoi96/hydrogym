import { describe, expect, it } from 'vitest';
import { snapshotFreshness } from './freshness';

describe('snapshotFreshness', () => {
  const now = Date.parse('2026-08-24T20:00:00Z');

  it('distinguishes current, stale, and unavailable API snapshots', () => {
    expect(snapshotFreshness('2026-08-24T19:50:00Z', now)).toBe('current');
    expect(snapshotFreshness('2026-08-24T19:40:00Z', now)).toBe('stale');
    expect(snapshotFreshness(null, now)).toBe('unknown');
    expect(snapshotFreshness('not-a-date', now)).toBe('unknown');
  });
});
