export type SnapshotFreshness = 'current' | 'stale' | 'unknown';

export function snapshotFreshness(
  fetchedAt: string | null,
  now = Date.now(),
  staleAfterMs = 15 * 60 * 1_000
): SnapshotFreshness {
  if (fetchedAt === null) return 'unknown';
  const timestamp = Date.parse(fetchedAt);
  if (!Number.isFinite(timestamp)) return 'unknown';
  return now - timestamp > staleAfterMs ? 'stale' : 'current';
}
