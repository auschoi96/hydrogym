export function formatTimestamp(value: string | number | null): string {
  if (value === null) return 'Not available';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Not available';
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(date);
}

export function formatMetric(value: number | null, digits = 3): string {
  return value === null || !Number.isFinite(value) ? '—' : value.toFixed(digits);
}

export function compactId(value: string, head = 10): string {
  return value.length <= head + 4 ? value : `${value.slice(0, head)}…${value.slice(-4)}`;
}

export function prettyJson(value: unknown): string {
  if (typeof value === 'string') return value;
  return JSON.stringify(value, null, 2);
}
