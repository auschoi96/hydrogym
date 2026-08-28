import type { ReviewerIdentity } from '../shared/contracts.js';

const IDENTITY_HEADERS = ['x-forwarded-email', 'x-forwarded-preferred-username', 'x-forwarded-user'] as const;

export function reviewerIdentity(
  headers: Record<string, string | string[] | undefined>,
  localOverride = process.env.CODEX_HYDROGYM_REVIEWER
): ReviewerIdentity {
  for (const header of IDENTITY_HEADERS) {
    const rawValue = headers[header];
    const value = Array.isArray(rawValue) ? rawValue[0] : rawValue;
    if (value?.trim()) {
      return { id: value.trim(), source: 'databricks_proxy' };
    }
  }

  if (localOverride?.trim()) {
    return { id: localOverride.trim(), source: 'local_override' };
  }

  return { id: null, source: 'unavailable' };
}
