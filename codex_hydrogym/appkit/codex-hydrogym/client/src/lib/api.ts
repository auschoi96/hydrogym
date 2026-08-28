import { useEffect, useState } from 'react';
import type { ApiErrorResponse } from '@shared/contracts';

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(message: string, status: number, code: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

export async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  });
  const body = (await response.json().catch(() => null)) as T | ApiErrorResponse | { error?: string } | null;
  if (!response.ok) {
    const apiError = body as ApiErrorResponse | null;
    const pluginError = body as { error?: string } | null;
    throw new ApiError(
      apiError?.error?.message ?? pluginError?.error ?? `Request failed with status ${response.status}`,
      response.status,
      apiError?.error?.code ?? 'REQUEST_FAILED'
    );
  }
  if (body === null) throw new ApiError('The server returned an empty response.', response.status, 'EMPTY_RESPONSE');
  return body as T;
}

interface ResourceState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
}

export function useApiResource<T>(path: string) {
  const [revision, setRevision] = useState(0);
  const [state, setState] = useState<ResourceState<T>>({ data: null, loading: true, error: null });

  useEffect(() => {
    const controller = new AbortController();
    fetchJson<T>(path, { signal: controller.signal })
      .then((data) => setState({ data, loading: false, error: null }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setState({ data: null, loading: false, error: error instanceof Error ? error : new Error(String(error)) });
      });
    return () => controller.abort();
  }, [path, revision]);

  const refresh = () => {
    setState((current) => ({ ...current, loading: true, error: null }));
    setRevision((value) => value + 1);
  };

  return { ...state, refresh };
}
