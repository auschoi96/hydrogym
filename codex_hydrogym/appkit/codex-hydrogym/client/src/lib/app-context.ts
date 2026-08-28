import { useOutletContext } from 'react-router';
import type { AppMeta } from '@shared/contracts';

export interface AppOutletContext {
  meta: AppMeta | null;
  metaLoading: boolean;
  metaError: Error | null;
  refreshMeta: () => void;
}

export function useAppContext() {
  return useOutletContext<AppOutletContext>();
}
