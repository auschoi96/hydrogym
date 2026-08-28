import { describe, expect, it } from 'vitest';
import { reviewerIdentity } from './identity.js';

describe('reviewerIdentity', () => {
  it('prefers Databricks forwarded email', () => {
    expect(
      reviewerIdentity(
        {
          'x-forwarded-email': ' reviewer@databricks.com ',
          'x-forwarded-preferred-username': undefined,
          'x-forwarded-user': undefined,
        },
        'local@example.com'
      )
    ).toEqual({ id: 'reviewer@databricks.com', source: 'databricks_proxy' });
  });

  it('uses an explicit local override only when proxy identity is absent', () => {
    expect(
      reviewerIdentity(
        {
          'x-forwarded-email': undefined,
          'x-forwarded-preferred-username': undefined,
          'x-forwarded-user': undefined,
        },
        'local@example.com'
      )
    ).toEqual({ id: 'local@example.com', source: 'local_override' });
  });
});
