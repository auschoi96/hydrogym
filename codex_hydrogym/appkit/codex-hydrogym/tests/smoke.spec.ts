import { expect, test } from '@playwright/test';

test('codex_hydrogym control cockpit loads its solver evidence boundary', async ({ page }) => {
  await page.goto('/');

  await expect(
    page.getByRole('heading', { name: 'Prove the fluid task before training a controller', exact: true })
  ).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole('heading', { name: 'Kolmogorov vorticity field', exact: true })).toBeVisible({
    timeout: 30_000,
  });
  await expect(page.getByText('Reference · not PPO evidence', { exact: true })).toBeVisible();
  await expect(
    page.getByRole('heading', { name: 'Why controller training remains locked', exact: true })
  ).toBeVisible();
  await expect(page.getByRole('button', { name: 'Run PPO on H100', exact: true })).toHaveCount(0);
  await expect(page.getByRole('navigation', { name: 'Primary navigation' }).first()).toBeVisible();
});
