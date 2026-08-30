# Instrument calibration scope and production discrepancy

This package calibrates the **protocol rule**, not a production treatment and
not the decision currently executed by the notebook. The preregistration says
the group-clustered 95% interval must be wholly above zero
(`codex_hydrogym/agent_eval/AGENT_REVISION_PROTOCOL.md:45`). The notebook instead
sets its decision from bare `paired_mean_delta > 0` signs
(`codex_hydrogym/notebooks/coding_agent_memalign_proof.py:368-376`); it computes
no group-clustered interval there. Therefore the calibrated interval rule is
currently absent from the notebook's executed decision path. Changing that
path is a preregistration decision for a human and is intentionally out of
scope here.

The calibration DGP is `tier mean + Normal(0, 1.0**2)`, with 500 consecutive
explicit seeds beginning at 20260826, ten independent group clusters per arm,
and the frozen two-sided 95% Student critical value 2.262157162798205. The
source of that value is unambiguous for this harness:
`codex_hydrogym/gate0/ensemble_replication.py:154-157` calls it ten clusters,
df=9. The same field name is reused for different designs elsewhere:
`ensemble_diagnostic.py:128-129,163-166` uses 3.182446305284263 for four
clusters/df=3, while `re100_v3.py:272-273,295-298` uses 2.200985160082949 for
12 clusters/df=11. These constants are design-specific, not interchangeable.
The harness therefore rejects every group count other than ten and records
both the critical value and df in every arm result.

The notebook is not imported because it is a Databricks export with top-level
execution. A source contract test AST-compares its exact paired subtraction
with the harness copy, so operand, sign, or treatment-label drift fails CI.

## Frozen-seed result

For the constants above, the measured null false-positive rate is **0.032**
(16/500; 95% Wilson binomial CI **[0.0197915, 0.0513449]**). Aggregate positive
95% interval coverage is **0.9495** (1899/2000; 95% Wilson CI
**[0.9390120, 0.9582645]**), and aggregate detection is
**0.748**. Thus this frozen experiment is consistent with calibration at the
nominal 95% interval level; the null estimate is below 5% and coverage is
approximately 95%.

Sensitivity by tier pair (`true delta: detection, coverage`) is: `(0,1)`
`1.0: 0.520, 0.950`; `(1,3)` `2.0: 0.982, 0.940`; `(2,4)`
`2.0: 0.978, 0.960`; `(3,4)` `1.0: 0.512, 0.948`. Detection is monotone in
absolute true delta within the prospectively fixed 0.10 tolerance. Mean noisy
Spearman rank correlation is 0.9952, above the fixed 0.90 threshold; no test
compares recovered scores with `TIER_BASE_SCORE` values.
