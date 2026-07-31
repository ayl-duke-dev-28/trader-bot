# Trader bot stress report

- Generated: `2026-07-31T19:05:43.332586+00:00`
- Seed: `7`
- Synthetic history: `520` business days; simulated `252` days
- Saved ML model included: `yes`
- Total runtime: `3.92s`
- Safety verdict: **PASS**

| Scenario | Status | Return | Max drawdown | Worst day | Trades | VaR blocks | Max gross | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | PASS | 1.16% | -5.07% | -0.81% | 117 | 0 | 66.22% | 0.66s |
| flash_crash | PASS | -14.46% | -19.16% | -17.98% | 68 | 0 | 63.16% | 0.46s |
| prolonged_bear | PASS | -1.87% | -2.56% | -0.72% | 62 | 0 | 62.86% | 0.42s |
| volatility_spike | PASS | -31.10% | -31.93% | -3.51% | 71 | 64 | 63.16% | 0.45s |
| missing_data | PASS | 1.62% | -4.71% | -0.82% | 107 | 0 | 63.37% | 0.65s |
| high_cost | PASS | -27.05% | -27.07% | -1.28% | 113 | 0 | 65.88% | 0.67s |
| macro_contraction | PASS | 0.37% | -0.72% | -0.16% | 80 | 0 | 19.26% | 0.59s |

## Interpretation

PASS means all software safety invariants and configured exposure limits held. WARN means the simulation stayed operational but crossed a stated risk or runtime budget. FAIL means an exception, insolvency, non-finite accounting value, negative cash, or cap violation.

Synthetic scenarios test behavior under controlled shocks; they do not predict returns or replace historical out-of-sample backtests.
