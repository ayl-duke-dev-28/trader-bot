# Trader bot stress report

- Generated: `2026-07-31T16:47:03.476879+00:00`
- Seed: `7`
- Synthetic history: `520` business days; simulated `252` days
- Saved ML model included: `yes`
- Total runtime: `2.65s`
- Safety verdict: **WARN**

| Scenario | Status | Return | Max drawdown | Worst day | Trades | Stops | Max gross | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | PASS | 1.16% | -5.07% | -0.81% | 117 | 1 | 66.22% | 0.40s |
| flash_crash | PASS | -14.46% | -19.16% | -17.98% | 68 | 3 | 63.16% | 0.30s |
| prolonged_bear | PASS | -1.87% | -2.56% | -0.72% | 62 | 2 | 62.86% | 0.28s |
| volatility_spike | WARN | -57.88% | -58.39% | -3.51% | 120 | 1 | 63.16% | 0.37s |
| missing_data | PASS | 1.62% | -4.71% | -0.82% | 107 | 1 | 63.37% | 0.47s |
| high_cost | PASS | -27.05% | -27.07% | -1.28% | 113 | 1 | 65.88% | 0.39s |
| macro_contraction | PASS | 0.37% | -0.72% | -0.16% | 80 | 1 | 19.26% | 0.42s |

## Findings

- **volatility_spike:** max drawdown -58.4% exceeds the 35% stress budget

## Interpretation

PASS means all software safety invariants and configured exposure limits held. WARN means the simulation stayed operational but crossed a stated risk or runtime budget. FAIL means an exception, insolvency, non-finite accounting value, negative cash, or cap violation.

Synthetic scenarios test behavior under controlled shocks; they do not predict returns or replace historical out-of-sample backtests.
