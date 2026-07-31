# Commodity diversification diagnosis

## Result

Forced diversification did not improve this strategy. Both diversification
levels reduced return and Sharpe without reducing maximum drawdown.

| Metric | Baseline | Moderate diversification | Broad diversification |
| --- | ---: | ---: | ---: |
| Holdings selected | 2–4 | 3–5 | 5–9 |
| Position cap | 40% | 30% | 25% |
| Commodity-group cap | None | 50% | 35% |
| Final equity | $300,799 | $181,040 | $185,598 |
| Total return | 200.80% | 81.04% | 85.60% |
| CAGR | 11.25% | 5.91% | 6.17% |
| Sharpe | 0.75 | 0.48 | 0.55 |
| Volatility | 15.94% | 13.88% | 12.38% |
| Maximum drawdown | -26.22% | -26.80% | -27.89% |
| Profitable four-month windows | 23/31 | 18/31 | 18/31 |
| Average exposure | 64.70% | 67.18% | 62.33% |
| Transaction costs | $10,366 | $6,750 | $5,721 |

## What was going wrong

The baseline looks diversified by ticker count, but its average effective
position count was only 2.21. The selector chose two holdings in 22 of 31
windows. Some losing periods were highly concentrated, including 67.57% average
precious-metals exposure in the 2023-04 through 2023-08 test window.

That concentration was visible, but it was not the source of the overall
underperformance versus equities:

- Broad diversification lowered direct costs by $4,645 and average exposure by
  only 2.37 percentage points, yet lost 115.20 percentage points of cumulative
  return relative to baseline.
- Volatility declined, but Sharpe also declined. The result was not simply a
  lower-risk version of the same return stream.
- Profitable windows fell from 23 to 18. Adding lower-ranked commodities diluted
  the strongest trends, while group caps cut multiple related winners during
  broad commodity moves.
- Maximum drawdown did not improve. Commodity groups can become correlated
  during inflation, dollar, liquidity, and growth shocks, so holding more funds
  did not guarantee better crisis diversification.
- The worst windows were reversals in energy or metals after trailing momentum
  had already selected them. More simultaneous positions did not fix that lag.

## Conclusion

Keep the baseline as the best of the tested commodity-only variants. Do not use
the moderate or broad diversification rules in paper trading.

The next useful experiments should target the actual weak points: slower
turnover, next-open execution, futures-curve carry/roll filters, and fundamental
energy signals such as gas prices, weather, outages, and spark spreads. Those
experiments need frozen acceptance criteria or a new holdout because choosing
them after seeing this history creates meta-overfitting risk.
