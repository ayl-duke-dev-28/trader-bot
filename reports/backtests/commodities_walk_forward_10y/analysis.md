# Commodity walk-forward analysis

## Verdict

The strategy is promising **as a commodity allocation strategy**. It beat `DBC`
on return, volatility-adjusted return, and drawdown over 31 completed
four-month out-of-sample windows. It did not beat `SPY` on CAGR or Sharpe, so
the evidence supports it as a diversifying commodity sleeve rather than a
replacement for the equity strategy.

| Metric | Strategy | DBC | SPY |
| --- | ---: | ---: | ---: |
| Final equity from $100,000 | $300,798 | $255,967 | $400,439 |
| Total return | 200.80% | 155.97% | 300.44% |
| CAGR | 11.25% | 9.53% | 14.37% |
| Sharpe | 0.75 | 0.60 | 0.84 |
| Annualized volatility | 15.94% | 17.81% | 17.92% |
| Maximum drawdown | -26.22% | -41.71% | -33.72% |

Against `DBC`, the strategy added 1.72 percentage points of annual return,
improved Sharpe by 0.15, and reduced maximum drawdown by 15.49 percentage
points. Against `SPY`, it gave up 3.12 percentage points of annual return and
0.09 of Sharpe while reducing maximum drawdown by 7.50 percentage points.

## Four-month consistency

- 23 of 31 windows were profitable: **74.2%**.
- Mean four-month return: **3.95%**.
- Median four-month return: **2.40%**.
- Best window: **+27.34%**, 2025-08-16 through 2025-12-15.
- Worst window: **-12.18%**, 2022-08-16 through 2022-12-15.

| Window | Return | Sharpe | Window drawdown | Selected parameters |
| --- | ---: | ---: | ---: | --- |
| 2025-08-16 to 2025-12-15 | +27.34% | 4.30 | -5.63% | 126-day momentum, top 4 |
| 2020-04-16 to 2020-08-15 | +20.75% | 2.93 | -6.49% | 126-day momentum, top 2 |
| 2021-12-16 to 2022-04-15 | +16.23% | 1.99 | -11.92% | 126-day momentum, top 2 |
| 2022-08-16 to 2022-12-15 | -12.18% | -2.75 | -15.15% | 126-day momentum, top 2 |
| 2024-04-16 to 2024-08-15 | -9.25% | -1.51 | -13.72% | 63-day momentum, top 4 |
| 2023-04-16 to 2023-08-15 | -8.31% | -2.16 | -9.69% | 252-day momentum, top 4 |

The main portfolio drawdown began after the 2022-03-08 equity peak, bottomed on
2024-01-17, and recovered on 2025-10-03. That recovery time is long enough that
the strategy should not be described as low-risk despite its improvement over
`DBC`.

## Stability, exposure, and costs

The selector chose 126-day momentum in 23 of 31 windows and the 126-day/top-2
combination in 20 windows. This is more stable than a selector that jumps among
parameters every period, although the candidate set still needs testing on a
future holdout that is not used for further tuning.

- Average gross exposure: **64.70%**.
- Median gross exposure: **71.39%**.
- Completely in cash: 50 of 2,596 sessions, or **1.93%**.
- Total turnover: **62.02 times portfolio value** across the full run.
- Transaction costs paid: **$10,366** using 10 bps per dollar traded.

Reconstructing the same return path before the modeled turnover charge gives an
estimated final value of approximately **$320,051**. The reported net value is
**$300,799**, so direct charges plus their lost compounding reduced ending
wealth by about **$19,252**. Cost sensitivity is the clearest implementation
concern.

## Limitations

- This trades commodity ETFs/ETPs, not direct futures, spot commodities, or
  physical power. Futures roll effects and fund expenses are embedded in fund
  prices rather than modeled explicitly.
- The universe is today's fixed list of long-lived funds. It avoids short-history
  entrants but still has present-universe selection bias.
- Yahoo adjusted daily bars do not model bid/ask spreads, market impact, taxes,
  rejected orders, or tracking differences between a signal close and the next
  executable price.
- `SPY` is context, not the mandate benchmark. `DBC` is the relevant comparison.
- This is a price-trend strategy. It does not yet use power-market fundamentals,
  futures-curve carry, heat rates, weather, generation outages, or spark spreads.

## Recommendation

Keep it as a research candidate for a capped commodity sleeve. Before paper
trading it, run a frozen-parameter holdout, raise costs to 25 and 50 bps, model
next-open execution, and compare quarterly rebalancing against monthly
rebalancing. Do not add the strategy to the live bot solely from this run.
