# Institutional Strategy Research and Bot Mapping

Research date: 2026-07-25

This note records the public evidence used for the implementation. Historical
results are not a promise of future profitability.

## What transfers to this bot

1. **Slow trend / momentum.** AQR's time-series momentum research uses the sign
   of lagged returns across liquid markets, with positions scaled by lagged
   volatility. The ensemble momentum vote now blends 3-, 6-, and 12-month
   direction and scales conviction by 60-day realized volatility.
2. **Cross-sectional confirmation.** The existing relative-strength gate keeps
   individual equities only when they outperform QQQ. This is a conservative,
   long-only analogue of cross-sectional momentum.
3. **Volatility-targeted gross exposure.** In a QQQ risk-on regime, the selected
   candidate scales the normal 80% gross anchor by target volatility divided by
   trailing QQQ volatility, clamped to 60%-95%. The 20% risk-off cap remains
   authoritative. When exposure is above target plus the configured band, the
   weakest positions are closed first and QQQ is preserved until last.
4. **Cost discipline.** The backtest charges explicit basis-point costs, uses a
   rebalance band, and was checked at both 5 bps and 10 bps. The live path uses
   only completed daily bars for signals and exposure decisions.

## What was intentionally not copied

True high-frequency market making depends on direct feeds, co-location, queue
position, subsecond cross-security models, exchange fees/rebates, and real-time
inventory hedging. An hourly Alpaca/yfinance bot cannot reproduce those economics.
Short-horizon reversal was also not promoted to a standalone strategy because
public transaction-cost evidence shows that its high turnover often consumes the
gross spread.

## Primary sources

- AQR, [Time Series Momentum original-paper data](https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Original-Paper-Data)
- AQR, [A Century of Evidence on Trend-Following Investing](https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing)
- AQR, [Momentum Indices methodology and data](https://www.aqr.com/insights/datasets/momentum-indices-monthly)
- Moreira and Muir, [Volatility Managed Portfolios](https://www.nber.org/papers/w22208)
- AQR, [Trading Costs of Asset Pricing Anomalies](https://www.aqr.com/insights/research/working-paper/trading-costs-of-asset-pricing-anomalies)
- SEC DERA, [High-Frequency Trading Synchronizes Prices in Financial Markets](https://www.sec.gov/about/divisions-offices/division-economic-risk-analysis/staff-papers-analyses/21jan15_gerig_high-frequency-trading)
- NBER, [Momentum Trading, Return Chasing, and Predictable Crashes](https://www.nber.org/papers/w20660)

## Validation snapshot

The candidate matrix was fixed at 20/60-day volatility and 16%/18% targets. On
five years of cached daily history for the first 50 configured symbols plus QQQ:

| Run | CAGR | Sharpe | Max drawdown | Trades |
| --- | ---: | ---: | ---: | ---: |
| Walk-forward baseline, 5 bps | 10.50% | 0.861 | -17.71% | 906 |
| Walk-forward 60d / 18%, 5 bps | 12.01% | 0.958 | -16.78% | 931 |
| Fixed-model baseline, 10 bps | 11.29% | 0.895 | -18.87% | 1,006 |
| Fixed-model 60d / 18%, 10 bps | 11.39% | 0.896 | -18.44% | 1,031 |

The walk-forward comparison used 29 rolling prior-only model windows. The 10 bps
comparison is only a stressed-cost diagnostic because it reused a stored model.
The universe is not point-in-time and the feature has not completed paper
verification, so `risk.volatility_targeting.enabled` remains `false`.
