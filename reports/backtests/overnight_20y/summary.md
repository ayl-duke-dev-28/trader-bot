# Close-to-next-open backtest

Period requested: 2006-08-20 through 2026-08-21 (Yahoo end date is exclusive).
Prices: Yahoo Finance adjusted daily OHLC, so splits and cash distributions are reflected.
Execution: buy at each adjusted close, sell at the next adjusted open; fully in cash intraday.
Costs: the stated basis points are charged at both the close buy and next-open sell.
No taxes, commissions beyond modeled costs, capacity limits, or auction-fill constraints.

| symbol | strategy | cost_bps_per_side | total_return | cagr | annual_volatility | sharpe | max_drawdown | ending_value | trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | buy_and_hold | - | 746.3% | 11.3% | 19.4% | 0.65 | -55.2% | $846,317 | 5030 |
| SPY | intraday | - | 92.9% | 3.3% | 15.0% | 0.29 | -46.7% | $192,935 | 5031 |
| SPY | overnight | 0.0 | 338.5% | 7.7% | 11.8% | 0.69 | -29.4% | $438,485 | 5030 |
| SPY | overnight | 1.0 | 60.3% | 2.4% | 11.8% | 0.26 | -29.8% | $160,336 | 5030 |
| SPY | overnight | 2.0 | -41.4% | -2.6% | 11.7% | -0.17 | -55.9% | $58,623 | 5030 |
| SPY | overnight | 5.0 | -97.1% | -16.3% | 11.7% | -1.46 | -97.2% | $2,864 | 5030 |
| QQQ | buy_and_hold | - | 2064.1% | 16.7% | 22.1% | 0.81 | -53.4% | $2,164,094 | 5030 |
| QQQ | intraday | - | 147.3% | 4.6% | 17.8% | 0.34 | -49.4% | $247,285 | 5031 |
| QQQ | overnight | 0.0 | 773.3% | 11.5% | 12.9% | 0.91 | -27.4% | $873,322 | 5030 |
| QQQ | overnight | 1.0 | 219.3% | 6.0% | 12.9% | 0.52 | -30.8% | $319,339 | 5030 |
| QQQ | overnight | 2.0 | 16.8% | 0.8% | 12.9% | 0.13 | -35.3% | $116,758 | 5030 |
| QQQ | overnight | 5.0 | -94.3% | -13.4% | 12.9% | -1.05 | -94.6% | $5,703 | 5030 |
| IWM | buy_and_hold | - | 454.1% | 9.0% | 24.5% | 0.47 | -58.6% | $554,082 | 5030 |
| IWM | intraday | - | -37.1% | -2.3% | 20.2% | -0.01 | -61.3% | $62,902 | 5031 |
| IWM | overnight | 0.0 | 777.1% | 11.5% | 14.0% | 0.85 | -28.8% | $877,118 | 5030 |
| IWM | overnight | 1.0 | 220.7% | 6.0% | 14.0% | 0.49 | -30.5% | $320,727 | 5030 |
| IWM | overnight | 2.0 | 17.3% | 0.8% | 14.0% | 0.13 | -45.8% | $117,265 | 5030 |
| IWM | overnight | 5.0 | -94.3% | -13.3% | 14.0% | -0.95 | -94.6% | $5,728 | 5030 |
| DIA | buy_and_hold | - | 621.1% | 10.4% | 18.5% | 0.63 | -51.9% | $721,102 | 5030 |
| DIA | intraday | - | 91.3% | 3.3% | 14.0% | 0.30 | -43.6% | $191,266 | 5031 |
| DIA | overnight | 0.0 | 277.2% | 6.9% | 11.1% | 0.65 | -28.8% | $377,249 | 5030 |
| DIA | overnight | 1.0 | 37.9% | 1.6% | 11.1% | 0.20 | -29.2% | $137,945 | 5030 |
| DIA | overnight | 2.0 | -49.6% | -3.4% | 11.1% | -0.25 | -56.2% | $50,436 | 5030 |
| DIA | overnight | 5.0 | -97.5% | -16.9% | 11.1% | -1.61 | -97.7% | $2,464 | 5030 |
| VTI | buy_and_hold | - | 740.9% | 11.3% | 19.6% | 0.64 | -55.5% | $840,888 | 5030 |
| VTI | intraday | - | -33.3% | -2.0% | 15.3% | -0.06 | -55.8% | $66,729 | 5031 |
| VTI | overnight | 0.0 | 1161.8% | 13.5% | 12.0% | 1.12 | -30.8% | $1,261,833 | 5030 |
| VTI | overnight | 1.0 | 361.4% | 8.0% | 12.0% | 0.70 | -31.3% | $461,402 | 5030 |
| VTI | overnight | 2.0 | 68.7% | 2.7% | 12.0% | 0.28 | -31.8% | $168,700 | 5030 |
| VTI | overnight | 5.0 | -91.8% | -11.8% | 12.0% | -0.98 | -92.3% | $8,241 | 5030 |
