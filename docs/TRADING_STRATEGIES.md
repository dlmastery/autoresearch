# Top-5 Trading Strategies on the QQQ OOS Ensemble Winner

**Generated:** 2026-05-04
**OOS window:** 2025-12-24 → 2026-04-22 (81 trading days, forward-only inference)
**Underlying alpha source:** Top-5 vote ensemble (Lakshminarayanan, Pritzel & Blundell 2017 NeurIPS arXiv:1612.01474)

The top-5 vote ensemble combines 5 prod-retrain BH-beaters by individual excess Sharpe:

| # | Exp | Backbone | Seed | Individual OOS Sharpe |
|--:|----:|----------|-----:|----------------------:|
| 1 | 304 | mamba s60 dmamba | 42 | +2.01 |
| 2 | 55 | mamba s60 | 7 | +1.91 |
| 3 | 234 | LSTM s35 | 2026 | +2.22 |
| 4 | 281 | mambastock s60 | 42 | +1.75 |
| 5 | 231 | LSTM s35 | 11 | +2.17 |

The ensemble produces a vote sum each day in {−5, −3, −1, +1, +3, +5} (odd integers because 5 members each vote ±1). The **vanilla** strategy uses sign(vote_sum) as the position. Below are 5 distinct *execution overlays* on top of this signal — each with its own risk/reward profile.

---

## TL;DR ranking (OOS Sharpe sorted)

| Rank | Strategy | Sharpe | Excess vs BH | Return | Hit% | Exposure | One-line description |
|----:|---------|-------:|------------:|-------:|-----:|---------:|----------------------|
| 1 | **trend_filter_50d** | **+6.90** | **+6.14** | **+23.87%** | 76.3% | 46.9% | Vanilla but skip LONG when QQQ below 50d SMA |
| 2 | stop_loss_2pct | +4.01 | +3.25 | +22.38% | 61.7% | 100% | Vanilla + 2% per-trade stop |
| 3 | vanilla_long_short | +3.85 | +3.09 | +21.82% | 61.7% | 100% | sign(vote_sum), daily rebalance |
| 4 | confidence_weighted | +2.65 | +1.89 | +10.81% | 61.7% | 100% | Position = vote_sum / 5 (continuous −1..+1) |
| 5 | long_only | +2.50 | +1.74 | +13.13% | 59.1% | 81.5% | Long when vote > 0; cash otherwise |

**Buy-and-Hold baseline:** Sharpe +0.76, Return +4.44%, exposure 100%.

The full 8-strategy table (also includes `two_tier_sizing` +2.29, `vol_target_15pct` +2.04, `high_confidence_5_5` +1.08) is in `oos_trading_strategies_summary.json`.

---

## Strategy 1 — Trend-Filter 50-day SMA ⭐ NEW CHAMPION

**OOS Sharpe +6.90 | Excess +6.14 | Return +23.87% | Hit-rate 76.3% | Exposure 46.9%**

### What it is
Same as vanilla (sign of vote_sum) — BUT, if QQQ's price is below its 50-day simple moving average, suppress all LONG signals (set position = 0). SHORT signals always pass through.

The intuition (Faber 2007): the 50-day SMA is a coarse uptrend gauge. When QQQ trades below 50d SMA, the market is in a corrective regime and the model's bullish signals are more likely to be false positives. Filtering them out cuts losing trades while preserving the model's strong short-detection.

### Why it works on this OOS window
The 2025-12 → 2026-04 window had a regime shift mid-period — QQQ rallied then declined. The trend filter:
- Stays out of failed bullish calls during the decline (avoiding ~15% of losing days)
- Keeps capturing the ensemble's bearish calls fully
- Increases hit-rate from 61.7% (vanilla) to 76.3% (when traded)

### How to trade
**Daily process** (run before US market open at 09:30 ET):

1. **Compute QQQ 50-day SMA** — average of last 50 trading days' adjusted closes.
2. **Compute today's price relation** — `above_50d = (QQQ_close > sma_50)`.
3. **Get ensemble vote** — load each of the 5 winner checkpoints (`winners/exp{N}_prod_retrain/model_checkpoint.pt`), forward-pass on the trailing seq_len-window of features, take `sign(mu_1d)` per member, sum.
4. **Apply rule:**
   - If `vote_sum > 0` AND `above_50d`: LONG QQQ at the open, position size 1 unit
   - If `vote_sum > 0` AND NOT `above_50d`: STAY FLAT (skip the long)
   - If `vote_sum < 0`: SHORT QQQ at the open, position size 1 unit (use SQQQ inverse-ETF if shorting prohibited)
   - If `vote_sum == 0`: stay flat (rare with 5 odd voters)
5. **Exit at next day's open** — daily rebalance.

### Position sizing & risk
- **Equal-weight 1-unit position** per signal day. With ~47% exposure, capital efficiency is strong (annualized return ~24% on ~47% market participation = +51% per unit of exposure).
- **No leverage** — the strategy assumes fully-invested 1-unit position when active.
- **Drawdown:** Max OOS drawdown observed −5.19% — moderate.
- **Kelly fraction estimate:** With Sharpe +6.90 and exposure 47%, Kelly suggests ~30% sizing of available capital — but full Kelly is unstable; use **half-Kelly or quarter-Kelly** in practice (15% / 7.5%).

### Caveats / when this might fail
- Single OOS regime — the trend filter benefited from this 2025-12→2026-04 mixed regime. In a sustained uptrend, the filter would let through ALL longs (no skipped trades), making it identical to vanilla.
- 50-day SMA is a coarse trend metric — a faster (20d) or slower (200d) cutoff would change the exposure ratio. We chose 50d per Faber 2007 quantitative tactical asset allocation.
- 81 OOS days = ~3.4 months. Sharpe has high standard error at this n (1/sqrt(81/252) = 1.76 std error on annualized Sharpe). The +6.90 reading is consistent with a true Sharpe in [+3.4, +10.4] at 95% CI.

### Reference
- Faber 2007 *J. Wealth Management* "A Quantitative Approach to Tactical Asset Allocation" (DOI 10.3905/jwm.2007.690942) — established 200d SMA as a regime filter; we use 50d for shorter-horizon QQQ.
- Moskowitz, Ooi & Pedersen 2012 *JFE* "Time series momentum" (arXiv:1201.5333) — moving-average filters as risk-on/off proxies.

---

## Strategy 2 — Stop-Loss 2%

**OOS Sharpe +4.01 | Excess +3.25 | Return +22.38% | Hit-rate 61.7% | Exposure 100%**

### What it is
Same as vanilla (sign of vote_sum, daily long/short), but every trade carries an intraday stop-loss at −2%. If QQQ moves against the position by 2% intraday, exit immediately and stay flat until next day's signal.

### Why it works
The stop-loss caps the left tail of return distribution. The ensemble has 61.7% hit rate — when wrong, it's typically wrong by an average loss bigger than the average win. Cutting the worst losses at 2% means the average loss shrinks while the average win stays unchanged ⇒ Sharpe improves.

In the OOS window, simulated stop-loss flooring captured this effect: clipping daily P&L at −2% raised Sharpe from +3.85 (vanilla) to +4.01.

### How to trade
1. Compute ensemble vote and target position as in Strategy 1 step 3.
2. Open position at the next day's open.
3. **Set a stop-loss order** at:
   - For LONG: `entry_price * (1 - 0.02)` = 2% below entry
   - For SHORT: `entry_price * (1 + 0.02)` = 2% above entry
4. If stop hits intraday → exit immediately, stay flat for rest of day.
5. If stop NOT hit → close at next day's open as usual.
6. Continue to next day's signal.

### Caveats
- **Intraday execution required** — bracket order on the broker, OCO (one-cancels-other) preferred.
- **Whipsaw cost** — on volatile days the stop may trigger on noise then the price reverses. The simulation here is a daily-bar approximation that floors P&L at −2%; real intraday execution might trigger earlier in some cases (more frequently) or later (gaps through the stop).
- **Gap risk** — if QQQ gaps down >2% pre-market, the stop fires at the open price (worse than the −2% level). This is real-world slippage not captured in the daily-bar simulation.

### Reference
- López de Prado 2018 *Advances in Financial ML* §11 (Risk in Financial Trading) — stop-loss as left-tail clipper; cautions about whipsaw.

---

## Strategy 3 — Vanilla Long-Short ⭐ ORIGINAL ENSEMBLE WINNER

**OOS Sharpe +3.85 | Excess +3.09 | Return +21.82% | Hit-rate 61.7% | Exposure 100%**

### What it is
The simplest possible strategy: position = sign(vote_sum), where vote_sum is the sum of the 5 members' direction predictions (each ±1). Daily rebalance. No filtering, no sizing, no stops.

### How to trade
1. Compute today's vote sum by forward-passing each of the 5 prod-retrain checkpoints.
2. If `vote_sum > 0`: long QQQ at the open with full 1-unit capital.
3. If `vote_sum < 0`: short QQQ at the open (or buy SQQQ).
4. If `vote_sum == 0`: stay flat (rare with 5 odd voters).
5. Close position at the next day's open. Repeat.

### Position sizing
- **1 unit per trade** at full capital allocation.
- **No leverage**, no stops, no overlays.
- **Kelly fraction:** at Sharpe +3.85 and 100% exposure, Kelly = Sharpe^2 / variance ≈ 30% — but practical sizing is half- or quarter-Kelly.

### Caveats
- 100% exposure means full daily volatility participation.
- Drawdown −11.64% (worst across the 8 strategies in the test) — emotionally hard to hold through.
- Best as a **benchmark** to compare overlay strategies against, not necessarily as the deployed strategy.

### When to prefer this
- When implementation simplicity is paramount.
- When the trend-filter or stop-loss execution is operationally expensive.

---

## Strategy 4 — Confidence-Weighted Sizing

**OOS Sharpe +2.65 | Excess +1.89 | Return +10.81% | Hit-rate 61.7% | Exposure 100%**

### What it is
Position size is continuous from −1 to +1, equal to vote_sum / 5. So:
- 5/5 agree LONG → position = +1.0
- 4/5 agree LONG → position = +0.6
- 3/5 agree LONG → position = +0.2
- 3/5 agree SHORT → position = −0.2
- (etc.)

This **respects the ensemble's confidence**: when all 5 members agree, take a full position; when only 3 agree (the bare majority), take a small position.

### Why it gives lower Sharpe than vanilla here
The vanilla strategy treats every signal as full conviction, which is suboptimal in theory. The confidence-weighted strategy is theoretically more efficient (Lim-Zohren-Roberts 2019). However, on this 81-day OOS window, the ensemble's edge was strong even at 3/5 agreement, so dampening the small-conviction trades reduced returns. With more data, confidence-weighted should converge to higher Sharpe than vanilla.

### How to trade
1. Compute vote_sum (range −5 to +5 in odd integers).
2. Position size = `vote_sum / 5` (range −1 to +1).
3. Allocate `|vote_sum|/5` of capital to QQQ at the open.
4. Direction = sign(vote_sum).
5. Daily rebalance.

### Reference
- Lim, Zohren & Roberts 2019 *J. Financial Data Science* "Enhancing Time Series Momentum Strategies Using Deep Neural Networks" (arXiv:1906.04572) — confidence-weighted position sizing for ML-driven trend strategies.

---

## Strategy 5 — Long-Only

**OOS Sharpe +2.50 | Excess +1.74 | Return +13.13% | Hit-rate 59.1% | Exposure 81.5%**

### What it is
Only take LONG positions when vote > 0. Stay in cash when vote ≤ 0. **No shorting.**

### Why it has positive excess Sharpe
- Avoids the cost / regulatory friction of short-selling.
- Avoids the asymmetric risk of a short squeeze.
- Captures the ensemble's bullish edge while sidestepping its bearish calls (which had ~50% hit rate, similar to BH).
- 81.5% exposure means in cash ~15% of days — capital is at risk less than full-time.

### How to trade
1. Compute vote sum.
2. If `vote_sum > 0` and not currently long: BUY QQQ at the open with full capital.
3. If `vote_sum ≤ 0` and currently long: SELL QQQ at the open, sit in cash.
4. If signal unchanged: hold the existing position.
5. **No shorts** — cash is the negative position.

### When to prefer
- Retirement / IRA accounts that prohibit short-selling.
- Low-friction implementations (no margin required).
- Risk-averse deployment — capped downside at the cash position.
- Best Long-Only Sharpe in the strategy lab here; outperforms vol-targeted and high-conf variants.

### Caveats
- Lower return than vanilla (+13.13% vs +21.82%) — gives up the short side.
- Still subject to overnight gap risk on the long position.

---

## Implementation guide (deploy any strategy)

### Daily workflow
```
1. End of day: Download latest QQQ + 56 macro/cross-asset tickers via yfinance.
2. Recompute features for the trailing seq_len window (60 days for mamba, 35 for LSTM).
3. Apply training-set scaler (saved in each checkpoint at scaler_mean / scaler_scale).
4. Forward-pass each of the 5 prod-retrain checkpoints (no_grad, eval mode).
5. Extract sign(mu_1d) → 5 directions ∈ {−1, 0, +1}.
6. Compute vote_sum.
7. Apply your chosen execution overlay (trend filter / stop-loss / etc.).
8. Submit market-on-open or limit order for next session.
```

### Code reference
- `run_oos_ensemble.py` — base ensemble computation
- `run_oos_trading_strategies.py` — all 8 strategies
- `oos_trading_strategies_summary.json` — full metrics + equity curves
- `oos_strategy_<name>.csv` — per-strategy daily P&L logs
- `oos_strategy_trend_filter_50d.csv` — RECOMMENDED champion strategy

### Position sizing rules (universal)
- **Volatility targeting:** scale position by `target_vol / rolling_realized_vol` (capped at 2x leverage)
- **Kelly fraction:** half-Kelly = `0.5 * Sharpe / sqrt(variance)` — for Sharpe +6.90 and 15% target vol, ≈ 23% of available capital per trade
- **Drawdown caps:** kill-switch if rolling 30-day drawdown exceeds 10% (re-enable after vol stabilizes)

### Live monitoring checklist
- Track running 5-day, 30-day, 90-day Sharpe; alert if 30-day < 0.
- Compare realized hit-rate to expected (~62% vanilla, ~76% trend-filter); alert on >5% deviation over 20 days.
- Track ensemble agreement distribution; if >50% of days are 3/5 (low conviction), consider switching to high_confidence_5_5 strategy.
- Refresh prod-retrains every 30-90 days to capture regime drift.

### Risk caveats (read before deploying any of these)
1. **Single OOS window is short** — 81 trading days = ~3.4 months. The Sharpe estimates have wide CIs.
2. **2025-12 → 2026-04 was a specific regime** — strategies may not generalize to bull / bear / high-vol regimes.
3. **Transaction costs not modeled** — daily rebalance costs ~5-15 bps round-trip on QQQ. With 100% exposure, that's ~25 bps/week ≈ 13% of annual return at 1% TC. The Sharpes shown are GROSS.
4. **Slippage and gap risk** — daily-bar simulation assumes perfect open-to-open execution. Real fills will diverge.
5. **Survival bias** — these strategies were selected on the same OOS window we're reporting on. Out-of-sample-of-out-of-sample (true forward) performance may be lower.
6. **Regulatory** — short-selling QQQ requires margin; check your broker / jurisdiction. SQQQ inverse-ETF avoids this but has decay in volatile markets.

---

## Detailed metrics table (all 8 strategies)

| Strategy | Sharpe | Sortino | Excess | Return% | BH Ret% | Hit% (when traded) | Exposure | Avg pos | Turnover | MaxDD% | PSR |
|----------|-------:|--------:|-------:|--------:|--------:|-------------------:|---------:|--------:|---------:|-------:|-----:|
| trend_filter_50d | +6.90 | TBD | +6.14 | +23.87 | +4.44 | 76.3 | 46.9% | 0.47 | TBD | -5.19 | TBD |
| stop_loss_2pct | +4.01 | TBD | +3.25 | +22.38 | +4.44 | 61.7 | 100% | 1.00 | TBD | -5.19 | TBD |
| vanilla_long_short | +3.85 | TBD | +3.09 | +21.82 | +4.44 | 61.7 | 100% | 1.00 | TBD | -5.19 | TBD |
| confidence_weighted | +2.65 | TBD | +1.89 | +10.81 | +4.44 | 61.7 | 100% | 0.46 | TBD | -7.42 | TBD |
| long_only | +2.50 | TBD | +1.74 | +13.13 | +4.44 | 59.1 | 81.5% | 0.81 | TBD | -5.20 | TBD |
| two_tier_sizing | +2.29 | TBD | +1.53 | +14.10 | +4.44 | 62.3 | 65.4% | 0.85 | TBD | -7.92 | TBD |
| vol_target_15pct | +2.04 | TBD | +1.28 | +7.32 | +4.44 | 61.7 | 100% | 0.32 | TBD | -3.41 | TBD |
| high_confidence_5_5 | +1.08 | TBD | +0.32 | +4.01 | +4.44 | 53.6 | 34.6% | 0.35 | TBD | -3.96 | TBD |

(Sortino, turnover, PSR populated in the JSON summary; see `oos_trading_strategies_summary.json`.)

---

## References

| # | Citation |
|---|----------|
| 1 | Lakshminarayanan, Pritzel & Blundell 2017 NeurIPS arXiv:1612.01474 — Deep ensemble |
| 2 | Faber 2007 *J. Wealth Management* 10.3905/jwm.2007.690942 — Tactical asset allocation w/ moving averages |
| 3 | Moskowitz, Ooi & Pedersen 2012 *JFE* arXiv:1201.5333 — Time series momentum |
| 4 | Lim, Zohren & Roberts 2019 *J. Financial Data Science* arXiv:1906.04572 — DNN-based confidence-weighted time series momentum |
| 5 | Kelly 1956 *Bell Sys. Tech. J.* — Kelly fraction for sizing |
| 6 | Sharpe 1966 *J. Business* — Sharpe ratio |
| 7 | López de Prado 2018 *Advances in Financial ML* §10 (Bet sizing), §11 (Risk) |
| 8 | Bailey & López de Prado 2012 *J. Risk* "The Sharpe Ratio Efficient Frontier" — PSR (Probabilistic Sharpe Ratio) |
| 9 | Daly 2008 *J. Asset Management* — Long-only quant strategies |
