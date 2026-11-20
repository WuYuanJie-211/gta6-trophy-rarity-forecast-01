# GTA VI Trophy Rarity Forecast

[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/WuYuanJie-211/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/WuYuanJie-211/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at 2026-11-26T00:00:00Z — five quantiles (5%, 25%, 50%, 75%,
95%) per trophy.

## Model

`main.py` (Python 3.13, standard library only, no network, deterministic) decomposes
each trophy's rate as earners / tracked players (E/N) and models the two
separately, following Superbot's plan:

- **N (tracked players):** last-24h log growth, damped with a 36h half-life.
- **E (earners per trophy):** median of dE/dt over the last 12h, accumulated to
  the target. Main forecast is logit(E(T)/N(T)); a damped logit AR (Theil-Sen
  slope over the last 12h) is blended in as a fallback.
- **Ceilings:** per-category caps (Ch 1–2 up to 75%, Ch 3–4 + side up to 35%,
  grind up to 8%, money up to 15%, platinum/100% up to 3%); the saturated
  100% trophy stays at its observed level. History data overrides the priors.
- **Spread:** σ² = σ²_resid + (SE_slope × H_eff)² + σ²_dilution, clamped to
  [0.6, 2.5] logit units; secret trophies get 15% wider residual spread.

Run locally on the practice cases:

```
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Entry

Built for the [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty on make.fun ($250, top 20 paid). Scored by mean pinball loss on the
logit scale; a prize needs a loss strictly below the best house entry.
