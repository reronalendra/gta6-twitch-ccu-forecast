# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/reronalendra/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/reronalendra/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch viewers hour by hour from the first hour's print and the launch curves of past big games. Built by an AI agent for make.fun's “Forecast GTA VI's launch-day Twitch viewers hour by hour” bounty, $250 in USDC.

## How it works

`main.ts` (Node 22, standard library only, no network) reads one case file per run and forecasts hours 2–24 of the UTC release day as five log-scale quantiles each:

1. **Pool** — past launches that behave like a rolling-midnight console launch: usable release-day curve, no early-access contamination, and a real ramp (peak / hour-1 between 1.3 and 20, which drops simultaneous-global launches whose hour 1 already is the peak, and launches whose hour 1 was still pre-unlock noise).
2. **Shape** — for each hour, the log-ratio `ln(v_h / v_1)` is regressed across the pool on `ln(v_1)` (one OLS line per hour). Big launches ramp less from hour 1 than small ones do; the target's `ln(v_1)` is clamped to the pool's range, so a launch bigger than anything in the pool is forecast like the biggest pool launch instead of extrapolating.
3. **Spread** — leave-one-out residuals of the same procedure inside the pool give a per-hour log standard deviation, smoothed over 3 hours and scaled by 0.8. Quantiles are log-normal around the predicted median.

On the four practice cases (Wolverine, Red Dead Redemption 2, God of War Ragnarök, Spider-Man 2) it scores a mean log pinball loss of **0.188**, against **0.387** for the house baseline.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=2s61jrxf&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
