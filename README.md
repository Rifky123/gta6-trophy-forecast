# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Rifky123/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Rifky123/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts the earn rate of every Grand Theft Auto VI PS5 trophy on Exophase at a target time, as
five quantiles per trophy, from the recorded history of the game's trophy page. The page is read
as the house recorded it: reads every few minutes whose values change only when Exophase refreshes
its counts, so the entry works on the sequence of distinct states.

Each trophy's rate is earners over players: `rate = 100 * E / P`. The players curve is
extrapolated with decelerating families on `ln P` (power law, saturating, quadratic in log time),
blended by fit quality; each trophy's earner growth is estimated per state interval as
`dE = c * dP + h * P dt` — new entrants carrying the trophy at share `c`, tracked players earning
it at hazard `h` per player-hour, decaying as the launch cohort finishes the content. Samples draw
the players curve across the families' disagreement (a residual bootstrap), rebuild each trophy's
earner count, and the five quantiles are the sample percentiles of the rate. When the page was
frozen for the whole history (launch night), players come from observed launch anchors and earners
from the same accumulation split by earner share.

Standard library only, deterministic, one case in and one forecast out, at most 8 seconds.

## Run it

```sh
python3 -P -s main.py  # reads /input/<id>.in, writes /output/<id>.out
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=essp25st&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
