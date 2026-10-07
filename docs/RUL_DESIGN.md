# RUL design

Monday's contract for gearbox remaining useful life. The physics fit stays in [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin). This repo consumes its scored series and its gearbox log.

Related docs: [WEEK_11_TODO.md](WEEK_11_TODO.md), [handover.md](../handover.md), [README.md](../README.md), [AGENT.md](../AGENT.md).

## Target

RUL is the number of calendar days until the next logged gearbox event.

A gearbox event is a failure-log row with `Component == "GEARBOX"`. That includes repairs and noise lines, not only the turbines Week 10 treated as primary failures. Week 10 names are `Turbine_ID`, `Timestamp`, `Component`, and `Remarks`. `GearboxFailure` in the twin is the same record after filtering.

The label uses the C-MAPSS piecewise-linear cap, stated here explicitly:

```text
RUL(t) = min(RUL_max, days_to_event)
RUL_max = 90
```

Far from the event the label is flat at 90. Inside the last 90 days it falls by one day per day. `60` is a callable sensitivity (`rul_max=60`), not the default.

90 is the same cutoff as `DEFAULT_BUFFER_DAYS` in the twin. Week 10 refused to train the thermal model and the detector on that window. The RUL window is that same interval.

## Cadence

One row per turbine per calendar day, in UTC.

The twin emits 10-minute series. `daily_from_scored_frame` rolls them up. The twin's own 10-minute EWMA (span 36) and rolling windows (6, 36, 144 samples) stay inside the detector. They are not copied onto the RUL row.

The calendar day of the log timestamp is dropped, because that SCADA day is partial. The last kept day has `days_to_event >= 1`.

After an event, the clock resets toward the next gearbox event on that turbine. This matters for T09, which has two log lines about a year apart.

## Censoring

A turbine-day with no later gearbox event gets `rul = 90` and `censored = True`. The true RUL is at least 90. Those rows stay in the training table so the model sees healthy operation. Metrics do not treat 90 as an exact label for them.

A day that does have a later event, but sits more than 90 days before it, is not censored. Its label is the flat cap, `rul = 90`, and `days_to_event` keeps the true gap. That is the C-MAPSS early-life plateau.

## What Week 10 must already have computed

Do not refit the lumped ODE, the gradient-boosting corrector, or the isolation forest here. Build the input frame in the twin from an already fitted pipeline:

- `HybridThermalCorrector.dual_hybrid_residuals` → `oil_residual`, `bear_residual`
- degradation = `-residual` (higher means hotter than the ODE expected), from `degradation_signal`
- `HybridResidualPipeline.score` → `anomaly_score` (negated isolation-forest decision function; higher is more anomalous; not a probability in `[0, 1]`)
- `build_fusion_feature_frame` → `theta_oil`, `theta_bear`, `delta_bear_oil`

`theta_oil` and `theta_bear` are `(gear_temp - nac_temp) / power`: temperature rise per unit power.

Daily columns written by [src/rul/labels.py](../src/rul/labels.py):

| Column | Meaning |
| --- | --- |
| `oil_deg_mean`, `oil_deg_std` | Daily mean and sample std of `-oil_residual` |
| `bear_deg_mean`, `bear_deg_std` | Same for the bearing residual |
| `anomaly_score_mean`, `anomaly_score_max` | Daily mean and max of the twin score |
| `oil_deg_slope_7d`, `bear_deg_slope_7d` | Slope of the daily mean over the trailing 7 observed days |
| `theta_oil_mean`, `theta_bear_mean`, `delta_bear_oil_mean` | Daily means of the fusion features |

The first 6 days of each turbine have a null slope. Sample standard deviation is null when a day has only one 10-minute row.

## How many gearbox events

`GEARBOX_FAILURES` in the twin config lists **4 events on 3 turbines**:

| Turbine | Timestamp (UTC) | Remarks |
| --- | --- | --- |
| T01 | 2016-07-18 02:10 | Gearbox pump damaged |
| T06 | 2017-10-17 08:38 | Gearbox bearings damaged |
| T09 | 2016-10-11 08:06 | Gearbox repaired |
| T09 | 2017-10-18 08:32 | Gearbox noise |

Week 10's `FAILURE_TURBINES` is only T01 and T06. T09 is two log lines on one machine. Every other turbine is right-censored for this target.

`summarize_events` recounts the frame you pass. A CSV under the twin's `data/raw/edp/` can differ from this catalog. The scored test fixture is synthetic. The failure fixture repeats these four lines and adds one non-gearbox row so the component filter has something to drop. Four is the honest catalog until a local log says otherwise. That is a single-digit failure set. A P10/P50/P90 with checked coverage on these events is the claim this project can support.

## Evaluation

Leave-one-failure-out, one fold per gearbox event that still has labeled rows:

- Test: that event's rows with `0 < days_to_event <= 90`.
- Train: every other turbine, including censored healthy rows labeled 90.
- The held-out turbine is absent from training. Holding out either T09 event also drops T09's other event.

Folds with no training rows are still returned.

Scored only on the test window:

- MAE on the median (q50)
- Pinball loss at q = 0.1, 0.5, and 0.9
- Coverage of the interval `[q10, q90]`

Baselines, fit inside the training fold only:

1. Constant: every quantile equals 90. The interval has zero width. On a decreasing window, coverage counts only the day whose true RUL is already 90.
2. Linear: `rul = clip(a + b * anomaly_score_mean, 0, 90)`, ordinary least squares on the training rows. The score is unbounded. The point prediction is copied to q10, q50, and q90 until a quantile model exists, so this interval is degenerate too.

Call these from [src/rul/evaluate.py](../src/rul/evaluate.py): `leave_one_failure_out`, `constant_baseline`, `fit_linear_anomaly_baseline`, `score_predictions`, `evaluate_baselines`.

### Quantile model (LightGBM)

Three `LGBMRegressor` models at q = 0.1, 0.5, and 0.9 with `objective="quantile"`. Inputs are exactly `DAILY_FEATURE_COLUMNS` from [src/rul/labels.py](../src/rul/labels.py). Censored rows at the cap stay in training. Implementation: [src/rul/quantile.py](../src/rul/quantile.py).

After prediction, each quantile is clipped to `[0, 90]` and reordered so q10 ≤ q50 ≤ q90 on every row.

**Monotonic sanity.** The build tries `monotone_constraints = -1` on `anomaly_score_mean` and `anomaly_score_max` (higher anomaly → lower RUL). LightGBM 4.x in this environment rejects monotone constraints with the quantile objective; the code falls back to unconstrained quantile trees and sets `monotone_applied = False` on the metrics row. Either way, partial dependence on `anomaly_score_mean` (other features at training medians) must be non-increasing for q50 within 1e-4 days when constraints are active; when they are not, the PDP table is still written for manual review.

**Coverage vs nominal 80%.** Interval coverage uses `[q10, q90]` against true RUL on the held-out window. Nominal central coverage is 80%. Report `coverage_gap = pi_coverage - 0.8` per event in `coverage.csv`. Expect the band to be **too narrow early** in the run (`days_to_event > 45`): most training labels sit on the 90-day plateau, so the raw quantile interval undercovers there. Wednesday conformal calibration is meant to fix that; this slice does not widen intervals.

**Headline figures.** [src/rul/plots.py](../src/rul/plots.py) writes one PNG per held-out event: true RUL, P50, and the P10–P90 band over the last 90 days before failure, with a dashed line at the maintenance threshold.

Run leave-one-failure-out quantile evaluation (baselines + model) with:

```powershell
python scripts/evaluate_quantile_rul.py --labeled path/to/daily_labeled.csv --out reports/rul
# or from 10-minute twin output:
python scripts/evaluate_quantile_rul.py --scored path/to/scored_10min.csv --failures path/to/gearbox_log.csv --out reports/rul
```

Outputs: `metrics.csv`, `coverage.csv`, `warnings.csv`, `pdp_anomaly_mean.csv`, and `figures/*_rul_band.png`.

**Scored export.** A full fleet scored frame (all six Week 10 columns: `oil_residual`, `bear_residual`, `anomaly_score`, `theta_oil`, `theta_bear`, `delta_bear_oil`) was not present under [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin) `results/` at implementation time (residual parquets and layer dumps do not include the full scored contract). Fill the warning column below after exporting a scored CSV from the twin without refitting the ODE or detector.

### Maintenance planning lead time (P10)

Use the **pessimistic** quantile: the first calendar day in the held-out window when **P10 RUL drops below 14 days**. The lead time is `days_to_event` on that day — how many days before failure the twin would have warned an O&M planner to schedule gearbox inspection.

| Event | Turbine | P10 first below 14 d (days before failure) |
| --- | --- | --- |
| T01 \| 2016-07-18 | T01 | — |
| T06 \| 2017-10-17 | T06 | — |
| T09 \| 2016-10-11 | T09 | — |
| T09 \| 2017-10-18 | T09 | — |

Replace em dashes with integers from `warnings.csv` after running the CLI on a twin scored export. Example sentence for interviews: *The twin would have given the O&M planner X days of warning before the gearbox event on T06.*

## What this means for maintenance

A capped RUL is a gearbox inspection planning window. Ninety days is the horizon on which Week 10 already withholds healthy training, so a prediction inside that horizon is a reason to schedule a look at the oil and bearing temperatures, the pump, and the maintenance log. A constant prediction of 90 days means the daily score still looks like the healthy rows. With four catalogued events, the interval is a statement about coverage on those run-to-failure windows, not a fleet-wide day-count you can book a crane against.
