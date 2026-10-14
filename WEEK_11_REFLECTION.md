# Week 11 reflection

**Date:** 2026-10-13

## What did I build?

This week closed the capstone loop from Week 10’s hybrid thermal twin to something an O&M desk could actually open:

- **Gearbox RUL head** in [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults): daily aggregates of hybrid residuals, anomaly score, and θ features → LightGBM **P10/P50/P90**, **MAPIE CQR** at 80%/90%, **leave-one-failure-out** evaluation on four catalogued EDP gearbox events, maintenance lead-time from pessimistic P10, and **TreeSHAP** rolled up to six sensor groups ([docs/EXPLANATIONS.md](docs/EXPLANATIONS.md), `results/shap_sensor_ranking.csv`).
- **Serving layer** in [wind-digital-twin/service/](https://github.com/mehmetertac/wind-digital-twin): FastAPI **`POST /twin/score`** (SCADA window → physics-expected oil temperature, hybrid residual, anomaly score + threshold) and **`POST /twin/rul`** (daily features → quantiles, conformal band, top-five sensor attributions), models loaded once from `service/models/`, pytest smokes, GitHub Actions CI, Docker compose for API + Streamlit.
- **Streamlit dashboard** (three panels): turbine selector, scrub through the failure window — actual vs physics oil temperature with residual, anomaly score vs threshold, RUL P50 with CQR band, global P50 SHAP bar chart. README assets: `docs/figures/twin_dashboard.png` and `.gif`.

The physics ODE, GBM corrector, and isolation forest were **not** refit in grid-gnn-faults; the serving export retrains the Week 10 detector only for **synthetic smoke** artifacts so CI never downloads EDP.

## What’s still fuzzy?

**RUL label convention.** The default is C-MAPSS-style `min(90, days_to_event)` with censored healthy rows pinned at 90. That matches Week 10’s 90-day training buffer, but it is not the same object a survival model would use: censored turbines are “at least 90 days,” not “exactly 90.” Calling `rul_max=60` in [src/rul/labels.py](src/rul/labels.py) changes the plateau and the LOFO windows — any interview answer should state which cap was used.

**Four events, two on T09.** Holding out one T09 line drops the other from training. The label resets after the first log line; that is honest for the maintenance log, but it is not a independent sample size of four turbines.

**Conformal validity.** Split conformal coverage is a **marginal** guarantee under exchangeability. Here, calibration tails are tiny, time-ordered, and drawn from turbines that are not exchangeable with the held-out run-to-failure window. The **pooled** row in `conformal_coverage.csv` is a summary, not one trial. Gaps versus 80%/90% are expected noise, not a knob to over-tune. The deployment API uses **fixed CQR widening constants** fit on synthetic data — honest for demo, not a fleet guarantee.

**Capped regression vs survival.** A Cox or discrete-time survival model would treat healthy turbines as right-censored without pretending `RUL = 90` is a realizable day count. Quantile boosting on a capped target is easier to ship and aligns with the C-MAPSS literature, but it **overstates precision** on censored rows and cannot emit “probability of failure before day *d*” without extra work.

**SHAP on a LOFO fold.** Global sensor ranks describe the **held-out training fold**, not a causal “this bearing caused the failure.” With one-digit events, local timelines are fragile; the synthetic pytest panel already showed windows too short for a field-facing “first negative day” sentence.

## How does this twin compare to CMS / OEM tooling — and what would an O&M manager push back on?

From Week 10 ([WEEK_10_REFLECTION.md](https://github.com/mehmetertac/wind-digital-twin/blob/main/WEEK_10_REFLECTION.md)) and field notes: typical **CMS** on turbines emphasizes **vibration** (envelope, band energy, ISO severity), **oil debris / particle counters**, and OEM-curated thresholds with work-order integration. EDP gives **10-minute thermal SCADA only** — no accelerometers, no ferrography.

| OEM / field CMS habit | This capstone twin |
| --- | --- |
| Vibration leads temperature by days–weeks on many gear faults | Thermal hybrid residual + IF only |
| Fixed high-temperature alarms | Load-conditioned physics expectation + learned healthy correction |
| Fleet benchmark KPIs with vendor support | Four gearbox log lines, LOFO, explicit coverage caveat |
| “Replace bearing in ~N hours” work packages | Calendar-day **P10/P50/P90** advisory band + conformal width |

**Pushback I would expect in a real O&M review:**

1. **“Do not book a crane on your P10.”** Four labeled events cannot support fleet RUL SLAs; the honest output is trend + band + inspection planning horizon, not a day-count contract.
2. **“Your anomaly score is not a probability.”** It is a negated isolation-forest score on engineered windows — comparable to healthy history, not to a calibrated fault probability.
3. **“Which failure mode?”** T01 pump, T06 bearing, T09 repair/noise are different root causes; one thermal score and one RUL head do not separate them — vibration and oil chemistry would.
4. **“Why should I trust SHAP?”** Sensor ranks help prioritize which **daily features** moved the **regression** RUL; they are not a substitute for borescope or CMS triage.
5. **“Temperature lags the physics.”** At 10-minute SCADA, sub-hour bearing transients are smoothed away; this complements CMS, it does not replace early mechanical warning.

That is the right bar: **advisory twin for inspection scheduling**, aligned with Week 10’s maintenance narrative, not a certified OEM replacement.

## Next

- IEEE 39-bus GNN fault-localization notebook (still open in [docs/WEEK_11_TODO.md](docs/WEEK_11_TODO.md)).
- Re-export `service/models/` and SHAP ranks on a **real** EDP scored CSV when the fleet export exists; fill the P10 lead-time table in [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md).
