# Week 11 TODO

Capstone week: finish the wind-turbine twin from [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin) with a probabilistic gearbox RUL, then add one grid-side GNN demo. Do not refit the thermal model.

Reading list: [RUL_DESIGN.md](RUL_DESIGN.md), [handover.md](../handover.md), [README.md](../README.md), [AGENT.md](../AGENT.md).

## Monday — definition, dataset, baselines

- [x] Write the RUL target, 90-day C-MAPSS cap, daily cadence, and censoring rule in [RUL_DESIGN.md](RUL_DESIGN.md).
- [x] Build [src/rul/labels.py](../src/rul/labels.py): daily aggregates of the Week 10 hybrid residual, anomaly score, and theta features, joined to gearbox events. Keep censored healthy turbines. Record the event count with `summarize_events` (catalog is 4).
- [x] Write [src/rul/evaluate.py](../src/rul/evaluate.py): constant baseline, linear anomaly baseline, leave-one-failure-out, MAE, pinball at 0.1/0.5/0.9, and interval coverage.

## Tuesday and Wednesday — call the protocol

- [x] Fit a model that emits q10, q50, and q90. Score it with `evaluate_baselines` / `score_predictions`. Do not invent a second split.
- [ ] Add conformal intervals and check coverage on the held-out run-to-failure windows. State the single-digit event count in the write-up.

## Thursday — explanation, deployment, grid demo

- [ ] SHAP or attention: which daily sensors move the RUL prediction.
- [ ] A small Streamlit or FastAPI view of the twin.
- [ ] One notebook: GNN fault localization on the IEEE 39-bus system with torch-geometric and pandapower. One accuracy table, one figure.

## Friday

- [ ] Write `WEEK_11_REFLECTION.md`.

## All week

- Probabilistic outputs on every prediction.
- Time-ordered, leave-one-failure-out evaluation. The failure window does not enter training.
- A maintenance or grid-operations section in each README.
- Docs updated before any push. Tests run before commit.
