# handover.md

**Last updated:** 2026-10-06

Week 11 Monday slice: gearbox RUL labels and leave-one-failure-out baselines on top of [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin). The thermal model is not refit here.

---

## What is done

| Item | Status |
|------|--------|
| Git remote `origin` → [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults) | Done |
| [AGENT.md](AGENT.md) agent rules, pre-commit file-size check + pytest | Done |
| [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md) — 90-day C-MAPSS cap, daily contract, 4-event catalog, LOFO | Done |
| [docs/WEEK_11_TODO.md](docs/WEEK_11_TODO.md) — Monday checked; Tue–Fri open | Done |
| `src/rul/labels.py` — daily aggregate, gearbox join, censoring | Done |
| `src/rul/evaluate.py` — constant and linear baselines, MAE, pinball, coverage | Done |
| Tests on fixtures (no EDP download, no twin import) | 15 passed on 2026-10-06 |
| Quantile model, conformal intervals, SHAP, deployment, IEEE 39-bus notebook | Not started |

### Test run (2026-10-06)

`pytest tests/ -q`: 15 passed. `python scripts/check_file_size.py`: all scanned files are at or under 1,000 lines.

---

## Repo layout

```
grid-gnn-faults/
├── AGENT.md
├── handover.md
├── README.md
├── docs/RUL_DESIGN.md
├── docs/WEEK_11_TODO.md
├── src/rul/
│   ├── labels.py
│   └── evaluate.py
├── scripts/check_file_size.py
├── tests/
└── tests/fixtures/
```

---

## Core module API

| Symbol | Module | Purpose |
|--------|--------|---------|
| `daily_from_scored_frame` | `rul.labels` | 10-minute hybrid residual, anomaly score, and theta features → turbine-day |
| `attach_rul_labels` | `rul.labels` | Capped RUL, censor flag, next gearbox event |
| `summarize_events` | `rul.labels` | Count gearbox log lines actually passed in |
| `leave_one_failure_out` | `rul.evaluate` | Test the 90-day window; train on every other turbine |
| `constant_baseline` | `rul.evaluate` | Predict `RUL_max` at every quantile |
| `fit_linear_anomaly_baseline` | `rul.evaluate` | OLS from `anomaly_score_mean`, training fold only |
| `score_predictions` | `rul.evaluate` | MAE, pinball at 0.1/0.5/0.9, interval coverage |
| `evaluate_baselines` | `rul.evaluate` | Both baselines on every fold |

Input columns from the twin, already computed: `oil_residual`, `bear_residual`, `anomaly_score`, `theta_oil`, `theta_bear`, `delta_bear_oil`. Failure log columns: `Turbine_ID`, `Timestamp`, `Component`, `Remarks` with `Component == "GEARBOX"`.

Catalog in the twin config, also stored as `CATALOGUED_GEARBOX_EVENTS`: **4 events, 3 turbines** (T01 2016-07-18, T06 2017-10-17, T09 2016-10-11, T09 2017-10-18). `summarize_events` is the count that matters for a local CSV.

**CLI**

```powershell
pip install -e ".[dev]"
pytest tests/ -q
python scripts/check_file_size.py
```

---

## Suggested next step

Tuesday: a q10/q50/q90 model that calls `evaluate_baselines` and `score_predictions`. Keep the split. Do not refit the ODE.

---

## Key commit

Not committed in this session.
