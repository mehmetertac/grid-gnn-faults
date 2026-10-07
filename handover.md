# handover.md

**Last updated:** 2026-10-07

Week 11 Tuesday slice: LightGBM quantile gearbox RUL (q10/q50/q90) on Monday daily features, LOFO figures, and P10 maintenance lead-time table in [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md). The thermal model is not refit here.

---

## What is done

| Item | Status |
|------|--------|
| Git remote `origin` → [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults) | Done |
| [AGENT.md](AGENT.md) agent rules, pre-commit file-size check + pytest | Done |
| [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md) — cap, LOFO, quantile model, coverage note, P10 warning table | Done |
| [docs/WEEK_11_TODO.md](docs/WEEK_11_TODO.md) — Monday + Tuesday quantile checked | Done |
| `src/rul/labels.py` — daily aggregate, gearbox join, censoring (pandas 2.x day count fix) | Done |
| `src/rul/evaluate.py` — constant and linear baselines, MAE, pinball, coverage | Done |
| `src/rul/quantile.py` — LightGBM q10/q50/q90, LOFO, PDP, warning lead time | Done |
| `src/rul/plots.py` — true RUL vs P50 with P10–P90 band PNGs | Done |
| `scripts/evaluate_quantile_rul.py` — CLI → `reports/rul/` | Done |
| Tests on fixtures (no EDP download, no twin import) | 22 passed on 2026-10-07 |
| Twin scored CSV export for four catalog events (warning table filled) | Pending export |
| Conformal intervals, SHAP, deployment, IEEE 39-bus notebook | Not started |

### Test run (2026-10-07)

`pytest tests/ -q`: 22 passed. `python scripts/check_file_size.py`: all scanned files are at or under 1,000 lines.

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
│   ├── evaluate.py
│   ├── quantile.py
│   └── plots.py
├── scripts/
│   ├── check_file_size.py
│   └── evaluate_quantile_rul.py
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
| `fit_quantile_models` | `rul.quantile` | Three LightGBM quantile boosters on daily features |
| `evaluate_quantile_model` | `rul.quantile` | LOFO quantile metrics, coverage, warnings, PDP |
| `write_all_band_figures` | `rul.plots` | Headline PNG per held-out event |

Input columns from the twin, already computed: `oil_residual`, `bear_residual`, `anomaly_score`, `theta_oil`, `theta_bear`, `delta_bear_oil`. Failure log columns: `Turbine_ID`, `Timestamp`, `Component`, `Remarks` with `Component == "GEARBOX"`.

Catalog in the twin config, also stored as `CATALOGUED_GEARBOX_EVENTS`: **4 events, 3 turbines** (T01 2016-07-18, T06 2017-10-17, T09 2016-10-11, T09 2017-10-18). `summarize_events` is the count that matters for a local CSV.

**CLI**

```powershell
pip install -e ".[dev]"
pytest tests/ -q
python scripts/check_file_size.py
python scripts/evaluate_quantile_rul.py --labeled daily.csv --out reports/rul
```

---

## Suggested next step

Wednesday: conformal intervals on the held-out run-to-failure windows; check coverage after calibration. Do not refit the ODE.

---

## Key commit

`57ff954` — LightGBM quantile gearbox RUL, LOFO band figures, pandas 2.x label fix (prior: `074f1ad`).
