# handover.md

**Last updated:** 2026-10-08

Week 11 Wednesday slice: MAPIE CQR on gearbox RUL (80% and 90%), honest LOFO coverage tables, raw vs conformal band figures, and [scripts/run_twin.py](scripts/run_twin.py) end-to-end timeline. The thermal model is not refit here.

---

## What is done

| Item | Status |
|------|--------|
| Git remote `origin` → [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults) | Done |
| [AGENT.md](AGENT.md) agent rules, pre-commit file-size check + pytest | Done |
| [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md) — cap, LOFO, quantile + CQR, coverage caveat, P10 table | Done |
| [docs/WEEK_11_TODO.md](docs/WEEK_11_TODO.md) — Monday through Wednesday conformal checked | Done |
| `src/rul/labels.py` — daily aggregate, gearbox join, censoring (pandas 2.x day count fix) | Done |
| `src/rul/evaluate.py` — constant and linear baselines, MAE, pinball, coverage | Done |
| `src/rul/quantile.py` — LightGBM q10/q50/q90 (+ optional quantile list), LOFO, PDP, warnings | Done |
| `src/rul/conformal.py` — time-ordered CQR calibration, coverage and width vs horizon | Done |
| `src/rul/plots.py` — raw + CQR band PNGs; four-panel twin timeline | Done |
| `scripts/evaluate_quantile_rul.py` — CLI → `reports/rul/` incl. conformal CSVs | Done |
| `scripts/run_twin.py` — SCADA → hybrid twin → conformal RUL per turbine-day | Done |
| Tests on fixtures (no EDP download, no twin import in pytest) | 27 passed on 2026-10-08 |
| Twin scored CSV export for four catalog events (warning table filled) | Pending export |
| SHAP, deployment, IEEE 39-bus notebook | Not started |

### Test run (2026-10-08)

Use the project venv (`.venv`) after `pip install -e ".[dev]"` (includes `mapie`).

`pytest tests/ -q`: 27 passed. `python scripts/check_file_size.py`: all scanned files are at or under 1,000 lines.

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
│   ├── conformal.py
│   └── plots.py
├── scripts/
│   ├── check_file_size.py
│   ├── evaluate_quantile_rul.py
│   └── run_twin.py
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
| `fit_quantile_models` | `rul.quantile` | LightGBM quantile boosters on daily features |
| `evaluate_quantile_model` | `rul.quantile` | LOFO quantile metrics, coverage, warnings, PDP |
| `evaluate_conformal_rul` | `rul.conformal` | LOFO CQR at 80%/90%, pooled coverage, width table |
| `predict_conformal_rul_for_frame` | `rul.conformal` | Fleet train/cal → conformal RUL on a predict frame |
| `write_all_band_figures` | `rul.plots` | Headline PNG per held-out event (raw + CQR) |
| `plot_headline_timeline` | `rul.plots` | Four-panel twin timeline with RUL band |

Import `rul.conformal` directly (not via `import rul`) so pytest does not load MAPIE before LightGBM quantile tests.

Input columns from the twin, already computed: `oil_residual`, `bear_residual`, `anomaly_score`, `theta_oil`, `theta_bear`, `delta_bear_oil`. Failure log columns: `Turbine_ID`, `Timestamp`, `Component`, `Remarks` with `Component == "GEARBOX"`.

Catalog in the twin config, also stored as `CATALOGUED_GEARBOX_EVENTS`: **4 events, 3 turbines** (T01 2016-07-18, T06 2017-10-17, T09 2016-10-11, T09 2017-10-18). `summarize_events` is the count that matters for a local CSV.

**CLI**

```powershell
pip install -e ".[dev]"
pytest tests/ -q
python scripts/check_file_size.py
python scripts/evaluate_quantile_rul.py --labeled daily.csv --out reports/rul
python scripts/run_twin.py --turbine T06 --out reports/twin
```

---

## Suggested next step

Thursday: SHAP or attention on daily features; small Streamlit/FastAPI twin view. Do not refit the ODE.

---

## Key commit

`68dc925` — MAPIE CQR RUL at 80%/90%, conformal coverage CSVs, `run_twin.py` timeline (prior: `57ff954` quantile RUL).
