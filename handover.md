# handover.md

**Last updated:** 2026-10-11

Week 11 Thursday slice: TreeSHAP on LOFO P10/P50 RUL models, sensor-level ranking CSV for deployment, global beeswarm + local timeline figures. Prior: MAPIE CQR, `run_twin.py`, quantile LOFO. The thermal model is not refit here.

---

## What is done

| Item | Status |
|------|--------|
| Git remote `origin` → [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults) | Done |
| [AGENT.md](AGENT.md) agent rules, pre-commit file-size check + pytest | Done |
| [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md) — cap, LOFO, quantile + CQR, coverage caveat, P10 table | Done |
| [docs/WEEK_11_TODO.md](docs/WEEK_11_TODO.md) — Monday through Wednesday conformal checked; Thursday SHAP checked | Done |
| [docs/EXPLANATIONS.md](docs/EXPLANATIONS.md) — sensor map, SHAP outputs, field-intuition section | Done |
| `src/rul/labels.py` — daily aggregate, gearbox join, censoring (pandas 2.x day count fix) | Done |
| `src/rul/evaluate.py` — constant and linear baselines, MAE, pinball, coverage | Done |
| `src/rul/quantile.py` — LightGBM q10/q50/q90 (+ optional quantile list), LOFO, PDP, warnings, bundle save/load | Done |
| `src/rul/conformal.py` — time-ordered CQR calibration, coverage and width vs horizon | Done |
| `src/rul/plots.py` — raw + CQR band PNGs; four-panel twin timeline | Done |
| `src/rul/explain.py` — TreeSHAP global/local, sensor aggregation, ranking CSV | Done |
| `scripts/evaluate_quantile_rul.py` — CLI → `reports/rul/` incl. conformal CSVs | Done |
| `scripts/run_twin.py` — SCADA → hybrid twin → conformal RUL per turbine-day | Done |
| `scripts/explain_rul.py` — LOFO bundle + SHAP → `results/` | Done |
| Tests on fixtures (no EDP download, no twin import in pytest) | 34 passed on 2026-10-11 |
| Twin scored CSV export for four catalog events (warning table filled) | Pending export |
| Streamlit/FastAPI deployment, IEEE 39-bus notebook | Not started |

### Test run (2026-10-11)

Use the project venv (`.venv`) after `pip install -e ".[dev]"` (includes `mapie`, `shap`, `joblib`).

`pytest tests/ -q`: 34 passed. `python scripts/check_file_size.py`: all scanned files are at or under 1,000 lines.

---

## Repo layout

```
grid-gnn-faults/
├── AGENT.md
├── handover.md
├── README.md
├── docs/RUL_DESIGN.md
├── docs/WEEK_11_TODO.md
├── docs/EXPLANATIONS.md
├── src/rul/
│   ├── labels.py
│   ├── evaluate.py
│   ├── quantile.py
│   ├── conformal.py
│   ├── plots.py
│   └── explain.py
├── scripts/
│   ├── check_file_size.py
│   ├── evaluate_quantile_rul.py
│   ├── explain_rul.py
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
| `save_quantile_bundle` / `load_quantile_bundle` | `rul.quantile` | Joblib persistence for SHAP and deployment |
| `evaluate_quantile_model` | `rul.quantile` | LOFO quantile metrics, coverage, warnings, PDP |
| `evaluate_conformal_rul` | `rul.conformal` | LOFO CQR at 80%/90%, pooled coverage, width table |
| `predict_conformal_rul_for_frame` | `rul.conformal` | Fleet train/cal → conformal RUL on a predict frame |
| `write_all_band_figures` | `rul.plots` | Headline PNG per held-out event (raw + CQR) |
| `plot_headline_timeline` | `rul.plots` | Four-panel twin timeline with RUL band |
| `run_explain` | `rul.explain` | SHAP figures + `shap_sensor_ranking.csv` for one LOFO fold |

Import `rul.conformal` and `rul.explain` directly (not via `import rul`) so pytest does not load MAPIE/SHAP before lighter tests.

Input columns from the twin, already computed: `oil_residual`, `bear_residual`, `anomaly_score`, `theta_oil`, `theta_bear`, `delta_bear_oil`. Failure log columns: `Turbine_ID`, `Timestamp`, `Component`, `Remarks` with `Component == "GEARBOX"`.

Catalog in the twin config, also stored as `CATALOGUED_GEARBOX_EVENTS`: **4 events, 3 turbines** (T01 2016-07-18, T06 2017-10-17, T09 2016-10-11, T09 2017-10-18). `summarize_events` is the count that matters for a local CSV.

**CLI**

```powershell
pip install -e ".[dev]"
pytest tests/ -q
python scripts/check_file_size.py
python scripts/evaluate_quantile_rul.py --labeled daily.csv --out reports/rul
python scripts/run_twin.py --turbine T06 --out reports/twin
python scripts/explain_rul.py --labeled daily.csv --out results
```

---

## Suggested next step

Friday reflection doc; Streamlit/FastAPI twin view reading `results/shap_sensor_ranking.csv`; IEEE 39-bus GNN notebook. Do not refit the ODE.

---

## Key commit

(Pending) — TreeSHAP sensor explanations, bundle save/load, `explain_rul.py` (prior: `68dc925` MAPIE CQR RUL, `run_twin.py` timeline).
