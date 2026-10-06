# grid-gnn-faults

Probabilistic remaining useful life (RUL) for the Week 10 wind-turbine gearbox twin, with conformal intervals still to come. Later this repo also carries SHAP or attention explanations, a Streamlit/FastAPI view, and a small GNN fault-localization demo on the IEEE 39-bus system.

The thermal ODE, learned residual, and hybrid-residual detector stay in [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin). This repo aggregates the series that twin already emits.

Design: [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md). Status: [handover.md](handover.md). Week checklist: [docs/WEEK_11_TODO.md](docs/WEEK_11_TODO.md). Agent rules: [AGENT.md](AGENT.md).

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pre-commit install
pytest tests/ -q
python scripts/check_file_size.py
```

## What this means in maintenance terms

RUL here is calendar days until the next logged gearbox event, capped at 90. Ninety days is the same pre-failure buffer Week 10 already keeps out of thermal training, so a prediction inside that window is an inspection planning horizon for the oil, the bearing temperatures, and the gearbox maintenance log.

The published EDP catalog has four gearbox events on three turbines (T01 pump damage, T06 bearing damage, and two T09 lines). Turbines with no gearbox line are labeled as healthy at the cap and kept in training. A constant prediction of 90 days means the daily anomaly score still looks like those healthy rows. With a single-digit event count, the useful output is a P10/P50/P90 band whose coverage has been checked on the held-out run-to-failure windows.

## License

Apache-2.0 — see [LICENSE](LICENSE).
