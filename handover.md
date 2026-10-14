# handover.md

**Last updated:** 2026-10-14

Week 11 Friday slice: probabilistic gearbox RUL (quantile + CQR + LOFO + SHAP), deployment in [wind-digital-twin/service](https://github.com/mehmetertac/wind-digital-twin/tree/main/service), and [WEEK_11_REFLECTION.md](WEEK_11_REFLECTION.md). The thermal model is not refit here.

---

## What is done

| Item | Status |
|------|--------|
| Git remote `origin` → [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults) | Done |
| [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md), [docs/EXPLANATIONS.md](docs/EXPLANATIONS.md) | Done |
| `src/rul/*` — labels, evaluate, quantile, conformal, plots, explain | Done |
| Scripts — `evaluate_quantile_rul.py`, `run_twin.py`, `explain_rul.py` | Done |
| Tests on fixtures (34+ passed; no EDP download in CI) | Done |
| **Deployment** — FastAPI `/twin/score`, `/twin/rul` + Streamlit dashboard in wind-digital-twin | Done |
| [WEEK_11_REFLECTION.md](WEEK_11_REFLECTION.md) | Done |
| Twin scored CSV export for four catalog events (warning table filled) | Pending export |
| IEEE 39-bus GNN notebook | Not started |

### Test run (2026-10-13)

`pytest tests/ -q` in grid-gnn-faults; serving tests run in wind-digital-twin (`pytest tests/ service/tests/ -q`, 64 passed after deployment).

---

## Repo layout

```
grid-gnn-faults/
├── WEEK_11_REFLECTION.md
├── handover.md
├── docs/RUL_DESIGN.md
├── src/rul/
├── scripts/
└── tests/
```

Serving layer (sibling repo):

```
wind-digital-twin/service/
├── src/twin_service/
├── dashboard/
├── models/
└── scripts/export_models.py
```

---

## Core module API

Unchanged from prior handover — see [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md) and `src/rul/`.

**Deployment (wind-digital-twin)**

| Endpoint | Purpose |
|----------|---------|
| `POST /twin/score` | SCADA window → expected temp, residual, anomaly score |
| `POST /twin/rul` | Daily features → P10/P50/P90, CQR band, top-5 sensor attributions |
| `GET /twin/series` | Dashboard timeline for failure-window scrub |

---

## Suggested next step

IEEE 39-bus GNN fault-localization notebook (one accuracy table, one figure). Re-export twin scored CSV and fill P10 lead-time table in RUL_DESIGN when fleet export exists.

---

## Key commit

`4f16067` — Week 11 reflection and deployment docs (prior: `76a3dc6` TreeSHAP sensor explanations).
