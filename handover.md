# handover.md

**Last updated:** 2026-11-07

Week 11 gearbox RUL + deployment remain in place. **Track C day 2** added GCN training, hop-distance metrics, edge-drop robustness check, and README figures. Capstone polish and tags live in sibling [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin).

---

## What is done

| Item | Status |
|------|--------|
| Git remote `origin` → [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults) | Done |
| [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md), [docs/EXPLANATIONS.md](docs/EXPLANATIONS.md) | Done |
| `src/rul/*` — labels, evaluate, quantile, conformal, plots, explain, [scored_export.py](src/rul/scored_export.py) | Done |
| **`scripts/run_twin.py --fleet`** — fleet `scored_10min.csv` + `gearbox_failures.csv` | Done |
| Synthetic fleet LOFO table in [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md) + twin README (P10 lead **`none`** on cap-saturated run) | Done |
| **Deployment** — FastAPI `/twin/score`, `/twin/rul` + Streamlit in wind-digital-twin | Done |
| **Track C day 2** — GCN vs MLP, hop metrics, [docs/figures/fault_error_layout.png](docs/figures/fault_error_layout.png) | Done |
| Twin scored CSV export for four catalog events (warning table filled) | Done (synthetic EDP, local `reports/`) |
| Git tag **`v0.1.0`** pushed to origin | Done |
| wind-digital-twin README v1.0.0 polish + tag **`v1.0.0`** | Done (sibling repo) |

### Test run (2026-11-07)

`pytest tests/test_scored_export.py tests/test_grid_faults.py -q` — fleet concat + grid tests (run full `pytest tests/ -q` before push).

---

## Repo layout

```
grid-gnn-faults/
├── handover.md
├── docs/GRID_FAULTS.md
├── docs/RUL_DESIGN.md
├── notebooks/01_fault_localization_gnn.ipynb
├── src/rul/
├── src/grid/
├── scripts/run_twin.py          # --fleet for scored export
├── scripts/evaluate_quantile_rul.py
├── scripts/train_fault_gnn.py
└── tests/
```

---

## Core module API

**RUL fleet export**

| Entry | Purpose |
|-------|---------|
| [scripts/run_twin.py](scripts/run_twin.py) | `--fleet` → per-turbine hybrid score without refitting ODE elsewhere |
| [src/rul/scored_export.py](src/rul/scored_export.py) | `concat_scored_frames` for fleet CSV |
| [scripts/evaluate_quantile_rul.py](scripts/evaluate_quantile_rul.py) | LOFO metrics, `warnings.csv`, conformal tables |

**Grid fault localization** — see [docs/GRID_FAULTS.md](docs/GRID_FAULTS.md).

---

## Suggested next step

Re-run fleet export on **real** EDP when downloaded; expect P10 lead integers to move off `none` only if daily quantiles leave the 90-day cap before failure. Optional: re-simulate voltages after line outage for a stronger GNN robustness story.

---

## Key commit

Pending local commit — synthetic fleet export, T09 in sibling synthetic EDP, filled RUL README tables (prior: `c26d0ec` Track C day 2).
