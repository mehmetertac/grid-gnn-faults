# handover.md

**Last updated:** 2026-10-17

Week 11 gearbox RUL + deployment remain in place. **Track C day 2** added GCN training, hop-distance metrics, edge-drop robustness check, and README figures. Capstone polish and tags live in sibling [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin).

---

## What is done

| Item | Status |
|------|--------|
| Git remote `origin` → [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults) | Done |
| [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md), [docs/EXPLANATIONS.md](docs/EXPLANATIONS.md) | Done |
| `src/rul/*` — labels, evaluate, quantile, conformal, plots, explain | Done |
| **Deployment** — FastAPI `/twin/score`, `/twin/rul` + Streamlit in wind-digital-twin | Done |
| **Track C day 1** — `src/grid/*`, dataset, MLP baseline | Done |
| **Track C day 2** — [`src/grid/model.py`](src/grid/model.py), [`scripts/train_fault_gnn.py`](scripts/train_fault_gnn.py), GCN vs MLP table, [docs/figures/fault_error_layout.png](docs/figures/fault_error_layout.png) | Done |
| GCN test acc ≈ 0.662 vs MLP 0.956 (seed 0); mean hop error 0.85 vs 0.06 | Done (local `results/`) |
| Twin scored CSV export for four catalog events (warning table filled) | Pending export |
| wind-digital-twin README v1.0.0 polish + tag | See wind-digital-twin handover |

### Test run (2026-10-17)

`pytest tests/test_grid_faults.py -q` — 10 passed (torch/pandapower grid tests).

---

## Repo layout

```
grid-gnn-faults/
├── handover.md
├── docs/GRID_FAULTS.md
├── docs/figures/fault_error_layout.png
├── notebooks/01_fault_localization_gnn.ipynb
├── src/rul/
├── src/grid/          # simulate, graph, baseline, model
├── scripts/build_fault_dataset.py
├── scripts/train_fault_gnn.py
└── tests/
```

---

## Core module API

**Grid fault localization (`src/grid/`)**

| Module | Purpose |
|--------|---------|
| [simulate.py](src/grid/simulate.py) | case39 scenarios, Ybus bolted during-fault voltages |
| [graph.py](src/grid/graph.py) | Topology, split, z-score, PyG `Data`, ΔV plot |
| [baseline.py](src/grid/baseline.py) | Flattened MLP + min-`vm_fault` heuristic |
| [model.py](src/grid/model.py) | `FaultGCN`, hop matrix, edge-drop, train/eval, error-layout figure |

**RUL** — unchanged; see [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md).

---

## Suggested next step

Re-export twin scored CSV when fleet export exists; fill P10 lead-time table in [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md). Optional: re-simulate voltages after line outage for a stronger GNN robustness story.

---

## Key commit

(Uncommitted Track C day 2 — prior: `d4e2ea1` IEEE 39-bus dataset + MLP baseline.)
