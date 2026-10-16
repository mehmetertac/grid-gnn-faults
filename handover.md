# handover.md

**Last updated:** 2026-10-15

Week 11 gearbox RUL + deployment remain in place. **Track C day 1** added the IEEE 39-bus fault dataset, graph construction, and MLP baseline. GNN training is not done yet.

---

## What is done

| Item | Status |
|------|--------|
| Git remote `origin` → [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults) | Done |
| [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md), [docs/EXPLANATIONS.md](docs/EXPLANATIONS.md) | Done |
| `src/rul/*` — labels, evaluate, quantile, conformal, plots, explain | Done |
| Scripts — `evaluate_quantile_rul.py`, `run_twin.py`, `explain_rul.py` | Done |
| **Deployment** — FastAPI `/twin/score`, `/twin/rul` + Streamlit dashboard in wind-digital-twin | Done |
| [WEEK_11_REFLECTION.md](WEEK_11_REFLECTION.md) | Done |
| **Track C day 1** — `src/grid/*`, [docs/GRID_FAULTS.md](docs/GRID_FAULTS.md), `scripts/build_fault_dataset.py`, [notebooks/01_fault_localization_gnn.ipynb](notebooks/01_fault_localization_gnn.ipynb) | Done |
| 1,950-graph dataset (50 scenarios × 39 buses), PyG `.pt` + `.npz`, ΔV sanity figure | Done (local `results/`) |
| MLP baseline (test acc ≈ 0.956, top-3 ≈ 0.982); min-`vm_fault` heuristic (1.0) | Done |
| Twin scored CSV export for four catalog events (warning table filled) | Pending export |
| IEEE 39-bus **GNN** training + accuracy table vs MLP | Not started |

### Test run (2026-10-15)

`pytest tests/ -q` in grid-gnn-faults (RUL fixtures + `tests/test_grid_faults.py` with `importorskip` for torch/pandapower).

---

## Repo layout

```
grid-gnn-faults/
├── WEEK_11_REFLECTION.md
├── handover.md
├── docs/RUL_DESIGN.md
├── docs/GRID_FAULTS.md
├── notebooks/01_fault_localization_gnn.ipynb
├── src/rul/
├── src/grid/
├── scripts/build_fault_dataset.py
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

**RUL** — unchanged; see [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md) and `src/rul/`.

**Grid fault localization (`src/grid/`)**

| Module | Purpose |
|--------|---------|
| [simulate.py](src/grid/simulate.py) | `load_case39()`, `generate_fault_dataset()`, Ybus bolted during-fault voltages |
| [graph.py](src/grid/graph.py) | Topology, stratified split, z-score, PyG `Data`, ΔV plot |
| [baseline.py](src/grid/baseline.py) | Flattened MLP + min during-fault voltage heuristic |

Label: graph-level class `0…38` = faulted bus index in `net.bus.index` order. During-fault voltages: **`ybus_bolted`** for all 1,950 samples (case39 `calc_sc` branch path fails sag check after one probe).

**Deployment (wind-digital-twin)**

| Endpoint | Purpose |
|----------|---------|
| `POST /twin/score` | SCADA window → expected temp, residual, anomaly score |
| `POST /twin/rul` | Daily features → P10/P50/P90, CQR band, top-5 sensor attributions |
| `GET /twin/series` | Dashboard timeline for failure-window scrub |

---

## Suggested next step

Train a small GCN/GAT on `results/fault_graphs.pt` (or regenerate with `scripts/build_fault_dataset.py`), report test accuracy and top-3 vs the MLP in [docs/GRID_FAULTS.md](docs/GRID_FAULTS.md). Re-export twin scored CSV when fleet export exists.

---

## Key commit

`e4f7885` — IEEE 39-bus fault dataset, PyG graphs, and MLP baseline (prior: `c8c1411` handover hash update).
