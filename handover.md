# handover.md

**Last updated:** 2026-11-24

Week 11 gearbox RUL + deployment remain in place. **Track C day 2** added GCN training, hop-distance metrics, edge-drop robustness check, and README figures. Capstone polish and tags live in sibling [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin).

**Reproducibility sweep (2026-11-24):** `requires-python = ">=3.11,<3.13"`, hashed [requirements.lock](requirements.lock) (wheels-only, CPU PyTorch index), [scripts/regenerate_lock.ps1](scripts/regenerate_lock.ps1) + [scripts/merge_lock_hashes.py](scripts/merge_lock_hashes.py), CI smoke on `windows-latest` / `ubuntu-latest` × Python 3.11 / 3.12 ([.github/workflows/ci.yml](.github/workflows/ci.yml)). Pinned `pandas>=2.0,<3` so pandapower 3.1.x passes grid tests under locked installs.

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
| **Reproducibility** — Python pin, `requirements.lock`, wheels-only README, GitHub Actions CI | Done |

### Test run (2026-11-24)

Clean venv: lock install + `pytest tests/ -q` → **46 passed** (includes grid + RUL). Same command as CI (`MPLBACKEND=Agg`).

---

## Repo layout

```
grid-gnn-faults/
├── handover.md
├── requirements.lock
├── .github/workflows/ci.yml
├── docs/GRID_FAULTS.md
├── docs/RUL_DESIGN.md
├── notebooks/01_fault_localization_gnn.ipynb
├── scripts/regenerate_lock.ps1
├── scripts/merge_lock_hashes.py
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

**Install (repro)** — [README.md](README.md) Quickstart: `pip install --require-hashes -r requirements.lock`, then `pip install --no-deps -e .`

---

## Suggested next step

Re-run fleet export on **real** EDP when downloaded; expect P10 lead integers to move off `none` only if daily quantiles leave the 90-day cap before failure. Optional: re-simulate voltages after line outage for a stronger GNN robustness story.

---

## Key commit

Pending local commit — reproducibility sweep (Python pin, lockfile, CI, pandas cap for pandapower); prior functional work on fleet export / Track C.