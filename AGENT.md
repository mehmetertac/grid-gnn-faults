# AGENT.md — rules for AI agents

Guidance for agents working in **grid-gnn-faults**. Read this file first, then follow the linked docs.

Week 10 physics, the learned residual, and the hybrid-residual detector live in [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin). Do not refit them here.

---

## Documentation map

| Doc | Purpose |
|-----|---------|
| [README.md](README.md) | Entry point, setup, maintenance interpretation |
| [handover.md](handover.md) | Current status, layout, module API, next step |
| [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md) | RUL target, 90-day cap, daily features, event catalog, LOFO protocol |
| [docs/WEEK_11_TODO.md](docs/WEEK_11_TODO.md) | Week checklist |
| [src/rul/labels.py](src/rul/labels.py) | Daily aggregate and capped labels |
| [src/rul/evaluate.py](src/rul/evaluate.py) | Baselines, leave-one-failure-out, MAE, pinball, coverage |
| [src/rul/quantile.py](src/rul/quantile.py) | LightGBM q10/q50/q90, PDP, maintenance warning lead time |
| [src/rul/plots.py](src/rul/plots.py) | Headline RUL band figures |
| [scripts/evaluate_quantile_rul.py](scripts/evaluate_quantile_rul.py) | LOFO quantile CLI |
| [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin) | Week 10 twin: ODE, hybrid residual, anomaly score, gearbox log |

Headline contract: **probabilistic outputs on every prediction**, **leave-one-failure-out evaluation with no leakage from the failure window**, and a **maintenance-meaning** section in user-facing docs.

---

## Rules

### 1. File size limit

- **No file should exceed 1,000 lines.**
- If a file approaches or exceeds that limit, **stop and suggest a refactor** before adding more code.
- Pre-commit runs [`scripts/check_file_size.py`](scripts/check_file_size.py).

### 2. Documentation before every push

- **Update documentation before every push** to the repository.
- At minimum, check [README.md](README.md) and [handover.md](handover.md).

### 2a. Update handover.md on every push (required)

- Refresh **Last updated**, **What is done**, **Repo layout**, **Core module API**, **Suggested next step**, and **Key commit** on each push.
- If nothing functional changed, still bump **Last updated** and note "no functional change."

### 3. Tests — always, at least minimal

- **Always create at least minimal unit tests**, even for small changes.
- Add **integration** tests when wiring modules (scored frame → daily labels → LOFO baselines).
- Add **functional** tests when the project supports them (CLI smoke).
- Pattern: [tests/](tests/) uses fixtures. CI must not download EDP or import `wind_digital_twin`.

### 4. Run tests before commit or push

```powershell
pytest tests/ -q
python scripts/check_file_size.py
```

**Git hooks:** After creating the venv:

```powershell
pip install -e ".[dev]"
pre-commit install
```

Hooks run the file-size check and pytest (see [`.pre-commit-config.yaml`](.pre-commit-config.yaml)). If hooks do not exist yet, create them.

### 5. Keep reading in-repo docs

- Do not guess the RUL definition or the event count from memory. Use [docs/RUL_DESIGN.md](docs/RUL_DESIGN.md) and [handover.md](handover.md).
- Do not refit the thermal model. If a change needs new residuals, stop and consume the Week 10 outputs instead.

---

## Quick checklist (before push)

- [ ] No file > 1,000 lines (or a refactor is proposed)
- [ ] [README.md](README.md) updated if behavior or layout changed
- [ ] [handover.md](handover.md) updated (required on every push)
- [ ] New or changed logic has tests in [tests/](tests/)
- [ ] `pytest tests/ -q` passes
- [ ] `python scripts/check_file_size.py` passes
