# SHAP explanations for gearbox RUL

TreeSHAP on the leave-one-failure-out LightGBM quantile models explains which **daily fused features** move P10 and P50 RUL. Attributions are summed back to six **sensor-level** groups so maintenance and the deployment UI see names like “Gearbox bearing temperature” instead of column codes.

Implementation: [`src/rul/explain.py`](../src/rul/explain.py). CLI: [`scripts/explain_rul.py`](../scripts/explain_rul.py).

Related: [RUL_DESIGN.md](RUL_DESIGN.md), [handover.md](../handover.md).

## Sensor map (additive)

Each row’s sensor SHAP values sum to the row’s feature SHAP values (and to prediction minus the tree base value, up to numerical tolerance).

| Sensor | Daily features |
| --- | --- |
| Gearbox oil temperature | `oil_deg_mean`, `oil_deg_std`, `oil_deg_slope_7d` |
| Gearbox bearing temperature | `bear_deg_mean`, `bear_deg_std`, `bear_deg_slope_7d` |
| Oil temperature per unit power | `theta_oil_mean` |
| Bearing temperature per unit power | `theta_bear_mean` |
| Bearing-oil temperature difference | `delta_bear_oil_mean` |
| Hybrid anomaly score | `anomaly_score_mean`, `anomaly_score_max` |

There is no separate generator current or voltage channel in the RUL matrix; electrical coupling appears only through temperature per unit power. The anomaly score is the hybrid-residual detector output, not a thermocouple.

## Protocol

- **Model:** P10 and P50 boosters from a LOFO fit — train on every turbine **except** the held-out event’s turbine; the failure window never enters training (same rule as [evaluate.py](../src/rul/evaluate.py)).
- **Default hold-out:** T06 bearing event when present in the labeled frame; override with `--event` or `--turbine`.
- **Global SHAP:** training rows of that fold (beeswarm on P50 features; bar chart of mean \|SHAP\| per sensor for P10 and P50).
- **Local SHAP:** held-out window (`0 < days_to_event <= 90`); heatmap of signed P50 sensor SHAP vs days to event (top six sensors; all six groups exist in this project).

## Outputs

| Path | Content |
| --- | --- |
| `results/shap_sensor_ranking.csv` | One row per sensor per learner (`p10`, `p50`): `sensor`, `learner`, `mean_shap`, `mean_abs_shap`, `rank`, `n_rows`, `features` |
| `results/figures/shap_global.png` | P50 beeswarm + sensor bars |
| `results/figures/shap_local_timeline.png` | Local P50 sensor heatmap |
| `results/rul_quantile_bundle.joblib` | Saved quantile bundle after fit (optional `--model` to skip refit) |

```powershell
python scripts/explain_rul.py --labeled path/to/daily_labeled.csv --out results
# or rebuild daily from twin scored export:
python scripts/explain_rul.py --scored path/to/scored_10min.csv --failures path/to/gearbox_log.csv --out results
```

Re-run with a fleet scored export from [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin) when available; see the scored-export caveat in [RUL_DESIGN.md](RUL_DESIGN.md).

## Field intuition (held-out run)

Paragraph below is produced by `run_explain` / `--field-notes` from the labeled frame actually explained. On the **pytest synthetic panel** (held-out turbine T_A, `rul_max=10`), TreeSHAP did not show a clear “first negative” day for thermal vs power-normalized groups in the short test window — rerun on a twin daily export for T06 (or another catalog event) for a field-facing sentence.

On the held-out window, grouped thermal residuals and bearing-oil delta first pull P50 RUL down at not in the 90-day window, while temperature-per-unit-power features do so at not in the 90-day window. Neither group shows a sustained negative P50 SHAP in this window; check that the labeled frame has enough pre-failure daily rows.

After a real export, expect to compare whether **Gearbox bearing temperature** / **Gearbox oil temperature** turn negative before **Bearing temperature per unit power** — agreement supports a thermal-led gearbox story; anomaly-first or theta-first ordering is worth reporting as disagreement, not as a modeling failure.
