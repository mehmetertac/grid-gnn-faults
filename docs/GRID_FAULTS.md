# IEEE 39-bus fault localization (Track C day 1)

Three-phase bolted faults on [pandapower](https://pandapower.readthedocs.io/) `networks.case39()`, with graph-level labels (which bus faulted). This is the dataset and non-graph baseline for a later GNN.

## Dataset

| Item | Value |
|------|-------|
| Graphs | 1,950 (50 operating scenarios × 39 buses) |
| Nodes | 39 buses (`net.bus.index` order) |
| Node features (8) | pre-fault `vm_pu`, during-fault `vm_pu`, ΔV, slack/PV/PQ one-hot, pre-fault `p_mw`, `q_mvar` |
| Label | Integer 0…38 = faulted bus position (`Data.y` shape `[1]`) |
| Split | Stratified 60/20/20 by faulted bus (seed 0) → 1170 / 390 / 390 |
| Voltage method | **`ybus_bolted`** for all samples (see below) |

Operating points: scale all loads and generator `p_mw` by a random factor in `[0.80, 0.90] ∪ [1.10, 1.20]`, then `runpp`.

### During-fault voltages

`pandapower.shortcircuit.calc_sc` on case39 needs extra generator short-circuit fields and still produced invalid `Ybus` for this network. The pipeline therefore uses the **bolted 3φ profile from the pre-fault admittance matrix**:

\[
\mathbf{V}_\text{sc} = \mathbf{V}_\text{pf} - \mathbf{Z}_{:,f}\,\frac{V_{\text{pf},f}}{Z_{ff}}
\]

implemented via `_pd2ppc` + `makeYbus` ([`src/grid/simulate.py`](../src/grid/simulate.py)). One `calc_sc` attempt with `branch_results=True` and `use_pre_fault_voltage=True` is made per run; if it fails the sag check, all remaining samples use Ybus only.

Artifacts (gitignored under `results/`):

- `fault_raw.npz` — raw features before normalization
- `fault_dataset.npz` — `x_raw`, `x_norm`, splits, topology, normalization stats
- `fault_graphs.pt` — list of PyG `Data` objects
- `fault_dv_sample.png` — Kamada–Kawai layout, node colour = ΔV

Regenerate:

```powershell
pip install -e ".[dev,grid]"
python scripts/build_fault_dataset.py --out results --scenarios 50 --seed 0
```

## Graph

Edges: in-service lines and two-winding transformers, undirected, weight = `1/|Z_pu|` (admittance magnitude). See [`src/grid/graph.py`](../src/grid/graph.py).

## Baselines (test split, seed 0)

| Model | Accuracy | Top-3 |
|-------|----------|-------|
| Flattened MLP (39×8 → 128 → 39) | 0.956 | 0.982 |
| Min during-fault `vm_pu` (oracle-style) | 1.000 | 1.000 |

The voltage-minimum rule is a sanity check: with Ybus bolted faults the faulted bus is at 0 pu. It is **not** the same as argmax(ΔV), because a distant bus with a higher pre-fault voltage can show a larger ΔV while not being the fault.

Tomorrow’s GNN should be compared to the MLP, not the voltage minimum alone.

## Grid operations note

For operators, the MLP and (later) GNN answer: *given PMU-like voltages and injections during a fault, which bus is the fault location?* The baseline shows much of the signal is local voltage depression; the GNN must use topology to beat or match the MLP when measurements are noisy or incomplete.
