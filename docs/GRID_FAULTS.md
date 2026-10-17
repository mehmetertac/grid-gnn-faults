# IEEE 39-bus fault localization (Track C)

Three-phase bolted faults on [pandapower](https://pandapower.readthedocs.io/) `networks.case39()`, with **graph-level** labels (which bus faulted). Dataset, MLP baseline, and 2-layer GCN live under [`src/grid/`](../src/grid/).

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

## Models (test split, seed 0)

Train GCN: `python scripts/train_fault_gnn.py --data results --out results` → `gnn_metrics.json`, `fault_gcn.pt`, [docs/figures/fault_error_layout.png](../docs/figures/fault_error_layout.png).

| Model | Accuracy | Top-3 | Mean hop error | Acc (one random line out) | Hops (one line out) |
|-------|----------|-------|----------------|---------------------------|---------------------|
| Flattened MLP (39×8 → 128 → 39) | 0.956 | 0.982 | 0.059 | 0.956 | 0.059 |
| 2-layer GCN, hidden 64, mean pool | 0.662 | 0.800 | 0.851 | 0.664 | 0.882 |
| Min during-fault `vm_pu` (oracle) | 1.000 | 1.000 | 0.000 | — | — |

**Hop error:** mean shortest-path distance on the undirected case39 graph between predicted and true bus (0 = exact). Implemented in [`src/grid/model.py`](../src/grid/model.py).

**Robustness:** one random undirected branch removed per test graph (both directed edges); node features are **not** re-simulated. The MLP ignores edges, so metrics are unchanged. The GCN sees a sparser graph; on this dataset the MLP already captures nearly all voltage-depression signal, so the GCN does not beat it on clean data.

The voltage-minimum rule is a sanity check only (Ybus bolted faults pin the fault bus near 0 pu). It is **not** argmax(ΔV).

## Grid operations note

For operators, these models answer: *given PMU-like voltages and injections during a fault, which bus is the fault location?* Top-1 accuracy alone hides whether errors are **one hop** away or across the system — hop distance is the operations-facing metric. A graph model is the credible path when buses are unobserved or topology changes; this demo reports honestly that the flattened MLP wins on clean Ybus features.
