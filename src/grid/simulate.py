"""Generate IEEE 39-bus fault samples with pandapower."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.idx_bus import VA, VM
from pandapower.pypower.makeYbus import makeYbus
from pandapower.shortcircuit import calc_sc

VoltageMethod = Literal["branch_sc", "ybus_bolted", "load_injection"]


@dataclass(frozen=True)
class FaultSample:
    """One graph sample before normalization."""

    x_raw: np.ndarray  # (39, 8)
    label: int  # faulted bus position 0..38
    scenario_id: int
    voltage_method: VoltageMethod


def load_case39():
    """Return a deep-copied IEEE 39-bus network with short-circuit fields filled."""
    net = pn.case39()
    net.ext_grid["s_sc_max_mva"] = net.ext_grid.get("s_sc_max_mva", 10_000.0)
    net.ext_grid["s_sc_min_mva"] = net.ext_grid.get("s_sc_min_mva", 8_000.0)
    net.ext_grid["rx_max"] = net.ext_grid.get("rx_max", 0.1)
    net.ext_grid["rx_min"] = net.ext_grid.get("rx_min", 0.1)
    if "vn_kv" not in net.gen.columns:
        net.gen["vn_kv"] = net.gen["bus"].map(net.bus["vn_kv"])
    for col, val in (("xdss_pu", 0.25), ("rdss_ohm", 0.01), ("cos_phi", 0.85)):
        if col not in net.gen.columns:
            net.gen[col] = val
    return net


def _draw_load_scale(rng: np.random.Generator) -> float:
    if rng.random() < 0.5:
        return float(rng.uniform(0.80, 0.90))
    return float(rng.uniform(1.10, 1.20))


def _apply_operating_point(net, scale: float) -> None:
    net.load["p_mw"] = net.load["p_mw"] * scale
    net.load["q_mvar"] = net.load["q_mvar"] * scale
    net.gen["p_mw"] = net.gen["p_mw"] * scale


def _bus_type_onehot(net) -> np.ndarray:
    n = len(net.bus)
    slack_bus = int(net.ext_grid.bus.iloc[0])
    gen_buses = set(net.gen.bus.astype(int))
    out = np.zeros((n, 3), dtype=np.float32)
    bus_to_pos = {int(b): i for i, b in enumerate(net.bus.index)}
    for b in net.bus.index:
        pos = bus_to_pos[int(b)]
        if int(b) == slack_bus:
            out[pos, 0] = 1.0
        elif int(b) in gen_buses:
            out[pos, 1] = 1.0
        else:
            out[pos, 2] = 1.0
    return out


def _node_features(net, vm_pre: np.ndarray, vm_fault: np.ndarray) -> np.ndarray:
    dv = vm_pre - vm_fault
    type_oh = _bus_type_onehot(net)
    p_mw = net.res_bus.p_mw.values.astype(np.float32)
    q_mvar = net.res_bus.q_mvar.values.astype(np.float32)
    return np.column_stack(
        [
            vm_pre.astype(np.float32),
            vm_fault.astype(np.float32),
            dv.astype(np.float32),
            type_oh,
            p_mw,
            q_mvar,
        ]
    ).astype(np.float32)


def _ybus_bolted_voltages(net, fault_pos: int) -> np.ndarray:
    _, ppci = _pd2ppc(net)
    ybus, _, _ = makeYbus(ppci["baseMVA"], ppci["bus"], ppci["branch"])
    zbus = np.linalg.inv(ybus.toarray())
    v_pf = ppci["bus"][:, VM] * np.exp(1j * np.deg2rad(ppci["bus"][:, VA]))
    v_sc = v_pf - zbus[:, fault_pos] * v_pf[fault_pos] / zbus[fault_pos, fault_pos]
    return np.abs(v_sc).astype(np.float64)


def _assemble_branch_voltages(net) -> np.ndarray:
    n = len(net.bus)
    vm = np.full(n, np.nan, dtype=np.float64)
    bus_to_pos = {int(b): i for i, b in enumerate(net.bus.index)}

    for idx, row in net.line.iterrows():
        if not row.in_service:
            continue
        fb = bus_to_pos[int(row.from_bus)]
        tb = bus_to_pos[int(row.to_bus)]
        vm[fb] = float(net.res_line_sc.at[idx, "vm_from_pu"])
        vm[tb] = float(net.res_line_sc.at[idx, "vm_to_pu"])

    for idx, row in net.trafo.iterrows():
        if not row.in_service:
            continue
        fb = bus_to_pos[int(row.hv_bus)]
        tb = bus_to_pos[int(row.lv_bus)]
        vm[fb] = float(net.res_trafo_sc.at[idx, "vm_hv_pu"])
        vm[tb] = float(net.res_trafo_sc.at[idx, "vm_lv_pu"])

    return vm


def _passes_sag_check(vm_fault: np.ndarray, fault_pos: int) -> bool:
    if np.any(np.isnan(vm_fault)):
        return False
    min_pos = int(np.argmin(vm_fault))
    if min_pos != fault_pos:
        return False
    if vm_fault[fault_pos] > 0.05:
        return False
    if vm_fault[fault_pos] > vm_fault.min() + 1e-6:
        return False
    return True


def _branch_sc_voltages(net, fault_bus: int, fault_pos: int) -> np.ndarray | None:
    try:
        calc_sc(
            net,
            bus=fault_bus,
            fault="3ph",
            branch_results=True,
            use_pre_fault_voltage=True,
        )
    except (ValueError, NotImplementedError, AttributeError):
        return None
    vm = _assemble_branch_voltages(net)
    if not _passes_sag_check(vm, fault_pos):
        return None
    return vm


def _load_injection_voltages(net, fault_bus: int, fault_pos: int) -> np.ndarray | None:
    """Fallback: large load step at the fault bus via power flow."""
    trial = copy.deepcopy(net)
    bump_mw = 500.0
    trial.load.loc[trial.load.bus == fault_bus, "p_mw"] += bump_mw
    if not trial.load.bus.eq(fault_bus).any():
        pp.create_load(trial, bus=fault_bus, p_mw=bump_mw, q_mvar=0.0)
    try:
        pp.runpp(trial, init="results", calculate_voltage_angles=True)
    except pp.LoadflowNotConverged:
        return None
    vm = trial.res_bus.vm_pu.values.astype(np.float64)
    if not _passes_sag_check(vm, fault_pos):
        return None
    return vm


def _fault_voltages(
    net,
    fault_bus: int,
    fault_pos: int,
) -> tuple[np.ndarray, VoltageMethod]:
    vm_ybus = _ybus_bolted_voltages(net, fault_pos)
    if _passes_sag_check(vm_ybus, fault_pos):
        return vm_ybus, "ybus_bolted"

    vm_load = _load_injection_voltages(net, fault_bus, fault_pos)
    if vm_load is not None:
        return vm_load, "load_injection"

    return vm_ybus, "ybus_bolted"


def _run_power_flow(net, max_tries: int, rng: np.random.Generator) -> bool:
    for _ in range(max_tries):
        try:
            pp.runpp(net, calculate_voltage_angles=True)
            return True
        except pp.LoadflowNotConverged:
            scale = _draw_load_scale(rng)
            _apply_operating_point(net, scale)
    return False


def generate_fault_dataset(
    n_scenarios: int = 50,
    seed: int = 0,
    base_net=None,
    try_branch_sc: bool = True,
) -> tuple[list[FaultSample], np.ndarray, dict[str, int]]:
    """Build scenarios × buses fault samples."""
    template = load_case39() if base_net is None else copy.deepcopy(base_net)
    base_load_p = template.load["p_mw"].copy()
    base_load_q = template.load["q_mvar"].copy()
    base_gen_p = template.gen["p_mw"].copy()

    rng = np.random.default_rng(seed)
    bus_ids = template.bus.index.to_numpy()
    samples: list[FaultSample] = []
    method_counts: dict[str, int] = {"branch_sc": 0, "ybus_bolted": 0, "load_injection": 0}
    branch_sc_enabled = try_branch_sc

    for scenario_id in range(n_scenarios):
        net = copy.deepcopy(template)
        net.load["p_mw"] = base_load_p
        net.load["q_mvar"] = base_load_q
        net.gen["p_mw"] = base_gen_p

        scale = _draw_load_scale(rng)
        _apply_operating_point(net, scale)
        if not _run_power_flow(net, max_tries=3, rng=rng):
            continue

        vm_pre = net.res_bus.vm_pu.values.astype(np.float64)

        for fault_pos, fault_bus in enumerate(bus_ids):
            if branch_sc_enabled:
                vm_branch = _branch_sc_voltages(net, int(fault_bus), fault_pos)
                if vm_branch is not None:
                    vm_fault, method = vm_branch, "branch_sc"
                else:
                    branch_sc_enabled = False
                    vm_fault, method = _fault_voltages(net, int(fault_bus), fault_pos)
            else:
                vm_fault, method = _fault_voltages(net, int(fault_bus), fault_pos)
            method_counts[method] += 1
            x_raw = _node_features(net, vm_pre, vm_fault)
            samples.append(
                FaultSample(
                    x_raw=x_raw,
                    label=fault_pos,
                    scenario_id=scenario_id,
                    voltage_method=method,
                )
            )

    return samples, bus_ids, method_counts


def save_raw_dataset(
    out_dir: Path,
    samples: list[FaultSample],
    bus_ids: np.ndarray,
    method_counts: dict[str, int],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.stack([s.x_raw for s in samples], axis=0)
    y = np.array([s.label for s in samples], dtype=np.int64)
    scenario = np.array([s.scenario_id for s in samples], dtype=np.int64)
    methods = np.array([s.voltage_method for s in samples], dtype=object)

    path = out_dir / "fault_raw.npz"
    np.savez_compressed(
        path,
        x_raw=x,
        y=y,
        scenario_id=scenario,
        bus_ids=bus_ids,
        voltage_method=methods,
        method_counts=np.array([str(method_counts)], dtype=object),
    )
    return path


def generate_and_save(
    out_dir: Path | str = "results",
    n_scenarios: int = 50,
    seed: int = 0,
) -> tuple[list[FaultSample], np.ndarray, dict[str, int]]:
    samples, bus_ids, method_counts = generate_fault_dataset(
        n_scenarios=n_scenarios,
        seed=seed,
    )
    save_raw_dataset(Path(out_dir), samples, bus_ids, method_counts)
    return samples, bus_ids, method_counts
