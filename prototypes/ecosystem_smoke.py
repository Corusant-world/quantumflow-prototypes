#!/usr/bin/env python3
"""
ECOSYSTEM SMOKE TEST — team1 + team2 + team3 in ONE environment

Goal:
- Prove that all 3 prototypes can be imported together (no dependency conflicts)
- Run a tiny functional path for each (CPU or CUDA)
- Write a single JSON artifact with results

Usage:
  python prototypes/ecosystem_smoke.py
  DEVICE=cpu python prototypes/ecosystem_smoke.py
  DEVICE=cuda python prototypes/ecosystem_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Make `team{1,2,3}_*` importable when running:
#   python prototypes/ecosystem_smoke.py
_PROTOTYPES_DIR = Path(__file__).resolve().parent
if str(_PROTOTYPES_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTOTYPES_DIR))


def _pick_device() -> torch.device:
    dev = (os.environ.get("DEVICE") or "auto").strip().lower()
    if dev in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev in ("cpu",):
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_import_cuquantum() -> Dict[str, Any]:
    try:
        import cuquantum  # type: ignore

        has_cutensornet = False
        try:
            from cuquantum import cutensornet  # type: ignore  # noqa: F401

            has_cutensornet = True
        except Exception:
            has_cutensornet = False

        return {
            "available": True,
            "file": getattr(cuquantum, "__file__", None),
            "path": [str(p) for p in getattr(cuquantum, "__path__", [])] if hasattr(cuquantum, "__path__") else None,
            "has_cutensornet": bool(has_cutensornet),
            "version": getattr(cuquantum, "__version__", None),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def run() -> Dict[str, Any]:
    device = _pick_device()
    out: Dict[str, Any] = {
        "ts": time.time(),
        "device": str(device),
        "torch": {
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "cuquantum": _safe_import_cuquantum(),
        "team1": {},
        "team2": {},
        "team3": {},
    }

    # Team1: QuantumVacancyHunter
    t0 = time.perf_counter()
    try:
        from team1_quantum.src.core import QuantumVacancyHunter

        hunter = QuantumVacancyHunter(lattice_size=128, use_fp16=(device.type == "cuda"), device=device)
        m = hunter.find_stable_vacancies(batch_size=64, steps=6)
        out["team1"] = {
            "ok": True,
            "wall_s": float(time.perf_counter() - t0),
            "metrics": {
                "compute_time_ms": float(m.compute_time_ms),
                "fidelity": float(m.fidelity),
                "active_qubits": int(m.active_qubits),
            },
        }
    except Exception as e:
        out["team1"] = {"ok": False, "error": str(e), "wall_s": float(time.perf_counter() - t0)}

    # Team2: EnergyGridSystem + process()
    t0 = time.perf_counter()
    try:
        from team2_energy.src.core import EnergyGridSystem, process as energy_process

        sys_ = EnergyGridSystem(num_nodes=128)
        sys_.step_simulation()
        before = sys_.get_metrics()
        sys_.optimize_topology(steps=30)
        sys_.step_simulation()
        after = sys_.get_metrics()
        proc = energy_process({"num_nodes": 128, "opt_steps": 30})
        out["team2"] = {
            "ok": True,
            "wall_s": float(time.perf_counter() - t0),
            "before": before,
            "after": after,
            "process_device": proc.get("device") if isinstance(proc, dict) else None,
        }
    except Exception as e:
        out["team2"] = {"ok": False, "error": str(e), "wall_s": float(time.perf_counter() - t0)}

    # Team3: quantum_neural_process (must work on CPU too)
    t0 = time.perf_counter()
    try:
        from team3_innovation.src.core import quantum_neural_process

        x = torch.randn(64, 128, device=device)
        y, metrics = quantum_neural_process(x, qubits=8)
        out["team3"] = {
            "ok": True,
            "wall_s": float(time.perf_counter() - t0),
            "out_shape": list(y.shape),
            "out_mean": float(y.mean().detach().cpu().item()),
            "metrics": metrics,
        }
    except Exception as e:
        out["team3"] = {"ok": False, "error": str(e), "wall_s": float(time.perf_counter() - t0)}

    return out


def main() -> None:
    out = run()
    out_dir = Path(__file__).parent / "ecosystem_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latest.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"✅ ECOSYSTEM SMOKE COMPLETE: {out_path}")
    print(json.dumps({k: out.get(k) for k in ["device", "cuquantum", "team1", "team2", "team3"]}, indent=2))


if __name__ == "__main__":
    main()


