"""
Team1 Quantum — Demo (runnable on CPU or CUDA)

Goal: show a tiny end-to-end usage of the public contract:
- QuantumVacancyHunter(...).find_stable_vacancies(...)
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import torch

# Make `team1_quantum` importable when running `python demo/demo.py`
_PROTOTYPES_DIR = Path(__file__).resolve().parents[2]
if str(_PROTOTYPES_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTOTYPES_DIR))

from team1_quantum.src.core import QuantumVacancyHunter


def _pick_device() -> torch.device:
    dev = (os.environ.get("DEVICE") or "auto").strip().lower()
    if dev in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = _pick_device()
    print(f"[team1 demo] device={device} cuda_available={torch.cuda.is_available()}")

    hunter = QuantumVacancyHunter(lattice_size=512, use_fp16=(device.type == "cuda"), device=device)
    m = hunter.find_stable_vacancies(batch_size=64, steps=6)

    print("[team1 demo] metrics:")
    print(f"  compute_time_ms={m.compute_time_ms:.3f}")
    print(f"  fidelity={m.fidelity:.4f}")
    print(f"  active_qubits={m.active_qubits}")


if __name__ == "__main__":
    main()