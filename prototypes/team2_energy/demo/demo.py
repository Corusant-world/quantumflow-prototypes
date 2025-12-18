"""
Team2 Energy — Demo (CPU/CUDA)

Goal: show a tiny end-to-end usage of the public contract:
- EnergyGridSystem optimize + metrics
- process() helper
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import torch

# Make `team2_energy` importable when running `python demo/demo.py`
_PROTOTYPES_DIR = Path(__file__).resolve().parents[2]
if str(_PROTOTYPES_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTOTYPES_DIR))

from team2_energy.src.core import EnergyGridSystem, HAS_GPU, process as energy_process


def _pick_device() -> torch.device:
    dev = (os.environ.get("DEVICE") or "auto").strip().lower()
    if dev in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = _pick_device()
    print(f"[team2 demo] device={device} cuda_available={torch.cuda.is_available()} HAS_GPU={HAS_GPU}")

    sys_ = EnergyGridSystem(num_nodes=128)
    sys_.step_simulation()
    before = sys_.get_metrics()
    sys_.optimize_topology(steps=30)
    sys_.step_simulation()
    after = sys_.get_metrics()

    print("[team2 demo] before:", before)
    print("[team2 demo] after :", after)

    proc = energy_process({"num_nodes": 128, "opt_steps": 30}, device_override=device)
    print("[team2 demo] process():", proc)


if __name__ == "__main__":
    main()