"""
Team3 Innovation — Demo (CPU/CUDA)

Goal: show a tiny end-to-end usage of the public contract:
- quantum_neural_process(x, qubits=8)
- measure_tflops() (only meaningful on CUDA)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

# Make `team3_innovation` importable when running `python demo/demo.py`
_PROTOTYPES_DIR = Path(__file__).resolve().parents[2]
if str(_PROTOTYPES_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(_PROTOTYPES_DIR))

from team3_innovation.src.core import measure_tflops, quantum_neural_process  # noqa: E402


def _pick_device() -> torch.device:
    dev = (os.environ.get("DEVICE") or "auto").strip().lower()
    if dev in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = _pick_device()
    print(f"[team3 demo] device={device} cuda_available={torch.cuda.is_available()}")

    x = torch.randn(64, 128, device=device)
    y, metrics = quantum_neural_process(x, qubits=8)
    out_mean = float(y.mean().detach().cpu().item())
    print(f"[team3 demo] out_shape={list(y.shape)} out_mean={out_mean:.4f}")
    print(f"[team3 demo] metrics={metrics}")

    if device.type == "cuda":
        stats = measure_tflops(n=8192, iters=40, dtype="bf16")
        print(
            "[team3 demo] tflops_proxy="
            + str(stats.get("tflops"))
            + " dtype="
            + str(stats.get("dtype"))
            + " n="
            + str(stats.get("n"))
        )


if __name__ == "__main__":
    main()



