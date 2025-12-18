"""
team2_energy — src/core.py

Canonical prototype module for Team 2 (Energy).
This file must be importable as a normal Python module (no notebook magic).

Exports (contract):
- EnergyGridSystem (tests + benchmarks)
- TNP_System (benchmarks)
- HAS_GPU (tests + demo)
- run_tnp_demo / process (back-compat entrypoints)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn

# Contract for tests (`tests/test_core.py`)
HAS_GPU: bool = bool(torch.cuda.is_available())
device = torch.device("cuda" if HAS_GPU else "cpu")


class EnergyGridSystem:
    """
    EnergyGridSystem (contracted by `tests/test_core.py`)

    Intent:
    - GPU-friendly physics-style loss that can run on CUDA when available.
    - Topology optimization that changes loss (test checks m2 != m1).
    """

    def __init__(self, num_nodes: int = 100, seed: int = 1337):
        if num_nodes <= 1:
            raise ValueError("num_nodes must be > 1")
        self.num_nodes = int(num_nodes)
        self.device = device
        self.dtype = torch.float16 if (self.device.type == "cuda") else torch.float32

        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed) + self.num_nodes)

        # Node potentials (toy surrogate of voltage angles)
        self._v = torch.randn(self.num_nodes, device=self.device, dtype=self.dtype, generator=g)

        # Symmetric conductance matrix (topology); zero diagonal
        G = torch.rand(self.num_nodes, self.num_nodes, device=self.device, dtype=self.dtype, generator=g)
        G = 0.5 * (G + G.T)
        G.fill_diagonal_(0.0)
        self._G = G

        self._metrics: Dict[str, float] = {"total_loss_mw": float("nan")}

    def _loss(self) -> torch.Tensor:
        # Loss = sum_{i,j} G_ij * (v_i - v_j)^2  (Joule heating proxy)
        v = self._v
        dv = v.unsqueeze(1) - v.unsqueeze(0)
        loss = (self._G * (dv * dv)).sum()
        # scale to MW-ish numbers
        return loss.to(torch.float32) * 1e3

    def step_simulation(self) -> None:
        # Simple stochastic dynamics: random drift + stable clamp
        with torch.no_grad():
            noise = torch.randn_like(self._v) * (0.01 if self.device.type == "cuda" else 0.005)
            self._v = (0.995 * self._v + noise).clamp(-3.0, 3.0)
            L = self._loss()
            self._metrics["total_loss_mw"] = float(L.detach().cpu().item())

    def get_metrics(self) -> Dict[str, float]:
        return dict(self._metrics)

    def validation_check(self) -> Tuple[bool, str]:
        val = float(self._metrics.get("total_loss_mw", float("nan")))
        if not np.isfinite(val):
            return False, "loss is not finite"
        if val < 0:
            return False, "loss is negative"
        return True, "ok"

    def optimize_topology(self, steps: int = 50, lr: float = 5e-4) -> None:
        if steps <= 0:
            raise ValueError("steps must be > 0")
        # Optimize conductances to reduce loss; keep symmetric & non-negative.
        G = self._G.to(torch.float32).clone().detach().requires_grad_(True)

        for _ in range(int(steps)):
            v = self._v.to(torch.float32)
            dv = v.unsqueeze(1) - v.unsqueeze(0)
            loss = (G * (dv * dv)).sum() * 1e3
            loss.backward()
            with torch.no_grad():
                G -= float(lr) * G.grad
                G.clamp_(0.0, 1.0)
                G.fill_diagonal_(0.0)
                if G.grad is not None:
                    G.grad.zero_()

        with torch.no_grad():
            Gsym = 0.5 * (G + G.T)
            Gsym.fill_diagonal_(0.0)
            self._G = Gsym.to(self.dtype)
            self._metrics["total_loss_mw"] = float(self._loss().detach().cpu().item())


class TNP_System(nn.Module):
    """
    Thermodynamic Neural Processor (benchmark-only class).
    Uses GPU-friendly matmul + gradients to create a heavy CUDA workload on H100.
    """

    def __init__(self, dim: int = 512, layers: int = 4, kT: float = 4.14e-21):
        super().__init__()
        self.dim = int(dim)
        self.layers = int(layers)
        self.kT = float(kT)  # Boltzmann factor for Landauer
        self.model = self._build_model()
        self.thermo_kernel = self._compile_thermo_kernel()
        self.energy_stats = {"total_joules": 0.0, "landauer_min": 0.0, "dissipation": 0.0}

    def _build_model(self) -> nn.Module:
        seq: List[nn.Module] = []
        for _ in range(self.layers):
            seq.extend([nn.Linear(self.dim, self.dim), nn.ReLU()])
        return nn.Sequential(*seq).to(device)

    def _compile_thermo_kernel(self):
        # Custom CUDA kernel stub (nsight-ready): Stochastic Langevin dynamics for energy landscape.
        # Placeholder for full PTX (to be nsight_compute optimized).
        def thermo_energy(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
            dissipation = torch.norm(grad, dim=-1) ** 2 * 1e-12  # Joule heating proxy
            landauer = (
                self.kT
                * torch.log(torch.tensor(2.0, device=x.device))
                * x.numel()
                * torch.ones_like(x[:, 0])
            )
            return dissipation + landauer

        return thermo_energy

    def simulate_thermo(self, batch_size: int = 1024, steps: int = 100) -> Dict[str, float]:
        # IMPORTANT: keep x as a leaf each iteration; otherwise x.grad becomes None (and benchmark crashes).
        x = torch.randn(int(batch_size), self.dim, device=device, requires_grad=True)

        total_energy = 0.0
        for _ in range(int(steps)):
            pred = self.model(x)
            loss = pred.norm()
            loss.backward()

            grad = x.grad
            if grad is None:
                raise RuntimeError("TNP_System: x.grad is None after backward()")

            energy = self.thermo_kernel(x, grad)
            total_energy += float(energy.mean().detach().item())

            with torch.no_grad():
                x = x - 0.01 * grad + torch.randn_like(x) * 0.01  # Langevin step

            # New leaf for next step
            x = x.detach().requires_grad_(True)

        landauer_min = float(self.kT * np.log(2) * int(batch_size) * self.dim * int(steps))
        self.energy_stats = {
            "total_joules": float(total_energy),
            "landauer_min": landauer_min,
            "dissipation": float(total_energy - landauer_min),
            "efficiency": float((1.0 - (total_energy / max(1e-12, landauer_min))) * 100.0),
        }
        return dict(self.energy_stats)


def run_tnp_demo() -> Dict[str, float]:
    tnp = TNP_System()
    stats = tnp.simulate_thermo(batch_size=256, steps=20)
    print("TNP Energy Stats:", stats)
    return stats


def process(input_data: Any = None, device_override: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Back-compat entrypoint used by `src/main.py` / `src/__init__.py`.
    Keeps CPU-safe defaults; GPU will be used automatically if available.
    """
    _ = device_override  # device is derived globally from HAS_GPU for this prototype

    num_nodes = 256
    opt_steps = 60
    if isinstance(input_data, dict):
        num_nodes = int(input_data.get("num_nodes", num_nodes))
        opt_steps = int(input_data.get("opt_steps", opt_steps))

    sys_ = EnergyGridSystem(num_nodes=num_nodes)
    sys_.step_simulation()
    before = sys_.get_metrics()
    ok0, msg0 = sys_.validation_check()

    sys_.optimize_topology(steps=opt_steps)
    sys_.step_simulation()
    after = sys_.get_metrics()
    ok1, msg1 = sys_.validation_check()

    return {
        "device": "cuda" if HAS_GPU else "cpu",
        "energy_grid": {
            "before": before,
            "after": after,
            "sanity_before": {"ok": ok0, "msg": msg0},
            "sanity_after": {"ok": ok1, "msg": msg1},
        },
    }