# QuantumFlow — GPU-Accelerated Prototypes Ecosystem

**Achieving 95%+ GPU utilization on NVIDIA H100 with reproducible benchmarks, cuQuantum integration, and ecosystem compatibility proof.**

## Quick Start

### Installation

```bash
# CPU (no CUDA required)
python -m pip install -U pip
python -m pip install -r prototypes/requirements.cpu.txt

# NVIDIA GPU (CUDA 12, without cuQuantum)
python -m pip install -U pip
python -m pip install -r prototypes/requirements.gpu-cu12.txt

# NVIDIA GPU (CUDA 12) + cuQuantum (Team3 acceleration)
python -m pip install -U pip
python -m pip install -r prototypes/requirements.gpu-cu12-cuquantum.txt
```

### Run Ecosystem Smoke Test

```bash
python prototypes/ecosystem_smoke.py
```

## Key Results (tested on NVIDIA H100 PCIe)

| Component | What it proves | Key metric | Evidence |
|---|---|---:|---|
| Team1 Quantum | Tensor Core-heavy screening workload (CPU fallback) | NVML GPU util avg **95.19%** | `prototypes/team1_quantum/benchmarks/results/latest.json` |
| Team2 Energy | Differentiable thermo + grid optimization (CPU fallback) | NVML GPU util avg **95.44%** | `prototypes/team2_energy/benchmarks/results/latest.json` |
| Team3 Innovation | cuQuantum contraction + sustained soak | NVML GPU util avg **95.47%** + `cuquantum_used=true` | `prototypes/team3_innovation/benchmarks/results/latest.json` |

## Sample Benchmark Results

### Ecosystem Smoke Test

```json
{
  "timestamp": "2025-12-14T...",
  "device": "cuda",
  "teams": {
    "team1_quantum": {"status": "OK", "device": "cuda"},
    "team2_energy": {"status": "OK", "device": "cuda"},
    "team3_innovation": {"status": "OK", "device": "cuda", "cuquantum_available": true}
  }
}
```

### Team1 Quantum

```json
{
  "nvml": {
    "gpu_util_avg": 95.19,
    "gpu_util_max": 100.0,
    "mem_used_mb_max": 3921
  },
  "benchmarks": {
    "tflops_est": 453.2
  }
}
```

### Team2 Energy

```json
{
  "nvml": {
    "gpu_util_avg": 95.44,
    "gpu_util_max": 100.0,
    "mem_used_mb_max": 3735
  },
  "benchmarks": {
    "tflops_est": 451.1,
    "tnp_efficiency": 96.89
  }
}
```

### Team3 Innovation

```json
{
  "nvml": {
    "gpu_util_avg": 95.47,
    "gpu_util_max": 100.0,
    "mem_used_mb_max": 4531
  },
  "benchmarks": {
    "tflops_est": 449.3,
    "cuquantum_used": true,
    "cuquantum_path": {
      "dim": 4096,
      "iters": 316,
      "tflops_est": 5.42
    }
  }
}
```

## Links

- GitHub Repository: https://github.com/<ORG>/<REPO>
- Documentation: https://github.com/<ORG>/<REPO>/blob/main/README.md
- Docker Images: https://github.com/<ORG>/<REPO>/pkgs/container/quantumflow
- PyPI: https://pypi.org/project/quantumflow-prototypes/
- Release Notes: https://github.com/<ORG>/<REPO>/releases/tag/v0.1.0

