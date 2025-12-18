# Benchmarks (Prototypes)

This repository ships three GPU-first prototypes under `prototypes/`. Each prototype has a benchmark harness that writes a single JSON artifact:

- `prototypes/<team>/benchmarks/results/latest.json`

## Hardware

Primary target: **NVIDIA H100 (80GB class)**.

## How to run

### Team1

```bash
python prototypes/team1_quantum/benchmarks/run_benchmarks.py
```

### Team2

```bash
python prototypes/team2_energy/benchmarks/run_benchmarks.py
```

### Team3 (includes cuQuantum proof)

```bash
python prototypes/team3_innovation/benchmarks/run_benchmarks.py
```

## What we record

Each `latest.json` contains:
- **Dependency probe**: Python, platform, torch, CUDA availability, optional CuPy/cuQuantum.
- **NVML sampling** (via `nvidia-ml-py` → `pynvml` import): GPU utilization (avg/max) and memory peak.
- **Workload-specific metrics**:
  - Team1: screening metrics + Tensor Core soak TFLOPS estimate
  - Team2: thermo efficiency + Tensor Core soak TFLOPS estimate
  - Team3: TFLOPS proxy + cuQuantum contraction path (`cuquantum_used=true`) + sustained soak

## Source of truth

Do not copy/paste metrics into code. Treat these artifacts as the authoritative record:
- `prototypes/team1_quantum/benchmarks/results/latest.json`
- `prototypes/team2_energy/benchmarks/results/latest.json`
- `prototypes/team3_innovation/benchmarks/results/latest.json`



