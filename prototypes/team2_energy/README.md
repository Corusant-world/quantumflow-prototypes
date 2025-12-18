# Team2 Energy — Grid + Thermo Kernel (GPU-first, CPU fallback)

## What it is
Components:
- `EnergyGridSystem`: topology/loss optimization on a grid-like graph.
- `TNP_System`: differentiable thermo workload (gradients + steps) plus Tensor Core soak for measurable utilization.

## Install

```bash
python -m pip install -r requirements.txt
```

If you want a single shared environment for all 3 prototypes, use `prototypes/README.md`.

## Run

Demo (CPU/CUDA):

```bash
python demo/demo.py
```

Benchmark (writes `benchmarks/results/latest.json`):

```bash
python benchmarks/run_benchmarks.py
```

Env:
- `BENCH_SECONDS` (default 120)
- `BENCH_RESERVE_GB` (default 0)

## Metrics (H100; from `benchmarks/results/latest.json`)
- **nvml.gpu_util_avg**: 95.44% (max 100%)
- **TensorCore soak TFLOPS(est)**: ~451
- **mem_used_mb_max**: 3735 MB
- **TNP thermo efficiency**: ~96.89%

## Ecosystem
Smoke test: `python prototypes/ecosystem_smoke.py` (see `prototypes/ECOSYSTEM_COMPATIBILITY.md`).
