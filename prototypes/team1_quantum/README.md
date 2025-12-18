# Team1 Quantum — Vacancy/Stability Hunter (GPU-first, CPU fallback)

## What it is
`QuantumVacancyHunter` is a GPU-heavy (when CUDA is available) surrogate workload for vacancy/stability-style screening.
The benchmark intentionally includes **Tensor Core** matmul soak (FP16/BF16) to produce reproducible utilization.

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
- `BENCH_SECONDS` (default 60)
- `BENCH_RESERVE_GB` (default 0)

## Metrics (H100; from `benchmarks/results/latest.json`)
- **nvml.gpu_util_avg**: 95.19% (max 100%)
- **TensorCore soak TFLOPS(est)**: ~453
- **mem_used_mb_max**: 3921 MB

## Ecosystem
Smoke test: `python prototypes/ecosystem_smoke.py` (see `prototypes/ECOSYSTEM_COMPATIBILITY.md`).
