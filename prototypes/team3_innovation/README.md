# Team3 Innovation — QNP + cuQuantum Proof (GPU-first, CPU fallback)

## What it is
Team3 is the only prototype where **cuQuantum is directly relevant**.

Components:
- `quantum_neural_process(x, qubits=...)`: fast quantum-inspired block (CPU or CUDA).
- Benchmark harness:
  - TFLOPS proxy (Tensor Cores via BF16 matmul)
  - cuQuantum contraction path → `cuquantum_used=true`
  - GPU soak to sustain high NVML utilization

## Install

Base:

```bash
python -m pip install -r requirements.txt
```

If you want a single shared environment for all 3 prototypes, use `prototypes/README.md`.

cuQuantum (required for `cuquantum_used=true`; NGC extra index):

```bash
python -m pip install -U --extra-index-url https://pypi.ngc.nvidia.com cuquantum-python
```

Check:

```bash
python -c "from cuquantum import cutensornet, custatevec; print('OK')"
```

## Run

Demo (CPU/CUDA):

```bash
python demo/demo.py
```

Benchmark (writes `benchmarks/results/latest.json`):

```bash
python benchmarks/run_benchmarks.py
```

## Metrics (H100; from `benchmarks/results/latest.json`)
- **cuquantum_used**: true
- **nvml.gpu_util_avg**: 95.47% (max 100%)
- **mem_used_mb_max**: 4531 MB
- **TensorCore soak TFLOPS(est)**: ~449
- **cuQuantum contraction**: dim=4096, iters=316, tflops_est≈5.42

## Ecosystem
Smoke test: `python prototypes/ecosystem_smoke.py` (see `prototypes/ECOSYSTEM_COMPATIBILITY.md`).
