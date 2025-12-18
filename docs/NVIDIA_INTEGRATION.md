# NVIDIA Integration Notes (Prototypes)

This repo is NVIDIA-first by design. The prototypes target GPU utilization, reproducible metrics, and optional cuQuantum acceleration where it makes sense.

## CUDA / PyTorch CUDA

All three prototypes run on CPU fallback and on NVIDIA GPUs via PyTorch CUDA.

## Tensor Cores

The benchmark harnesses include BF16/FP16 matmul soak workloads to drive Tensor Cores and sustain high utilization. This is recorded via:
- TFLOPS proxy estimates
- NVML utilization sampling

## NVML metrics

We record:
- `nvml.gpu_util_avg`, `nvml.gpu_util_max`
- `nvml.mem_used_mb_max`

Implementation detail:
- We depend on `nvidia-ml-py` (it provides a `pynvml` import path).
- Do **not** install the deprecated `pynvml` pip package directly (it can introduce import-hook issues).

## cuQuantum (Team3 only)

Team3 uses cuQuantum contractions to provide a concrete “NVIDIA-native” quantum acceleration proof:
- `cuquantum_used=true`
- `benchmarks.cuquantum_path.used=true`

Install via NGC extra index (see `prototypes/requirements.gpu-cu12-cuquantum.txt`).



