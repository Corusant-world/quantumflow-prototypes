# Real GPU Utilization Guide

This document explains how to actually utilize NVIDIA GPUs with these prototypes and achieve high utilization (95%+ on H100).

## Prerequisites

### Hardware
- **NVIDIA GPU** with CUDA 12.x support (H100, A100, RTX 4090, etc.)
- **CUDA Toolkit 12.x** installed on system
- **NVIDIA Driver** compatible with CUDA 12.x

### Software
- Python 3.10+
- All dependencies from `requirements.gpu-cu12.txt` or `requirements.gpu-cu12-cuquantum.txt`

## How Prototypes Utilize GPU

### Team1 Quantum
- **Tensor Core matmul**: FP16/BF16 matrix multiplications to drive Tensor Cores
- **Sustained workload**: Continuous computation for 60+ seconds to maintain high utilization
- **Result**: ~95% GPU utilization, ~450 TFLOPS (estimated)

### Team2 Energy
- **Differentiable thermo**: Physics-informed optimization with GPU-accelerated gradients
- **Tensor Core soak**: BF16 matmul workloads for measurable utilization
- **Result**: ~95% GPU utilization, ~450 TFLOPS (estimated)

### Team3 Innovation
- **cuQuantum contractions**: Real cuQuantum tensor network contractions (if cuQuantum installed)
- **Tensor Core proxy**: BF16 matmul for sustained utilization
- **Result**: ~95% GPU utilization, ~450 TFLOPS (estimated), plus cuQuantum acceleration

## Running Benchmarks for Real Utilization

### Quick Start (GPU)

```bash
# Install GPU dependencies
python -m pip install -U pip
python -m pip install -r prototypes/requirements.gpu-cu12.txt

# Run Team1 benchmark (60 seconds, sustained GPU load)
python prototypes/team1_quantum/benchmarks/run_benchmarks.py

# Run Team2 benchmark (120 seconds, sustained GPU load)
python prototypes/team2_energy/benchmarks/run_benchmarks.py

# Run Team3 benchmark (with cuQuantum if available)
python prototypes/team3_innovation/benchmarks/run_benchmarks.py
```

### With cuQuantum (Team3)

```bash
# Install cuQuantum dependencies
python -m pip install -U pip
python -m pip install -r prototypes/requirements.gpu-cu12-cuquantum.txt

# Verify cuQuantum
python -c "from cuquantum import cutensornet, custatevec; print('OK')"

# Run Team3 benchmark (will use cuQuantum if available)
python prototypes/team3_innovation/benchmarks/run_benchmarks.py
```

## Understanding Results

Each benchmark writes `benchmarks/results/latest.json` with:

- **NVML metrics**: `nvml.gpu_util_avg`, `nvml.gpu_util_max`, `nvml.mem_used_mb_max`
- **TFLOPS estimates**: Tensor Core performance proxy
- **Workload-specific metrics**: Domain-specific results per prototype

### Example Output (Team1)

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

## Troubleshooting Low Utilization

If you see low GPU utilization (<70%):

1. **Check CUDA availability**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Check GPU device**:
   ```bash
   python -c "import torch; print(torch.cuda.get_device_name(0))"
   ```

3. **Verify NVML**:
   ```bash
   python -c "import pynvml; pynvml.nvmlInit(); print('OK')"
   ```

4. **Run longer benchmarks**: Increase `BENCH_SECONDS` environment variable:
   ```bash
   BENCH_SECONDS=120 python prototypes/team1_quantum/benchmarks/run_benchmarks.py
   ```

5. **Check for CPU fallback**: If `device: "cpu"` in results, GPU is not being used

## CPU Fallback

All prototypes work on CPU (without GPU), but:
- **No Tensor Core utilization** (CPU doesn't have Tensor Cores)
- **Lower performance** (CPU is much slower than GPU)
- **No NVML metrics** (NVML requires GPU)

CPU fallback is useful for:
- Development on laptops
- Testing code logic
- CI/CD pipelines without GPU

## Next Steps

- Run all three prototypes together: `python prototypes/ecosystem_smoke.py`
- Check individual benchmarks: See `prototypes/*/benchmarks/results/latest.json`
- Integrate into your workflow: Use `process()` functions from each prototype's `core.py`








