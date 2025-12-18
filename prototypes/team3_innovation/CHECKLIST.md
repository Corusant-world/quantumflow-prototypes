# Team3 Innovation — CHECKLIST (fact-based)

Source of truth: `benchmarks/results/latest.json`

## How to reproduce

```bash
python benchmarks/run_benchmarks.py
```

## GPU metrics (H100)
From `benchmarks/results/latest.json`:
- `cuquantum_used`: **true**
- `nvml.gpu_util_avg`: **95.47%**
- `nvml.gpu_util_max`: **100%**
- `nvml.mem_used_mb_max`: **4531 MB**
- `benchmarks.gpu_soak_matmul.tflops_est`: **~449**

## cuQuantum proof
- `benchmarks.cuquantum_path.used`: **true**
- `benchmarks.cuquantum_path.dim`: **4096**

## Status
- **PASS** (avg ≥ 85% target met; cuQuantum used)

