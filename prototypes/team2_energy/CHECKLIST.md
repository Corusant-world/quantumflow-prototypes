# Team2 Energy — CHECKLIST (fact-based)

Source of truth: `benchmarks/results/latest.json`

## How to reproduce

```bash
python benchmarks/run_benchmarks.py
```

## GPU metrics (H100)
From `benchmarks/results/latest.json`:
- `nvml.gpu_util_avg`: **95.44%**
- `nvml.gpu_util_max`: **100%**
- `nvml.mem_used_mb_max`: **3735 MB**
- `benchmarks.gpu_soak_matmul.tflops_est`: **~451**

## Domain metrics
- `benchmarks.tnp_thermo.stats.efficiency`: **~96.89%**

## Status
- **PASS** (avg ≥ 70%)

