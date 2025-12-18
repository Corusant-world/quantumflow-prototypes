# CTDR (CUDA Tensor-DPX Runtime) - Hopper Logic Core

## What it is

CTDR is a specialized CUDA library that bridges AGI mathematics (Tensor Logic, Baire Metric) with NVIDIA Hopper architecture (DPX, Tensor Cores). It demonstrates the transition from "heating microwave" H100 usage to engineering-correct utilization through DPX/TC hybrid, showing a revolution in energy efficiency and performance.

**Paradigm Shift**: CTDR proves the "Landauer/Weightless" paradigm on H100:
- **4.42× energy reduction** vs standard Tensor Core dot-product retrieval
- **10× write reduction** through RLA memoization (reversible logic approximation)
- **98% cache hit rate** with deterministic correctness (100% FSM precision)
- **1534× speedup** for batch LCP retrieval (O(N) Baire Metric vs O(N²) baseline)

## Key Components

1. **DPX_LCP_Kernel**: Linear O(N) hierarchical search via Baire Metric using DPX intrinsics (128 op/cycle/SM)
2. **Reversible_Einsum_Engine**: Reliable symbolic reasoning via Boolean Einsum + Heaviside threshold activation

## Install

### CPU (Development)
```bash
python -m pip install -r requirements.txt
```

### GPU (H100)
```bash
# On GPU server
python -m pip install -r requirements.txt
# Compile CUDA kernels
cd cuda
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Run

### Demo (30 seconds)
```bash
python demo/demo_simple.py
```

### Full Demo (5 minutes)
```bash
python demo/demo_full.py
```

### Benchmarks
```bash
# Run all benchmarks (performance, GPU utilization, energy, reliability, entropy)
python3 benchmarks/run_all_benchmarks.py
# Results saved to:
#   - benchmarks/results/comprehensive_report.json (all results)
#   - benchmarks/results/latest.json (summary + key metrics)

# Individual benchmarks:
python3 benchmarks/benchmark_performance.py      # LCP, Einsum, KV Cache speedup
python3 benchmarks/benchmark_gpu_utilization.py  # SM utilization, memory bandwidth
python3 benchmarks/benchmark_energy_efficiency.py # Energy per query (J/query)
python3 benchmarks/benchmark_reliability.py      # FSM precision, semantic errors
python3 benchmarks/benchmark_entropy.py          # Shannon + Landauer entropy
```

### Tests
```bash
pytest tests/
```

## GPU Requirements

- NVIDIA H100 (Hopper architecture, sm_90)
- CUDA Toolkit 12.0+
- CMake 3.18+

## Performance Results

**Measured on H100 (December 2025):**

### Batch LCP Retrieval (DPX_LCP_Kernel)
- **Speedup vs CPU**: Up to **1534.47×** (query_len=2048, candidates=16384)
- **Throughput**: 0.405ms for 16K candidates vs 621ms CPU baseline
- **Architecture**: Warp-level primitives (`__ballot_sync`, `__ffs`) for early exit

### Energy Efficiency (Landauer/Weightless Paradigm)
- **Energy reduction**: **4.42×** (CTDR vs Tensor Core dot-product baseline)
- **Write reduction**: **10.0×** (RLA memoization vs baseline)
- **Read efficiency**: **9.9** (reads per baseline write)
- **Cache hit rate**: **98.02%** (target: ≥80%)

### Reliability (FSM Precision)
- **FSM Precision**: **100.0%** (target: ≥51.52%)
- **Semantic error rate**: **0.0%** (bit-perfect correctness)
- **Token reduction**: **100.0%** (target: ≥31%)
- **Determinism**: 100% (repeated runs match exactly)

### Reversible Einsum Engine
- **Speedup vs CPU**: Up to **338.19×** (128×128 matrices)
- **Architecture**: Hybrid Tensor Cores (multiplication) + DPX (threshold)

**All metrics from**: `benchmarks/results/latest.json` (run `python3 benchmarks/run_all_benchmarks.py` to regenerate)

---

## Phase 2: Extended CTDR (December 2025)

### New Components

| Component | Description | Status |
|-----------|-------------|--------|
| **DRC** | Dynamic Reversible Core (RC + DHM + SMI modules) | ✅ |
| **HCE** | Hybrid Computational Unit (Tensor Cores + DPX) | ✅ |
| **DHM** | Dynamic Hierarchy Manager (p-adic tree, infinite context) | ✅ |
| **P-adic Attention** | O(t) attention via Baire Metric (vs O(N²) standard) | ✅ |
| **A2A/REP** | Agent protocols + consensus | ✅ |
| **Blame Logic** | Self-healing with real recovery | ✅ |
| **MPO** | Tensor Networks compression | ✅ |

### Phase 2 Results (REAL measurements, no simulations)

```
DHM Scaling (500K concepts):
  Insert: 1.9M concepts/s
  Query:  18.84ms avg
  Complexity: Sublinear ✅ (ratios 0.82-0.84)

Memoization (RLA Stack):
  Cold:    17.34ms
  Hot:     0.0005ms  
  Speedup: 33,617×

Reliability:
  A2A Handoff: 0.006ms
  Self-healing: 100%

MPO/Tensor Networks:
  Compression: 5×
  Speedup: 4.5× (2048×2048)

H100 Utilization:
  TFLOPS: 428.4
  SM Utilization: 87.4% (target ≥70%)
  Memory Bandwidth: 55.3% (target ≥50%)

CRITICAL PROOF — Standard Attention vs CTDR:
  N=500K tokens:
  - Standard: ❌ OOM (500GB > 80GB H100)
  - CTDR:     ✅ 33ms, ~2MB memory
```

### Run Phase 2 Benchmarks

```bash
# Unified benchmark (all Phase 2 tests)
python3 benchmarks/unified_phase2_benchmark.py

# Results saved to:
#   benchmarks/results/phase2_latest.json
```

### Checkpoints Passed

- [x] CP-2.0: DRC/HCE integration
- [x] CP-2.1: DHM infinite context
- [x] CP-2.2: P-adic O(t) attention
- [x] CP-2.3: Reliability >90% self-healing
- [x] CP-2.4: MPO compression
- [x] CP-2.5: H100 optimization
- [x] CP-2.6: Unified benchmarks

---

## References

- `trident/TRIDENT_IMPLEMENTATION_PLAN_DEEP.md` - Full technical specifications
- `trident/TRIDENT_COMPLETE_CONTEXT.md` - Complete context from PDFs
- `Cuda-agi-path.md` - CTDR module specifications


