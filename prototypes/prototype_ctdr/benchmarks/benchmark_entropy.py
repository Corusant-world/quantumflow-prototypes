"""
CTDR Entropy Benchmarks (Unified Metric)

We report:
- Information entropy (Shannon) of outputs
- Thermodynamic lower bound via Landauer (energy per erase) as implemented in RLAStack
- Symmetric A/B comparison: Baseline (no memoization) vs RLA (with memoization)

This benchmark is CPU/GPU agnostic; it measures the information-theoretic and RLA-level metrics.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kv_cache_steering import KVCacheSteeringDPX
from src.rla_stack import RLAStack
from src.core import reversible_einsum, einsum_cpu

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "entropy.json"


def run_entropy_benchmark(num_operations: int = 200, unique_keys: int = 20) -> Dict[str, Any]:
    """
    Run symmetric A/B comparison:
    - Baseline: no memoization (every operation = compute + write)
    - RLA: with memoization (first compute + write, then read from cache)
    """
    print("=" * 60, flush=True)
    print("CTDR ENTROPY BENCHMARK (A/B Comparison)", flush=True)
    print("=" * 60, flush=True)

    # Prepare inputs (stable / reproducible)
    rng = np.random.RandomState(123)
    A = rng.rand(16, 16) > 0.5
    B = rng.rand(16, 16) > 0.5
    threshold = 0.5

    # ============================================================
    # BASELINE PATH: No memoization (worst-case)
    # ============================================================
    print("\n[Baseline] Running without memoization...", flush=True)
    baseline_rla = RLAStack()
    baseline_cache = KVCacheSteeringDPX(rla_stack=baseline_rla)
    
    baseline_outputs = []
    for i in range(num_operations):
        key = f"einsum_{i % unique_keys}"
        # Baseline: always compute, always write (no cache lookup)
        result = reversible_einsum(A, B, threshold)
        baseline_cache.put_with_rla(key, result)  # This increments writes
        baseline_outputs.append(result)
    
    baseline_stats = baseline_rla.get_stats()
    baseline_writes = baseline_stats["memory_writes"]
    baseline_reads = baseline_stats["memory_reads"]
    baseline_cache_stats = baseline_cache.get_stats()
    
    print(f"  Baseline writes: {baseline_writes}, reads: {baseline_reads}", flush=True)
    print(f"  Baseline cache hits: {baseline_cache_stats['cache_hits']}, misses: {baseline_cache_stats['cache_misses']}", flush=True)

    # ============================================================
    # RLA PATH: With memoization
    # ============================================================
    print("\n[RLA] Running with memoization...", flush=True)
    rla = RLAStack()
    cache = KVCacheSteeringDPX(rla_stack=rla)
    
    rla_outputs = []
    for i in range(num_operations):
        key = f"einsum_{i % unique_keys}"
        # RLA: try to get from cache first (this increments reads/hits)
        cached = cache.get_with_rla(key)
        if cached is not None:
            # Cache hit: read instead of compute+write
            rla_outputs.append(cached)
        else:
            # Cache miss: compute, then put (this increments writes)
            result = reversible_einsum(A, B, threshold)
            cache.put_with_rla(key, result)
            rla_outputs.append(result)
    
    rla_stats = rla.get_stats()
    rla_writes = rla_stats["memory_writes"]
    rla_reads = rla_stats["memory_reads"]
    rla_cache_stats = cache.get_stats()
    rla_cache_hits = rla_cache_stats["cache_hits"]
    rla_cache_misses = rla_cache_stats["cache_misses"]
    rla_cache_hit_rate = rla_cache_stats["cache_hit_rate"]
    
    print(f"  RLA writes: {rla_writes}, reads: {rla_reads}", flush=True)
    print(f"  RLA cache hits: {rla_cache_hits}, misses: {rla_cache_misses}, hit rate: {rla_cache_hit_rate:.2f}%", flush=True)

    # ============================================================
    # Information Entropy (Shannon) - same for both paths
    # ============================================================
    if rla_outputs:
        flat = np.concatenate([o.astype(np.int8).flatten() for o in rla_outputs])
    else:
        flat = np.array([], dtype=np.int8)
    
    info_entropy_bits = rla.compute_information_entropy(flat)

    # ============================================================
    # Thermodynamic Entropy (Landauer)
    # ============================================================
    thermo_energy_baseline = rla.compute_thermodynamic_entropy(baseline_writes)
    thermo_energy_rla = rla.compute_thermodynamic_entropy(rla_writes)
    
    # Energy reduction factor
    energy_reduction = thermo_energy_baseline / thermo_energy_rla if rla_writes > 0 else 0.0

    # ============================================================
    # Comparison Metrics
    # ============================================================
    write_reduction = baseline_writes / rla_writes if rla_writes > 0 else 0.0
    read_efficiency = rla_reads / baseline_writes if baseline_writes > 0 else 0.0  # How many reads replaced writes
    target_reduction = 2.0
    meets_target = write_reduction >= target_reduction

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "num_operations": num_operations,
            "unique_keys": unique_keys,
        },
        "information_entropy_bits": float(info_entropy_bits),
        "thermodynamic": {
            "k_b": rla.K_B,
            "temperature_k": rla.T,
            "e_min_joules": rla.E_MIN,
            "baseline_energy_joules": float(thermo_energy_baseline),
            "rla_energy_joules": float(thermo_energy_rla),
            "energy_reduction_factor": float(energy_reduction),
        },
        "baseline_stats": {
            "memory_writes": baseline_writes,
            "memory_reads": baseline_reads,
            "cache_hits": baseline_cache_stats["cache_hits"],
            "cache_misses": baseline_cache_stats["cache_misses"],
            "cache_hit_rate": baseline_cache_stats["cache_hit_rate"],
        },
        "rla_stats": {
            "memory_writes": rla_writes,
            "memory_reads": rla_reads,
            "cache_hits": rla_cache_hits,
            "cache_misses": rla_cache_misses,
            "cache_hit_rate": rla_cache_hit_rate,
            "total_operations": rla_stats["total_operations"],
            "memoization_cache_size": rla_stats["memoization_cache_size"],
            "entropy_log_entries": rla_stats["entropy_log_entries"],
        },
        "comparison": {
            "write_reduction_factor": float(write_reduction),
            "read_efficiency": float(read_efficiency),
            "energy_reduction_factor": float(energy_reduction),
            "target_reduction": target_reduction,
            "meets_target": meets_target,
        },
        "meets_target": meets_target,
        "notes": [
            "Information entropy is computed on the concatenated outputs (Shannon).",
            "Thermodynamic metric uses Landauer lower bound per erase (E_min).",
            "Baseline: no memoization (every operation = compute + write).",
            "RLA: with memoization (first compute + write, then read from cache).",
            f"Cache hit rate: {rla_cache_hit_rate:.2f}% (target: ≥80% for production).",
        ],
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Write reduction: {write_reduction:.2f}x (target: ≥{target_reduction}x)", flush=True)
    print(f"Energy reduction: {energy_reduction:.2f}x", flush=True)
    print(f"Cache hit rate: {rla_cache_hit_rate:.2f}%", flush=True)
    print(f"Read efficiency: {read_efficiency:.2f} (reads per baseline write)", flush=True)
    print(f"Meets target: {meets_target}", flush=True)
    print(f"\nResults saved to: {RESULTS_FILE}", flush=True)
    
    return result


if __name__ == "__main__":
    run_entropy_benchmark()


