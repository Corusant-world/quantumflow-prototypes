"""
CTDR Simple Demo (30 seconds)
Quick demonstration of CTDR capabilities with real kernels and metrics
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import lcp_cpu, dpx_lcp, reversible_einsum, einsum_cpu, CUDA_AVAILABLE
from src.kv_cache_steering import KVCacheSteeringDPX
from src.rla_stack import RLAStack

def main():
    print("=" * 60)
    print("CTDR Simple Demo (~30 seconds)")
    print("=" * 60)
    print(f"\nCUDA kernels available: {CUDA_AVAILABLE}\n")

    start_time = time.perf_counter()

    # 1) DPX_LCP_Kernel: CPU vs CUDA comparison
    print("1. DPX_LCP_Kernel (Baire Metric, O(N))")
    print("-" * 60)
    s1, s2 = "hello world", "hello there"
    cpu_time = time.perf_counter()
    cpu_result = lcp_cpu(s1, s2)
    cpu_time = (time.perf_counter() - cpu_time) * 1000
    
    cuda_time = time.perf_counter()
    cuda_result = dpx_lcp(s1, s2)
    cuda_time = (time.perf_counter() - cuda_time) * 1000
    
    speedup = cpu_time / cuda_time if cuda_time > 0 else 0.0
    print(f"   Input: '{s1}' vs '{s2}'")
    print(f"   CPU:   {cpu_result} ({cpu_time:.3f} ms)")
    print(f"   CUDA:  {cuda_result} ({cuda_time:.3f} ms)")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   ✅ Correctness: {'PASS' if cpu_result == cuda_result else 'FAIL'}")

    # 2) Reversible_Einsum_Engine: Boolean Einsum + Heaviside
    print("\n2. Reversible_Einsum_Engine (Boolean Einsum + Heaviside)")
    print("-" * 60)
    A = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    B = np.array([[1, 1], [0, 0], [1, 0]], dtype=bool)
    
    # CPU baseline
    cpu_time = time.perf_counter()
    cpu_result = einsum_cpu(A, B)
    cpu_time = (time.perf_counter() - cpu_time) * 1000
    
    # CTDR (CUDA or CPU fallback)
    cuda_time = time.perf_counter()
    cuda_result = reversible_einsum(A, B, threshold=0.5)
    cuda_time = (time.perf_counter() - cuda_time) * 1000
    
    speedup = cpu_time / cuda_time if cuda_time > 0 else 0.0
    print(f"   Input: A{A.shape} @ B{B.shape}")
    print(f"   CPU:   {cpu_time:.3f} ms")
    print(f"   CTDR:  {cuda_time:.3f} ms")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Output shape: {cuda_result.shape}")
    print(f"   ✅ Correctness: {'PASS' if np.array_equal(cpu_result, cuda_result) else 'FAIL'}")

    # 3) KV_Cache_Steering_DPX: Memoization demonstration
    print("\n3. KV_Cache_Steering_DPX (Memoization)")
    print("-" * 60)
    cache = KVCacheSteeringDPX(sram_size=16, l2_size=64)
    
    # First put (cache miss)
    cache.put("einsum_result", cuda_result.astype(np.float32))
    stats1 = cache.get_stats()
    
    # Multiple gets (cache hits)
    for i in range(5):
        _ = cache.get("einsum_result")
    
    stats2 = cache.get_stats()
    print(f"   Cache hits: {stats2['cache_hits']}")
    print(f"   Cache misses: {stats2['cache_misses']}")
    print(f"   Cache hit rate: {stats2['cache_hit_rate']:.2f}%")
    print(f"   ✅ Memoization: {'WORKING' if stats2['cache_hit_rate'] > 0 else 'NOT WORKING'}")

    # 4) RLA Stack: Entropy metrics
    print("\n4. RLA Stack (Entropy Metrics)")
    print("-" * 60)
    rla = RLAStack()
    cache_rla = KVCacheSteeringDPX(sram_size=16, l2_size=64, rla_stack=rla)
    
    # Perform operations with memoization
    for i in range(10):
        key = f"op_{i % 3}"  # 3 unique keys, 10 operations
        cached = cache_rla.get_with_rla(key)
        if cached is None:
            result = reversible_einsum(A, B, threshold=0.5)
            cache_rla.put_with_rla(key, result.astype(np.float32))
    
    rla_stats = rla.get_stats()
    print(f"   Memory writes: {rla_stats['memory_writes']}")
    print(f"   Memory reads: {rla_stats['memory_reads']}")
    print(f"   Cache hit rate: {rla_stats['cache_hit_rate']:.2f}%")
    print(f"   Write reduction: {10.0 / rla_stats['memory_writes']:.2f}x (target: ≥2.0x)")

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 60)
    print(f"✅ Demo complete ({elapsed:.1f}s)")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Run full benchmarks: python3 benchmarks/run_all_benchmarks.py")
    print("  - View results: cat benchmarks/results/latest.json | python3 -m json.tool")
    print("  - Full demo: python3 demo/demo_full.py")

if __name__ == "__main__":
    main()

