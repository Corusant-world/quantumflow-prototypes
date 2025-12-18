"""
CTDR Full Demo (5 minutes)
Complete demonstration with benchmarks, comparisons, and real metrics
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import lcp_cpu, dpx_lcp, einsum_cpu, reversible_einsum, CUDA_AVAILABLE
from src.kv_cache_steering import KVCacheSteeringDPX
from src.rla_stack import RLAStack

def main():
    print("=" * 60)
    print("CTDR Full Demo (~5 minutes)")
    print("=" * 60)
    print(f"\nCUDA kernels available: {CUDA_AVAILABLE}\n")
    
    start_time = time.perf_counter()
    
    # ============================================================
    # 1. DPX_LCP_Kernel: Batch retrieval demonstration
    # ============================================================
    print("1. DPX_LCP_Kernel (Batch Retrieval)")
    print("-" * 60)
    
    # Test cases for correctness
    test_cases = [
        ("hello", "hell", 4),
        ("world", "word", 3),
        ("test", "test", 4),
        ("abc", "xyz", 0),
    ]
    
    print("   Correctness tests:")
    all_pass = True
    for s1, s2, expected in test_cases:
        cpu_result = lcp_cpu(s1, s2)
        cuda_result = dpx_lcp(s1, s2)
        status = "✅" if cpu_result == cuda_result == expected else "❌"
        if cpu_result != cuda_result or cpu_result != expected:
            all_pass = False
        print(f"   {status} LCP('{s1}', '{s2}') = CPU:{cpu_result} CUDA:{cuda_result} (Expected: {expected})")
    
    print(f"   ✅ Correctness: {'PASS' if all_pass else 'FAIL'}")
    
    # Performance comparison
    print("\n   Performance (CPU vs CUDA):")
    sizes = [100, 1000, 10000]
    for n in sizes:
        s1 = "a" * n
        s2 = "a" * (n - 1) + "b"
        
        # CPU
        cpu_time = time.perf_counter()
        _ = lcp_cpu(s1, s2)
        cpu_time = (time.perf_counter() - cpu_time) * 1000
        
        # CUDA
        cuda_time = time.perf_counter()
        _ = dpx_lcp(s1, s2)
        cuda_time = (time.perf_counter() - cuda_time) * 1000
        
        speedup = cpu_time / cuda_time if cuda_time > 0 else 0.0
        print(f"   Size {n:5d}: CPU={cpu_time:6.3f}ms CUDA={cuda_time:6.3f}ms Speedup={speedup:6.2f}x")
    
    # ============================================================
    # 2. Reversible_Einsum_Engine: Boolean Einsum + Heaviside
    # ============================================================
    print("\n2. Reversible_Einsum_Engine (Boolean Einsum + Heaviside)")
    print("-" * 60)
    
    # Test cases
    test_matrices = [
        (2, 2, 2),
        (4, 4, 4),
        (16, 16, 16),
        (64, 64, 64),
    ]
    
    print("   Performance (CPU vs CTDR):")
    for I, J, K in test_matrices:
        rng = np.random.RandomState(42)
        A = (rng.rand(I, J) > 0.5).astype(bool)
        B = (rng.rand(J, K) > 0.5).astype(bool)
        
        # CPU baseline
        cpu_time = time.perf_counter()
        cpu_result = einsum_cpu(A, B)
        cpu_time = (time.perf_counter() - cpu_time) * 1000
        
        # CTDR (CUDA or CPU fallback)
        cuda_time = time.perf_counter()
        cuda_result = reversible_einsum(A, B, threshold=0.5)
        cuda_time = (time.perf_counter() - cuda_time) * 1000
        
        speedup = cpu_time / cuda_time if cuda_time > 0 else 0.0
        correct = np.array_equal(cpu_result, cuda_result)
        print(f"   {I}×{J} @ {J}×{K}: CPU={cpu_time:6.3f}ms CTDR={cuda_time:6.3f}ms Speedup={speedup:6.2f}x {'✅' if correct else '❌'}")
    
    # ============================================================
    # 3. KV_Cache_Steering_DPX: Memoization and cache performance
    # ============================================================
    print("\n3. KV_Cache_Steering_DPX (Memoization)")
    print("-" * 60)
    
    cache = KVCacheSteeringDPX(sram_size=100, l2_size=1000)
    rng = np.random.RandomState(123)
    
    # Simulate repeated operations with memoization
    num_operations = 200
    unique_keys = 20
    
    print(f"   Simulating {num_operations} operations with {unique_keys} unique keys:")
    
    baseline_time = time.perf_counter()
    for i in range(num_operations):
        key = f"op_{i % unique_keys}"
        cached = cache.get(key)
        if cached is None:
            # Simulate computation
            A = (rng.rand(16, 16) > 0.5).astype(bool)
            B = (rng.rand(16, 16) > 0.5).astype(bool)
            result = reversible_einsum(A, B, threshold=0.5)
            cache.put(key, result.astype(np.float32))
    baseline_time = time.perf_counter() - baseline_time
    
    cache_stats = cache.get_stats()
    print(f"   Total operations: {cache_stats['total_queries']}")
    print(f"   Cache hits: {cache_stats['cache_hits']}")
    print(f"   Cache misses: {cache_stats['cache_misses']}")
    print(f"   Cache hit rate: {cache_stats['cache_hit_rate']:.2f}%")
    print(f"   Time: {baseline_time:.3f}s")
    print(f"   ✅ Cache hit rate: {'PASS' if cache_stats['cache_hit_rate'] >= 80.0 else 'FAIL'} (target: ≥80%)")
    
    # ============================================================
    # 4. RLA Stack: Entropy metrics and write reduction
    # ============================================================
    print("\n4. RLA Stack (Entropy Metrics & Write Reduction)")
    print("-" * 60)
    
    rla = RLAStack()
    cache_rla = KVCacheSteeringDPX(sram_size=100, l2_size=1000, rla_stack=rla)
    
    num_operations = 200
    unique_keys = 20
    
    print(f"   Baseline: {num_operations} operations (no memoization) = {num_operations} writes")
    print(f"   RLA: {num_operations} operations (with memoization):")
    
    for i in range(num_operations):
        key = f"einsum_{i % unique_keys}"
        cached = cache_rla.get_with_rla(key)
        if cached is None:
            A = (rng.rand(16, 16) > 0.5).astype(bool)
            B = (rng.rand(16, 16) > 0.5).astype(bool)
            result = reversible_einsum(A, B, threshold=0.5)
            cache_rla.put_with_rla(key, result.astype(np.float32))
    
    rla_stats = rla.get_stats()
    write_reduction = num_operations / rla_stats['memory_writes'] if rla_stats['memory_writes'] > 0 else 0.0
    
    print(f"   Memory writes: {rla_stats['memory_writes']}")
    print(f"   Memory reads: {rla_stats['memory_reads']}")
    print(f"   Write reduction: {write_reduction:.2f}x (target: ≥2.0x)")
    print(f"   Cache hit rate: {rla_stats['cache_hit_rate']:.2f}%")
    print(f"   ✅ Write reduction: {'PASS' if write_reduction >= 2.0 else 'FAIL'}")
    
    # ============================================================
    # 5. Summary: Key metrics
    # ============================================================
    elapsed = time.perf_counter() - start_time
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"   Total time: {elapsed:.1f}s")
    print(f"   CUDA available: {CUDA_AVAILABLE}")
    print(f"   Cache hit rate: {cache_stats['cache_hit_rate']:.2f}%")
    print(f"   Write reduction: {write_reduction:.2f}x")
    print("\n" + "=" * 60)
    print("✅ Full demo complete")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Run all benchmarks: python3 benchmarks/run_all_benchmarks.py")
    print("  - View comprehensive report: cat benchmarks/results/comprehensive_report.json | python3 -m json.tool")
    print("  - View latest metrics: cat benchmarks/results/latest.json | python3 -m json.tool")

if __name__ == "__main__":
    main()
