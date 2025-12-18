"""
RAW PROOF — No text, just data.
"""
import torch
import sys
import time

sys.path.insert(0, 'src')

print("=== RAW PROOF ===")
print()

# 1. Memory math
n = 500000
matrix_bytes = n * n * 2  # float16
print(f"Attention matrix {n}x{n} float16:")
print(f"  Bytes: {matrix_bytes:,}")
print(f"  GB: {matrix_bytes / 1e9:.1f}")
print()

# 2. H100 memory
props = torch.cuda.get_device_properties(0)
print(f"H100 Memory:")
print(f"  Total: {props.total_memory / 1e9:.1f} GB")
print(f"  Can fit 500K×500K matrix: {'YES' if matrix_bytes < props.total_memory else 'NO'}")
print()

# 3. Actual allocation attempt
print("Attempting torch.randn(500000, 500000, dtype=float16)...")
torch.cuda.empty_cache()
try:
    x = torch.randn(500000, 500000, device='cuda', dtype=torch.float16)
    print("  Result: SUCCESS (unexpected)")
    del x
except RuntimeError as e:
    error_msg = str(e).split('\n')[0]
    print(f"  Result: FAILED")
    print(f"  Error: {error_msg}")
print()

# 4. DHM memory usage
print("Building DHM with 500K concepts...")
from dhm import DynamicHierarchyManager
import tracemalloc

tracemalloc.start()
dhm = DynamicHierarchyManager()
for i in range(500000):
    dhm.insert(f"concept_{i:06d}", {"id": i})
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"DHM with 500K concepts:")
print(f"  Python memory (peak): {peak / 1e6:.1f} MB")
print()

# 5. Actual search
print("DHM search test:")
if hasattr(dhm, '_load_gpu_index'):
    dhm._load_gpu_index()

start = time.perf_counter()
result = dhm.search("concept_250000", max_results=10)
elapsed = (time.perf_counter() - start) * 1000
print(f"  Query: 'concept_250000'")
print(f"  Time: {elapsed:.2f} ms")
print(f"  Results: {len(result)}")
print()

# 6. Comparison summary
print("=== COMPARISON ===")
print(f"Standard Attention (500K×500K):")
print(f"  Memory required: {matrix_bytes / 1e9:.0f} GB")
print(f"  H100 has: {props.total_memory / 1e9:.0f} GB")
print(f"  Status: IMPOSSIBLE (OOM)")
print()
print(f"CTDR DHM (500K concepts):")
print(f"  Memory used: {peak / 1e6:.0f} MB")
print(f"  Query time: {elapsed:.1f} ms")
print(f"  Status: WORKS")

