"""
CTDR Demo - Main demo script
"""

from src.core import dpx_lcp, reversible_einsum
import numpy as np

def main():
    print("=== CTDR (CUDA Tensor-DPX Runtime) Demo ===\n")
    
    # LCP Demo
    print("1. DPX_LCP_Kernel Demo:")
    s1 = "hello world"
    s2 = "hello there"
    lcp_result = dpx_lcp(s1, s2)
    print(f"   LCP('{s1}', '{s2}') = {lcp_result}\n")
    
    # Einsum Demo
    print("2. Reversible_Einsum_Engine Demo:")
    A = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    B = np.array([[1, 1], [0, 0], [1, 0]], dtype=bool)
    result = reversible_einsum(A, B, threshold=0.5)
    print(f"   Einsum result shape: {result.shape}")
    print(f"   Result:\n{result}\n")
    
    print("=== Demo Complete ===")

if __name__ == "__main__":
    main()


