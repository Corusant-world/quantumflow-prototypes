"""
CTDR Core API - Interface to CUDA Kernels
"""

import numpy as np

# Try to import CUDA kernels
CUDA_AVAILABLE = False
try:
    # Try relative import first (when used as package)
    try:
        from ..cuda.kernels import run_dpx_lcp_with_fallback, run_reversible_einsum_with_fallback
        CUDA_AVAILABLE = True
    except (ImportError, ValueError):
        # Try absolute import (when used as script)
        import sys
        from pathlib import Path
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        from cuda.kernels import run_dpx_lcp_with_fallback, run_reversible_einsum_with_fallback
        CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    import sys
    if sys.stderr:
        print("Warning: CUDA kernels not available. Using CPU fallback.", file=sys.stderr)


def lcp_cpu(s1: str, s2: str) -> int:
    """
    CPU baseline для LCP (Longest Common Prefix).
    
    Примеры:
    - lcp_cpu("hello", "hell") → 4
    - lcp_cpu("abc", "def") → 0
    """
    n = min(len(s1), len(s2))
    for i in range(n):
        if s1[i] != s2[i]:
            return i
    return n


def einsum_cpu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    CPU baseline для Boolean Einsum.
    
    C[i,k] = Σ_j (A[i,j] AND B[j,k])
    
    Args:
        A: Boolean массив shape (I, J)
        B: Boolean массив shape (J, K)
    
    Returns:
        Boolean массив shape (I, K)
    """
    # Простая реализация через numpy
    # C[i,k] = any_j (A[i,j] AND B[j,k])
    I, J = A.shape
    J2, K = B.shape
    if J != J2:
        raise ValueError(f"Shape mismatch: A.shape[1]={J} != B.shape[0]={J2}")
    
    # Boolean matrix multiplication: C = A @ B (where @ is AND-OR)
    C = np.zeros((I, K), dtype=bool)
    for i in range(I):
        for k in range(K):
            # C[i,k] = OR_j (A[i,j] AND B[j,k])
            C[i, k] = np.any(A[i, :] & B[:, k])
    
    return C


def dpx_lcp(s1: str, s2: str) -> int:
    """DPX LCP Kernel wrapper."""
    # CTDR CUDA binding expects short2-encoded bytes, not raw utf-8 bytes.
    from .encoding import encode_to_short2
    if CUDA_AVAILABLE:
        try:
            return run_dpx_lcp_with_fallback(encode_to_short2(s1), encode_to_short2(s2))
        except Exception:
            return lcp_cpu(s1, s2)
    else:
        return lcp_cpu(s1, s2)


def reversible_einsum(A: np.ndarray, B: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Reversible Einsum Engine wrapper."""
    if CUDA_AVAILABLE:
        try:
            return run_reversible_einsum_with_fallback(A, B, threshold)
        except Exception:
            # Fallback to CPU if CUDA fails
            pass
    
    # CPU fallback
    try:
        import torch
        # Convert bool to float for einsum, then convert back
        A_torch = torch.tensor(A, dtype=torch.float32)
        B_torch = torch.tensor(B, dtype=torch.float32)
        C = torch.einsum('ij,jk->ik', A_torch, B_torch)
        return (C >= threshold).numpy().astype(bool)
    except ImportError:
        # Fallback to pure numpy if torch not available
        return einsum_cpu(A, B).astype(float) >= threshold
    except Exception:
        # Fallback to pure numpy if torch fails for any reason
        return einsum_cpu(A, B).astype(float) >= threshold

