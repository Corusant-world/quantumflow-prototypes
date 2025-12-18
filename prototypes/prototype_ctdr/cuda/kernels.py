"""
Python bindings for CTDR CUDA kernels
Provides high-level interface to DPX_LCP_Kernel and Reversible_Einsum_Engine
"""

import sys
from pathlib import Path
import numpy as np

# Try to import compiled module
try:
    import ctdr_python
    CUDA_MODULE_AVAILABLE = True
except ImportError:
    CUDA_MODULE_AVAILABLE = False


def run_dpx_lcp(s1: bytes, s2: bytes) -> int:
    """
    Run DPX LCP Kernel on GPU.
    
    Args:
        s1: bytes - encoded string 1 (short2 format)
        s2: bytes - encoded string 2 (short2 format)
        
    Returns:
        int - LCP length
        
    Raises:
        ImportError: if CUDA module not compiled
        RuntimeError: if CUDA operation fails
    """
    if not CUDA_MODULE_AVAILABLE:
        raise ImportError(
            "ctdr_python module not found. "
            "Please compile the CUDA kernels first: "
            "cd build && cmake .. && make"
        )
    
    try:
        result = ctdr_python.dpx_lcp(s1, s2)
        return result
    except Exception as e:
        raise RuntimeError(f"CUDA kernel execution failed: {e}") from e


def run_dpx_lcp_with_fallback(s1: bytes, s2: bytes) -> int:
    """
    Run DPX LCP Kernel with CPU fallback.
    
    If CUDA is not available, falls back to CPU implementation.
    
    Args:
        s1: bytes - encoded string 1 (short2 format)
        s2: bytes - encoded string 2 (short2 format)
        
    Returns:
        int - LCP length
    """
    if CUDA_MODULE_AVAILABLE:
        try:
            return run_dpx_lcp(s1, s2)
        except (ImportError, RuntimeError):
            # Fallback to CPU
            pass
    
    # CPU fallback
    from ..src.core import lcp_cpu
    from ..src.encoding import decode_from_short2
    
    try:
        s1_str = decode_from_short2(s1)
        s2_str = decode_from_short2(s2)
        return lcp_cpu(s1_str, s2_str)
    except Exception as e:
        raise RuntimeError(f"CPU fallback failed: {e}") from e


def run_reversible_einsum(A: np.ndarray, B: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Run Reversible Einsum Engine on GPU.
    
    Boolean Einsum: C[i,k] = Σ_j (A[i,j] AND B[j,k])
    Heaviside threshold: H(x) = 1 if x ≥ threshold else 0
    
    Args:
        A: numpy.ndarray - Boolean matrix A (shape: [m, k])
        B: numpy.ndarray - Boolean matrix B (shape: [k, n])
        threshold: float - Heaviside threshold (default: 0.5)
        
    Returns:
        numpy.ndarray - Boolean matrix C (shape: [m, n])
        
    Raises:
        ImportError: if CUDA module not compiled
        RuntimeError: if CUDA operation fails
        ValueError: if input shapes are invalid
    """
    if not CUDA_MODULE_AVAILABLE:
        raise ImportError(
            "ctdr_python module not found. "
            "Please compile the CUDA kernels first: "
            "cd build && cmake .. && make"
        )
    
    # Validate inputs
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise ValueError("A and B must be numpy arrays")
    
    if A.dtype != bool or B.dtype != bool:
        raise ValueError("A and B must be boolean arrays (dtype=bool)")
    
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays")
    
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Shape mismatch: A.shape[1]={A.shape[1]} != B.shape[0]={B.shape[0]}"
        )
    
    try:
        result = ctdr_python.reversible_einsum(A, B, threshold)
        return result
    except Exception as e:
        raise RuntimeError(f"CUDA kernel execution failed: {e}") from e


def run_reversible_einsum_with_fallback(A: np.ndarray, B: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Run Reversible Einsum Engine with CPU fallback.
    
    If CUDA is not available, falls back to CPU implementation.
    
    Args:
        A: numpy.ndarray - Boolean matrix A (shape: [m, k])
        B: numpy.ndarray - Boolean matrix B (shape: [k, n])
        threshold: float - Heaviside threshold (default: 0.5)
        
    Returns:
        numpy.ndarray - Boolean matrix C (shape: [m, n])
    """
    if CUDA_MODULE_AVAILABLE:
        try:
            return run_reversible_einsum(A, B, threshold)
        except (ImportError, RuntimeError):
            # Fallback to CPU
            pass
    
    # CPU fallback
    from ..src.core import einsum_cpu
    
    try:
        # CPU baseline: Boolean Einsum + Heaviside
        C = einsum_cpu(A, B)
        # Apply Heaviside threshold: convert to float, compare with threshold
        C_float = C.astype(float)
        return (C_float >= threshold).astype(bool)
    except Exception as e:
        raise RuntimeError(f"CPU fallback failed: {e}") from e


def dpx_lcp_batch(query: np.ndarray, candidates: np.ndarray, max_len: int = 256) -> np.ndarray:
    """
    Batch LCP computation using DPX kernel.
    
    Compares ONE query against MANY candidates in parallel on GPU.
    This is the KEY to million-scale DHM.
    
    Args:
        query: np.ndarray - Query encoded as uint16 [max_len]
        candidates: np.ndarray - Candidates encoded as uint16 [num_candidates, max_len]
        max_len: int - Maximum path length
        
    Returns:
        np.ndarray - LCP values for each candidate [num_candidates]
    """
    if not CUDA_MODULE_AVAILABLE:
        raise ImportError("CUDA module required for dpx_lcp_batch")
    
    try:
        return ctdr_python.dpx_lcp_batch(query, candidates, max_len)
    except Exception as e:
        raise RuntimeError(f"DPX LCP batch failed: {e}") from e


def dpx_similarity_batch(query: np.ndarray, candidates: np.ndarray, max_len: int = 256) -> np.ndarray:
    """
    Batch similarity computation using DPX kernel.
    
    Computes p-adic similarity: 1.0 / (1.0 + 2^(-lcp))
    
    Args:
        query: np.ndarray - Query encoded as uint16 [max_len]
        candidates: np.ndarray - Candidates encoded as uint16 [num_candidates, max_len]
        max_len: int - Maximum path length
        
    Returns:
        np.ndarray - Similarity values for each candidate [num_candidates]
    """
    if not CUDA_MODULE_AVAILABLE:
        raise ImportError("CUDA module required for dpx_similarity_batch")
    
    try:
        return ctdr_python.dpx_similarity_batch(query, candidates, max_len)
    except Exception as e:
        raise RuntimeError(f"DPX similarity batch failed: {e}") from e


def dpx_lcp_batch_cpu(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    CPU implementation of batch LCP for testing/fallback.
    
    Uses vectorized NumPy operations.
    NOTE: This is NOT our target path - DPX GPU is the goal.
    """
    # Compare query with all candidates
    matches = (candidates == query)
    
    # Cumulative product to find first mismatch
    cumsum = np.cumprod(matches, axis=1)
    
    # LCP = sum of matches before first mismatch
    lcps = cumsum.sum(axis=1)
    
    return lcps.astype(np.int32)
