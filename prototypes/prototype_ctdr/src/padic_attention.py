"""
P-adic Attention — O(N) attention mechanism using DPX LCP.

Based on Baire Metric: d(x,y) = p^(-LCP(x,y))
- LCP = Longest Common Prefix
- Closer in p-adic space = longer shared prefix = higher similarity

Complexity: O(N) vs O(N²) for standard attention.
"""

import numpy as np
from typing import List, Tuple, Optional
import sys
import os

# Add paths for imports
src_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_path)
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from encoding import encode_to_short2

# Try to import GPU module
_ctdr_python = None
try:
    import ctdr_python
    _ctdr_python = ctdr_python
except ImportError:
    pass

GPU_AVAILABLE = (_ctdr_python is not None) and hasattr(_ctdr_python, 'dpx_lcp')


def _compute_lcp_cpu(s1: str, s2: str) -> int:
    """CPU fallback for LCP computation."""
    lcp = 0
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] == s2[i]:
            lcp += 1
        else:
            break
    return lcp


def _compute_lcp_gpu(s1_encoded: bytes, s2_encoded: bytes) -> int:
    """GPU LCP via DPX."""
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")
    return _ctdr_python.dpx_lcp(s1_encoded, s2_encoded)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def p_adic_similarity(lcp: int, p: int = 2) -> float:
    """
    P-adic similarity from LCP.
    
    distance = p^(-LCP)
    similarity = 1 / (1 + distance)
    
    When LCP=0: distance=1, similarity=0.5
    When LCP→∞: distance→0, similarity→1
    """
    if lcp <= 0:
        return 0.5  # No common prefix
    distance = p ** (-lcp)
    return 1.0 / (1.0 + distance)


class PadicAttention:
    """
    P-adic Attention mechanism.
    
    Uses Baire Metric (LCP-based) for computing attention weights.
    Complexity: O(N) where N = number of keys.
    
    Standard attention: O(N²) for N queries × N keys
    P-adic attention: O(N) per query (single pass through keys)
    """
    
    def __init__(self, p: int = 2, use_gpu: bool = True):
        """
        Args:
            p: Prime base for p-adic metric (default 2)
            use_gpu: Use GPU acceleration if available
        """
        self.p = p
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self._key_cache: List[Tuple[str, bytes]] = []
        self._gpu_index_loaded = False
    
    def set_keys(self, keys: List[str]):
        """
        Pre-encode keys for batch processing.
        Loads index to GPU if available.
        
        Args:
            keys: List of key strings
        """
        self._key_cache = [(k, encode_to_short2(k)) for k in keys]
        self._gpu_index_loaded = False
        
        # Load to GPU for batch processing
        if self.use_gpu and len(keys) > 0:
            try:
                encoded_keys = [encode_to_short2(k) for k in keys]
                concatenated = b''.join(encoded_keys)
                _ctdr_python.dpx_lcp_index_load(concatenated, len(keys))
                self._gpu_index_loaded = True
            except Exception:
                self._gpu_index_loaded = False
    
    def compute_similarities(self, query: str) -> np.ndarray:
        """
        Compute p-adic similarities between query and all keys.
        
        Complexity: O(N) where N = len(keys)
        
        Uses batch LCP on GPU for efficiency (single kernel launch).
        
        Args:
            query: Query string
            
        Returns:
            np.ndarray: Similarity scores [N]
        """
        if not self._key_cache:
            return np.array([])
        
        n_keys = len(self._key_cache)
        
        if self.use_gpu and self._gpu_index_loaded:
            # Batch GPU path: single kernel for all keys
            query_encoded = encode_to_short2(query)
            lcps = _ctdr_python.dpx_lcp_index_query(query_encoded)
            
            # Vectorized similarity computation
            lcps = np.array(lcps, dtype=np.float32)
            distances = np.power(float(self.p), -lcps)
            similarities = 1.0 / (1.0 + distances)
            return similarities.astype(np.float32)
        else:
            # CPU path: vectorized where possible
            query_encoded = encode_to_short2(query)
            similarities = np.zeros(n_keys, dtype=np.float32)
            
            for i, (key_str, key_encoded) in enumerate(self._key_cache):
                lcp = _compute_lcp_cpu(query, key_str)
                similarities[i] = p_adic_similarity(lcp, self.p)
            
            return similarities
    
    def attention(self, query: str, values: np.ndarray) -> np.ndarray:
        """
        Compute attention-weighted sum of values.
        
        attention_weights = softmax(similarities)
        output = sum(attention_weights * values)
        
        Args:
            query: Query string
            values: Value matrix [N, D] where N=num_keys, D=value_dim
            
        Returns:
            np.ndarray: Attention output [D]
        """
        similarities = self.compute_similarities(query)
        if len(similarities) == 0:
            return np.zeros(values.shape[1] if len(values.shape) > 1 else 1)
        
        # Scale similarities for softmax
        attention_weights = softmax(similarities * 10)  # Scale factor for sharper attention
        
        # Weighted sum
        if len(values.shape) == 1:
            return np.sum(attention_weights * values)
        else:
            return np.dot(attention_weights, values)
    
    def batch_attention(self, queries: List[str], values: np.ndarray) -> np.ndarray:
        """
        Batch attention for multiple queries.
        
        Complexity: O(Q * N) where Q=num_queries, N=num_keys
        Standard attention would be O(Q * N * D) for dot-product
        
        Args:
            queries: List of query strings [Q]
            values: Value matrix [N, D]
            
        Returns:
            np.ndarray: Attention outputs [Q, D]
        """
        outputs = []
        for query in queries:
            output = self.attention(query, values)
            outputs.append(output)
        return np.array(outputs)


def standard_attention(query_vec: np.ndarray, key_vecs: np.ndarray, value_vecs: np.ndarray) -> np.ndarray:
    """
    Standard dot-product attention for comparison.
    
    Complexity: O(N * D) for single query, O(N²) for N queries
    
    Args:
        query_vec: Query vector [D]
        key_vecs: Key matrix [N, D]
        value_vecs: Value matrix [N, D]
        
    Returns:
        np.ndarray: Attention output [D]
    """
    # Dot product: query @ keys^T -> [N]
    scores = np.dot(key_vecs, query_vec)
    
    # Softmax
    attention_weights = softmax(scores)
    
    # Weighted sum
    return np.dot(attention_weights, value_vecs)


# Convenience function
def p_adic_attention_gpu(query: str, keys: List[str], values: List[float], p: int = 2) -> float:
    """
    P-adic attention through LCP on GPU.
    
    Formula:
    - LCP(query, key_i) = length of common prefix
    - distance = p^(-LCP)
    - similarity = 1.0 / (1.0 + distance)
    - attention_weights = softmax(similarities)
    - result = sum(attention_weights * values)
    
    Complexity: O(N) instead of O(N²)
    
    Args:
        query: Query string
        keys: List of key strings
        values: List of values (floats)
        p: Prime base (default 2)
        
    Returns:
        float: Attention-weighted result
    """
    attn = PadicAttention(p=p, use_gpu=True)
    attn.set_keys(keys)
    values_arr = np.array(values, dtype=np.float32)
    return float(attn.attention(query, values_arr))

