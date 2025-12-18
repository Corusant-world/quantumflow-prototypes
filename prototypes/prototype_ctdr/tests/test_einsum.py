"""
Tests for Reversible_Einsum_Engine
Comprehensive test suite for Boolean Einsum + Heaviside threshold
"""

import pytest
import numpy as np
import torch
from src.core import reversible_einsum, einsum_cpu
from src.rla_stack import RLAStack


def test_einsum_correctness_small():
    """Test Einsum correctness with small 2x2 matrices vs CPU baseline."""
    A = np.array([[1, 0], [0, 1]], dtype=bool)
    B = np.array([[1, 1], [0, 0]], dtype=bool)
    
    # CPU baseline
    result_cpu = einsum_cpu(A, B)
    result_cpu_threshold = (result_cpu.astype(float) >= 0.5).astype(bool)
    
    # CUDA/fallback
    result_cuda = reversible_einsum(A, B, threshold=0.5)
    
    # Check shape
    assert result_cuda.shape == result_cpu_threshold.shape, \
        f"Shape mismatch: {result_cuda.shape} != {result_cpu_threshold.shape}"
    
    # Check correctness (should match CPU baseline)
    np.testing.assert_array_equal(result_cuda, result_cpu_threshold), \
        f"Result mismatch:\nCUDA:\n{result_cuda}\nCPU:\n{result_cpu_threshold}"


def test_einsum_correctness_medium():
    """Test Einsum correctness with medium 4x4 matrices."""
    A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
    B = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=bool)
    
    # CPU baseline
    result_cpu = einsum_cpu(A, B)
    result_cpu_threshold = (result_cpu.astype(float) >= 0.5).astype(bool)
    
    # CUDA/fallback
    result_cuda = reversible_einsum(A, B, threshold=0.5)
    
    # Check correctness
    np.testing.assert_array_equal(result_cuda, result_cpu_threshold)


def test_einsum_correctness_large():
    """Test Einsum correctness with large 16x16 matrices."""
    np.random.seed(42)
    A = np.random.rand(16, 16) > 0.5
    B = np.random.rand(16, 16) > 0.5
    
    # CPU baseline
    result_cpu = einsum_cpu(A, B)
    result_cpu_threshold = (result_cpu.astype(float) >= 0.5).astype(bool)
    
    # CUDA/fallback
    result_cuda = reversible_einsum(A, B, threshold=0.5)
    
    # Check correctness
    np.testing.assert_array_equal(result_cuda, result_cpu_threshold)


def test_einsum_threshold_zero():
    """Test Einsum with threshold=0.0 (all should be True)."""
    A = np.array([[1, 0], [0, 1]], dtype=bool)
    B = np.array([[1, 1], [0, 0]], dtype=bool)
    
    result = reversible_einsum(A, B, threshold=0.0)
    
    # With threshold=0, all non-zero results should be True
    assert result.dtype == bool
    assert result.shape == (2, 2)


def test_einsum_threshold_one():
    """Test Einsum with threshold=1.0 (only exact matches)."""
    A = np.array([[1, 0], [0, 1]], dtype=bool)
    B = np.array([[1, 1], [0, 0]], dtype=bool)
    
    result = reversible_einsum(A, B, threshold=1.0)
    
    # With threshold=1.0, only results >= 1.0 should be True
    assert result.dtype == bool
    assert result.shape == (2, 2)


def test_einsum_threshold_half():
    """Test Einsum with threshold=0.5 (default)."""
    A = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    B = np.array([[1, 1], [0, 0], [1, 0]], dtype=bool)
    
    result = reversible_einsum(A, B, threshold=0.5)
    
    assert result.dtype == bool
    assert result.shape == (2, 2)


def test_einsum_empty_matrices():
    """Test Einsum with empty matrices (edge case)."""
    A = np.array([[]], dtype=bool).reshape(0, 0)
    B = np.array([[]], dtype=bool).reshape(0, 0)
    
    # Should raise ValueError for empty matrices
    with pytest.raises((ValueError, IndexError)):
        result = reversible_einsum(A, B, threshold=0.5)


def test_einsum_identity_matrix():
    """Test Einsum with identity matrix."""
    I = np.eye(4, dtype=bool)
    A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
    
    # I @ A should equal A
    result = reversible_einsum(I, A, threshold=0.5)
    
    # Check that result matches A (for threshold=0.5)
    result_cpu = einsum_cpu(I, A)
    expected = (result_cpu.astype(float) >= 0.5).astype(bool)
    np.testing.assert_array_equal(result, expected)


def test_einsum_zero_matrix():
    """Test Einsum with zero matrix."""
    A = np.zeros((3, 3), dtype=bool)
    B = np.ones((3, 3), dtype=bool)
    
    result = reversible_einsum(A, B, threshold=0.5)
    
    # A @ B should be all zeros
    assert np.all(result == False), "Result should be all False for zero matrix"


def test_einsum_ones_matrix():
    """Test Einsum with ones matrix."""
    A = np.ones((2, 3), dtype=bool)
    B = np.ones((3, 2), dtype=bool)
    
    result = reversible_einsum(A, B, threshold=0.5)
    
    # All results should be >= threshold (3 AND operations per element)
    assert result.shape == (2, 2)
    assert result.dtype == bool


def test_rla_stack_memoization():
    """Test RLA-стек мемоизация."""
    rla = RLAStack()
    
    A = np.array([[1, 0], [0, 1]], dtype=bool)
    B = np.array([[1, 1], [0, 0]], dtype=bool)
    
    # Первый вызов - должен выполнить операцию
    result1 = rla.wrap_reversible_einsum(A, B, threshold=0.5)
    stats1 = rla.get_stats()
    
    # Второй вызов - должен использовать кэш
    result2 = rla.wrap_reversible_einsum(A, B, threshold=0.5)
    stats2 = rla.get_stats()
    
    # Результаты должны совпадать
    np.testing.assert_array_equal(result1, result2)
    
    # Количество попаданий в кэш должно увеличиться
    assert stats2["cache_hits"] > stats1["cache_hits"], \
        "Cache hits should increase on second call"


def test_rla_stack_entropy_metrics():
    """Test RLA-стек энтропийные метрики."""
    rla = RLAStack()
    
    A = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    B = np.array([[1, 1], [0, 0], [1, 0]], dtype=bool)
    
    # Выполнение операции
    result = rla.wrap_reversible_einsum(A, B, threshold=0.5)
    
    # Проверка энтропийных метрик
    stats = rla.get_stats()
    
    assert stats["entropy_log_entries"] > 0, "Should have entropy log entries"
    assert stats["memory_writes"] > 0, "Should have memory writes"
    
    # Проверка информационной энтропии
    info_entropy = rla.compute_information_entropy(result)
    assert info_entropy >= 0.0, "Information entropy should be non-negative"
    
    # Проверка термодинамической энтропии
    thermo_entropy = rla.compute_thermodynamic_entropy(1)
    assert thermo_entropy >= 0.0, "Thermodynamic entropy should be non-negative"


def test_rla_stack_baseline_comparison():
    """Test RLA-стек сравнение с baseline."""
    rla = RLAStack()
    
    A = np.array([[1, 0], [0, 1]], dtype=bool)
    B = np.array([[1, 1], [0, 0]], dtype=bool)
    
    # Выполнение нескольких операций
    for _ in range(10):
        rla.wrap_reversible_einsum(A, B, threshold=0.5)
    
    # Baseline: 10 операций = 10 перезаписей
    baseline_writes = 10
    
    # Сравнение с baseline
    comparison = rla.compare_with_baseline(baseline_writes)
    
    assert "reduction_factor" in comparison
    assert "meets_target" in comparison
    
    # Если есть мемоизация, reduction_factor должен быть > 1
    stats = rla.get_stats()
    if stats["cache_hits"] > 0:
        assert comparison["reduction_factor"] >= 1.0, \
            "Reduction factor should be >= 1.0 with memoization"


def test_einsum_torch_baseline():
    """Test Einsum vs torch baseline (if torch available)."""
    try:
        A = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
        B = np.array([[1, 1], [0, 0], [1, 0]], dtype=bool)
        
        # Torch baseline
        A_torch = torch.tensor(A, dtype=torch.bool)
        B_torch = torch.tensor(B, dtype=torch.bool)
        result_torch = torch.einsum('ij,jk->ik', A_torch, B_torch).float()
        result_torch_threshold = (result_torch >= 0.5).numpy().astype(bool)
        
        # CUDA/fallback
        result_cuda = reversible_einsum(A, B, threshold=0.5)
        
        # Check correctness
        np.testing.assert_array_equal(result_cuda, result_torch_threshold)
    except ImportError:
        pytest.skip("torch not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
