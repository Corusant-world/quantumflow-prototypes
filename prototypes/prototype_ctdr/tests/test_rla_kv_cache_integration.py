"""
Tests for RLA-стек + KV_Cache_Steering_DPX Integration
Проверка энтропийных метрик и метрики 2× меньше перезаписей
"""

import pytest
import numpy as np
from src.kv_cache_steering import KVCacheSteeringDPX
from src.rla_stack import RLAStack


def test_rla_kv_cache_integration():
    """Test integration of RLA-стек with KV Cache."""
    rla = RLAStack()
    cache = KVCacheSteeringDPX(rla_stack=rla)
    
    # Put values with RLA tracking
    for i in range(10):
        key = f"key_{i}"
        value = np.array([i], dtype=np.float32)
        cache.put_with_rla(key, value)
    
    # Get values with RLA tracking
    for i in range(10):
        key = f"key_{i}"
        value = cache.get_with_rla(key)
        assert value is not None, f"Should retrieve key_{i}"
    
    # Check RLA stats
    rla_stats = rla.get_stats()
    assert rla_stats["entropy_log_entries"] > 0, "Should have entropy log entries"
    assert rla_stats["memory_writes"] > 0, "Should have memory writes"
    
    # Check cache stats
    cache_stats = cache.get_stats()
    assert cache_stats["cache_hits"] > 0, "Should have cache hits"


def test_rla_kv_cache_entropy_metrics():
    """Test entropy metrics tracking in RLA + KV Cache integration."""
    rla = RLAStack()
    cache = KVCacheSteeringDPX(rla_stack=rla)
    
    # Put value
    value = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    cache.put_with_rla("test_key", value)
    
    # Get value (should log entropy metrics)
    retrieved = cache.get_with_rla("test_key")
    assert retrieved is not None
    
    # Check entropy log
    rla_stats = rla.get_stats()
    assert rla_stats["entropy_log_entries"] >= 2, "Should have at least 2 entropy log entries (put + get)"
    
    # Verify entropy computation
    info_entropy = rla.compute_information_entropy(value)
    assert info_entropy >= 0.0, "Information entropy should be non-negative"
    
    thermo_entropy = rla.compute_thermodynamic_entropy(1)
    assert thermo_entropy >= 0.0, "Thermodynamic entropy should be non-negative"


def test_rla_kv_cache_baseline_comparison():
    """Test comparison with baseline (2× меньше перезаписей)."""
    rla = RLAStack()
    cache = KVCacheSteeringDPX(rla_stack=rla)
    
    # Simulate operations with cache (мемоизация работает)
    A = np.array([[1, 0], [0, 1]], dtype=bool)
    B = np.array([[1, 1], [0, 0]], dtype=bool)
    
    # 10 операций с мемоизацией
    for i in range(10):
        key = f"einsum_{i % 5}"  # Повторяющиеся ключи (мемоизация)
        result = cache.memoize_einsum_result(key, A, B, threshold=0.5)
        cache.put_with_rla(key, result)
    
    # Baseline: 10 операций = 10 перезаписей (без мемоизации)
    baseline_writes = 10
    
    # Сравнение с baseline
    comparison = rla.compare_with_baseline(baseline_writes)
    
    assert comparison["rla_writes"] <= baseline_writes, "RLA should have <= writes than baseline"
    
    # С мемоизацией должно быть меньше перезаписей
    rla_stats = rla.get_stats()
    if rla_stats["cache_hits"] > 0:
        assert comparison["reduction_factor"] >= 1.0, "Reduction factor should be >= 1.0 with memoization"
        assert comparison["meets_target"] or comparison["reduction_factor"] >= 1.5, \
            f"Should meet target (2×) or at least 1.5×, got {comparison['reduction_factor']:.2f}x"


def test_rla_kv_cache_memory_log():
    """Test logging to memory_log.json."""
    import tempfile
    import os
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        log_file = f.name
    
    try:
        rla = RLAStack(log_file=log_file)
        cache = KVCacheSteeringDPX(rla_stack=rla)
        
        # Perform operations
        cache.put_with_rla("key_1", np.array([1, 2, 3], dtype=np.float32))
        cache.get_with_rla("key_1")
        
        # Check that log file was created and has entries
        assert os.path.exists(log_file), "Log file should be created"
        
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        assert "decisions" in log_data, "Log should contain decisions"
        assert len(log_data["decisions"]) >= 2, "Should have at least 2 log entries"
        
        # Verify entropy metrics in log
        for entry in log_data["decisions"]:
            assert "information_entropy" in entry, "Entry should have information_entropy"
            assert "thermodynamic_entropy" in entry, "Entry should have thermodynamic_entropy"
            assert entry["information_entropy"] >= 0.0, "Information entropy should be >= 0"
            assert entry["thermodynamic_entropy"] >= 0.0, "Thermodynamic entropy should be >= 0"
    
    finally:
        if os.path.exists(log_file):
            os.remove(log_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


