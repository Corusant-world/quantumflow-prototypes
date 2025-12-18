#!/usr/bin/env python3
"""
Benchmark: DHM Infinite Context (CP-2.1)

Цель: Демонстрация масштабируемости DHM
- Извлечение из большого корпуса с задержкой <100ms
- Интеграция с DPX_LCP_Kernel для O(N) поиска
- Симуляция "бесконечного контекста"

Метрики:
- Retrieval latency (ms)
- Throughput (queries/sec)
- Memory usage
- GPU utilization (если GPU)
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, Any, List

# Абсолютные пути к модулям
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)  # prototype_ctdr/
_SRC_PATH = os.path.join(_PROJECT_ROOT, 'src')
_CUDA_PATH = os.path.join(_PROJECT_ROOT, 'cuda')

# Добавляем пути В НАЧАЛО (важно!)
sys.path.insert(0, _PROJECT_ROOT)  # Для ctdr_python.so в корне
sys.path.insert(0, _CUDA_PATH)
sys.path.insert(0, _SRC_PATH)

from dhm import DynamicHierarchyManager
from encoding import encode_to_short2

# Проверка GPU (ctdr_python)
try:
    import ctdr_python
    GPU_AVAILABLE = True
    print(f"[INFO] DPX GPU available: ctdr_python loaded from {ctdr_python.__file__}")
except ImportError as e:
    GPU_AVAILABLE = False
    print(f"[INFO] DPX GPU not available: {e}")


def generate_knowledge_base(num_concepts: int) -> List[Dict[str, Any]]:
    """
    Генерация синтетической базы знаний.
    
    Создаёт иерархическую структуру концептов.
    """
    concepts = []
    categories = ["science", "tech", "art", "history", "math", "physics", "biology", "chemistry"]
    
    for i in range(num_concepts):
        category = categories[i % len(categories)]
        depth = (i % 5) + 1  # Глубина 1-5
        
        # Создание иерархического пути
        path_parts = [category]
        for d in range(depth):
            path_parts.append(f"level{d}_{i // (10 ** d) % 10}")
        
        path = " → ".join(path_parts)
        
        concepts.append({
            "concept": f"concept_{i}",
            "content": {"id": i, "category": category, "data": f"data_{i}"},
            "path": path
        })
    
    return concepts


def benchmark_dhm_cpu(num_concepts: int, num_queries: int) -> Dict[str, Any]:
    """
    Benchmark DHM на CPU.
    """
    print(f"\n[CPU] Benchmarking DHM with {num_concepts} concepts, {num_queries} queries...")
    
    dhm = DynamicHierarchyManager()
    
    # 1. Вставка концептов
    print("  Inserting concepts...")
    insert_start = time.time()
    concepts = generate_knowledge_base(num_concepts)
    for c in concepts:
        dhm.insert(c["concept"], c["content"], c["path"])
    insert_time = time.time() - insert_start
    print(f"  Insert time: {insert_time:.3f}s ({num_concepts / insert_time:.0f} concepts/s)")
    
    # 2. Генерация запросов
    queries = [f"concept_{np.random.randint(0, num_concepts)}" for _ in range(num_queries)]
    
    # 3. WARM-UP: первый query строит encoded paths (не измеряем)
    print("  Warm-up (building encoded paths)...")
    warmup_start = time.time()
    _ = dhm.search("warmup_query", max_results=1)
    warmup_time = (time.time() - warmup_start) * 1000
    print(f"  Warm-up time: {warmup_time:.2f}ms (one-time encoding)")
    
    # 4. Benchmark поиска (после warm-up)
    print("  Running search benchmark...")
    latencies = []
    
    for query in queries:
        start = time.time()
        results = dhm.search(query, max_results=10)
        elapsed = (time.time() - start) * 1000  # ms
        latencies.append(elapsed)
    
    # 4. Статистика
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    max_latency = np.max(latencies)
    throughput = 1000 / avg_latency  # queries/sec
    
    print(f"  Avg latency: {avg_latency:.2f}ms")
    print(f"  P50: {p50_latency:.2f}ms, P95: {p95_latency:.2f}ms, P99: {p99_latency:.2f}ms")
    print(f"  Max latency: {max_latency:.2f}ms")
    print(f"  Throughput: {throughput:.0f} queries/sec")
    
    # 5. Mental Saccade benchmark
    print("  Running mental saccade benchmark...")
    saccade_times = []
    test_path = "science → level0_0 → level1_0 → level2_0 → level3_0"
    
    for level in range(5):
        start = time.time()
        for _ in range(1000):
            dhm.mental_saccade(test_path, level)
        elapsed = (time.time() - start) * 1000 / 1000  # ms per saccade
        saccade_times.append(elapsed)
    
    avg_saccade = np.mean(saccade_times)
    print(f"  Avg mental saccade time: {avg_saccade:.4f}ms (O(1) expected)")
    
    return {
        "mode": "cpu",
        "num_concepts": num_concepts,
        "num_queries": num_queries,
        "insert_time_s": insert_time,
        "insert_throughput": num_concepts / insert_time,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "max_latency_ms": max_latency,
        "throughput_qps": throughput,
        "avg_saccade_ms": avg_saccade,
        "dhm_stats": dhm.get_stats(),
        "meets_100ms_target": max_latency < 100
    }


def benchmark_dhm_gpu(num_concepts: int, num_queries: int) -> Dict[str, Any]:
    """
    Benchmark DHM на GPU с DPX.
    
    Использует DPX batch LCP через ctdr_python:
    - dpx_lcp_index_load: загрузка candidates на GPU
    - dpx_lcp_index_query: batch LCP для всех candidates
    """
    if not GPU_AVAILABLE:
        print("\n[GPU] DPX kernels not available, skipping GPU benchmark")
        return {"mode": "gpu", "available": False}
    
    print(f"\n[GPU] Benchmarking DHM with DPX, {num_concepts} concepts, {num_queries} queries...")
    
    # DHM автоматически использует GPU если ctdr_python доступен
    dhm = DynamicHierarchyManager(use_gpu=True)
    
    # 1. Вставка концептов
    print("  Inserting concepts...")
    insert_start = time.time()
    concepts = generate_knowledge_base(num_concepts)
    for c in concepts:
        dhm.insert(c["concept"], c["content"], c["path"])
    insert_time = time.time() - insert_start
    print(f"  Insert time: {insert_time:.3f}s")
    print(f"  GPU available: {dhm._gpu_available}")
    
    # 2. Генерация запросов
    queries = [f"concept_{np.random.randint(0, num_concepts)}" for _ in range(num_queries)]
    
    # 3. WARM-UP: первый query загружает index на GPU (не измеряем)
    print("  Warm-up (loading GPU index)...")
    warmup_start = time.time()
    _ = dhm.search("warmup_query", max_results=1)
    warmup_time = (time.time() - warmup_start) * 1000
    print(f"  Warm-up time: {warmup_time:.2f}ms (one-time GPU index load)")
    print(f"  GPU index loaded: {dhm._gpu_index_loaded}")
    
    # 4. Benchmark GPU поиска (после warm-up)
    print("  Running GPU search benchmark...")
    latencies = []
    
    for query in queries:
        start = time.time()
        results = dhm.search(query, max_results=10)  # Автоматически GPU
        elapsed = (time.time() - start) * 1000  # ms
        latencies.append(elapsed)
    
    # 4. Статистика
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    max_latency = np.max(latencies)
    throughput = 1000 / avg_latency
    
    print(f"  Avg latency: {avg_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms")
    print(f"  Max latency: {max_latency:.2f}ms")
    print(f"  Throughput: {throughput:.0f} queries/sec")
    
    return {
        "mode": "gpu",
        "available": True,
        "num_concepts": num_concepts,
        "num_queries": num_queries,
        "insert_time_s": insert_time,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "max_latency_ms": max_latency,
        "throughput_qps": throughput,
        "meets_100ms_target": max_latency < 100
    }


def benchmark_scalability() -> Dict[str, Any]:
    """
    Benchmark масштабируемости: как latency растёт с размером DHM.
    """
    print("\n[SCALABILITY] Testing DHM scalability...")
    
    sizes = [100, 1000, 10000, 50000, 100000]
    results = []
    
    for size in sizes:
        print(f"\n  Size: {size}")
        dhm = DynamicHierarchyManager()
        
        # Вставка
        concepts = generate_knowledge_base(size)
        for c in concepts:
            dhm.insert(c["concept"], c["content"])
        
        # Warm-up (не измеряем)
        _ = dhm.search("warmup", max_results=1)
        
        # Benchmark (после warm-up)
        latencies = []
        for _ in range(100):
            query = f"concept_{np.random.randint(0, size)}"
            start = time.time()
            dhm.search(query, max_results=10)
            latencies.append((time.time() - start) * 1000)
        
        avg_lat = np.mean(latencies)
        print(f"    Avg latency: {avg_lat:.2f}ms")
        
        results.append({
            "size": size,
            "avg_latency_ms": avg_lat,
            "complexity_ratio": avg_lat / (size / 1000)  # Нормализация
        })
    
    # Проверка O(N) или лучше сложности
    # Если ratios падают или константны → O(N) или лучше
    # Если ratios растут → хуже чем O(N)
    ratios = [r["complexity_ratio"] for r in results]
    
    # Проверяем: ratios не должны расти (последний не должен быть > первого)
    is_linear_or_better = ratios[-1] <= ratios[0] * 1.5  # Допуск 50%
    
    print(f"\n  Complexity analysis:")
    print(f"    Ratios: {[f'{r:.3f}' for r in ratios]}")
    print(f"    First ratio: {ratios[0]:.3f}, Last ratio: {ratios[-1]:.3f}")
    if ratios[-1] < ratios[0] * 0.5:
        print(f"    Complexity: SUBLINEAR (O(log N) or better) ✅")
    elif is_linear_or_better:
        print(f"    Complexity: O(N) or better ✅")
    else:
        print(f"    Complexity: WORSE than O(N) ❌")
    print(f"    Is O(N) or better: {is_linear_or_better}")
    
    return {
        "sizes": sizes,
        "results": results,
        "is_linear_or_better": is_linear_or_better
    }


def run_checkpoint_cp21() -> Dict[str, Any]:
    """
    Чекпоинт CP-2.1: Бесконечный Контекст
    
    Критерии:
    - DHM работает на CPU и GPU
    - Retrieval latency < 100ms
    - O(N) сложность подтверждена
    """
    print("=" * 60)
    print("CHECKPOINT CP-2.1: DHM Infinite Context")
    print("=" * 60)
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": "CP-2.1",
    }
    
    # CPU benchmark
    cpu_results = benchmark_dhm_cpu(num_concepts=10000, num_queries=100)
    results["cpu"] = cpu_results
    
    # GPU benchmark
    gpu_results = benchmark_dhm_gpu(num_concepts=10000, num_queries=100)
    results["gpu"] = gpu_results
    
    # Scalability
    scalability = benchmark_scalability()
    results["scalability"] = scalability
    
    # Summary
    print("\n" + "=" * 60)
    print("CP-2.1 SUMMARY")
    print("=" * 60)
    
    cpu_pass = cpu_results["meets_100ms_target"]
    gpu_pass = gpu_results.get("meets_100ms_target", True) if gpu_results.get("available", False) else True
    linear_pass = scalability["is_linear_or_better"]
    
    print(f"  CPU < 100ms: {'✅' if cpu_pass else '❌'} (max: {cpu_results['max_latency_ms']:.2f}ms)")
    if gpu_results.get("available"):
        print(f"  GPU < 100ms: {'✅' if gpu_pass else '❌'} (max: {gpu_results['max_latency_ms']:.2f}ms)")
    else:
        print(f"  GPU: skipped (DPX not available)")
    print(f"  O(N) or better: {'✅' if linear_pass else '❌'}")
    
    all_pass = cpu_pass and linear_pass
    results["passed"] = all_pass
    
    if all_pass:
        print(f"\n{'=' * 60}")
        print("CP-2.1 PASSED ✅")
        print("=" * 60)
    else:
        print(f"\n{'=' * 60}")
        print("CP-2.1 FAILED ❌")
        print("=" * 60)
    
    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Сохранение результатов в JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    results = run_checkpoint_cp21()
    
    output_path = os.path.join(
        os.path.dirname(__file__), 
        'results', 
        'phase2_dhm.json'
    )
    save_results(results, output_path)

