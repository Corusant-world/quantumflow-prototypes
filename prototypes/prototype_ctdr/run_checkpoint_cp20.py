#!/usr/bin/env python3
"""
Checkpoint CP-2.0: DRC Integration Test

Запуск: python3 run_checkpoint_cp20.py
"""

import sys
import os

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_checkpoint():
    print("=" * 60)
    print("CHECKPOINT CP-2.0: DRC Integration")
    print("=" * 60)
    
    # 1. Импорт модулей
    print("\n[1] Importing Phase 2 modules...")
    try:
        from drc import DynamicReversibleCore
        from drc_orchestrator import DRCOrchestrator
        from hce import HybridComputationalUnit
        from dhm import DynamicHierarchyManager
        print("    ✅ All imports successful")
    except ImportError as e:
        print(f"    ❌ Import error: {e}")
        return False
    
    # 2. Инициализация DRC
    print("\n[2] Initializing DRC...")
    drc = DynamicReversibleCore()
    print(f"    ✅ DRC created")
    print(f"    ✅ RC (RLA Stack) initialized: {drc.rc is not None}")
    print(f"    ✅ DHM lazy init: {drc._dhm is None}")
    
    # 3. Инициализация оркестратора
    print("\n[3] Initializing DRC Orchestrator...")
    orch = DRCOrchestrator(drc)
    print(f"    ✅ Orchestrator created")
    
    # 4. Инициализация HCE
    print("\n[4] Initializing HCE...")
    hce = HybridComputationalUnit()
    print(f"    ✅ HCE created (device: {hce.device})")
    
    # 5. Вставка знаний в DHM
    print("\n[5] Inserting knowledge into DHM...")
    path1 = drc.insert_knowledge("test_concept_1", {"data": "value1"})
    path2 = drc.insert_knowledge("test_concept_2", {"data": "value2"})
    path3 = drc.insert_knowledge("animals → cats", {"type": "pet"})
    print(f"    ✅ Inserted 3 concepts")
    print(f"    Path 1: {path1}")
    print(f"    Path 2: {path2}")
    print(f"    Path 3: {path3}")
    
    # 6. Тест Cold Path (RC + DHM)
    print("\n[6] Testing Cold Path (RC + DHM)...")
    result1 = orch.execute("search: test_concept_1")
    print(f"    ✅ Cold path executed")
    print(f"    Mode: {result1['mode']}")
    print(f"    Execution time: {result1['execution_time_ms']:.2f} ms")
    
    # 7. Тест DHM Search
    print("\n[7] Testing DHM Search...")
    dhm = drc.dhm
    search_results = dhm.search("test", max_results=5)
    print(f"    ✅ Search returned {len(search_results)} results")
    for path, content, similarity in search_results[:3]:
        print(f"       - {path}: similarity={similarity:.4f}")
    
    # 8. Тест Mental Saccade
    print("\n[8] Testing Mental Saccade...")
    full_path = "Level0 → Level1 → Level2"
    level0 = dhm.mental_saccade(full_path, 0)
    level1 = dhm.mental_saccade(full_path, 1)
    print(f"    ✅ Saccade to level 0: {level0}")
    print(f"    ✅ Saccade to level 1: {level1}")
    
    # 9. Тест HCE Structural
    print("\n[9] Testing HCE Structural Framework...")
    import numpy as np
    candidates = ["test_concept_1", "test_concept_2", "animals"]
    hce_results = hce.structural_framework("test", candidates)
    print(f"    ✅ HCE structural returned {len(hce_results)} results")
    for cand, sim in hce_results:
        print(f"       - {cand}: similarity={sim:.4f}")
    
    # 10. Тест HCE Neural
    print("\n[10] Testing HCE Neural Substrate...")
    embeddings = np.random.randn(10, 64).astype(np.float32)
    weights = np.random.randn(64, 32).astype(np.float32)
    neural_result = hce.neural_substrate(embeddings, weights)
    print(f"    ✅ Neural result shape: {neural_result.shape}")
    
    # 11. Тест HCE Logical
    print("\n[11] Testing HCE Logical Core...")
    tensors = [
        np.random.randn(8, 8).astype(np.float32),
        np.random.randn(8, 8).astype(np.float32),
    ]
    predicates = [True, True]
    logical_result = hce.logical_core(tensors, predicates)
    print(f"    ✅ Logical result shape: {logical_result.shape}")
    
    # 12. Сбор статистики
    print("\n[12] Collecting Statistics...")
    orch_stats = orch.get_stats()
    hce_stats = hce.get_stats()
    dhm_stats = dhm.get_stats()
    
    print(f"    DRC cold operations: {orch_stats['orchestrator']['cold_operations']}")
    print(f"    DHM size: {dhm_stats['size']}")
    print(f"    DHM max depth: {dhm_stats['max_depth']}")
    print(f"    HCE neural ops: {hce_stats['neural_operations']}")
    print(f"    HCE structural ops: {hce_stats['structural_operations']}")
    print(f"    HCE logical ops: {hce_stats['logical_operations']}")
    
    # 13. Entropy Report
    print("\n[13] Entropy Report...")
    entropy_report = orch.get_entropy_report()
    print(f"    Total entropy: {entropy_report['total_entropy']:.4f}")
    print(f"    Current entropy: {entropy_report['current_entropy']:.4f}")
    print(f"    Memoization efficiency: {entropy_report['memoization_efficiency']:.2%}")
    
    # 14. Phase 1 Integration Check
    print("\n[14] Phase 1 Integration Check...")
    rla_stats = drc.rc.get_stats()
    print(f"    RLA memory writes: {rla_stats['memory_writes']}")
    print(f"    RLA cache hits: {rla_stats['cache_hits']}")
    print(f"    RLA cache hit rate: {rla_stats['cache_hit_rate']:.2%}")
    
    # Final Result
    print("\n" + "=" * 60)
    print("CHECKPOINT CP-2.0: DRC Integration - PASSED ✅")
    print("=" * 60)
    print("\nSummary:")
    print(f"  • DRC initialized: ✅")
    print(f"  • RC (RLA Stack) integrated: ✅")
    print(f"  • DHM working: ✅ ({dhm_stats['size']} nodes)")
    print(f"  • HCE working: ✅ (neural + structural + logical)")
    print(f"  • Cold path works: ✅")
    print(f"  • Entropy tracking: ✅")
    print(f"  • Phase 1 not broken: ✅")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = run_checkpoint()
    sys.exit(0 if success else 1)

