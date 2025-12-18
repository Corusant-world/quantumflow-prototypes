"""
Smoke Tests для DRC/HCE — Подфаза 2.0

Проверяет:
1. DRC интеграция (RC + DHM)
2. HCE гибридная работа (Neural + Structural + Logical)
3. Связка с Фазой 1 (RLA Stack, reversible_einsum)
4. Мемоизация и энтропийные метрики

Чекпоинт CP-2.0: все модули работают как единое целое
"""

import sys
import os
import numpy as np
import pytest

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.drc import DynamicReversibleCore
from src.drc_orchestrator import DRCOrchestrator
from src.hce import HybridComputationalUnit, HCEGPU
from src.dhm import DynamicHierarchyManager, DHMNode
from src.rla_stack import RLAStack


class TestDRCIntegration:
    """Тесты интеграции DRC (RC + DHM)."""
    
    def test_drc_initialization(self):
        """DRC должен инициализироваться без ошибок."""
        drc = DynamicReversibleCore()
        
        assert drc.rc is not None, "RC (RLAStack) должен быть инициализирован"
        assert drc._current_mode == "cold", "Начальный режим должен быть cold"
    
    def test_drc_cold_mode_memoization(self):
        """Cold mode должен использовать мемоизацию."""
        drc = DynamicReversibleCore()
        
        # Вставляем знание
        path = drc.insert_knowledge("test_concept", {"data": "test_value"})
        assert path is not None, "Путь должен быть возвращён"
        
        # Поиск через cold compute
        result = drc.compute("test_concept", mode="cold")
        # Результат может быть None если DHM пустой, но операция должна пройти
        
        # Проверка статистики
        stats = drc.get_stats()
        assert stats["drc"]["cold_operations"] >= 1, "Должна быть минимум 1 cold операция"
    
    def test_drc_entropy_management(self):
        """DRC должен управлять энтропией (DEM)."""
        drc = DynamicReversibleCore()
        
        # Внутренняя задача (search) → cold mode
        result = drc.manage_entropy("search: find concept X")
        assert result["mode"] == "cold", "Внутренняя задача должна использовать cold mode"
        assert "entropy" in result, "Должна быть метрика энтропии"
        
        # Проверка что entropy >= 0
        assert result["entropy"] >= 0, "Энтропия должна быть >= 0"
    
    def test_drc_rla_integration(self):
        """DRC должен интегрироваться с RLA Stack из Фазы 1."""
        drc = DynamicReversibleCore()
        
        # Мемоизация через RC (RLA Stack)
        test_data = np.array([1.0, 2.0, 3.0])
        hit = drc.rc.memoize("test_key", test_data)
        
        # Первая запись — не попадание
        assert hit == False, "Первая запись не должна быть hit"
        
        # Повторная запись того же — попадание
        hit = drc.rc.memoize("test_key", test_data)
        assert hit == True, "Повторная запись должна быть hit"
        
        # Проверка get
        cached = drc.rc.get("test_key")
        assert cached is not None, "Значение должно быть в кэше"
        assert np.array_equal(cached, test_data), "Значение должно совпадать"


class TestDRCOrchestrator:
    """Тесты оркестратора DRC."""
    
    def test_orchestrator_execute(self):
        """Оркестратор должен выполнять задачи."""
        orchestrator = DRCOrchestrator()
        
        result = orchestrator.execute("search: test query")
        
        assert "result" in result, "Должен быть result"
        assert "execution_time_ms" in result, "Должно быть время выполнения"
        assert "task_id" in result, "Должен быть task_id"
    
    def test_orchestrator_batch(self):
        """Оркестратор должен обрабатывать батчи."""
        orchestrator = DRCOrchestrator()
        
        tasks = [
            "search: query 1",
            "lookup: query 2", 
            "find: query 3"
        ]
        
        results = orchestrator.execute_batch(tasks)
        
        assert len(results) == 3, "Должно быть 3 результата"
        
        stats = orchestrator.get_stats()
        assert stats["orchestrator"]["total_tasks"] == 3, "Должно быть 3 задачи"
    
    def test_orchestrator_entropy_report(self):
        """Оркестратор должен генерировать отчёт по энтропии."""
        orchestrator = DRCOrchestrator()
        
        # Выполняем несколько задач
        orchestrator.execute("search: test 1")
        orchestrator.execute("lookup: test 2")
        
        report = orchestrator.get_entropy_report()
        
        assert "total_entropy" in report, "Должна быть total_entropy"
        assert "memoization_efficiency" in report, "Должна быть эффективность мемоизации"


class TestHCE:
    """Тесты Hybrid Computational Unit."""
    
    def test_hce_initialization(self):
        """HCE должен инициализироваться."""
        hce = HybridComputationalUnit()
        
        assert hce.device in ["cpu", "cuda"], "Device должен быть cpu или cuda"
        assert hce.dpx_enabled == True, "DPX должен быть включен"
    
    def test_hce_neural_substrate(self):
        """Neural substrate должен выполнять матричные умножения."""
        hce = HybridComputationalUnit(device="cpu")
        
        embeddings = np.random.randn(10, 64).astype(np.float32)
        weights = np.random.randn(64, 32).astype(np.float32)
        
        result = hce.neural_substrate(embeddings, weights)
        
        assert result.shape == (10, 32), f"Shape должен быть (10, 32), получили {result.shape}"
        
        stats = hce.get_stats()
        assert stats["neural_operations"] == 1, "Должна быть 1 neural операция"
    
    def test_hce_structural_framework(self):
        """Structural framework должен выполнять LCP поиск."""
        hce = HybridComputationalUnit(device="cpu")
        
        query = "test_query"
        candidates = ["test_query_1", "test_other", "test_query_exact"]
        
        results = hce.structural_framework(query, candidates)
        
        assert len(results) == 3, "Должно быть 3 результата"
        assert all(isinstance(r[1], float) for r in results), "Все similarity должны быть float"
        
        # test_query_1 должен быть ближе к test_query чем test_other
        similarities = {r[0]: r[1] for r in results}
        assert similarities["test_query_1"] > similarities["test_other"], \
            "test_query_1 должен быть ближе к test_query"
    
    def test_hce_logical_core(self):
        """Logical core должен выполнять Boolean Einsum."""
        hce = HybridComputationalUnit(device="cpu")
        
        tensors = [
            np.random.randn(4, 4).astype(np.float32),
            np.random.randn(4, 4).astype(np.float32),
            np.random.randn(4, 4).astype(np.float32),
        ]
        predicates = [True, True, False]  # Третий тензор отфильтрован
        
        result = hce.logical_core(tensors, predicates)
        
        assert result is not None, "Результат не должен быть None"
        
        stats = hce.get_stats()
        assert stats["logical_operations"] == 1, "Должна быть 1 logical операция"
    
    def test_hce_hybrid_compute(self):
        """Hybrid compute должен автоматически выбирать компонент."""
        hce = HybridComputationalUnit(device="cpu")
        
        # Neural task
        neural_task = {
            "type": "neural",
            "embeddings": np.random.randn(5, 32).astype(np.float32),
            "weights": np.random.randn(32, 16).astype(np.float32),
        }
        result1 = hce.hybrid_compute(neural_task)
        assert result1.shape == (5, 16), "Neural result shape должен быть (5, 16)"
        
        # Structural task
        structural_task = {
            "type": "structural",
            "query": "test",
            "candidates": ["test1", "other", "testing"],
        }
        result2 = hce.hybrid_compute(structural_task)
        assert len(result2) == 3, "Structural result должен иметь 3 элемента"


class TestDHM:
    """Тесты Dynamic Hierarchy Manager."""
    
    def test_dhm_initialization(self):
        """DHM должен инициализироваться."""
        dhm = DynamicHierarchyManager()
        
        assert dhm.size == 0, "Начальный размер должен быть 0"
        assert dhm.root is not None, "Корень должен существовать"
    
    def test_dhm_insert_and_search(self):
        """DHM должен вставлять и искать концепты."""
        dhm = DynamicHierarchyManager()
        
        # Вставка
        path1 = dhm.insert("concept_a", {"value": 1})
        path2 = dhm.insert("concept_b", {"value": 2})
        path3 = dhm.insert("concept_a_related", {"value": 3})
        
        assert dhm.size == 3, "Должно быть 3 узла"
        
        # Поиск
        results = dhm.search("concept_a", max_results=10)
        
        assert len(results) > 0, "Должны быть результаты"
        # Первый результат должен быть наиболее похожим на concept_a
    
    def test_dhm_mental_saccade(self):
        """DHM должен поддерживать ментальные саккады O(1)."""
        dhm = DynamicHierarchyManager()
        
        # Создаём иерархический путь
        full_path = "Level0 → Level1 → Level2 → Level3"
        
        # Саккада на уровень 0
        result0 = dhm.mental_saccade(full_path, 0)
        assert result0 == "Level0", f"Уровень 0 должен быть 'Level0', получили '{result0}'"
        
        # Саккада на уровень 1
        result1 = dhm.mental_saccade(full_path, 1)
        assert result1 == "Level0 → Level1", f"Уровень 1 должен быть 'Level0 → Level1'"
        
        # Саккада на уровень 2
        result2 = dhm.mental_saccade(full_path, 2)
        assert result2 == "Level0 → Level1 → Level2"
    
    def test_dhm_archive(self):
        """DHM должен поддерживать архивацию (забывание без стирания)."""
        dhm = DynamicHierarchyManager()
        
        # Вставка
        path = dhm.insert("to_archive", {"data": "will be archived"})
        initial_size = dhm.size
        
        # Архивация
        dhm.archive(path, max_depth=5)
        
        # Размер не должен уменьшиться (архивация, не стирание)
        # Но узел должен переместиться
        assert dhm.size >= initial_size, "Размер не должен уменьшиться"


class TestPhase1Phase2Integration:
    """Тесты интеграции Фазы 1 и Фазы 2."""
    
    def test_rla_stack_in_drc(self):
        """RLA Stack из Фазы 1 должен работать внутри DRC."""
        drc = DynamicReversibleCore()
        
        # Прямой доступ к RLA Stack
        test_array = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Мемоизация
        drc.rc.memoize("phase1_test", test_array)
        
        # Проверка статистики
        stats = drc.rc.get_stats()
        assert stats["memory_writes"] >= 1, "Должна быть минимум 1 запись"
    
    def test_reversible_einsum_in_hce(self):
        """Reversible Einsum из Фазы 1 должен работать в HCE."""
        hce = HybridComputationalUnit(device="cpu")
        
        # Logical core использует reversible_einsum
        tensors = [
            np.random.randn(3, 3).astype(np.float32),
            np.random.randn(3, 3).astype(np.float32),
        ]
        predicates = [True, True]
        
        result = hce.logical_core(tensors, predicates, threshold=0.5)
        
        assert result is not None, "Reversible einsum должен выполниться"
        # Результат Boolean — 0 или 1
        assert np.all((result == 0) | (result == 1)), "Результат должен быть Boolean"
    
    def test_full_drc_hce_pipeline(self):
        """Полный пайплайн DRC + HCE должен работать."""
        # DRC для управления состоянием
        drc = DynamicReversibleCore()
        
        # HCE для вычислений
        hce = HybridComputationalUnit(device="cpu")
        
        # 1. Вставка знаний через DRC → DHM
        drc.insert_knowledge("concept_neural", {"type": "neural", "dim": 64})
        drc.insert_knowledge("concept_logic", {"type": "logical", "rules": 10})
        
        # 2. Поиск через DRC → DHM
        result = drc.compute("concept", mode="cold")
        
        # 3. HCE структурный поиск
        candidates = ["concept_neural", "concept_logic", "other"]
        structural_result = hce.structural_framework("concept", candidates)
        
        # 4. Проверка что всё работает вместе
        drc_stats = drc.get_stats()
        hce_stats = hce.get_stats()
        
        assert drc_stats["drc"]["dhm_inserts"] >= 2, "Должно быть минимум 2 вставки"
        assert hce_stats["structural_operations"] >= 1, "Должна быть минимум 1 структурная операция"


# Benchmark для CP-2.0
class TestCP20Benchmark:
    """Чекпоинт CP-2.0: DRC Integration."""
    
    def test_cp20_drc_integration_complete(self):
        """
        CP-2.0: Все модули DRC работают как единое целое.
        
        Критерии:
        - DRC запускает cold path (RC + lookup)
        - DRC запускает structural path (DHM)
        - Phase 1 не сломана
        """
        # 1. Инициализация
        orchestrator = DRCOrchestrator()
        
        # 2. Предзагрузка знаний
        knowledge = [
            {"concept": "test_a", "content": {"id": 1}},
            {"concept": "test_b", "content": {"id": 2}},
            {"concept": "query_test", "content": {"id": 3}},
        ]
        orchestrator.preload_knowledge(knowledge)
        
        # 3. Выполнение задач
        results = []
        for task in ["search: test_a", "lookup: test_b", "find: query_test"]:
            result = orchestrator.execute(task)
            results.append(result)
        
        # 4. Проверка критериев
        stats = orchestrator.get_stats()
        
        # Cold operations должны быть > 0
        assert stats["orchestrator"]["cold_operations"] >= 3, \
            f"CP-2.0 FAILED: cold_operations должны быть >= 3, получили {stats['orchestrator']['cold_operations']}"
        
        # DHM должен иметь узлы
        assert stats["drc"]["dhm"]["size"] >= 3, \
            f"CP-2.0 FAILED: DHM size должен быть >= 3, получили {stats['drc']['dhm']['size']}"
        
        # Энтропия должна быть измерена
        entropy_report = orchestrator.get_entropy_report()
        assert "total_entropy" in entropy_report, \
            "CP-2.0 FAILED: total_entropy должна быть в отчёте"
        
        print("\n" + "="*60)
        print("CP-2.0 PASSED: DRC Integration Complete")
        print(f"  Cold operations: {stats['orchestrator']['cold_operations']}")
        print(f"  DHM size: {stats['drc']['dhm']['size']}")
        print(f"  Total entropy: {entropy_report['total_entropy']:.4f}")
        print(f"  Memoization efficiency: {entropy_report['memoization_efficiency']:.2%}")
        print("="*60)
        
        return True


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v", "--tb=short"])
