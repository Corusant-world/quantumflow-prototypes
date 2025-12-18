"""
Dynamic Reversible Core (DRC) — Архитектура обратимого ядра AGI.

Состоит из 3 модулей:
1. RC (Reversible Core) — квази-реверсивные вычисления через мемоизацию (RLA Stack)
2. DHM (Dynamic Hierarchy Manager) — иерархическая память (p-адическое дерево)
3. SMI (Sensorimotor Interface) — заземление символов (Фаза 3)

DRC обеспечивает Dynamic Entropy Management (DEM):
- "Холодное" ядро (RC) для внутренних вычислений (низкая энтропия)
- "Горячее" ядро (SMI) для взаимодействия с миром (высокая энтропия)
- DHM как диспетчер, оркестрирующий переключение
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time
import hashlib

from rla_stack import RLAStack


@dataclass
class DRCStats:
    """Статистика работы DRC."""
    cold_operations: int = 0
    hot_operations: int = 0
    total_entropy: float = 0.0
    memoization_hits: int = 0
    memoization_misses: int = 0
    dhm_lookups: int = 0
    dhm_inserts: int = 0


class DynamicReversibleCore:
    """
    Dynamic Reversible Core (DRC) — Архитектурное ядро AGI.
    
    Интегрирует RC (RLA Stack из Фазы 1) + DHM (Фаза 2) + SMI (Фаза 3).
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Инициализация DRC.
        
        Args:
            log_file: Путь к файлу для логирования энтропийных решений
        """
        # RC модуль (Reversible Core) — уже реализован в Фазе 1
        self.rc = RLAStack(log_file=log_file)
        
        # DHM модуль (Dynamic Hierarchy Manager) — реализуется в Фазе 2
        # Импорт откладывается чтобы избежать циклических зависимостей
        self._dhm = None
        
        # SMI модуль (Sensorimotor Interface) — будет в Фазе 3
        self._smi = None
        
        # Статистика
        self.stats = DRCStats()
        
        # Режим работы
        self._current_mode = "cold"  # "cold" (RC) или "hot" (SMI)
    
    @property
    def dhm(self):
        """Lazy initialization DHM."""
        if self._dhm is None:
            from dhm import DynamicHierarchyManager
            self._dhm = DynamicHierarchyManager()
        return self._dhm
    
    @property
    def smi(self):
        """SMI placeholder — будет реализован в Фазе 3."""
        return self._smi
    
    def has_memoized(self, task: str) -> bool:
        """
        Проверка наличия результата в мемоизации.
        
        Args:
            task: Задача для проверки
            
        Returns:
            True если результат есть в кэше
        """
        key = self._generate_key(task)
        return self.rc.get(key) is not None
    
    def get_memoized(self, task: str) -> Optional[Any]:
        """
        Получение мемоизированного результата.
        
        Args:
            task: Задача
            
        Returns:
            Результат из кэша или None
        """
        key = self._generate_key(task)
        result = self.rc.get(key)
        if result is not None:
            self.stats.memoization_hits += 1
        else:
            self.stats.memoization_misses += 1
        return result
    
    def memoize(self, task: str, result: Any) -> bool:
        """
        Мемоизация результата.
        
        Args:
            task: Задача
            result: Результат для сохранения
            
        Returns:
            True если было попадание в кэш (перезапись не нужна)
        """
        import numpy as np
        key = self._generate_key(task)
        
        # Конвертация result в numpy array если это не так
        if not isinstance(result, np.ndarray):
            result = np.array([result])
        
        return self.rc.memoize(key, result)
    
    def compute(self, task: str, mode: str = "cold") -> Any:
        """
        Выполнение задачи через DRC.
        
        Args:
            task: Задача для выполнения
            mode: "cold" (RC - низкая энтропия) или "hot" (SMI - высокая энтропия)
        
        Returns:
            Результат вычисления
        """
        self._current_mode = mode
        
        if mode == "cold":
            return self._compute_cold(task)
        elif mode == "hot":
            return self._compute_hot(task)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_cold(self, task: str) -> Any:
        """
        Вычисление в холодном режиме (RC + DHM).
        
        Использует мемоизацию для минимизации энтропии.
        """
        self.stats.cold_operations += 1
        
        # 1. Проверка мемоизации через RC (RLA Stack)
        cached = self.get_memoized(task)
        if cached is not None:
            return cached
        
        # 2. Поиск через DHM (иерархический поиск)
        result = self._dhm_search(task)
        
        # 3. Мемоизация результата
        if result is not None:
            self.memoize(task, result)
        
        return result
    
    def _compute_hot(self, task: str) -> Any:
        """
        Вычисление в горячем режиме (SMI).
        
        Взаимодействие с миром — высокая энтропия.
        """
        self.stats.hot_operations += 1
        
        if self._smi is None:
            raise NotImplementedError("SMI will be implemented in Phase 3")
        
        return self._smi.process(task)
    
    def _dhm_search(self, task: str) -> Optional[Any]:
        """
        Поиск через DHM.
        
        Args:
            task: Запрос для поиска
            
        Returns:
            Результат поиска или None
        """
        self.stats.dhm_lookups += 1
        
        # DHM search возвращает список (path, content, similarity)
        results = self.dhm.search(task, max_results=1)
        
        if results:
            path, content, similarity = results[0]
            return content
        
        return None
    
    def manage_entropy(self, task: str) -> Dict[str, Any]:
        """
        Динамическое управление энтропией (DEM).
        
        Автоматически выбирает режим работы (cold/hot) в зависимости от задачи.
        
        Args:
            task: Задача для выполнения
            
        Returns:
            Словарь с результатом, энтропией и режимом
        """
        # Определение типа задачи
        is_internal = self._is_internal_task(task)
        mode = "cold" if is_internal else "hot"
        
        try:
            result = self.compute(task, mode=mode)
            success = True
        except NotImplementedError:
            # SMI не реализован — fallback на cold
            result = self.compute(task, mode="cold")
            mode = "cold"
            success = True
        except Exception as e:
            result = None
            success = False
        
        # Получение энтропии
        entropy = self.get_entropy()
        self.stats.total_entropy += entropy
        
        return {
            "result": result,
            "entropy": entropy,
            "mode": mode,
            "success": success
        }
    
    def _is_internal_task(self, task: str) -> bool:
        """
        Определение типа задачи (внутренняя vs внешняя).
        
        Внутренние задачи → холодное ядро (RC)
        Внешние задачи → горячее ядро (SMI)
        """
        internal_keywords = ["search", "reason", "lookup", "find", "query", "retrieve"]
        task_lower = task.lower()
        return any(keyword in task_lower for keyword in internal_keywords)
    
    def get_entropy(self) -> float:
        """
        Получение текущей энтропии системы.
        
        Returns:
            Информационная энтропия (биты)
        """
        # Вычисление энтропии на основе статистики кэша
        total_ops = self.stats.memoization_hits + self.stats.memoization_misses
        if total_ops == 0:
            return 0.0
        
        # Энтропия = процент промахов (больше промахов = больше энтропии)
        miss_rate = self.stats.memoization_misses / total_ops
        
        # Нормализованная энтропия (0-1)
        return miss_rate
    
    def _generate_key(self, task: str) -> str:
        """Генерация ключа для мемоизации."""
        return hashlib.md5(task.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики DRC.
        
        Returns:
            Словарь со статистикой
        """
        rc_stats = self.rc.get_stats()
        
        return {
            "drc": {
                "cold_operations": self.stats.cold_operations,
                "hot_operations": self.stats.hot_operations,
                "total_entropy": self.stats.total_entropy,
                "current_mode": self._current_mode,
                "dhm_lookups": self.stats.dhm_lookups,
                "dhm_inserts": self.stats.dhm_inserts,
            },
            "rc": rc_stats,
            "dhm": {
                "size": self._dhm.size if self._dhm else 0,
            }
        }
    
    def insert_knowledge(self, concept: str, content: Any, path: Optional[str] = None) -> str:
        """
        Вставка знания в DHM.
        
        Args:
            concept: Название концепта
            content: Содержимое
            path: Опциональный путь в дереве
            
        Returns:
            p-адический путь к узлу
        """
        self.stats.dhm_inserts += 1
        return self.dhm.insert(concept, content, path)

