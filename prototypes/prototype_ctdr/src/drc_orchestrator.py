"""
DRC Orchestrator — Оркестратор для управления переключением между модулями DRC.

Обеспечивает:
- Автоматический выбор режима (cold/hot)
- Сбор статистики работы
- Батчевую обработку задач
- Мониторинг энтропии
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time

from drc import DynamicReversibleCore


@dataclass
class OrchestratorStats:
    """Статистика оркестратора."""
    total_tasks: int = 0
    cold_operations: int = 0
    hot_operations: int = 0
    total_entropy: float = 0.0
    execution_times: List[float] = field(default_factory=list)
    mode_switches: int = 0
    last_mode: str = "cold"


class DRCOrchestrator:
    """
    Оркестратор для управления DRC.
    
    Отвечает за:
    - Переключение между модулями (RC ↔ SMI)
    - Мониторинг энтропии
    - Сбор метрик
    """
    
    def __init__(self, drc: Optional[DynamicReversibleCore] = None, log_file: Optional[str] = None):
        """
        Инициализация оркестратора.
        
        Args:
            drc: Экземпляр DRC (если None — создаётся новый)
            log_file: Путь к файлу для логирования
        """
        self.drc = drc if drc else DynamicReversibleCore(log_file=log_file)
        self.stats = OrchestratorStats()
    
    def execute(self, task: str) -> Dict[str, Any]:
        """
        Выполнение задачи с автоматическим выбором режима.
        
        Args:
            task: Задача для выполнения
            
        Returns:
            Результат выполнения с метаданными
        """
        start_time = time.time()
        
        # Выполнение через DRC с автоматическим выбором режима
        result = self.drc.manage_entropy(task)
        
        execution_time = time.time() - start_time
        
        # Обновление статистики
        self.stats.total_tasks += 1
        self.stats.execution_times.append(execution_time)
        
        if result["mode"] == "cold":
            self.stats.cold_operations += 1
        else:
            self.stats.hot_operations += 1
        
        # Отслеживание переключений режима
        if result["mode"] != self.stats.last_mode:
            self.stats.mode_switches += 1
            self.stats.last_mode = result["mode"]
        
        self.stats.total_entropy += result["entropy"]
        
        # Добавление метаданных к результату
        result["execution_time_ms"] = execution_time * 1000
        result["task_id"] = self.stats.total_tasks
        
        return result
    
    def execute_batch(self, tasks: List[str]) -> List[Dict[str, Any]]:
        """
        Батчевая обработка задач.
        
        Args:
            tasks: Список задач
            
        Returns:
            Список результатов
        """
        results = []
        for task in tasks:
            result = self.execute(task)
            results.append(result)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики оркестратора.
        
        Returns:
            Словарь со статистикой
        """
        avg_time = (sum(self.stats.execution_times) / len(self.stats.execution_times) 
                   if self.stats.execution_times else 0.0)
        
        return {
            "orchestrator": {
                "total_tasks": self.stats.total_tasks,
                "cold_operations": self.stats.cold_operations,
                "hot_operations": self.stats.hot_operations,
                "cold_ratio": (self.stats.cold_operations / self.stats.total_tasks 
                              if self.stats.total_tasks > 0 else 0.0),
                "total_entropy": self.stats.total_entropy,
                "avg_entropy_per_task": (self.stats.total_entropy / self.stats.total_tasks
                                        if self.stats.total_tasks > 0 else 0.0),
                "mode_switches": self.stats.mode_switches,
                "avg_execution_time_ms": avg_time * 1000,
            },
            "drc": self.drc.get_stats()
        }
    
    def reset_stats(self):
        """Сброс статистики."""
        self.stats = OrchestratorStats()
    
    def preload_knowledge(self, knowledge_items: List[Dict[str, Any]]):
        """
        Предзагрузка знаний в DHM.
        
        Args:
            knowledge_items: Список словарей {"concept": str, "content": Any, "path": Optional[str]}
        """
        for item in knowledge_items:
            self.drc.insert_knowledge(
                concept=item["concept"],
                content=item["content"],
                path=item.get("path")
            )
    
    def get_entropy_report(self) -> Dict[str, Any]:
        """
        Получение отчёта по энтропии.
        
        Returns:
            Отчёт с метриками энтропии
        """
        drc_stats = self.drc.get_stats()
        
        return {
            "total_entropy": self.stats.total_entropy,
            "avg_entropy_per_task": (self.stats.total_entropy / self.stats.total_tasks
                                    if self.stats.total_tasks > 0 else 0.0),
            "current_entropy": self.drc.get_entropy(),
            "entropy_trend": "decreasing" if self.drc.get_entropy() < 0.5 else "increasing",
            "memoization_efficiency": drc_stats["rc"]["cache_hit_rate"],
        }

