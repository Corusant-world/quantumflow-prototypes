"""
Blame Logic — Реверсивный анализ графа состояний для трассировки ошибок.

РЕАЛЬНАЯ ИМПЛЕМЕНТАЦИЯ (не симуляция):
- При ошибке DHM — реально перестраивает индекс и повторяет
- При ошибке мемоизации — реально очищает кэш и повторяет
- При ошибке A2A — реально переотправляет сообщение

Self-healing target: >90% автономное восстановление
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import traceback


class OperationType(Enum):
    """Типы операций для трассировки."""
    KERNEL_CALL = "kernel_call"
    MEMORY_ACCESS = "memory_access"
    CACHE_OPERATION = "cache_operation"
    DHM_SEARCH = "dhm_search"
    DHM_INSERT = "dhm_insert"
    ATTENTION_COMPUTE = "attention_compute"
    MEMOIZATION = "memoization"
    A2A_SEND = "a2a_send"
    A2A_RECEIVE = "a2a_receive"
    REP_GRADIENT = "rep_gradient"


@dataclass
class StateNode:
    """Узел в графе состояний."""
    id: str
    operation: OperationType
    timestamp: float
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    agent_id: Optional[str] = None
    parent_id: Optional[str] = None
    
    # Для реального восстановления
    retry_func: Optional[Callable] = None
    retry_args: Optional[tuple] = None
    retry_kwargs: Optional[Dict] = None


@dataclass
class RecoveryResult:
    """Результат попытки восстановления."""
    success: bool
    action_taken: str
    original_error: str
    new_result: Any = None
    recovery_time_ms: float = 0.0


class BlameLogic:
    """
    Blame Logic с РЕАЛЬНЫМ восстановлением.
    
    НЕ использует random. Реально выполняет recovery actions.
    """
    
    SELF_HEALING_TARGET = 0.9
    
    def __init__(self):
        self.state_graph: List[StateNode] = []
        self.error_nodes: List[str] = []
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        self._node_counter = 0
        
        # Реальные компоненты для восстановления (устанавливаются извне)
        self._dhm = None
        self._rla_stack = None
        self._a2a_protocol = None
    
    def set_components(self, dhm=None, rla_stack=None, a2a_protocol=None):
        """Установка реальных компонентов для восстановления."""
        if dhm is not None:
            self._dhm = dhm
        if rla_stack is not None:
            self._rla_stack = rla_stack
        if a2a_protocol is not None:
            self._a2a_protocol = a2a_protocol
    
    def _generate_node_id(self, operation: OperationType) -> str:
        """Генерация уникального ID узла."""
        self._node_counter += 1
        return f"{operation.value}_{self._node_counter}"
    
    def record_operation(
        self,
        operation: OperationType,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        success: bool = True,
        error: Optional[str] = None,
        agent_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        retry_func: Optional[Callable] = None,
        retry_args: Optional[tuple] = None,
        retry_kwargs: Optional[Dict] = None
    ) -> str:
        """
        Запись операции в граф состояний.
        
        Args:
            retry_func: Функция для повторного выполнения при ошибке
            retry_args: Аргументы для retry_func
            retry_kwargs: Keyword аргументы для retry_func
        """
        node_id = self._generate_node_id(operation)
        
        error_tb = None
        if not success and error:
            error_tb = traceback.format_exc()
        
        node = StateNode(
            id=node_id,
            operation=operation,
            timestamp=time.time(),
            inputs=inputs,
            outputs=outputs,
            success=success,
            error=error,
            error_traceback=error_tb,
            agent_id=agent_id,
            parent_id=parent_id,
            retry_func=retry_func,
            retry_args=retry_args or (),
            retry_kwargs=retry_kwargs or {}
        )
        
        self.state_graph.append(node)
        
        if not success:
            self.error_nodes.append(node_id)
        
        return node_id
    
    def _find_node(self, node_id: str) -> Optional[StateNode]:
        """Поиск узла по ID."""
        for node in self.state_graph:
            if node.id == node_id:
                return node
        return None
    
    def trace_error(self, error_node_id: str) -> List[StateNode]:
        """Реверсивный поиск источника ошибки."""
        error_node = self._find_node(error_node_id)
        if not error_node:
            return []
        
        trace = [error_node]
        current = error_node
        
        while current.parent_id:
            parent = self._find_node(current.parent_id)
            if not parent:
                break
            trace.append(parent)
            current = parent
        
        return list(reversed(trace))
    
    def attempt_recovery(self, error_node_id: str) -> RecoveryResult:
        """
        РЕАЛЬНАЯ попытка восстановления.
        
        Выполняет конкретные действия в зависимости от типа ошибки.
        """
        node = self._find_node(error_node_id)
        if not node:
            self.failed_recoveries += 1
            return RecoveryResult(
                success=False,
                action_taken="none",
                original_error="Node not found"
            )
        
        start_time = time.time()
        result = None
        
        try:
            # РЕАЛЬНОЕ восстановление в зависимости от типа операции
            
            if node.operation == OperationType.DHM_SEARCH:
                result = self._recover_dhm_search(node)
            
            elif node.operation == OperationType.DHM_INSERT:
                result = self._recover_dhm_insert(node)
            
            elif node.operation == OperationType.MEMOIZATION:
                result = self._recover_memoization(node)
            
            elif node.operation == OperationType.A2A_SEND:
                result = self._recover_a2a_send(node)
            
            elif node.operation == OperationType.A2A_RECEIVE:
                result = self._recover_a2a_receive(node)
            
            elif node.retry_func is not None:
                # Общий случай: есть retry функция
                result = self._recover_with_retry_func(node)
            
            else:
                # Нет способа восстановления
                self.failed_recoveries += 1
                return RecoveryResult(
                    success=False,
                    action_taken="no_recovery_method",
                    original_error=node.error or "Unknown error",
                    recovery_time_ms=(time.time() - start_time) * 1000
                )
            
            if result.success:
                self.successful_recoveries += 1
            else:
                self.failed_recoveries += 1
            
            result.recovery_time_ms = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            self.failed_recoveries += 1
            return RecoveryResult(
                success=False,
                action_taken="recovery_exception",
                original_error=f"{node.error} -> Recovery failed: {str(e)}",
                recovery_time_ms=(time.time() - start_time) * 1000
            )
    
    def _recover_dhm_search(self, node: StateNode) -> RecoveryResult:
        """Реальное восстановление DHM search."""
        if self._dhm is None:
            return RecoveryResult(
                success=False,
                action_taken="dhm_not_available",
                original_error=node.error or "DHM error"
            )
        
        # Действие 1: Перезагрузить GPU индекс
        try:
            if hasattr(self._dhm, '_gpu_index_loaded'):
                self._dhm._gpu_index_loaded = False
            if hasattr(self._dhm, '_load_gpu_index'):
                self._dhm._load_gpu_index()
        except Exception:
            pass  # Продолжаем даже если не удалось
        
        # Действие 2: Повторить поиск
        query = node.inputs.get("query", "")
        max_results = node.inputs.get("max_results", 10)
        
        try:
            results = self._dhm.search(query, max_results=max_results)
            return RecoveryResult(
                success=True,
                action_taken="dhm_index_reload_and_retry",
                original_error=node.error or "DHM search error",
                new_result=results
            )
        except Exception as e:
            return RecoveryResult(
                success=False,
                action_taken="dhm_retry_failed",
                original_error=f"{node.error} -> {str(e)}"
            )
    
    def _recover_dhm_insert(self, node: StateNode) -> RecoveryResult:
        """Реальное восстановление DHM insert."""
        if self._dhm is None:
            return RecoveryResult(
                success=False,
                action_taken="dhm_not_available",
                original_error=node.error or "DHM error"
            )
        
        concept = node.inputs.get("concept", "")
        content = node.inputs.get("content")
        path = node.inputs.get("path")
        
        try:
            result_path = self._dhm.insert(concept, content, path)
            return RecoveryResult(
                success=True,
                action_taken="dhm_insert_retry",
                original_error=node.error or "DHM insert error",
                new_result=result_path
            )
        except Exception as e:
            return RecoveryResult(
                success=False,
                action_taken="dhm_insert_retry_failed",
                original_error=f"{node.error} -> {str(e)}"
            )
    
    def _recover_memoization(self, node: StateNode) -> RecoveryResult:
        """Реальное восстановление мемоизации."""
        if self._rla_stack is None:
            return RecoveryResult(
                success=False,
                action_taken="rla_not_available",
                original_error=node.error or "Memoization error"
            )
        
        # Действие: очистить проблемный ключ из кэша
        key = node.inputs.get("key", "")
        
        try:
            if key in self._rla_stack.memoization_cache:
                del self._rla_stack.memoization_cache[key]
            
            # Повторить операцию если есть данные
            value = node.inputs.get("value")
            if value is not None:
                import numpy as np
                if not isinstance(value, np.ndarray):
                    value = np.array([value])
                self._rla_stack.memoize(key, value)
            
            return RecoveryResult(
                success=True,
                action_taken="cache_clear_and_retry",
                original_error=node.error or "Memoization error"
            )
        except Exception as e:
            return RecoveryResult(
                success=False,
                action_taken="memoization_retry_failed",
                original_error=f"{node.error} -> {str(e)}"
            )
    
    def _recover_a2a_send(self, node: StateNode) -> RecoveryResult:
        """Реальное восстановление A2A send."""
        if self._a2a_protocol is None:
            return RecoveryResult(
                success=False,
                action_taken="a2a_not_available",
                original_error=node.error or "A2A error"
            )
        
        receiver_id = node.inputs.get("receiver_id", "")
        intent = node.inputs.get("intent", "")
        payload = node.inputs.get("payload", {})
        
        try:
            # Повторная отправка
            msg = self._a2a_protocol.send(receiver_id, intent, payload)
            return RecoveryResult(
                success=True,
                action_taken="a2a_resend",
                original_error=node.error or "A2A send error",
                new_result=msg
            )
        except Exception as e:
            return RecoveryResult(
                success=False,
                action_taken="a2a_resend_failed",
                original_error=f"{node.error} -> {str(e)}"
            )
    
    def _recover_a2a_receive(self, node: StateNode) -> RecoveryResult:
        """Реальное восстановление A2A receive."""
        if self._a2a_protocol is None:
            return RecoveryResult(
                success=False,
                action_taken="a2a_not_available",
                original_error=node.error or "A2A error"
            )
        
        try:
            # Повторная попытка получения
            msg = self._a2a_protocol.receive()
            return RecoveryResult(
                success=msg is not None,
                action_taken="a2a_retry_receive",
                original_error=node.error or "A2A receive error",
                new_result=msg
            )
        except Exception as e:
            return RecoveryResult(
                success=False,
                action_taken="a2a_receive_failed",
                original_error=f"{node.error} -> {str(e)}"
            )
    
    def _recover_with_retry_func(self, node: StateNode) -> RecoveryResult:
        """Восстановление через сохраненную retry функцию."""
        try:
            result = node.retry_func(*node.retry_args, **node.retry_kwargs)
            return RecoveryResult(
                success=True,
                action_taken="retry_func_executed",
                original_error=node.error or "Unknown error",
                new_result=result
            )
        except Exception as e:
            return RecoveryResult(
                success=False,
                action_taken="retry_func_failed",
                original_error=f"{node.error} -> {str(e)}"
            )
    
    def get_self_healing_rate(self) -> float:
        """Текущий показатель самоисцеления."""
        total = self.successful_recoveries + self.failed_recoveries
        if total == 0:
            return 1.0
        return self.successful_recoveries / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика Blame Logic."""
        healing_rate = self.get_self_healing_rate()
        return {
            "total_operations": len(self.state_graph),
            "error_count": len(self.error_nodes),
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "self_healing_rate": healing_rate,
            "meets_target": healing_rate >= self.SELF_HEALING_TARGET,
            "target": self.SELF_HEALING_TARGET,
            "components_connected": {
                "dhm": self._dhm is not None,
                "rla_stack": self._rla_stack is not None,
                "a2a": self._a2a_protocol is not None
            }
        }
    
    def clear(self):
        """Очистка графа состояний."""
        self.state_graph.clear()
        self.error_nodes.clear()
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        self._node_counter = 0
