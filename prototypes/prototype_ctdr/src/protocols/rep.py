"""
REP (Reconciliation and Equilibrium Protocol) — Протокол согласования решений.

Вместо передачи полного контекста агенты обмениваются "градиентами чувствительности" —
информацией о том, как изменится их решение при вариации входных условий.
Это позволяет быстро находить консенсус (равновесие Нэша) без длительных диалогов.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import time


@dataclass
class SensitivityGradient:
    """Градиент чувствительности для REP."""
    agent_id: str
    decision: Any  # Текущее решение агента
    sensitivity_vector: np.ndarray  # Как изменится решение при вариации входов
    confidence: float  # Уверенность агента (0-1)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "agent_id": self.agent_id,
            "decision": self.decision if not isinstance(self.decision, np.ndarray) 
                       else self.decision.tolist(),
            "sensitivity_vector": self.sensitivity_vector.tolist(),
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensitivityGradient':
        """Десериализация из словаря."""
        return cls(
            agent_id=data["agent_id"],
            decision=data["decision"],
            sensitivity_vector=np.array(data["sensitivity_vector"]),
            confidence=data["confidence"],
            timestamp=data.get("timestamp", time.time())
        )


class REPProtocol:
    """
    Reconciliation and Equilibrium Protocol.
    
    Протокол согласования решений через обмен "градиентами чувствительности".
    Позволяет быстро находить консенсус (равновесие Нэша) без длительных диалогов.
    
    Алгоритм:
    1. Каждый агент вычисляет свой градиент чувствительности
    2. Градиенты агрегируются
    3. Проверяется сходимость (дивергенция градиентов)
    4. Если дивергенция мала — консенсус достигнут
    """
    
    CONSENSUS_THRESHOLD = 0.9
    MAX_ITERATIONS = 10
    
    def __init__(self):
        """Инициализация протокола."""
        self.gradients: List[SensitivityGradient] = []
        self.consensus_history: List[Dict[str, Any]] = []
        self.iteration_count = 0
    
    def submit_gradient(self, agent_id: str, decision: Any, 
                       sensitivity_vector: np.ndarray, confidence: float):
        """
        Отправка градиента чувствительности.
        
        Args:
            agent_id: ID агента
            decision: Текущее решение
            sensitivity_vector: Градиент чувствительности
            confidence: Уверенность (0-1)
        """
        gradient = SensitivityGradient(
            agent_id=agent_id,
            decision=decision,
            sensitivity_vector=sensitivity_vector,
            confidence=confidence
        )
        self.gradients.append(gradient)
    
    def compute_divergence(self) -> float:
        """
        Вычисление дивергенции градиентов.
        
        Returns:
            Дивергенция (0 = полное согласие, 1+ = разногласие)
        """
        if len(self.gradients) < 2:
            return 0.0
        
        # Собираем все градиенты в массив
        gradients_array = np.array([g.sensitivity_vector for g in self.gradients])
        
        # Средний градиент
        mean_gradient = np.mean(gradients_array, axis=0)
        
        # Дивергенция = среднее отклонение от среднего
        divergence = np.mean([
            np.linalg.norm(g.sensitivity_vector - mean_gradient)
            for g in self.gradients
        ])
        
        return float(divergence)
    
    def find_consensus(self, threshold: Optional[float] = None) -> Optional[Any]:
        """
        Поиск консенсуса через анализ градиентов.
        
        Args:
            threshold: Порог для консенсуса (по умолчанию CONSENSUS_THRESHOLD)
        
        Returns:
            Консенсусное решение или None
        """
        if threshold is None:
            threshold = self.CONSENSUS_THRESHOLD
        
        if not self.gradients:
            return None
        
        self.iteration_count += 1
        
        # Вычисление дивергенции
        divergence = self.compute_divergence()
        
        # Проверка сходимости
        consensus_reached = divergence < (1 - threshold)
        
        # Логирование
        self.consensus_history.append({
            "iteration": self.iteration_count,
            "divergence": divergence,
            "num_agents": len(self.gradients),
            "consensus_reached": consensus_reached,
            "timestamp": time.time()
        })
        
        if consensus_reached:
            # Взвешенное среднее решений
            return self._compute_weighted_decision()
        
        return None
    
    def _compute_weighted_decision(self) -> Any:
        """
        Вычисление взвешенного среднего решений.
        
        Returns:
            Взвешенное решение
        """
        if not self.gradients:
            return None
        
        total_weight = sum(g.confidence for g in self.gradients)
        if total_weight == 0:
            return self.gradients[0].decision
        
        # Пытаемся вычислить взвешенное среднее
        try:
            # Если решения — числа
            if all(isinstance(g.decision, (int, float)) for g in self.gradients):
                weighted_sum = sum(g.decision * g.confidence for g in self.gradients)
                return weighted_sum / total_weight
            
            # Если решения — numpy arrays
            if all(isinstance(g.decision, np.ndarray) for g in self.gradients):
                weighted_sum = sum(g.decision * g.confidence for g in self.gradients)
                return weighted_sum / total_weight
            
            # Иначе — возвращаем решение с максимальной уверенностью
            best_gradient = max(self.gradients, key=lambda g: g.confidence)
            return best_gradient.decision
            
        except Exception:
            # Fallback
            best_gradient = max(self.gradients, key=lambda g: g.confidence)
            return best_gradient.decision
    
    def iterate_until_consensus(self, 
                               agents: List['Agent'],
                               max_iterations: Optional[int] = None) -> Optional[Any]:
        """
        Итеративный поиск консенсуса.
        
        Args:
            agents: Список агентов
            max_iterations: Максимальное количество итераций
            
        Returns:
            Консенсусное решение или None
        """
        if max_iterations is None:
            max_iterations = self.MAX_ITERATIONS
        
        for i in range(max_iterations):
            # Сбор градиентов от всех агентов
            self.gradients.clear()
            for agent in agents:
                gradient = agent.compute_sensitivity_gradient()
                self.submit_gradient(
                    agent_id=agent.id,
                    decision=gradient["decision"],
                    sensitivity_vector=gradient["sensitivity"],
                    confidence=gradient["confidence"]
                )
            
            # Проверка консенсуса
            consensus = self.find_consensus()
            if consensus is not None:
                return consensus
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики протокола.
        
        Returns:
            Словарь со статистикой
        """
        divergence = self.compute_divergence()
        
        return {
            "num_gradients": len(self.gradients),
            "iteration_count": self.iteration_count,
            "current_divergence": divergence,
            "consensus_reached": divergence < (1 - self.CONSENSUS_THRESHOLD),
            "history_length": len(self.consensus_history),
            "agents": [g.agent_id for g in self.gradients]
        }
    
    def reset(self):
        """Сброс протокола."""
        self.gradients.clear()
        self.iteration_count = 0


# Базовый интерфейс агента для REP
class Agent:
    """Базовый интерфейс агента для REP протокола."""
    
    def __init__(self, agent_id: str):
        self.id = agent_id
    
    def compute_sensitivity_gradient(self) -> Dict[str, Any]:
        """
        Вычисление градиента чувствительности.
        
        Returns:
            Словарь с decision, sensitivity, confidence
        """
        raise NotImplementedError("Subclasses must implement compute_sensitivity_gradient")

