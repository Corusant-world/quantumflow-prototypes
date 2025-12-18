"""
A2A (Agent-to-Agent Protocol) — Стандартизированный протокол обмена намерениями.

Исключает использование естественного языка для межмашинного взаимодействия.
Использует структурированные данные (JSON) для снижения объёма токенов 
и устранения семантической двусмысленности.

Handoff Latency target: <100ms
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import json
import time
import uuid


@dataclass
class A2AMessage:
    """Структурированное сообщение A2A."""
    sender_id: str
    receiver_id: str
    intent: str  # Тип намерения: query, response, handoff, error, ack
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def to_json(self) -> str:
        """Сериализация в JSON."""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'A2AMessage':
        """Десериализация из JSON."""
        data = json.loads(json_str)
        return cls(**data)
    
    def to_bytes(self) -> bytes:
        """Сериализация в bytes."""
        return self.to_json().encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'A2AMessage':
        """Десериализация из bytes."""
        return cls.from_json(data.decode('utf-8'))


class A2AProtocol:
    """
    Agent-to-Agent Protocol.
    
    Стандартизированный протокол обмена намерениями между агентами.
    
    Интенты:
    - query: Запрос данных/действия
    - response: Ответ на запрос
    - handoff: Передача управления другому агенту
    - error: Сообщение об ошибке
    - ack: Подтверждение получения
    """
    
    HANDOFF_LATENCY_TARGET_MS = 100.0
    
    def __init__(self, agent_id: str):
        """
        Инициализация протокола.
        
        Args:
            agent_id: Идентификатор текущего агента
        """
        self.agent_id = agent_id
        self.message_queue: List[A2AMessage] = []
        self.sent_messages: List[A2AMessage] = []
        self.received_messages: List[A2AMessage] = []
        self.handoff_latencies: List[float] = []
    
    def send(self, receiver_id: str, intent: str, payload: Dict[str, Any]) -> A2AMessage:
        """
        Отправка структурированного сообщения.
        
        Args:
            receiver_id: ID получателя
            intent: Тип намерения
            payload: Данные сообщения
            
        Returns:
            Созданное сообщение
        """
        message = A2AMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            intent=intent,
            payload=payload
        )
        
        self.message_queue.append(message)
        self.sent_messages.append(message)
        
        return message
    
    def receive(self) -> Optional[A2AMessage]:
        """
        Получение сообщения из очереди для текущего агента.
        
        Returns:
            Сообщение или None
        """
        for i, msg in enumerate(self.message_queue):
            if msg.receiver_id == self.agent_id:
                message = self.message_queue.pop(i)
                self.received_messages.append(message)
                return message
        return None
    
    def handoff(self, receiver_id: str, task: str, state: Dict[str, Any]) -> A2AMessage:
        """
        Передача управления другому агенту.
        
        Args:
            receiver_id: ID агента-получателя
            task: Описание задачи
            state: Текущее состояние
            
        Returns:
            Сообщение handoff
        """
        start_time = time.time()
        
        payload = {
            "task": task,
            "state": state,
            "handoff_start": start_time
        }
        
        message = self.send(receiver_id, "handoff", payload)
        
        # Измерение латентности (симуляция)
        latency = (time.time() - start_time) * 1000
        self.handoff_latencies.append(latency)
        
        return message
    
    def acknowledge(self, original_message: A2AMessage) -> A2AMessage:
        """
        Подтверждение получения сообщения.
        
        Args:
            original_message: Исходное сообщение
            
        Returns:
            Сообщение ack
        """
        return self.send(
            original_message.sender_id,
            "ack",
            {"original_message_id": original_message.message_id}
        )
    
    def query(self, receiver_id: str, query_type: str, params: Dict[str, Any]) -> A2AMessage:
        """
        Отправка запроса.
        
        Args:
            receiver_id: ID получателя
            query_type: Тип запроса
            params: Параметры запроса
            
        Returns:
            Сообщение query
        """
        payload = {
            "query_type": query_type,
            "params": params
        }
        return self.send(receiver_id, "query", payload)
    
    def respond(self, original_message: A2AMessage, result: Any, 
                success: bool = True, error: Optional[str] = None) -> A2AMessage:
        """
        Ответ на запрос.
        
        Args:
            original_message: Исходный запрос
            result: Результат
            success: Успешность
            error: Сообщение об ошибке
            
        Returns:
            Сообщение response
        """
        payload = {
            "original_message_id": original_message.message_id,
            "result": result,
            "success": success,
            "error": error
        }
        return self.send(original_message.sender_id, "response", payload)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики протокола.
        
        Returns:
            Словарь со статистикой
        """
        avg_latency = (sum(self.handoff_latencies) / len(self.handoff_latencies)
                      if self.handoff_latencies else 0.0)
        max_latency = max(self.handoff_latencies) if self.handoff_latencies else 0.0
        
        return {
            "agent_id": self.agent_id,
            "sent_count": len(self.sent_messages),
            "received_count": len(self.received_messages),
            "queue_size": len(self.message_queue),
            "handoff_count": len(self.handoff_latencies),
            "avg_handoff_latency_ms": avg_latency,
            "max_handoff_latency_ms": max_latency,
            "meets_latency_target": max_latency < self.HANDOFF_LATENCY_TARGET_MS
        }

