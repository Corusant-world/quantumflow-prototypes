"""
RLA-стек (Программная Аппроксимация Обратимой Логики)
Использование DPX для минимизации информационных потерь (стирания)

Цель: 2× меньше перезаписей vs классический LLM
Принцип Ландауэра: E_min = k_B * T * ln(2) ≈ 2.9 × 10^(-21) Дж @ 300K
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import json
from pathlib import Path
from datetime import datetime

if TYPE_CHECKING:
    from .kv_cache_steering import KVCacheSteeringDPX


class RLAStack:
    """
    RLA-стек для минимизации энтропии через DPX-мемоизацию.
    
    Отслеживает:
    - Количество перезаписей в память/регистры
    - Информационную энтропию (Шенноновская)
    - Термодинамическую энтропию (Ландауэр)
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Инициализация RLA-стека.
        
        Args:
            log_file: Путь к файлу для логирования решений (опционально)
        """
        self.log_file = log_file
        self.memory_writes = 0  # Счетчик перезаписей
        self.memory_reads = 0   # Счетчик чтений
        self.cache_hits = 0     # Счетчик попаданий в кэш
        self.cache_misses = 0   # Счетчик промахов кэша
        
        # Мемоизация промежуточных результатов
        self.memoization_cache: Dict[str, np.ndarray] = {}
        
        # Энтропийные метрики
        self.entropy_log: List[Dict] = []
        
        # Константы
        self.K_B = 1.380649e-23  # Постоянная Больцмана (Дж/К)
        self.T = 300.0  # Температура (К)
        self.E_MIN = self.K_B * self.T * np.log(2)  # Минимальная энергия Ландауэра
        
    def memoize(self, key: str, value: np.ndarray) -> bool:
        """
        Мемоизация результата с проверкой на перезапись.
        
        Args:
            key: Ключ для мемоизации
            value: Значение для сохранения
            
        Returns:
            True если было попадание в кэш (перезапись не нужна), False если нужна перезапись
        """
        if key in self.memoization_cache:
            # Проверка на идентичность (избегаем перезаписи)
            cached_value = self.memoization_cache[key]
            if np.array_equal(cached_value, value):
                # Значение уже в кэше и идентично - перезапись не нужна
                return True  # Попадание в кэш, перезапись не нужна
            else:
                # Перезапись необходима (значение изменилось)
                self.memory_writes += 1
        else:
            # Первая запись
            self.memory_writes += 1
        
        # Сохраняем значение
        self.memoization_cache[key] = value.copy()
        return False
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Получение значения из кэша.
        
        Args:
            key: Ключ для поиска
            
        Returns:
            Значение из кэша или None
        """
        if key in self.memoization_cache:
            self.cache_hits += 1
            self.memory_reads += 1
            return self.memoization_cache[key].copy()
        else:
            self.cache_misses += 1
            return None
    
    def compute_information_entropy(self, data: np.ndarray) -> float:
        """
        Вычисление информационной энтропии (Шенноновская).
        
        Мера непредсказуемости данных.
        
        Args:
            data: Массив данных
            
        Returns:
            Информационная энтропия (биты)
        """
        # Нормализация данных
        data_flat = data.flatten().astype(float)
        if data_flat.size == 0:
            return 0.0
        
        # Вычисление вероятностей
        unique, counts = np.unique(data_flat, return_counts=True)
        probabilities = counts / data_flat.size
        
        # Энтропия Шеннона: H = -Σ p_i * log2(p_i)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def compute_thermodynamic_entropy(self, num_operations: int) -> float:
        """
        Вычисление термодинамической энтропии (Ландауэр).
        
        Энергия на операцию стирания информации.
        
        Args:
            num_operations: Количество операций стирания
            
        Returns:
            Термодинамическая энтропия (Дж)
        """
        # Энергия = количество операций * минимальная энергия Ландауэра
        energy = num_operations * self.E_MIN
        return energy
    
    def log_entropy_decision(self, operation: str, info_entropy: float, 
                            thermo_entropy: float, memoized: bool):
        """
        Логирование энтропийного решения.
        
        Args:
            operation: Название операции
            info_entropy: Информационная энтропия
            thermo_entropy: Термодинамическая энтропия
            memoized: Была ли использована мемоизация
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "information_entropy": float(info_entropy),
            "thermodynamic_entropy": float(thermo_entropy),
            "memoized": memoized,
            "memory_writes": self.memory_writes,
            "memory_reads": self.memory_reads,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }
        
        self.entropy_log.append(entry)
        
        # Сохранение в файл, если указан
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Загрузка существующих записей
            if log_path.exists() and log_path.stat().st_size > 0:
                try:
                    with open(log_path, 'r') as f:
                        data = json.load(f)
                    if "decisions" not in data:
                        data = {"decisions": []}
                except (json.JSONDecodeError, ValueError):
                    # Если файл поврежден, создаем новый
                    data = {"decisions": []}
            else:
                data = {"decisions": []}
            
            data["decisions"].append(entry)
            
            # Сохранение
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
    
    def wrap_reversible_einsum(self, A: np.ndarray, B: np.ndarray, 
                              threshold: float = 0.5) -> np.ndarray:
        """
        Обертка для reversible_einsum с RLA-мемоизацией.
        
        Args:
            A: Матрица A
            B: Матрица B
            threshold: Порог Heaviside
            
        Returns:
            Результат reversible_einsum
        """
        # Генерация ключа для мемоизации
        import hashlib
        key_data = f"{A.tobytes()}{B.tobytes()}{threshold}".encode()
        key = hashlib.md5(key_data).hexdigest()
        
        # Проверка кэша
        cached_result = self.get(key)
        if cached_result is not None:
            # Вычисление энтропийных метрик
            info_entropy = self.compute_information_entropy(cached_result)
            thermo_entropy = self.compute_thermodynamic_entropy(0)  # Нет операций стирания
            
            self.log_entropy_decision(
                "reversible_einsum",
                info_entropy,
                thermo_entropy,
                memoized=True
            )
            return cached_result
        
        # Выполнение операции
        from .core import reversible_einsum
        result = reversible_einsum(A, B, threshold)
        
        # Мемоизация результата
        self.memoize(key, result)
        
        # Вычисление энтропийных метрик
        info_entropy = self.compute_information_entropy(result)
        thermo_entropy = self.compute_thermodynamic_entropy(1)  # Одна операция стирания
        
        self.log_entropy_decision(
            "reversible_einsum",
            info_entropy,
            thermo_entropy,
            memoized=False
        )
        
        return result
    
    def get_stats(self) -> Dict:
        """
        Получение статистики RLA-стека.
        
        Returns:
            Словарь со статистикой
        """
        total_operations = self.memory_writes + self.memory_reads
        cache_hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses) 
                         if (self.cache_hits + self.cache_misses) > 0 else 0.0)
        
        return {
            "memory_writes": self.memory_writes,
            "memory_reads": self.memory_reads,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "total_operations": total_operations,
            "memoization_cache_size": len(self.memoization_cache),
            "entropy_log_entries": len(self.entropy_log),
        }
    
    def compare_with_baseline(self, baseline_writes: int) -> Dict:
        """
        Сравнение с baseline (классический LLM).
        
        Метрика: 2× меньше перезаписей vs baseline
        
        Args:
            baseline_writes: Количество перезаписей в baseline
            
        Returns:
            Словарь с результатами сравнения
        """
        reduction_factor = baseline_writes / self.memory_writes if self.memory_writes > 0 else 0.0
        target_reduction = 2.0
        
        return {
            "baseline_writes": baseline_writes,
            "rla_writes": self.memory_writes,
            "reduction_factor": reduction_factor,
            "target_reduction": target_reduction,
            "meets_target": reduction_factor >= target_reduction,
        }

