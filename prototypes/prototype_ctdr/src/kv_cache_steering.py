"""
KV_Cache_Steering_DPX: DPX-ускорённая мемоизация для устранения Catastrophic Forgetting

Двухуровневая система памяти:
- SRAM: Для частых обращений (горячие состояния)
- L2 Cache (50 МБ на H100): Для менее частых состояний
- HBM3 (3 ТБ/с): Минимизация обращений через эффективное управление кэшем

Метрики:
- Cache hit rate: ≥80%
- Latency reduction: ≥7× (цель ≥10×)
- Token Reduction: ≥31% за счет DHM-мемоизации
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass
class CacheEntry:
    """Запись в кэше KV."""
    key: str
    value: np.ndarray
    access_count: int = 0
    last_access: float = 0.0
    frequency: float = 0.0  # Частота обращения (для ранжирования)


class KVCacheSteeringDPX:
    """
    KV_Cache_Steering_DPX: Управление KV Cache через DPX-мемоизацию.
    
    Использует DPX_LCP_Kernel для O(N) поиска в кэше через Baire Metric.
    Двухуровневая система: SRAM (горячие) + L2 Cache (менее частые).
    """
    
    def __init__(self, 
                 sram_size: int = 1024,  # Размер SRAM кэша (количество записей)
                 l2_size: int = 10000,   # Размер L2 кэша (количество записей)
                 log_file: Optional[str] = None,
                 rla_stack: Optional['RLAStack'] = None):
        """
        Инициализация KV_Cache_Steering_DPX.
        
        Args:
            sram_size: Размер SRAM кэша (горячие состояния)
            l2_size: Размер L2 кэша (менее частые состояния)
            log_file: Путь к файлу для логирования метрик
            rla_stack: Опциональный RLA-стек для отслеживания энтропийных метрик
        """
        self.sram_size = sram_size
        self.l2_size = l2_size
        
        # Двухуровневая система памяти
        self.sram_cache: OrderedDict[str, CacheEntry] = OrderedDict()  # SRAM: горячие состояния
        self.l2_cache: OrderedDict[str, CacheEntry] = OrderedDict()    # L2: менее частые
        
        # Метрики производительности
        self.cache_hits = 0
        self.cache_misses = 0
        self.sram_hits = 0
        self.l2_hits = 0
        self.total_queries = 0
        
        # Интеграция с DPX_LCP_Kernel для поиска
        self._dpx_lcp_available = False
        try:
            from .core import dpx_lcp
            self._dpx_lcp_available = True
            self._dpx_lcp_func = dpx_lcp
        except ImportError:
            # Fallback на CPU LCP
            from .core import lcp_cpu
            self._dpx_lcp_func = lcp_cpu
        
        # Интеграция с Reversible_Einsum_Engine для мемоизации
        self._reversible_einsum_available = False
        try:
            from .core import reversible_einsum
            self._reversible_einsum_available = True
            self._reversible_einsum_func = reversible_einsum
        except ImportError:
            self._reversible_einsum_func = None
        
        # Логирование
        self.log_file = log_file
        self.metrics_log: List[Dict] = []
        
        # Временные метки для вычисления частоты
        import time
        self._time_func = time.time
        
        # Интеграция с RLA-стеком (опционально)
        self._rla_stack = rla_stack
        
    def _compute_lcp(self, s1: str, s2: str) -> int:
        """
        Вычисление LCP через DPX_LCP_Kernel или CPU fallback.
        
        Args:
            s1: Строка 1
            s2: Строка 2
            
        Returns:
            Длина LCP
        """
        if self._dpx_lcp_available:
            # Используем DPX_LCP_Kernel для O(N) поиска
            return self._dpx_lcp_func(s1, s2)
        else:
            # CPU fallback - используем lcp_cpu напрямую
            from .core import lcp_cpu
            return lcp_cpu(s1, s2)
    
    def _find_similar_key(self, query_key: str, cache: OrderedDict[str, CacheEntry], 
                         threshold: float = 0.8) -> Optional[str]:
        """
        Поиск похожего ключа в кэше через Baire Metric (LCP).
        
        Args:
            query_key: Ключ для поиска
            cache: Кэш для поиска
            threshold: Порог схожести (0.0-1.0)
            
        Returns:
            Найденный ключ или None
        """
        if not cache:
            return None
        
        max_similarity = 0.0
        best_key = None
        
        for cached_key in cache.keys():
            # Вычисление LCP через DPX_LCP_Kernel
            lcp_len = self._compute_lcp(query_key, cached_key)
            
            # Нормализация: similarity = LCP / max(len(query), len(cached))
            max_len = max(len(query_key), len(cached_key))
            if max_len == 0:
                similarity = 1.0
            else:
                similarity = lcp_len / max_len
            
            if similarity >= threshold and similarity > max_similarity:
                max_similarity = similarity
                best_key = cached_key
        
        return best_key if max_similarity >= threshold else None
    
    def _update_access_stats(self, entry: CacheEntry):
        """Обновление статистики доступа к записи."""
        entry.access_count += 1
        entry.last_access = self._time_func()
        
        # Вычисление частоты (обращений в секунду)
        # Упрощенная модель: частота = количество обращений / время с момента создания
        if entry.last_access > 0:
            time_diff = entry.last_access - (entry.last_access - 1.0)  # Упрощение
            entry.frequency = entry.access_count / max(time_diff, 1.0)
    
    def _promote_to_sram(self, key: str, entry: CacheEntry):
        """Перемещение записи в SRAM (горячие состояния)."""
        # Удаление из L2, если там есть
        if key in self.l2_cache:
            del self.l2_cache[key]
        
        # Добавление в SRAM
        self.sram_cache[key] = entry
        
        # Ограничение размера SRAM (вытеснение наименее используемых)
        if len(self.sram_cache) > self.sram_size:
            # CAKE алгоритм: вытеснение наименее частых
            least_frequent = min(self.sram_cache.items(), 
                               key=lambda x: x[1].frequency)
            least_key = least_frequent[0]
            least_entry = least_frequent[1]
            
            # Перемещение в L2
            self.l2_cache[least_key] = least_entry
            del self.sram_cache[least_key]
    
    def _evict_from_l2(self):
        """Вытеснение из L2 кэша (CAKE алгоритм)."""
        if len(self.l2_cache) <= self.l2_size:
            return
        
        # Вытеснение наименее частых записей
        sorted_entries = sorted(self.l2_cache.items(), 
                              key=lambda x: x[1].frequency)
        
        # Удаление 10% наименее частых
        num_to_evict = max(1, len(self.l2_cache) // 10)
        for i in range(num_to_evict):
            key, _ = sorted_entries[i]
            del self.l2_cache[key]
    
    def get(self, key: str, similarity_threshold: float = 0.8) -> Optional[np.ndarray]:
        """
        Получение значения из кэша с поиском через DPX_LCP_Kernel.
        
        Args:
            key: Ключ для поиска
            similarity_threshold: Порог схожести для поиска (0.0-1.0)
            
        Returns:
            Значение из кэша или None
        """
        self.total_queries += 1
        
        # 1. Точное совпадение в SRAM
        if key in self.sram_cache:
            entry = self.sram_cache[key]
            self._update_access_stats(entry)
            self.cache_hits += 1
            self.sram_hits += 1
            
            # Перемещение в начало (LRU)
            self.sram_cache.move_to_end(key)
            return entry.value.copy()
        
        # 2. Точное совпадение в L2
        if key in self.l2_cache:
            entry = self.l2_cache[key]
            self._update_access_stats(entry)
            self.cache_hits += 1
            self.l2_hits += 1
            
            # Продвижение в SRAM (горячее состояние)
            self._promote_to_sram(key, entry)
            return entry.value.copy()
        
        # 3. Поиск похожего ключа через DPX_LCP_Kernel (Baire Metric)
        # Сначала в SRAM
        similar_key = self._find_similar_key(key, self.sram_cache, similarity_threshold)
        if similar_key:
            entry = self.sram_cache[similar_key]
            self._update_access_stats(entry)
            self.cache_hits += 1
            self.sram_hits += 1
            return entry.value.copy()
        
        # Затем в L2
        similar_key = self._find_similar_key(key, self.l2_cache, similarity_threshold)
        if similar_key:
            entry = self.l2_cache[similar_key]
            self._update_access_stats(entry)
            self.cache_hits += 1
            self.l2_hits += 1
            
            # Продвижение в SRAM
            self._promote_to_sram(similar_key, entry)
            return entry.value.copy()
        
        # Промах кэша
        self.cache_misses += 1
        return None
    
    def put(self, key: str, value: np.ndarray, frequency: float = 1.0):
        """
        Добавление значения в кэш.
        
        Args:
            key: Ключ
            value: Значение
            frequency: Начальная частота обращения
        """
        entry = CacheEntry(
            key=key,
            value=value.copy(),
            access_count=1,
            last_access=self._time_func(),
            frequency=frequency
        )
        
        # Добавление в SRAM (горячее состояние)
        self.sram_cache[key] = entry
        
        # Ограничение размера SRAM
        if len(self.sram_cache) > self.sram_size:
            # Перемещение наименее частых в L2
            least_frequent = min(self.sram_cache.items(), 
                               key=lambda x: x[1].frequency)
            least_key = least_frequent[0]
            least_entry = least_frequent[1]
            
            self.l2_cache[least_key] = least_entry
            del self.sram_cache[least_key]
        
        # Ограничение размера L2
        self._evict_from_l2()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики производительности.
        
        Returns:
            Словарь со статистикой
        """
        total_queries = self.total_queries if self.total_queries > 0 else 1
        cache_hit_rate = (self.cache_hits / total_queries) * 100.0
        
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "sram_hits": self.sram_hits,
            "l2_hits": self.l2_hits,
            "sram_size": len(self.sram_cache),
            "l2_size": len(self.l2_cache),
            "sram_capacity": self.sram_size,
            "l2_capacity": self.l2_size,
        }
    
    def log_metrics(self):
        """Логирование метрик производительности."""
        stats = self.get_stats()
        entry = {
            "timestamp": datetime.now().isoformat(),
            **stats
        }
        
        self.metrics_log.append(entry)
        
        # Сохранение в файл, если указан
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Загрузка существующих записей
            if log_path.exists() and log_path.stat().st_size > 0:
                try:
                    with open(log_path, 'r') as f:
                        data = json.load(f)
                    if "metrics" not in data:
                        data = {"metrics": []}
                except (json.JSONDecodeError, ValueError):
                    # Если файл поврежден, создаем новый
                    data = {"metrics": []}
            else:
                data = {"metrics": []}
            
            data["metrics"].append(entry)
            
            # Сохранение
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
    
    def consolidate_ltm(self, keys: List[str]) -> Dict[str, np.ndarray]:
        """
        Консолидация LTM (Long-Term Memory) через DPX-мемоизацию.
        
        Args:
            keys: Список ключей для консолидации
            
        Returns:
            Словарь с консолидированными значениями
        """
        consolidated = {}
        
        for key in keys:
            value = self.get(key)
            if value is not None:
                consolidated[key] = value
        
        return consolidated
    
    def memoize_einsum_result(self, key: str, A: np.ndarray, B: np.ndarray, 
                             threshold: float = 0.5) -> np.ndarray:
        """
        Мемоизация результата reversible_einsum через KV Cache.
        
        Интеграция с Reversible_Einsum_Engine для мемоизации промежуточных результатов.
        
        Args:
            key: Ключ для мемоизации
            A: Матрица A
            B: Матрица B
            threshold: Порог Heaviside
            
        Returns:
            Результат reversible_einsum (из кэша или вычисленный)
        """
        # Проверка кэша
        cached_result = self.get(key)
        if cached_result is not None:
            return cached_result
        
        # Выполнение операции через Reversible_Einsum_Engine
        if self._reversible_einsum_available and self._reversible_einsum_func:
            result = self._reversible_einsum_func(A, B, threshold)
        else:
            # Fallback на CPU
            from .core import einsum_cpu
            C = einsum_cpu(A, B)
            result = (C.astype(float) >= threshold).astype(bool)
        
        # Сохранение в кэш
        self.put(key, result, frequency=1.0)
        
        return result
    
    def integrate_rla_stack(self, rla_stack: 'RLAStack'):
        """
        Интеграция с RLA-стеком для отслеживания энтропийных метрик.
        
        Args:
            rla_stack: Экземпляр RLAStack для отслеживания энтропии
        """
        self._rla_stack = rla_stack
    
    def get_with_rla(self, key: str, similarity_threshold: float = 0.8) -> Optional[np.ndarray]:
        """
        Получение значения из кэша с отслеживанием энтропийных метрик через RLA-стек.
        
        Args:
            key: Ключ для поиска
            similarity_threshold: Порог схожести для поиска
            
        Returns:
            Значение из кэша или None
        """
        # Сохраняем состояние до вызова get() для определения cache hit/miss
        cache_hits_before = self.cache_hits
        
        value = self.get(key, similarity_threshold)
        
        # Отслеживание энтропийных метрик через RLA-стек
        if hasattr(self, '_rla_stack') and self._rla_stack:
            if value is not None:
                # Cache hit: инкрементируем memory_reads в RLA стеке
                # (чтение из кэша = read операция, не write)
                self._rla_stack.memory_reads += 1
                self._rla_stack.cache_hits += 1
                
                # Вычисление информационной энтропии
                info_entropy = self._rla_stack.compute_information_entropy(value)
                
                # Термодинамическая энтропия (чтение из кэша = 0 операций стирания)
                thermo_entropy = self._rla_stack.compute_thermodynamic_entropy(0)
                
                # Логирование (если кэш попал - мемоизация работает)
                self._rla_stack.log_entropy_decision(
                    "kv_cache_get",
                    info_entropy,
                    thermo_entropy,
                    memoized=True  # Кэш попадание = мемоизация
                )
            else:
                # Cache miss: инкрементируем cache_misses в RLA стеке
                self._rla_stack.cache_misses += 1
        
        return value
    
    def put_with_rla(self, key: str, value: np.ndarray, frequency: float = 1.0):
        """
        Добавление значения в кэш с отслеживанием энтропийных метрик через RLA-стек.
        
        Args:
            key: Ключ
            value: Значение
            frequency: Начальная частота обращения
        """
        # Проверка, нужно ли перезаписывать (через RLA-стек)
        if hasattr(self, '_rla_stack') and self._rla_stack:
            # Проверка на перезапись
            existing = self.get(key)
            was_memoized = False
            
            if existing is not None:
                # Перезапись необходима - используем RLA-стек для отслеживания
                was_memoized = self._rla_stack.memoize(key, value)
                
                # Логирование
                info_entropy = self._rla_stack.compute_information_entropy(value)
                thermo_entropy = self._rla_stack.compute_thermodynamic_entropy(1 if not was_memoized else 0)
                
                self._rla_stack.log_entropy_decision(
                    "kv_cache_put",
                    info_entropy,
                    thermo_entropy,
                    memoized=was_memoized
                )
            else:
                # Первая запись - используем RLA-стек
                was_memoized = self._rla_stack.memoize(key, value)
                
                # Логирование
                info_entropy = self._rla_stack.compute_information_entropy(value)
                thermo_entropy = self._rla_stack.compute_thermodynamic_entropy(1 if not was_memoized else 0)
                
                self._rla_stack.log_entropy_decision(
                    "kv_cache_put",
                    info_entropy,
                    thermo_entropy,
                    memoized=was_memoized
                )
        
        # Добавление в кэш
        self.put(key, value, frequency)

