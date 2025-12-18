"""
Dynamic Hierarchy Manager (DHM) — "Древо Смыслов".

DHM — это "файловая система" Когнитивной ОС. Организует все знания 
в самобалансирующееся p-адическое дерево.

КЛЮЧЕВАЯ ОПТИМИЗАЦИЯ: DPX Batch LCP (GPU)
- Используем dpx_lcp_index_load/query для batch поиска на GPU
- 1 query vs 1M candidates = параллельные threads на H100
- Warp-level first-mismatch kernel (128-bit chunks)
- Цель: миллионы концептов, <100ms retrieval
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import struct
import sys
import os

# Добавляем путь к ctdr_python (в корне prototype_ctdr/)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@dataclass
class DHMNode:
    """Узел в p-адическом дереве."""
    path: str
    content: Any
    children: Dict[str, 'DHMNode'] = field(default_factory=dict)
    depth: int = 0
    access_count: int = 0


def encode_to_short2_padded(text: str, max_len: int) -> bytes:
    """
    Кодирование строки в short2 формат с padding до max_len символов.
    
    Результат: max_len * 2 bytes (каждый символ = 2 bytes как uint16)
    """
    encoded = bytearray()
    for i in range(max_len):
        if i < len(text):
            val = ord(text[i])
        else:
            val = 0  # Padding
        encoded.extend(struct.pack('<H', val))  # uint16 little-endian
    return bytes(encoded)


class DynamicHierarchyManager:
    """
    Dynamic Hierarchy Manager — "Древо Смыслов".
    
    Использует DPX Batch LCP Kernel для O(N) параллельного поиска на GPU.
    
    GPU Functions (from ctdr_python):
    - dpx_lcp_index_load(candidates_bytes, num_candidates) — загрузка на GPU
    - dpx_lcp_index_query(query_bytes) — batch LCP, возвращает массив LCP
    - dpx_lcp_index_query_top1() — возвращает (best_idx, best_lcp) без копирования
    """
    
    PATH_SEPARATOR = " → "
    MAX_PATH_LEN = 128  # Символов (= 256 bytes в short2)
    
    def __init__(self, use_gpu: bool = True):
        """
        Инициализация DHM.
        
        Args:
            use_gpu: Использовать GPU (DPX) если доступен
        """
        self.root = DHMNode(path="", content=None, children={}, depth=0, access_count=0)
        self.size = 0
        
        # FLAT INDEX
        self._paths: List[str] = []
        self._contents: List[Any] = []
        self._path_to_idx: Dict[str, int] = {}
        
        # GPU state
        self._use_gpu = use_gpu
        self._gpu_available = False
        self._gpu_index_loaded = False
        self._check_gpu()
    
    def _check_gpu(self):
        """Проверка доступности GPU/DPX."""
        if not self._use_gpu:
            return
        
        try:
            import ctdr_python
            self._gpu_available = True
            self._ctdr = ctdr_python
        except ImportError:
            self._gpu_available = False
            self._ctdr = None
    
    def insert(self, concept: str, content: Any, path: Optional[str] = None) -> str:
        """
        Вставка концепта. O(1) amortized.
        """
        if path is None:
            path = concept
        
        idx = len(self._paths)
        self._paths.append(path)
        self._contents.append(content)
        self._path_to_idx[path] = idx
        self.size += 1
        
        # Помечаем что GPU index нужно перезагрузить
        self._gpu_index_loaded = False
        
        return path
    
    def _load_gpu_index(self):
        """
        Загрузка всех путей на GPU для batch search.
        
        Вызывается один раз после вставок, затем query работает быстро.
        """
        if not self._gpu_available or self._gpu_index_loaded:
            return
        
        if self.size == 0:
            return
        
        # Encode все пути в short2 формат
        # Каждый path → MAX_PATH_LEN * 2 bytes
        all_encoded = bytearray()
        for path in self._paths:
            encoded = encode_to_short2_padded(path, self.MAX_PATH_LEN)
            all_encoded.extend(encoded)
        
        # Загрузка на GPU
        try:
            self._ctdr.dpx_lcp_index_load(bytes(all_encoded), self.size)
            self._gpu_index_loaded = True
        except Exception as e:
            print(f"Warning: GPU index load failed: {e}")
            self._gpu_index_loaded = False
    
    def search(self, query: str, max_results: int = 10) -> List[Tuple[str, Any, float]]:
        """
        Поиск через DPX Batch LCP на GPU.
        
        GPU: dpx_lcp_index_query возвращает массив LCP для всех candidates
        CPU fallback: vectorized numpy (НЕ наш целевой путь)
        """
        if self.size == 0:
            return []
        
        if self._gpu_available:
            return self._search_gpu(query, max_results)
        else:
            return self._search_cpu(query, max_results)
    
    def _search_gpu(self, query: str, max_results: int) -> List[Tuple[str, Any, float]]:
        """
        GPU search using DPX batch kernel.
        
        Использует dpx_lcp_index_query для batch LCP.
        """
        # Загрузка index если нужно
        self._load_gpu_index()
        
        if not self._gpu_index_loaded:
            return self._search_cpu(query, max_results)
        
        # Encode query
        query_encoded = encode_to_short2_padded(query, self.MAX_PATH_LEN)
        
        try:
            # GPU batch LCP
            lcps = self._ctdr.dpx_lcp_index_query(query_encoded)
            lcps = np.array(lcps, dtype=np.int32)
            
            # P-adic similarity: 1.0 / (1.0 + 2^(-lcp))
            # lcps в единицах uint16 (= символы)
            distances = np.power(2.0, -lcps.astype(float))
            similarities = 1.0 / (1.0 + distances)
            
            # Exact matches
            for i, path in enumerate(self._paths):
                if path == query:
                    similarities[i] = 1.0
            
            # Top-K
            if max_results < len(similarities):
                top_indices = np.argpartition(similarities, -max_results)[-max_results:]
                top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            else:
                top_indices = np.argsort(similarities)[::-1]
            
            results = []
            for idx in top_indices[:max_results]:
                results.append((
                    self._paths[idx],
                    self._contents[idx],
                    float(similarities[idx])
                ))
            
            return results
            
        except Exception as e:
            print(f"Warning: GPU search failed: {e}, falling back to CPU")
            return self._search_cpu(query, max_results)
    
    def _search_cpu(self, query: str, max_results: int) -> List[Tuple[str, Any, float]]:
        """
        CPU fallback search.
        
        NOTE: This is NOT our target path. DPX GPU is the goal.
        Used only when GPU not available.
        """
        # Vectorized LCP computation
        n = len(self._paths)
        
        # Encode all paths
        encoded_paths = np.zeros((n, self.MAX_PATH_LEN), dtype=np.uint16)
        for i, path in enumerate(self._paths):
            for j, char in enumerate(path[:self.MAX_PATH_LEN]):
                encoded_paths[i, j] = ord(char)
        
        # Encode query
        query_encoded = np.zeros(self.MAX_PATH_LEN, dtype=np.uint16)
        for j, char in enumerate(query[:self.MAX_PATH_LEN]):
            query_encoded[j] = ord(char)
        
        # Vectorized comparison
        matches = (encoded_paths == query_encoded)
        cumsum = np.cumprod(matches, axis=1)
        lcps = cumsum.sum(axis=1)
        
        # P-adic similarity
        distances = np.power(2.0, -lcps.astype(float))
        similarities = 1.0 / (1.0 + distances)
        
        # Exact matches
        for i, path in enumerate(self._paths):
            if path == query:
                similarities[i] = 1.0
        
        # Top-K
        if max_results < len(similarities):
            top_indices = np.argpartition(similarities, -max_results)[-max_results:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        else:
            top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices[:max_results]:
            results.append((
                self._paths[idx],
                self._contents[idx],
                float(similarities[idx])
            ))
        
        return results
    
    def search_top1(self, query: str) -> Optional[Tuple[str, Any, float]]:
        """
        Быстрый top-1 search через GPU (без копирования всего массива).
        
        Использует dpx_lcp_index_query_top1.
        """
        if self.size == 0:
            return None
        
        if not self._gpu_available:
            results = self._search_cpu(query, 1)
            return results[0] if results else None
        
        # Загрузка index если нужно
        self._load_gpu_index()
        
        if not self._gpu_index_loaded:
            results = self._search_cpu(query, 1)
            return results[0] if results else None
        
        # Encode query и set на GPU
        query_encoded = encode_to_short2_padded(query, self.MAX_PATH_LEN)
        
        try:
            self._ctdr.dpx_lcp_index_set_query(query_encoded)
            best_idx, best_lcp = self._ctdr.dpx_lcp_index_query_top1()
            
            # P-adic similarity
            distance = 2.0 ** (-best_lcp)
            similarity = 1.0 / (1.0 + distance)
            
            # Exact match check
            if self._paths[best_idx] == query:
                similarity = 1.0
            
            return (self._paths[best_idx], self._contents[best_idx], float(similarity))
            
        except Exception as e:
            print(f"Warning: GPU top1 failed: {e}")
            results = self._search_cpu(query, 1)
            return results[0] if results else None
    
    def mental_saccade(self, from_path: str, to_level: int) -> Optional[str]:
        """
        Ментальная Саккада: O(1) прыжок между уровнями.
        """
        path_parts = from_path.split(self.PATH_SEPARATOR)
        
        if to_level < 0 or to_level >= len(path_parts):
            return None
        
        return self.PATH_SEPARATOR.join(path_parts[:to_level + 1])
    
    def archive(self, path: str, max_depth: int = 10):
        """Архивация (забывание через перемещение)."""
        if path not in self._path_to_idx:
            return
        
        idx = self._path_to_idx[path]
        old_path = self._paths[idx]
        
        depth = old_path.count(self.PATH_SEPARATOR) + 1
        if depth < max_depth:
            new_path = f"archive{self.PATH_SEPARATOR}{depth}{self.PATH_SEPARATOR}{old_path}"
            self._paths[idx] = new_path
            del self._path_to_idx[path]
            self._path_to_idx[new_path] = idx
            self._gpu_index_loaded = False  # Нужно перезагрузить index
    
    def get(self, path: str) -> Optional[Any]:
        """O(1) доступ по пути."""
        if path in self._path_to_idx:
            return self._contents[self._path_to_idx[path]]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика DHM."""
        max_depth = 0
        for path in self._paths:
            depth = path.count(self.PATH_SEPARATOR) + 1
            max_depth = max(max_depth, depth)
        
        return {
            "size": self.size,
            "max_depth": max_depth,
            "cache_size": len(self._path_to_idx),
            "gpu_available": self._gpu_available,
            "gpu_index_loaded": self._gpu_index_loaded,
            "max_path_len": self.MAX_PATH_LEN,
        }
    
    def clear(self):
        """Очистка."""
        self.root = DHMNode(path="", content=None, children={}, depth=0, access_count=0)
        self.size = 0
        self._paths.clear()
        self._contents.clear()
        self._path_to_idx.clear()
        self._gpu_index_loaded = False
        
        # Очистка GPU index
        if self._gpu_available and self._ctdr:
            try:
                self._ctdr.dpx_lcp_index_clear()
            except:
                pass


class DHMWithDPX(DynamicHierarchyManager):
    """
    DHM с принудительным использованием DPX.
    
    Выбрасывает ошибку если DPX недоступен.
    """
    
    def __init__(self):
        super().__init__(use_gpu=True)
        
        if not self._gpu_available:
            raise ImportError(
                "DPX GPU required for DHMWithDPX. "
                "Compile CUDA kernels first: cd cuda && make"
            )
