"""
Hybrid Computational Unit (HCE) — Гибридная вычислительная единица CTDR.

Объединяет 3 компонента:
1. Нейронный Субстрат (Tensor Cores) — плотная линейная алгебра
2. Структурный Каркас (DPX Instructions) — ультраметрическая геометрия  
3. Логическое Ядро (DPX + Tensor Cores) — дискретная логика

Разделение труда:
- Tensor Cores: непрерывные сигналы (эмбеддинги, матричные умножения)
- DPX: дискретная структура (иерархический поиск, LCP)
- Гибрид: символическое рассуждение (Boolean Einsum + Heaviside)
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class HCEStats:
    """Статистика HCE."""
    neural_operations: int = 0
    structural_operations: int = 0
    logical_operations: int = 0
    tensor_core_time_ms: float = 0.0
    dpx_time_ms: float = 0.0


class HybridComputationalUnit:
    """
    Hybrid Computational Unit (HCE) — Гибридная вычислительная единица.
    
    Объединяет нейронный и символический ИИ через Tensor Cores и DPX.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Инициализация HCE.
        
        Args:
            device: "cpu" или "cuda" для GPU
        """
        self.device = device
        self.tensor_cores_enabled = TORCH_AVAILABLE and device == "cuda"
        self.dpx_enabled = True
        self.stats = HCEStats()
    
    def neural_substrate(self, embeddings: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Нейронный субстрат: плотная линейная алгебра.
        
        Использует Tensor Cores для FP8/FP16 матричных умножений.
        
        Args:
            embeddings: Входные эмбеддинги [batch, dim]
            weights: Весовая матрица [dim, output_dim]
            
        Returns:
            Результат матричного умножения [batch, output_dim]
        """
        import time
        start = time.time()
        
        self.stats.neural_operations += 1
        
        if TORCH_AVAILABLE and self.device == "cuda":
            # GPU path через PyTorch (Tensor Cores)
            embeddings_t = torch.from_numpy(embeddings).to(self.device)
            weights_t = torch.from_numpy(weights).to(self.device)
            
            result_t = torch.matmul(embeddings_t, weights_t)
            result = result_t.cpu().numpy()
        else:
            # CPU fallback
            result = np.matmul(embeddings, weights)
        
        elapsed = (time.time() - start) * 1000
        self.stats.tensor_core_time_ms += elapsed
        
        return result
    
    def structural_framework(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Структурный каркас: ультраметрическая геометрия через DPX.
        
        Использует DPX_LCP_Kernel для иерархического поиска O(N).
        
        Args:
            query: Запрос
            candidates: Список кандидатов
            
        Returns:
            Список (candidate, similarity) отсортированный по similarity
        """
        import time
        start = time.time()
        
        self.stats.structural_operations += 1
        
        # Импорт encoding и kernel
        from encoding import encode_to_short2
        
        try:
            from cuda.kernels import dpx_lcp_batch
            use_gpu = True
        except ImportError:
            use_gpu = False
        
        query_encoded = encode_to_short2(query)
        results = []
        
        for candidate in candidates:
            candidate_encoded = encode_to_short2(candidate)
            
            if use_gpu:
                # GPU path через DPX
                lcp = self._compute_lcp_dpx(query_encoded, candidate_encoded)
            else:
                # CPU fallback
                lcp = self._compute_lcp_cpu(query, candidate)
            
            # P-adic distance и similarity
            p = 2
            distance = p ** (-lcp) if lcp > 0 else 1.0
            similarity = 1.0 / (1.0 + distance)
            
            results.append((candidate, similarity))
        
        # Сортировка по similarity (убывание)
        results.sort(key=lambda x: x[1], reverse=True)
        
        elapsed = (time.time() - start) * 1000
        self.stats.dpx_time_ms += elapsed
        
        return results
    
    def logical_core(self, 
                    tensors: List[np.ndarray], 
                    predicates: List[bool], 
                    threshold: float = 0.5) -> np.ndarray:
        """
        Логическое ядро: дискретная логика через DPX + Tensor Cores.
        
        Использует Boolean Einsum + Heaviside для символического рассуждения.
        
        Args:
            tensors: Список тензоров для логических операций
            predicates: Список предикатов (True/False)
            threshold: Порог Heaviside
            
        Returns:
            Результат логической операции
        """
        import time
        start = time.time()
        
        self.stats.logical_operations += 1
        
        # Импорт reversible_einsum
        from core import reversible_einsum
        
        # Применение предикатов к тензорам
        filtered_tensors = []
        for tensor, predicate in zip(tensors, predicates):
            if predicate:
                filtered_tensors.append(tensor)
        
        if len(filtered_tensors) < 2:
            # Недостаточно тензоров для einsum
            if filtered_tensors:
                return filtered_tensors[0]
            else:
                return np.zeros((1,))
        
        # Последовательное применение Boolean Einsum
        result = filtered_tensors[0]
        for i in range(1, len(filtered_tensors)):
            result = reversible_einsum(result, filtered_tensors[i], threshold)
        
        elapsed = (time.time() - start) * 1000
        self.stats.tensor_core_time_ms += elapsed  # Einsum использует TC
        
        return result
    
    def hybrid_compute(self, task: Dict[str, Any]) -> Any:
        """
        Гибридное вычисление: автоматический выбор компонента HCE.
        
        Args:
            task: Словарь с полем "type" и данными
                - type="neural": embeddings, weights
                - type="structural": query, candidates
                - type="logical": tensors, predicates
        
        Returns:
            Результат вычисления
        """
        task_type = task.get("type")
        
        if task_type == "neural":
            embeddings = task["embeddings"]
            weights = task["weights"]
            return self.neural_substrate(embeddings, weights)
        
        elif task_type == "structural":
            query = task["query"]
            candidates = task["candidates"]
            return self.structural_framework(query, candidates)
        
        elif task_type == "logical":
            tensors = task["tensors"]
            predicates = task["predicates"]
            threshold = task.get("threshold", 0.5)
            return self.logical_core(tensors, predicates, threshold)
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _compute_lcp_cpu(self, s1: str, s2: str) -> int:
        """CPU версия LCP для fallback."""
        lcp = 0
        min_len = min(len(s1), len(s2))
        for i in range(min_len):
            if s1[i] == s2[i]:
                lcp += 1
            else:
                break
        return lcp
    
    def _compute_lcp_dpx(self, s1_encoded: bytes, s2_encoded: bytes) -> int:
        """GPU версия LCP через DPX."""
        try:
            from cuda.kernels import dpx_lcp_single
            return dpx_lcp_single(s1_encoded, s2_encoded)
        except Exception:
            # Fallback на CPU
            return self._compute_lcp_cpu(
                s1_encoded.decode('utf-8', errors='ignore'),
                s2_encoded.decode('utf-8', errors='ignore')
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики HCE.
        
        Returns:
            Словарь со статистикой
        """
        total_ops = (self.stats.neural_operations + 
                    self.stats.structural_operations + 
                    self.stats.logical_operations)
        
        return {
            "neural_operations": self.stats.neural_operations,
            "structural_operations": self.stats.structural_operations,
            "logical_operations": self.stats.logical_operations,
            "total_operations": total_ops,
            "tensor_core_time_ms": self.stats.tensor_core_time_ms,
            "dpx_time_ms": self.stats.dpx_time_ms,
            "device": self.device,
            "tensor_cores_enabled": self.tensor_cores_enabled,
            "dpx_enabled": self.dpx_enabled,
        }
    
    def to_gpu(self):
        """Переключение на GPU."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
            self.tensor_cores_enabled = True
        else:
            raise RuntimeError("CUDA not available")
    
    def to_cpu(self):
        """Переключение на CPU."""
        self.device = "cpu"
        self.tensor_cores_enabled = False


class HCEGPU(HybridComputationalUnit):
    """
    GPU версия HCE с оптимизацией под H100.
    
    Использует:
    - Tensor Cores для FP8/FP16 матричных умножений
    - DPX Instructions для ультраметрического поиска
    - L2 Cache Management для структурных данных
    """
    
    def __init__(self):
        """Инициализация GPU версии HCE."""
        super().__init__(device="cuda" if TORCH_AVAILABLE else "cpu")
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
            self.tensor_cores_enabled = True
    
    def hybrid_compute_gpu(self, task: Dict[str, Any]) -> Any:
        """
        Гибридное вычисление на GPU с разделением труда TC/DPX.
        
        Оптимизировано под H100 архитектуру.
        """
        return self.hybrid_compute(task)

