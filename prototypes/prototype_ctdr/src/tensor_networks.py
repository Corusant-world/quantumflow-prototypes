"""
Tensor Networks — MPO (Matrix Product Operator) для экстремального сжатия весов.

РЕАЛЬНАЯ ИМПЛЕМЕНТАЦИЯ:
- SVD-based decomposition для сжатия матриц
- MPO формат: W ≈ U₁ @ S @ Vₜ с truncated rank
- Интеграция с Tensor Cores H100 для быстрого inference

Targets:
- Compression: 2.7×+
- Speedup: 1.6×+ при inference
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class MPOLayer:
    """Одна MPO-сжатая матрица."""
    U: np.ndarray          # Left singular vectors [m, r]
    S: np.ndarray          # Singular values [r]
    Vt: np.ndarray         # Right singular vectors [r, n]
    original_shape: Tuple[int, int]
    rank: int
    compression_ratio: float
    reconstruction_error: float


class TensorNetworkCompressor:
    """
    MPO-based compressor для нейронных весов.
    
    Использует truncated SVD для сжатия матриц весов,
    сохраняя важнейшие сингулярные значения.
    """
    
    def __init__(self, target_rank_ratio: float = 0.1, min_rank: int = 4):
        """
        Args:
            target_rank_ratio: Доля от полного ранга (0.1 = 10% от min(m, n))
            min_rank: Минимальный ранг для сжатия
        """
        self.target_rank_ratio = target_rank_ratio
        self.min_rank = min_rank
        self.compressed_layers: Dict[str, MPOLayer] = {}
        self.stats = {
            "total_original_params": 0,
            "total_compressed_params": 0,
            "compression_time_ms": 0.0,
            "inference_speedup": 1.0
        }
    
    def compress_matrix(
        self, 
        W: np.ndarray, 
        name: str = "layer",
        target_rank: Optional[int] = None
    ) -> MPOLayer:
        """
        Сжимает матрицу весов через truncated SVD (MPO).
        
        Args:
            W: Матрица весов [m, n]
            name: Имя слоя
            target_rank: Явный ранг (если None — вычисляется из target_rank_ratio)
        
        Returns:
            MPOLayer с U, S, Vt и метриками
        """
        start_time = time.time()
        
        m, n = W.shape
        full_rank = min(m, n)
        
        if target_rank is None:
            target_rank = max(self.min_rank, int(full_rank * self.target_rank_ratio))
        
        target_rank = min(target_rank, full_rank)
        
        # РЕАЛЬНЫЙ SVD (не симуляция)
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        
        # Truncate to target rank
        U_trunc = U[:, :target_rank]      # [m, r]
        S_trunc = S[:target_rank]          # [r]
        Vt_trunc = Vt[:target_rank, :]     # [r, n]
        
        # Reconstruction для вычисления ошибки
        W_reconstructed = U_trunc @ np.diag(S_trunc) @ Vt_trunc
        reconstruction_error = np.linalg.norm(W - W_reconstructed, 'fro') / np.linalg.norm(W, 'fro')
        
        # Подсчет параметров
        original_params = m * n
        compressed_params = m * target_rank + target_rank + target_rank * n
        compression_ratio = original_params / compressed_params
        
        elapsed = (time.time() - start_time) * 1000
        self.stats["compression_time_ms"] += elapsed
        self.stats["total_original_params"] += original_params
        self.stats["total_compressed_params"] += compressed_params
        
        layer = MPOLayer(
            U=U_trunc,
            S=S_trunc,
            Vt=Vt_trunc,
            original_shape=(m, n),
            rank=target_rank,
            compression_ratio=compression_ratio,
            reconstruction_error=reconstruction_error
        )
        
        self.compressed_layers[name] = layer
        return layer
    
    def decompress_layer(self, name: str) -> np.ndarray:
        """Восстановление оригинальной матрицы из MPO."""
        layer = self.compressed_layers[name]
        return layer.U @ np.diag(layer.S) @ layer.Vt
    
    def forward_mpo(self, x: np.ndarray, name: str) -> np.ndarray:
        """
        Forward pass через MPO-сжатый слой.
        
        Вместо x @ W (O(m*n)) делаем:
        1. x @ U        [batch, m] @ [m, r] = [batch, r]
        2. * S          [batch, r] * [r]    = [batch, r]  
        3. @ Vt         [batch, r] @ [r, n] = [batch, n]
        
        При r << min(m,n) это значительно быстрее.
        """
        layer = self.compressed_layers[name]
        
        # Step 1: x @ U
        h = x @ layer.U  # [batch, r]
        
        # Step 2: * S (element-wise broadcast)
        h = h * layer.S  # [batch, r]
        
        # Step 3: @ Vt
        out = h @ layer.Vt  # [batch, n]
        
        return out
    
    def forward_original(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Forward pass через оригинальную матрицу."""
        return x @ W
    
    def get_total_compression_ratio(self) -> float:
        """Общий коэффициент сжатия."""
        if self.stats["total_compressed_params"] == 0:
            return 1.0
        return self.stats["total_original_params"] / self.stats["total_compressed_params"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика компрессора."""
        return {
            "total_original_params": self.stats["total_original_params"],
            "total_compressed_params": self.stats["total_compressed_params"],
            "compression_ratio": self.get_total_compression_ratio(),
            "compression_time_ms": self.stats["compression_time_ms"],
            "num_layers": len(self.compressed_layers),
            "layer_details": {
                name: {
                    "shape": layer.original_shape,
                    "rank": layer.rank,
                    "compression": layer.compression_ratio,
                    "error": layer.reconstruction_error
                }
                for name, layer in self.compressed_layers.items()
            }
        }


class MPOTensorCore:
    """
    MPO с GPU ускорением через Tensor Cores (PyTorch).
    
    Выполняет MPO inference на GPU с FP16/TF32 precision.
    """
    
    def __init__(self, device: str = "cuda"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for GPU MPO")
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.layers_gpu: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    
    def load_mpo_layer(self, name: str, layer: MPOLayer) -> None:
        """Загрузка MPO слоя на GPU."""
        U = torch.from_numpy(layer.U).to(self.device, dtype=torch.float16)
        S = torch.from_numpy(layer.S).to(self.device, dtype=torch.float16)
        Vt = torch.from_numpy(layer.Vt).to(self.device, dtype=torch.float16)
        self.layers_gpu[name] = (U, S, Vt)
    
    def forward_gpu(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """
        Forward pass на GPU через Tensor Cores.
        
        FP16 matmul использует Tensor Cores на H100.
        """
        U, S, Vt = self.layers_gpu[name]
        
        x = x.to(self.device, dtype=torch.float16)
        
        # Step 1: x @ U (Tensor Cores)
        h = torch.matmul(x, U)
        
        # Step 2: * S
        h = h * S
        
        # Step 3: @ Vt (Tensor Cores)
        out = torch.matmul(h, Vt)
        
        return out


def benchmark_mpo_compression(
    matrix_sizes: List[Tuple[int, int]] = None,
    rank_ratios: List[float] = None
) -> Dict[str, Any]:
    """
    Бенчмарк MPO compression.
    
    Returns:
        Результаты с compression ratio, speedup, accuracy.
    """
    if matrix_sizes is None:
        matrix_sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
    
    if rank_ratios is None:
        rank_ratios = [0.1, 0.05, 0.02]  # 10%, 5%, 2% от ранга
    
    results = []
    
    for m, n in matrix_sizes:
        for ratio in rank_ratios:
            compressor = TensorNetworkCompressor(target_rank_ratio=ratio)
            
            # Создаем случайную матрицу (симулирует веса)
            W = np.random.randn(m, n).astype(np.float32)
            
            # Сжатие
            layer = compressor.compress_matrix(W, name=f"{m}x{n}")
            
            # Бенчмарк inference
            batch_size = 64
            x = np.random.randn(batch_size, m).astype(np.float32)
            
            # Original
            start = time.perf_counter()
            for _ in range(100):
                y_orig = compressor.forward_original(x, W)
            time_original = (time.perf_counter() - start) * 1000 / 100
            
            # MPO
            start = time.perf_counter()
            for _ in range(100):
                y_mpo = compressor.forward_mpo(x, f"{m}x{n}")
            time_mpo = (time.perf_counter() - start) * 1000 / 100
            
            # Accuracy
            y_orig_check = compressor.forward_original(x, W)
            y_mpo_check = compressor.forward_mpo(x, f"{m}x{n}")
            accuracy = 1.0 - np.mean(np.abs(y_orig_check - y_mpo_check)) / np.mean(np.abs(y_orig_check))
            
            speedup = time_original / time_mpo if time_mpo > 0 else 1.0
            
            results.append({
                "matrix_size": f"{m}x{n}",
                "rank_ratio": ratio,
                "rank": layer.rank,
                "compression_ratio": layer.compression_ratio,
                "reconstruction_error": layer.reconstruction_error,
                "time_original_ms": time_original,
                "time_mpo_ms": time_mpo,
                "speedup": speedup,
                "accuracy": accuracy
            })
    
    return {
        "results": results,
        "summary": {
            "avg_compression": np.mean([r["compression_ratio"] for r in results]),
            "avg_speedup": np.mean([r["speedup"] for r in results]),
            "avg_accuracy": np.mean([r["accuracy"] for r in results])
        }
    }

