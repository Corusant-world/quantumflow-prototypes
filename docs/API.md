# API Reference

Complete API documentation for GPU utilization tools. These prototypes are designed to achieve **95%+ GPU utilization** on NVIDIA H100 through Tensor Core acceleration and cuQuantum integration.

## Table of Contents

- [Installation](#installation)
- [Team1 Quantum](#team1-quantum-vacancystability-hunter)
- [Team2 Energy](#team2-energy-grid--thermo-kernel)
- [Team3 Innovation](#team3-innovation-quantum-neural-process--cuquantum)
- [Ecosystem Smoke Test](#ecosystem-smoke-test)
- [Usage Examples](#usage-examples)

## Installation

```bash
# CPU (no CUDA required)
python -m pip install -r prototypes/requirements.cpu.txt

# NVIDIA GPU (CUDA 12, without cuQuantum)
python -m pip install -r prototypes/requirements.gpu-cu12.txt

# NVIDIA GPU (CUDA 12) + cuQuantum (Team3 acceleration)
python -m pip install -r prototypes/requirements.gpu-cu12-cuquantum.txt
```

## Team1 Quantum — Vacancy/Stability Hunter

GPU-heavy screening workload that drives Tensor Cores to achieve 95%+ utilization.

### Classes

#### `QuantumVacancyHunter`

GPU-heavy (when CUDA is available) vacancy/stability search surrogate. Uses repeated matrix multiplications with FP16 option to hit Tensor Cores.

**Constructor:**

```python
QuantumVacancyHunter(
    lattice_size: int = 128,
    use_fp16: bool = False,
    device: Optional[torch.device] = None
)
```

**Parameters:**
- `lattice_size` (int): Size of the lattice (must be > 0). Default: 128
- `use_fp16` (bool): Use FP16 precision for Tensor Core acceleration. Default: False
- `device` (Optional[torch.device]): Target device. If None, uses CUDA if available, else CPU

**Properties:**
- `device` (torch.device): Current device (cuda or cpu)
- `dtype` (torch.dtype): Current dtype (float16 if use_fp16 and cuda, else float32)
- `hamiltonian_d` (int): Dimension of the Hamiltonian matrix

**Methods:**

##### `generate_lattice_batch(batch_size: int) -> torch.Tensor`

Generate a batch of lattice states.

**Parameters:**
- `batch_size` (int): Number of lattice states to generate (must be > 0)

**Returns:**
- `torch.Tensor`: Batch of lattice states, shape `(batch_size, lattice_size)`, on `self.device` with `self.dtype`

**Example:**
```python
hunter = QuantumVacancyHunter(lattice_size=256, use_fp16=True)
batch = hunter.generate_lattice_batch(batch_size=128)
# batch.shape == (128, 256), dtype == torch.float16 (if CUDA)
```

##### `find_stable_vacancies(batch_size: int = 128, steps: int = 10) -> QuantumMetrics`

Find stable vacancies through iterative computation. This is the main workload that drives GPU utilization.

**Parameters:**
- `batch_size` (int): Number of lattice states to process (must be > 0). Default: 128
- `steps` (int): Number of iteration steps (must be > 0). Default: 10

**Returns:**
- `QuantumMetrics`: Dataclass containing:
  - `compute_time_ms` (float): Computation time in milliseconds
  - `fidelity` (float): Fidelity metric in (0, 1]
  - `active_qubits` (int): Number of active qubits
  - `details` (Dict[str, Any]): Additional metadata (device, dtype, lattice_size, batch_size, steps)

**Example:**
```python
hunter = QuantumVacancyHunter(lattice_size=128, use_fp16=True, device=torch.device("cuda"))
metrics = hunter.find_stable_vacancies(batch_size=256, steps=20)
print(f"Compute time: {metrics.compute_time_ms} ms")
print(f"Fidelity: {metrics.fidelity}")
print(f"Active qubits: {metrics.active_qubits}")
```

### Functions

#### `process(input_data: Any = None, device_override: Optional[torch.device] = None) -> Dict[str, Any]`

Back-compatibility entrypoint for demos and legacy code.

**Parameters:**
- `input_data` (Any): Optional dict with keys:
  - `lattice_size` (int): Default 128
  - `batch_size` (int): Default 128
  - `steps` (int): Default 10
  - `use_fp16` (bool): Default True
- `device_override` (Optional[torch.device]): Target device. If None, auto-detects CUDA/CPU

**Returns:**
- `Dict[str, Any]`: Result dictionary with:
  - `device` (str): Device used
  - `dtype` (str): Dtype used
  - `result_shape` (List[int]): Shape of result `[batch_size, lattice_size]`
  - `metrics` (Dict): Metrics from `find_stable_vacancies()`

**Example:**
```python
result = process({
    "lattice_size": 256,
    "batch_size": 512,
    "steps": 50,
    "use_fp16": True
}, device_override=torch.device("cuda"))
```

#### `create_bio_quantum_engine() -> Dict[str, Any]`

Tiny 'bio-quantum' primitive used by demos. Exposes a Hadamard-like amplitude transform.

**Returns:**
- `Dict[str, Any]`: Dictionary with:
  - `quantum_amp` (Callable): Function that applies Hadamard-like transform
  - `hadamard` (torch.Tensor): 2x2 Hadamard matrix

**Example:**
```python
engine = create_bio_quantum_engine()
psi = torch.tensor([1.0, 0.0])
transformed = engine["quantum_amp"](psi)
```

### Data Classes

#### `QuantumMetrics`

Result dataclass from `find_stable_vacancies()`.

**Fields:**
- `compute_time_ms` (float): Computation time in milliseconds
- `fidelity` (float): Fidelity metric in (0, 1]
- `active_qubits` (int): Number of active qubits
- `details` (Dict[str, Any]): Additional metadata

## Team2 Energy — Grid + Thermo Kernel

Differentiable thermo workload with grid optimization, plus Tensor Core soak for measurable utilization.

### Classes

#### `EnergyGridSystem`

GPU-friendly physics-style loss that can run on CUDA when available. Performs topology optimization.

**Constructor:**

```python
EnergyGridSystem(num_nodes: int = 100, seed: int = 1337)
```

**Parameters:**
- `num_nodes` (int): Number of nodes in the grid (must be > 1). Default: 100
- `seed` (int): Random seed. Default: 1337

**Properties:**
- `device` (torch.device): Current device (cuda or cpu)
- `dtype` (torch.dtype): Current dtype (float16 on CUDA, float32 on CPU)
- `num_nodes` (int): Number of nodes

**Methods:**

##### `optimize_topology(steps: int = 50, lr: float = 5e-4) -> None`

Optimize conductances to reduce loss. Keeps symmetric & non-negative.

**Parameters:**
- `steps` (int): Number of optimization steps (must be > 0). Default: 50
- `lr` (float): Learning rate. Default: 5e-4

**Example:**
```python
grid = EnergyGridSystem(num_nodes=200)
grid.optimize_topology(steps=100, lr=1e-3)
```

##### `get_metrics() -> Dict[str, float]`

Get current metrics.

**Returns:**
- `Dict[str, float]`: Dictionary with:
  - `total_loss_mw` (float): Total loss in MW (Joule heating proxy)

**Example:**
```python
metrics = grid.get_metrics()
print(f"Total loss: {metrics['total_loss_mw']} MW")
```

##### `step_simulation() -> None`

Perform one step of stochastic dynamics simulation.

**Example:**
```python
for _ in range(100):
    grid.step_simulation()
```

##### `validation_check() -> Tuple[bool, str]`

Validate current state.

**Returns:**
- `Tuple[bool, str]`: (is_valid, message)

#### `TNP_System`

Differentiable thermo workload (gradients + steps) plus Tensor Core soak for measurable utilization.

**Constructor:**

```python
TNP_System(batch_size: int = 1024, steps: int = 100, seed: int = 4242)
```

**Parameters:**
- `batch_size` (int): Batch size for thermo simulation (must be > 0). Default: 1024
- `steps` (int): Number of simulation steps (must be > 0). Default: 100
- `seed` (int): Random seed. Default: 4242

**Methods:**

##### `simulate_thermo(batch_size: int = 1024, steps: int = 100) -> Dict[str, float]`

Run thermo simulation with GPU-accelerated gradients.

**Parameters:**
- `batch_size` (int): Batch size (must be > 0). Default: 1024
- `steps` (int): Number of steps (must be > 0). Default: 100

**Returns:**
- `Dict[str, float]`: Dictionary with:
  - `efficiency` (float): Thermo efficiency metric
  - `final_temp` (float): Final temperature
  - Other simulation metrics

**Example:**
```python
tnp = TNP_System()
result = tnp.simulate_thermo(batch_size=2048, steps=200)
print(f"Efficiency: {result['efficiency']}")
```

### Constants

#### `HAS_GPU: bool`

Boolean indicating if CUDA is available.

**Example:**
```python
from team2_energy.src.core import HAS_GPU
if HAS_GPU:
    print("GPU available")
```

### Functions

#### `process(input_data: Dict[str, Any], device_override: Optional[torch.device] = None) -> Dict[str, Any]`

Back-compatibility entrypoint.

**Parameters:**
- `input_data` (Dict[str, Any]): Input configuration
- `device_override` (Optional[torch.device]): Target device

**Returns:**
- `Dict[str, Any]`: Result dictionary

## Team3 Innovation — Quantum Neural Process + cuQuantum

Quantum-inspired feature processor with real cuQuantum integration (if cuQuantum installed).

### Functions

#### `quantum_neural_process(inp: torch.Tensor, qubits: int = 8, out_dim: int = 32) -> Tuple[torch.Tensor, Dict[str, float]]`

Fast quantum-inspired block (CPU or CUDA). Processes input tensor through quantum-inspired transformation.

**Parameters:**
- `inp` (torch.Tensor): Input tensor, shape `[B, D]` where B is batch size, D is feature dimension
- `qubits` (int): Number of qubits for quantum-inspired processing. Default: 8
- `out_dim` (int): Output dimension (must be 32 for contract). Default: 32

**Returns:**
- `Tuple[torch.Tensor, Dict[str, float]]`: 
  - Output tensor, shape `[B, 32]`
  - Metrics dictionary with:
    - `time_sec` (float): Computation time in seconds
    - `batch` (float): Batch size
    - `in_dim` (float): Input dimension
    - `qubits` (float): Number of qubits used

**Raises:**
- `ValueError`: If `inp` is not rank-2 tensor

**Example:**
```python
from team3_innovation.src.core import quantum_neural_process
import torch

x = torch.randn(64, 128, device="cuda")  # [batch=64, features=128]
out, metrics = quantum_neural_process(x, qubits=12)

print(f"Output shape: {out.shape}")  # (64, 32)
print(f"Computation time: {metrics['time_sec']} seconds")
```

#### `measure_tflops(n: int = 8192, iters: int = 40, dtype: str = "bf16") -> Dict[str, Any]`

Simple GEMM proxy to stress Tensor Cores. Used by benchmarks to push GPU utilization.

**Parameters:**
- `n` (int): Matrix dimension (n x n). Default: 8192
- `iters` (int): Number of iterations. Default: 40
- `dtype` (str): Data type: "bf16", "fp16", or "fp32". Default: "bf16"

**Returns:**
- `Dict[str, Any]`: Dictionary with:
  - `device` (str): "cuda" or "cpu"
  - `tflops` (float): Estimated TFLOPS (0.0 on CPU)
  - `dtype` (str): Data type used
  - `n` (float): Matrix dimension

**Example:**
```python
from team3_innovation.src.core import measure_tflops

# Measure BF16 Tensor Core performance
result = measure_tflops(n=16384, iters=100, dtype="bf16")
print(f"TFLOPS: {result['tflops']}")
```

#### `process(input_data: Optional[dict] = None, device: Optional[torch.device] = None) -> Dict[str, Any]`

Back-compatibility entrypoint.

**Parameters:**
- `input_data` (Optional[dict]): Optional dict with keys:
  - `batch` (int): Batch size. Default: 256
  - `in_dim` (int): Input dimension. Default: 256
  - `qubits` (int): Number of qubits. Default: 8
- `device` (Optional[torch.device]): Target device. If None, auto-detects CUDA/CPU

**Returns:**
- `Dict[str, Any]`: Dictionary with:
  - `out_shape` (List[int]): Output shape
  - `out_mean` (float): Mean of output
  - `metrics` (Dict): Metrics from `quantum_neural_process()`
  - `cuq_available` (bool): Whether cuQuantum is available

**Example:**
```python
result = process({
    "batch": 512,
    "in_dim": 512,
    "qubits": 16
}, device=torch.device("cuda"))
```

### Constants

#### `CUQ_AVAILABLE: bool`

Boolean indicating if cuQuantum is available (requires cuQuantum installation).

**Example:**
```python
from team3_innovation.src.core import CUQ_AVAILABLE
if CUQ_AVAILABLE:
    print("cuQuantum available")
```

## Ecosystem Smoke Test

One-command compatibility proof that all 3 prototypes can run together in one environment.

### Usage

```bash
# Auto-detect device (CUDA if available, else CPU)
python prototypes/ecosystem_smoke.py

# Force CPU
DEVICE=cpu python prototypes/ecosystem_smoke.py

# Force CUDA
DEVICE=cuda python prototypes/ecosystem_smoke.py
```

### Output

Writes `prototypes/ecosystem_results/latest.json` with:
- `timestamp` (str): ISO timestamp
- `device` (str): Device used ("cuda" or "cpu")
- `teams` (Dict): Status for each team:
  - `team1_quantum`: Status and device
  - `team2_energy`: Status and device
  - `team3_innovation`: Status, device, and `cuquantum_available` flag

## Usage Examples

### Example 1: GPU Utilization Validation

```python
from team1_quantum.src.core import QuantumVacancyHunter
import torch

# Create GPU workload
hunter = QuantumVacancyHunter(lattice_size=256, use_fp16=True, device=torch.device("cuda"))

# Run sustained workload (drives Tensor Cores)
metrics = hunter.find_stable_vacancies(batch_size=512, steps=100)

print(f"GPU utilization target: 95%+")
print(f"Compute time: {metrics.compute_time_ms} ms")
print(f"Fidelity: {metrics.fidelity}")
```

### Example 2: Tensor Core Performance Measurement

```python
from team3_innovation.src.core import measure_tflops

# Measure BF16 Tensor Core performance
result = measure_tflops(n=16384, iters=200, dtype="bf16")
print(f"Tensor Core TFLOPS: {result['tflops']}")
```

### Example 3: Quantum-Inspired Processing

```python
from team3_innovation.src.core import quantum_neural_process
import torch

# Process quantum-inspired features
x = torch.randn(128, 256, device="cuda")
out, metrics = quantum_neural_process(x, qubits=16)

print(f"Output shape: {out.shape}")
print(f"Processing time: {metrics['time_sec']} seconds")
```

### Example 4: Energy Grid Optimization

```python
from team2_energy.src.core import EnergyGridSystem

# Create and optimize grid
grid = EnergyGridSystem(num_nodes=200)
grid.optimize_topology(steps=100, lr=1e-3)

# Get metrics
metrics = grid.get_metrics()
print(f"Total loss: {metrics['total_loss_mw']} MW")
```

### Example 5: Thermo Simulation

```python
from team2_energy.src.core import TNP_System

# Run thermo simulation
tnp = TNP_System(batch_size=2048, steps=200)
result = tnp.simulate_thermo()

print(f"Efficiency: {result['efficiency']}")
print(f"Final temperature: {result['final_temp']}")
```

## GPU Utilization Best Practices

1. **Use FP16/BF16 for Tensor Cores**: Enable `use_fp16=True` in Team1, use BF16 in Team3
2. **Sustained workloads**: Run benchmarks for 60+ seconds to maintain high utilization
3. **Large batch sizes**: Use `batch_size >= 128` for better GPU utilization
4. **Monitor utilization**: Check `nvml.gpu_util_avg` in benchmark results (target: ≥85%)
5. **cuQuantum for Team3**: Install cuQuantum for real quantum circuit acceleration

## See Also

- **Real GPU utilization guide**: `docs/REAL_GPU_UTILIZATION.md`
- **Benchmark methodology**: `docs/BENCHMARKS.md`
- **NVIDIA integration notes**: `docs/NVIDIA_INTEGRATION.md`
