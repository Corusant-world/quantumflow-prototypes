# file: prototypes/team1_quantum/tests/test_core.py
import pytest
import torch
from src.core import QuantumVacancyHunter

def test_initialization():
    hunter = QuantumVacancyHunter(lattice_size=100)
    assert hunter.device.type in ["cuda", "cpu"]
    assert hunter.hamiltonian_d > 0

def test_simulation_run():
    hunter = QuantumVacancyHunter(lattice_size=128)
    batch_size = 128
    metrics = hunter.find_stable_vacancies(batch_size=batch_size)
    
    assert metrics.compute_time_ms > 0
    assert 0 <= metrics.fidelity <= 1.0
    assert metrics.active_qubits <= (128 * batch_size)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA")
def test_gpu_allocation():
    hunter = QuantumVacancyHunter(lattice_size=1024, use_fp16=True)
    assert hunter.dtype == torch.float16
    
    # Check memory is actually on GPU
    tensor = hunter.generate_lattice_batch(10)
    assert tensor.is_cuda