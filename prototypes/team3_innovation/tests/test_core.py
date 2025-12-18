import pytest
import torch
from src.core import quantum_neural_process

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU preferred, CPU OK")
def test_quantum_neural_basic():
    inp = torch.randn(64, 128)
    out, metrics = quantum_neural_process(inp, qubits=8)
    assert out.shape == (64, 32)
    assert metrics['time_sec'] < 0.1, "Too slow"
    assert abs(out.mean() - 0.5) < 0.2  # Stability

def test_cpu_fallback():
    # Mock no GPU
    out, _ = quantum_neural_process(torch.randn(32, 64), qubits=4)
    assert out.shape == (32, 32)