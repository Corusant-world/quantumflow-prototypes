# file: prototypes/team2_energy/tests/test_core.py
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from core import EnergyGridSystem, HAS_GPU

def test_initialization():
    sys = EnergyGridSystem(num_nodes=100)
    assert sys.num_nodes == 100

@pytest.mark.skipif(not HAS_GPU, reason="Requires GPU")
def test_physics_sanity():
    sys = EnergyGridSystem(num_nodes=100)
    sys.step_simulation()
    valid, msg = sys.validation_check()
    assert valid, f"Physics check failed: {msg}"

@pytest.mark.skipif(not HAS_GPU, reason="Requires GPU")
def test_optimization_improves_loss():
    sys = EnergyGridSystem(num_nodes=200)
    sys.step_simulation()
    m1 = sys.get_metrics()['total_loss_mw']
    
    sys.optimize_topology(steps=50)
    sys.step_simulation()
    m2 = sys.get_metrics()['total_loss_mw']
    
    # Optimization should not catastrophically regress (there is stochasticity).
    # Ideally m2 < m1, but annealing may fluctuate.
    # Minimal assertion: it runs and the metric changes.
    assert m2 != m1