# Dependencies Verification

This document verifies that all dependencies required by the prototypes are installable and functional.

## Verification Status

### CPU Requirements (`prototypes/requirements.cpu.txt`)

- ✅ **numpy>=1.24.0** — Standard scientific computing library
- ✅ **torch>=2.0.0** — PyTorch (CPU build)
- ✅ **pytest>=7.0.0** — Testing framework
- ✅ **nvidia-ml-py>=12.0.0** — NVML Python bindings (harmless on CPU, provides `pynvml` import)

**Verification:** All packages install via `pip install -r prototypes/requirements.cpu.txt`

### GPU Requirements (`prototypes/requirements.gpu-cu12.txt`)

Includes all CPU requirements plus:

- ✅ **cupy-cuda12x>=12.0.0** — CuPy for CUDA 12.x (used by Team3 and optional GPU probes)

**Verification:** 
- Installs from PyPI
- Requires CUDA 12.x runtime
- Provides `cupy` module for GPU array operations

### GPU + cuQuantum Requirements (`prototypes/requirements.gpu-cu12-cuquantum.txt`)

Includes all GPU requirements plus:

- ✅ **cuquantum-python>=25.0.0** — cuQuantum Python bindings from NVIDIA NGC index

**Verification:**
- Installs from NGC PyPI index: `--extra-index-url https://pypi.ngc.nvidia.com`
- Provides `cuquantum` module
- Imports: `from cuquantum import cutensornet, custatevec` work correctly
- Version tested: `25.03.0.post0` (may vary)

## Known Issues / Pitfalls

### NVML Import

- **Do NOT install** deprecated `pynvml` pip package directly
- Use `nvidia-ml-py` instead (it provides the `pynvml` import path)
- Installing `pynvml` directly can cause import-hook conflicts

### cuQuantum Shadowing

If you see:
```
cuquantum.__file__ = None
cuquantum loaded via _NamespaceLoader
```

This indicates `cuquantum` is being shadowed by a folder in `dist-packages/`. Remove the shadowing folder and keep only the pip-installed bindings.

## Installation Commands

### CPU (no CUDA required)
```bash
python -m pip install -U pip
python -m pip install -r prototypes/requirements.cpu.txt
```

### GPU (CUDA 12, without cuQuantum)
```bash
python -m pip install -U pip
python -m pip install -r prototypes/requirements.gpu-cu12.txt
```

### GPU (CUDA 12) + cuQuantum
```bash
python -m pip install -U pip
python -m pip install -r prototypes/requirements.gpu-cu12-cuquantum.txt
```

## Verification Test

After installation, verify dependencies:

```bash
# CPU
python -c "import torch, numpy, pytest, pynvml; print('CPU deps OK')"

# GPU
python -c "import torch, cupy; assert torch.cuda.is_available(); print('GPU deps OK')"

# GPU + cuQuantum
python -c "from cuquantum import cutensornet, custatevec; print('cuQuantum OK')"
```

## Last Verified

- Date: 2025-12-14
- Python: 3.10.12
- Platform: Linux (Ubuntu 22.04)
- CUDA: 12.4
- PyTorch: 2.6.0+cu124
- CuPy: 13.6.0
- cuQuantum: 25.03.0.post0
