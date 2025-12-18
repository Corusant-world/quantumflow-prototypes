## Goal

Make sure **team1_quantum + team2_energy + team3_innovation**:
- install into **one** Python environment
- import together (no dependency conflicts)
- have one smoke test
- run on CPU fallback and on NVIDIA GPUs

## One-command smoke test

```bash
python prototypes/ecosystem_smoke.py
```

Force mode:
- CPU: `DEVICE=cpu python prototypes/ecosystem_smoke.py`
- CUDA: `DEVICE=cuda python prototypes/ecosystem_smoke.py`

Artifact: `prototypes/ecosystem_results/latest.json`

Note: `ecosystem_smoke.py` adds `prototypes/` to `sys.path`, so `PYTHONPATH=prototypes` is not required.

## Installation

CPU:

```bash
python -m pip install -U pip
python -m pip install -r prototypes/requirements.cpu.txt
```

GPU (CUDA 12):

```bash
python -m pip install -U pip
python -m pip install -r prototypes/requirements.gpu-cu12.txt
```

cuQuantum (Team3 only):

```bash
python -m pip install -U pip
python -m pip install -r prototypes/requirements.gpu-cu12-cuquantum.txt
python -c "from cuquantum import cutensornet, custatevec; print('OK')"
```

## Known pitfalls

- Do **not** install the deprecated `pynvml` pip package directly. Use `nvidia-ml-py` (it provides the `pynvml` import).
- If you see `cuquantum.__file__ = None` and `_NamespaceLoader`, you likely have `cuquantum` shadowing in `dist-packages/`. Remove the shadowing folder and keep only the pip-installed bindings.


