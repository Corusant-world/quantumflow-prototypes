# Release v0.1.0 — First Public Release

## Key Results (tested on NVIDIA H100 PCIe)

| Component | What it proves | Key metric | Evidence |
|---|---|---:|---|
| Team1 Quantum | Tensor Core-heavy screening workload (CPU fallback) | NVML GPU util avg **95.19%** | `prototypes/team1_quantum/benchmarks/results/latest.json` |
| Team2 Energy | Differentiable thermo + grid optimization (CPU fallback) | NVML GPU util avg **95.44%** | `prototypes/team2_energy/benchmarks/results/latest.json` |
| Team3 Innovation | cuQuantum contraction + sustained soak | NVML GPU util avg **95.47%** + `cuquantum_used=true` | `prototypes/team3_innovation/benchmarks/results/latest.json` |

## Quick Start

### CPU (no CUDA required)

```bash
python -m pip install -U pip
python -m pip install -r prototypes/requirements.cpu.txt
python prototypes/ecosystem_smoke.py
```

### NVIDIA GPU (CUDA 12, without cuQuantum)

```bash
python -m pip install -U pip
python -m pip install -r prototypes/requirements.gpu-cu12.txt
DEVICE=cuda python prototypes/ecosystem_smoke.py
```

### NVIDIA GPU (CUDA 12) + cuQuantum (Team3 acceleration)

```bash
python -m pip install -U pip
python -m pip install -r prototypes/requirements.gpu-cu12-cuquantum.txt
python -c "from cuquantum import cutensornet, custatevec; print('OK')"
DEVICE=cuda python prototypes/ecosystem_smoke.py
```

## What's New

- Three GPU-first prototypes (Team1 Quantum, Team2 Energy, Team3 Innovation)
- Ecosystem compatibility proof (one environment, zero conflicts)
- Reproducible benchmarks with JSON artifacts
- cuQuantum integration demonstration (Team3)
- CPU fallback for all prototypes
- One-command installation via aggregated requirements files

## Known Issues / Pitfalls

- **NVML import**: Do not install deprecated `pynvml` pip package. Use `nvidia-ml-py` (it provides the `pynvml` import).
- **cuQuantum shadowing**: If you see `cuquantum.__file__ = None` and `_NamespaceLoader`, you likely have `cuquantum` shadowing in `dist-packages/`. Remove the shadowing folder and keep only the pip-installed bindings.

## Links

- GitHub Repository: https://github.com/<ORG>/<REPO>
- Documentation: https://github.com/<ORG>/<REPO>/blob/main/README.md
- Docker Images: https://github.com/<ORG>/<REPO>/pkgs/container/quantumflow
- PyPI: https://pypi.org/project/quantumflow-prototypes/
- HuggingFace: https://huggingface.co/<ORG>/quantumflow

## Artifacts

This release includes sample benchmark artifacts (attached to this release):
- `release-artifacts/ecosystem_results.sample.json` — ecosystem smoke test results
- `release-artifacts/team1_latest.sample.json` — Team1 benchmark results (95.19% GPU util)
- `release-artifacts/team2_latest.sample.json` — Team2 benchmark results (95.44% GPU util)
- `release-artifacts/team3_latest.sample.json` — Team3 benchmark results (95.47% GPU util + cuQuantum)

These are **samples** for reference. Runtime artifacts are generated on each benchmark run and written to:
- `prototypes/ecosystem_results/latest.json`
- `prototypes/<team>/benchmarks/results/latest.json`

