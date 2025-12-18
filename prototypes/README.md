# Prototypes — install + run (Team1 + Team2 + Team3)

This folder contains 3 independent Python prototypes that are designed to run **together in one environment**:
- `team1_quantum`
- `team2_energy`
- `team3_innovation`

## Install (one command)

### Option A — CPU / laptop

```bash
python -m pip install -U pip
python -m pip install -r prototypes/requirements.cpu.txt
```

### Option B — NVIDIA GPU (CUDA 12, without cuQuantum)

```bash
python -m pip install -U pip
python -m pip install -r prototypes/requirements.gpu-cu12.txt
```

### Option C — NVIDIA GPU (CUDA 12) + cuQuantum (Team3 acceleration)

```bash
python -m pip install -U pip
python -m pip install -r prototypes/requirements.gpu-cu12-cuquantum.txt
```

## Ecosystem smoke test (all 3 prototypes together)

```bash
python prototypes/ecosystem_smoke.py
```

Force mode:
- CPU: `DEVICE=cpu python prototypes/ecosystem_smoke.py`
- CUDA: `DEVICE=cuda python prototypes/ecosystem_smoke.py`

Artifact: `prototypes/ecosystem_results/latest.json` (generated on each run).

## Demos

```bash
python prototypes/team1_quantum/demo/demo.py
python prototypes/team2_energy/demo/demo.py
python prototypes/team3_innovation/demo/demo.py
```

## Benchmarks

```bash
python prototypes/team1_quantum/benchmarks/run_benchmarks.py
python prototypes/team2_energy/benchmarks/run_benchmarks.py
python prototypes/team3_innovation/benchmarks/run_benchmarks.py
```


