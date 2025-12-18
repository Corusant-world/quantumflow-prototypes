# Contributing

## Scope

This repository contains:
- **Public-facing prototypes** under `prototypes/` (NVIDIA-facing deliverable)
- An **agent/orchestration stack** under `backend/` and `frontend/`

If you contribute to public-facing materials, keep them **English-only**.

## Development workflow

1) Run the ecosystem smoke test:

```bash
python prototypes/ecosystem_smoke.py
```

2) Run prototype demos:

```bash
python prototypes/team1_quantum/demo/demo.py
python prototypes/team2_energy/demo/demo.py
python prototypes/team3_innovation/demo/demo.py
```

3) Run benchmarks (writes `benchmarks/results/latest.json` per team):

```bash
python prototypes/team1_quantum/benchmarks/run_benchmarks.py
python prototypes/team2_energy/benchmarks/run_benchmarks.py
python prototypes/team3_innovation/benchmarks/run_benchmarks.py
```

## Pull request guidelines

- Keep changes minimal and reviewable.
- Do not introduce new dependencies unless justified and documented.
- Do not commit runtime artifacts (e.g., `prototypes/ecosystem_results/latest.json`).



