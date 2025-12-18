# Release Artifacts (Samples)

This directory contains **sample** benchmark artifacts for reference. These are not runtime artifacts — they are examples of what the benchmarks produce.

## Files

- `ecosystem_results.sample.json` — Example output from `prototypes/ecosystem_smoke.py`
- `team1_latest.sample.json` — Example output from `prototypes/team1_quantum/benchmarks/run_benchmarks.py`
- `team2_latest.sample.json` — Example output from `prototypes/team2_energy/benchmarks/run_benchmarks.py`
- `team3_latest.sample.json` — Example output from `prototypes/team3_innovation/benchmarks/run_benchmarks.py`

## Usage

These samples are included in GitHub Releases (v0.1.0+) to demonstrate:
- What the benchmark artifacts look like
- The structure of JSON output
- Example metrics (GPU utilization, TFLOPS, etc.)

## Runtime Artifacts

**Note:** Runtime artifacts are generated on each benchmark run and written to:
- `prototypes/ecosystem_results/latest.json` (ecosystem smoke test)
- `prototypes/<team>/benchmarks/results/latest.json` (team-specific benchmarks)

These runtime artifacts are **not committed** to the repository (see `.gitignore`). Only samples are included in releases.
