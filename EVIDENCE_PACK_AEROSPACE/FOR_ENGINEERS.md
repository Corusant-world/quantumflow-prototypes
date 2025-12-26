# SIGMA: Engineering Verification Manual

This manual is for technical auditors at NVIDIA, SpaceX, and NASA.

## 1. Quick Verification (The "Less Dumb" Way)

To verify our claims of 160ns latency and O(N) determinism:

1. **Install Dependencies**: `pip install numpy torch matplotlib opencv-python`
2. **Run the Stress Test**: `python tools/starship_v3_shadow_flight.py`
3. **Audit the Code**: Read `SDK/NASA_cFS/src/sigma_core.c`. Notice the total absence of dynamic memory and recursion.

## 2. Hardware Proofs (Receipts)

We do not ask for trust. We provide evidence. 

All files in `EVIDENCE/` were generated on a live **NVIDIA H100 PCIe (80GB)**. 
- You can cross-reference the timestamps in `H100_FINAL_HARDWARE_RECEIPT.json` with the logs from our `shadow_flight_results.json`.

## 3. Architecture Deep-Dive

SIGMA operates by projecting the telemetry state into an **Ultrametric Baire Metric**. This transforms the search problem from a non-deterministic hash lookup into a deterministic **Prefix Match**.

- **DPX Core**: Utilizes the `viadp` instruction set for quasi-reversible computation.
- **TMA**: Bypasses CPU L1/L2 caches to stream sensor data directly into the Indexing manifold.

## 4. Compliance Audit

Run our automated auditor to see how we map each line of code to your requirements:
`python tools/audit_traceability.py`
