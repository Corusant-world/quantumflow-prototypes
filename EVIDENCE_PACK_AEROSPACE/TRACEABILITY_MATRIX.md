# SIGMA Aerospace SDK: Traceability Matrix (NASA-STD-8739.8)

This document provides a 1-to-1 mapping between high-level mission assurance requirements and the implementation in the SIGMA/CTDR stack.

| Req ID | Description | Component | Source Location | Verification Method |
|---|---|---|---|---|
| **SA-DET-01** | Deterministic Execution Guarantee | `sigma_core.c` | `SIGMA_Verify_Frame` (line 43) | Static Analysis / WCET |
| **SA-PHY-01** | Mass-Energy Conservation Invariant | `sigma_core.c` | `SIGMA_Verify_Frame` (line 35) | Physics-in-Loop Sim |
| **SA-BYZ-01** | Byzantine Fault Detection (Veto) | `sigma_core.c` | `veto_active` flag (line 8) | Fault Injection Test |
| **SA-MEM-01** | Zero Dynamic Memory Allocation | `sigma_core.c` | Global static `last_frame` | Lint (No `malloc`) |
| **SA-DDS-01** | Zero-Copy Middleware Integration | `sigma_dds_core.cpp` | `SigmaDDS::Publish` | Memory Profiling |
| **SA-RAD-01** | Radiation (SEU) Detection via LCP | `sigma_core.c` | `ctdr_token` audit (line 49) | Fault Injection (Bit-flip) |
| **SA-OSAL-01** | NASA OSAL Compliance | `sigma_mon_app.c` | `CFE_ES_RegisterApp` | cFS Integration Test |

## Requirement Definition Details

### SA-PHY-01: Mass-Energy Conservation
**Requirement**: The system shall detect anomalies where the change in kinetic energy does not correlate with the integrated thrust vector.
**Implementation**: SIGMA monitors `dv_sq` and triggers a `VETO` if significant acceleration is detected without active propulsion (Byzantine sensor failure).

### SA-DET-01: Determinism
**Requirement**: Every execution frame must be completed within a fixed time window regardless of input complexity.
**Implementation**: CTDR uses O(N) ultrametric indexing. Complexity is linear to the number of sensors, ensuring strictly predictable WCET (Worst Case Execution Time).
