# SIGMA: Tool Qualification Plan (TQL-1 for DO-178C)

For DO-178C Level A certification, all tools used in the software development and verification life cycle must be qualified.

## 1. Compiler Qualification (CompCert)
SIGMA requires the use of **CompCert**, a formally verified C compiler.
*   **Reason**: Traditional compilers (GCC/Clang) can introduce "miscompilation" bugs. CompCert is mathematically proven to maintain the semantics of the source code.
*   **Qualification Level**: TQL-1 (highest level).

## 2. Static Analysis Qualification (Polyspace/Astrée)
To ensure the absence of runtime errors (division by zero, overflow), we use **Astrée**.
*   **Scope**: `sigma_core.c` and `sigma_dds_core.cpp`.
*   **Qualification Level**: TQL-2.

## 3. Unit Testing Qualification (VectorCAST/C++)
Automated testing of MC/DC coverage.
*   **Scope**: Full branch coverage of the `SIGMA_Verify_Frame` decision logic.
*   **Qualification Level**: TQL-2.

## 4. Hardware/Simulation Loop (HIL/PIL)
Testing the H100/DPX kernels in a Processor-in-the-Loop (PIL) environment to verify WCET (Worst Case Execution Time).
