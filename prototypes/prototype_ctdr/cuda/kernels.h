/**
 * Header file for CUDA kernels
 * Forward declarations for kernel functions
 */

#ifndef CTDR_KERNELS_H
#define CTDR_KERNELS_H

#include <cuda_runtime.h>
#include <cuda_stdint.h>

// Forward declaration of CUDA kernels
// Note: __global__ functions can be called from host code

__global__ void dpx_lcp_kernel(
    const short2* __restrict__ s1_encoded,
    const short2* __restrict__ s2_encoded,
    int* __restrict__ lcp_result,
    int n
);

// Correctness-first u16 kernel: compares 16-bit code units sequentially.
__global__ void dpx_lcp_u16_kernel(
    const uint16_t* __restrict__ s1_u16,
    const uint16_t* __restrict__ s2_u16,
    int* __restrict__ lcp_result,
    int n_u16
);

__global__ void reversible_einsum_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    uint8_t* __restrict__ C,
    float threshold,
    int m, int n, int k
);

#endif // CTDR_KERNELS_H

