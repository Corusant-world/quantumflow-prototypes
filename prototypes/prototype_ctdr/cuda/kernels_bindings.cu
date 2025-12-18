/**
 * Pybind11 bindings for CTDR CUDA kernels.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <cuda_stdint.h>
#include "kernels.h"  // Forward declarations

namespace py = pybind11;

// Local kernels are defined in this translation unit to guarantee device code
// is linked into `ctdr_python` (avoid cross-shared-lib CUDA device linking issues).
__global__ void dpx_lcp_u16_kernel_local_blockmin(
    const uint16_t* __restrict__ s1_u16,
    const uint16_t* __restrict__ s2_u16,
    int* __restrict__ lcp_result,
    int n_u16
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int candidate = n_u16;
    if (idx < n_u16) {
        candidate = (s1_u16[idx] != s2_u16[idx]) ? idx : n_u16;
    }

    // Reduce to per-block minimum to avoid global atomic contention.
    extern __shared__ int shmin[];
    shmin[threadIdx.x] = candidate;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            int other = shmin[threadIdx.x + stride];
            if (other < shmin[threadIdx.x]) {
                shmin[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMin(lcp_result, shmin[0]);
    }
}

__global__ void dpx_lcp_u16_batch_kernel_local_min(
    const uint16_t* __restrict__ query_u16,
    const uint16_t* __restrict__ candidates_u16, // shape: [num_candidates, n_u16]
    int* __restrict__ out_lcp_u16,                // shape: [num_candidates]
    int n_u16,
    int num_candidates
) {
    int cand = blockIdx.x;
    if (cand >= num_candidates) return;
    const uint16_t* cand_u16 = candidates_u16 + static_cast<long long>(cand) * static_cast<long long>(n_u16);

    int local_min = n_u16;
    for (int i = threadIdx.x; i < n_u16; i += blockDim.x) {
        if (query_u16[i] != cand_u16[i]) {
            if (i < local_min) local_min = i;
        }
    }

    // Block reduce min.
    extern __shared__ int shmin[];
    shmin[threadIdx.x] = local_min;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            int other = shmin[threadIdx.x + stride];
            if (other < shmin[threadIdx.x]) shmin[threadIdx.x] = other;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_lcp_u16[cand] = shmin[0];
    }
}

// Warp-first-mismatch kernel:
// - 1 warp per candidate
// - compare 128-bit chunks (8x u16) per lane
// - early exit on first mismatch using ballot+ffs
// This is the intended DPX-friendly shape: branchless-ish, integer compares, high parallelism.
__global__ void dpx_lcp_u16_batch_kernel_warp_first_mismatch(
    const uint16_t* __restrict__ query_u16,
    const uint16_t* __restrict__ candidates_u16, // [num_candidates, n_u16]
    int* __restrict__ out_lcp_u16,                // [num_candidates]
    int n_u16,
    int num_candidates
) {
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
    const int lane = threadIdx.x & 31;
    const int warp_id_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int cand = blockIdx.x * warps_per_block + warp_id_in_block;
    if (cand >= num_candidates) return;

    // candidates are laid out as u16; for our typical sizes (query_len=2048), each candidate stride is 4096 bytes,
    // which is 16-byte aligned. cudaMalloc gives base alignment, so uint4 loads are safe.
    const uint16_t* cand_u16 = candidates_u16 + static_cast<long long>(cand) * static_cast<long long>(n_u16);
    const uint4* q4 = reinterpret_cast<const uint4*>(query_u16);
    const uint4* c4 = reinterpret_cast<const uint4*>(cand_u16);
    const int n_u4 = (n_u16 * static_cast<int>(sizeof(uint16_t))) / static_cast<int>(sizeof(uint4)); // bytes/16

    int found = n_u16; // default: identical
    for (int base = 0; base < n_u4; base += 32) {
        const int i4 = base + lane;
        uint4 qv = {0, 0, 0, 0};
        uint4 cv = {0, 0, 0, 0};
        if (i4 < n_u4) {
            qv = q4[i4];
            cv = c4[i4];
        }

        const uint32_t d0 = (qv.x ^ cv.x);
        const uint32_t d1 = (qv.y ^ cv.y);
        const uint32_t d2 = (qv.z ^ cv.z);
        const uint32_t d3 = (qv.w ^ cv.w);
        const int has = (d0 | d1 | d2 | d3) != 0;

        unsigned mask = __ballot_sync(FULL_MASK, has);
        if (mask != 0u) {
            const int first_lane = __ffs(mask) - 1; // 0..31

            // compute first mismatch u16 index for THIS lane (only meaningful on mismatch lanes)
            int lane_u16 = n_u16;
            if (has) {
                // each uint4 covers 16 bytes => 8 u16
                const int chunk_u16_base = i4 * 8;
                // scan four u32 words, each covers 2 u16
                const uint32_t ds[4] = {d0, d1, d2, d3};
                #pragma unroll
                for (int w = 0; w < 4; w++) {
                    const uint32_t x = ds[w];
                    if (x != 0u) {
                        const int u16_base = chunk_u16_base + (w * 2);
                        if ((x & 0x0000FFFFu) != 0u) { lane_u16 = u16_base + 0; }
                        else { lane_u16 = u16_base + 1; }
                        break;
                    }
                }
            }

            found = __shfl_sync(FULL_MASK, lane_u16, first_lane);
            break;
        }
    }

    // Convert u16-index to LCP length in u16 units (first mismatch index).
    // If identical => n_u16.
    if (lane == 0) {
        out_lcp_u16[cand] = found;
    }
}

__global__ void argmax_int_kernel_local(
    const int* __restrict__ values,
    int* __restrict__ out_best_idx,
    int* __restrict__ out_best_val,
    int n
) {
    int best_val = -2147483647;
    int best_idx = 0;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int v = values[i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    extern __shared__ int sh[];
    int* sh_val = sh;
    int* sh_idx = sh + blockDim.x;
    sh_val[threadIdx.x] = best_val;
    sh_idx[threadIdx.x] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            int v_other = sh_val[threadIdx.x + stride];
            int i_other = sh_idx[threadIdx.x + stride];
            if (v_other > sh_val[threadIdx.x]) {
                sh_val[threadIdx.x] = v_other;
                sh_idx[threadIdx.x] = i_other;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_best_idx[0] = sh_idx[0];
        out_best_val[0] = sh_val[0];
    }
}

// ---------------------------------------------------------------------------
// Local kernel for Reversible Einsum (u8 inputs/outputs).
// Defined here to guarantee device code is linked into ctdr_python.
// ---------------------------------------------------------------------------
__global__ void reversible_einsum_u8_kernel_local(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    uint8_t* __restrict__ C,
    float threshold,
    int m, int n, int k
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (int j = 0; j < k; j++) {
        float a_val = (A[row * k + j] != 0) ? 1.0f : 0.0f;
        float b_val = (B[j * n + col] != 0) ? 1.0f : 0.0f;
        sum += a_val * b_val;
    }

    float diff = sum - threshold;
    C[row * n + col] = (diff >= 0.0f) ? 1 : 0;
}

static void throw_cuda(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return;
    throw std::runtime_error(std::string(what) + ": " + std::string(cudaGetErrorString(err)));
}

static void validate_short2_bytes(const std::string& s, const char* name) {
    if (s.empty()) {
        throw std::runtime_error(std::string(name) + " is empty");
    }
    if ((s.length() % sizeof(short2)) != 0) {
        throw std::runtime_error(
            std::string(name) + " must be short2-encoded bytes (length multiple of 4). "
            "Encode via src/encoding.py: encode_to_short2(text)."
        );
    }
}

/**
 * Wrapper for dpx_lcp_kernel
 * 
 * Args:
 *   s1: bytes - encoded string 1
 *   s2: bytes - encoded string 2
 * 
 * Returns:
 *   int - LCP length
 */
int dpx_lcp_wrapper(py::bytes s1_bytes, py::bytes s2_bytes) {
    // Convert Python bytes to std::string preserving embedded nulls (short2 encoding contains 0 bytes).
    char* s1_buf = nullptr;
    char* s2_buf = nullptr;
    py::ssize_t s1_len = 0;
    py::ssize_t s2_len = 0;
    if (PyBytes_AsStringAndSize(s1_bytes.ptr(), &s1_buf, &s1_len) != 0) {
        throw std::runtime_error("Failed to read s1 bytes");
    }
    if (PyBytes_AsStringAndSize(s2_bytes.ptr(), &s2_buf, &s2_len) != 0) {
        throw std::runtime_error("Failed to read s2 bytes");
    }
    std::string s1_str(s1_buf, static_cast<size_t>(s1_len));
    std::string s2_str(s2_buf, static_cast<size_t>(s2_len));
    
    validate_short2_bytes(s1_str, "s1");
    validate_short2_bytes(s2_str, "s2");
    
    // Calculate number of short2 pairs
    int n1 = static_cast<int>(s1_str.length() / sizeof(short2));
    int n2 = static_cast<int>(s2_str.length() / sizeof(short2));
    int n = (n1 < n2) ? n1 : n2;
    
    if (n == 0) {
        return 0;
    }
    
    // GPU buffers are cached to avoid cudaMalloc/cudaFree per call.
    // This is crucial for beating CPU on small/medium sizes.
    static uint16_t* d_s1_u16 = nullptr;
    static uint16_t* d_s2_u16 = nullptr;
    static int* d_lcp_result = nullptr;
    static int capacity_u16 = 0;
    
    cudaError_t err;
    int n_u16 = n * 2;

    if (capacity_u16 < n_u16 || d_s1_u16 == nullptr || d_s2_u16 == nullptr || d_lcp_result == nullptr) {
        if (d_s1_u16) cudaFree(d_s1_u16);
        if (d_s2_u16) cudaFree(d_s2_u16);
        if (d_lcp_result) cudaFree(d_lcp_result);
        d_s1_u16 = nullptr;
        d_s2_u16 = nullptr;
        d_lcp_result = nullptr;
        capacity_u16 = n_u16;

        throw_cuda(cudaMalloc((void**)&d_s1_u16, capacity_u16 * sizeof(uint16_t)), "cudaMalloc failed for s1");
        throw_cuda(cudaMalloc((void**)&d_s2_u16, capacity_u16 * sizeof(uint16_t)), "cudaMalloc failed for s2");
        throw_cuda(cudaMalloc((void**)&d_lcp_result, sizeof(int)), "cudaMalloc failed for lcp_result");
    }
    
    // Copy data to GPU
    throw_cuda(cudaMemcpy(d_s1_u16, s1_str.data(), n_u16 * sizeof(uint16_t), cudaMemcpyHostToDevice), "cudaMemcpy failed for s1");
    throw_cuda(cudaMemcpy(d_s2_u16, s2_str.data(), n_u16 * sizeof(uint16_t), cudaMemcpyHostToDevice), "cudaMemcpy failed for s2");

    // Launch kernel (local, parallel)
    // Initialize result to n_u16 (\"no mismatch found\") and atomicMin updates it.
    int init_val = n_u16;
    throw_cuda(cudaMemcpy(d_lcp_result, &init_val, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy init failed for lcp_result");

    int threads = 256;
    int blocks = (n_u16 + threads - 1) / threads;
    size_t shmem = static_cast<size_t>(threads) * sizeof(int);
    dpx_lcp_u16_kernel_local_blockmin<<<blocks, threads, shmem>>>(d_s1_u16, d_s2_u16, d_lcp_result, n_u16);
    
    throw_cuda(cudaGetLastError(), "Kernel launch failed");
    throw_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    
    // Copy result back
    int h_lcp_result = 0;
    throw_cuda(cudaMemcpy(&h_lcp_result, d_lcp_result, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy failed for result");
    
    int result = h_lcp_result;

    return result;
}

// Batch index state (candidates live on GPU between queries).
static uint16_t* g_d_candidates_u16 = nullptr;
static int* g_d_out_lcp_u16 = nullptr;
static int* g_d_best_idx = nullptr;
static int* g_d_best_val = nullptr;
static int g_capacity_candidates = 0;
static int g_capacity_n_u16 = 0;
static int g_num_candidates = 0;
static uint16_t* g_d_query_u16 = nullptr;
static int g_capacity_query_u16 = 0;

bool dpx_lcp_index_load(py::bytes candidates_bytes, int num_candidates) {
    char* buf = nullptr;
    py::ssize_t blen = 0;
    if (PyBytes_AsStringAndSize(candidates_bytes.ptr(), &buf, &blen) != 0) {
        throw std::runtime_error("Failed to read candidates bytes");
    }
    std::string cand_str(buf, static_cast<size_t>(blen));
    validate_short2_bytes(cand_str, "candidates");
    if (num_candidates <= 0) {
        throw std::runtime_error("num_candidates must be > 0");
    }
    if ((cand_str.size() % static_cast<size_t>(num_candidates)) != 0) {
        throw std::runtime_error("candidates_bytes length must be divisible by num_candidates");
    }
    const int bytes_per_candidate = static_cast<int>(cand_str.size() / static_cast<size_t>(num_candidates));
    if ((bytes_per_candidate % static_cast<int>(sizeof(short2))) != 0) {
        throw std::runtime_error("bytes_per_candidate must be multiple of 4 (short2)");
    }
    const int n_pairs = bytes_per_candidate / static_cast<int>(sizeof(short2));
    const int n_u16 = n_pairs * 2;

    if (g_d_candidates_u16 == nullptr || g_capacity_candidates < num_candidates || g_capacity_n_u16 < n_u16) {
        if (g_d_candidates_u16) cudaFree(g_d_candidates_u16);
        if (g_d_out_lcp_u16) cudaFree(g_d_out_lcp_u16);
        if (g_d_best_idx) cudaFree(g_d_best_idx);
        if (g_d_best_val) cudaFree(g_d_best_val);
        g_d_candidates_u16 = nullptr;
        g_d_out_lcp_u16 = nullptr;
        g_d_best_idx = nullptr;
        g_d_best_val = nullptr;
        g_capacity_candidates = num_candidates;
        g_capacity_n_u16 = n_u16;
        const size_t bytes_candidates = static_cast<size_t>(g_capacity_candidates) * static_cast<size_t>(g_capacity_n_u16) * sizeof(uint16_t);
        throw_cuda(cudaMalloc((void**)&g_d_candidates_u16, bytes_candidates), "cudaMalloc failed for candidates");
        throw_cuda(cudaMalloc((void**)&g_d_out_lcp_u16, static_cast<size_t>(g_capacity_candidates) * sizeof(int)), "cudaMalloc failed for out_lcp");
        throw_cuda(cudaMalloc((void**)&g_d_best_idx, sizeof(int)), "cudaMalloc failed for best_idx");
        throw_cuda(cudaMalloc((void**)&g_d_best_val, sizeof(int)), "cudaMalloc failed for best_val");
    }
    if (g_d_query_u16 == nullptr || g_capacity_query_u16 < n_u16) {
        if (g_d_query_u16) cudaFree(g_d_query_u16);
        g_d_query_u16 = nullptr;
        g_capacity_query_u16 = n_u16;
        throw_cuda(cudaMalloc((void**)&g_d_query_u16, static_cast<size_t>(g_capacity_query_u16) * sizeof(uint16_t)), "cudaMalloc failed for query");
    }

    throw_cuda(cudaMemcpy(g_d_candidates_u16, cand_str.data(), static_cast<size_t>(num_candidates) * static_cast<size_t>(n_u16) * sizeof(uint16_t), cudaMemcpyHostToDevice),
              "cudaMemcpy failed for candidates");
    g_num_candidates = num_candidates;
    return true;
}

py::array_t<int> dpx_lcp_index_query(py::bytes query_bytes) {
    if (g_d_candidates_u16 == nullptr || g_d_out_lcp_u16 == nullptr || g_num_candidates <= 0) {
        throw std::runtime_error("Index not loaded. Call dpx_lcp_index_load(candidates_bytes, num_candidates) first.");
    }
    char* buf = nullptr;
    py::ssize_t blen = 0;
    if (PyBytes_AsStringAndSize(query_bytes.ptr(), &buf, &blen) != 0) {
        throw std::runtime_error("Failed to read query bytes");
    }
    std::string q_str(buf, static_cast<size_t>(blen));
    validate_short2_bytes(q_str, "query");
    const int n_pairs = static_cast<int>(q_str.size() / sizeof(short2));
    const int n_u16 = n_pairs * 2;
    if (n_u16 > g_capacity_query_u16 || n_u16 != g_capacity_n_u16) {
        throw std::runtime_error("Query length must match index candidate length.");
    }

    throw_cuda(cudaMemcpy(g_d_query_u16, q_str.data(), static_cast<size_t>(n_u16) * sizeof(uint16_t), cudaMemcpyHostToDevice),
              "cudaMemcpy failed for query");

    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks = (g_num_candidates + warps_per_block - 1) / warps_per_block;
    dpx_lcp_u16_batch_kernel_warp_first_mismatch<<<blocks, threads>>>(
        g_d_query_u16, g_d_candidates_u16, g_d_out_lcp_u16, n_u16, g_num_candidates
    );
    throw_cuda(cudaGetLastError(), "Kernel launch failed (batch_warp)");
    throw_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed (batch_warp)");

    auto out = py::array_t<int>({g_num_candidates});
    py::buffer_info out_buf = out.request();
    throw_cuda(cudaMemcpy(out_buf.ptr, g_d_out_lcp_u16, static_cast<size_t>(g_num_candidates) * sizeof(int), cudaMemcpyDeviceToHost),
              "cudaMemcpy failed for batch result");
    return out;
}

bool dpx_lcp_index_set_query(py::bytes query_bytes) {
    if (g_d_candidates_u16 == nullptr || g_d_out_lcp_u16 == nullptr || g_num_candidates <= 0) {
        throw std::runtime_error("Index not loaded. Call dpx_lcp_index_load(...) first.");
    }
    char* buf = nullptr;
    py::ssize_t blen = 0;
    if (PyBytes_AsStringAndSize(query_bytes.ptr(), &buf, &blen) != 0) {
        throw std::runtime_error("Failed to read query bytes");
    }
    std::string q_str(buf, static_cast<size_t>(blen));
    validate_short2_bytes(q_str, "query");
    const int n_pairs = static_cast<int>(q_str.size() / sizeof(short2));
    const int n_u16 = n_pairs * 2;
    if (n_u16 != g_capacity_n_u16) {
        throw std::runtime_error("Query length must match index candidate length.");
    }

    if (g_d_query_u16 == nullptr || g_capacity_query_u16 < n_u16) {
        if (g_d_query_u16) cudaFree(g_d_query_u16);
        g_d_query_u16 = nullptr;
        g_capacity_query_u16 = n_u16;
        throw_cuda(cudaMalloc((void**)&g_d_query_u16, static_cast<size_t>(g_capacity_query_u16) * sizeof(uint16_t)), "cudaMalloc failed for query");
    }

    throw_cuda(cudaMemcpy(g_d_query_u16, q_str.data(), static_cast<size_t>(n_u16) * sizeof(uint16_t), cudaMemcpyHostToDevice),
              "cudaMemcpy failed for query");
    return true;
}

py::tuple dpx_lcp_index_query_top1() {
    if (g_d_candidates_u16 == nullptr || g_d_out_lcp_u16 == nullptr || g_d_query_u16 == nullptr || g_num_candidates <= 0) {
        throw std::runtime_error("Index/query not ready. Call dpx_lcp_index_load(...) and dpx_lcp_index_set_query(...) first.");
    }

    const int threads = 256; // 8 warps per block
    const int warps_per_block = threads / 32;
    const int blocks = (g_num_candidates + warps_per_block - 1) / warps_per_block;
    dpx_lcp_u16_batch_kernel_warp_first_mismatch<<<blocks, threads>>>(
        g_d_query_u16, g_d_candidates_u16, g_d_out_lcp_u16, g_capacity_n_u16, g_num_candidates
    );
    throw_cuda(cudaGetLastError(), "Kernel launch failed (batch_warp)");

    const int red_threads = 256;
    const size_t shmem_red = static_cast<size_t>(red_threads) * sizeof(int) * 2;
    argmax_int_kernel_local<<<1, red_threads, shmem_red>>>(g_d_out_lcp_u16, g_d_best_idx, g_d_best_val, g_num_candidates);
    throw_cuda(cudaGetLastError(), "Kernel launch failed (argmax)");
    throw_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed (top1)");

    int h_idx = 0;
    int h_val = 0;
    throw_cuda(cudaMemcpy(&h_idx, g_d_best_idx, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy failed for best_idx");
    throw_cuda(cudaMemcpy(&h_val, g_d_best_val, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy failed for best_val");
    return py::make_tuple(h_idx, h_val);
}

void dpx_lcp_index_clear() {
    if (g_d_candidates_u16) cudaFree(g_d_candidates_u16);
    if (g_d_out_lcp_u16) cudaFree(g_d_out_lcp_u16);
    if (g_d_query_u16) cudaFree(g_d_query_u16);
    if (g_d_best_idx) cudaFree(g_d_best_idx);
    if (g_d_best_val) cudaFree(g_d_best_val);
    g_d_candidates_u16 = nullptr;
    g_d_out_lcp_u16 = nullptr;
    g_d_query_u16 = nullptr;
    g_d_best_idx = nullptr;
    g_d_best_val = nullptr;
    g_capacity_candidates = 0;
    g_capacity_n_u16 = 0;
    g_capacity_query_u16 = 0;
    g_num_candidates = 0;
}

/**
 * Wrapper for reversible_einsum_kernel
 * 
 * Args:
 *   A: numpy.ndarray - Boolean matrix A (shape: [m, k])
 *   B: numpy.ndarray - Boolean matrix B (shape: [k, n])
 *   threshold: float - Heaviside threshold
 * 
 * Returns:
 *   numpy.ndarray - Boolean matrix C (shape: [m, n])
 */
py::array_t<bool> reversible_einsum_wrapper(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> A,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> B,
    float threshold
) {
    // Get buffer info
    py::buffer_info A_buf = A.request();
    py::buffer_info B_buf = B.request();
    
    // Validate shapes
    if (A_buf.ndim != 2 || B_buf.ndim != 2) {
        throw std::runtime_error("Input arrays must be 2D");
    }
    
    int m = A_buf.shape[0];
    int k_A = A_buf.shape[1];
    int k_B = B_buf.shape[0];
    int n = B_buf.shape[1];
    
    if (k_A != k_B) {
        throw std::runtime_error(
            "Shape mismatch: A.shape[1]=" + std::to_string(k_A) + 
            " != B.shape[0]=" + std::to_string(k_B)
        );
    }
    
    int k = k_A;
    
    // Allocate GPU memory
    uint8_t *d_A, *d_B, *d_C;
    cudaError_t err;
    
    throw_cuda(cudaMalloc((void**)&d_A, m * k * sizeof(uint8_t)), "cudaMalloc failed for A");
    
    err = cudaMalloc((void**)&d_B, k * n * sizeof(uint8_t));
    if (err != cudaSuccess) { cudaFree(d_A); throw std::runtime_error("cudaMalloc failed for B: " + std::string(cudaGetErrorString(err))); }
    
    err = cudaMalloc((void**)&d_C, m * n * sizeof(uint8_t));
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); throw std::runtime_error("cudaMalloc failed for C: " + std::string(cudaGetErrorString(err))); }
    
    // Copy data to GPU
    throw_cuda(cudaMemcpy(d_A, A_buf.ptr, m * k * sizeof(uint8_t), cudaMemcpyHostToDevice), "cudaMemcpy failed for A");
    
    throw_cuda(cudaMemcpy(d_B, B_buf.ptr, k * n * sizeof(uint8_t), cudaMemcpyHostToDevice), "cudaMemcpy failed for B");
    
    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    
    reversible_einsum_u8_kernel_local<<<gridSize, blockSize>>>(
        d_A, d_B, d_C, threshold, m, n, k
    );
    
    throw_cuda(cudaGetLastError(), "Kernel launch failed");
    
    throw_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    
    // Allocate output array
    auto result = py::array_t<bool>({m, n});
    py::buffer_info result_buf = result.request();
    
    // Copy result back
    throw_cuda(cudaMemcpy(result_buf.ptr, d_C, m * n * sizeof(uint8_t), cudaMemcpyDeviceToHost), "cudaMemcpy failed for result");
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return result;
}

PYBIND11_MODULE(ctdr_python, m) {
    m.doc() = "CTDR CUDA Kernels - DPX_LCP_Kernel and Reversible_Einsum_Engine bindings";
    
    m.def("dpx_lcp", &dpx_lcp_wrapper,
          "DPX LCP Kernel wrapper",
          py::arg("s1"), py::arg("s2"));

    m.def("dpx_lcp_index_load", &dpx_lcp_index_load,
          "Load candidates (short2 bytes) to GPU index for batch LCP queries",
          py::arg("candidates_bytes"), py::arg("num_candidates"));
    m.def("dpx_lcp_index_query", &dpx_lcp_index_query,
          "Query GPU index with a short2-encoded query; returns LCP lengths (u16 count) per candidate",
          py::arg("query_bytes"));
    m.def("dpx_lcp_index_set_query", &dpx_lcp_index_set_query,
          "Set query bytes once (kept resident on GPU) for repeated top1 queries",
          py::arg("query_bytes"));
    m.def("dpx_lcp_index_query_top1", &dpx_lcp_index_query_top1,
          "Run batch LCP and return (best_idx, best_lcp_u16) without copying full array to host");
    m.def("dpx_lcp_index_clear", &dpx_lcp_index_clear,
          "Free GPU index buffers for batch LCP");
    
    m.def("reversible_einsum", &reversible_einsum_wrapper,
          "Reversible Einsum Engine wrapper (Boolean Einsum + Heaviside threshold)",
          py::arg("A"), py::arg("B"), py::arg("threshold"));
}

