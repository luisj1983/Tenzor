#pragma once

/**
 * @file cuda_launch_utils.cuh
 * @brief CUDA kernel launch utilities with occupancy-based block sizing
 *
 * Provides optimal_launch_config() to replace hardcoded BLOCK_SIZE = 256
 * with cudaOccupancyMaxPotentialBlockSize for better GPU utilization
 * across different architectures.
 */

#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace tenzor {
namespace cuda {

/**
 * @brief RAII guard that makes a CUDA device current and restores the previous
 *        device on scope exit.
 *
 * Single shared implementation (extracted from cudnn_sdpa.cpp's DeviceRestore)
 * used by every op wrapper that launches kernels or uses a library handle on a
 * tensor's device. cuBLAS/cuSOLVER/cuSPARSE/cuDNN handles and per-device
 * workspaces are keyed by the *current* device, so the wrapper must make the
 * tensor's device current before fetching them — otherwise the op silently
 * runs on whatever device happened to be current on the calling thread,
 * corrupting multi-GPU execution.
 *
 * @code
 * CudaDeviceGuard dev_guard(input.device().index);
 * cublasHandle_t h = CuBLASHandlePool::get(stream);  // bound to the right GPU
 * @endcode
 */
class CudaDeviceGuard {
public:
    explicit CudaDeviceGuard(int device) : prev_(0), cur_(device) {
        // Capture the previously-current device so it can be restored. If this
        // fails, the destructor's restore would be meaningless, so fail fast.
        cudaError_t err = cudaGetDevice(&prev_);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CudaDeviceGuard: cudaGetDevice failed: ") +
                cudaGetErrorString(err));
        }
        if (device >= 0 && device != prev_) {
            // A failed switch must abort here. The class contract is that the
            // tensor's device is current before handles/workspaces/launches are
            // fetched; swallowing this error would run the op on the wrong
            // device, exactly the silent multi-GPU corruption this guard exists
            // to prevent. Reset cur_ so the destructor does not attempt a
            // restore predicated on a switch that never happened.
            err = cudaSetDevice(device);
            if (err != cudaSuccess) {
                cur_ = prev_;
                throw std::runtime_error(
                    std::string("CudaDeviceGuard: cudaSetDevice(") +
                    std::to_string(device) + ") failed: " +
                    cudaGetErrorString(err));
            }
        }
    }
    ~CudaDeviceGuard() {
        if (cur_ >= 0 && cur_ != prev_) {
            // Best-effort restore in a destructor: cannot throw. Discarding the
            // status is acceptable here because the constructor already proved
            // prev_ is a valid current device.
            cudaSetDevice(prev_);
        }
    }
    CudaDeviceGuard(const CudaDeviceGuard&) = delete;
    CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;
    CudaDeviceGuard(CudaDeviceGuard&&) = delete;
    CudaDeviceGuard& operator=(CudaDeviceGuard&&) = delete;
private:
    int prev_;
    int cur_;
};

/**
 * @brief Compute optimal grid and block dimensions using CUDA occupancy API.
 *
 * Uses cudaOccupancyMaxPotentialBlockSize to determine the block size
 * that maximizes occupancy for a given kernel. Results are cached per
 * kernel function pointer to avoid repeated API calls.
 *
 * @tparam KernelFunc CUDA kernel function type
 * @param kernel Pointer to the CUDA kernel function
 * @param num_elements Total number of elements to process
 * @param dynamic_smem Dynamic shared memory per block (bytes)
 * @return Pair of (num_blocks, block_size)
 */
template<typename KernelFunc>
inline std::pair<int, int> optimal_launch_config(
    KernelFunc kernel, int64_t num_elements, size_t dynamic_smem = 0)
{
    // Cache block size per (kernel function pointer, device) to avoid repeated
    // cudaOccupancyMaxPotentialBlockSize calls (~10-100us each). The optimal
    // block size is architecture-dependent, so on heterogeneous multi-GPU hosts
    // the cache must be keyed by device, not just the kernel pointer.
    static thread_local std::unordered_map<const void*,
                                           std::unordered_map<int, int>> block_cache;
    const void* key = reinterpret_cast<const void*>(kernel);
    int cur_device = 0;
    cudaGetDevice(&cur_device);

    int block_size;
    auto kit = block_cache.find(key);
    auto it = (kit != block_cache.end()) ? kit->second.find(cur_device)
                                         : std::unordered_map<int, int>::iterator{};
    bool found = (kit != block_cache.end()) && (it != kit->second.end());
    if (found) {
        block_size = it->second;
    } else {
        int min_grid_size = 0;
        block_size = 256;  // fallback
        cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &block_size, kernel, dynamic_smem, 0);
        if (err != cudaSuccess) {
            block_size = 256;  // fallback on error
        }
        block_cache[key][cur_device] = block_size;
    }

    int num_blocks = static_cast<int>(
        std::min(static_cast<int64_t>((num_elements + block_size - 1) / block_size),
                 static_cast<int64_t>(2147483647)));  // CUDA x-dim max: 2^31-1
    num_blocks = std::max(1, num_blocks);

    return {num_blocks, block_size};
}

/**
 * @brief Compute grid size for a given block size and element count.
 *
 * Simple utility when block size is already known (e.g., from a previous
 * optimal_launch_config call or when using a fixed block size).
 *
 * @param num_elements Total number of elements to process
 * @param block_size Threads per block
 * @return Number of blocks to launch
 */
inline int compute_grid_size(int64_t num_elements, int block_size = 256) {
    // Clamp in int64 BEFORE narrowing; computing the block count in `int`
    // overflows for very large tensors (> ~2^31 * block_size elements).
    int64_t nb = std::min<int64_t>((num_elements + block_size - 1) / block_size,
                                   2147483647LL);  // CUDA x-dim max: 2^31-1
    return static_cast<int>(std::max<int64_t>(1, nb));
}

/**
 * @brief Conservative 1D launch config (256 threads) for general-purpose kernels.
 *
 * Uses a safe block size that works for ALL kernels regardless of register
 * pressure. For kernel-specific optimization, use OCCUPANCY_CONFIG instead.
 */
inline void compute_launch_config_1d(int64_t n, dim3& grid, dim3& block) {
    constexpr int kBlockSize = 256;
    // Compute the block count in int64 and clamp to the CUDA gridDim.x limit
    // (2^31-1) before narrowing. For n > ~2^31*256 the previous int cast
    // overflowed to a negative/garbage grid size; kernels here use grid-stride
    // loops, so a clamped grid still covers all n correctly.
    int64_t num_blocks = (n + kBlockSize - 1) / kBlockSize;
    if (num_blocks < 1) num_blocks = 1;
    constexpr int64_t kMaxGridX = 2147483647;  // CUDA gridDim.x maximum
    if (num_blocks > kMaxGridX) num_blocks = kMaxGridX;
    block = dim3(static_cast<unsigned int>(kBlockSize), 1, 1);
    grid  = dim3(static_cast<unsigned int>(num_blocks), 1, 1);
}

/**
 * @brief Compute occupancy-based grid/block from a kernel pointer and
 * element count, storing into the provided dim3 variables.
 */
#define OCCUPANCY_CONFIG(kernel_ptr, numel, grid_var, block_var) \
    do { \
        auto [_nb, _bs] = optimal_launch_config((kernel_ptr), (numel)); \
        (grid_var)  = dim3(static_cast<unsigned int>(_nb)); \
        (block_var) = dim3(static_cast<unsigned int>(_bs)); \
    } while (0)

/**
 * @brief RAII wrapper for cudaMalloc/cudaFree allocations (synchronous).
 */
struct CudaBuffer {
    void* ptr = nullptr;
    explicit CudaBuffer(size_t bytes) {
        if (bytes > 0) {
            auto err = cudaMalloc(&ptr, bytes);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
            }
        }
    }
    ~CudaBuffer() { if (ptr) cudaFree(ptr); }
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    template<typename T> T* as() { return static_cast<T*>(ptr); }
};

/**
 * @brief RAII wrapper for cudaMallocAsync/cudaFreeAsync allocations.
 *
 * Ensures async allocations are freed even when exceptions occur between
 * allocation and deallocation. Non-copyable, move-only.
 */
struct CudaAsyncBuffer {
    void* ptr = nullptr;
    cudaStream_t stream = nullptr;

    CudaAsyncBuffer() = default;

    CudaAsyncBuffer(size_t bytes, cudaStream_t s) : stream(s) {
        auto err = cudaMallocAsync(&ptr, bytes, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMallocAsync failed: ") + cudaGetErrorString(err));
        }
    }

    ~CudaAsyncBuffer() {
        if (ptr) cudaFreeAsync(ptr, stream);
    }

    CudaAsyncBuffer(const CudaAsyncBuffer&) = delete;
    CudaAsyncBuffer& operator=(const CudaAsyncBuffer&) = delete;

    CudaAsyncBuffer(CudaAsyncBuffer&& other) noexcept
        : ptr(other.ptr), stream(other.stream) {
        other.ptr = nullptr;
    }

    CudaAsyncBuffer& operator=(CudaAsyncBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaFreeAsync(ptr, stream);
            ptr = other.ptr;
            stream = other.stream;
            other.ptr = nullptr;
        }
        return *this;
    }

    template<typename T>
    T* as() { return static_cast<T*>(ptr); }
};

} // namespace cuda
} // namespace tenzor
