/**
 * @file nccl_backend.hpp
 * @brief NCCL backend for GPU-to-GPU distributed communication
 *
 * Implements efficient GPU collective operations using NVIDIA NCCL or AMD RCCL.
 */

#pragma once

#include "distributed.hpp"
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <string>

// Forward declare NCCL types to avoid dependency in header
#if defined(TENZOR_HAS_NCCL)
    #ifdef TENZOR_USE_ROCM
        #include <rccl/rccl.h>
    #else
        #include <nccl.h>
    #endif
#else
    // Stub definitions for builds without NCCL
    typedef void* ncclComm_t;
    typedef struct { char internal[128]; } ncclUniqueId;
    typedef enum { ncclFloat = 0, ncclDouble = 1, ncclInt = 2, ncclInt64 = 3 } ncclDataType_t;
    typedef enum { ncclSum = 0, ncclProd = 1, ncclMax = 2, ncclMin = 3 } ncclRedOp_t;
    typedef enum { ncclSuccess = 0 } ncclResult_t;
#endif

namespace tenzor {
namespace distributed {

/// Convert a ReduceOp to the corresponding NCCL reduction operator.
/// Shared by NCCLBackend and the process-group collective wrappers (single
/// source of truth — previously duplicated in process_group.cpp).
inline ncclRedOp_t to_nccl_reduce_op(ReduceOp op) {
    switch (op) {
        case ReduceOp::SUM:
        case ReduceOp::AVG:  // Average implemented as sum + divide
            return ncclSum;
        case ReduceOp::PRODUCT:
            return ncclProd;
        case ReduceOp::MIN:
            return ncclMin;
        case ReduceOp::MAX:
            return ncclMax;
        default:
            throw std::invalid_argument("Unsupported reduce operation for NCCL");
    }
}

/// Convert a DType to the corresponding NCCL data type.
inline ncclDataType_t to_nccl_datatype(DType dtype) {
    switch (dtype) {
        case DType::Float32: return ncclFloat;
        case DType::Float64: return ncclDouble;
        case DType::Int32:   return ncclInt;
        case DType::Int64:   return ncclInt64;
        default:
            throw std::invalid_argument(
                "Unsupported dtype for NCCL: " + std::to_string(static_cast<int>(dtype)));
    }
}

/**
 * @brief NCCL communication backend for GPU operations.
 *
 * Provides high-performance GPU-to-GPU communication using NCCL (NVIDIA)
 * or RCCL (AMD). Supports multi-GPU and multi-node configurations.
 *
 * Features:
 * - Ring all-reduce with bandwidth-optimal communication
 * - GPU Direct RDMA for inter-node communication
 * - Multi-stream support for overlapped computation/communication
 * - Topology-aware communication optimization
 * - Support for all standard collective operations
 *
 * Requirements:
 * - NCCL 2.0+ (NVIDIA) or RCCL (AMD)
 * - CUDA or ROCm runtime
 * - Network configuration for multi-node (NCCL_SOCKET_IFNAME, etc.)
 */
class NCCLBackend : public CommunicationBackend {
public:
    /**
     * @brief Construct NCCL backend.
     */
    NCCLBackend();

    /**
     * @brief Destructor - cleanup NCCL communicators.
     */
    ~NCCLBackend() override;

    // CommunicationBackend interface

    auto initialize(int rank, int world_size, const std::string& master_addr,
                   int master_port) -> void override;

    auto broadcast(Tensor& tensor, int src_rank) -> void override;

    auto all_reduce(Tensor& tensor, ReduceOp op) -> void override;

    auto all_reduce_async(Tensor& tensor, ReduceOp op,
                          void* stream) -> void override;

    auto supports_async_stream() const -> bool override { return true; }

    auto reduce(Tensor& tensor, int dst_rank, ReduceOp op) -> void override;

    auto all_gather(const Tensor& tensor, std::vector<Tensor>& output) -> void override;

    // Phase E (E1): NCCL all_gather_async / reduce_scatter_async overrides --
    // launch on caller's stream without cudaDeviceSynchronize so the collective
    // can overlap with default-stream compute. Caller responsible for
    // cudaStreamWaitEvent / cudaStreamSynchronize before reading output.
    auto all_gather_async(const Tensor& tensor, std::vector<Tensor>& output,
                          void* stream) -> void override;

    auto gather(const Tensor& tensor, std::vector<Tensor>& output, int dst_rank) -> void override;

    auto scatter(const std::vector<Tensor>& tensors, Tensor& output, int src_rank) -> void override;

    auto reduce_scatter(const std::vector<Tensor>& tensors, Tensor& output, ReduceOp op) -> void override;

    auto reduce_scatter_async(const std::vector<Tensor>& tensors, Tensor& output,
                              ReduceOp op, void* stream) -> void override;

    auto send(const Tensor& tensor, int dst_rank) -> void override;

    auto recv(Tensor& tensor, int src_rank) -> void override;

    // A4-extended: native NCCL all-to-all via ncclGroupStart + Send/Recv pairs.
    auto all_to_all_single(Tensor& output, const Tensor& input) -> void override;

    auto barrier() -> void override;

    auto finalize() -> void override;

    auto backend_type() const -> Backend override { return Backend::NCCL; }

    auto supports_device(Device::Type device_type) const -> bool override;

    // NCCL-specific methods

    /**
     * @brief Get NCCL communicator for a device.
     *
     * @param device_id GPU device ID
     * @return NCCL communicator handle
     */
    auto get_communicator(int device_id) -> ncclComm_t;

    /**
     * @brief Get unique ID for NCCL group initialization.
     *
     * Should be called on rank 0 and broadcast to all ranks.
     *
     * @return NCCL unique ID
     */
    auto get_unique_id() -> ncclUniqueId;

private:
    int rank_{-1};
    int world_size_{-1};
    std::string master_addr_;
    int master_port_{29500};

    // NCCL communicators per device
    std::unordered_map<int, ncclComm_t> communicators_;
    ncclUniqueId unique_id_;
    bool initialized_{false};

    // Helper methods


    /**
     * @brief Initialize communicator for a device.
     */
    auto init_communicator(int device_id) -> void;

    /**
     * @brief Validate tensor is on GPU device.
     */
    auto validate_gpu_tensor(const Tensor& tensor) -> void;

    /**
     * @brief Get device ID from tensor.
     */
    auto get_device_id(const Tensor& tensor) -> int;

    /**
     * @brief Exchange unique ID across all ranks.
     *
     * Uses TCP sockets to exchange NCCL unique ID.
     */
    auto exchange_unique_id() -> void;

    /**
     * @brief Create TCP socket for rank coordination.
     */
    auto create_socket_connection(bool is_master) -> int;

    /**
     * @brief Close socket connection.
     */
    auto close_socket(int socket_fd) -> void;
};

/**
 * @brief Helper to check NCCL errors.
 */
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    #define NCCL_CHECK(cmd) \
        do { \
            ncclResult_t result = cmd; \
            if (result != ncclSuccess) { \
                throw std::runtime_error( \
                    std::string("NCCL error: ") + ncclGetErrorString(result) + \
                    " at " + __FILE__ + ":" + std::to_string(__LINE__) \
                ); \
            } \
        } while(0)
#else
    #define NCCL_CHECK(cmd) cmd
#endif

} // namespace distributed
} // namespace tenzor
