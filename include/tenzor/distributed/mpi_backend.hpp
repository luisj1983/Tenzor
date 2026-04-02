/**
 * @file mpi_backend.hpp
 * @brief MPI backend for distributed communication
 *
 * Implements the CommunicationBackend interface using MPI collective and
 * point-to-point operations. Supports both CPU tensors natively and GPU
 * tensors via host-staging when MPI is not CUDA-aware.
 */

#pragma once

#include "distributed.hpp"
#include <vector>

#ifdef TENZOR_HAS_MPI
    #include <mpi.h>
#else
    // Stub types for builds without MPI
    typedef int MPI_Comm;
    typedef int MPI_Datatype;
    typedef int MPI_Op;
    #define MPI_COMM_WORLD 0
#endif

namespace tenzor {
namespace distributed {

/**
 * @brief MPI communication backend.
 *
 * Provides distributed collective and point-to-point operations using MPI.
 * Supports all standard MPI collectives: all-reduce, broadcast, all-gather,
 * gather, scatter, reduce-scatter, barrier, send, and recv.
 *
 * GPU Tensor Support:
 * - If MPI is CUDA-aware (detected at initialization), GPU buffers are
 *   passed directly to MPI calls for zero-copy communication.
 * - Otherwise, tensors are staged through pinned host memory automatically.
 *
 * Requirements:
 * - MPI 3.0+ (for MPI_Iallreduce and one-sided features)
 * - Build with TENZOR_HAS_MPI defined
 * - mpirun/mpiexec for launching multi-process jobs
 */
class MPIBackend : public CommunicationBackend {
public:
    /**
     * @brief Construct MPI backend.
     *
     * Does not call MPI_Init; use initialize() to set up MPI state.
     */
    MPIBackend();

    /**
     * @brief Destructor - calls finalize() if still initialized.
     */
    ~MPIBackend() override;

    // Non-copyable
    MPIBackend(const MPIBackend&) = delete;
    MPIBackend& operator=(const MPIBackend&) = delete;

    // CommunicationBackend interface

    /**
     * @brief Initialize MPI backend.
     *
     * Calls MPI_Init if MPI has not already been initialized, then
     * validates rank/world_size against the MPI communicator. The
     * master_addr and master_port parameters are ignored (MPI manages
     * its own transport).
     *
     * @param rank Expected process rank
     * @param world_size Expected world size
     * @param master_addr Ignored (MPI handles transport)
     * @param master_port Ignored (MPI handles transport)
     */
    auto initialize(int rank, int world_size, const std::string& master_addr,
                    int master_port) -> void override;

    auto broadcast(Tensor& tensor, int src_rank) -> void override;

    auto all_reduce(Tensor& tensor, ReduceOp op) -> void override;

    auto reduce(Tensor& tensor, int dst_rank, ReduceOp op) -> void override;

    auto all_gather(const Tensor& tensor, std::vector<Tensor>& output) -> void override;

    auto gather(const Tensor& tensor, std::vector<Tensor>& output, int dst_rank) -> void override;

    auto scatter(const std::vector<Tensor>& tensors, Tensor& output, int src_rank) -> void override;

    auto reduce_scatter(const std::vector<Tensor>& tensors, Tensor& output, ReduceOp op) -> void override;

    auto barrier() -> void override;

    /**
     * @brief Send tensor to destination rank (point-to-point).
     *
     * @param tensor Tensor to send
     * @param dst_rank Destination rank
     */
    auto send(const Tensor& tensor, int dst_rank) -> void;

    /**
     * @brief Receive tensor from source rank (point-to-point).
     *
     * @param tensor Pre-allocated tensor to receive into
     * @param src_rank Source rank
     */
    auto recv(Tensor& tensor, int src_rank) -> void;

    auto finalize() -> void override;

    auto backend_type() const -> Backend override { return Backend::MPI; }

    auto supports_device(Device::Type device_type) const -> bool override;

private:
    int rank_{-1};
    int world_size_{-1};
    bool initialized_{false};
    bool owns_mpi_init_{false};  ///< True if we called MPI_Init ourselves
    bool cuda_aware_{false};     ///< True if MPI supports GPU-direct

#ifdef TENZOR_HAS_MPI
    MPI_Comm comm_{MPI_COMM_WORLD};
#endif

    // Helper methods

    /**
     * @brief Convert ReduceOp to MPI_Op.
     */
    auto to_mpi_op(ReduceOp op) -> MPI_Op;

    /**
     * @brief Convert DType to MPI_Datatype.
     */
    auto to_mpi_datatype(DType dtype) -> MPI_Datatype;

    /**
     * @brief Detect if the MPI implementation is CUDA-aware.
     *
     * Checks the MPIX_CUDA_AWARE_SUPPORT macro (OpenMPI) and/or the
     * TENZOR_MPI_CUDA_AWARE environment variable.
     */
    auto detect_cuda_aware() -> bool;

    /**
     * @brief Get a host-accessible pointer for a tensor.
     *
     * If the tensor is on CPU, returns data_ptr() directly.
     * If on GPU and MPI is CUDA-aware, returns data_ptr() directly.
     * Otherwise, copies to a pinned host buffer and returns its pointer.
     *
     * @param tensor Input tensor
     * @param host_buf Output: host buffer (populated if staging was needed)
     * @return Pointer suitable for passing to MPI
     */
    auto get_send_ptr(const Tensor& tensor, Tensor& host_buf) -> const void*;

    /**
     * @brief Get a host-accessible pointer for receiving into a tensor.
     *
     * Similar to get_send_ptr but for receive operations. If staging is
     * needed, allocates a host buffer; caller must copy back to GPU after
     * the MPI call completes.
     *
     * @param tensor Target tensor
     * @param host_buf Output: host buffer (populated if staging was needed)
     * @return Pointer suitable for passing to MPI
     */
    auto get_recv_ptr(Tensor& tensor, Tensor& host_buf) -> void*;

    /**
     * @brief Copy staged host buffer back to GPU tensor if needed.
     */
    auto copy_back_if_staged(Tensor& tensor, const Tensor& host_buf) -> void;

    /**
     * @brief Validate MPI is initialized before operations.
     */
    auto validate_initialized() const -> void;
};

} // namespace distributed
} // namespace tenzor
