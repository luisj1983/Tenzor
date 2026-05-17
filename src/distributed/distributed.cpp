/**
 * @file distributed.cpp
 * @brief Implementation of distributed training API and process group
 */

#include "tenzor/distributed/distributed.hpp"
#if defined(TENZOR_HAS_NCCL)
#include "tenzor/distributed/nccl_backend.hpp"
#endif
#include "tenzor/distributed/gloo_backend.hpp"
#ifdef TENZOR_HAS_MPI
#include "tenzor/distributed/mpi_backend.hpp"
#endif
#include "tenzor/utils/error.hpp"
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <iostream>  // D.2: dtor error logging
#include "tenzor/utils/log.hpp"  // D.1: TENZOR_LOG_ERROR

namespace tenzor {
namespace distributed {

// ============================================================================
// Backend conversion utilities
// ============================================================================

auto backend_to_string(Backend backend) -> std::string {
    switch (backend) {
        case Backend::NCCL: return "nccl";
        case Backend::GLOO: return "gloo";
        case Backend::MPI: return "mpi";
        default: return "unknown";
    }
}

auto string_to_backend(const std::string& backend) -> Backend {
    if (backend == "nccl") return Backend::NCCL;
    if (backend == "gloo") return Backend::GLOO;
    if (backend == "mpi") return Backend::MPI;
    throw std::invalid_argument("Unknown backend: " + backend);
}

// ============================================================================
// ProcessGroup Implementation
// ============================================================================

ProcessGroup::ProcessGroup(std::unique_ptr<CommunicationBackend> backend,
                          int rank, int world_size)
    : backend_(std::move(backend)), rank_(rank), world_size_(world_size) {

    if (rank_ < 0 || rank_ >= world_size_) {
        throw std::invalid_argument(
            "ProcessGroup: rank " + std::to_string(rank_) +
            " must be in range [0, " + std::to_string(world_size_) + ")"
        );
    }

    if (world_size_ <= 0) {
        throw std::invalid_argument(
            "ProcessGroup: world_size must be positive, got " +
            std::to_string(world_size_)
        );
    }

    if (!backend_) {
        throw std::invalid_argument("ProcessGroup: backend cannot be null");
    }
}

ProcessGroup::~ProcessGroup() {
    if (backend_) {
        // D.2: log destructor finalize failures via tenzor logger (D.1).
        try {
            backend_->finalize();
        } catch (const std::exception& e) {
            TENZOR_LOG_ERROR(
                "ProcessGroup::~ProcessGroup: backend finalize() threw: {}",
                e.what());
        } catch (...) {
            TENZOR_LOG_ERROR(
                "ProcessGroup::~ProcessGroup: backend finalize() threw an "
                "unknown exception type");
        }
    }
}

auto ProcessGroup::broadcast(Tensor& tensor, int src_rank) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_->broadcast(tensor, src_rank);
}

auto ProcessGroup::all_reduce(Tensor& tensor, ReduceOp op) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_->all_reduce(tensor, op);
}

auto ProcessGroup::all_reduce_async(Tensor& tensor, ReduceOp op,
                                     void* stream) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_->all_reduce_async(tensor, op, stream);
}

auto ProcessGroup::supports_async_stream() const -> bool {
    return backend_->supports_async_stream();
}

auto ProcessGroup::reduce(Tensor& tensor, int dst_rank, ReduceOp op) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_->reduce(tensor, dst_rank, op);
}

auto ProcessGroup::all_gather(const Tensor& tensor, std::vector<Tensor>& output) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_->all_gather(tensor, output);
}

auto ProcessGroup::all_gather_async(const Tensor& tensor, std::vector<Tensor>& output,
                                     void* stream) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_->all_gather_async(tensor, output, stream);
}

auto ProcessGroup::gather(const Tensor& tensor, std::vector<Tensor>& output, int dst_rank) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_->gather(tensor, output, dst_rank);
}

auto ProcessGroup::scatter(const std::vector<Tensor>& tensors, Tensor& output, int src_rank) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_->scatter(tensors, output, src_rank);
}

auto ProcessGroup::reduce_scatter(const std::vector<Tensor>& tensors, Tensor& output, ReduceOp op) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_->reduce_scatter(tensors, output, op);
}

// Phase E (E1): async stream-based reduce-scatter wrapper. Backend default
// impl is the sync fallback (ignores stream); NCCL/RCCL backends override
// to launch on the user-provided stream so the collective overlaps with
// default-stream compute.
auto ProcessGroup::reduce_scatter_async(const std::vector<Tensor>& tensors, Tensor& output,
                                        ReduceOp op, void* stream) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_->reduce_scatter_async(tensors, output, op, stream);
}

// A4-extended: all-to-all single forwarder. Default backend impl is the
// composed fallback (all_gather + slice); NCCL overrides with native
// ncclGroupStart + ncclSend/Recv pairs (see NCCLBackend).
auto ProcessGroup::all_to_all_single(Tensor& output, const Tensor& input) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_->all_to_all_single(output, input);
}

// A4-extended: default `all_to_all_single` for backends without a native
// primitive. CommunicationBackend itself has no rank/world_size accessor
// (those live on ProcessGroup), so the default fallback can only handle
// the trivial world_size == 1 / single-rank case via memcpy. Multi-rank
// callers MUST go through a backend that overrides this method (NCCL),
// or use the newer `ProcessGroupBase` hierarchy where the
// `all_gather + slice` fallback is rank-aware.
auto CommunicationBackend::all_to_all_single(Tensor& output, const Tensor& input) -> void {
    auto input_shape = input.shape();
    if (input_shape.empty()) {
        throw std::invalid_argument(
            "CommunicationBackend::all_to_all_single: input must have ≥ 1 dimension.");
    }
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());
    std::vector<int64_t> out_shape_vec(output.shape().begin(), output.shape().end());
    if (in_shape_vec != out_shape_vec) {
        throw std::invalid_argument(
            "CommunicationBackend::all_to_all_single: output and input must have identical shape.");
    }
    if (output.dtype() != input.dtype()) {
        throw std::invalid_argument(
            "CommunicationBackend::all_to_all_single: output and input must have identical dtype.");
    }

    // The trivial single-rank case is a memcpy. For multi-rank, the
    // backend must override.
    int64_t world_size_guess = input_shape[0];
    if (world_size_guess == 1) {
        std::memcpy(output.data_ptr(),
                    input.data_ptr(),
                    static_cast<size_t>(input.numel()) * dtype_size(input.dtype()));
        return;
    }

    throw std::runtime_error(
        "CommunicationBackend::all_to_all_single: no default fallback for "
        "world_size > 1 (this backend lacks a rank accessor). Use NCCL which "
        "overrides this method natively, or switch to the ProcessGroupBase "
        "hierarchy whose default does support multi-rank via all_gather + slice.");
}

auto ProcessGroup::send(const Tensor& tensor, int dst_rank) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_->send(tensor, dst_rank);
}

auto ProcessGroup::recv(Tensor& tensor, int src_rank) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_->recv(tensor, src_rank);
}

auto ProcessGroup::barrier() -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_->barrier();
}

auto ProcessGroup::create_from_env(Backend backend) -> std::shared_ptr<ProcessGroup> {
    // Read environment variables
    const char* rank_env = std::getenv("RANK");
    const char* world_size_env = std::getenv("WORLD_SIZE");
    const char* master_addr_env = std::getenv("MASTER_ADDR");
    const char* master_port_env = std::getenv("MASTER_PORT");

    if (!rank_env || !world_size_env) {
        throw std::runtime_error(
            "ProcessGroup::create_from_env: RANK and WORLD_SIZE environment "
            "variables must be set"
        );
    }

    int rank = std::atoi(rank_env);
    int world_size = std::atoi(world_size_env);
    std::string master_addr = master_addr_env ? master_addr_env : "localhost";
    int master_port = master_port_env ? std::atoi(master_port_env) : 29500;

    return create_process_group(backend, rank, world_size, master_addr, master_port);
}

auto ProcessGroup::create_process_group(
    Backend backend,
    int rank,
    int world_size,
    const std::string& master_addr,
    int master_port
) -> std::shared_ptr<ProcessGroup> {
    // Validate parameters before backend initialization
    if (rank < 0 || rank >= world_size) {
        throw std::invalid_argument(
            "ProcessGroup: rank must be in range [0, world_size). "
            "Got rank=" + std::to_string(rank) + ", world_size=" + std::to_string(world_size)
        );
    }

    if (world_size <= 0) {
        throw std::invalid_argument(
            "ProcessGroup: world_size must be positive. "
            "Got world_size=" + std::to_string(world_size)
        );
    }

    std::unique_ptr<CommunicationBackend> comm_backend;

    switch (backend) {
        case Backend::NCCL:
#if defined(TENZOR_HAS_NCCL)
            comm_backend = std::make_unique<NCCLBackend>();
#else
            throw std::runtime_error("NCCL backend requires CUDA/ROCm with NCCL installed");
#endif
            break;
        case Backend::GLOO:
            comm_backend = std::make_unique<GlooBackend>();
            break;
        case Backend::MPI:
#ifdef TENZOR_HAS_MPI
            comm_backend = std::make_unique<MPIBackend>();
            break;
#else
            throw std::runtime_error(
                "MPI backend requested but Tenzor was built without MPI support. "
                "Rebuild with -DTENZOR_BUILD_MPI=ON after installing an MPI "
                "implementation (OpenMPI, MPICH, or Intel MPI). Launch MPI "
                "programs with mpirun/mpiexec, e.g. `mpirun -n 4 ./your_program`.\n"
                "\n"
                "Alternative backends that do not require a rebuild:\n"
                "  - NCCL: High-performance GPU-to-GPU communication (requires CUDA/ROCm + NCCL)\n"
                "  - Gloo: CPU-based communication over TCP/IP sockets"
            );
#endif
        default:
            throw std::invalid_argument("Unknown backend");
    }

    // Initialize backend
    comm_backend->initialize(rank, world_size, master_addr, master_port);

    return std::make_shared<ProcessGroup>(
        std::move(comm_backend), rank, world_size
    );
}

// ============================================================================
// GradientBucket Implementation
// ============================================================================

GradientBucket::GradientBucket(size_t max_size_mb)
    : max_size_bytes_(max_size_mb * 1024 * 1024) {
}

auto GradientBucket::add_gradient(const Tensor& gradient) -> bool {
    size_t gradient_bytes = gradient.numel() * dtype_size(gradient.dtype());
    gradients_.push_back(gradient);
    size_bytes_ += gradient_bytes;
    return is_full();
}

auto GradientBucket::reset() -> void {
    gradients_.clear();
    size_bytes_ = 0;
}

// ============================================================================
// DistributedContext Implementation
// ============================================================================

std::shared_ptr<ProcessGroup> DistributedContext::global_process_group_;
std::mutex DistributedContext::mutex_;

auto DistributedContext::initialize(
    Backend backend,
    int rank,
    int world_size,
    const std::string& master_addr,
    int master_port
) -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    if (global_process_group_) {
        throw std::runtime_error(
            "DistributedContext: already initialized. Call finalize() first."
        );
    }

    global_process_group_ = ProcessGroup::create_process_group(
        backend, rank, world_size, master_addr, master_port
    );
}

auto DistributedContext::initialize_from_env(Backend backend) -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    if (global_process_group_) {
        throw std::runtime_error(
            "DistributedContext: already initialized. Call finalize() first."
        );
    }

    global_process_group_ = ProcessGroup::create_from_env(backend);
}

auto DistributedContext::get_process_group() -> std::shared_ptr<ProcessGroup> {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!global_process_group_) {
        throw std::runtime_error(
            "DistributedContext: not initialized. Call initialize() first."
        );
    }

    return global_process_group_;
}

auto DistributedContext::is_initialized() -> bool {
    std::lock_guard<std::mutex> lock(mutex_);
    return global_process_group_ != nullptr;
}

auto DistributedContext::get_rank() -> int {
    auto pg = get_process_group();
    return pg->rank();
}

auto DistributedContext::get_world_size() -> int {
    auto pg = get_process_group();
    return pg->world_size();
}

auto DistributedContext::finalize() -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    global_process_group_.reset();
}

// ============================================================================
// Convenience Functions
// ============================================================================

auto init_process_group(
    const std::string& backend,
    int rank,
    int world_size,
    const std::string& master_addr,
    int master_port
) -> void {

    Backend backend_enum = string_to_backend(backend);

    if (rank == -1 || world_size == -1) {
        // Read from environment
        DistributedContext::initialize_from_env(backend_enum);
    } else {
        DistributedContext::initialize(
            backend_enum, rank, world_size, master_addr, master_port
        );
    }
}

auto destroy_process_group() -> void {
    DistributedContext::finalize();
}

auto get_rank() -> int {
    return DistributedContext::get_rank();
}

auto get_world_size() -> int {
    return DistributedContext::get_world_size();
}

auto is_initialized() -> bool {
    return DistributedContext::is_initialized();
}

auto barrier() -> void {
    auto pg = DistributedContext::get_process_group();
    pg->barrier();
}

auto all_reduce(Tensor& tensor, ReduceOp op) -> void {
    auto pg = DistributedContext::get_process_group();
    pg->all_reduce(tensor, op);
}

auto broadcast(Tensor& tensor, int src_rank) -> void {
    auto pg = DistributedContext::get_process_group();
    pg->broadcast(tensor, src_rank);
}

auto send(const Tensor& tensor, int dst_rank) -> void {
    auto pg = DistributedContext::get_process_group();
    pg->send(tensor, dst_rank);
}

auto recv(Tensor& tensor, int src_rank) -> void {
    auto pg = DistributedContext::get_process_group();
    pg->recv(tensor, src_rank);
}

} // namespace distributed
} // namespace tenzor
