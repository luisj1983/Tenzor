/**
 * @file mpi_backend.cpp
 * @brief Implementation of MPI communication backend
 *
 * Implements collective and point-to-point operations using MPI.
 * GPU tensors are staged through host memory when MPI is not CUDA-aware.
 */

#include "tenzor/distributed/mpi_backend.hpp"
#include "tenzor/utils/error.hpp"  // NotImplementedError (S25 / audit-12) — needed in both branches

#ifdef TENZOR_HAS_MPI

#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/math.hpp"  // round() for integer ReduceOp::AVG rounding
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <iostream>  // D.2: dtor error logging
#include "tenzor/utils/log.hpp"  // D.1: TENZOR_LOG_ERROR

namespace tenzor {
namespace distributed {

// ============================================================================
// MPI error checking macro
// ============================================================================

#define MPI_CHECK(call) \
    do { \
        int mpi_err = (call); \
        if (mpi_err != MPI_SUCCESS) { \
            char err_string[MPI_MAX_ERROR_STRING]; \
            int err_len = 0; \
            MPI_Error_string(mpi_err, err_string, &err_len); \
            throw std::runtime_error( \
                std::string("MPI error: ") + std::string(err_string, err_len) + \
                " at " + __FILE__ + ":" + std::to_string(__LINE__) \
            ); \
        } \
    } while (0)

namespace {

// MPI's classic collective APIs take the per-rank element count as an `int`.
// tensor.numel() is int64; for >2^31-element tensors the narrowing cast would
// silently truncate and MPI would transfer fewer elements, corrupting the
// result with no error. Validate the range and throw a clear message instead.
auto checked_mpi_count(int64_t numel, const char* op) -> int {
    if (numel < 0 || numel > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            std::string("MPIBackend::") + op + ": element count " +
            std::to_string(numel) + " exceeds INT_MAX; tensors with more than " +
            std::to_string(std::numeric_limits<int>::max()) +
            " elements require MPI large-count APIs or chunking");
    }
    return static_cast<int>(numel);
}

// ReduceOp::AVG has no native MPI collective; it is implemented as MPI_SUM
// followed by an element-wise division by world_size. Divide IN PLACE through
// the caller's existing storage so aliased Variables/views/grads are preserved
// (the H6 aliasing fix). For floating/complex dtypes a same-dtype scalar divide
// keeps full precision. For integer dtypes a same-dtype divide would truncate
// toward zero (integer division), so widen to Float64, divide, round to nearest,
// then narrow back and write the result through the original storage.
auto apply_avg_in_place(Tensor& tensor, int world_size) -> void {
    if (world_size <= 0) {
        return;
    }

    if (is_integer_type(tensor.dtype())) {
        const DType orig = tensor.dtype();
        Tensor widened = tensor.to(DType::Float64);
        auto divisor = tenzor::full({1}, static_cast<double>(world_size),
                                    DType::Float64, widened.device());
        widened /= divisor;
        Tensor rounded = tenzor::round(widened).to(orig);
        // Write back through existing storage (zero_() + operator+=) to preserve
        // aliasing; plain assignment would rebind the impl and detach views.
        tensor.zero_();
        tensor += rounded.to(tensor.device());
        return;
    }

    auto scalar = tenzor::full({1}, static_cast<double>(world_size),
                               tensor.dtype(), tensor.device());
    tensor /= scalar;
}

} // anonymous namespace

// ============================================================================
// Construction / destruction
// ============================================================================

MPIBackend::MPIBackend() = default;

MPIBackend::~MPIBackend() {
    // D.2: log destructor finalize failures via tenzor logger (D.1).
    try {
        finalize();
    } catch (const std::exception& e) {
        TENZOR_LOG_ERROR("MPIBackend::~MPIBackend: finalize() threw: {}",
                         e.what());
    } catch (...) {
        TENZOR_LOG_ERROR(
            "MPIBackend::~MPIBackend: finalize() threw an unknown "
            "exception type");
    }
}

// ============================================================================
// Initialization
// ============================================================================

auto MPIBackend::initialize(int rank, int world_size,
                            const std::string& /*master_addr*/,
                            int /*master_port*/) -> void
{
    if (initialized_) {
        throw std::runtime_error("MPIBackend: already initialized");
    }

    // Initialize MPI if not already done
    int mpi_initialized = 0;
    MPI_CHECK(MPI_Initialized(&mpi_initialized));

    if (!mpi_initialized) {
        int provided = 0;
        MPI_CHECK(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided));
        if (provided < MPI_THREAD_MULTIPLE) {
            throw std::runtime_error(
                "MPIBackend: MPI_THREAD_MULTIPLE not supported by this MPI "
                "implementation (got thread level " + std::to_string(provided) +
                "). Multi-threaded operation may be unsafe."
            );
        }
        owns_mpi_init_ = true;
    }

    // Query actual rank and world_size from MPI
    int mpi_rank = 0;
    int mpi_world_size = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size));

    // Validate against caller-provided values
    if (rank >= 0 && rank != mpi_rank) {
        throw std::invalid_argument(
            "MPIBackend: requested rank " + std::to_string(rank) +
            " but MPI reports rank " + std::to_string(mpi_rank)
        );
    }
    if (world_size > 0 && world_size != mpi_world_size) {
        throw std::invalid_argument(
            "MPIBackend: requested world_size " + std::to_string(world_size) +
            " but MPI reports " + std::to_string(mpi_world_size)
        );
    }

    rank_ = mpi_rank;
    world_size_ = mpi_world_size;
    comm_ = MPI_COMM_WORLD;

    // Detect CUDA-aware MPI
    cuda_aware_ = detect_cuda_aware();

    initialized_ = true;
}

// ============================================================================
// Collective operations
// ============================================================================

auto MPIBackend::broadcast(Tensor& tensor, int src_rank) -> void {
    validate_initialized();

    Tensor host_buf;
    void* ptr = get_recv_ptr(tensor, host_buf);

    MPI_CHECK(MPI_Bcast(
        ptr,
        checked_mpi_count(tensor.numel(), "broadcast"),
        to_mpi_datatype(tensor.dtype()),
        src_rank,
        comm_
    ));

    copy_back_if_staged(tensor, host_buf);
}

auto MPIBackend::all_reduce(Tensor& tensor, ReduceOp op) -> void {
    validate_initialized();

    Tensor host_buf;
    void* ptr = get_recv_ptr(tensor, host_buf);

    MPI_CHECK(MPI_Allreduce(
        MPI_IN_PLACE,
        ptr,
        checked_mpi_count(tensor.numel(), "all_reduce"),
        to_mpi_datatype(tensor.dtype()),
        to_mpi_op(op),
        comm_
    ));

    copy_back_if_staged(tensor, host_buf);

    // Handle AVG: MPI doesn't have a native average op. apply_avg_in_place
    // divides IN PLACE through the caller's storage (preserving aliased
    // Variables/views — the H6 fix) and computes integer averages in a Float64
    // accumulator with round-to-nearest instead of truncating via integer
    // division.
    if (op == ReduceOp::AVG) {
        apply_avg_in_place(tensor, world_size_);
    }
}

auto MPIBackend::reduce(Tensor& tensor, int dst_rank, ReduceOp op) -> void {
    validate_initialized();

    Tensor host_buf;
    void* ptr = get_recv_ptr(tensor, host_buf);

    if (rank_ == dst_rank) {
        MPI_CHECK(MPI_Reduce(
            MPI_IN_PLACE,
            ptr,
            checked_mpi_count(tensor.numel(), "reduce"),
            to_mpi_datatype(tensor.dtype()),
            to_mpi_op(op),
            dst_rank,
            comm_
        ));
    } else {
        MPI_CHECK(MPI_Reduce(
            ptr,
            nullptr,
            checked_mpi_count(tensor.numel(), "reduce"),
            to_mpi_datatype(tensor.dtype()),
            to_mpi_op(op),
            dst_rank,
            comm_
        ));
    }

    copy_back_if_staged(tensor, host_buf);

    // MPI has no AVG collective (ReduceOp::AVG maps to MPI_SUM); only the
    // destination rank holds the reduced result, so divide there to get the
    // mean. Non-root ranks keep their unchanged local tensor. apply_avg_in_place
    // preserves aliasing and rounds integer averages instead of truncating.
    if (op == ReduceOp::AVG && rank_ == dst_rank) {
        apply_avg_in_place(tensor, world_size_);
    }
}

auto MPIBackend::all_gather(const Tensor& tensor, std::vector<Tensor>& output) -> void {
    validate_initialized();

    if (output.size() != static_cast<size_t>(world_size_)) {
        throw std::invalid_argument(
            "all_gather: output size must equal world_size"
        );
    }

    Tensor send_host_buf;
    // Intel MPI declares the send buffer parameter as non-const void*;
    // OpenMPI uses const void*. get_send_ptr() returns const to reflect
    // that we don't modify the sender, so cast it away at the API
    // boundary (MPI is not supposed to write into this buffer anyway).
    void* send_ptr = const_cast<void*>(get_send_ptr(tensor, send_host_buf));

    // Allocate contiguous receive buffer
    size_t total_elements = tensor.numel() * world_size_;
    Tensor recv_buf = empty(
        {static_cast<int64_t>(total_elements)},
        tensor.dtype(),
        Device::cpu()
    );

    const int gather_count = checked_mpi_count(tensor.numel(), "all_gather");
    MPI_CHECK(MPI_Allgather(
        send_ptr,
        gather_count,
        to_mpi_datatype(tensor.dtype()),
        recv_buf.data_ptr(),
        gather_count,
        to_mpi_datatype(tensor.dtype()),
        comm_
    ));

    // Split into output tensors
    for (int i = 0; i < world_size_; ++i) {
        Tensor chunk = recv_buf.slice(
            0,
            i * tensor.numel(),
            (i + 1) * tensor.numel()
        );
        // Move to the same device as the input tensor
        if (tensor.device().type != Device::Type::CPU) {
            output[i] = chunk.to(tensor.device());
        } else {
            output[i] = chunk;
        }
    }
}

auto MPIBackend::gather(const Tensor& tensor, std::vector<Tensor>& output,
                        int dst_rank) -> void {
    validate_initialized();

    if (dst_rank < 0 || dst_rank >= world_size_) {
        throw std::invalid_argument("gather: invalid dst_rank");
    }

    Tensor send_host_buf;
    void* send_ptr = const_cast<void*>(get_send_ptr(tensor, send_host_buf));

    if (rank_ == dst_rank) {
        if (output.size() != static_cast<size_t>(world_size_)) {
            throw std::invalid_argument(
                "gather: output size must equal world_size on dst_rank"
            );
        }

        // Allocate contiguous receive buffer
        size_t total_elements = tensor.numel() * world_size_;
        Tensor recv_buf = empty(
            {static_cast<int64_t>(total_elements)},
            tensor.dtype(),
            Device::cpu()
        );

        const int gather_count = checked_mpi_count(tensor.numel(), "gather");
        MPI_CHECK(MPI_Gather(
            send_ptr,
            gather_count,
            to_mpi_datatype(tensor.dtype()),
            recv_buf.data_ptr(),
            gather_count,
            to_mpi_datatype(tensor.dtype()),
            dst_rank,
            comm_
        ));

        // Split into output tensors
        for (int i = 0; i < world_size_; ++i) {
            Tensor chunk = recv_buf.slice(
                0,
                i * tensor.numel(),
                (i + 1) * tensor.numel()
            );
            if (tensor.device().type != Device::Type::CPU) {
                output[i] = chunk.to(tensor.device());
            } else {
                output[i] = chunk;
            }
        }
    } else {
        MPI_CHECK(MPI_Gather(
            send_ptr,
            checked_mpi_count(tensor.numel(), "gather"),
            to_mpi_datatype(tensor.dtype()),
            nullptr,
            0,
            to_mpi_datatype(tensor.dtype()),
            dst_rank,
            comm_
        ));
    }
}

auto MPIBackend::scatter(const std::vector<Tensor>& tensors, Tensor& output,
                         int src_rank) -> void {
    validate_initialized();

    if (src_rank < 0 || src_rank >= world_size_) {
        throw std::invalid_argument("scatter: invalid src_rank");
    }

    Tensor recv_host_buf;
    void* recv_ptr = get_recv_ptr(output, recv_host_buf);

    if (rank_ == src_rank) {
        if (tensors.size() != static_cast<size_t>(world_size_)) {
            throw std::invalid_argument(
                "scatter: tensors size must equal world_size on src_rank"
            );
        }

        // Build contiguous send buffer
        size_t total_elements = output.numel() * world_size_;
        Tensor send_buf = empty(
            {static_cast<int64_t>(total_elements)},
            output.dtype(),
            Device::cpu()
        );

        for (int i = 0; i < world_size_; ++i) {
            // Validate each chunk matches the per-rank output exactly before
            // copying. A smaller input would cause an out-of-bounds read; a
            // larger one would be silently truncated. (The NCCL scatter path
            // performs the same check.)
            if (tensors[i].numel() != output.numel()) {
                throw std::invalid_argument(
                    "scatter: tensors[" + std::to_string(i) + "] numel " +
                    std::to_string(tensors[i].numel()) +
                    " does not match output numel " +
                    std::to_string(output.numel()));
            }
            if (tensors[i].dtype() != output.dtype()) {
                throw std::invalid_argument(
                    "scatter: tensors[" + std::to_string(i) +
                    "] dtype does not match output dtype");
            }
            Tensor src_cpu = tensors[i];
            if (src_cpu.device().type != Device::Type::CPU) {
                src_cpu = src_cpu.to(Device::cpu());
            }
            // .to(cpu()) preserves strides; ensure contiguous before the flat
            // memcpy so a non-contiguous source is not misread.
            src_cpu = src_cpu.contiguous();
            std::memcpy(
                static_cast<char*>(send_buf.data_ptr()) +
                    i * output.numel() * output.element_size(),
                src_cpu.data_ptr(),
                output.numel() * output.element_size()
            );
        }

        const int scatter_count = checked_mpi_count(output.numel(), "scatter");
        MPI_CHECK(MPI_Scatter(
            send_buf.data_ptr(),
            scatter_count,
            to_mpi_datatype(output.dtype()),
            recv_ptr,
            scatter_count,
            to_mpi_datatype(output.dtype()),
            src_rank,
            comm_
        ));
    } else {
        MPI_CHECK(MPI_Scatter(
            nullptr,
            0,
            to_mpi_datatype(output.dtype()),
            recv_ptr,
            checked_mpi_count(output.numel(), "scatter"),
            to_mpi_datatype(output.dtype()),
            src_rank,
            comm_
        ));
    }

    copy_back_if_staged(output, recv_host_buf);
}

auto MPIBackend::reduce_scatter(const std::vector<Tensor>& tensors,
                                Tensor& output, ReduceOp op) -> void {
    validate_initialized();

    if (tensors.empty()) {
        throw std::invalid_argument("reduce_scatter: tensors cannot be empty");
    }

    // Concatenate input tensors into contiguous buffer
    std::vector<Tensor> cpu_tensors;
    cpu_tensors.reserve(tensors.size());
    for (const auto& t : tensors) {
        if (t.device().type != Device::Type::CPU) {
            cpu_tensors.push_back(t.to(Device::cpu()));
        } else {
            cpu_tensors.push_back(t);
        }
    }
    // Validate dtype of every chunk before concatenation: cat() would otherwise
    // either throw obscurely or coerce, and MPI requires a uniform datatype.
    for (size_t i = 0; i < cpu_tensors.size(); ++i) {
        if (cpu_tensors[i].dtype() != output.dtype()) {
            throw std::invalid_argument(
                "reduce_scatter: input tensor dtype does not match output dtype");
        }
    }

    Tensor send_buf = cat(cpu_tensors, 0);

    // MPI_Reduce_scatter_block requires the send buffer to hold exactly
    // output.numel() * world_size elements. Validate the caller's contract
    // explicitly instead of letting MPI perform an out-of-bounds read.
    // (The NCCL reduce_scatter path performs the same check.)
    if (send_buf.numel() != output.numel() * world_size_) {
        throw std::invalid_argument(
            "reduce_scatter: concatenated input numel " +
            std::to_string(send_buf.numel()) +
            " must equal output numel * world_size (" +
            std::to_string(output.numel()) + " * " +
            std::to_string(world_size_) + ")");
    }

    Tensor recv_host_buf;
    void* recv_ptr = get_recv_ptr(output, recv_host_buf);

    // MPI_Reduce_scatter_block: each rank gets an equal-sized chunk
    MPI_CHECK(MPI_Reduce_scatter_block(
        send_buf.data_ptr(),
        recv_ptr,
        checked_mpi_count(output.numel(), "reduce_scatter"),
        to_mpi_datatype(output.dtype()),
        to_mpi_op(op),
        comm_
    ));

    copy_back_if_staged(output, recv_host_buf);

    // MPI has no AVG collective (ReduceOp::AVG maps to MPI_SUM); divide IN PLACE
    // afterwards, mirroring MPIBackend::all_reduce. apply_avg_in_place preserves
    // precision and caller aliasing, and rounds integer averages instead of
    // truncating via integer division.
    if (op == ReduceOp::AVG) {
        apply_avg_in_place(output, world_size_);
    }
}

auto MPIBackend::barrier() -> void {
    validate_initialized();
    MPI_CHECK(MPI_Barrier(comm_));
}

// ============================================================================
// Point-to-point operations
// ============================================================================

auto MPIBackend::send(const Tensor& tensor, int dst_rank) -> void {
    validate_initialized();

    if (dst_rank < 0 || dst_rank >= world_size_) {
        throw std::invalid_argument(
            "send: invalid dst_rank " + std::to_string(dst_rank)
        );
    }

    Tensor host_buf;
    void* ptr = const_cast<void*>(get_send_ptr(tensor, host_buf));

    MPI_CHECK(MPI_Send(
        ptr,
        checked_mpi_count(tensor.numel(), "send"),
        to_mpi_datatype(tensor.dtype()),
        dst_rank,
        /*tag=*/0,
        comm_
    ));
}

auto MPIBackend::recv(Tensor& tensor, int src_rank) -> void {
    validate_initialized();

    if (src_rank < 0 || src_rank >= world_size_) {
        throw std::invalid_argument(
            "recv: invalid src_rank " + std::to_string(src_rank)
        );
    }

    Tensor host_buf;
    void* ptr = get_recv_ptr(tensor, host_buf);

    MPI_CHECK(MPI_Recv(
        ptr,
        checked_mpi_count(tensor.numel(), "recv"),
        to_mpi_datatype(tensor.dtype()),
        src_rank,
        /*tag=*/0,
        comm_,
        MPI_STATUS_IGNORE
    ));

    copy_back_if_staged(tensor, host_buf);
}

// ============================================================================
// Finalization
// ============================================================================

auto MPIBackend::finalize() -> void {
    if (!initialized_) {
        return;
    }

    if (owns_mpi_init_) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
    }

    initialized_ = false;
}

auto MPIBackend::supports_device(Device::Type device_type) const -> bool {
    // MPI natively supports CPU. GPU is supported via staging or CUDA-aware MPI.
    if (device_type == Device::Type::CPU) {
        return true;
    }
    if (device_type == Device::Type::CUDA || device_type == Device::Type::ROCm) {
        return true;  // We handle staging transparently
    }
    return false;
}

// ============================================================================
// Private helpers
// ============================================================================

auto MPIBackend::to_mpi_op(ReduceOp op) -> MPI_Op {
    switch (op) {
        case ReduceOp::SUM:
        case ReduceOp::AVG:   // Handle as SUM, then divide
            return MPI_SUM;
        case ReduceOp::PRODUCT:
            return MPI_PROD;
        case ReduceOp::MIN:
            return MPI_MIN;
        case ReduceOp::MAX:
            return MPI_MAX;
        case ReduceOp::BAND:
            return MPI_BAND;
        case ReduceOp::BOR:
            return MPI_BOR;
        case ReduceOp::BXOR:
            return MPI_BXOR;
        default:
            throw std::invalid_argument("Unsupported reduce operation for MPI");
    }
}

auto MPIBackend::to_mpi_datatype(DType dtype) -> MPI_Datatype {
    switch (dtype) {
        case DType::Float32:
            return MPI_FLOAT;
        case DType::Float64:
            return MPI_DOUBLE;
        case DType::Int32:
            return MPI_INT;
        case DType::Int64:
            return MPI_LONG_LONG;
        case DType::Int8:
            return MPI_INT8_T;
        case DType::Int16:
            return MPI_SHORT;
        case DType::UInt8:
            return MPI_UINT8_T;
        default:
            throw std::invalid_argument(
                "Unsupported dtype for MPI: " +
                std::to_string(static_cast<int>(dtype))
            );
    }
}

auto MPIBackend::detect_cuda_aware() -> bool {
    // Check environment variable override first
    const char* env = std::getenv("TENZOR_MPI_CUDA_AWARE");
    if (env) {
        return std::string(env) == "1" || std::string(env) == "true";
    }

    // Check OpenMPI's CUDA-aware support macro
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    return true;
#endif

    return false;
}

auto MPIBackend::get_send_ptr(const Tensor& tensor, Tensor& host_buf) -> const void* {
    if (tensor.device().type == Device::Type::CPU || cuda_aware_) {
        return tensor.data_ptr();
    }

    // Stage through host memory
    host_buf = tensor.to(Device::cpu());
    return host_buf.data_ptr();
}

auto MPIBackend::get_recv_ptr(Tensor& tensor, Tensor& host_buf) -> void* {
    if (tensor.device().type == Device::Type::CPU || cuda_aware_) {
        if (tensor.is_contiguous()) {
            return tensor.data_ptr();
        }
        // Non-contiguous: MPI treats the buffer as flat-contiguous, so a
        // strided view would be read/written incorrectly. Stage a contiguous
        // copy (preserving the current values for in-place ops) and copy back.
        host_buf = tensor.contiguous();
        return host_buf.data_ptr();
    }

    // Non-CUDA-aware MPI: stage through host memory. The receive buffer must be
    // initialized with the device tensor's CURRENT values, not left uninitialized.
    // The in-place collectives (all_reduce/reduce with MPI_IN_PLACE) read each
    // rank's local contribution FROM this same buffer; an uninitialized empty()
    // buffer would make the local rank contribute garbage to the reduction.
    // Copying device->host here is also correct for the overwrite-only paths
    // (broadcast root, scatter/gather recv) — the staged values are simply
    // overwritten by MPI before copy_back_if_staged writes them back.
    host_buf = tensor.to(Device::cpu()).contiguous();
    return host_buf.data_ptr();
}

auto MPIBackend::copy_back_if_staged(Tensor& tensor, const Tensor& host_buf) -> void {
    // Copy back whenever we staged into host_buf — either to return data from
    // the host bounce (non-cuda-aware GPU) or from the contiguous copy made
    // for a non-contiguous input.
    //
    // Write the staged result back THROUGH the caller's existing storage rather
    // than rebinding the reference (`tensor = host_buf.to(...)`). Tensor's
    // copy-assignment is defaulted and rebinds the intrusive_ptr<TensorImpl> to
    // a fresh impl/storage, which detaches any Variable/view/grad aliasing the
    // original buffer (the same H6 aliasing hazard the all_reduce AVG path was
    // rewritten to avoid). Tenzor's Tensor does not expose `copy_`, so use the
    // established in-place primitive pair (zero_() + operator+=) — both write
    // through the existing strides/storage, preserving aliases, and handle every
    // supported dtype the collectives accept.
    if (host_buf.data_ptr() != nullptr) {
        tensor.zero_();
        tensor += host_buf.to(tensor.device());
    }
}

auto MPIBackend::validate_initialized() const -> void {
    if (!initialized_) {
        throw std::runtime_error(
            "MPIBackend: not initialized. Call initialize() first."
        );
    }
}

} // namespace distributed
} // namespace tenzor

#else // !TENZOR_HAS_MPI

// Provide stub implementation that throws at runtime
namespace tenzor {
namespace distributed {

MPIBackend::MPIBackend() = default;
MPIBackend::~MPIBackend() = default;

auto MPIBackend::initialize(int /*rank*/, int /*world_size*/,
                            const std::string& /*master_addr*/,
                            int /*master_port*/) -> void {
    throw std::runtime_error(
        "MPIBackend: MPI support not available. "
        "Please rebuild with TENZOR_HAS_MPI=ON and a valid MPI installation."
    );
}

auto MPIBackend::broadcast(Tensor&, int) -> void {
    throw NotImplementedError("MPIBackend: MPI not available");
}

auto MPIBackend::all_reduce(Tensor&, ReduceOp) -> void {
    throw NotImplementedError("MPIBackend: MPI not available");
}

auto MPIBackend::reduce(Tensor&, int, ReduceOp) -> void {
    throw NotImplementedError("MPIBackend: MPI not available");
}

auto MPIBackend::all_gather(const Tensor&, std::vector<Tensor>&) -> void {
    throw NotImplementedError("MPIBackend: MPI not available");
}

auto MPIBackend::gather(const Tensor&, std::vector<Tensor>&, int) -> void {
    throw NotImplementedError("MPIBackend: MPI not available");
}

auto MPIBackend::scatter(const std::vector<Tensor>&, Tensor&, int) -> void {
    throw NotImplementedError("MPIBackend: MPI not available");
}

auto MPIBackend::reduce_scatter(const std::vector<Tensor>&, Tensor&, ReduceOp) -> void {
    throw NotImplementedError("MPIBackend: MPI not available");
}

auto MPIBackend::barrier() -> void {
    throw NotImplementedError("MPIBackend: MPI not available");
}

auto MPIBackend::send(const Tensor&, int) -> void {
    throw NotImplementedError("MPIBackend: MPI not available");
}

auto MPIBackend::recv(Tensor&, int) -> void {
    throw NotImplementedError("MPIBackend: MPI not available");
}

auto MPIBackend::finalize() -> void {}

auto MPIBackend::supports_device(Device::Type) const -> bool {
    return false;
}

// Private helpers - stubs
auto MPIBackend::to_mpi_op(ReduceOp) -> MPI_Op { return 0; }
auto MPIBackend::to_mpi_datatype(DType) -> MPI_Datatype { return 0; }
auto MPIBackend::detect_cuda_aware() -> bool { return false; }
auto MPIBackend::get_send_ptr(const Tensor&, Tensor&) -> const void* { return nullptr; }
auto MPIBackend::get_recv_ptr(Tensor&, Tensor&) -> void* { return nullptr; }
auto MPIBackend::copy_back_if_staged(Tensor&, const Tensor&) -> void {}
auto MPIBackend::validate_initialized() const -> void {}

} // namespace distributed
} // namespace tenzor

#endif // TENZOR_HAS_MPI
