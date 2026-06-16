/**
 * @file nccl_backend.cpp
 * @brief Implementation of NCCL backend for GPU communication
 */

#include "tenzor/distributed/nccl_backend.hpp"
#include "tenzor/utils/error.hpp"  // NotImplementedError (S25 / audit-12) — needed in both branches

#if defined(TENZOR_HAS_NCCL)

#include "tenzor/utils/log.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include <stdexcept>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <iostream>
#include "tenzor/utils/log.hpp"  // D.1: TENZOR_LOG_ERROR

#if defined(TENZOR_USE_CUDA)
    #include <cuda_runtime.h>
    #define GPU_CHECK(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                throw std::runtime_error( \
                    std::string("CUDA error: ") + cudaGetErrorString(err) \
                ); \
            } \
        } while(0)
#elif defined(TENZOR_USE_ROCM)
    #include <hip/hip_runtime.h>
    #define GPU_CHECK(call) \
        do { \
            hipError_t err = call; \
            if (err != hipSuccess) { \
                throw std::runtime_error( \
                    std::string("HIP error: ") + hipGetErrorString(err) \
                ); \
            } \
        } while(0)
#else
    #define GPU_CHECK(call) call
#endif

namespace tenzor {
namespace distributed {

// ============================================================================
// NCCLBackend Implementation
// ============================================================================

NCCLBackend::NCCLBackend() = default;

NCCLBackend::~NCCLBackend() {
    // D.2: log destructor finalize failures via the central tenzor logger
    // (D.1). Throwing from a destructor is forbidden (terminates the
    // process), but silent swallow made hung shutdowns invisible.
    try {
        finalize();
    } catch (const std::exception& e) {
        TENZOR_LOG_ERROR("NCCLBackend::~NCCLBackend: finalize() threw: {}",
                         e.what());
    } catch (...) {
        TENZOR_LOG_ERROR(
            "NCCLBackend::~NCCLBackend: finalize() threw an unknown "
            "exception type");
    }
}

auto NCCLBackend::initialize(
    int rank,
    int world_size,
    const std::string& master_addr,
    int master_port
) -> void {

    if (initialized_) {
        throw std::runtime_error("NCCLBackend: already initialized");
    }

    rank_ = rank;
    world_size_ = world_size;
    master_addr_ = master_addr;
    master_port_ = master_port;

#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    // Exchange NCCL unique ID across all ranks
    exchange_unique_id();
    initialized_ = true;
#else
    throw std::runtime_error(
        "NCCLBackend: NCCL requires CUDA or ROCm support. "
        "Please rebuild with TENZOR_BUILD_CUDA=ON or TENZOR_BUILD_ROCM=ON"
    );
#endif
}

auto NCCLBackend::broadcast(Tensor& tensor, int src_rank) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(tensor);
    int device_id = get_device_id(tensor);
    ncclComm_t comm = get_communicator(device_id);

    ncclDataType_t nccl_dtype = to_nccl_datatype(tensor.dtype());

    NCCL_CHECK(ncclBroadcast(
        tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        nccl_dtype,
        src_rank,
        comm,
        nullptr  // Use default stream
    ));

    // Synchronize to ensure completion
    GPU_CHECK(cudaDeviceSynchronize());
#else
    throw NotImplementedError("NCCLBackend: NCCL not available");
#endif
}

auto NCCLBackend::all_reduce(Tensor& tensor, ReduceOp op) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(tensor);
    int device_id = get_device_id(tensor);
    ncclComm_t comm = get_communicator(device_id);

    ncclDataType_t nccl_dtype = to_nccl_datatype(tensor.dtype());
    ncclRedOp_t nccl_op = to_nccl_reduce_op(op);

    NCCL_CHECK(ncclAllReduce(
        tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        nccl_dtype,
        nccl_op,
        comm,
        nullptr  // Use default stream
    ));

    // Synchronize to ensure completion
    GPU_CHECK(cudaDeviceSynchronize());

    // If AVG operation, divide by world size
    if (op == ReduceOp::AVG) {
        tensor = tensor / static_cast<float>(world_size_);
    }
#else
    throw NotImplementedError("NCCLBackend: NCCL not available");
#endif
}

auto NCCLBackend::all_reduce_async(Tensor& tensor, ReduceOp op,
                                    void* stream) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(tensor);
    int device_id = get_device_id(tensor);
    ncclComm_t comm = get_communicator(device_id);

    ncclDataType_t nccl_dtype = to_nccl_datatype(tensor.dtype());
    ncclRedOp_t nccl_op = to_nccl_reduce_op(op);

    // Launch NCCL all-reduce on the caller-provided stream.
    // Unlike all_reduce(), we do NOT call cudaDeviceSynchronize() --
    // the caller is responsible for synchronization via CUDA events.
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    NCCL_CHECK(ncclAllReduce(
        tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        nccl_dtype,
        nccl_op,
        comm,
        cuda_stream
    ));

    // Note: AVG handling (divide by world_size) is done by the caller
    // after sync, since we cannot safely do tensor math on an async
    // stream without additional synchronization.
#else
    (void)stream;
    throw NotImplementedError("NCCLBackend: NCCL not available");
#endif
}

auto NCCLBackend::reduce(Tensor& tensor, int dst_rank, ReduceOp op) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(tensor);
    int device_id = get_device_id(tensor);
    ncclComm_t comm = get_communicator(device_id);

    ncclDataType_t nccl_dtype = to_nccl_datatype(tensor.dtype());
    ncclRedOp_t nccl_op = to_nccl_reduce_op(op);

    NCCL_CHECK(ncclReduce(
        tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        nccl_dtype,
        nccl_op,
        dst_rank,
        comm,
        nullptr
    ));

    GPU_CHECK(cudaDeviceSynchronize());
#else
    throw NotImplementedError("NCCLBackend: NCCL not available");
#endif
}

auto NCCLBackend::all_gather(const Tensor& tensor, std::vector<Tensor>& output) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(tensor);
    int device_id = get_device_id(tensor);
    ncclComm_t comm = get_communicator(device_id);

    if (output.size() != static_cast<size_t>(world_size_)) {
        throw std::invalid_argument(
            "all_gather: output size must equal world_size"
        );
    }

    ncclDataType_t nccl_dtype = to_nccl_datatype(tensor.dtype());

    // NCCL all-gather expects contiguous output buffer
    // Create temporary contiguous buffer
    size_t total_elements = tensor.numel() * world_size_;

    Tensor gathered = empty(
        {static_cast<int64_t>(total_elements)},
        tensor.dtype(),
        tensor.device()
    );

    NCCL_CHECK(ncclAllGather(
        tensor.data_ptr(),
        gathered.data_ptr(),
        tensor.numel(),
        nccl_dtype,
        comm,
        nullptr
    ));

    GPU_CHECK(cudaDeviceSynchronize());

    // Split gathered tensor into output vector
    for (int i = 0; i < world_size_; ++i) {
        output[i] = gathered.slice(0, i * tensor.numel(), (i + 1) * tensor.numel());
    }
#else
    throw NotImplementedError("NCCLBackend: NCCL not available");
#endif
}

auto NCCLBackend::gather(const Tensor& tensor, std::vector<Tensor>& output, int dst_rank) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(tensor);

    if (dst_rank < 0 || dst_rank >= world_size_) {
        throw std::invalid_argument("gather: invalid dst_rank");
    }

    int device_id = get_device_id(tensor);
    ncclComm_t comm = get_communicator(device_id);

    // NCCL doesn't have native gather, so we implement it using send/recv
    // within a group to enable communication/computation overlap

    NCCL_CHECK(ncclGroupStart());

    if (rank_ == dst_rank) {
        // Destination rank: receive from all other ranks
        if (output.size() != static_cast<size_t>(world_size_)) {
            throw std::invalid_argument("gather: output size must equal world_size");
        }

        for (int src = 0; src < world_size_; ++src) {
            if (src == rank_) {
                // Copy own data
                if (output[src].device() != tensor.device() ||
                    output[src].numel() != tensor.numel() ||
                    output[src].dtype() != tensor.dtype()) {
                    throw std::invalid_argument("gather: output tensor mismatch");
                }
                NCCL_CHECK(ncclRecv(
                    output[src].data_ptr(),
                    tensor.numel(),
                    to_nccl_datatype(tensor.dtype()),
                    rank_,  // Receive from self
                    comm,
                    nullptr  // Use default stream
                ));
            } else {
                // Receive from other ranks
                validate_gpu_tensor(output[src]);
                NCCL_CHECK(ncclRecv(
                    output[src].data_ptr(),
                    tensor.numel(),
                    to_nccl_datatype(tensor.dtype()),
                    src,
                    comm,
                    nullptr
                ));
            }
        }
    } else {
        // Non-destination ranks: send to destination
        NCCL_CHECK(ncclSend(
            tensor.data_ptr(),
            tensor.numel(),
            to_nccl_datatype(tensor.dtype()),
            dst_rank,
            comm,
            nullptr
        ));
    }

    NCCL_CHECK(ncclGroupEnd());

    // Synchronize to ensure gather completes
    GPU_CHECK(cudaDeviceSynchronize());
#else
    (void)tensor;
    (void)output;
    (void)dst_rank;
    throw NotImplementedError("NCCLBackend: NCCL not available");
#endif
}

auto NCCLBackend::scatter(const std::vector<Tensor>& tensors, Tensor& output, int src_rank) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(output);

    if (src_rank < 0 || src_rank >= world_size_) {
        throw std::invalid_argument("scatter: invalid src_rank");
    }

    int device_id = get_device_id(output);
    ncclComm_t comm = get_communicator(device_id);

    // NCCL doesn't have native scatter, so we implement it using send/recv
    // within a group to enable communication/computation overlap

    NCCL_CHECK(ncclGroupStart());

    if (rank_ == src_rank) {
        // Source rank: send to all other ranks
        if (tensors.size() != static_cast<size_t>(world_size_)) {
            throw std::invalid_argument("scatter: tensors size must equal world_size");
        }

        for (int dst = 0; dst < world_size_; ++dst) {
            validate_gpu_tensor(tensors[dst]);

            if (dst == rank_) {
                // Send to self (copy)
                if (tensors[dst].device() != output.device() ||
                    tensors[dst].numel() != output.numel() ||
                    tensors[dst].dtype() != output.dtype()) {
                    throw std::invalid_argument("scatter: tensor mismatch");
                }
                NCCL_CHECK(ncclSend(
                    tensors[dst].data_ptr(),
                    tensors[dst].numel(),
                    to_nccl_datatype(tensors[dst].dtype()),
                    rank_,  // Send to self
                    comm,
                    nullptr
                ));
            } else {
                // Send to other ranks
                NCCL_CHECK(ncclSend(
                    tensors[dst].data_ptr(),
                    tensors[dst].numel(),
                    to_nccl_datatype(tensors[dst].dtype()),
                    dst,
                    comm,
                    nullptr
                ));
            }
        }
    }

    // All ranks receive their portion
    NCCL_CHECK(ncclRecv(
        output.data_ptr(),
        output.numel(),
        to_nccl_datatype(output.dtype()),
        src_rank,
        comm,
        nullptr
    ));

    NCCL_CHECK(ncclGroupEnd());

    // Synchronize to ensure scatter completes
    GPU_CHECK(cudaDeviceSynchronize());
#else
    (void)tensors;
    (void)output;
    (void)src_rank;
    throw NotImplementedError("NCCLBackend: NCCL not available");
#endif
}

auto NCCLBackend::reduce_scatter(const std::vector<Tensor>& tensors, Tensor& output, ReduceOp op) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    if (tensors.empty()) {
        throw std::invalid_argument("reduce_scatter: tensors cannot be empty");
    }

    validate_gpu_tensor(tensors[0]);
    int device_id = get_device_id(tensors[0]);
    ncclComm_t comm = get_communicator(device_id);

    ncclDataType_t nccl_dtype = to_nccl_datatype(tensors[0].dtype());
    ncclRedOp_t nccl_op = to_nccl_reduce_op(op);

    // Concatenate input tensors
    std::vector<Tensor> concat_list(tensors.begin(), tensors.end());
    Tensor concatenated = cat(concat_list, 0);

    NCCL_CHECK(ncclReduceScatter(
        concatenated.data_ptr(),
        output.data_ptr(),
        output.numel(),
        nccl_dtype,
        nccl_op,
        comm,
        nullptr
    ));

    GPU_CHECK(cudaDeviceSynchronize());
#else
    throw NotImplementedError("NCCLBackend: NCCL not available");
#endif
}

// Phase E (E1): async on caller stream. Mirrors all_reduce_async pattern --
// launch on cuda_stream, do NOT cudaDeviceSynchronize. Caller is responsible
// for synchronization (cudaStreamSynchronize / cudaStreamWaitEvent on a
// downstream consumer stream).
auto NCCLBackend::reduce_scatter_async(const std::vector<Tensor>& tensors, Tensor& output,
                                       ReduceOp op, void* stream) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    if (tensors.empty()) {
        throw std::invalid_argument("reduce_scatter_async: tensors cannot be empty");
    }
    validate_gpu_tensor(tensors[0]);
    int device_id = get_device_id(tensors[0]);
    ncclComm_t comm = get_communicator(device_id);

    ncclDataType_t nccl_dtype = to_nccl_datatype(tensors[0].dtype());
    ncclRedOp_t nccl_op = to_nccl_reduce_op(op);

    std::vector<Tensor> concat_list(tensors.begin(), tensors.end());
    Tensor concatenated = cat(concat_list, 0);

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    NCCL_CHECK(ncclReduceScatter(
        concatenated.data_ptr(),
        output.data_ptr(),
        output.numel(),
        nccl_dtype,
        nccl_op,
        comm,
        cuda_stream
    ));
    // No cudaDeviceSynchronize -- caller owns the wait.
#else
    (void)stream;
    (void)tensors;
    (void)output;
    (void)op;
    throw NotImplementedError("NCCLBackend: NCCL not available");
#endif
}

// Phase E (E1): NCCL all_gather_async override. Default base impl falls back
// to sync all_gather. Real async launches on caller stream so gather can
// overlap with default-stream compute.
auto NCCLBackend::all_gather_async(const Tensor& tensor, std::vector<Tensor>& output,
                                    void* stream) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(tensor);

    if (output.size() != static_cast<size_t>(world_size_)) {
        throw std::invalid_argument(
            "all_gather_async: output size must equal world_size");
    }

    int device_id = get_device_id(tensor);
    ncclComm_t comm = get_communicator(device_id);
    ncclDataType_t nccl_dtype = to_nccl_datatype(tensor.dtype());

    // NCCL all-gather expects a contiguous output of world_size * input.numel().
    // Allocate that, launch async, then expose per-rank views via output[].
    // The slice operations themselves are metadata-only; safe before stream
    // completion as long as the consumer waits on the event before reading.
    int64_t per_rank = tensor.numel();
    std::vector<int64_t> out_shape{static_cast<int64_t>(world_size_) * per_rank};
    Tensor full = empty(out_shape, tensor.dtype(), tensor.device());

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    NCCL_CHECK(ncclAllGather(
        tensor.data_ptr(),
        full.data_ptr(),
        per_rank,
        nccl_dtype,
        comm,
        cuda_stream
    ));

    for (int r = 0; r < world_size_; ++r) {
        output[r] = full.slice(0, r * per_rank, (r + 1) * per_rank);
    }
#else
    (void)stream;
    // Sync fallback
    all_gather(tensor, output);
#endif
}

auto NCCLBackend::send(const Tensor& tensor, int dst_rank) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(tensor);

    if (dst_rank < 0 || dst_rank >= world_size_) {
        throw std::invalid_argument(
            "NCCLBackend::send: invalid dst_rank " + std::to_string(dst_rank)
        );
    }
    if (dst_rank == rank_) {
        throw std::invalid_argument(
            "NCCLBackend::send: cannot send to self (rank " + std::to_string(rank_) + ")"
        );
    }

    int device_id = get_device_id(tensor);
    ncclComm_t comm = get_communicator(device_id);

    ncclDataType_t nccl_dtype = to_nccl_datatype(tensor.dtype());

    NCCL_CHECK(ncclSend(
        tensor.data_ptr(),
        tensor.numel(),
        nccl_dtype,
        dst_rank,
        comm,
        nullptr  // Use default stream
    ));

    GPU_CHECK(cudaDeviceSynchronize());
#else
    (void)tensor;
    (void)dst_rank;
    throw NotImplementedError("NCCLBackend: NCCL not available");
#endif
}

auto NCCLBackend::recv(Tensor& tensor, int src_rank) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(tensor);

    if (src_rank < 0 || src_rank >= world_size_) {
        throw std::invalid_argument(
            "NCCLBackend::recv: invalid src_rank " + std::to_string(src_rank)
        );
    }
    if (src_rank == rank_) {
        throw std::invalid_argument(
            "NCCLBackend::recv: cannot recv from self (rank " + std::to_string(rank_) + ")"
        );
    }

    int device_id = get_device_id(tensor);
    ncclComm_t comm = get_communicator(device_id);

    ncclDataType_t nccl_dtype = to_nccl_datatype(tensor.dtype());

    NCCL_CHECK(ncclRecv(
        tensor.data_ptr(),
        tensor.numel(),
        nccl_dtype,
        src_rank,
        comm,
        nullptr  // Use default stream
    ));

    GPU_CHECK(cudaDeviceSynchronize());
#else
    (void)tensor;
    (void)src_rank;
    throw NotImplementedError("NCCLBackend: NCCL not available");
#endif
}

auto NCCLBackend::all_to_all_single(Tensor& output, const Tensor& input) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    // A4-extended NCCL override: native ncclGroupStart + ncclSend/Recv pairs
    // (mirrors NCCLProcessGroup::all_to_all_single in process_group.cpp,
    // which is the ProcessGroupBase-hierarchy version). The input has shape
    // [world_size * chunk, ...]; each rank slices its outgoing chunk for
    // every peer and submits one Send + one Recv per peer inside a single
    // NCCL group call so the runtime can fuse them.
    validate_gpu_tensor(input);
    validate_gpu_tensor(output);

    auto input_shape = input.shape();
    if (input_shape.empty()) {
        throw std::invalid_argument(
            "NCCLBackend::all_to_all_single: input must have >= 1 dimension.");
    }
    const int64_t total = input_shape[0];
    if (total % world_size_ != 0) {
        throw std::invalid_argument(
            "NCCLBackend::all_to_all_single: input.shape[0] (" +
            std::to_string(total) + ") must be divisible by world_size (" +
            std::to_string(world_size_) + ").");
    }
    std::vector<int64_t> in_shape_vec(input_shape.begin(), input_shape.end());
    std::vector<int64_t> out_shape_vec(output.shape().begin(), output.shape().end());
    if (in_shape_vec != out_shape_vec || output.dtype() != input.dtype()) {
        throw std::invalid_argument(
            "NCCLBackend::all_to_all_single: output and input must have identical shape and dtype.");
    }

    const int64_t chunk = total / world_size_;
    int64_t inner = 1;
    for (size_t i = 1; i < input_shape.size(); ++i) inner *= input_shape[i];
    const int64_t chunk_numel = chunk * inner;
    const size_t elem_size = dtype_size(input.dtype());

    int device_id = get_device_id(input);
    ncclComm_t comm = get_communicator(device_id);
    ncclDataType_t nccl_dtype = to_nccl_datatype(input.dtype());

    // Group all Send/Recv pairs so NCCL can pipeline them.
    NCCL_CHECK(ncclGroupStart());
    const char* in_ptr  = static_cast<const char*>(input.data_ptr());
    char*       out_ptr = static_cast<char*>(output.data_ptr());
    for (int peer = 0; peer < world_size_; ++peer) {
        const void* send_ptr = in_ptr  + static_cast<size_t>(peer) * static_cast<size_t>(chunk_numel) * elem_size;
        void*       recv_ptr = out_ptr + static_cast<size_t>(peer) * static_cast<size_t>(chunk_numel) * elem_size;
        if (peer == rank_) {
            // Self-chunk: cudaMemcpy directly; NCCL forbids self Send/Recv.
            GPU_CHECK(cudaMemcpyAsync(recv_ptr, send_ptr,
                                       static_cast<size_t>(chunk_numel) * elem_size,
                                       cudaMemcpyDeviceToDevice));
        } else {
            NCCL_CHECK(ncclSend(send_ptr, chunk_numel, nccl_dtype, peer, comm, nullptr));
            NCCL_CHECK(ncclRecv(recv_ptr, chunk_numel, nccl_dtype, peer, comm, nullptr));
        }
    }
    NCCL_CHECK(ncclGroupEnd());

    GPU_CHECK(cudaDeviceSynchronize());
#else
    (void)output;
    (void)input;
    throw NotImplementedError("NCCLBackend: NCCL not available");
#endif
}

auto NCCLBackend::barrier() -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    // NCCL doesn't have native barrier, use all-reduce on dummy tensor
    int device_id = 0;
    GPU_CHECK(cudaGetDevice(&device_id));

    Tensor dummy = zeros({1}, DType::Float32, Device::cuda(device_id));
    all_reduce(dummy, ReduceOp::SUM);
#else
    throw NotImplementedError("NCCLBackend: NCCL not available");
#endif
}

auto NCCLBackend::finalize() -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    for (auto& [device_id, comm] : communicators_) {
        if (comm != nullptr) {
            ncclCommDestroy(comm);
        }
    }
    communicators_.clear();
    initialized_ = false;
#endif
}

auto NCCLBackend::supports_device(Device::Type device_type) const -> bool {
    return device_type == Device::Type::CUDA || device_type == Device::Type::ROCm;
}

auto NCCLBackend::get_communicator(int device_id) -> ncclComm_t {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    auto it = communicators_.find(device_id);
    if (it != communicators_.end()) {
        return it->second;
    }

    // Initialize new communicator
    init_communicator(device_id);
    return communicators_[device_id];
#else
    throw NotImplementedError("NCCLBackend: NCCL not available");
#endif
}

auto NCCLBackend::get_unique_id() -> ncclUniqueId {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));
    return id;
#else
    ncclUniqueId dummy;
    return dummy;
#endif
}

// ============================================================================
// Private Helper Methods
// ============================================================================


auto NCCLBackend::init_communicator(int device_id) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    GPU_CHECK(cudaSetDevice(device_id));

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, world_size_, unique_id_, rank_));

    communicators_[device_id] = comm;
#endif
}

auto NCCLBackend::validate_gpu_tensor(const Tensor& tensor) -> void {
    if (tensor.device().type != Device::Type::CUDA &&
        tensor.device().type != Device::Type::ROCm) {
        throw std::invalid_argument(
            "NCCLBackend: tensor must be on GPU device, got " +
            tensor.device().to_string()
        );
    }
}

auto NCCLBackend::get_device_id(const Tensor& tensor) -> int {
    return tensor.device().index;
}

auto NCCLBackend::exchange_unique_id() -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    if (rank_ == 0) {
        // Master rank: generate unique ID and broadcast
        unique_id_ = get_unique_id();

        // Create server socket
        int server_fd = create_socket_connection(true);

        // Send unique ID to all other ranks
        for (int i = 1; i < world_size_; ++i) {
            int client_fd;
            do {
                client_fd = accept(server_fd, nullptr, nullptr);
            } while (client_fd < 0 && errno == EINTR);
            if (client_fd < 0) {
                throw std::runtime_error("Failed to accept connection from rank " +
                    std::to_string(i) + ": " + strerror(errno));
            }

            ssize_t sent;
            do {
                sent = ::send(client_fd, &unique_id_, sizeof(unique_id_), 0);
            } while (sent < 0 && errno == EINTR);
            if (sent != static_cast<ssize_t>(sizeof(unique_id_))) {
                close_socket(client_fd);
                throw std::runtime_error("Failed to send unique ID to rank " +
                    std::to_string(i) + ": " + strerror(errno));
            }

            close_socket(client_fd);
        }

        close_socket(server_fd);

    } else {
        // Worker ranks: receive unique ID from master
        int socket_fd = create_socket_connection(false);

        ssize_t received;
        do {
            received = ::recv(socket_fd, &unique_id_, sizeof(unique_id_), MSG_WAITALL);
        } while (received < 0 && errno == EINTR);
        if (received != static_cast<ssize_t>(sizeof(unique_id_))) {
            close_socket(socket_fd);
            throw std::runtime_error("Failed to receive unique ID from master: " +
                std::string(strerror(errno)));
        }

        close_socket(socket_fd);
    }
#endif
}

auto NCCLBackend::create_socket_connection(bool is_master) -> int {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        throw std::runtime_error("Failed to create socket");
    }

    // Apply socket timeouts (default 300s, configurable via TENZOR_COMM_TIMEOUT)
    const char* timeout_env = std::getenv("TENZOR_COMM_TIMEOUT");
    int timeout_sec = timeout_env ? std::atoi(timeout_env) : 300;
    if (timeout_sec > 0) {
        struct timeval tv{};
        tv.tv_sec = timeout_sec;
        tv.tv_usec = 0;
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    }

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(master_port_);

    if (is_master) {
        // Server mode — default to INADDR_ANY
        addr.sin_addr.s_addr = INADDR_ANY;

        // Respect NCCL_SOCKET_IFNAME to bind to a specific network interface
        const char* ifname = std::getenv("NCCL_SOCKET_IFNAME");
        if (ifname) {
            struct ifaddrs* ifaddr = nullptr;
            if (getifaddrs(&ifaddr) == 0) {
                bool found = false;
                for (auto* ifa = ifaddr; ifa; ifa = ifa->ifa_next) {
                    if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET &&
                        std::strcmp(ifa->ifa_name, ifname) == 0) {
                        addr.sin_addr = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr)->sin_addr;
                        found = true;
                        break;
                    }
                }
                freeifaddrs(ifaddr);
                if (!found) {
                    // Audit I.4: unified logger.
                    TENZOR_LOG_WARN("NCCL_SOCKET_IFNAME={} not found, falling back "
                                    "to INADDR_ANY", ifname);
                }
            }
        }

        int opt = 1;
        setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        if (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(sockfd);
            throw std::runtime_error("Failed to bind socket");
        }

        if (listen(sockfd, world_size_) < 0) {
            close(sockfd);
            throw std::runtime_error("Failed to listen on socket");
        }

        return sockfd;

    } else {
        // Client mode
        struct hostent* server = gethostbyname(master_addr_.c_str());
        if (!server) {
            close(sockfd);
            throw std::runtime_error("Failed to resolve master address: " + master_addr_);
        }

        // Validate the resolver result before copying: a non-IPv4 record
        // (AF_INET6, h_length==16) or a hostile DNS response would otherwise
        // overflow the 4-byte sin_addr field.
        if (server->h_addrtype != AF_INET ||
            server->h_length != static_cast<int>(sizeof(struct in_addr)) ||
            server->h_addr == nullptr) {
            close(sockfd);
            throw std::runtime_error(
                "Resolver returned a non-IPv4 address for master: " + master_addr_);
        }

        std::memcpy(&addr.sin_addr.s_addr, server->h_addr, sizeof(struct in_addr));

        // Retry connection with exponential backoff
        int retries = 10;
        for (int i = 0; i < retries; ++i) {
            if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                return sockfd;
            }
            usleep(100000 * (1 << i));  // 100ms, 200ms, 400ms, ...
        }

        close(sockfd);
        throw std::runtime_error("Failed to connect to master after " + std::to_string(retries) + " retries");
    }
}

auto NCCLBackend::close_socket(int socket_fd) -> void {
    if (socket_fd >= 0) {
        ::close(socket_fd);
    }
}

} // namespace distributed
} // namespace tenzor

#else  // !TENZOR_HAS_NCCL

// Stub implementation for builds without NCCL. The class still links so it can
// be instantiated and feature-detected; every collective throws
// NotImplementedError. This mirrors the MPIBackend stub branch and is what lets
// test_nccl_backend_smoke construct the backend and skip cleanly when NCCL is
// unavailable (its try_init catches the initialize() throw).
namespace tenzor {
namespace distributed {

NCCLBackend::NCCLBackend() = default;
NCCLBackend::~NCCLBackend() = default;

auto NCCLBackend::initialize(int, int, const std::string&, int) -> void {
    throw NotImplementedError("NCCLBackend: NCCL not available");
}
auto NCCLBackend::broadcast(Tensor&, int) -> void {
    throw NotImplementedError("NCCLBackend: NCCL not available");
}
auto NCCLBackend::all_reduce(Tensor&, ReduceOp) -> void {
    throw NotImplementedError("NCCLBackend: NCCL not available");
}
auto NCCLBackend::all_reduce_async(Tensor&, ReduceOp, void*) -> void {
    throw NotImplementedError("NCCLBackend: NCCL not available");
}
auto NCCLBackend::reduce(Tensor&, int, ReduceOp) -> void {
    throw NotImplementedError("NCCLBackend: NCCL not available");
}
auto NCCLBackend::all_gather(const Tensor&, std::vector<Tensor>&) -> void {
    throw NotImplementedError("NCCLBackend: NCCL not available");
}
auto NCCLBackend::all_gather_async(const Tensor&, std::vector<Tensor>&, void*) -> void {
    throw NotImplementedError("NCCLBackend: NCCL not available");
}
auto NCCLBackend::gather(const Tensor&, std::vector<Tensor>&, int) -> void {
    throw NotImplementedError("NCCLBackend: NCCL not available");
}
auto NCCLBackend::scatter(const std::vector<Tensor>&, Tensor&, int) -> void {
    throw NotImplementedError("NCCLBackend: NCCL not available");
}
auto NCCLBackend::reduce_scatter(const std::vector<Tensor>&, Tensor&, ReduceOp) -> void {
    throw NotImplementedError("NCCLBackend: NCCL not available");
}
auto NCCLBackend::reduce_scatter_async(const std::vector<Tensor>&, Tensor&, ReduceOp, void*) -> void {
    throw NotImplementedError("NCCLBackend: NCCL not available");
}
auto NCCLBackend::send(const Tensor&, int) -> void {
    throw NotImplementedError("NCCLBackend: NCCL not available");
}
auto NCCLBackend::recv(Tensor&, int) -> void {
    throw NotImplementedError("NCCLBackend: NCCL not available");
}
auto NCCLBackend::all_to_all_single(Tensor&, const Tensor&) -> void {
    throw NotImplementedError("NCCLBackend: NCCL not available");
}
auto NCCLBackend::barrier() -> void {
    throw NotImplementedError("NCCLBackend: NCCL not available");
}
auto NCCLBackend::finalize() -> void {}
auto NCCLBackend::supports_device(Device::Type) const -> bool {
    return false;
}

} // namespace distributed
} // namespace tenzor

#endif // TENZOR_HAS_NCCL
