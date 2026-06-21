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

    // Validate the port before it reaches htons() in create_socket_connection;
    // an out-of-range value would otherwise be silently truncated to 16 bits,
    // routing the rendezvous socket to an unintended/colliding port (matching
    // the guards in GlooProcessGroup / NCCLProcessGroup::bootstrap_unique_id).
    if (master_port < 1 || master_port > 65535) {
        throw std::invalid_argument(
            "NCCLBackend: master_port " + std::to_string(master_port) +
            " out of range [1, 65535]"
        );
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

    // Stage a contiguous buffer (see all_reduce): a non-contiguous receive
    // buffer would otherwise be written as if contiguous and corrupted.
    const bool staged = !tensor.is_contiguous();
    Tensor work = staged ? tensor.contiguous() : tensor;

    NCCL_CHECK(ncclBroadcast(
        work.data_ptr(),
        work.data_ptr(),
        work.numel(),
        nccl_dtype,
        src_rank,
        comm,
        nullptr  // Use default stream
    ));

    // Synchronize to ensure completion
    GPU_CHECK(cudaDeviceSynchronize());

    if (staged) {
        // Write the result back THROUGH the caller's existing storage rather
        // than rebinding the reference (`tensor = work`), which would detach
        // any Variable/view/external buffer aliasing the original
        // non-contiguous storage. Tensor exposes no copy_, so use the
        // established in-place primitive pair (zero_() + operator+=).
        tensor.zero_();
        tensor += work;
    }
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

    // NCCL treats data_ptr() as a flat contiguous buffer; a non-contiguous
    // input would be read/written as if contiguous and corrupt the result.
    // Stage a contiguous copy and propagate the reduced values back.
    const bool staged = !tensor.is_contiguous();
    Tensor work = staged ? tensor.contiguous() : tensor;

    NCCL_CHECK(ncclAllReduce(
        work.data_ptr(),
        work.data_ptr(),
        work.numel(),
        nccl_dtype,
        nccl_op,
        comm,
        nullptr  // Use default stream
    ));

    // Synchronize to ensure completion
    GPU_CHECK(cudaDeviceSynchronize());

    // If AVG operation, divide by world size
    if (op == ReduceOp::AVG) {
        // In-place, dtype/device-matched divide so the result keeps the
        // tensor's dtype (a scalar-float rebind would not).
        auto scalar = tenzor::full({1}, static_cast<double>(world_size_),
                                   work.dtype(), work.device());
        work /= scalar;
    }

    if (staged) {
        // Write back THROUGH the caller's storage (see broadcast): rebinding
        // `tensor = work` would detach aliased Variables/views/external buffers
        // sharing the original non-contiguous storage.
        tensor.zero_();
        tensor += work;
    }
#else
    throw NotImplementedError("NCCLBackend: NCCL not available");
#endif
}

auto NCCLBackend::all_reduce_async(Tensor& tensor, ReduceOp op,
                                    void* stream) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(tensor);

    // ReduceOp::AVG is implemented as ncclSum followed by an in-place divide by
    // world_size. The async path cannot safely perform that divide on the
    // caller's stream without extra synchronization, so it would otherwise
    // return an un-averaged SUM under the AVG label — silently wrong, unlike the
    // synchronous all_reduce() which divides internally. Reject AVG explicitly
    // rather than producing a wrong result; callers needing an async average
    // must use ncclSum here and divide by world_size after their own sync.
    if (op == ReduceOp::AVG) {
        throw std::invalid_argument(
            "NCCLBackend::all_reduce_async: ReduceOp::AVG is not supported on the "
            "asynchronous path (the averaging divide cannot be applied on the "
            "caller's stream). Use ReduceOp::SUM and divide by world_size after "
            "synchronizing, or use the synchronous all_reduce().");
    }

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

    // NCCL has no native AVG op (ReduceOp::AVG maps to ncclSum); only the
    // destination rank holds the reduced result, so divide there to get the
    // mean. Non-root ranks keep their unchanged local tensor.
    if (op == ReduceOp::AVG && rank_ == dst_rank) {
        auto scalar = tenzor::full({1}, static_cast<double>(world_size_),
                                   tensor.dtype(), tensor.device());
        tensor /= scalar;
    }
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

    // Stage a contiguous source buffer (read-only input): NCCL reads numel
    // elements as a flat contiguous block.
    Tensor in = tensor.is_contiguous() ? tensor : tensor.contiguous();

    NCCL_CHECK(ncclAllGather(
        in.data_ptr(),
        gathered.data_ptr(),
        in.numel(),
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
                // Copy own data with a local device-to-device memcpy. A
                // self ncclRecv has NO matching self ncclSend (this rank never
                // sends to itself), so posting it inside the group deadlocks
                // the whole collective. Mirrors all_to_all_single's self path.
                GPU_CHECK(cudaMemcpyAsync(
                    output[src].data_ptr(),
                    tensor.data_ptr(),
                    static_cast<size_t>(tensor.numel()) * tensor.dtype_size(),
                    cudaMemcpyDeviceToDevice,
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

    // ncclReduceScatter reads recvcount*nranks elements from the send buffer.
    // Validate the concatenated input matches output.numel()*world_size_ (and
    // dtype/device) so a caller contract violation raises a clear error instead
    // of a silent device-side out-of-bounds read.
    if (concatenated.numel() != output.numel() * world_size_) {
        throw std::invalid_argument(
            "reduce_scatter: concatenated input numel (" +
            std::to_string(concatenated.numel()) +
            ") must equal output.numel() (" + std::to_string(output.numel()) +
            ") * world_size (" + std::to_string(world_size_) + ")"
        );
    }
    if (output.dtype() != concatenated.dtype()) {
        throw std::invalid_argument(
            "reduce_scatter: output dtype must match input dtype"
        );
    }
    if (output.device() != concatenated.device()) {
        throw std::invalid_argument(
            "reduce_scatter: output device must match input device"
        );
    }

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

    // NCCL has no native AVG reduce op (ReduceOp::AVG maps to ncclSum); divide by
    // world size afterwards, mirroring NCCLBackend::all_reduce.
    if (op == ReduceOp::AVG) {
        auto scalar = tenzor::full({1}, static_cast<double>(world_size_),
                                   output.dtype(), output.device());
        output /= scalar;
    }
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

    // Stage a contiguous source buffer; NCCL reads numel elements as a flat
    // contiguous block, so a non-contiguous view would send the wrong bytes.
    Tensor in = tensor.is_contiguous() ? tensor : tensor.contiguous();

    NCCL_CHECK(ncclSend(
        in.data_ptr(),
        in.numel(),
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
    // NCCL doesn't have native barrier, use all-reduce on dummy tensor.
    // Use the communicator's OWN device, not whatever device happens to be
    // current. A single backend instance binds exactly one device (enforced in
    // init_communicator); using the ambient cudaGetDevice() could pick a
    // different device, which would make get_communicator() throw the
    // "single backend instance supports only one device" error or try to build
    // a comm from the already-consumed unique_id. Fall back to the ambient
    // device only if no communicator has been created yet (initialize() always
    // builds one, so this is just a safety net).
    int device_id = 0;
    if (!communicators_.empty()) {
        device_id = communicators_.begin()->first;
    } else {
        GPU_CHECK(cudaGetDevice(&device_id));
    }

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
    // A single ncclUniqueId identifies exactly one communicator clique and is
    // consumed by the first ncclCommInitRank that uses it. unique_id_ was
    // exchanged once in exchange_unique_id(), so this backend instance can only
    // build one communicator. Creating a second communicator for a different
    // device from the already-consumed unique_id (with the same rank) is invalid
    // NCCL usage and would hang/error. Enforce one device per backend instance
    // rather than silently constructing a broken multi-device setup. Multi-GPU
    // per process must use one NCCLBackend instance (and one unique_id) per
    // device.
    if (!communicators_.empty() &&
        communicators_.find(device_id) == communicators_.end()) {
        throw std::runtime_error(
            "NCCLBackend: a single backend instance supports only one device "
            "(unique_id is exchanged once and consumed by the first "
            "communicator). Requested device " + std::to_string(device_id) +
            " but a communicator for a different device already exists. Use one "
            "NCCLBackend instance per device for multi-GPU-per-process."
        );
    }

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
