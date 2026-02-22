/**
 * @file nccl_backend.cpp
 * @brief Implementation of NCCL backend for GPU communication
 */

#include "tenzor/distributed/nccl_backend.hpp"

#if defined(TENZOR_HAS_NCCL)

#include "tenzor/utils/error.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include <stdexcept>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>

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
    try {
        finalize();
    } catch (...) {
        // Ignore errors during destruction
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
    throw std::runtime_error("NCCLBackend: NCCL not available");
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
    throw std::runtime_error("NCCLBackend: NCCL not available");
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
    throw std::runtime_error("NCCLBackend: NCCL not available");
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
    throw std::runtime_error("NCCLBackend: NCCL not available");
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
    throw std::runtime_error("NCCLBackend: NCCL not available");
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
    throw std::runtime_error("NCCLBackend: NCCL not available");
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
    throw std::runtime_error("NCCLBackend: NCCL not available");
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
    throw std::runtime_error("NCCLBackend: NCCL not available");
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
    throw std::runtime_error("NCCLBackend: NCCL not available");
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
    throw std::runtime_error("NCCLBackend: NCCL not available");
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

auto NCCLBackend::to_nccl_reduce_op(ReduceOp op) -> ncclRedOp_t {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    switch (op) {
        case ReduceOp::SUM:
        case ReduceOp::AVG:  // Handle average as sum, then divide
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
#else
    return ncclSum;
#endif
}

auto NCCLBackend::to_nccl_datatype(DType dtype) -> ncclDataType_t {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    switch (dtype) {
        case DType::Float32:
            return ncclFloat;
        case DType::Float64:
            return ncclDouble;
        case DType::Int32:
            return ncclInt;
        case DType::Int64:
            return ncclInt64;
        default:
            throw std::invalid_argument(
                "Unsupported dtype for NCCL: " + std::to_string(static_cast<int>(dtype))
            );
    }
#else
    return ncclFloat;
#endif
}

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
            int client_fd = accept(server_fd, nullptr, nullptr);
            if (client_fd < 0) {
                throw std::runtime_error("Failed to accept connection from rank " + std::to_string(i));
            }

            ssize_t sent = send(client_fd, &unique_id_, sizeof(unique_id_), 0);
            if (sent != sizeof(unique_id_)) {
                throw std::runtime_error("Failed to send unique ID to rank " + std::to_string(i));
            }

            close_socket(client_fd);
        }

        close_socket(server_fd);

    } else {
        // Worker ranks: receive unique ID from master
        int socket_fd = create_socket_connection(false);

        ssize_t received = recv(socket_fd, &unique_id_, sizeof(unique_id_), MSG_WAITALL);
        if (received != sizeof(unique_id_)) {
            throw std::runtime_error("Failed to receive unique ID from master");
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

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(master_port_);

    if (is_master) {
        // Server mode
        addr.sin_addr.s_addr = INADDR_ANY;

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

        std::memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);

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

#endif // TENZOR_HAS_NCCL
