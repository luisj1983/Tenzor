/**
 * @file process_group.cpp
 * @brief Implementation of simplified ProcessGroupBase wrappers
 *
 * Implements GlooProcessGroup (and NCCLProcessGroup when NCCL is available)
 * by delegating to the existing ProcessGroup + CommunicationBackend infrastructure
 * (Gloo) or by directly owning and managing NCCL communicators (NCCL).
 */

#include "tenzor/distributed/process_group.hpp"
#include "tenzor/distributed/distributed.hpp"
#include "tenzor/distributed/gloo_backend.hpp"
#include <stdexcept>

#if defined(TENZOR_HAS_NCCL)
    #ifdef TENZOR_USE_ROCM
        #include <rccl/rccl.h>
    #else
        #include <nccl.h>
    #endif
    #include "tenzor/backend/loader.hpp"  // for is_backend_registry_alive()
    #include "tenzor/ops/creation.hpp"    // for zeros(), empty()
    #include "tenzor/ops/transform.hpp"   // for cat()
    #include <cstring>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <netdb.h>

    #if defined(TENZOR_USE_CUDA)
        #include <cuda_runtime.h>
        #define NCCL_PG_GPU_CHECK(call) \
            do { \
                cudaError_t err = call; \
                if (err != cudaSuccess) { \
                    throw std::runtime_error( \
                        std::string("CUDA error in NCCLProcessGroup: ") + \
                        cudaGetErrorString(err) \
                    ); \
                } \
            } while(0)
    #elif defined(TENZOR_USE_ROCM)
        #include <hip/hip_runtime.h>
        #define NCCL_PG_GPU_CHECK(call) \
            do { \
                hipError_t err = call; \
                if (err != hipSuccess) { \
                    throw std::runtime_error( \
                        std::string("HIP error in NCCLProcessGroup: ") + \
                        hipGetErrorString(err) \
                    ); \
                } \
            } while(0)
    #else
        #define NCCL_PG_GPU_CHECK(call) call
    #endif

    #define NCCL_PG_CHECK(cmd) \
        do { \
            ncclResult_t result = cmd; \
            if (result != ncclSuccess) { \
                throw std::runtime_error( \
                    std::string("NCCL error in NCCLProcessGroup: ") + \
                    ncclGetErrorString(result) + \
                    " at " + __FILE__ + ":" + std::to_string(__LINE__) \
                ); \
            } \
        } while(0)

namespace {

/// Convert ReduceOp to ncclRedOp_t
auto to_nccl_reduce_op(tenzor::distributed::ReduceOp op) -> ncclRedOp_t {
    using tenzor::distributed::ReduceOp;
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
            throw std::invalid_argument(
                "NCCLProcessGroup: unsupported reduce operation"
            );
    }
}

/// Convert DType to ncclDataType_t
auto to_nccl_datatype(tenzor::DType dtype) -> ncclDataType_t {
    switch (dtype) {
        case tenzor::DType::Float32:
            return ncclFloat;
        case tenzor::DType::Float64:
            return ncclDouble;
        case tenzor::DType::Int32:
            return ncclInt;
        case tenzor::DType::Int64:
            return ncclInt64;
        default:
            throw std::invalid_argument(
                "NCCLProcessGroup: unsupported dtype for NCCL: " +
                std::to_string(static_cast<int>(dtype))
            );
    }
}

} // anonymous namespace
#endif // TENZOR_HAS_NCCL

namespace tenzor::distributed {

// ============================================================================
// GlooProcessGroup Implementation
// ============================================================================

GlooProcessGroup::GlooProcessGroup(int rank, int world_size,
                                   const std::string& master_addr,
                                   int master_port)
    : rank_(rank), world_size_(world_size) {

    if (rank < 0 || rank >= world_size) {
        throw std::invalid_argument(
            "GlooProcessGroup: rank " + std::to_string(rank) +
            " must be in range [0, " + std::to_string(world_size) + ")"
        );
    }

    if (world_size <= 0) {
        throw std::invalid_argument(
            "GlooProcessGroup: world_size must be positive, got " +
            std::to_string(world_size)
        );
    }

    // Create the underlying ProcessGroup using Gloo backend
    pg_ = ProcessGroup::create_process_group(
        Backend::GLOO, rank, world_size, master_addr, master_port
    );
}

GlooProcessGroup::~GlooProcessGroup() = default;

auto GlooProcessGroup::all_reduce(Tensor& tensor, ReduceOp op) -> void {
    pg_->all_reduce(tensor, op);
}

auto GlooProcessGroup::broadcast(Tensor& tensor, int src_rank) -> void {
    pg_->broadcast(tensor, src_rank);
}

auto GlooProcessGroup::all_gather(std::vector<Tensor>& output, const Tensor& input) -> void {
    pg_->all_gather(input, output);
}

auto GlooProcessGroup::reduce_scatter(Tensor& output, std::span<const Tensor> input) -> void {
    std::vector<Tensor> input_vec(input.begin(), input.end());
    pg_->reduce_scatter(input_vec, output);
}

auto GlooProcessGroup::barrier() -> void {
    pg_->barrier();
}

// ============================================================================
// NCCLProcessGroup Implementation
// ============================================================================

#ifdef TENZOR_HAS_NCCL

NCCLProcessGroup::NCCLProcessGroup(int rank, int world_size,
                                   const std::string& master_addr,
                                   int master_port)
    : rank_(rank), world_size_(world_size) {

    if (rank < 0 || rank >= world_size) {
        throw std::invalid_argument(
            "NCCLProcessGroup: rank " + std::to_string(rank) +
            " must be in range [0, " + std::to_string(world_size) + ")"
        );
    }

    if (world_size <= 0) {
        throw std::invalid_argument(
            "NCCLProcessGroup: world_size must be positive, got " +
            std::to_string(world_size)
        );
    }

#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    // Step 1: Exchange ncclUniqueId via TCP bootstrap.
    // bootstrap_unique_id() allocates a heap ncclUniqueId and stashes it
    // in comm_ for us to retrieve here.
    bootstrap_unique_id(master_addr, master_port);

    // Step 2: Create the NCCL communicator on the current GPU
    int device_id = 0;
    NCCL_PG_GPU_CHECK(cudaGetDevice(&device_id));
    NCCL_PG_GPU_CHECK(cudaSetDevice(device_id));

    // Extract the bootstrapped unique ID, then replace comm_ with the
    // real NCCL communicator
    ncclUniqueId* id_ptr = static_cast<ncclUniqueId*>(comm_);
    ncclUniqueId id;
    std::memcpy(&id, id_ptr, sizeof(ncclUniqueId));
    delete id_ptr;
    comm_ = nullptr;

    ncclComm_t nccl_comm = nullptr;
    NCCL_PG_CHECK(ncclCommInitRank(&nccl_comm, world_size, id, rank));
    comm_ = static_cast<void*>(nccl_comm);
#else
    (void)master_addr;
    (void)master_port;
    throw std::runtime_error(
        "NCCLProcessGroup: NCCL requires CUDA or ROCm support. "
        "Please rebuild with TENZOR_BUILD_CUDA=ON or TENZOR_BUILD_ROCM=ON"
    );
#endif
}

NCCLProcessGroup::~NCCLProcessGroup() {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    if (comm_ != nullptr) {
        // Guard against static destruction ordering issues:
        // if the backend registry is already torn down, CUDA/NCCL
        // resources may no longer be valid.
        if (is_backend_registry_alive()) {
            ncclComm_t nccl_comm = static_cast<ncclComm_t>(comm_);
            ncclCommDestroy(nccl_comm);
        }
        comm_ = nullptr;
    }
#endif
}

auto NCCLProcessGroup::bootstrap_unique_id(
    const std::string& master_addr,
    int master_port
) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    // Allocate a heap ncclUniqueId that we'll temporarily stash in comm_
    auto* id_ptr = new ncclUniqueId{};

    if (rank_ == 0) {
        // Rank 0: generate the unique ID
        NCCL_PG_CHECK(ncclGetUniqueId(id_ptr));

        // Create server socket to send the ID to all other ranks
        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) {
            delete id_ptr;
            throw std::runtime_error(
                "NCCLProcessGroup: failed to create bootstrap socket"
            );
        }

        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(master_port));

        if (bind(server_fd, reinterpret_cast<struct sockaddr*>(&addr),
                 sizeof(addr)) < 0) {
            ::close(server_fd);
            delete id_ptr;
            throw std::runtime_error(
                "NCCLProcessGroup: failed to bind bootstrap socket on port " +
                std::to_string(master_port)
            );
        }

        if (listen(server_fd, world_size_) < 0) {
            ::close(server_fd);
            delete id_ptr;
            throw std::runtime_error(
                "NCCLProcessGroup: failed to listen on bootstrap socket"
            );
        }

        // Accept connections from all other ranks and send the unique ID
        for (int i = 1; i < world_size_; ++i) {
            int client_fd = accept(server_fd, nullptr, nullptr);
            if (client_fd < 0) {
                ::close(server_fd);
                delete id_ptr;
                throw std::runtime_error(
                    "NCCLProcessGroup: failed to accept connection from rank " +
                    std::to_string(i)
                );
            }

            ssize_t sent = send(client_fd, id_ptr, sizeof(ncclUniqueId), 0);
            ::close(client_fd);

            if (sent != static_cast<ssize_t>(sizeof(ncclUniqueId))) {
                ::close(server_fd);
                delete id_ptr;
                throw std::runtime_error(
                    "NCCLProcessGroup: failed to send unique ID to rank " +
                    std::to_string(i)
                );
            }
        }

        ::close(server_fd);

    } else {
        // Non-zero ranks: connect to rank 0 and receive the unique ID
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            delete id_ptr;
            throw std::runtime_error(
                "NCCLProcessGroup: failed to create client socket"
            );
        }

        struct hostent* server = gethostbyname(master_addr.c_str());
        if (!server) {
            ::close(sockfd);
            delete id_ptr;
            throw std::runtime_error(
                "NCCLProcessGroup: failed to resolve master address: " +
                master_addr
            );
        }

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        std::memcpy(&addr.sin_addr.s_addr, server->h_addr,
                     static_cast<size_t>(server->h_length));
        addr.sin_port = htons(static_cast<uint16_t>(master_port));

        // Retry connection with exponential backoff
        constexpr int max_retries = 10;
        bool connected = false;
        for (int attempt = 0; attempt < max_retries; ++attempt) {
            if (connect(sockfd, reinterpret_cast<struct sockaddr*>(&addr),
                        sizeof(addr)) == 0) {
                connected = true;
                break;
            }
            // Exponential backoff: 100ms, 200ms, 400ms, ...
            usleep(static_cast<useconds_t>(100000) << attempt);
        }

        if (!connected) {
            ::close(sockfd);
            delete id_ptr;
            throw std::runtime_error(
                "NCCLProcessGroup: failed to connect to master at " +
                master_addr + ":" + std::to_string(master_port) +
                " after " + std::to_string(max_retries) + " retries"
            );
        }

        ssize_t received = recv(sockfd, id_ptr, sizeof(ncclUniqueId),
                                MSG_WAITALL);
        ::close(sockfd);

        if (received != static_cast<ssize_t>(sizeof(ncclUniqueId))) {
            delete id_ptr;
            throw std::runtime_error(
                "NCCLProcessGroup: failed to receive unique ID from master"
            );
        }
    }

    // Stash the unique ID pointer in comm_ for the constructor to retrieve
    comm_ = static_cast<void*>(id_ptr);
#else
    (void)master_addr;
    (void)master_port;
#endif
}

auto NCCLProcessGroup::validate_gpu_tensor(const Tensor& tensor) const -> void {
    if (tensor.device().type != Device::Type::CUDA &&
        tensor.device().type != Device::Type::ROCm) {
        throw std::invalid_argument(
            "NCCLProcessGroup: tensor must be on GPU device, got " +
            tensor.device().to_string()
        );
    }
}

auto NCCLProcessGroup::get_device_id(const Tensor& tensor) const -> int {
    return tensor.device().index;
}

auto NCCLProcessGroup::all_reduce(Tensor& tensor, ReduceOp op) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(tensor);

    ncclComm_t nccl_comm = static_cast<ncclComm_t>(comm_);
    ncclDataType_t nccl_dtype = to_nccl_datatype(tensor.dtype());
    ncclRedOp_t nccl_op = to_nccl_reduce_op(op);

    NCCL_PG_CHECK(ncclAllReduce(
        tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        nccl_dtype,
        nccl_op,
        nccl_comm,
        nullptr  // default CUDA stream
    ));

    NCCL_PG_GPU_CHECK(cudaDeviceSynchronize());

    // AVG = SUM followed by division
    if (op == ReduceOp::AVG) {
        tensor = tensor / static_cast<float>(world_size_);
    }
#else
    (void)tensor;
    (void)op;
    throw std::runtime_error("NCCLProcessGroup: NCCL not available");
#endif
}

auto NCCLProcessGroup::all_reduce_async(Tensor& tensor, ReduceOp op,
                                         void* stream) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(tensor);

    ncclComm_t nccl_comm = static_cast<ncclComm_t>(comm_);
    ncclDataType_t nccl_dtype = to_nccl_datatype(tensor.dtype());
    ncclRedOp_t nccl_op = to_nccl_reduce_op(op);

    // Launch NCCL all-reduce on the caller-provided stream.
    // Unlike all_reduce(), we do NOT call cudaDeviceSynchronize() --
    // the caller is responsible for synchronization via CUDA events.
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    NCCL_PG_CHECK(ncclAllReduce(
        tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        nccl_dtype,
        nccl_op,
        nccl_comm,
        cuda_stream
    ));

    // Note: AVG handling (divide by world_size) is done by the caller
    // (DDP) after sync, since we cannot safely do tensor math on an
    // async stream without additional synchronization.
#else
    (void)tensor;
    (void)op;
    (void)stream;
    throw std::runtime_error("NCCLProcessGroup: NCCL not available");
#endif
}

auto NCCLProcessGroup::broadcast(Tensor& tensor, int src_rank) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(tensor);

    ncclComm_t nccl_comm = static_cast<ncclComm_t>(comm_);
    ncclDataType_t nccl_dtype = to_nccl_datatype(tensor.dtype());

    NCCL_PG_CHECK(ncclBroadcast(
        tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        nccl_dtype,
        src_rank,
        nccl_comm,
        nullptr  // default CUDA stream
    ));

    NCCL_PG_GPU_CHECK(cudaDeviceSynchronize());
#else
    (void)tensor;
    (void)src_rank;
    throw std::runtime_error("NCCLProcessGroup: NCCL not available");
#endif
}

auto NCCLProcessGroup::all_gather(std::vector<Tensor>& output,
                                   const Tensor& input) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(input);

    if (static_cast<int>(output.size()) != world_size_) {
        throw std::invalid_argument(
            "NCCLProcessGroup::all_gather: output size (" +
            std::to_string(output.size()) + ") must equal world_size (" +
            std::to_string(world_size_) + ")"
        );
    }

    ncclComm_t nccl_comm = static_cast<ncclComm_t>(comm_);
    ncclDataType_t nccl_dtype = to_nccl_datatype(input.dtype());

    // NCCL all-gather writes into a contiguous buffer; allocate one
    size_t total_elements = input.numel() * world_size_;
    Tensor gathered = empty(
        {static_cast<int64_t>(total_elements)},
        input.dtype(),
        input.device()
    );

    NCCL_PG_CHECK(ncclAllGather(
        input.data_ptr(),
        gathered.data_ptr(),
        input.numel(),
        nccl_dtype,
        nccl_comm,
        nullptr  // default CUDA stream
    ));

    NCCL_PG_GPU_CHECK(cudaDeviceSynchronize());

    // Split the contiguous result into per-rank output tensors
    for (int i = 0; i < world_size_; ++i) {
        output[i] = gathered.slice(
            0,
            i * input.numel(),
            (i + 1) * input.numel()
        );
    }
#else
    (void)output;
    (void)input;
    throw std::runtime_error("NCCLProcessGroup: NCCL not available");
#endif
}

auto NCCLProcessGroup::reduce_scatter(Tensor& output,
                                       std::span<const Tensor> input) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    if (input.empty()) {
        throw std::invalid_argument(
            "NCCLProcessGroup::reduce_scatter: input cannot be empty"
        );
    }

    validate_gpu_tensor(input[0]);

    ncclComm_t nccl_comm = static_cast<ncclComm_t>(comm_);
    ncclDataType_t nccl_dtype = to_nccl_datatype(input[0].dtype());

    // Concatenate input tensors into contiguous buffer
    std::vector<Tensor> input_vec(input.begin(), input.end());
    Tensor concatenated = cat(input_vec, 0);

    NCCL_PG_CHECK(ncclReduceScatter(
        concatenated.data_ptr(),
        output.data_ptr(),
        output.numel(),
        nccl_dtype,
        ncclSum,
        nccl_comm,
        nullptr  // default CUDA stream
    ));

    NCCL_PG_GPU_CHECK(cudaDeviceSynchronize());
#else
    (void)output;
    (void)input;
    throw std::runtime_error("NCCLProcessGroup: NCCL not available");
#endif
}

auto NCCLProcessGroup::barrier() -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    // NCCL has no native barrier; perform an all-reduce on a dummy tensor
    int device_id = 0;
    NCCL_PG_GPU_CHECK(cudaGetDevice(&device_id));

    Tensor dummy = zeros({1}, DType::Float32, Device::cuda(device_id));
    all_reduce(dummy, ReduceOp::SUM);
#else
    throw std::runtime_error("NCCLProcessGroup: NCCL not available");
#endif
}

#endif // TENZOR_HAS_NCCL

} // namespace tenzor::distributed
