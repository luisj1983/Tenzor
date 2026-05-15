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
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include <stdexcept>
#include <memory>
#include <cstdlib>

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

auto GlooProcessGroup::all_to_all_single(Tensor& output, const Tensor& input) -> void {
    // Gloo lacks a native `all_to_all` in this codebase; fall back to the
    // base-class all_gather-based default. This is bandwidth-suboptimal
    // (uses world_size * input_size) but is correct on top of Gloo's
    // existing all_gather. The native primitive (when added to GlooBackend)
    // should override this method.
    ProcessGroupBase::all_to_all_single(output, input);
}

// ============================================================================
// ProcessGroupBase::split default: throw. Backends override.
// ============================================================================
auto ProcessGroupBase::split(int color, int key)
    -> std::shared_ptr<ProcessGroupBase>
{
    (void)color; (void)key;
    throw std::runtime_error(
        "ProcessGroupBase::split: not implemented for this backend. "
        "NCCLProcessGroup supports ncclCommSplit; other backends require a "
        "concrete override.");
}

// ============================================================================
// ProcessGroupBase non-pure default: all_to_all_single via all_gather + slice.
// ============================================================================
//
// Contract:
//   * `input` and `output` must have the same shape and dtype.
//   * `input.shape()[0]` must be divisible by `world_size()`.
//   * On return, `output` is rebound to a freshly-allocated tensor produced
//     by `cat()`; the caller's pre-allocated buffer is discarded. Backends
//     that override this method (e.g. NCCL) preserve the pre-allocated buffer.
//
// This default is bandwidth-suboptimal (uses world_size^2 chunk volume across
// the wire instead of world_size). Production GPU paths should override
// with a native group-Send/Recv-pair primitive — see NCCLProcessGroup.
auto ProcessGroupBase::all_to_all_single(Tensor& output, const Tensor& input) -> void {
    const int ws = world_size();
    if (input.shape().empty()) {
        throw std::invalid_argument(
            "ProcessGroupBase::all_to_all_single: input must have at least 1 dimension");
    }
    const int64_t total = input.shape()[0];
    if (total % ws != 0) {
        throw std::invalid_argument(
            "ProcessGroupBase::all_to_all_single: input.shape[0] (" +
            std::to_string(total) + ") must be divisible by world_size (" +
            std::to_string(ws) + ")"
        );
    }
    auto same_shape = [](std::span<const int64_t> a, std::span<const int64_t> b) {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
    };
    if (!same_shape(output.shape(), input.shape()) || output.dtype() != input.dtype()) {
        throw std::invalid_argument(
            "ProcessGroupBase::all_to_all_single: output must match input shape and dtype");
    }

    // 1. all_gather: every rank receives every peer's full input.
    std::vector<Tensor> gathered(ws);
    all_gather(gathered, input);

    // 2. From peer r's gathered tensor, take the slice destined for *my* rank.
    //    That slice goes into position r of the output (we receive peer-ordered
    //    contributions).
    const int my_rank = rank();
    const int64_t chunk = total / ws;
    std::vector<Tensor> per_peer(ws);
    for (int r = 0; r < ws; ++r) {
        per_peer[r] = gathered[r].slice(0, my_rank * chunk, (my_rank + 1) * chunk);
    }
    output = cat(per_peer, 0);
}

// ============================================================================
// NCCLProcessGroup Implementation
// ============================================================================

#ifdef TENZOR_HAS_NCCL

// Private constructor (audit A3-extended): wraps an already-created
// ncclComm_t produced by `ncclCommSplit`. Skips the TCP bootstrap — the
// caller (`NCCLProcessGroup::split`) has already done the synchronous
// collective that constructs the new communicator.
NCCLProcessGroup::NCCLProcessGroup(int rank, int world_size, void* comm)
    : rank_(rank), world_size_(world_size), comm_(comm) {
    if (rank < 0 || rank >= world_size) {
        throw std::invalid_argument(
            "NCCLProcessGroup(split): rank " + std::to_string(rank) +
            " must be in range [0, " + std::to_string(world_size) + ")");
    }
    if (comm == nullptr) {
        throw std::invalid_argument(
            "NCCLProcessGroup(split): comm must not be null");
    }
}

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

#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
namespace {

// RAII wrapper for a POSIX fd. close() on destruction; release() gives up
// ownership. Used so any throw path (including future ones) cleans up the
// socket without repeating `::close(fd); delete id_ptr;` at every call site.
class FdGuard {
public:
    FdGuard() = default;
    explicit FdGuard(int fd) : fd_(fd) {}
    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
    FdGuard(FdGuard&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    FdGuard& operator=(FdGuard&& other) noexcept {
        if (this != &other) {
            reset();
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }
    ~FdGuard() { reset(); }

    [[nodiscard]] int get() const noexcept { return fd_; }
    [[nodiscard]] bool valid() const noexcept { return fd_ >= 0; }
    void reset(int new_fd = -1) noexcept {
        if (fd_ >= 0) ::close(fd_);
        fd_ = new_fd;
    }

private:
    int fd_{-1};
};

// Read TENZOR_NCCL_BOOTSTRAP_TIMEOUT_SEC (default 30) for socket timeouts.
int bootstrap_timeout_seconds() {
    const char* env = std::getenv("TENZOR_NCCL_BOOTSTRAP_TIMEOUT_SEC");
    if (env == nullptr || env[0] == '\0') return 30;
    int seconds = std::atoi(env);
    if (seconds <= 0) return 30;
    return seconds;
}

// Apply SO_RCVTIMEO + SO_SNDTIMEO so a dead peer does not deadlock
// bootstrap. Called once per socket, right after creation.
void apply_socket_timeouts(int fd) {
    struct timeval tv;
    tv.tv_sec = bootstrap_timeout_seconds();
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

} // anonymous namespace
#endif

auto NCCLProcessGroup::bootstrap_unique_id(
    const std::string& master_addr,
    int master_port
) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    // Port validation: htons() of an out-of-range value silently truncates,
    // which used to route rank 0 to the wrong port. Fail fast instead.
    if (master_port < 1 || master_port > 65535) {
        throw std::invalid_argument(
            "NCCLProcessGroup: master_port must be in [1, 65535], got " +
            std::to_string(master_port));
    }

    // Heap-allocate the ncclUniqueId through a unique_ptr so every error
    // path (including future additions) cleans up automatically. We only
    // call .release() at the very end, on success.
    auto id = std::make_unique<ncclUniqueId>();

    if (rank_ == 0) {
        NCCL_PG_CHECK(ncclGetUniqueId(id.get()));

        FdGuard server(socket(AF_INET, SOCK_STREAM, 0));
        if (!server.valid()) {
            throw std::runtime_error(
                "NCCLProcessGroup: failed to create bootstrap socket");
        }
        apply_socket_timeouts(server.get());

        int opt = 1;
        setsockopt(server.get(), SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(master_port));

        if (bind(server.get(), reinterpret_cast<struct sockaddr*>(&addr),
                 sizeof(addr)) < 0) {
            throw std::runtime_error(
                "NCCLProcessGroup: failed to bind bootstrap socket on port " +
                std::to_string(master_port));
        }

        if (listen(server.get(), world_size_) < 0) {
            throw std::runtime_error(
                "NCCLProcessGroup: failed to listen on bootstrap socket");
        }

        for (int i = 1; i < world_size_; ++i) {
            FdGuard client(accept(server.get(), nullptr, nullptr));
            if (!client.valid()) {
                throw std::runtime_error(
                    "NCCLProcessGroup: failed to accept connection from rank " +
                    std::to_string(i));
            }
            apply_socket_timeouts(client.get());

            ssize_t sent = send(client.get(), id.get(), sizeof(ncclUniqueId), 0);
            if (sent != static_cast<ssize_t>(sizeof(ncclUniqueId))) {
                throw std::runtime_error(
                    "NCCLProcessGroup: failed to send unique ID to rank " +
                    std::to_string(i));
            }
        }
    } else {
        FdGuard sock(socket(AF_INET, SOCK_STREAM, 0));
        if (!sock.valid()) {
            throw std::runtime_error(
                "NCCLProcessGroup: failed to create client socket");
        }
        apply_socket_timeouts(sock.get());

        struct hostent* server = gethostbyname(master_addr.c_str());
        if (!server) {
            throw std::runtime_error(
                "NCCLProcessGroup: failed to resolve master address: " +
                master_addr);
        }

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        std::memcpy(&addr.sin_addr.s_addr, server->h_addr,
                     static_cast<size_t>(server->h_length));
        addr.sin_port = htons(static_cast<uint16_t>(master_port));

        constexpr int max_retries = 10;
        bool connected = false;
        for (int attempt = 0; attempt < max_retries; ++attempt) {
            if (connect(sock.get(), reinterpret_cast<struct sockaddr*>(&addr),
                        sizeof(addr)) == 0) {
                connected = true;
                break;
            }
            usleep(static_cast<useconds_t>(100000) << attempt);
        }

        if (!connected) {
            throw std::runtime_error(
                "NCCLProcessGroup: failed to connect to master at " +
                master_addr + ":" + std::to_string(master_port) +
                " after " + std::to_string(max_retries) + " retries");
        }

        ssize_t received = recv(sock.get(), id.get(), sizeof(ncclUniqueId),
                                MSG_WAITALL);
        if (received != static_cast<ssize_t>(sizeof(ncclUniqueId))) {
            throw std::runtime_error(
                "NCCLProcessGroup: failed to receive unique ID from master");
        }
    }

    // Success: hand the heap-allocated id off to the constructor via comm_.
    // The constructor is responsible for the sole matching `delete` — see
    // the site around line 210 in this file.
    comm_ = static_cast<void*>(id.release());
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

auto NCCLProcessGroup::all_to_all_single(Tensor& output, const Tensor& input) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(input);
    validate_gpu_tensor(output);

    if (input.shape().empty()) {
        throw std::invalid_argument(
            "NCCLProcessGroup::all_to_all_single: input must have at least 1 dimension");
    }
    const int64_t total = input.shape()[0];
    if (total % world_size_ != 0) {
        throw std::invalid_argument(
            "NCCLProcessGroup::all_to_all_single: input.shape[0] (" +
            std::to_string(total) + ") must be divisible by world_size (" +
            std::to_string(world_size_) + ")"
        );
    }
    auto same_shape = [](std::span<const int64_t> a, std::span<const int64_t> b) {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
    };
    if (!same_shape(output.shape(), input.shape()) || output.dtype() != input.dtype()) {
        throw std::invalid_argument(
            "NCCLProcessGroup::all_to_all_single: output must match input shape and dtype");
    }

    // Each rank sends/receives `chunk_elems` elements per peer.
    const int64_t per_peer_elems = input.numel() / world_size_;
    const int64_t elem_bytes = static_cast<int64_t>(dtype_size(input.dtype()));
    const int64_t per_peer_bytes = per_peer_elems * elem_bytes;

    auto* send_base = static_cast<const std::byte*>(input.data_ptr());
    auto* recv_base = static_cast<std::byte*>(output.data_ptr());

    ncclComm_t nccl_comm = static_cast<ncclComm_t>(comm_);
    ncclDataType_t nccl_dtype = to_nccl_datatype(input.dtype());

    // Group the W * 2 - 2 (skip self) ncclSend/ncclRecv calls so NCCL can fuse
    // them into a single kernel launch and avoid deadlocks.
    NCCL_PG_CHECK(ncclGroupStart());
    for (int peer = 0; peer < world_size_; ++peer) {
        const std::byte* send_chunk = send_base + peer * per_peer_bytes;
        std::byte* recv_chunk = recv_base + peer * per_peer_bytes;
        if (peer == rank_) {
            // Self-copy via cudaMemcpy on the default stream. ncclSend/Recv to
            // self is supported but a direct copy avoids the NCCL group cost.
            NCCL_PG_GPU_CHECK(cudaMemcpyAsync(
                recv_chunk, send_chunk, per_peer_bytes,
                cudaMemcpyDeviceToDevice, /*stream=*/0));
            continue;
        }
        NCCL_PG_CHECK(ncclSend(
            send_chunk, per_peer_elems, nccl_dtype, peer, nccl_comm,
            /*stream=*/nullptr));
        NCCL_PG_CHECK(ncclRecv(
            recv_chunk, per_peer_elems, nccl_dtype, peer, nccl_comm,
            /*stream=*/nullptr));
    }
    NCCL_PG_CHECK(ncclGroupEnd());

    NCCL_PG_GPU_CHECK(cudaDeviceSynchronize());
#else
    (void)output;
    (void)input;
    throw std::runtime_error("NCCLProcessGroup: NCCL not available");
#endif
}

auto NCCLProcessGroup::split(int color, int key)
    -> std::shared_ptr<ProcessGroupBase>
{
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    // NCCL_SPLIT_NOCOLOR (-1): rank opts out, receives nullptr.
    if (color == NCCL_SPLIT_NOCOLOR) {
        // Still collective — must participate so peers don't deadlock.
        ncclComm_t parent = static_cast<ncclComm_t>(comm_);
        ncclComm_t new_comm = nullptr;
        NCCL_PG_CHECK(ncclCommSplit(parent, NCCL_SPLIT_NOCOLOR, key,
                                    &new_comm, /*config=*/nullptr));
        return nullptr;
    }

    ncclComm_t parent = static_cast<ncclComm_t>(comm_);
    ncclComm_t new_comm = nullptr;
    NCCL_PG_CHECK(ncclCommSplit(parent, color, key, &new_comm,
                                /*config=*/nullptr));
    if (new_comm == nullptr) {
        throw std::runtime_error("NCCLProcessGroup::split: ncclCommSplit "
                                  "returned a null communicator");
    }

    int new_rank = -1;
    int new_size = -1;
    NCCL_PG_CHECK(ncclCommUserRank(new_comm, &new_rank));
    NCCL_PG_CHECK(ncclCommCount(new_comm, &new_size));

    // `std::make_shared` can't access the private constructor, so use
    // shared_ptr directly with a fresh allocation.
    return std::shared_ptr<NCCLProcessGroup>(
        new NCCLProcessGroup(new_rank, new_size, new_comm));
#else
    (void)color; (void)key;
    throw std::runtime_error("NCCLProcessGroup::split: NCCL not available");
#endif
}

#endif // TENZOR_HAS_NCCL

} // namespace tenzor::distributed
