/**
 * @file nccl_process_group.cpp
 * @brief NCCLProcessGroup implementation — compiled into the GPU backend DSO.
 *
 * Extracted from src/distributed/process_group.cpp so that the host
 * tenzor_core library stays free of <nccl.h>/<rccl/rccl.h> includes (which
 * clash with the CUDA runtime headers in a combined CUDA+ROCm build because
 * the ROCm include path shadows /usr/include/nccl.h). This TU is built as
 * part of tenzor_backend_cuda (NVIDIA NCCL, cudaStream_t) and — once RCCL
 * support lands — tenzor_backend_rocm (AMD RCCL, hipStream_t). It is NOT
 * compiled into tenzor_core. TENZOR_HAS_NCCL is defined by the owning
 * backend target; the surrounding #ifdef mirrors the host header's guard so
 * this file compiles to an empty TU when the collective library is absent.
 */

#include "tenzor/distributed/process_group.hpp"
#include "tenzor/distributed/distributed.hpp"
#include "tenzor/utils/error.hpp"  // NotImplementedError
#include "tenzor/ops/math.hpp"     // round() for integer ReduceOp::AVG rounding
#include <stdexcept>

#if defined(TENZOR_HAS_NCCL)
    #ifdef TENZOR_USE_ROCM
        #include <rccl/rccl.h>
    #else
        #include <nccl.h>
    #endif
    #include "tenzor/backend/loader.hpp"  // for is_backend_registry_alive()
    #include "tenzor/distributed/nccl_backend.hpp"  // shared to_nccl_reduce_op/to_nccl_datatype
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
        // Map CUDA runtime API names to HIP equivalents (same technique as
        // ddp.cpp/fsdp.cpp/zero_optimizer.cpp and nccl_backend.cpp in this
        // codebase) so the collective body below, written against the cuda*
        // names, compiles unchanged for ROCm.
        #define cudaStream_t hipStream_t
        #define cudaSetDevice hipSetDevice
        #define cudaGetDevice hipGetDevice
        #define cudaDeviceSynchronize hipDeviceSynchronize
        #define cudaMemcpyAsync hipMemcpyAsync
        #define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
        #define cudaStreamSynchronize hipStreamSynchronize
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

#endif // TENZOR_HAS_NCCL

namespace tenzor::distributed {

// ============================================================================
// NCCLProcessGroup Implementation
// ============================================================================

#ifdef TENZOR_HAS_NCCL

// Private constructor (audit A3-extended): wraps an already-created
// ncclComm_t produced by `ncclCommSplit`. Skips the TCP bootstrap — the
// caller (`NCCLProcessGroup::split`) has already done the synchronous
// collective that constructs the new communicator.
NCCLProcessGroup::NCCLProcessGroup(int rank, int world_size, void* comm,
                                   int comm_device)
    : rank_(rank), world_size_(world_size), comm_(comm),
      comm_device_(comm_device) {
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

    // Step 2: Create the NCCL communicator on the current GPU and RECORD that
    // device. All later collectives/barrier are bound to this device; using the
    // ambient cudaGetDevice() at call time could pick a different device than
    // the comm is bound to (undefined NCCL behaviour).
    int device_id = 0;
    NCCL_PG_GPU_CHECK(cudaGetDevice(&device_id));
    NCCL_PG_GPU_CHECK(cudaSetDevice(device_id));
    comm_device_ = device_id;

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

        // Validate the resolver result before copying: gethostbyname can
        // return an IPv6 (AF_INET6, h_length==16) or otherwise oversized
        // record (e.g. from a hostile/compromised DNS response), and copying
        // h_length bytes into the 4-byte sin_addr would overflow the struct.
        if (server->h_addrtype != AF_INET ||
            server->h_length != static_cast<int>(sizeof(struct in_addr)) ||
            server->h_addr == nullptr) {
            throw std::runtime_error(
                "NCCLProcessGroup: resolver returned a non-IPv4 address for "
                "master: " + master_addr);
        }

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        std::memcpy(&addr.sin_addr.s_addr, server->h_addr,
                     sizeof(struct in_addr));
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
    // The communicator is bound to exactly one device (comm_device_). Running a
    // collective on a tensor located on a different device is undefined NCCL
    // behaviour (the comm rank is bound to comm_device_), so reject it loudly
    // instead of silently corrupting/hanging. comm_device_ == -1 only for a
    // not-yet-initialized instance, which the collectives never reach.
    if (comm_device_ >= 0 && tensor.device().index != comm_device_) {
        throw std::invalid_argument(
            "NCCLProcessGroup: tensor is on device " +
            std::to_string(tensor.device().index) +
            " but this process group's communicator is bound to device " +
            std::to_string(comm_device_) +
            ". Use one NCCLProcessGroup per device.");
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

    // NCCL treats data_ptr() as a flat contiguous buffer; a non-contiguous
    // input/output would be read/written as if contiguous and corrupt the
    // result. Stage a contiguous copy and propagate the reduced values back
    // (mirrors NCCLBackend::all_reduce).
    const bool staged = !tensor.is_contiguous();
    Tensor work = staged ? tensor.contiguous() : tensor;

    NCCL_PG_CHECK(ncclAllReduce(
        work.data_ptr(),
        work.data_ptr(),
        work.numel(),
        nccl_dtype,
        nccl_op,
        nccl_comm,
        nullptr  // default CUDA stream
    ));

    NCCL_PG_GPU_CHECK(cudaDeviceSynchronize());

    // AVG = SUM followed by division
    if (op == ReduceOp::AVG && world_size_ > 1) {
        // Mirroring MPIProcessGroup::all_reduce: in-place divide preserves the
        // staged buffer's storage. Integer AVG rounds to nearest (matching Gloo
        // / MPIBackend) rather than truncating via integer division.
        if (is_integer_type(work.dtype())) {
            const DType orig = work.dtype();
            Tensor widened = work.to(DType::Float64);
            auto divisor = tenzor::full({1}, static_cast<double>(world_size_),
                                        DType::Float64, widened.device());
            widened /= divisor;
            Tensor rounded = tenzor::round(widened).to(orig);
            work.zero_();
            work += rounded.to(work.device());
        } else {
            auto scalar = tenzor::full({1}, static_cast<double>(world_size_),
                                        work.dtype(), work.device());
            work /= scalar;
        }
    }

    if (staged) {
        // Write back THROUGH the caller's storage rather than rebinding
        // `tensor = work`. A plain rebind swaps the intrusive_ptr to a fresh
        // contiguous impl, detaching any aliased Variable/view/grad that shares
        // the original strided storage (H6 aliasing hazard — the NCCLBackend and
        // MPIProcessGroup paths avoid it the same way). zero_() += writes through
        // the original strides/storage.
        tensor.zero_();
        tensor += work;
    }
#else
    (void)tensor;
    (void)op;
    throw NotImplementedError("NCCLProcessGroup: NCCL not available");
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
    // Unlike all_reduce(), we do NOT call cudaDeviceSynchronize() for the
    // non-AVG ops -- the caller is responsible for synchronization via CUDA
    // events.
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

    // ReduceOp::AVG maps to ncclSum (to_nccl_reduce_op), so the collective only
    // produces the SUM; the averaging divide must still be applied. Previously
    // this was skipped entirely (a comment claimed the caller divides post-sync,
    // but callers such as ZeROStage1Optimizer do NOT -- they pass AVG expecting a
    // true average), so the AVG async path silently returned a world_size-times
    // too-large SUM. There is no stream-aware elementwise op to enqueue the
    // divide on cuda_stream, so for AVG we synchronize the comm stream, perform
    // the divide, then fully synchronize so the averaged result is visible before
    // returning. This makes AVG effectively synchronous (correctness over
    // overlap); SUM/PRODUCT/MIN/MAX remain truly async. Integer dtypes widen to
    // Float64 + round-to-nearest, matching the synchronous all_reduce / MPI /
    // Gloo AVG.
    if (op == ReduceOp::AVG && world_size_ > 1) {
        NCCL_PG_GPU_CHECK(cudaStreamSynchronize(cuda_stream));
        if (is_integer_type(tensor.dtype())) {
            const DType orig = tensor.dtype();
            Tensor widened = tensor.to(DType::Float64);
            auto divisor = tenzor::full({1}, static_cast<double>(world_size_),
                                        DType::Float64, widened.device());
            widened /= divisor;
            Tensor rounded = tenzor::round(widened).to(orig);
            tensor.zero_();
            tensor += rounded.to(tensor.device());
        } else {
            auto scalar = tenzor::full({1}, static_cast<double>(world_size_),
                                        tensor.dtype(), tensor.device());
            tensor /= scalar;
        }
        NCCL_PG_GPU_CHECK(cudaDeviceSynchronize());
    }
#else
    (void)tensor;
    (void)op;
    (void)stream;
    throw NotImplementedError("NCCLProcessGroup: NCCL not available");
#endif
}

auto NCCLProcessGroup::broadcast(Tensor& tensor, int src_rank) -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    validate_gpu_tensor(tensor);

    ncclComm_t nccl_comm = static_cast<ncclComm_t>(comm_);
    ncclDataType_t nccl_dtype = to_nccl_datatype(tensor.dtype());

    // Stage a contiguous buffer: a non-contiguous receive buffer would
    // otherwise be written as if contiguous and corrupted (mirrors
    // NCCLBackend::broadcast).
    const bool staged = !tensor.is_contiguous();
    Tensor work = staged ? tensor.contiguous() : tensor;

    NCCL_PG_CHECK(ncclBroadcast(
        work.data_ptr(),
        work.data_ptr(),
        work.numel(),
        nccl_dtype,
        src_rank,
        nccl_comm,
        nullptr  // default CUDA stream
    ));

    NCCL_PG_GPU_CHECK(cudaDeviceSynchronize());

    if (staged) {
        // Write back through the caller's storage (see all_reduce) to preserve
        // aliased Variables/views instead of rebinding `tensor = work`.
        tensor.zero_();
        tensor += work;
    }
#else
    (void)tensor;
    (void)src_rank;
    throw NotImplementedError("NCCLProcessGroup: NCCL not available");
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

    // NCCL reads input.data_ptr() as a flat contiguous block; a non-contiguous
    // input would be read as if contiguous and gather corrupted data. Stage a
    // contiguous send buffer.
    Tensor send = input.is_contiguous() ? input : input.contiguous();

    // NCCL all-gather writes into a contiguous buffer; allocate one
    size_t total_elements = input.numel() * world_size_;
    Tensor gathered = empty(
        {static_cast<int64_t>(total_elements)},
        input.dtype(),
        input.device()
    );

    NCCL_PG_CHECK(ncclAllGather(
        send.data_ptr(),
        gathered.data_ptr(),
        send.numel(),
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
    throw NotImplementedError("NCCLProcessGroup: NCCL not available");
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

    // NCCL writes output.data_ptr() as a flat contiguous block; a
    // non-contiguous output would be written as if contiguous and corrupted.
    // Stage a contiguous receive buffer and propagate the result back.
    const bool staged = !output.is_contiguous();
    Tensor work = staged ? output.contiguous() : output;

    // Validate that the total input equals output * world_size (the same
    // check MPI and NCCLBackend enforce); a mismatch makes ncclReduceScatter
    // read/write past the buffers it was given.
    if (concatenated.numel() != work.numel() * static_cast<int64_t>(world_size_)) {
        throw std::invalid_argument(
            "NCCLProcessGroup::reduce_scatter: total input size (" +
            std::to_string(concatenated.numel()) + ") must equal output size (" +
            std::to_string(work.numel()) + ") * world_size (" +
            std::to_string(world_size_) + ")");
    }

    // SUM-only by interface contract (see ProcessGroupBase::reduce_scatter);
    // averaging is performed caller-side by dividing the output by world_size.
    NCCL_PG_CHECK(ncclReduceScatter(
        concatenated.data_ptr(),
        work.data_ptr(),
        work.numel(),
        nccl_dtype,
        ncclSum,
        nccl_comm,
        nullptr  // default CUDA stream
    ));

    NCCL_PG_GPU_CHECK(cudaDeviceSynchronize());

    if (staged) {
        // Write back through the caller's storage (see all_reduce) to preserve
        // aliased Variables/views instead of rebinding `output = work`.
        output.zero_();
        output += work;
    }
#else
    (void)output;
    (void)input;
    throw NotImplementedError("NCCLProcessGroup: NCCL not available");
#endif
}

auto NCCLProcessGroup::barrier() -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    // NCCL has no native barrier; perform an all-reduce on a dummy tensor.
    // L7 fix: cache the dummy tensor per-instance to avoid the per-call
    // allocation on the hot path. Lazy-init keyed on (device_id, dtype).
    // Use the communicator's OWN device (comm_device_), not the ambient
    // cudaGetDevice(): the comm is bound to one device, and a dummy on a
    // different current device would mismatch the comm and be undefined.
    const int device_id = comm_device_;
    if (!barrier_dummy_.has_value() ||
        barrier_dummy_->device().index != device_id) {
        // This TU is compiled once per owning backend DSO (see file header
        // comment) with exactly one of TENZOR_USE_CUDA/TENZOR_USE_ROCM active,
        // so the dummy's device type must match whichever backend this object
        // was built for instead of hardcoding CUDA (which would bind an
        // RCCL-backed communicator to a nonexistent/mismatched CUDA device).
#if defined(TENZOR_USE_CUDA)
        barrier_dummy_.emplace(zeros({1}, DType::Float32, Device::cuda(device_id)));
#elif defined(TENZOR_USE_ROCM)
        barrier_dummy_.emplace(zeros({1}, DType::Float32, Device::rocm(device_id)));
#endif
    }
    all_reduce(*barrier_dummy_, ReduceOp::SUM);
#else
    throw NotImplementedError("NCCLProcessGroup: NCCL not available");
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
    throw NotImplementedError("NCCLProcessGroup: NCCL not available");
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
        // With NCCL_SPLIT_NOCOLOR, NCCL is documented to return a null
        // communicator for the opting-out rank, so there is nothing to destroy.
        // Guard the contract: if a future runtime ever returns a real comm here,
        // destroy it rather than leaking it.
        if (new_comm != nullptr) {
            ncclCommDestroy(new_comm);
        }
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
    // shared_ptr directly with a fresh allocation. The child communicator is
    // bound to the same GPU device as this parent communicator.
    return std::shared_ptr<NCCLProcessGroup>(
        new NCCLProcessGroup(new_rank, new_size, new_comm, comm_device_));
#else
    (void)color; (void)key;
    throw NotImplementedError("NCCLProcessGroup::split: NCCL not available");
#endif
}

#endif // TENZOR_HAS_NCCL

} // namespace tenzor::distributed
