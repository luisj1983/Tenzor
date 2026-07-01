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
#include "tenzor/ops/math.hpp"   // round() for integer ReduceOp::AVG rounding
#include <atomic>
#include <stdexcept>
#include "tenzor/utils/error.hpp"  // NotImplementedError (S25 / audit-12)
#include <memory>
#include <cstdlib>
#include <cstring>
#include <limits>

#ifdef TENZOR_HAS_MPI
#include "tenzor/distributed/mpi_backend.hpp"
#include <mpi.h>
#endif

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
// GlooProcessGroup Implementation
// ============================================================================

GlooProcessGroup::GlooProcessGroup(int rank, int world_size,
                                   const std::string& master_addr,
                                   int master_port)
    : rank_(rank), world_size_(world_size),
      master_addr_(master_addr), master_port_(master_port) {

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

    // Validate the port before it reaches htons() in the rendezvous socket
    // setup; an out-of-range value would otherwise be silently truncated to 16
    // bits, routing the group to an unintended/colliding port.
    if (master_port < 1 || master_port > 65535) {
        throw std::invalid_argument(
            "GlooProcessGroup: master_port " + std::to_string(master_port) +
            " out of range [1, 65535]"
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
    // Inf-F5: native paired send/recv all_to_all on Gloo's TCP transport.
    //
    // Sends only chunk-sized payloads per peer (vs. the base-class
    // all_gather fallback which sends full input to every peer — O(W²)
    // chunk volume on the wire). With paired ordering (low-rank
    // sends-first; high-rank recvs-first) the loop is deadlock-free.
    //
    // Contract validation mirrors the base class so callers see a single
    // consistent error shape regardless of override path.
    const int ws = world_size();
    const int my_rank = rank();
    if (input.shape().empty()) {
        throw std::invalid_argument(
            "GlooProcessGroup::all_to_all_single: input must have at least 1 dimension");
    }
    const int64_t total = input.shape()[0];
    if (total % ws != 0) {
        throw std::invalid_argument(
            "GlooProcessGroup::all_to_all_single: input.shape[0] (" +
            std::to_string(total) + ") must be divisible by world_size (" +
            std::to_string(ws) + ")");
    }
    auto same_shape = [](std::span<const int64_t> a, std::span<const int64_t> b) {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
    };
    if (!same_shape(output.shape(), input.shape()) ||
        output.dtype() != input.dtype()) {
        throw std::invalid_argument(
            "GlooProcessGroup::all_to_all_single: output must match input shape and dtype");
    }

    const int64_t chunk = total / ws;

    // Build per-peer receive buffers (fresh contiguous tensors); the
    // i-th slot becomes the chunk received from peer i (with i == my_rank
    // being the local-copy fast path).
    auto chunk_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    chunk_shape[0] = chunk;
    std::vector<Tensor> per_peer(ws);

    // Local copy: my chunk of `input` ends up in `per_peer[my_rank]` —
    // no network round-trip.
    per_peer[my_rank] =
        input.slice(0, my_rank * chunk, (my_rank + 1) * chunk).contiguous();

    // For each remote peer, exchange one chunk. Pair (a, b) where a<b:
    // a sends first then receives; b receives first then sends. This
    // avoids the deadlock you'd get if both peers blocked on send.
    for (int peer = 0; peer < ws; ++peer) {
        if (peer == my_rank) continue;
        auto send_chunk =
            input.slice(0, peer * chunk, (peer + 1) * chunk).contiguous();
        Tensor recv_chunk = zeros(chunk_shape, input.dtype(), input.device());
        if (my_rank < peer) {
            pg_->send(send_chunk, peer);
            pg_->recv(recv_chunk, peer);
        } else {
            pg_->recv(recv_chunk, peer);
            pg_->send(send_chunk, peer);
        }
        per_peer[peer] = std::move(recv_chunk);
    }

    // Assemble: peer-ordered concatenation along dim 0.
    // H5 fix: write into the caller's pre-allocated `output` storage in-place
    // instead of rebinding (which would orphan any autograd Variable / view
    // aliased to it). Use the stride-aware zero_()/+= pair rather than a flat
    // std::memcpy so a non-contiguous (strided) output is written correctly
    // through its own strides instead of being corrupted as if contiguous.
    auto assembled = cat(per_peer, 0).reshape(
        std::vector<int64_t>(output.shape().begin(), output.shape().end()));
    output.zero_();
    output += assembled;
}

// Inf-F4: collective sub-PG creation. Builds a fresh GlooProcessGroup on
// a derived TCP port for ranks sharing the same color, ordered by key.
auto GlooProcessGroup::split(int color, int key)
    -> std::shared_ptr<ProcessGroupBase>
{
    // Build per-rank (color, key, rank) and all_gather over the parent.
    // Pack as Int32 triples to keep the all_gather simple.
    auto local = zeros({3}, DType::Int32, Device::cpu());
    auto* lp = local.data<int32_t>();
    lp[0] = static_cast<int32_t>(color);
    lp[1] = static_cast<int32_t>(key);
    lp[2] = static_cast<int32_t>(rank_);

    std::vector<Tensor> gathered(world_size_);
    all_gather(gathered, local);

    // For each peer, extract (color, key, peer_rank).
    struct Member { int color; int key; int rank; };
    std::vector<Member> members;
    members.reserve(world_size_);
    for (int r = 0; r < world_size_; ++r) {
        auto* g = gathered[r].data<int32_t>();
        members.push_back({g[0], g[1], g[2]});
    }

    // M3 fix: previously used `master_port_ + 1000 + color` which collides
    // on nested splits or same-color splits from sibling parent PGs. Now
    // derive an offset that depends on both the parent's port AND a
    // process-local counter of splits performed so far, so every split-
    // derived child PG gets a unique port within the process.
    //
    // For multi-process distribution the counter is process-local but the
    // sequence is identical across ranks (since they all execute split()
    // collectively in the same order) — so the derived port matches.
    //
    // CRITICAL: every rank MUST advance this counter in lockstep, INCLUDING
    // ranks that opt out (color < 0). If the opt-out early-return below
    // skipped the fetch_add, an opting-out rank's counter would lag, and on
    // the next collective split() the participating ranks would compute a
    // different my_split_id (hence a different child port) than they did on
    // ranks that took this same split — colliding/mismatched ports and a
    // hung child rendezvous. Increment BEFORE the opt-out check.
    static std::atomic<int> split_counter{0};
    const int my_split_id = split_counter.fetch_add(1, std::memory_order_relaxed);

    // Opt-out path: this rank doesn't participate. The counter was already
    // advanced above so this rank stays in sync with participating peers.
    if (color < 0) {
        return nullptr;
    }

    // Collect members sharing my color, sort by (key, rank) for deterministic
    // new-rank ordering (matches MPI_Comm_split convention).
    std::vector<Member> my_group;
    for (const auto& m : members) {
        if (m.color == color) my_group.push_back(m);
    }
    std::sort(my_group.begin(), my_group.end(),
              [](const Member& a, const Member& b) {
                  if (a.key != b.key) return a.key < b.key;
                  return a.rank < b.rank;
              });

    // Find my position in the sorted group → new rank.
    int new_rank = -1;
    for (size_t i = 0; i < my_group.size(); ++i) {
        if (my_group[i].rank == rank_) {
            new_rank = static_cast<int>(i);
            break;
        }
    }
    if (new_rank < 0) {
        throw std::runtime_error(
            "GlooProcessGroup::split: internal — local rank not found in own color group");
    }
    int new_world_size = static_cast<int>(my_group.size());

    // Use a large enough stride (10000) so colors within one split don't
    // collide with the next split's color-0.
    const int new_port = master_port_ + 1000 + my_split_id * 10000 + color;

    // Validate the derived port fits in a TCP port (uint16). With stride 10000,
    // only a few splits exhaust the range from the default 29500; previously the
    // out-of-range value was silently truncated by htons() in the child PG's
    // sockaddr, routing it to an unintended/colliding port (hang or collision).
    // Fail fast instead, mirroring the NCCL bootstrap path's [1,65535] guard.
    if (new_port < 1 || new_port > 65535) {
        throw std::runtime_error(
            "GlooProcessGroup::split: derived port " + std::to_string(new_port) +
            " out of range [1, 65535] (master_port=" + std::to_string(master_port_) +
            ", split_id=" + std::to_string(my_split_id) + ", color=" +
            std::to_string(color) + "); too many splits for the available port range");
    }

    return std::make_shared<GlooProcessGroup>(
        new_rank, new_world_size, master_addr_, new_port);
}

// ============================================================================
// ProcessGroupBase::split default: throw with a documented contract.
//
// M8 note: this is intentionally a non-pure-virtual default so:
//   (a) future backends that don't support sub-group creation can still
//       inherit from ProcessGroupBase without forcing them to provide a
//       split implementation,
//   (b) tests that use a mock ProcessGroup don't need to override split,
//   (c) the existing override set (NCCL via ncclCommSplit, MPI via
//       MPI_Comm_split, Gloo via gather + new rendezvous) demonstrates
//       the contract for backends that DO support split.
//
// If you're adding a new ProcessGroup subclass that supports sub-group
// creation, override this method.
// ============================================================================
auto ProcessGroupBase::split(int color, int key)
    -> std::shared_ptr<ProcessGroupBase>
{
    (void)color; (void)key;
    throw std::runtime_error(
        "ProcessGroupBase::split: not implemented for this backend. "
        "Currently supported: NCCLProcessGroup (ncclCommSplit), "
        "MPIProcessGroup (MPI_Comm_split), GlooProcessGroup (gather + "
        "rendezvous). Add a `split()` override on your custom backend.");
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
    // H5 fix: in-place writeback so autograd aliases survive. Write THROUGH the
    // caller's storage with the stride-aware zero_()/+= pair (operator+= honours
    // output's strides) instead of a flat std::memcpy: a non-contiguous output
    // (a strided view) would otherwise have the assembled bytes written into it
    // as if contiguous, corrupting the view and any aliasing tensor. Reshape the
    // 1-D assembled buffer to the output shape so the elementwise add lines up.
    auto assembled_b = cat(per_peer, 0).reshape(
        std::vector<int64_t>(output.shape().begin(), output.shape().end()));
    output.zero_();
    output += assembled_b;
}

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
        barrier_dummy_.emplace(zeros({1}, DType::Float32, Device::cuda(device_id)));
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

// ============================================================================
// Inf-F1: MPIProcessGroup Implementation
// ============================================================================

#ifdef TENZOR_HAS_MPI

namespace {

inline auto reduce_op_to_mpi(ReduceOp op) -> MPI_Op {
    switch (op) {
        case ReduceOp::SUM:     return MPI_SUM;
        case ReduceOp::AVG:     return MPI_SUM;  // divide by world_size on caller
        case ReduceOp::PRODUCT: return MPI_PROD;
        case ReduceOp::MIN:     return MPI_MIN;
        case ReduceOp::MAX:     return MPI_MAX;
        default:
            throw std::invalid_argument(
                "MPIProcessGroup: unsupported ReduceOp");
    }
}

inline auto dtype_to_mpi(DType dtype) -> MPI_Datatype {
    switch (dtype) {
        case DType::Float32: return MPI_FLOAT;
        case DType::Float64: return MPI_DOUBLE;
        case DType::Int32:   return MPI_INT;
        case DType::Int64:   return MPI_LONG_LONG_INT;
        case DType::Int8:    return MPI_INT8_T;
        case DType::UInt8:   return MPI_UINT8_T;
        case DType::Int16:   return MPI_INT16_T;
        case DType::UInt16:  return MPI_UINT16_T;
        case DType::UInt32:  return MPI_UINT32_T;
        case DType::UInt64:  return MPI_UINT64_T;
        case DType::Bool:    return MPI_C_BOOL;
        default:
            // Sentinel — callers must validate dtype before reaching this
            // path. Returning MPI_BYTE was unsafe because MPI_SUM on raw
            // bytes is undefined. Use `validate_mpi_reducible_dtype` below.
            return MPI_BYTE;
    }
}

// H3 fix: explicit gate that throws on any dtype MPI cannot natively
// reduce (F16/BF16/Complex/quantized). Callers must widen to F32 first.
inline auto validate_mpi_reducible_dtype(DType dtype, const char* op) -> void {
    switch (dtype) {
        case DType::Float32: case DType::Float64:
        case DType::Int8:  case DType::Int16: case DType::Int32: case DType::Int64:
        case DType::UInt8: case DType::UInt16: case DType::UInt32: case DType::UInt64:
        case DType::Bool:
            return;
        default:
            throw std::runtime_error(
                std::string("MPIProcessGroup::") + op +
                ": dtype " + std::string(dtype_name(dtype)) +
                " has no native MPI representation. Widen to Float32 before reducing.");
    }
}

#define MPI_PG_CHECK(call) \
    do { \
        int err__ = (call); \
        if (err__ != MPI_SUCCESS) { \
            char err_buf__[MPI_MAX_ERROR_STRING]; int err_len__ = 0; \
            MPI_Error_string(err__, err_buf__, &err_len__); \
            throw std::runtime_error( \
                std::string("MPIProcessGroup: MPI call failed: ") + err_buf__); \
        } \
    } while (0)

// MPI's classic collective APIs take the per-rank element count as an `int`.
// tensor.numel() is int64; for tensors with >2^31 elements the narrowing cast
// would silently truncate (or wrap negative) and MPI would transfer the wrong
// element count, silently corrupting the result. Validate the range and throw a
// clear message instead. (Mirrors MPIBackend::checked_mpi_count.)
inline auto checked_mpi_pg_count(int64_t numel, const char* op) -> int {
    if (numel < 0 || numel > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            std::string("MPIProcessGroup::") + op + ": element count " +
            std::to_string(numel) + " exceeds INT_MAX; tensors with more than " +
            std::to_string(std::numeric_limits<int>::max()) +
            " elements require MPI large-count APIs or chunking");
    }
    return static_cast<int>(numel);
}

// MPI reads/writes the buffer as flat-contiguous. A CUDA (or other non-CPU)
// tensor's data_ptr() is a device pointer that non-CUDA-aware MPI would
// dereference on the host (crash/garbage); a non-contiguous tensor would be
// transferred as if contiguous (corruption). MPIProcessGroup uses MPI_COMM_WORLD
// directly without the CUDA-aware staging the MPIBackend performs, so require
// the tensor to be a contiguous CPU tensor and fail loudly otherwise. Callers
// must stage GPU tensors to host (tensor.to(cpu())) and make views contiguous
// before invoking these collectives.
inline auto require_mpi_pg_buffer(const Tensor& t, const char* op) -> void {
    if (t.device().type != Device::Type::CPU) {
        throw std::invalid_argument(
            std::string("MPIProcessGroup::") + op +
            ": tensor must be on CPU (MPI_COMM_WORLD path is not CUDA-aware here); "
            "stage GPU tensors to host before reducing");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(
            std::string("MPIProcessGroup::") + op +
            ": tensor must be contiguous; MPI transfers the buffer as flat "
            "contiguous and would corrupt a strided view");
    }
}

}  // anonymous namespace

MPIProcessGroup::MPIProcessGroup(int rank, int world_size,
                                 const std::string& /*master_addr*/,
                                 int /*master_port*/)
    : rank_(rank), world_size_(world_size) {
    // Ensure MPI is initialized — match MPIBackend::initialize semantics.
    int inited = 0;
    MPI_PG_CHECK(MPI_Initialized(&inited));
    if (!inited) {
        int provided = 0;
        MPI_PG_CHECK(MPI_Init_thread(nullptr, nullptr,
                                     MPI_THREAD_MULTIPLE, &provided));
    }
    MPI_Comm world = MPI_COMM_WORLD;
    comm_ = static_cast<void*>(world);
    owns_comm_ = false;  // MPI_COMM_WORLD is owned by MPI itself

    // Sanity-check rank/world_size against the communicator.
    int comm_rank = -1, comm_size = -1;
    MPI_PG_CHECK(MPI_Comm_rank(world, &comm_rank));
    MPI_PG_CHECK(MPI_Comm_size(world, &comm_size));
    if (comm_rank != rank || comm_size != world_size) {
        throw std::invalid_argument(
            "MPIProcessGroup: requested (rank=" + std::to_string(rank) +
            ", world_size=" + std::to_string(world_size) +
            ") does not match MPI_COMM_WORLD (rank=" +
            std::to_string(comm_rank) + ", size=" + std::to_string(comm_size) + ")");
    }
}

MPIProcessGroup::MPIProcessGroup(int rank, int world_size,
                                 void* comm, bool owns)
    : rank_(rank), world_size_(world_size),
      comm_(comm), owns_comm_(owns) {}

MPIProcessGroup::~MPIProcessGroup() {
    if (owns_comm_ && comm_ != nullptr) {
        MPI_Comm c = reinterpret_cast<MPI_Comm>(comm_);
        if (c != MPI_COMM_NULL) {
            MPI_Comm_free(&c);  // ignore error at destruction time
        }
    }
}

auto MPIProcessGroup::validate_initialized() const -> void {
    if (comm_ == nullptr) {
        throw std::runtime_error(
            "MPIProcessGroup: communicator is null (uninitialized)");
    }
}

auto MPIProcessGroup::all_reduce(Tensor& tensor, ReduceOp op) -> void {
    validate_initialized();
    validate_mpi_reducible_dtype(tensor.dtype(), "all_reduce");
    require_mpi_pg_buffer(tensor, "all_reduce");
    auto comm = reinterpret_cast<MPI_Comm>(comm_);
    MPI_PG_CHECK(MPI_Allreduce(MPI_IN_PLACE, tensor.data_ptr(),
                               checked_mpi_pg_count(tensor.numel(), "all_reduce"),
                               dtype_to_mpi(tensor.dtype()),
                               reduce_op_to_mpi(op), comm));
    if (op == ReduceOp::AVG && world_size_ > 1) {
        // In-place divide preserves the caller's storage pointer so aliased
        // Variables/views see the result. Integer AVG must round to nearest
        // (matching Gloo / MPIBackend), not truncate via integer division:
        // widen to Float64, divide, round, narrow back, write through.
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
    }
}

auto MPIProcessGroup::broadcast(Tensor& tensor, int src_rank) -> void {
    validate_initialized();
    require_mpi_pg_buffer(tensor, "broadcast");
    auto comm = reinterpret_cast<MPI_Comm>(comm_);
    MPI_PG_CHECK(MPI_Bcast(tensor.data_ptr(),
                           checked_mpi_pg_count(tensor.numel(), "broadcast"),
                           dtype_to_mpi(tensor.dtype()),
                           src_rank, comm));
}

auto MPIProcessGroup::all_gather(std::vector<Tensor>& output,
                                 const Tensor& input) -> void {
    validate_initialized();
    if (static_cast<int>(output.size()) != world_size_) {
        throw std::invalid_argument(
            "MPIProcessGroup::all_gather: output.size() must equal world_size");
    }
    require_mpi_pg_buffer(input, "all_gather");
    auto comm = reinterpret_cast<MPI_Comm>(comm_);
    // Use a flat contiguous buffer for the gather, then slice into output[].
    int64_t per_rank_numel = input.numel();
    const int gather_count = checked_mpi_pg_count(per_rank_numel, "all_gather");
    Tensor flat = zeros({static_cast<int64_t>(world_size_) * per_rank_numel},
                        input.dtype(), input.device());
    MPI_PG_CHECK(MPI_Allgather(input.data_ptr(),
                               gather_count,
                               dtype_to_mpi(input.dtype()),
                               flat.data_ptr(),
                               gather_count,
                               dtype_to_mpi(input.dtype()),
                               comm));
    auto in_shape = input.shape();
    for (int r = 0; r < world_size_; ++r) {
        output[r] = flat.slice(0, r * per_rank_numel, (r + 1) * per_rank_numel)
                        .reshape(std::vector<int64_t>(in_shape.begin(), in_shape.end()));
    }
}

auto MPIProcessGroup::reduce_scatter(Tensor& output,
                                     std::span<const Tensor> input) -> void {
    validate_initialized();
    validate_mpi_reducible_dtype(output.dtype(), "reduce_scatter");
    require_mpi_pg_buffer(output, "reduce_scatter");
    if (static_cast<int>(input.size()) != world_size_) {
        throw std::invalid_argument(
            "MPIProcessGroup::reduce_scatter: input.size() must equal world_size");
    }
    for (const auto& chunk : input) {
        if (chunk.dtype() != output.dtype()) {
            throw std::invalid_argument(
                "MPIProcessGroup::reduce_scatter: input chunk dtype must match output dtype");
        }
    }
    auto comm = reinterpret_cast<MPI_Comm>(comm_);
    // MPI_Reduce_scatter_block expects a contiguous send buffer with one
    // chunk per peer. Concatenate input[] along dim 0 first.
    std::vector<Tensor> in_vec(input.begin(), input.end());
    Tensor send_buf = cat(in_vec, 0).contiguous();
    // The send buffer must hold exactly output.numel() * world_size elements;
    // a caller-contract violation would otherwise yield an out-of-bounds read.
    if (send_buf.numel() != output.numel() * world_size_) {
        throw std::invalid_argument(
            "MPIProcessGroup::reduce_scatter: concatenated input numel " +
            std::to_string(send_buf.numel()) +
            " must equal output numel * world_size (" +
            std::to_string(output.numel()) + " * " +
            std::to_string(world_size_) + ")");
    }
    // SUM-only by interface contract (see ProcessGroupBase::reduce_scatter);
    // averaging is performed caller-side by dividing the output by world_size.
    MPI_PG_CHECK(MPI_Reduce_scatter_block(
        send_buf.data_ptr(), output.data_ptr(),
        checked_mpi_pg_count(output.numel(), "reduce_scatter"),
        dtype_to_mpi(output.dtype()),
        MPI_SUM, comm));
}

auto MPIProcessGroup::all_to_all_single(Tensor& output,
                                        const Tensor& input) -> void {
    // Inf-F3: native MPI_Alltoallv on the owning communicator. Validates
    // the same contract as the base class.
    validate_initialized();
    if (input.shape().empty()) {
        throw std::invalid_argument(
            "MPIProcessGroup::all_to_all_single: input must have at least 1 dim");
    }
    const int64_t total = input.shape()[0];
    if (total % world_size_ != 0) {
        throw std::invalid_argument(
            "MPIProcessGroup::all_to_all_single: input.shape[0] (" +
            std::to_string(total) + ") must be divisible by world_size (" +
            std::to_string(world_size_) + ")");
    }
    auto same_shape = [](std::span<const int64_t> a, std::span<const int64_t> b) {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
    };
    if (!same_shape(output.shape(), input.shape()) ||
        output.dtype() != input.dtype()) {
        throw std::invalid_argument(
            "MPIProcessGroup::all_to_all_single: output must match input shape and dtype");
    }
    require_mpi_pg_buffer(input, "all_to_all_single");
    require_mpi_pg_buffer(output, "all_to_all_single");
    auto comm = reinterpret_cast<MPI_Comm>(comm_);
    const int chunk_elems =
        checked_mpi_pg_count(input.numel() / world_size_, "all_to_all_single");
    // Equal-size chunks → MPI_Alltoall suffices and avoids per-peer displs.
    MPI_PG_CHECK(MPI_Alltoall(input.data_ptr(),  chunk_elems,
                              dtype_to_mpi(input.dtype()),
                              output.data_ptr(), chunk_elems,
                              dtype_to_mpi(output.dtype()),
                              comm));
}

auto MPIProcessGroup::split(int color, int key)
    -> std::shared_ptr<ProcessGroupBase> {
    // Inf-F2: native MPI_Comm_split. Maps `color < 0` to MPI_UNDEFINED so
    // the opting-out rank receives MPI_COMM_NULL → returns nullptr (same
    // contract as NCCLProcessGroup::split).
    validate_initialized();
    auto parent = reinterpret_cast<MPI_Comm>(comm_);
    int mpi_color = (color < 0) ? MPI_UNDEFINED : color;
    MPI_Comm new_comm = MPI_COMM_NULL;
    MPI_PG_CHECK(MPI_Comm_split(parent, mpi_color, key, &new_comm));
    if (new_comm == MPI_COMM_NULL) {
        return nullptr;  // we opted out
    }
    int new_rank = -1, new_size = -1;
    MPI_PG_CHECK(MPI_Comm_rank(new_comm, &new_rank));
    MPI_PG_CHECK(MPI_Comm_size(new_comm, &new_size));
    // shared_ptr can't reach the private ctor; use new directly.
    return std::shared_ptr<MPIProcessGroup>(
        new MPIProcessGroup(new_rank, new_size,
                            static_cast<void*>(new_comm),
                            /*owns=*/true));
}

auto MPIProcessGroup::barrier() -> void {
    validate_initialized();
    auto comm = reinterpret_cast<MPI_Comm>(comm_);
    MPI_PG_CHECK(MPI_Barrier(comm));
}

#endif  // TENZOR_HAS_MPI

}  // namespace tenzor::distributed
