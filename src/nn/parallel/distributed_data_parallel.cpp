/**
 * @file distributed_data_parallel.cpp
 * @brief Implementation of DistributedDataParallel for multi-node training
 */

#include "tenzor/nn/parallel/distributed_data_parallel.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/utils/error.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>

#ifdef TENZOR_USE_ROCM
    #include <hip/hip_runtime.h>
    #define GPU_CHECK(call) \
        do { \
            hipError_t err = call; \
            if (err != hipSuccess) { \
                throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
            } \
        } while(0)
    #define NCCL_CHECK(call) \
        do { \
            ncclResult_t err = call; \
            if (err != ncclSuccess) { \
                throw std::runtime_error(std::string("RCCL error: ") + ncclGetErrorString(err)); \
            } \
        } while(0)
#elif defined(TENZOR_USE_CUDA)
    #include <cuda_runtime.h>
    #ifdef TENZOR_HAS_NCCL
        #include <nccl.h>
    #endif
    #define GPU_CHECK(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
            } \
        } while(0)
    #ifdef TENZOR_HAS_NCCL
        #define NCCL_CHECK(call) \
            do { \
                ncclResult_t err = call; \
                if (err != ncclSuccess) { \
                    throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(err)); \
                } \
            } while(0)
    #else
        #define NCCL_CHECK(call) \
            throw std::runtime_error("NCCL not available - recompile with NCCL support")
    #endif
#else
    #define GPU_CHECK(call) call
    #define NCCL_CHECK(call) call
#endif

namespace tenzor {
namespace nn {

// ============================================================================
// ProcessGroup Implementation
// ============================================================================

ProcessGroup::ProcessGroup(int rank, int world_size, const std::string& backend)
    : rank_(rank), world_size_(world_size), backend_(backend) {

    if (rank_ < 0 || rank_ >= world_size_) {
        throw std::invalid_argument(
            "ProcessGroup: rank " + std::to_string(rank_) +
            " must be in range [0, " + std::to_string(world_size_) + ")"
        );
    }

    if (world_size_ <= 0) {
        throw std::invalid_argument(
            "ProcessGroup: world_size must be positive, got " + std::to_string(world_size_)
        );
    }

    if (backend_ != "nccl" && backend_ != "gloo" && backend_ != "mpi") {
        throw std::invalid_argument(
            "ProcessGroup: unsupported backend '" + backend_ + "'. Supported: nccl, gloo, mpi"
        );
    }
}

ProcessGroup::~ProcessGroup() {
#if defined(TENZOR_USE_ROCM) || defined(TENZOR_USE_CUDA)
    // Destroy all NCCL communicators
    std::lock_guard<std::mutex> lock(comm_mutex_);
    for (auto& [device_id, comm] : communicators_) {
        if (comm != nullptr) {
            ncclCommDestroy(comm);
        }
    }
    communicators_.clear();
#endif
}

auto ProcessGroup::get_communicator(int device_id) -> ncclComm_t {
    std::lock_guard<std::mutex> lock(comm_mutex_);

    auto it = communicators_.find(device_id);
    if (it != communicators_.end()) {
        return it->second;
    }

    throw std::runtime_error(
        "ProcessGroup: communicator for device " + std::to_string(device_id) +
        " not initialized. Call init_communicator() first."
    );
}

auto ProcessGroup::init_communicator(int device_id, const ncclUniqueId& unique_id) -> void {
#if defined(TENZOR_USE_ROCM) || defined(TENZOR_USE_CUDA)
    std::lock_guard<std::mutex> lock(comm_mutex_);

    // Check if already initialized
    if (communicators_.find(device_id) != communicators_.end()) {
        return;  // Already initialized
    }

    // Set device
    #ifdef TENZOR_USE_ROCM
        GPU_CHECK(hipSetDevice(device_id));
    #else
        GPU_CHECK(cudaSetDevice(device_id));
    #endif

    // Initialize NCCL communicator
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, world_size_, unique_id, rank_));

    communicators_[device_id] = comm;
#else
    (void)device_id;
    (void)unique_id;
    throw std::runtime_error("ProcessGroup: NCCL support not compiled");
#endif
}

auto ProcessGroup::broadcast(Tensor& tensor, int src_rank, int device_id) -> void {
#if defined(TENZOR_USE_ROCM) || defined(TENZOR_USE_CUDA)
    auto comm = get_communicator(device_id);

    // Set device
    #ifdef TENZOR_USE_ROCM
        GPU_CHECK(hipSetDevice(device_id));
    #else
        GPU_CHECK(cudaSetDevice(device_id));
    #endif

    // Determine NCCL data type
    ncclDataType_t nccl_type;
    switch (tensor.dtype()) {
        case DType::Float32: nccl_type = ncclFloat32; break;
        case DType::Float64: nccl_type = ncclFloat64; break;
        case DType::Int32:   nccl_type = ncclInt32; break;
        case DType::Int64:   nccl_type = ncclInt64; break;
        default:
            throw std::runtime_error("ProcessGroup: unsupported dtype for broadcast");
    }

    // Broadcast
    NCCL_CHECK(ncclBroadcast(
        tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        nccl_type,
        src_rank,
        comm,
        nullptr  // Use default stream
    ));

    // Synchronize
    #ifdef TENZOR_USE_ROCM
        GPU_CHECK(hipDeviceSynchronize());
    #else
        GPU_CHECK(cudaDeviceSynchronize());
    #endif
#else
    (void)tensor;
    (void)src_rank;
    (void)device_id;
    throw std::runtime_error("ProcessGroup: NCCL support not compiled");
#endif
}

auto ProcessGroup::all_reduce(Tensor& tensor, ncclRedOp_t op, int device_id) -> void {
#if defined(TENZOR_USE_ROCM) || defined(TENZOR_USE_CUDA)
    auto comm = get_communicator(device_id);

    // Set device
    #ifdef TENZOR_USE_ROCM
        GPU_CHECK(hipSetDevice(device_id));
    #else
        GPU_CHECK(cudaSetDevice(device_id));
    #endif

    // For now, use the new Phase 4 distributed training API instead of direct NCCL
    // This old DDP implementation will be deprecated in favor of the new distributed module
    throw std::runtime_error(
        "ProcessGroup::all_reduce - Please use the new distributed training API "
        "from tenzor/distributed/distributed.hpp instead of the old DDP API"
    );

#if 0  // Old NCCL code - disabled
    // Determine NCCL data type
    ncclDataType_t nccl_type;
    switch (tensor.dtype()) {
        case DType::Float32: nccl_type = ncclFloat; break;
        case DType::Float64: nccl_type = ncclDouble; break;
        case DType::Int32:   nccl_type = ncclInt; break;
        case DType::Int64:   nccl_type = ncclInt64; break;
        default:
            throw std::runtime_error("ProcessGroup: unsupported dtype for all_reduce");
    }

    // All-reduce (in-place)
    NCCL_CHECK(ncclAllReduce(
        tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        nccl_type,
        op,
        comm,
        nullptr  // Use default stream
    ));

    // Synchronize
    #ifdef TENZOR_USE_ROCM
        GPU_CHECK(hipDeviceSynchronize());
    #else
        GPU_CHECK(cudaDeviceSynchronize());
    #endif
#else
    (void)tensor;
    (void)op;
    (void)device_id;
    throw std::runtime_error("ProcessGroup: NCCL support not compiled");
#endif
}

auto ProcessGroup::barrier() -> void {
#if defined(TENZOR_USE_ROCM) || defined(TENZOR_USE_CUDA)
    // Barrier using NCCL all-reduce on a dummy tensor
    if (communicators_.empty()) {
        return;  // No communicators initialized
    }

    // Use first communicator for barrier
    auto& [device_id, comm] = *communicators_.begin();

    #ifdef TENZOR_USE_ROCM
        GPU_CHECK(hipSetDevice(device_id));
    #else
        GPU_CHECK(cudaSetDevice(device_id));
    #endif

    // Create dummy tensor for barrier
    int dummy = 0;
    #ifdef TENZOR_USE_ROCM
        int* d_dummy;
        GPU_CHECK(hipMalloc(&d_dummy, sizeof(int)));
        GPU_CHECK(hipMemcpy(d_dummy, &dummy, sizeof(int), hipMemcpyHostToDevice));
    #else
        int* d_dummy;
        GPU_CHECK(cudaMalloc(&d_dummy, sizeof(int)));
        GPU_CHECK(cudaMemcpy(d_dummy, &dummy, sizeof(int), cudaMemcpyHostToDevice));
    #endif

    // All-reduce for barrier
    NCCL_CHECK(ncclAllReduce(
        d_dummy,
        d_dummy,
        1,
        ncclInt32,
        ncclSum,
        comm,
        nullptr
    ));

    // Synchronize and cleanup
    #ifdef TENZOR_USE_ROCM
        GPU_CHECK(hipDeviceSynchronize());
        GPU_CHECK(hipFree(d_dummy));
    #else
        GPU_CHECK(cudaDeviceSynchronize());
        GPU_CHECK(cudaFree(d_dummy));
    #endif
#else
    // CPU-only: no-op barrier
#endif
}

// ============================================================================
// GradientBucket Implementation
// ============================================================================

GradientBucket::GradientBucket(size_t bucket_size_mb)
    : max_size_bytes_(bucket_size_mb * 1024 * 1024) {
}

auto GradientBucket::add_gradient(Variable* param) -> bool {
    if (!param || !param->has_grad()) {
        return false;
    }

    auto grad_opt = param->grad();
    if (!grad_opt.has_value()) {
        return false;
    }

    auto& grad = grad_opt.value();
    size_t grad_size = grad.numel() * grad.dtype_size();

    params_.push_back(param);
    size_bytes_ += grad_size;

    return is_full();
}

auto GradientBucket::reset() -> void {
    params_.clear();
    size_bytes_ = 0;
}

// ============================================================================
// DistributedDataParallel Implementation
// ============================================================================

DistributedDataParallel::DistributedDataParallel(
    std::shared_ptr<Module> module,
    std::shared_ptr<ProcessGroup> process_group,
    std::vector<int> device_ids,
    int output_device,
    bool broadcast_buffers,
    bool find_unused_parameters,
    bool gradient_as_bucket_view,
    size_t bucket_size_mb
) : module_(module),
    process_group_(process_group),
    device_ids_(device_ids),
    output_device_(output_device),
    broadcast_buffers_(broadcast_buffers),
    find_unused_parameters_(find_unused_parameters),
    gradient_as_bucket_view_(gradient_as_bucket_view),
    bucket_size_mb_(bucket_size_mb) {

    if (!module_) {
        throw std::invalid_argument("DistributedDataParallel: module cannot be null");
    }

    if (!process_group_) {
        throw std::invalid_argument("DistributedDataParallel: process_group cannot be null");
    }

    // Auto-detect devices if not provided
    if (device_ids_.empty()) {
#if defined(TENZOR_USE_ROCM)
        int device_count = 0;
        hipError_t err = hipGetDeviceCount(&device_count);
        if (err != hipSuccess || device_count == 0) {
            throw std::runtime_error("DistributedDataParallel: No HIP devices available");
        }
        for (int i = 0; i < device_count; ++i) {
            device_ids_.push_back(i);
        }
#elif defined(TENZOR_USE_CUDA)
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            throw std::runtime_error("DistributedDataParallel: No CUDA devices available");
        }
        for (int i = 0; i < device_count; ++i) {
            device_ids_.push_back(i);
        }
#else
        throw std::runtime_error("DistributedDataParallel: GPU support not compiled");
#endif
    }

    if (device_ids_.empty()) {
        throw std::invalid_argument("DistributedDataParallel: device_ids cannot be empty");
    }

    // Set output device
    if (output_device_ == -1) {
        output_device_ = device_ids_[0];
    }

    // Validate that output_device is in device_ids
    if (std::find(device_ids_.begin(), device_ids_.end(), output_device_) == device_ids_.end()) {
        throw std::invalid_argument(
            "DistributedDataParallel: output_device must be in device_ids"
        );
    }

    // Validate devices
    validate_devices();

    // Initialize distributed training
    initialize_distributed();
}

auto DistributedDataParallel::forward(const Variable& input) -> Variable {
    // First forward pass: setup complete
    if (first_forward_) {
        first_forward_ = false;
    }

    // Single device optimization: just use module directly
    if (device_ids_.size() == 1 && process_group_->world_size() == 1) {
        return module_->forward(input);
    }

    // Execute forward pass on local module
    auto output = module_->forward(input);

    // Register backward hook for gradient synchronization
    // Note: In a full autograd implementation, we would register a hook here
    // that automatically triggers gradient synchronization when backward() is called.
    //
    // For now, users need to manually call synchronize_gradients() after loss.backward()
    // or integrate this into the backward engine.
    //
    // Example integration:
    // output.register_hook([this]() {
    //     this->synchronize_gradients();
    // });

    return output;
}

auto DistributedDataParallel::parameters() -> std::vector<std::shared_ptr<Variable>> {
    return module_->parameters();
}

auto DistributedDataParallel::named_parameters()
    -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> {
    return module_->named_parameters();
}

auto DistributedDataParallel::train(bool mode) -> void {
    module_->train(mode);
}

auto DistributedDataParallel::eval() -> void {
    train(false);
}

auto DistributedDataParallel::join() -> void {
    // Wait for all processes to reach this point
    process_group_->barrier();
}

auto DistributedDataParallel::initialize_distributed() -> void {
#if defined(TENZOR_USE_ROCM) || defined(TENZOR_USE_CUDA)
    // Initialize NCCL communicators for each device
    ncclUniqueId id;

    // Rank 0 generates unique ID
    if (process_group_->rank() == 0) {
        NCCL_CHECK(ncclGetUniqueId(&id));
    }

    // Broadcast unique ID to all processes using TCP sockets
    // This implements proper inter-process communication for multi-node setups

    if (process_group_->rank() == 0) {
        // Master rank: Create TCP server to broadcast unique ID
        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) {
            throw std::runtime_error("DistributedDataParallel: Failed to create broadcast socket");
        }

        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        const char* master_port_env = std::getenv("MASTER_PORT");
        int broadcast_port = master_port_env ? std::atoi(master_port_env) : 29501;

        struct sockaddr_in addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(broadcast_port);

        if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            ::close(server_fd);
            throw std::runtime_error("DistributedDataParallel: Failed to bind broadcast socket");
        }

        if (listen(server_fd, process_group_->world_size()) < 0) {
            ::close(server_fd);
            throw std::runtime_error("DistributedDataParallel: Failed to listen on broadcast socket");
        }

        // Accept connections and send unique ID to all workers
        for (int i = 1; i < process_group_->world_size(); ++i) {
            int client_fd = accept(server_fd, nullptr, nullptr);
            if (client_fd < 0) {
                ::close(server_fd);
                throw std::runtime_error("DistributedDataParallel: Failed to accept connection from rank " + std::to_string(i));
            }

            ssize_t sent = send(client_fd, &id, sizeof(id), 0);
            if (sent != sizeof(id)) {
                ::close(client_fd);
                ::close(server_fd);
                throw std::runtime_error("DistributedDataParallel: Failed to send unique ID to rank " + std::to_string(i));
            }

            ::close(client_fd);
        }

        ::close(server_fd);

    } else {
        // Worker ranks: Connect to master and receive unique ID
        const char* master_addr_env = std::getenv("MASTER_ADDR");
        const char* master_port_env = std::getenv("MASTER_PORT");

        std::string master_addr = master_addr_env ? std::string(master_addr_env) : "localhost";
        int broadcast_port = master_port_env ? std::atoi(master_port_env) : 29501;

        int client_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (client_fd < 0) {
            throw std::runtime_error("DistributedDataParallel: Failed to create client socket");
        }

        struct sockaddr_in addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(broadcast_port);

        struct hostent* server = gethostbyname(master_addr.c_str());
        if (!server) {
            ::close(client_fd);
            throw std::runtime_error("DistributedDataParallel: Failed to resolve master address: " + master_addr);
        }

        std::memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);

        // Retry connection with exponential backoff
        bool connected = false;
        for (int retry = 0; retry < 10; ++retry) {
            if (connect(client_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                connected = true;
                break;
            }
            usleep(100000 * (1 << retry));  // 100ms, 200ms, 400ms, ...
        }

        if (!connected) {
            ::close(client_fd);
            throw std::runtime_error("DistributedDataParallel: Failed to connect to master after retries");
        }

        ssize_t received = recv(client_fd, &id, sizeof(id), MSG_WAITALL);
        if (received != sizeof(id)) {
            ::close(client_fd);
            throw std::runtime_error("DistributedDataParallel: Failed to receive unique ID from master");
        }

        ::close(client_fd);
    }

    // Initialize communicator for each local device
    for (int device_id : device_ids_) {
        process_group_->init_communicator(device_id, id);
    }

    // Broadcast parameters from rank 0
    broadcast_parameters();

    // Broadcast buffers if requested
    if (broadcast_buffers_) {
        broadcast_buffers();
    }

    // Create gradient buckets
    create_gradient_buckets();

    // Register backward hooks
    register_backward_hooks();
#else
    throw std::runtime_error(
        "DistributedDataParallel: GPU support not compiled. "
        "Build with -DTENZOR_BUILD_CUDA=ON or -DTENZOR_BUILD_ROCM=ON"
    );
#endif
}

auto DistributedDataParallel::broadcast_parameters() -> void {
    auto params = module_->parameters();

    for (const auto& param : params) {
        if (!param) continue;

        auto& tensor = param->tensor();

        // Broadcast from rank 0 to all processes
        process_group_->broadcast(tensor, 0, output_device_);
    }
}

auto DistributedDataParallel::broadcast_buffers() -> void {
    auto buffers = module_->buffers();

    for (const auto& buffer : buffers) {
        if (!buffer) continue;

        auto& tensor = buffer->tensor();

        // Broadcast from rank 0 to all processes
        process_group_->broadcast(tensor, 0, output_device_);
    }
}

auto DistributedDataParallel::create_gradient_buckets() -> void {
    // Get all parameters
    auto params = module_->parameters();

    // Reverse order (parameters used later in backward pass first)
    std::reverse(params.begin(), params.end());

    // Create buckets
    GradientBucket current_bucket(bucket_size_mb_);

    for (const auto& param : params) {
        if (!param || !param->requires_grad()) {
            continue;
        }

        // Add to current bucket
        bool is_full = current_bucket.add_gradient(param.get());

        // Track parameter for synchronization
        parameters_to_sync_.push_back(param);

        // If bucket is full, save it and start a new one
        if (is_full) {
            buckets_.push_back(std::move(current_bucket));
            current_bucket = GradientBucket(bucket_size_mb_);
        }
    }

    // Add remaining parameters
    if (!current_bucket.is_empty()) {
        buckets_.push_back(std::move(current_bucket));
    }
}

auto DistributedDataParallel::register_backward_hooks() -> void {
    // In a full implementation, we would register autograd hooks here
    // that trigger bucket synchronization when gradients become ready.
    //
    // Example:
    // for (auto* param : parameters_to_sync_) {
    //     param->register_hook([this, param]() {
    //         this->gradient_ready_callback(param);
    //     });
    // }
    //
    // For now, synchronize_gradients() must be called manually after backward()
}

auto DistributedDataParallel::synchronize_bucket(GradientBucket& bucket) -> void {
#if defined(TENZOR_USE_ROCM) || defined(TENZOR_USE_CUDA)
    if (bucket.is_empty()) {
        return;
    }

    auto& params = bucket.parameters();

    // Synchronize each gradient in the bucket
    for (auto* param : params) {
        if (!param || !param->has_grad()) {
            continue;
        }

        auto grad_opt = param->grad();
        if (!grad_opt.has_value()) {
            continue;
        }

        auto grad = grad_opt.value();

        // All-reduce gradient across all processes
        process_group_->all_reduce(grad, ncclSum, output_device_);

        // Average by dividing by world size
        float scale = 1.0f / static_cast<float>(process_group_->world_size());
        grad = grad * scale;

        // Update parameter gradient
        param->grad() = grad;
    }
#else
    (void)bucket;
#endif
}

auto DistributedDataParallel::synchronize_gradients() -> void {
    // Single process optimization - no synchronization needed
    if (process_group_->world_size() == 1) {
        return;
    }

    std::lock_guard<std::mutex> lock(sync_mutex_);

    // Mark backward in progress
    backward_in_progress_ = true;

    // Synchronize each bucket
    for (auto& bucket : buckets_) {
        synchronize_bucket(bucket);
    }

    // Mark backward complete
    backward_in_progress_ = false;
    num_ready_gradients_ = 0;
}

auto DistributedDataParallel::validate_devices() -> void {
#if defined(TENZOR_USE_ROCM)
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);

    if (err != hipSuccess) {
        throw std::runtime_error(
            "DistributedDataParallel: HIP error - " + std::string(hipGetErrorString(err))
        );
    }

    for (int device_id : device_ids_) {
        if (device_id < 0 || device_id >= device_count) {
            throw std::invalid_argument(
                "DistributedDataParallel: invalid device_id " + std::to_string(device_id) +
                " (available: 0-" + std::to_string(device_count - 1) + ")"
            );
        }
    }
#elif defined(TENZOR_USE_CUDA)
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess) {
        throw std::runtime_error(
            "DistributedDataParallel: CUDA error - " + std::string(cudaGetErrorString(err))
        );
    }

    for (int device_id : device_ids_) {
        if (device_id < 0 || device_id >= device_count) {
            throw std::invalid_argument(
                "DistributedDataParallel: invalid device_id " + std::to_string(device_id) +
                " (available: 0-" + std::to_string(device_count - 1) + ")"
            );
        }
    }
#else
    throw std::runtime_error("DistributedDataParallel: GPU support not enabled");
#endif
}

auto DistributedDataParallel::get_nccl_datatype(DType dtype) -> ncclDataType_t {
#if defined(TENZOR_USE_CUDA)
    switch (dtype) {
        case DType::Float32: return ncclFloat;
        case DType::Float64: return ncclDouble;
        case DType::Int32:   return ncclInt;
        case DType::Int64:   return ncclInt64;
        default:
            throw std::runtime_error(
                "DistributedDataParallel: unsupported dtype for NCCL"
            );
    }
#else
    throw std::runtime_error("NCCL not available - CUDA support not enabled");
#endif
}

// ============================================================================
// Helper Functions
// ============================================================================

auto make_distributed_data_parallel(
    std::shared_ptr<Module> module,
    std::shared_ptr<ProcessGroup> process_group,
    std::vector<int> device_ids,
    int output_device
) -> std::shared_ptr<DistributedDataParallel> {
    return std::make_shared<DistributedDataParallel>(
        module,
        process_group,
        device_ids,
        output_device
    );
}

auto init_process_group(const std::string& backend) -> std::shared_ptr<ProcessGroup> {
    // Read environment variables
    const char* rank_env = std::getenv("RANK");
    const char* world_size_env = std::getenv("WORLD_SIZE");

    if (!rank_env || !world_size_env) {
        throw std::runtime_error(
            "init_process_group: RANK and WORLD_SIZE environment variables must be set"
        );
    }

    int rank = std::atoi(rank_env);
    int world_size = std::atoi(world_size_env);

    // Optional: read MASTER_ADDR and MASTER_PORT for multi-node
    const char* master_addr = std::getenv("MASTER_ADDR");
    const char* master_port = std::getenv("MASTER_PORT");

    if (master_addr) {
        // Multi-node setup with TCP-based initialization
        std::string addr(master_addr);
        std::string port_str = master_port ? std::string(master_port) : "29500";
        int port = std::atoi(port_str.c_str());

        // TCP-based initialization for multi-node distributed training
        // This implements a rendezvous mechanism for NCCL unique ID exchange

        // The actual NCCL unique ID exchange happens inside initialize_distributed()
        // which is called by the DistributedDataParallel constructor.
        //
        // Multi-node setup requirements:
        // 1. MASTER_ADDR: IP/hostname of rank 0 node
        // 2. MASTER_PORT: Port for coordination (default: 29500)
        // 3. RANK: Process rank (0 to WORLD_SIZE-1)
        // 4. WORLD_SIZE: Total number of processes
        //
        // Network requirements:
        // - All nodes must be able to reach MASTER_ADDR:MASTER_PORT
        // - Firewall rules must allow TCP connections
        // - For GPU Direct RDMA: Configure NCCL_SOCKET_IFNAME to network interface
        //
        // Example environment setup:
        // Rank 0 (master node):
        //   export MASTER_ADDR=192.168.1.100
        //   export MASTER_PORT=29500
        //   export RANK=0
        //   export WORLD_SIZE=4
        //
        // Rank 1-3 (worker nodes):
        //   export MASTER_ADDR=192.168.1.100
        //   export MASTER_PORT=29500
        //   export RANK=<rank_id>
        //   export WORLD_SIZE=4
    }

    return std::make_shared<ProcessGroup>(rank, world_size, backend);
}

auto destroy_process_group(std::shared_ptr<ProcessGroup> process_group) -> void {
    // Destructor handles cleanup
    process_group.reset();
}

} // namespace nn
} // namespace tenzor
