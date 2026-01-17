#include "rocm_backend.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/rocm_caching_allocator.hip.hpp"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <limits>
#include <cstdlib>
#include <cstdint>
#include <sstream>

namespace tenzor {

// ROCmBackend Implementation

ROCmBackend::ROCmBackend() {
    // Check if caching allocator is enabled via environment variable
    const char* enable_caching = std::getenv("TENZOR_ENABLE_CACHING_ALLOCATOR");
    use_caching_allocator_ = (enable_caching != nullptr && std::string(enable_caching) == "1");

    // Initialize HIP runtime by querying device count
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess && err != hipErrorNoDevice) {
        throw std::runtime_error(
            std::string("Failed to initialize ROCm backend: ") + hipGetErrorString(err)
        );
    }
}

auto ROCmBackend::name() const -> std::string_view {
    return "rocm";
}

auto ROCmBackend::device_count() const -> int32_t {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess) {
        // Return 0 if ROCm is not available or has no devices
        return 0;
    }
    return count;
}

auto ROCmBackend::is_available() const -> bool {
    return device_count() > 0;
}

auto ROCmBackend::get_device_info(int32_t device_id) const -> DeviceInfo {
    int count = device_count();
    if (device_id < 0 || device_id >= count) {
        throw std::out_of_range("Invalid ROCm device ID: " + std::to_string(device_id) +
                                " (available: 0-" + std::to_string(count - 1) + ")");
    }

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device_id);

    DeviceInfo info;
    info.name = props.name;
    info.vendor = "AMD";

    // Get driver version
    int driver_version = 0;
    hipDriverGetVersion(&driver_version);
    info.driver_version = std::to_string(driver_version / 100) + "." +
                          std::to_string(driver_version % 100);

    // Memory info
    info.total_memory = props.totalGlobalMem;
    size_t free_mem = 0, total_mem = 0;
    int current_device;
    hipGetDevice(&current_device);
    hipSetDevice(device_id);
    hipMemGetInfo(&free_mem, &total_mem);
    hipSetDevice(current_device);
    info.available_memory = free_mem;

    // Compute info
    info.compute_units = props.multiProcessorCount;
    info.max_threads_per_block = props.maxThreadsPerBlock;
    info.max_shared_memory = static_cast<int>(props.sharedMemPerBlock);
    info.warp_size = props.warpSize;  // 64 for AMD

    // GCN/RDNA version from architecture name
    info.major_version = props.major;
    info.minor_version = props.minor;

    // Feature support - AMD GPUs generally support these
    info.supports_fp16 = true;   // GCN 3rd gen+
    info.supports_fp64 = true;   // All GCN/RDNA
    info.supports_int8 = true;   // RDNA2+

    // Device type
    info.is_integrated = (props.integrated != 0);
    info.is_discrete = !info.is_integrated;

    // PCI info
    info.pci_bus_id = props.pciBusID;
    info.pci_device_id = props.pciDeviceID;

    return info;
}

auto ROCmBackend::allocate(size_t bytes, int32_t device_id) -> void* {
    // Handle empty tensors - HIP doesn't like 0-byte allocations
    if (bytes == 0) {
        return nullptr;
    }

    if (use_caching_allocator_) {
        return backend::rocm::RocmCachingAllocator::get().allocate(bytes, device_id);
    }

    void* ptr = nullptr;
    check_hip_error(hipSetDevice(device_id), "hipSetDevice in allocate");
    hipError_t err = hipMalloc(&ptr, bytes);
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("Failed to allocate device memory: ") + hipGetErrorString(err)
        );
    }
    return ptr;
}

auto ROCmBackend::deallocate(void* ptr) -> void {
    // Handle nullptr from empty tensor allocations
    if (ptr == nullptr) {
        return;
    }

    if (use_caching_allocator_) {
        // Note: we don't know the device_id here, but CachingAllocator tracks it
        // For proper integration, we'd need to look up the device from the pointer
        int device_id = 0;
        hipPointerAttribute_t attrs;
        if (hipPointerGetAttributes(&attrs, ptr) == hipSuccess) {
            device_id = attrs.device;
        }
        backend::rocm::RocmCachingAllocator::get().free(ptr, device_id);
        return;
    }

    check_hip_error(hipFree(ptr), "hipFree");
}

auto ROCmBackend::copy(void* dst, const void* src, size_t bytes, CopyKind kind) -> void {
    // Handle empty tensors
    if (bytes == 0) {
        return;
    }

    hipMemcpyKind hip_kind;
    switch (kind) {
        case CopyKind::HostToHost:
            hip_kind = hipMemcpyHostToHost;
            break;
        case CopyKind::HostToDevice:
            hip_kind = hipMemcpyHostToDevice;
            break;
        case CopyKind::DeviceToHost:
            hip_kind = hipMemcpyDeviceToHost;
            break;
        case CopyKind::DeviceToDevice:
            hip_kind = hipMemcpyDeviceToDevice;
            break;
    }

    hipError_t err = hipMemcpy(dst, src, bytes, hip_kind);
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("HIP copy failed: ") + hipGetErrorString(err)
        );
    }
}

auto ROCmBackend::synchronize(int32_t device_id) -> void {
    check_hip_error(hipSetDevice(device_id), "hipSetDevice in synchronize");
    check_hip_error(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

auto ROCmBackend::create_stream(int32_t device_id) -> StreamHandle {
    hipStream_t stream;
    check_hip_error(hipSetDevice(device_id), "hipSetDevice in create_stream");
    check_hip_error(hipStreamCreate(&stream), "hipStreamCreate");
    return static_cast<StreamHandle>(stream);
}

auto ROCmBackend::destroy_stream(StreamHandle stream) -> void {
    check_hip_error(hipStreamDestroy(static_cast<hipStream_t>(stream)), "hipStreamDestroy");
}

auto ROCmBackend::synchronize_stream(StreamHandle stream) -> void {
    check_hip_error(hipStreamSynchronize(static_cast<hipStream_t>(stream)), "hipStreamSynchronize");
}

auto ROCmBackend::dispatch(const std::string& op_name,
                           std::span<const Tensor> inputs,
                           const OpAttributes& attrs) -> std::vector<Tensor> {
    // Allow empty inputs for creation operations
    bool is_creation_op = (op_name == "zeros" || op_name == "ones" || op_name == "full" ||
                           op_name == "rand" || op_name == "randn");

    // Validate we have inputs (except for creation operations)
    if (inputs.empty() && !is_creation_op) {
        throw std::invalid_argument("dispatch requires at least one input tensor");
    }

    // Validate all inputs are on ROCm device (if any)
    for (const auto& tensor : inputs) {
        if (tensor.device().type != Device::Type::ROCm) {
            throw std::runtime_error(
                "ROCmBackend: All input tensors must be on ROCm device, got: " +
                tensor.device().to_string()
            );
        }
    }

    // Set HIP device - use first tensor's device or device from attrs
    int32_t device_id = 0;
    if (!inputs.empty()) {
        device_id = inputs[0].device().index;
    } else if (attrs.contains("device_id")) {
        device_id = std::stoi(attrs.at("device_id"));
    }
    check_hip_error(hipSetDevice(device_id), "hipSetDevice in dispatch");

    // Get or create stream (nullptr means default stream)
    hipStream_t stream = nullptr;
    if (attrs.contains("stream")) {
        stream = static_cast<hipStream_t>(
            reinterpret_cast<void*>(std::stoull(attrs.at("stream")))
        );
    }

    // Dispatch to appropriate HIP kernel based on operation name
    try {
        if (op_name == "add") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("add operation requires exactly 2 inputs");
            }
            return {rocm::add_kernel(inputs[0], inputs[1], stream)};
        }
        else if (op_name == "sub") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("sub operation requires exactly 2 inputs");
            }
            return {rocm::sub_kernel(inputs[0], inputs[1], stream)};
        }
        else if (op_name == "mul") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("mul operation requires exactly 2 inputs");
            }
            return {rocm::mul_kernel(inputs[0], inputs[1], stream)};
        }
        else if (op_name == "div") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("div operation requires exactly 2 inputs");
            }
            return {rocm::div_kernel(inputs[0], inputs[1], stream)};
        }
        else if (op_name == "matmul") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("matmul operation requires exactly 2 inputs");
            }
            return {rocm::matmul_kernel(inputs[0], inputs[1], stream)};
        }
        else if (op_name == "sum") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("sum operation requires exactly 1 input");
            }
            // Parse attributes - use INT64_MIN as sentinel for full reduction (when no dim specified)
            int64_t dim = INT64_MIN;
            bool keepdim = false;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            if (attrs.contains("keepdim")) {
                keepdim = (attrs.at("keepdim") == "1");
            }
            return {rocm::sum_kernel(inputs[0], dim, keepdim, stream)};
        }
        else if (op_name == "mean") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("mean operation requires exactly 1 input");
            }
            // Use INT64_MIN as sentinel for full reduction (when no dim specified)
            int64_t dim = INT64_MIN;
            bool keepdim = false;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            if (attrs.contains("keepdim")) {
                keepdim = (attrs.at("keepdim") == "1");
            }
            return {rocm::mean_kernel(inputs[0], dim, keepdim, stream)};
        }
        else if (op_name == "max") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("max operation requires exactly 1 input");
            }
            // Use INT64_MIN as sentinel for full reduction (when no dim specified)
            int64_t dim = INT64_MIN;
            bool keepdim = false;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            if (attrs.contains("keepdim")) {
                keepdim = (attrs.at("keepdim") == "1");
            }
            return {rocm::max_kernel(inputs[0], dim, keepdim, stream)};
        }
        else if (op_name == "min") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("min operation requires exactly 1 input");
            }
            // Use INT64_MIN as sentinel for full reduction (when no dim specified)
            int64_t dim = INT64_MIN;
            bool keepdim = false;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            if (attrs.contains("keepdim")) {
                keepdim = (attrs.at("keepdim") == "1");
            }
            return {rocm::min_kernel(inputs[0], dim, keepdim, stream)};
        }
        else if (op_name == "sqrt") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("sqrt operation requires exactly 1 input");
            }
            return {rocm::sqrt_kernel(inputs[0], stream)};
        }
        else if (op_name == "neg") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("neg operation requires exactly 1 input");
            }
            return {rocm::neg_kernel(inputs[0], stream)};
        }
        else if (op_name == "abs") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("abs operation requires exactly 1 input");
            }
            return {rocm::abs_kernel(inputs[0], stream)};
        }
        else if (op_name == "sign") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("sign operation requires exactly 1 input");
            }
            return {rocm::sign_kernel(inputs[0], stream)};
        }
        else if (op_name == "clamp") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("clamp operation requires exactly 1 input");
            }
            // Parse min and max from attributes
            float min_val = -std::numeric_limits<float>::infinity();
            float max_val = std::numeric_limits<float>::infinity();
            if (attrs.contains("min")) {
                min_val = std::stof(attrs.at("min"));
            }
            if (attrs.contains("max")) {
                max_val = std::stof(attrs.at("max"));
            }
            return {rocm::clamp_kernel(inputs[0], min_val, max_val, stream)};
        }
        else if (op_name == "log") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("log operation requires exactly 1 input");
            }
            return {rocm::log_kernel(inputs[0], stream)};
        }
        else if (op_name == "exp") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("exp operation requires exactly 1 input");
            }
            return {rocm::exp_kernel(inputs[0], stream)};
        }
        else if (op_name == "pow") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("pow operation requires exactly 1 input");
            }
            float exponent = 2.0f;
            if (attrs.contains("exponent")) {
                exponent = std::stof(attrs.at("exponent"));
            }
            return {rocm::pow_kernel(inputs[0], exponent, stream)};
        }
        else if (op_name == "relu") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("relu operation requires exactly 1 input");
            }
            return {rocm::relu_kernel(inputs[0], stream)};
        }
        else if (op_name == "relu_backward") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("relu_backward operation requires exactly 2 inputs");
            }
            return {rocm::relu_backward_kernel(inputs[0], inputs[1], stream)};
        }
        else if (op_name == "sigmoid") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("sigmoid operation requires exactly 1 input");
            }
            return {rocm::sigmoid_kernel(inputs[0], stream)};
        }
        else if (op_name == "sigmoid_backward") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("sigmoid_backward operation requires exactly 2 inputs");
            }
            return {rocm::sigmoid_backward_kernel(inputs[0], inputs[1], stream)};
        }
        else if (op_name == "tanh") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("tanh operation requires exactly 1 input");
            }
            return {rocm::tanh_kernel(inputs[0], stream)};
        }
        else if (op_name == "tanh_backward") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("tanh_backward operation requires exactly 2 inputs");
            }
            return {rocm::tanh_backward_kernel(inputs[0], inputs[1], stream)};
        }
        else if (op_name == "leaky_relu") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("leaky_relu operation requires exactly 1 input");
            }
            float alpha = 0.01f;
            if (attrs.contains("alpha")) {
                alpha = std::stof(attrs.at("alpha"));
            }
            return {rocm::leaky_relu_kernel(inputs[0], alpha, stream)};
        }
        else if (op_name == "leaky_relu_backward") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("leaky_relu_backward operation requires exactly 2 inputs");
            }
            float alpha = 0.01f;
            if (attrs.contains("alpha")) {
                alpha = std::stof(attrs.at("alpha"));
            }
            return {rocm::leaky_relu_backward_kernel(inputs[0], inputs[1], alpha, stream)};
        }
        else if (op_name == "softmax") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("softmax operation requires exactly 1 input");
            }
            int64_t dim = -1;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            return {rocm::softmax_kernel(inputs[0], dim, stream)};
        }
        else if (op_name == "softmax_backward") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("softmax_backward operation requires exactly 2 inputs");
            }
            int64_t dim = -1;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            return {rocm::softmax_backward_kernel(inputs[0], inputs[1], dim, stream)};
        }
        else if (op_name == "log_softmax") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("log_softmax operation requires exactly 1 input");
            }
            int64_t dim = -1;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            return {rocm::log_softmax_kernel(inputs[0], dim, stream)};
        }
        else if (op_name == "log_softmax_backward") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("log_softmax_backward operation requires exactly 2 inputs");
            }
            int64_t dim = -1;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            return {rocm::log_softmax_backward_kernel(inputs[0], inputs[1], dim, stream)};
        }
        else if (op_name == "zeros") {
            // Parse shape from attributes
            if (!attrs.contains("shape")) {
                throw std::invalid_argument("zeros operation requires 'shape' attribute");
            }
            // Shape is stored as comma-separated values
            std::vector<int64_t> shape;
            std::string shape_str = attrs.at("shape");
            size_t pos = 0;
            while ((pos = shape_str.find(',')) != std::string::npos) {
                shape.push_back(std::stoll(shape_str.substr(0, pos)));
                shape_str.erase(0, pos + 1);
            }
            if (!shape_str.empty()) {
                shape.push_back(std::stoll(shape_str));
            }

            DType dtype = DType::Float32;
            if (attrs.contains("dtype")) {
                // Parse dtype from string
                auto dtype_str = attrs.at("dtype");
                if (dtype_str == "float32") dtype = DType::Float32;
                else if (dtype_str == "float64") dtype = DType::Float64;
                else if (dtype_str == "int32") dtype = DType::Int32;
                else if (dtype_str == "int64") dtype = DType::Int64;
                else if (dtype_str == "bool") dtype = DType::Bool;
                else if (dtype_str == "uint8") dtype = DType::UInt8;
                else if (dtype_str == "int8") dtype = DType::Int8;
                else if (dtype_str == "float16") dtype = DType::Float16;
            }

            Device device = inputs.empty() ? Device::rocm(device_id) : inputs[0].device();
            return {rocm::zeros_kernel(shape, dtype, device, stream)};
        }
        else if (op_name == "ones") {
            if (!attrs.contains("shape")) {
                throw std::invalid_argument("ones operation requires 'shape' attribute");
            }
            std::vector<int64_t> shape;
            std::string shape_str = attrs.at("shape");
            size_t pos = 0;
            while ((pos = shape_str.find(',')) != std::string::npos) {
                shape.push_back(std::stoll(shape_str.substr(0, pos)));
                shape_str.erase(0, pos + 1);
            }
            if (!shape_str.empty()) {
                shape.push_back(std::stoll(shape_str));
            }

            DType dtype = DType::Float32;
            if (attrs.contains("dtype")) {
                auto dtype_str = attrs.at("dtype");
                if (dtype_str == "float32") dtype = DType::Float32;
                else if (dtype_str == "float64") dtype = DType::Float64;
                else if (dtype_str == "int32") dtype = DType::Int32;
                else if (dtype_str == "int64") dtype = DType::Int64;
                else if (dtype_str == "bool") dtype = DType::Bool;
                else if (dtype_str == "uint8") dtype = DType::UInt8;
                else if (dtype_str == "int8") dtype = DType::Int8;
                else if (dtype_str == "float16") dtype = DType::Float16;
            }

            Device device = inputs.empty() ? Device::rocm(device_id) : inputs[0].device();
            return {rocm::ones_kernel(shape, dtype, device, stream)};
        }
        else if (op_name == "full") {
            if (!attrs.contains("shape")) {
                throw std::invalid_argument("full operation requires 'shape' attribute");
            }
            if (!attrs.contains("value")) {
                throw std::invalid_argument("full operation requires 'value' attribute");
            }

            std::vector<int64_t> shape;
            std::string shape_str = attrs.at("shape");
            size_t pos = 0;
            while ((pos = shape_str.find(',')) != std::string::npos) {
                shape.push_back(std::stoll(shape_str.substr(0, pos)));
                shape_str.erase(0, pos + 1);
            }
            if (!shape_str.empty()) {
                shape.push_back(std::stoll(shape_str));
            }

            float value = std::stof(attrs.at("value"));

            DType dtype = DType::Float32;
            if (attrs.contains("dtype")) {
                auto dtype_str = attrs.at("dtype");
                if (dtype_str == "float32") dtype = DType::Float32;
                else if (dtype_str == "float64") dtype = DType::Float64;
                else if (dtype_str == "int32") dtype = DType::Int32;
                else if (dtype_str == "int64") dtype = DType::Int64;
                else if (dtype_str == "bool") dtype = DType::Bool;
                else if (dtype_str == "uint8") dtype = DType::UInt8;
                else if (dtype_str == "int8") dtype = DType::Int8;
                else if (dtype_str == "float16") dtype = DType::Float16;
            }

            Device device = inputs.empty() ? Device::rocm(device_id) : inputs[0].device();
            return {rocm::full_kernel(shape, value, dtype, device, stream)};
        }
        else if (op_name == "fill") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("fill operation requires exactly 1 input");
            }
            if (!attrs.contains("value")) {
                throw std::invalid_argument("fill operation requires 'value' attribute");
            }
            float value = std::stof(attrs.at("value"));
            return {rocm::fill_kernel(inputs[0], value, stream)};
        }
        else if (op_name == "expand") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("expand operation requires exactly 1 input");
            }
            if (!attrs.contains("shape")) {
                throw std::invalid_argument("expand operation requires 'shape' attribute");
            }

            std::vector<int64_t> shape;
            std::string shape_str = attrs.at("shape");
            size_t pos = 0;
            while ((pos = shape_str.find(',')) != std::string::npos) {
                shape.push_back(std::stoll(shape_str.substr(0, pos)));
                shape_str.erase(0, pos + 1);
            }
            if (!shape_str.empty()) {
                shape.push_back(std::stoll(shape_str));
            }

            return {rocm::expand_kernel(inputs[0], shape, static_cast<void*>(stream))};
        }
        else if (op_name == "rand") {
            if (!attrs.contains("shape")) {
                throw std::invalid_argument("rand operation requires 'shape' attribute");
            }

            std::vector<int64_t> shape;
            std::string shape_str = attrs.at("shape");
            size_t pos = 0;
            while ((pos = shape_str.find(',')) != std::string::npos) {
                shape.push_back(std::stoll(shape_str.substr(0, pos)));
                shape_str.erase(0, pos + 1);
            }
            if (!shape_str.empty()) {
                shape.push_back(std::stoll(shape_str));
            }

            DType dtype = DType::Float32;
            if (attrs.contains("dtype")) {
                auto dtype_str = attrs.at("dtype");
                if (dtype_str == "float32") dtype = DType::Float32;
                else if (dtype_str == "float64") dtype = DType::Float64;
                else if (dtype_str == "int32") dtype = DType::Int32;
                else if (dtype_str == "int64") dtype = DType::Int64;
                else if (dtype_str == "float16") dtype = DType::Float16;
            }

            Device device = inputs.empty() ? Device::rocm(device_id) : inputs[0].device();
            return {rocm::rand_kernel(shape, dtype, device, stream)};
        }
        else if (op_name == "randn") {
            if (!attrs.contains("shape")) {
                throw std::invalid_argument("randn operation requires 'shape' attribute");
            }

            std::vector<int64_t> shape;
            std::string shape_str = attrs.at("shape");
            size_t pos = 0;
            while ((pos = shape_str.find(',')) != std::string::npos) {
                shape.push_back(std::stoll(shape_str.substr(0, pos)));
                shape_str.erase(0, pos + 1);
            }
            if (!shape_str.empty()) {
                shape.push_back(std::stoll(shape_str));
            }

            DType dtype = DType::Float32;
            if (attrs.contains("dtype")) {
                auto dtype_str = attrs.at("dtype");
                if (dtype_str == "float32") dtype = DType::Float32;
                else if (dtype_str == "float64") dtype = DType::Float64;
                else if (dtype_str == "int32") dtype = DType::Int32;
                else if (dtype_str == "int64") dtype = DType::Int64;
                else if (dtype_str == "float16") dtype = DType::Float16;
            }

            Device device = inputs.empty() ? Device::rocm(device_id) : inputs[0].device();
            return {rocm::randn_kernel(shape, dtype, device, stream)};
        }
        else if (op_name == "contiguous") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("contiguous operation requires exactly 1 input");
            }
            return {rocm::contiguous_kernel(inputs[0], stream)};
        }
        else if (op_name == "clone") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("clone operation requires exactly 1 input");
            }
            return {rocm::clone_kernel(inputs[0], stream)};
        }
        else if (op_name == "reshape") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("reshape operation requires exactly 1 input");
            }
            // Parse shape from comma-separated string
            std::vector<int64_t> shape;
            if (attrs.contains("shape")) {
                std::string shape_str = attrs.at("shape");
                size_t pos = 0;
                while (pos < shape_str.size()) {
                    size_t comma = shape_str.find(',', pos);
                    if (comma == std::string::npos) {
                        shape.push_back(std::stoll(shape_str.substr(pos)));
                        break;
                    }
                    shape.push_back(std::stoll(shape_str.substr(pos, comma - pos)));
                    pos = comma + 1;
                }
            }
            return {rocm::reshape_kernel(inputs[0], shape, stream)};
        }
        else if (op_name == "transpose") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("transpose operation requires exactly 1 input");
            }
            int64_t dim0 = 0;
            int64_t dim1 = 1;
            if (attrs.contains("dim0")) {
                dim0 = std::stoll(attrs.at("dim0"));
            }
            if (attrs.contains("dim1")) {
                dim1 = std::stoll(attrs.at("dim1"));
            }
            return {rocm::transpose_kernel(inputs[0], dim0, dim1, stream)};
        }
        else if (op_name == "permute") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("permute operation requires exactly 1 input");
            }
            // Parse dims from comma-separated string
            std::vector<int64_t> dims;
            if (attrs.contains("dims")) {
                std::string dims_str = attrs.at("dims");
                size_t pos = 0;
                while (pos < dims_str.size()) {
                    size_t comma = dims_str.find(',', pos);
                    if (comma == std::string::npos) {
                        dims.push_back(std::stoll(dims_str.substr(pos)));
                        break;
                    }
                    dims.push_back(std::stoll(dims_str.substr(pos, comma - pos)));
                    pos = comma + 1;
                }
            }
            return {rocm::permute_kernel(inputs[0], dims, stream)};
        }
        else if (op_name == "squeeze") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("squeeze operation requires exactly 1 input");
            }
            int64_t dim = -1;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            return {rocm::squeeze_kernel(inputs[0], dim, stream)};
        }
        else if (op_name == "unsqueeze") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("unsqueeze operation requires exactly 1 input");
            }
            int64_t dim = 0;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            return {rocm::unsqueeze_kernel(inputs[0], dim, stream)};
        }
        else if (op_name == "batchnorm2d_mean_var") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("batchnorm2d_mean_var operation requires exactly 1 input");
            }
            // mean and variance tensors are passed in attrs as references
            // For now, we'll return them as outputs
            auto shape = inputs[0].shape();
            int64_t C = shape[1];
            Tensor mean({C}, inputs[0].dtype(), inputs[0].device());
            Tensor variance({C}, inputs[0].dtype(), inputs[0].device());
            rocm::batchnorm2d_mean_var(inputs[0], mean, variance, stream);
            return {mean, variance};
        }
        else if (op_name == "batchnorm2d_forward") {
            if (inputs.size() != 3) {
                throw std::invalid_argument("batchnorm2d_forward operation requires exactly 3 inputs (input, mean, variance)");
            }
            float epsilon = 1e-5f;
            if (attrs.contains("epsilon")) {
                epsilon = std::stof(attrs.at("epsilon"));
            }
            return {rocm::batchnorm2d_forward(inputs[0], inputs[1], inputs[2], epsilon, stream)};
        }
        else if (op_name == "batchnorm2d_forward_affine") {
            if (inputs.size() != 5) {
                throw std::invalid_argument("batchnorm2d_forward_affine operation requires exactly 5 inputs (input, mean, variance, gamma, beta)");
            }
            float epsilon = 1e-5f;
            if (attrs.contains("epsilon")) {
                epsilon = std::stof(attrs.at("epsilon"));
            }
            return {rocm::batchnorm2d_forward_affine(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, stream)};
        }
        else if (op_name == "batchnorm2d_update_running_stats") {
            if (inputs.size() != 4) {
                throw std::invalid_argument("batchnorm2d_update_running_stats operation requires exactly 4 inputs (running_mean, running_var, batch_mean, batch_var)");
            }
            float momentum = 0.1f;
            if (attrs.contains("momentum")) {
                momentum = std::stof(attrs.at("momentum"));
            }
            // Note: running_mean and running_var are modified in-place
            // This is a special case - we need to get mutable references
            // For now, we'll copy back the results
            Tensor running_mean = inputs[0];
            Tensor running_var = inputs[1];
            rocm::batchnorm2d_update_running_stats(running_mean, running_var, inputs[2], inputs[3], momentum, stream);
            return {running_mean, running_var};
        }
        else if (op_name == "batchnorm2d_backward") {
            if (inputs.size() != 5) {
                throw std::invalid_argument("batchnorm2d_backward operation requires exactly 5 inputs (grad_output, input, mean, variance, gamma)");
            }
            float epsilon = 1e-5f;
            if (attrs.contains("epsilon")) {
                epsilon = std::stof(attrs.at("epsilon"));
            }
            auto [grad_input, grad_gamma, grad_beta] = rocm::batchnorm2d_backward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, stream);
            return {grad_input, grad_gamma, grad_beta};
        }
        else if (op_name == "conv2d_forward") {
            // Parse conv2d parameters
            int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : 1;
            int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
            int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;
            int64_t groups = attrs.contains("groups") ? std::stoll(attrs.at("groups")) : 1;

            if (inputs.size() < 2 || inputs.size() > 3) {
                throw std::invalid_argument("conv2d_forward requires 2 or 3 inputs (input, weight, optional bias)");
            }

            const Tensor* bias_ptr = (inputs.size() == 3) ? &inputs[2] : nullptr;
            return {rocm::conv2d_forward_kernel(inputs[0], inputs[1], bias_ptr, stride, padding, dilation, groups, stream)};
        }
        else if (op_name == "conv2d_backward") {
            // Parse conv2d parameters
            int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : 1;
            int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
            int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;
            int64_t groups = attrs.contains("groups") ? std::stoll(attrs.at("groups")) : 1;
            // Handle both "true" and "1" as truthy values for boolean attributes
            auto parse_bool = [](const std::string& val) { return val == "true" || val == "1"; };
            bool compute_grad_input = attrs.contains("compute_grad_input") ? parse_bool(attrs.at("compute_grad_input")) : true;
            bool compute_grad_weight = attrs.contains("compute_grad_weight") ? parse_bool(attrs.at("compute_grad_weight")) : true;
            bool compute_grad_bias = attrs.contains("compute_grad_bias") ? parse_bool(attrs.at("compute_grad_bias")) : true;

            if (inputs.size() != 3) {
                throw std::invalid_argument("conv2d_backward requires 3 inputs (grad_output, input, weight)");
            }

            auto [grad_input, grad_weight, grad_bias] = rocm::conv2d_backward_kernel(
                inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups,
                compute_grad_input, compute_grad_weight, compute_grad_bias, stream);
            return {grad_input, grad_weight, grad_bias};
        }
        else if (op_name == "conv2d_backward_input") {
            // Parse conv2d parameters
            int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : 1;
            int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
            int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;
            int64_t groups = attrs.contains("groups") ? std::stoll(attrs.at("groups")) : 1;

            if (inputs.size() != 2) {
                throw std::invalid_argument("conv2d_backward_input requires 2 inputs (grad_output, weight)");
            }

            // Create a dummy input tensor for shape inference - we need the input shape from attrs
            std::vector<int64_t> input_shape;
            if (attrs.contains("input_shape")) {
                std::string shape_str = attrs.at("input_shape");
                std::stringstream ss(shape_str);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    input_shape.push_back(std::stoll(item));
                }
            }

            auto [grad_input, grad_weight, grad_bias] = rocm::conv2d_backward_kernel(
                inputs[0], Tensor(input_shape, inputs[0].dtype(), inputs[0].device()), inputs[1],
                stride, padding, dilation, groups,
                true, false, false, stream);
            return {grad_input};
        }
        else if (op_name == "conv2d_backward_weight") {
            // Parse conv2d parameters
            int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : 1;
            int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
            int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;
            int64_t groups = attrs.contains("groups") ? std::stoll(attrs.at("groups")) : 1;

            if (inputs.size() != 2) {
                throw std::invalid_argument("conv2d_backward_weight requires 2 inputs (grad_output, input)");
            }

            // Create a dummy weight tensor for shape inference - we need the weight shape from attrs
            std::vector<int64_t> weight_shape;
            if (attrs.contains("weight_shape")) {
                std::string shape_str = attrs.at("weight_shape");
                std::stringstream ss(shape_str);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    weight_shape.push_back(std::stoll(item));
                }
            }

            auto [grad_input, grad_weight, grad_bias] = rocm::conv2d_backward_kernel(
                inputs[0], inputs[1], Tensor(weight_shape, inputs[0].dtype(), inputs[0].device()),
                stride, padding, dilation, groups,
                false, true, false, stream);
            return {grad_weight};
        }
        else if (op_name == "conv2d_backward_bias") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("conv2d_backward_bias requires 1 input (grad_output)");
            }

            // Compute bias gradient by summing over batch, height, width dimensions
            // grad_bias[c] = sum(grad_output[:, c, :, :])
            auto grad_shape = inputs[0].shape();
            int64_t out_channels = grad_shape[1];

            // Create dummy tensors for the backward call
            std::vector<int64_t> dummy_input_shape = {grad_shape[0], out_channels, 1, 1};
            std::vector<int64_t> dummy_weight_shape = {out_channels, out_channels, 1, 1};

            auto [grad_input, grad_weight, grad_bias] = rocm::conv2d_backward_kernel(
                inputs[0],
                Tensor(dummy_input_shape, inputs[0].dtype(), inputs[0].device()),
                Tensor(dummy_weight_shape, inputs[0].dtype(), inputs[0].device()),
                1, 0, 1, 1,
                false, false, true, stream);
            return {grad_bias};
        }
        else if (op_name == "max_pool2d") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("max_pool2d operation requires exactly 1 input");
            }

            int64_t kernel_size = attrs.contains("kernel_size") ? std::stoll(attrs.at("kernel_size")) : 2;
            int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : kernel_size;
            int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;

            auto [output, indices] = rocm::maxpool2d_forward_hip(inputs[0], kernel_size, kernel_size, stride, stride, padding, padding, true);
            return {output, indices};
        }
        else if (op_name == "max_pool2d_backward") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("max_pool2d_backward operation requires exactly 2 inputs (grad_output, indices)");
            }

            int64_t H_in = attrs.contains("H_in") ? std::stoll(attrs.at("H_in")) : 0;
            int64_t W_in = attrs.contains("W_in") ? std::stoll(attrs.at("W_in")) : 0;

            // Reconstruct input shape from grad_output shape and H_in/W_in
            auto grad_shape = inputs[0].shape();
            std::vector<int64_t> input_shape = {grad_shape[0], grad_shape[1], H_in, W_in};

            return {rocm::maxpool2d_backward_hip(inputs[0], inputs[1], input_shape)};
        }
        else if (op_name == "avg_pool2d") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("avg_pool2d operation requires exactly 1 input");
            }

            int64_t kernel_size = attrs.contains("kernel_size") ? std::stoll(attrs.at("kernel_size")) : 2;
            int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : kernel_size;
            int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
            bool count_include_pad = attrs.contains("count_include_pad") ? (attrs.at("count_include_pad") == "true") : true;

            return {rocm::avgpool2d_forward_hip(inputs[0], kernel_size, kernel_size, stride, stride, padding, padding, count_include_pad)};
        }
        else if (op_name == "gather_relative_position_bias") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("gather_relative_position_bias operation requires exactly 2 inputs");
            }

            int64_t num_positions = attrs.contains("num_positions") ? std::stoll(attrs.at("num_positions")) : 0;
            int64_t num_heads = attrs.contains("num_heads") ? std::stoll(attrs.at("num_heads")) : 0;

            return {rocm::gather_relative_position_bias_kernel(inputs[0], inputs[1], num_positions, num_heads, stream)};
        }
        else if (op_name == "adaptive_avg_pool2d") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("adaptive_avg_pool2d operation requires exactly 1 input");
            }

            int64_t output_h = attrs.contains("output_h") ? std::stoll(attrs.at("output_h")) : 1;
            int64_t output_w = attrs.contains("output_w") ? std::stoll(attrs.at("output_w")) : 1;

            return {rocm::adaptive_avgpool2d_forward(inputs[0], output_h, output_w, stream)};
        }
        else if (op_name == "adaptive_avg_pool2d_backward") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("adaptive_avg_pool2d_backward operation requires exactly 1 input");
            }

            int64_t H_in = attrs.contains("H_in") ? std::stoll(attrs.at("H_in")) : 0;
            int64_t W_in = attrs.contains("W_in") ? std::stoll(attrs.at("W_in")) : 0;

            return {rocm::adaptive_avgpool2d_backward(inputs[0], H_in, W_in, stream)};
        }
        else {
            throw std::runtime_error("ROCmBackend: Unknown operation '" + op_name + "'");
        }
    }
    catch (const std::exception& e) {
        // Check for HIP errors and provide more context
        hipError_t hip_error = hipGetLastError();
        if (hip_error != hipSuccess) {
            throw std::runtime_error(
                "ROCmBackend: Operation '" + op_name + "' failed with HIP error: " +
                hipGetErrorString(hip_error) + " (Original exception: " + e.what() + ")"
            );
        }
        throw;
    }
}

auto ROCmBackend::get_device_properties(int32_t device_id) const -> hipDeviceProp_t {
    hipDeviceProp_t props;
    check_hip_error(hipGetDeviceProperties(&props, device_id), "hipGetDeviceProperties");
    return props;
}

void ROCmBackend::check_hip_error(hipError_t err, const char* operation) const {
    if (err != hipSuccess) {
        std::stringstream ss;
        ss << "ROCm operation '" << operation << "' failed: " << hipGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
}

// Factory function for backend creation
extern "C" {
    auto create_backend() -> std::unique_ptr<Backend> {
        return std::make_unique<ROCmBackend>();
    }
}

} // namespace tenzor
