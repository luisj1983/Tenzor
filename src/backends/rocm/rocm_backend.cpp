#include "rocm_backend.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/rocm_caching_allocator.hip.hpp"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <limits>
#include <cstdlib>
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
            // Parse attributes
            int64_t dim = -1;
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
            int64_t dim = -1;
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
            int64_t dim = -1;
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
            int64_t dim = -1;
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
