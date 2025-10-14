#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/caching_allocator.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <limits>
#include <cstdlib>

namespace tenzor {

// Forward declarations for CUDA kernels
// These will be implemented by kernel developers in separate .cu files
namespace cuda {
    // Binary operations
    auto add_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto sub_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto mul_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto div_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto matmul_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;

    // Unary operations
    auto sqrt_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto neg_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto abs_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto sign_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto log_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto exp_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;

    // Operations with parameters
    auto clamp_kernel(const Tensor& input, float min_val, float max_val, cudaStream_t stream) -> Tensor;
    auto pow_kernel(const Tensor& input, float exponent, cudaStream_t stream) -> Tensor;

    // Reduction operations
    auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto max_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto min_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;

    // Activation functions
    auto relu_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto sigmoid_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto tanh_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto leaky_relu_kernel(const Tensor& input, float alpha, cudaStream_t stream) -> Tensor;
    auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, cudaStream_t stream) -> Tensor;

    // Softmax operations
    auto softmax_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, cudaStream_t stream) -> Tensor;
    auto log_softmax_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, cudaStream_t stream) -> Tensor;

    // Transform operations
    auto contiguous_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto clone_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape, cudaStream_t stream) -> Tensor;
    auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1, cudaStream_t stream) -> Tensor;
    auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims, cudaStream_t stream) -> Tensor;
    auto squeeze_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto unsqueeze_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto expand_kernel(const Tensor& input, const std::vector<int64_t>& shape, void* stream) -> Tensor;

    // Fill operations
    auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto fill_kernel(const Tensor& tensor, float value, cudaStream_t stream) -> Tensor;

    // Random operations
    auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;

    // BatchNorm2d operations
    auto batchnorm2d_mean_var(const Tensor& input, Tensor& mean, Tensor& variance, cudaStream_t stream) -> void;
    auto batchnorm2d_forward(const Tensor& input, const Tensor& mean, const Tensor& variance, float epsilon, cudaStream_t stream) -> Tensor;
    auto batchnorm2d_forward_affine(const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, const Tensor& beta, float epsilon, cudaStream_t stream) -> Tensor;
    auto batchnorm2d_update_running_stats(Tensor& running_mean, Tensor& running_var, const Tensor& batch_mean, const Tensor& batch_var, float momentum, cudaStream_t stream) -> void;
    auto batchnorm2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, float epsilon, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
} // namespace cuda

class CUDABackend : public Backend {
public:
    CUDABackend() {
        // Check if caching allocator is enabled via environment variable
        const char* enable_caching = std::getenv("TENZOR_ENABLE_CACHING_ALLOCATOR");
        use_caching_allocator_ = (enable_caching != nullptr && std::string(enable_caching) == "1");
    }

    auto name() const -> std::string_view override {
        return "cuda";
    }

    auto device_count() const -> int32_t override {
        int count = 0;
        cudaGetDeviceCount(&count);
        return count;
    }

    auto is_available() const -> bool override {
        return device_count() > 0;
    }

    auto allocate(size_t bytes, int32_t device_id) -> void* override {
        // Handle empty tensors - CUDA doesn't like 0-byte allocations
        if (bytes == 0) {
            return nullptr;
        }

        if (use_caching_allocator_) {
            return backend::CachingAllocator::get().allocate(bytes, device_id);
        }

        void* ptr = nullptr;
        cudaSetDevice(device_id);
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("Failed to allocate device memory: ") + cudaGetErrorString(err)
            );
        }
        return ptr;
    }

    auto deallocate(void* ptr) -> void override {
        // Handle nullptr from empty tensor allocations
        if (ptr == nullptr) {
            return;
        }

        if (use_caching_allocator_) {
            // Note: we don't know the device_id here, but CachingAllocator tracks it
            // For proper integration, we'd need to look up the device from the pointer
            int device_id = 0;
            cudaPointerAttributes attrs;
            if (cudaPointerGetAttributes(&attrs, ptr) == cudaSuccess) {
                device_id = attrs.device;
            }
            backend::CachingAllocator::get().free(ptr, device_id);
            return;
        }

        cudaFree(ptr);
    }

    auto copy(void* dst, const void* src, size_t bytes, CopyKind kind) -> void override {
        // Handle empty tensors
        if (bytes == 0) {
            return;
        }

        cudaMemcpyKind cuda_kind;
        switch (kind) {
            case CopyKind::HostToHost: cuda_kind = cudaMemcpyHostToHost; break;
            case CopyKind::HostToDevice: cuda_kind = cudaMemcpyHostToDevice; break;
            case CopyKind::DeviceToHost: cuda_kind = cudaMemcpyDeviceToHost; break;
            case CopyKind::DeviceToDevice: cuda_kind = cudaMemcpyDeviceToDevice; break;
        }

        cudaError_t err = cudaMemcpy(dst, src, bytes, cuda_kind);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA copy failed: ") + cudaGetErrorString(err)
            );
        }
    }

    auto synchronize(int32_t device_id) -> void override {
        cudaSetDevice(device_id);
        cudaDeviceSynchronize();
    }

    auto create_stream(int32_t device_id) -> StreamHandle override {
        cudaStream_t stream;
        cudaSetDevice(device_id);
        cudaStreamCreate(&stream);
        return static_cast<StreamHandle>(stream);
    }

    auto destroy_stream(StreamHandle stream) -> void override {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    }

    auto synchronize_stream(StreamHandle stream) -> void override {
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    }

    auto dispatch(const std::string& op_name,
                 std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> std::vector<Tensor> override {
        // Allow empty inputs for creation operations
        bool is_creation_op = (op_name == "zeros" || op_name == "ones" || op_name == "full" ||
                               op_name == "rand" || op_name == "randn");

        // Validate we have inputs (except for creation operations)
        if (inputs.empty() && !is_creation_op) {
            throw std::invalid_argument("dispatch requires at least one input tensor");
        }

        // Validate all inputs are on CUDA device (if any)
        for (const auto& tensor : inputs) {
            if (tensor.device().type != Device::Type::CUDA) {
                throw std::runtime_error(
                    "CUDABackend: All input tensors must be on CUDA device, got: " +
                    tensor.device().to_string()
                );
            }
        }

        // Set CUDA device - use first tensor's device or device from attrs
        int32_t device_id = 0;
        if (!inputs.empty()) {
            device_id = inputs[0].device().index;
        } else if (attrs.contains("device_id")) {
            device_id = std::stoi(attrs.at("device_id"));
        }
        cudaSetDevice(device_id);

        // Get or create stream (nullptr means default stream)
        cudaStream_t stream = nullptr;
        if (attrs.contains("stream")) {
            stream = static_cast<cudaStream_t>(
                reinterpret_cast<void*>(std::stoull(attrs.at("stream")))
            );
        }

        // Dispatch to appropriate CUDA kernel based on operation name
        try {
            if (op_name == "add") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("add operation requires exactly 2 inputs");
                }
                return {cuda::add_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "sub") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("sub operation requires exactly 2 inputs");
                }
                return {cuda::sub_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "mul") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("mul operation requires exactly 2 inputs");
                }
                return {cuda::mul_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "div") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("div operation requires exactly 2 inputs");
                }
                return {cuda::div_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "matmul") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("matmul operation requires exactly 2 inputs");
                }
                return {cuda::matmul_kernel(inputs[0], inputs[1], stream)};
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
                return {cuda::sum_kernel(inputs[0], dim, keepdim, stream)};
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
                return {cuda::mean_kernel(inputs[0], dim, keepdim, stream)};
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
                return {cuda::max_kernel(inputs[0], dim, keepdim, stream)};
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
                return {cuda::min_kernel(inputs[0], dim, keepdim, stream)};
            }
            else if (op_name == "sqrt") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("sqrt operation requires exactly 1 input");
                }
                return {cuda::sqrt_kernel(inputs[0], stream)};
            }
            else if (op_name == "neg") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("neg operation requires exactly 1 input");
                }
                return {cuda::neg_kernel(inputs[0], stream)};
            }
            else if (op_name == "abs") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("abs operation requires exactly 1 input");
                }
                return {cuda::abs_kernel(inputs[0], stream)};
            }
            else if (op_name == "sign") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("sign operation requires exactly 1 input");
                }
                return {cuda::sign_kernel(inputs[0], stream)};
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
                return {cuda::clamp_kernel(inputs[0], min_val, max_val, stream)};
            }
            else if (op_name == "log") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("log operation requires exactly 1 input");
                }
                return {cuda::log_kernel(inputs[0], stream)};
            }
            else if (op_name == "exp") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("exp operation requires exactly 1 input");
                }
                return {cuda::exp_kernel(inputs[0], stream)};
            }
            else if (op_name == "pow") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("pow operation requires exactly 1 input");
                }
                float exponent = 2.0f;
                if (attrs.contains("exponent")) {
                    exponent = std::stof(attrs.at("exponent"));
                }
                return {cuda::pow_kernel(inputs[0], exponent, stream)};
            }
            else if (op_name == "relu") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("relu operation requires exactly 1 input");
                }
                return {cuda::relu_kernel(inputs[0], stream)};
            }
            else if (op_name == "relu_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("relu_backward operation requires exactly 2 inputs");
                }
                return {cuda::relu_backward_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "sigmoid") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("sigmoid operation requires exactly 1 input");
                }
                return {cuda::sigmoid_kernel(inputs[0], stream)};
            }
            else if (op_name == "sigmoid_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("sigmoid_backward operation requires exactly 2 inputs");
                }
                return {cuda::sigmoid_backward_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "tanh") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("tanh operation requires exactly 1 input");
                }
                return {cuda::tanh_kernel(inputs[0], stream)};
            }
            else if (op_name == "tanh_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("tanh_backward operation requires exactly 2 inputs");
                }
                return {cuda::tanh_backward_kernel(inputs[0], inputs[1], stream)};
            }
            else if (op_name == "leaky_relu") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("leaky_relu operation requires exactly 1 input");
                }
                float alpha = 0.01f;
                if (attrs.contains("alpha")) {
                    alpha = std::stof(attrs.at("alpha"));
                }
                return {cuda::leaky_relu_kernel(inputs[0], alpha, stream)};
            }
            else if (op_name == "leaky_relu_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("leaky_relu_backward operation requires exactly 2 inputs");
                }
                float alpha = 0.01f;
                if (attrs.contains("alpha")) {
                    alpha = std::stof(attrs.at("alpha"));
                }
                return {cuda::leaky_relu_backward_kernel(inputs[0], inputs[1], alpha, stream)};
            }
            else if (op_name == "softmax") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("softmax operation requires exactly 1 input");
                }
                int64_t dim = -1;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                return {cuda::softmax_kernel(inputs[0], dim, stream)};
            }
            else if (op_name == "softmax_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("softmax_backward operation requires exactly 2 inputs");
                }
                int64_t dim = -1;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                return {cuda::softmax_backward_kernel(inputs[0], inputs[1], dim, stream)};
            }
            else if (op_name == "log_softmax") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("log_softmax operation requires exactly 1 input");
                }
                int64_t dim = -1;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                return {cuda::log_softmax_kernel(inputs[0], dim, stream)};
            }
            else if (op_name == "log_softmax_backward") {
                if (inputs.size() != 2) {
                    throw std::invalid_argument("log_softmax_backward operation requires exactly 2 inputs");
                }
                int64_t dim = -1;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                return {cuda::log_softmax_backward_kernel(inputs[0], inputs[1], dim, stream)};
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

                Device device = inputs.empty() ? Device::cuda(0) : inputs[0].device();
                return {cuda::zeros_kernel(shape, dtype, device, stream)};
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

                Device device = inputs.empty() ? Device::cuda(0) : inputs[0].device();
                return {cuda::ones_kernel(shape, dtype, device, stream)};
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

                Device device = inputs.empty() ? Device::cuda(0) : inputs[0].device();
                return {cuda::full_kernel(shape, value, dtype, device, stream)};
            }
            else if (op_name == "fill") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("fill operation requires exactly 1 input");
                }
                if (!attrs.contains("value")) {
                    throw std::invalid_argument("fill operation requires 'value' attribute");
                }
                float value = std::stof(attrs.at("value"));
                return {cuda::fill_kernel(inputs[0], value, stream)};
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

                return {cuda::expand_kernel(inputs[0], shape, static_cast<void*>(stream))};
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

                Device device = inputs.empty() ? Device::cuda(0) : inputs[0].device();
                return {cuda::rand_kernel(shape, dtype, device, stream)};
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

                Device device = inputs.empty() ? Device::cuda(0) : inputs[0].device();
                return {cuda::randn_kernel(shape, dtype, device, stream)};
            }
            else if (op_name == "contiguous") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("contiguous operation requires exactly 1 input");
                }
                return {cuda::contiguous_kernel(inputs[0], stream)};
            }
            else if (op_name == "clone") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("clone operation requires exactly 1 input");
                }
                return {cuda::clone_kernel(inputs[0], stream)};
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
                return {cuda::reshape_kernel(inputs[0], shape, stream)};
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
                return {cuda::transpose_kernel(inputs[0], dim0, dim1, stream)};
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
                return {cuda::permute_kernel(inputs[0], dims, stream)};
            }
            else if (op_name == "squeeze") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("squeeze operation requires exactly 1 input");
                }
                int64_t dim = -1;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                return {cuda::squeeze_kernel(inputs[0], dim, stream)};
            }
            else if (op_name == "unsqueeze") {
                if (inputs.size() != 1) {
                    throw std::invalid_argument("unsqueeze operation requires exactly 1 input");
                }
                int64_t dim = 0;
                if (attrs.contains("dim")) {
                    dim = std::stoll(attrs.at("dim"));
                }
                return {cuda::unsqueeze_kernel(inputs[0], dim, stream)};
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
                cuda::batchnorm2d_mean_var(inputs[0], mean, variance, stream);
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
                return {cuda::batchnorm2d_forward(inputs[0], inputs[1], inputs[2], epsilon, stream)};
            }
            else if (op_name == "batchnorm2d_forward_affine") {
                if (inputs.size() != 5) {
                    throw std::invalid_argument("batchnorm2d_forward_affine operation requires exactly 5 inputs (input, mean, variance, gamma, beta)");
                }
                float epsilon = 1e-5f;
                if (attrs.contains("epsilon")) {
                    epsilon = std::stof(attrs.at("epsilon"));
                }
                return {cuda::batchnorm2d_forward_affine(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, stream)};
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
                cuda::batchnorm2d_update_running_stats(running_mean, running_var, inputs[2], inputs[3], momentum, stream);
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
                auto [grad_input, grad_gamma, grad_beta] = cuda::batchnorm2d_backward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, stream);
                return {grad_input, grad_gamma, grad_beta};
            }
            else {
                throw std::runtime_error("CUDABackend: Unknown operation '" + op_name + "'");
            }
        }
        catch (const std::exception& e) {
            // Check for CUDA errors and provide more context
            cudaError_t cuda_error = cudaGetLastError();
            if (cuda_error != cudaSuccess) {
                throw std::runtime_error(
                    "CUDABackend: Operation '" + op_name + "' failed with CUDA error: " +
                    cudaGetErrorString(cuda_error) + " (Original exception: " + e.what() + ")"
                );
            }
            throw;
        }
    }

private:
    bool use_caching_allocator_{false};
};

extern "C" {
    auto create_backend() -> std::unique_ptr<Backend> {
        return std::make_unique<CUDABackend>();
    }
}

} // namespace tenzor
