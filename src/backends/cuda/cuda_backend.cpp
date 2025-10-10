#include "tenzor/backend/backend.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <limits>

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
    auto expand_kernel(const Tensor& input, const std::vector<int64_t>& shape, void* stream) -> Tensor;

    // Fill operations
    auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto fill_kernel(const Tensor& tensor, float value, cudaStream_t stream) -> Tensor;

    // Random operations
    auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;
} // namespace cuda

class CUDABackend : public Backend {
public:
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
};

extern "C" {
    auto create_backend() -> std::unique_ptr<Backend> {
        return std::make_unique<CUDABackend>();
    }
}

} // namespace tenzor
