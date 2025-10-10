#include "tenzor/backend/backend.hpp"
#include <cstring>
#include <stdexcept>

namespace tenzor {

// Forward declarations for CPU kernels
namespace cpu {
    auto add_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto sub_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto mul_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto div_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto matmul_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto sqrt_kernel(const Tensor& input) -> Tensor;
    auto neg_kernel(const Tensor& input) -> Tensor;
    auto abs_kernel(const Tensor& input) -> Tensor;
    auto clamp_kernel(const Tensor& input, float min_val, float max_val) -> Tensor;
    auto log_kernel(const Tensor& input) -> Tensor;
    auto exp_kernel(const Tensor& input) -> Tensor;
    auto pow_kernel(const Tensor& input, float exponent) -> Tensor;
    auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto max_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto min_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;

    // Activation kernels
    auto relu_kernel(const Tensor& input) -> Tensor;
    auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto sigmoid_kernel(const Tensor& input) -> Tensor;
    auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto tanh_kernel(const Tensor& input) -> Tensor;
    auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto leaky_relu_kernel(const Tensor& input, float alpha) -> Tensor;
    auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha) -> Tensor;
    auto softmax_kernel(const Tensor& input, int64_t dim) -> Tensor;
    auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim) -> Tensor;
    auto log_softmax_kernel(const Tensor& input, int64_t dim) -> Tensor;
    auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim) -> Tensor;

    // Transform kernels
    auto contiguous_kernel(const Tensor& input) -> Tensor;
    auto fill_kernel(const Tensor& input, float value) -> Tensor;
    auto clone_kernel(const Tensor& input) -> Tensor;
    auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape) -> Tensor;
    auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor;
    auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims) -> Tensor;
    auto squeeze_kernel(const Tensor& input, int64_t dim) -> Tensor;
    auto unsqueeze_kernel(const Tensor& input, int64_t dim) -> Tensor;

    // BatchNorm kernels
    auto batchnorm2d_mean_var_kernel(const Tensor& input) -> std::vector<Tensor>;
    auto batchnorm2d_forward_kernel(const Tensor& input, const Tensor& mean, const Tensor& variance, float epsilon) -> Tensor;
    auto batchnorm2d_forward_affine_kernel(const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, const Tensor& beta, float epsilon) -> Tensor;
    auto batchnorm2d_update_running_stats_kernel(Tensor& running_mean, Tensor& running_var, const Tensor& batch_mean, const Tensor& batch_var, float momentum) -> void;
    auto batchnorm2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, float epsilon) -> std::vector<Tensor>;

    // Creation kernels
    auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;

    // Conv2d kernels
    auto conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv2d_backward_input_kernel(const Tensor& grad_output, const Tensor& weight, const std::vector<int64_t>& input_shape, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv2d_backward_weight_kernel(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& weight_shape, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv2d_backward_bias_kernel(const Tensor& grad_output) -> Tensor;
} // namespace cpu

class CPUBackend : public Backend {
public:
    auto name() const -> std::string_view override {
        return "cpu";
    }

    auto device_count() const -> int32_t override {
        return 1;
    }

    auto is_available() const -> bool override {
        return true;
    }

    auto allocate(size_t bytes, int32_t device_id) -> void* override {
        #ifdef _WIN32
            return _aligned_malloc(bytes, 64);
        #else
            void* ptr = nullptr;
            posix_memalign(&ptr, 64, bytes);
            return ptr;
        #endif
    }

    auto deallocate(void* ptr) -> void override {
        if (!ptr) return;
        #ifdef _WIN32
            _aligned_free(ptr);
        #else
            free(ptr);
        #endif
    }

    auto copy(void* dst, const void* src, size_t bytes, CopyKind kind) -> void override {
        std::memcpy(dst, src, bytes);
    }

    auto synchronize(int32_t device_id) -> void override {
        // CPU is always synchronized
    }

    auto create_stream(int32_t device_id) -> StreamHandle override {
        return nullptr;
    }

    auto destroy_stream(StreamHandle stream) -> void override {
        // No-op for CPU
    }

    auto synchronize_stream(StreamHandle stream) -> void override {
        // No-op for CPU
    }

    auto dispatch(const std::string& op_name,
                 std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> std::vector<Tensor> override {
        // Validate we have inputs
        if (inputs.empty()) {
            throw std::invalid_argument("dispatch requires at least one input tensor");
        }

        // Validate all inputs are on CPU device
        for (const auto& tensor : inputs) {
            if (tensor.device().type != Device::Type::CPU) {
                throw std::runtime_error(
                    "CPUBackend: All input tensors must be on CPU device, got: " +
                    tensor.device().to_string()
                );
            }
        }

        // Dispatch to appropriate kernel based on operation name
        if (op_name == "add") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("add operation requires exactly 2 inputs");
            }
            return {cpu::add_kernel(inputs[0], inputs[1])};
        }
        else if (op_name == "sub") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("sub operation requires exactly 2 inputs");
            }
            return {cpu::sub_kernel(inputs[0], inputs[1])};
        }
        else if (op_name == "mul") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("mul operation requires exactly 2 inputs");
            }
            return {cpu::mul_kernel(inputs[0], inputs[1])};
        }
        else if (op_name == "div") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("div operation requires exactly 2 inputs");
            }
            return {cpu::div_kernel(inputs[0], inputs[1])};
        }
        else if (op_name == "matmul") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("matmul operation requires exactly 2 inputs");
            }
            return {cpu::matmul_kernel(inputs[0], inputs[1])};
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
            return {cpu::sum_kernel(inputs[0], dim, keepdim)};
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
            return {cpu::mean_kernel(inputs[0], dim, keepdim)};
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
            return {cpu::max_kernel(inputs[0], dim, keepdim)};
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
            return {cpu::min_kernel(inputs[0], dim, keepdim)};
        }
        else if (op_name == "sqrt") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("sqrt operation requires exactly 1 input");
            }
            return {cpu::sqrt_kernel(inputs[0])};
        }
        else if (op_name == "neg") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("neg operation requires exactly 1 input");
            }
            return {cpu::neg_kernel(inputs[0])};
        }
        else if (op_name == "abs") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("abs operation requires exactly 1 input");
            }
            return {cpu::abs_kernel(inputs[0])};
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
            return {cpu::clamp_kernel(inputs[0], min_val, max_val)};
        }
        else if (op_name == "log") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("log operation requires exactly 1 input");
            }
            return {cpu::log_kernel(inputs[0])};
        }
        else if (op_name == "exp") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("exp operation requires exactly 1 input");
            }
            return {cpu::exp_kernel(inputs[0])};
        }
        else if (op_name == "pow") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("pow operation requires exactly 1 input");
            }
            float exponent = 2.0f;
            if (attrs.contains("exponent")) {
                exponent = std::stof(attrs.at("exponent"));
            }
            return {cpu::pow_kernel(inputs[0], exponent)};
        }
        else if (op_name == "relu") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("relu operation requires exactly 1 input");
            }
            return {cpu::relu_kernel(inputs[0])};
        }
        else if (op_name == "relu_backward") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("relu_backward operation requires exactly 2 inputs");
            }
            return {cpu::relu_backward_kernel(inputs[0], inputs[1])};
        }
        else if (op_name == "sigmoid") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("sigmoid operation requires exactly 1 input");
            }
            return {cpu::sigmoid_kernel(inputs[0])};
        }
        else if (op_name == "sigmoid_backward") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("sigmoid_backward operation requires exactly 2 inputs");
            }
            return {cpu::sigmoid_backward_kernel(inputs[0], inputs[1])};
        }
        else if (op_name == "tanh") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("tanh operation requires exactly 1 input");
            }
            return {cpu::tanh_kernel(inputs[0])};
        }
        else if (op_name == "tanh_backward") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("tanh_backward operation requires exactly 2 inputs");
            }
            return {cpu::tanh_backward_kernel(inputs[0], inputs[1])};
        }
        else if (op_name == "leaky_relu") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("leaky_relu operation requires exactly 1 input");
            }
            float alpha = 0.01f;
            if (attrs.contains("alpha")) {
                alpha = std::stof(attrs.at("alpha"));
            }
            return {cpu::leaky_relu_kernel(inputs[0], alpha)};
        }
        else if (op_name == "leaky_relu_backward") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("leaky_relu_backward operation requires exactly 2 inputs");
            }
            float alpha = 0.01f;
            if (attrs.contains("alpha")) {
                alpha = std::stof(attrs.at("alpha"));
            }
            return {cpu::leaky_relu_backward_kernel(inputs[0], inputs[1], alpha)};
        }
        else if (op_name == "softmax") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("softmax operation requires exactly 1 input");
            }
            int64_t dim = -1;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            return {cpu::softmax_kernel(inputs[0], dim)};
        }
        else if (op_name == "softmax_backward") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("softmax_backward operation requires exactly 2 inputs");
            }
            int64_t dim = -1;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            return {cpu::softmax_backward_kernel(inputs[0], inputs[1], dim)};
        }
        else if (op_name == "log_softmax") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("log_softmax operation requires exactly 1 input");
            }
            int64_t dim = -1;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            return {cpu::log_softmax_kernel(inputs[0], dim)};
        }
        else if (op_name == "log_softmax_backward") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("log_softmax_backward operation requires exactly 2 inputs");
            }
            int64_t dim = -1;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            return {cpu::log_softmax_backward_kernel(inputs[0], inputs[1], dim)};
        }
        else if (op_name == "contiguous") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("contiguous operation requires exactly 1 input");
            }
            return {cpu::contiguous_kernel(inputs[0])};
        }
        else if (op_name == "fill") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("fill operation requires exactly 1 input");
            }
            float value = 0.0f;
            if (attrs.contains("value")) {
                value = std::stof(attrs.at("value"));
            }
            return {cpu::fill_kernel(inputs[0], value)};
        }
        else if (op_name == "clone") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("clone operation requires exactly 1 input");
            }
            return {cpu::clone_kernel(inputs[0])};
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
            return {cpu::reshape_kernel(inputs[0], shape)};
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
            return {cpu::transpose_kernel(inputs[0], dim0, dim1)};
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
            return {cpu::permute_kernel(inputs[0], dims)};
        }
        else if (op_name == "squeeze") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("squeeze operation requires exactly 1 input");
            }
            int64_t dim = -1;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            return {cpu::squeeze_kernel(inputs[0], dim)};
        }
        else if (op_name == "unsqueeze") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("unsqueeze operation requires exactly 1 input");
            }
            int64_t dim = 0;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            return {cpu::unsqueeze_kernel(inputs[0], dim)};
        }
        else if (op_name == "batchnorm2d_mean_var") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("batchnorm2d_mean_var operation requires exactly 1 input");
            }
            return cpu::batchnorm2d_mean_var_kernel(inputs[0]);
        }
        else if (op_name == "batchnorm2d_forward") {
            if (inputs.size() != 3) {
                throw std::invalid_argument("batchnorm2d_forward operation requires exactly 3 inputs");
            }
            float epsilon = 1e-5f;
            if (attrs.contains("epsilon")) {
                epsilon = std::stof(attrs.at("epsilon"));
            }
            return {cpu::batchnorm2d_forward_kernel(inputs[0], inputs[1], inputs[2], epsilon)};
        }
        else if (op_name == "batchnorm2d_forward_affine") {
            if (inputs.size() != 5) {
                throw std::invalid_argument("batchnorm2d_forward_affine operation requires exactly 5 inputs");
            }
            float epsilon = 1e-5f;
            if (attrs.contains("epsilon")) {
                epsilon = std::stof(attrs.at("epsilon"));
            }
            return {cpu::batchnorm2d_forward_affine_kernel(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon)};
        }
        else if (op_name == "batchnorm2d_update_running_stats") {
            if (inputs.size() != 4) {
                throw std::invalid_argument("batchnorm2d_update_running_stats operation requires exactly 4 inputs");
            }
            float momentum = 0.1f;
            if (attrs.contains("momentum")) {
                momentum = std::stof(attrs.at("momentum"));
            }
            // Create copies that will be modified by the kernel
            Tensor updated_mean = inputs[0].clone();
            Tensor updated_var = inputs[1].clone();
            cpu::batchnorm2d_update_running_stats_kernel(updated_mean, updated_var, inputs[2], inputs[3], momentum);
            return {updated_mean, updated_var};
        }
        else if (op_name == "batchnorm2d_backward") {
            if (inputs.size() != 5) {
                throw std::invalid_argument("batchnorm2d_backward operation requires exactly 5 inputs");
            }
            float epsilon = 1e-5f;
            if (attrs.contains("epsilon")) {
                epsilon = std::stof(attrs.at("epsilon"));
            }
            return cpu::batchnorm2d_backward_kernel(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon);
        }
        else if (op_name == "zeros") {
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
            DType dtype = inputs.empty() ? DType::Float32 : inputs[0].dtype();
            Device device = inputs.empty() ? Device{Device::Type::CPU, 0} : inputs[0].device();
            if (attrs.contains("dtype")) {
                // Parse dtype if provided
                std::string dtype_str = attrs.at("dtype");
                if (dtype_str == "float32") dtype = DType::Float32;
                else if (dtype_str == "float64") dtype = DType::Float64;
                else if (dtype_str == "int32") dtype = DType::Int32;
                else if (dtype_str == "int64") dtype = DType::Int64;
            }
            return {cpu::zeros_kernel(shape, dtype, device)};
        }
        else if (op_name == "ones") {
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
            DType dtype = inputs.empty() ? DType::Float32 : inputs[0].dtype();
            Device device = inputs.empty() ? Device{Device::Type::CPU, 0} : inputs[0].device();
            if (attrs.contains("dtype")) {
                std::string dtype_str = attrs.at("dtype");
                if (dtype_str == "float32") dtype = DType::Float32;
                else if (dtype_str == "float64") dtype = DType::Float64;
                else if (dtype_str == "int32") dtype = DType::Int32;
                else if (dtype_str == "int64") dtype = DType::Int64;
            }
            return {cpu::ones_kernel(shape, dtype, device)};
        }
        else if (op_name == "rand") {
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
            DType dtype = inputs.empty() ? DType::Float32 : inputs[0].dtype();
            Device device = inputs.empty() ? Device{Device::Type::CPU, 0} : inputs[0].device();
            if (attrs.contains("dtype")) {
                std::string dtype_str = attrs.at("dtype");
                if (dtype_str == "float32") dtype = DType::Float32;
                else if (dtype_str == "float64") dtype = DType::Float64;
            }
            return {cpu::rand_kernel(shape, dtype, device)};
        }
        else if (op_name == "randn") {
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
            DType dtype = inputs.empty() ? DType::Float32 : inputs[0].dtype();
            Device device = inputs.empty() ? Device{Device::Type::CPU, 0} : inputs[0].device();
            if (attrs.contains("dtype")) {
                std::string dtype_str = attrs.at("dtype");
                if (dtype_str == "float32") dtype = DType::Float32;
                else if (dtype_str == "float64") dtype = DType::Float64;
            }
            return {cpu::randn_kernel(shape, dtype, device)};
        }
        else if (op_name == "conv2d_forward") {
            if (inputs.size() < 2) {
                throw std::invalid_argument("conv2d_forward operation requires at least 2 inputs (input, weight)");
            }
            const Tensor* bias = (inputs.size() >= 3) ? &inputs[2] : nullptr;
            int64_t stride = 1;
            int64_t padding = 0;
            int64_t dilation = 1;
            int64_t groups = 1;
            if (attrs.contains("stride")) {
                stride = std::stoll(attrs.at("stride"));
            }
            if (attrs.contains("padding")) {
                padding = std::stoll(attrs.at("padding"));
            }
            if (attrs.contains("dilation")) {
                dilation = std::stoll(attrs.at("dilation"));
            }
            if (attrs.contains("groups")) {
                groups = std::stoll(attrs.at("groups"));
            }
            return {cpu::conv2d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, groups)};
        }
        else if (op_name == "conv2d_backward_input") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("conv2d_backward_input operation requires exactly 2 inputs (grad_output, weight)");
            }
            // Parse input_shape from comma-separated string
            std::vector<int64_t> input_shape;
            if (attrs.contains("input_shape")) {
                std::string shape_str = attrs.at("input_shape");
                size_t pos = 0;
                while (pos < shape_str.size()) {
                    size_t comma = shape_str.find(',', pos);
                    if (comma == std::string::npos) {
                        input_shape.push_back(std::stoll(shape_str.substr(pos)));
                        break;
                    }
                    input_shape.push_back(std::stoll(shape_str.substr(pos, comma - pos)));
                    pos = comma + 1;
                }
            }
            int64_t stride = 1;
            int64_t padding = 0;
            int64_t dilation = 1;
            int64_t groups = 1;
            if (attrs.contains("stride")) {
                stride = std::stoll(attrs.at("stride"));
            }
            if (attrs.contains("padding")) {
                padding = std::stoll(attrs.at("padding"));
            }
            if (attrs.contains("dilation")) {
                dilation = std::stoll(attrs.at("dilation"));
            }
            if (attrs.contains("groups")) {
                groups = std::stoll(attrs.at("groups"));
            }
            return {cpu::conv2d_backward_input_kernel(inputs[0], inputs[1], input_shape, stride, padding, dilation, groups)};
        }
        else if (op_name == "conv2d_backward_weight") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("conv2d_backward_weight operation requires exactly 2 inputs (grad_output, input)");
            }
            // Parse weight_shape from comma-separated string
            std::vector<int64_t> weight_shape;
            if (attrs.contains("weight_shape")) {
                std::string shape_str = attrs.at("weight_shape");
                size_t pos = 0;
                while (pos < shape_str.size()) {
                    size_t comma = shape_str.find(',', pos);
                    if (comma == std::string::npos) {
                        weight_shape.push_back(std::stoll(shape_str.substr(pos)));
                        break;
                    }
                    weight_shape.push_back(std::stoll(shape_str.substr(pos, comma - pos)));
                    pos = comma + 1;
                }
            }
            int64_t stride = 1;
            int64_t padding = 0;
            int64_t dilation = 1;
            int64_t groups = 1;
            if (attrs.contains("stride")) {
                stride = std::stoll(attrs.at("stride"));
            }
            if (attrs.contains("padding")) {
                padding = std::stoll(attrs.at("padding"));
            }
            if (attrs.contains("dilation")) {
                dilation = std::stoll(attrs.at("dilation"));
            }
            if (attrs.contains("groups")) {
                groups = std::stoll(attrs.at("groups"));
            }
            return {cpu::conv2d_backward_weight_kernel(inputs[0], inputs[1], weight_shape, stride, padding, dilation, groups)};
        }
        else if (op_name == "conv2d_backward_bias") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("conv2d_backward_bias operation requires exactly 1 input (grad_output)");
            }
            return {cpu::conv2d_backward_bias_kernel(inputs[0])};
        }
        else {
            throw std::runtime_error("CPUBackend: Unknown operation '" + op_name + "'");
        }
    }
};

// Export factory function
extern "C" {
    auto create_backend() -> std::unique_ptr<Backend> {
        return std::make_unique<CPUBackend>();
    }
}

} // namespace tenzor
