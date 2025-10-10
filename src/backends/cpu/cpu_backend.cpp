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
