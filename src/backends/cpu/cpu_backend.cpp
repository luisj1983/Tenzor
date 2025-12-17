#include "tenzor/backend/backend.hpp"
#include <cstring>
#include <stdexcept>
#include <thread>
#include <fstream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

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
    auto sign_kernel(const Tensor& input) -> Tensor;
    auto clamp_kernel(const Tensor& input, float min_val, float max_val) -> Tensor;
    auto log_kernel(const Tensor& input) -> Tensor;
    auto exp_kernel(const Tensor& input) -> Tensor;
    auto pow_kernel(const Tensor& input, float exponent) -> Tensor;
    auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto max_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto min_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto argmax_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto argmin_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto argsort_kernel(const Tensor& input, int64_t dim, bool descending) -> Tensor;
    auto prod_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto var_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) -> Tensor;
    auto std_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) -> Tensor;
    auto norm_kernel(const Tensor& input, float p, int64_t dim, bool keepdim) -> Tensor;

    // Comparison operations
    auto eq_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto ne_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto lt_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto le_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto gt_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto ge_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto dot_kernel(const Tensor& a, const Tensor& b) -> Tensor;

    // Trigonometric operations
    auto sin_kernel(const Tensor& input) -> Tensor;
    auto cos_kernel(const Tensor& input) -> Tensor;
    auto tan_kernel(const Tensor& input) -> Tensor;
    auto asin_kernel(const Tensor& input) -> Tensor;
    auto acos_kernel(const Tensor& input) -> Tensor;
    auto atan_kernel(const Tensor& input) -> Tensor;
    auto sinh_kernel(const Tensor& input) -> Tensor;
    auto cosh_kernel(const Tensor& input) -> Tensor;

    // Rounding operations
    auto round_kernel(const Tensor& input) -> Tensor;
    auto floor_kernel(const Tensor& input) -> Tensor;
    auto ceil_kernel(const Tensor& input) -> Tensor;

    // Other math operations
    auto reciprocal_kernel(const Tensor& input) -> Tensor;

    // In-place operations
    auto add_inplace_kernel(Tensor& a, const Tensor& b) -> void;
    auto mul_inplace_kernel(Tensor& a, const Tensor& b) -> void;
    auto sub_inplace_kernel(Tensor& a, const Tensor& b) -> void;
    auto div_inplace_kernel(Tensor& a, const Tensor& b) -> void;

    // Activation kernels
    auto relu_kernel(const Tensor& input) -> Tensor;
    auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto sigmoid_kernel(const Tensor& input) -> Tensor;
    auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto tanh_kernel(const Tensor& input) -> Tensor;
    auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto gelu_kernel(const Tensor& input) -> Tensor;
    auto gelu_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto swish_kernel(const Tensor& input) -> Tensor;
    auto swish_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
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

    // Fused operation kernels
    auto fused_linear_relu_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor;
    auto fused_conv2d_relu_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding) -> Tensor;
    auto fused_batchnorm_relu_kernel(const Tensor& input, const Tensor& running_mean, const Tensor& running_var, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto fused_softmax_cross_entropy_kernel(const Tensor& logits, const Tensor& targets, const std::string& reduction) -> Tensor;
    auto fused_add_relu_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto fused_gelu_kernel(const Tensor& input) -> Tensor;
    auto fused_layer_norm_kernel(const Tensor& input, const std::vector<int64_t>& normalized_shape, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;

    // Indexing kernels
    auto index_select_kernel(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor;
    auto gather_kernel(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor;
    auto scatter_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) -> Tensor;
    auto masked_select_kernel(const Tensor& input, const Tensor& mask) -> Tensor;
    auto masked_fill_kernel(const Tensor& input, const Tensor& mask, float value) -> Tensor;
    auto where_kernel(const Tensor& condition, const Tensor& x, const Tensor& y) -> Tensor;
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

    auto get_device_info(int32_t device_id) const -> DeviceInfo override {
        if (device_id != 0) {
            throw std::out_of_range("CPU backend only has device 0");
        }

        DeviceInfo info;
        info.name = "CPU";
        info.vendor = "System";

        // Get number of hardware threads
        info.compute_units = std::thread::hardware_concurrency();
        if (info.compute_units == 0) {
            info.compute_units = 1;  // Fallback
        }

        // CPU always supports FP64 and usually FP16 via software
        info.supports_fp64 = true;
        info.supports_fp16 = true;
        info.is_integrated = true;

        // Try to get system memory info
        #ifdef __linux__
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal:") == 0) {
                size_t kb = 0;
                sscanf(line.c_str(), "MemTotal: %zu kB", &kb);
                info.total_memory = kb * 1024;
            } else if (line.find("MemAvailable:") == 0) {
                size_t kb = 0;
                sscanf(line.c_str(), "MemAvailable: %zu kB", &kb);
                info.available_memory = kb * 1024;
            }
        }
        #elif defined(_WIN32)
        MEMORYSTATUSEX memStatus;
        memStatus.dwLength = sizeof(memStatus);
        if (GlobalMemoryStatusEx(&memStatus)) {
            info.total_memory = memStatus.ullTotalPhys;
            info.available_memory = memStatus.ullAvailPhys;
        }
        #elif defined(__APPLE__)
        int64_t memsize;
        size_t len = sizeof(memsize);
        if (sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) == 0) {
            info.total_memory = static_cast<size_t>(memsize);
        }
        #endif

        return info;
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
            bool dim_specified = false;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
                dim_specified = true;
            }
            if (attrs.contains("keepdim")) {
                keepdim = (attrs.at("keepdim") == "1");
            }
            // Convert negative dim to positive when explicitly specified
            if (dim_specified && dim < 0) {
                int64_t ndim = static_cast<int64_t>(inputs[0].shape().size());
                dim = ndim + dim;
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
        else if (op_name == "argmax") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("argmax operation requires exactly 1 input");
            }
            int64_t dim = -1;
            bool keepdim = false;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            if (attrs.contains("keepdim")) {
                keepdim = (attrs.at("keepdim") == "1");
            }
            return {cpu::argmax_kernel(inputs[0], dim, keepdim)};
        }
        else if (op_name == "argmin") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("argmin operation requires exactly 1 input");
            }
            int64_t dim = -1;
            bool keepdim = false;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            if (attrs.contains("keepdim")) {
                keepdim = (attrs.at("keepdim") == "1");
            }
            return {cpu::argmin_kernel(inputs[0], dim, keepdim)};
        }
        else if (op_name == "prod") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("prod operation requires exactly 1 input");
            }
            int64_t dim = -1;
            bool keepdim = false;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            if (attrs.contains("keepdim")) {
                keepdim = (attrs.at("keepdim") == "1");
            }
            return {cpu::prod_kernel(inputs[0], dim, keepdim)};
        }
        else if (op_name == "var") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("var operation requires exactly 1 input");
            }
            int64_t dim = -1;
            bool keepdim = false;
            int64_t correction = 1;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            if (attrs.contains("keepdim")) {
                keepdim = (attrs.at("keepdim") == "1");
            }
            if (attrs.contains("correction")) {
                correction = std::stoll(attrs.at("correction"));
            }
            return {cpu::var_kernel(inputs[0], dim, keepdim, correction)};
        }
        else if (op_name == "std") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("std operation requires exactly 1 input");
            }
            int64_t dim = -1;
            bool keepdim = false;
            int64_t correction = 1;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            if (attrs.contains("keepdim")) {
                keepdim = (attrs.at("keepdim") == "1");
            }
            if (attrs.contains("correction")) {
                correction = std::stoll(attrs.at("correction"));
            }
            return {cpu::std_kernel(inputs[0], dim, keepdim, correction)};
        }
        else if (op_name == "norm") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("norm operation requires exactly 1 input");
            }
            float p = 2.0f;
            int64_t dim = -1;
            bool keepdim = false;
            if (attrs.contains("p")) {
                p = std::stof(attrs.at("p"));
            }
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            if (attrs.contains("keepdim")) {
                keepdim = (attrs.at("keepdim") == "1");
            }
            return {cpu::norm_kernel(inputs[0], p, dim, keepdim)};
        }
        else if (op_name == "argsort") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("argsort operation requires exactly 1 input");
            }
            int64_t dim = 0;
            bool descending = false;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            if (attrs.contains("descending")) {
                descending = (attrs.at("descending") == "1");
            }
            return {cpu::argsort_kernel(inputs[0], dim, descending)};
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
        else if (op_name == "sign") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("sign operation requires exactly 1 input");
            }
            return {cpu::sign_kernel(inputs[0])};
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
        else if (op_name == "clamp_min") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("clamp_min operation requires exactly 1 input");
            }
            float min_val = -std::numeric_limits<float>::infinity();
            if (attrs.contains("min")) {
                min_val = std::stof(attrs.at("min"));
            }
            return {cpu::clamp_kernel(inputs[0], min_val, std::numeric_limits<float>::infinity())};
        }
        else if (op_name == "clamp_max") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("clamp_max operation requires exactly 1 input");
            }
            float max_val = std::numeric_limits<float>::infinity();
            if (attrs.contains("max")) {
                max_val = std::stof(attrs.at("max"));
            }
            return {cpu::clamp_kernel(inputs[0], -std::numeric_limits<float>::infinity(), max_val)};
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
        else if (op_name == "gelu") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("gelu operation requires exactly 1 input");
            }
            return {cpu::gelu_kernel(inputs[0])};
        }
        else if (op_name == "gelu_backward") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("gelu_backward operation requires exactly 2 inputs");
            }
            return {cpu::gelu_backward_kernel(inputs[0], inputs[1])};
        }
        else if (op_name == "swish") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("swish operation requires exactly 1 input");
            }
            return {cpu::swish_kernel(inputs[0])};
        }
        else if (op_name == "swish_backward") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("swish_backward operation requires exactly 2 inputs");
            }
            return {cpu::swish_backward_kernel(inputs[0], inputs[1])};
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
        else if (op_name == "index_select") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("index_select operation requires exactly 2 inputs");
            }
            int64_t dim = 0;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            return {cpu::index_select_kernel(inputs[0], dim, inputs[1])};
        }
        else if (op_name == "gather") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("gather operation requires exactly 2 inputs");
            }
            int64_t dim = 0;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            return {cpu::gather_kernel(inputs[0], dim, inputs[1])};
        }
        else if (op_name == "scatter") {
            if (inputs.size() != 3) {
                throw std::invalid_argument("scatter operation requires exactly 3 inputs");
            }
            int64_t dim = 0;
            if (attrs.contains("dim")) {
                dim = std::stoll(attrs.at("dim"));
            }
            return {cpu::scatter_kernel(inputs[0], dim, inputs[1], inputs[2])};
        }
        else if (op_name == "masked_select") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("masked_select operation requires exactly 2 inputs");
            }
            return {cpu::masked_select_kernel(inputs[0], inputs[1])};
        }
        else if (op_name == "masked_fill") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("masked_fill operation requires exactly 2 inputs");
            }
            if (!attrs.contains("value")) {
                throw std::invalid_argument("masked_fill operation requires 'value' attribute");
            }
            float value = std::stof(attrs.at("value"));
            return {cpu::masked_fill_kernel(inputs[0], inputs[1], value)};
        }
        else if (op_name == "where") {
            if (inputs.size() != 3) {
                throw std::invalid_argument("where operation requires exactly 3 inputs");
            }
            return {cpu::where_kernel(inputs[0], inputs[1], inputs[2])};
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
        else if (op_name == "fused_linear_relu") {
            if (inputs.size() < 2) {
                throw std::invalid_argument("fused_linear_relu operation requires at least 2 inputs");
            }
            const Tensor* bias = (inputs.size() >= 3) ? &inputs[2] : nullptr;
            return {cpu::fused_linear_relu_kernel(inputs[0], inputs[1], bias)};
        }
        else if (op_name == "fused_conv2d_relu") {
            if (inputs.size() < 2) {
                throw std::invalid_argument("fused_conv2d_relu operation requires at least 2 inputs");
            }
            const Tensor* bias = (inputs.size() >= 3) ? &inputs[2] : nullptr;
            int64_t stride = 1;
            int64_t padding = 0;
            if (attrs.contains("stride")) {
                stride = std::stoll(attrs.at("stride"));
            }
            if (attrs.contains("padding")) {
                padding = std::stoll(attrs.at("padding"));
            }
            return {cpu::fused_conv2d_relu_kernel(inputs[0], inputs[1], bias, stride, padding)};
        }
        else if (op_name == "fused_batchnorm_relu") {
            if (inputs.size() != 5) {
                throw std::invalid_argument("fused_batchnorm_relu operation requires exactly 5 inputs");
            }
            float eps = 1e-5f;
            if (attrs.contains("eps")) {
                eps = std::stof(attrs.at("eps"));
            }
            return {cpu::fused_batchnorm_relu_kernel(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], eps)};
        }
        else if (op_name == "fused_softmax_cross_entropy") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("fused_softmax_cross_entropy operation requires exactly 2 inputs");
            }
            std::string reduction = "mean";
            if (attrs.contains("reduction")) {
                reduction = attrs.at("reduction");
            }
            return {cpu::fused_softmax_cross_entropy_kernel(inputs[0], inputs[1], reduction)};
        }
        else if (op_name == "fused_add_relu") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("fused_add_relu operation requires exactly 2 inputs");
            }
            return {cpu::fused_add_relu_kernel(inputs[0], inputs[1])};
        }
        else if (op_name == "fused_gelu") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("fused_gelu operation requires exactly 1 input");
            }
            return {cpu::fused_gelu_kernel(inputs[0])};
        }
        else if (op_name == "fused_layer_norm") {
            if (inputs.size() != 3) {
                throw std::invalid_argument("fused_layer_norm operation requires exactly 3 inputs");
            }
            // Parse normalized_shape from comma-separated string
            std::vector<int64_t> normalized_shape;
            if (attrs.contains("normalized_shape")) {
                std::string shape_str = attrs.at("normalized_shape");
                size_t pos = 0;
                while (pos < shape_str.size()) {
                    size_t comma = shape_str.find(',', pos);
                    if (comma == std::string::npos) {
                        normalized_shape.push_back(std::stoll(shape_str.substr(pos)));
                        break;
                    }
                    normalized_shape.push_back(std::stoll(shape_str.substr(pos, comma - pos)));
                    pos = comma + 1;
                }
            }
            float eps = 1e-5f;
            if (attrs.contains("eps")) {
                eps = std::stof(attrs.at("eps"));
            }
            return {cpu::fused_layer_norm_kernel(inputs[0], normalized_shape, inputs[1], inputs[2], eps)};
        }
        else if (op_name == "eq") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("eq operation requires exactly 2 inputs");
            }
            return {cpu::eq_kernel(inputs[0], inputs[1])};
        }
        else if (op_name == "ne") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("ne operation requires exactly 2 inputs");
            }
            return {cpu::ne_kernel(inputs[0], inputs[1])};
        }
        else if (op_name == "lt") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("lt operation requires exactly 2 inputs");
            }
            return {cpu::lt_kernel(inputs[0], inputs[1])};
        }
        else if (op_name == "le") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("le operation requires exactly 2 inputs");
            }
            return {cpu::le_kernel(inputs[0], inputs[1])};
        }
        else if (op_name == "gt") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("gt operation requires exactly 2 inputs");
            }
            return {cpu::gt_kernel(inputs[0], inputs[1])};
        }
        else if (op_name == "ge") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("ge operation requires exactly 2 inputs");
            }
            return {cpu::ge_kernel(inputs[0], inputs[1])};
        }
        else if (op_name == "dot") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("dot operation requires exactly 2 inputs");
            }
            return {cpu::dot_kernel(inputs[0], inputs[1])};
        }
        // Trigonometric operations
        else if (op_name == "sin") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("sin operation requires exactly 1 input");
            }
            return {cpu::sin_kernel(inputs[0])};
        }
        else if (op_name == "cos") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("cos operation requires exactly 1 input");
            }
            return {cpu::cos_kernel(inputs[0])};
        }
        else if (op_name == "tan") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("tan operation requires exactly 1 input");
            }
            return {cpu::tan_kernel(inputs[0])};
        }
        else if (op_name == "asin") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("asin operation requires exactly 1 input");
            }
            return {cpu::asin_kernel(inputs[0])};
        }
        else if (op_name == "acos") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("acos operation requires exactly 1 input");
            }
            return {cpu::acos_kernel(inputs[0])};
        }
        else if (op_name == "atan") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("atan operation requires exactly 1 input");
            }
            return {cpu::atan_kernel(inputs[0])};
        }
        else if (op_name == "sinh") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("sinh operation requires exactly 1 input");
            }
            return {cpu::sinh_kernel(inputs[0])};
        }
        else if (op_name == "cosh") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("cosh operation requires exactly 1 input");
            }
            return {cpu::cosh_kernel(inputs[0])};
        }
        // Rounding operations
        else if (op_name == "round") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("round operation requires exactly 1 input");
            }
            return {cpu::round_kernel(inputs[0])};
        }
        else if (op_name == "floor") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("floor operation requires exactly 1 input");
            }
            return {cpu::floor_kernel(inputs[0])};
        }
        else if (op_name == "ceil") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("ceil operation requires exactly 1 input");
            }
            return {cpu::ceil_kernel(inputs[0])};
        }
        // Other math operations
        else if (op_name == "reciprocal") {
            if (inputs.size() != 1) {
                throw std::invalid_argument("reciprocal operation requires exactly 1 input");
            }
            return {cpu::reciprocal_kernel(inputs[0])};
        }
        // In-place operations
        else if (op_name == "add_inplace") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("add_inplace operation requires exactly 2 inputs");
            }
            Tensor a_copy = inputs[0].clone();
            cpu::add_inplace_kernel(a_copy, inputs[1]);
            return {a_copy};
        }
        else if (op_name == "mul_inplace") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("mul_inplace operation requires exactly 2 inputs");
            }
            Tensor a_copy = inputs[0].clone();
            cpu::mul_inplace_kernel(a_copy, inputs[1]);
            return {a_copy};
        }
        else if (op_name == "sub_inplace") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("sub_inplace operation requires exactly 2 inputs");
            }
            Tensor a_copy = inputs[0].clone();
            cpu::sub_inplace_kernel(a_copy, inputs[1]);
            return {a_copy};
        }
        else if (op_name == "div_inplace") {
            if (inputs.size() != 2) {
                throw std::invalid_argument("div_inplace operation requires exactly 2 inputs");
            }
            Tensor a_copy = inputs[0].clone();
            cpu::div_inplace_kernel(a_copy, inputs[1]);
            return {a_copy};
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
