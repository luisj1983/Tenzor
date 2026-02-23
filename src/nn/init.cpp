/**
 * @file init.cpp
 * @brief Implementation of weight initialization utilities
 */

#include "tenzor/nn/init.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <numeric>
#include <cstring>

namespace tenzor::nn::init {

// ============================================================================
// Fan Calculation
// ============================================================================

auto calculate_fan_in_and_fan_out(const Tensor& tensor) -> std::pair<int64_t, int64_t> {
    auto shape = tensor.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 1) {
        throw std::invalid_argument(
            "calculate_fan_in_and_fan_out: tensor must have at least 1 dimension, "
            "got " + std::to_string(ndim) + "D");
    }

    if (ndim == 1) {
        // 1D: fan_in = fan_out = size
        return {shape[0], shape[0]};
    }

    int64_t num_output = shape[0];
    int64_t num_input = shape[1];

    // Receptive field size for conv layers (product of kernel dimensions)
    int64_t receptive_field_size = 1;
    for (int64_t i = 2; i < ndim; ++i) {
        receptive_field_size *= shape[i];
    }

    int64_t fan_in = num_input * receptive_field_size;
    int64_t fan_out = num_output * receptive_field_size;

    return {fan_in, fan_out};
}

auto calculate_gain(const std::string& nonlinearity, double param) -> double {
    if (nonlinearity == "linear" || nonlinearity == "conv1d" ||
        nonlinearity == "conv2d" || nonlinearity == "conv3d" ||
        nonlinearity == "conv_transpose1d" || nonlinearity == "conv_transpose2d" ||
        nonlinearity == "conv_transpose3d" || nonlinearity == "sigmoid") {
        return 1.0;
    } else if (nonlinearity == "tanh") {
        return 5.0 / 3.0;
    } else if (nonlinearity == "relu") {
        return std::sqrt(2.0);
    } else if (nonlinearity == "leaky_relu") {
        return std::sqrt(2.0 / (1.0 + param * param));
    } else if (nonlinearity == "selu") {
        return 3.0 / 4.0;  // Empirically derived
    }

    throw std::invalid_argument(
        "calculate_gain: unsupported nonlinearity '" + nonlinearity + "'");
}

// ============================================================================
// Helper: Write values into tensor
// ============================================================================

namespace {

// Fill tensor with values from a generator function
template<typename Generator>
void fill_tensor(Tensor& tensor, Generator gen) {
    // Always work on CPU, then copy to device if needed
    bool needs_device_transfer = (tensor.device().type != Device::Type::CPU);
    Tensor work_tensor = needs_device_transfer ? tensor.to(Device::cpu()) : tensor;

    int64_t numel = work_tensor.numel();

    if (work_tensor.dtype() == DType::Float32) {
        float* data = work_tensor.data<float>();
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = static_cast<float>(gen());
        }
    } else if (work_tensor.dtype() == DType::Float64) {
        double* data = work_tensor.data<double>();
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = gen();
        }
    } else if (work_tensor.dtype() == DType::Float16 || work_tensor.dtype() == DType::BFloat16) {
        // For half-precision types, fill via a Float32 temporary and convert
        Tensor temp({work_tensor.shape().begin(), work_tensor.shape().end()},
                    DType::Float32, Device::cpu());
        float* temp_data = temp.data<float>();
        for (int64_t i = 0; i < numel; ++i) {
            temp_data[i] = static_cast<float>(gen());
        }
        // Convert and copy back
        Tensor converted = temp.to(work_tensor.dtype());
        std::memcpy(work_tensor.data_ptr(), converted.data_ptr(),
                    numel * dtype_size(work_tensor.dtype()));
    } else if (work_tensor.dtype() == DType::Int32) {
        int32_t* data = work_tensor.data<int32_t>();
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = static_cast<int32_t>(gen());
        }
    } else if (work_tensor.dtype() == DType::Int64) {
        int64_t* data = work_tensor.data<int64_t>();
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = static_cast<int64_t>(gen());
        }
    } else {
        throw std::runtime_error("fill_tensor: unsupported dtype " +
            std::string(dtype_name(work_tensor.dtype())));
    }

    if (needs_device_transfer) {
        // Copy initialized CPU tensor back to original device
        Tensor result = work_tensor.to(tensor.device());
        // Copy the data back into the original tensor's storage
        int64_t byte_size = numel * dtype_size(tensor.dtype());
        if (tensor.device().type == Device::Type::CPU) {
            std::memcpy(tensor.data_ptr(), result.data_ptr(), byte_size);
        } else {
            // For GPU: replace the tensor data
            tensor = result;
        }
    }
}

// Thread-local RNG for initialization
auto get_rng() -> std::mt19937& {
    static thread_local std::mt19937 rng{std::random_device{}()};
    return rng;
}

} // anonymous namespace

// ============================================================================
// Xavier Initialization
// ============================================================================

auto xavier_uniform_(Tensor& tensor, double gain) -> Tensor& {
    auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(tensor);
    double a = gain * std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));

    std::uniform_real_distribution<double> dist(-a, a);
    fill_tensor(tensor, [&]() { return dist(get_rng()); });
    return tensor;
}

auto xavier_normal_(Tensor& tensor, double gain) -> Tensor& {
    auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(tensor);
    double std_val = gain * std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));

    std::normal_distribution<double> dist(0.0, std_val);
    fill_tensor(tensor, [&]() { return dist(get_rng()); });
    return tensor;
}

// ============================================================================
// Kaiming Initialization
// ============================================================================

auto kaiming_uniform_(Tensor& tensor, double a, FanMode mode,
                      const std::string& nonlinearity) -> Tensor& {
    auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(tensor);
    int64_t fan = (mode == FanMode::FanIn) ? fan_in : fan_out;

    double gain = calculate_gain(nonlinearity, a);
    double bound = gain * std::sqrt(3.0 / static_cast<double>(fan));

    std::uniform_real_distribution<double> dist(-bound, bound);
    fill_tensor(tensor, [&]() { return dist(get_rng()); });
    return tensor;
}

auto kaiming_normal_(Tensor& tensor, double a, FanMode mode,
                     const std::string& nonlinearity) -> Tensor& {
    auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(tensor);
    int64_t fan = (mode == FanMode::FanIn) ? fan_in : fan_out;

    double gain = calculate_gain(nonlinearity, a);
    double std_val = gain / std::sqrt(static_cast<double>(fan));

    std::normal_distribution<double> dist(0.0, std_val);
    fill_tensor(tensor, [&]() { return dist(get_rng()); });
    return tensor;
}

// ============================================================================
// Orthogonal Initialization
// ============================================================================

auto orthogonal_(Tensor& tensor, double gain) -> Tensor& {
    auto shape = tensor.shape();
    if (shape.size() < 2) {
        throw std::invalid_argument(
            "orthogonal_: tensor must have at least 2 dimensions, got " +
            std::to_string(shape.size()) + "D");
    }

    int64_t rows = shape[0];
    int64_t cols = 1;
    for (size_t i = 1; i < shape.size(); ++i) {
        cols *= shape[i];
    }

    // Generate random matrix
    bool needs_device_transfer = (tensor.device().type != Device::Type::CPU);
    Tensor flat = tenzor::randn({rows, cols}, DType::Float64, Device::cpu());

    // QR decomposition via modified Gram-Schmidt orthogonalization
    // This is simpler than full SVD and sufficient for initialization
    int64_t n = std::min(rows, cols);
    auto* data = flat.data<double>();

    for (int64_t j = 0; j < n; ++j) {
        // Orthogonalize column j against all previous columns
        for (int64_t i = 0; i < j; ++i) {
            // Compute dot product of column i and column j
            double dot = 0.0;
            for (int64_t r = 0; r < rows; ++r) {
                dot += data[r * cols + i] * data[r * cols + j];
            }
            // Subtract projection
            for (int64_t r = 0; r < rows; ++r) {
                data[r * cols + j] -= dot * data[r * cols + i];
            }
        }
        // Normalize column j
        double norm = 0.0;
        for (int64_t r = 0; r < rows; ++r) {
            norm += data[r * cols + j] * data[r * cols + j];
        }
        norm = std::sqrt(norm);
        if (norm > 1e-10) {
            for (int64_t r = 0; r < rows; ++r) {
                data[r * cols + j] /= norm;
            }
        }
    }

    // Apply gain
    if (gain != 1.0) {
        for (int64_t i = 0; i < rows * cols; ++i) {
            data[i] *= gain;
        }
    }

    // Convert to target dtype and reshape
    Tensor result = flat.to(tensor.dtype());
    result = tenzor::reshape(result, std::vector<int64_t>(shape.begin(), shape.end()));

    if (needs_device_transfer) {
        result = result.to(tensor.device());
    }

    // Copy into original tensor
    if (tensor.device().type == Device::Type::CPU) {
        int64_t byte_size = tensor.numel() * dtype_size(tensor.dtype());
        std::memcpy(tensor.data_ptr(), result.data_ptr(), byte_size);
    } else {
        tensor = result;
    }

    return tensor;
}

// ============================================================================
// Simple Initialization
// ============================================================================

auto uniform_(Tensor& tensor, double low, double high) -> Tensor& {
    std::uniform_real_distribution<double> dist(low, high);
    fill_tensor(tensor, [&]() { return dist(get_rng()); });
    return tensor;
}

auto normal_(Tensor& tensor, double mean, double std) -> Tensor& {
    std::normal_distribution<double> dist(mean, std);
    fill_tensor(tensor, [&]() { return dist(get_rng()); });
    return tensor;
}

auto constant_(Tensor& tensor, double value) -> Tensor& {
    fill_tensor(tensor, [value]() { return value; });
    return tensor;
}

auto zeros_(Tensor& tensor) -> Tensor& {
    if (tensor.device().type == Device::Type::CPU) {
        std::memset(tensor.data_ptr(), 0,
                    tensor.numel() * dtype_size(tensor.dtype()));
    } else {
        constant_(tensor, 0.0);
    }
    return tensor;
}

auto ones_(Tensor& tensor) -> Tensor& {
    return constant_(tensor, 1.0);
}

} // namespace tenzor::nn::init
