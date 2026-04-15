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
// LeCun Initialization
// ============================================================================

auto lecun_uniform_(Tensor& tensor) -> Tensor& {
    auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(tensor);
    double bound = std::sqrt(3.0 / static_cast<double>(fan_in));

    std::uniform_real_distribution<double> dist(-bound, bound);
    fill_tensor(tensor, [&]() { return dist(get_rng()); });
    return tensor;
}

auto lecun_normal_(Tensor& tensor) -> Tensor& {
    auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(tensor);
    double std_val = std::sqrt(1.0 / static_cast<double>(fan_in));

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

// ============================================================================
// Truncated Normal Initialization
// ============================================================================

auto trunc_normal_(Tensor& tensor,
                   double mean, double std,
                   double a, double b) -> Tensor& {
    // Normalize bounds to standard normal
    double l = (a - mean) / std;
    double u = (b - mean) / std;

    // CDF of standard normal (using erfc for numerical stability)
    auto norm_cdf = [](double x) -> double {
        return 0.5 * std::erfc(-x * M_SQRT1_2);
    };

    // Inverse CDF of standard normal (Beasley-Springer-Moro approximation)
    auto norm_icdf = [](double p) -> double {
        // Rational approximation from Peter Acklam
        static constexpr double a1 = -3.969683028665376e+01;
        static constexpr double a2 =  2.209460984245205e+02;
        static constexpr double a3 = -2.759285104469687e+02;
        static constexpr double a4 =  1.383577518672690e+02;
        static constexpr double a5 = -3.066479806614716e+01;
        static constexpr double a6 =  2.506628277459239e+00;

        static constexpr double b1 = -5.447609879822406e+01;
        static constexpr double b2 =  1.615858368580409e+02;
        static constexpr double b3 = -1.556989798598866e+02;
        static constexpr double b4 =  6.680131188771972e+01;
        static constexpr double b5 = -1.328068155288572e+01;

        static constexpr double c1 = -7.784894002430293e-03;
        static constexpr double c2 = -3.223964580411365e-01;
        static constexpr double c3 = -2.400758277161838e+00;
        static constexpr double c4 = -2.549732539343734e+00;
        static constexpr double c5 =  4.374664141464968e+00;
        static constexpr double c6 =  2.938163982698783e+00;

        static constexpr double d1 =  7.784695709041462e-03;
        static constexpr double d2 =  3.224671290700398e-01;
        static constexpr double d3 =  2.445134137142996e+00;
        static constexpr double d4 =  3.754408661907416e+00;

        static constexpr double p_low  = 0.02425;
        static constexpr double p_high = 1.0 - p_low;

        double q, r;
        if (p < p_low) {
            q = std::sqrt(-2.0 * std::log(p));
            return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
                   ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
        } else if (p <= p_high) {
            q = p - 0.5;
            r = q * q;
            return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q /
                   (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0);
        } else {
            q = std::sqrt(-2.0 * std::log(1.0 - p));
            return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
                    ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
        }
    };

    double cdf_l = norm_cdf(l);
    double cdf_u = norm_cdf(u);

    std::uniform_real_distribution<double> dist(cdf_l, cdf_u);
    fill_tensor(tensor, [&]() {
        double u_val = dist(get_rng());
        // Clamp to avoid numerical issues at tails
        u_val = std::clamp(u_val, cdf_l + 1e-10, cdf_u - 1e-10);
        return norm_icdf(u_val) * std + mean;
    });

    return tensor;
}

// ============================================================================
// Dirac Initialization
// ============================================================================

auto dirac_(Tensor& tensor, int64_t groups) -> Tensor& {
    auto shape = tensor.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 3 || ndim > 5) {
        throw std::invalid_argument(
            "dirac_: tensor must be 3D (Conv1d), 4D (Conv2d), or 5D (Conv3d), got " +
            std::to_string(ndim) + "D");
    }

    int64_t out_channels = shape[0];
    int64_t in_channels_per_group = shape[1];
    int64_t min_dim = std::min(out_channels, in_channels_per_group * groups);

    // Zero out first
    zeros_(tensor);

    // Work on CPU
    bool needs_device_transfer = (tensor.device().type != Device::Type::CPU);
    Tensor work_tensor = needs_device_transfer ? tensor.to(Device::cpu()) : tensor;

    // Compute center indices for spatial dimensions
    std::vector<int64_t> center;
    for (int64_t d = 2; d < ndim; ++d) {
        center.push_back(shape[d] / 2);
    }

    // Set weight[i, i/groups, center...] = 1 for i in [0, min_dim)
    if (work_tensor.dtype() == DType::Float32) {
        float* data = work_tensor.data<float>();
        for (int64_t i = 0; i < min_dim; ++i) {
            int64_t c_in = i % in_channels_per_group;
            // Compute flat index
            int64_t idx = i;
            for (int64_t d = 1; d < ndim; ++d) {
                idx *= shape[d];
                if (d == 1) idx += c_in;
                else idx += center[d - 2];
            }
            data[idx] = 1.0f;
        }
    } else if (work_tensor.dtype() == DType::Float64) {
        double* data = work_tensor.data<double>();
        for (int64_t i = 0; i < min_dim; ++i) {
            int64_t c_in = i % in_channels_per_group;
            int64_t idx = i;
            for (int64_t d = 1; d < ndim; ++d) {
                idx *= shape[d];
                if (d == 1) idx += c_in;
                else idx += center[d - 2];
            }
            data[idx] = 1.0;
        }
    } else {
        throw std::runtime_error("dirac_: unsupported dtype " +
            std::string(dtype_name(work_tensor.dtype())));
    }

    if (needs_device_transfer) {
        tensor = work_tensor.to(tensor.device());
    }

    return tensor;
}

// ============================================================================
// Sparse Initialization
// ============================================================================

auto sparse_(Tensor& tensor, double sparsity, double std) -> Tensor& {
    auto shape = tensor.shape();
    if (shape.size() != 2) {
        throw std::invalid_argument(
            "sparse_: tensor must be 2D, got " +
            std::to_string(shape.size()) + "D");
    }

    if (sparsity < 0.0 || sparsity >= 1.0) {
        throw std::invalid_argument(
            "sparse_: sparsity must be in [0, 1), got " +
            std::to_string(sparsity));
    }

    int64_t rows = shape[0];
    int64_t cols = shape[1];
    int64_t num_zeros = static_cast<int64_t>(std::ceil(sparsity * rows));

    // Fill with normal distribution first
    normal_(tensor, 0.0, std);

    // Work on CPU for the zeroing
    bool needs_device_transfer = (tensor.device().type != Device::Type::CPU);
    Tensor work_tensor = needs_device_transfer ? tensor.to(Device::cpu()) : tensor;

    // For each column, randomly zero out num_zeros rows
    std::vector<int64_t> row_indices(rows);
    std::iota(row_indices.begin(), row_indices.end(), 0);

    if (work_tensor.dtype() == DType::Float32) {
        float* data = work_tensor.data<float>();
        for (int64_t c = 0; c < cols; ++c) {
            std::shuffle(row_indices.begin(), row_indices.end(), get_rng());
            for (int64_t i = 0; i < num_zeros; ++i) {
                data[row_indices[i] * cols + c] = 0.0f;
            }
        }
    } else if (work_tensor.dtype() == DType::Float64) {
        double* data = work_tensor.data<double>();
        for (int64_t c = 0; c < cols; ++c) {
            std::shuffle(row_indices.begin(), row_indices.end(), get_rng());
            for (int64_t i = 0; i < num_zeros; ++i) {
                data[row_indices[i] * cols + c] = 0.0;
            }
        }
    } else {
        throw std::runtime_error("sparse_: unsupported dtype " +
            std::string(dtype_name(work_tensor.dtype())));
    }

    if (needs_device_transfer) {
        tensor = work_tensor.to(tensor.device());
    }

    return tensor;
}

} // namespace tenzor::nn::init
