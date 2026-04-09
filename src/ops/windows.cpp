/**
 * @file windows.cpp
 * @brief Implementation of window functions for spectral analysis
 */

#include "tenzor/ops/windows.hpp"
#include "tenzor/ops/creation.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tenzor {

namespace {

// Helper: compute window on CPU in Float64 (for precision), then cast to
// the requested dtype and device. Float64 is used regardless of the target
// so that:
//   - Float64 callers get full precision (previously they received a
//     Float32-computed result cast up, which lost half its mantissa bits
//     and failed tight symmetry tests),
//   - Float32/Float16/BFloat16 callers still see the correctly rounded
//     nearest-representable value for each sample.
auto make_window(int64_t size, bool periodic,
                 std::function<double(int64_t, int64_t)> fn,
                 DType dtype, Device device) -> Tensor {
    int64_t N = periodic ? size : (size > 1 ? size - 1 : 1);
    auto result = Tensor({size}, DType::Float64, Device::cpu());
    auto* data = result.data<double>();
    for (int64_t i = 0; i < size; ++i) {
        data[i] = fn(i, N);
    }
    if (dtype != DType::Float64) {
        result = result.to(dtype);
    }
    if (device != Device::cpu()) {
        result = result.to(device);
    }
    return result;
}

} // anonymous namespace

auto hann_window(int64_t size, bool periodic,
                 DType dtype, Device device) -> Tensor {
    return make_window(size, periodic, [](int64_t i, int64_t N) -> double {
        return 0.5 * (1.0 - std::cos(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(N)));
    }, dtype, device);
}

auto hamming_window(int64_t size, bool periodic,
                    double alpha, double beta,
                    DType dtype, Device device) -> Tensor {
    return make_window(size, periodic, [alpha, beta](int64_t i, int64_t N) -> double {
        return alpha - beta * std::cos(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(N));
    }, dtype, device);
}

auto blackman_window(int64_t size, bool periodic,
                     DType dtype, Device device) -> Tensor {
    return make_window(size, periodic, [](int64_t i, int64_t N) -> double {
        double x = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(N);
        return 0.42 - 0.5 * std::cos(x) + 0.08 * std::cos(2.0 * x);
    }, dtype, device);
}

auto bartlett_window(int64_t size, bool periodic,
                     DType dtype, Device device) -> Tensor {
    return make_window(size, periodic, [](int64_t i, int64_t N) -> double {
        double half_N = static_cast<double>(N) / 2.0;
        return 1.0 - std::abs((static_cast<double>(i) - half_N) / half_N);
    }, dtype, device);
}

auto kaiser_window(int64_t size, bool periodic, double beta,
                   DType dtype, Device device) -> Tensor {
    return make_window(size, periodic, [beta](int64_t i, int64_t N) -> double {
        double alpha = static_cast<double>(N) / 2.0;
        double r = (static_cast<double>(i) - alpha) / alpha;
        double x = beta * std::sqrt(std::max(0.0, 1.0 - r * r));
        // Approximate I0(x) using the series expansion
        double i0_x = 1.0;
        double term = 1.0;
        for (int k = 1; k <= 20; ++k) {
            term *= (x / (2.0 * k)) * (x / (2.0 * k));
            i0_x += term;
        }
        // I0(beta) for normalization
        double i0_b = 1.0;
        term = 1.0;
        for (int k = 1; k <= 20; ++k) {
            term *= (beta / (2.0 * k)) * (beta / (2.0 * k));
            i0_b += term;
        }
        return i0_x / i0_b;
    }, dtype, device);
}

} // namespace tenzor
