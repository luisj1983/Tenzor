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

// Helper: compute window on CPU and move to target device
auto make_window(int64_t size, bool periodic,
                 std::function<float(int64_t, int64_t)> fn,
                 DType dtype, Device device) -> Tensor {
    int64_t N = periodic ? size : (size > 1 ? size - 1 : 1);
    auto result = Tensor({size}, DType::Float32, Device::cpu());
    auto* data = result.data<float>();
    for (int64_t i = 0; i < size; ++i) {
        data[i] = fn(i, N);
    }
    if (dtype != DType::Float32) {
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
    return make_window(size, periodic, [](int64_t i, int64_t N) -> float {
        return 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / N));
    }, dtype, device);
}

auto hamming_window(int64_t size, bool periodic,
                    double alpha, double beta,
                    DType dtype, Device device) -> Tensor {
    float a = static_cast<float>(alpha);
    float b = static_cast<float>(beta);
    return make_window(size, periodic, [a, b](int64_t i, int64_t N) -> float {
        return a - b * std::cos(2.0f * static_cast<float>(M_PI) * i / N);
    }, dtype, device);
}

auto blackman_window(int64_t size, bool periodic,
                     DType dtype, Device device) -> Tensor {
    return make_window(size, periodic, [](int64_t i, int64_t N) -> float {
        float pi = static_cast<float>(M_PI);
        return 0.42f - 0.5f * std::cos(2.0f * pi * i / N)
                      + 0.08f * std::cos(4.0f * pi * i / N);
    }, dtype, device);
}

auto bartlett_window(int64_t size, bool periodic,
                     DType dtype, Device device) -> Tensor {
    return make_window(size, periodic, [](int64_t i, int64_t N) -> float {
        float half_N = static_cast<float>(N) / 2.0f;
        return 1.0f - std::abs((static_cast<float>(i) - half_N) / half_N);
    }, dtype, device);
}

auto kaiser_window(int64_t size, bool periodic, double beta,
                   DType dtype, Device device) -> Tensor {
    float b = static_cast<float>(beta);
    return make_window(size, periodic, [b](int64_t i, int64_t N) -> float {
        float alpha = static_cast<float>(N) / 2.0f;
        float r = (static_cast<float>(i) - alpha) / alpha;
        float x = b * std::sqrt(std::max(0.0f, 1.0f - r * r));
        // Approximate I0(x) using the series expansion
        float i0_x = 1.0f;
        float term = 1.0f;
        for (int k = 1; k <= 20; ++k) {
            term *= (x / (2.0f * k)) * (x / (2.0f * k));
            i0_x += term;
        }
        // I0(beta) for normalization
        float i0_b = 1.0f;
        term = 1.0f;
        for (int k = 1; k <= 20; ++k) {
            term *= (b / (2.0f * k)) * (b / (2.0f * k));
            i0_b += term;
        }
        return i0_x / i0_b;
    }, dtype, device);
}

} // namespace tenzor
