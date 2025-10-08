#pragma once

#include "../core/tensor.hpp"

namespace tenzor {

// Arithmetic operations
auto add(const Tensor& a, const Tensor& b) -> Tensor;
auto sub(const Tensor& a, const Tensor& b) -> Tensor;
auto mul(const Tensor& a, const Tensor& b) -> Tensor;
auto div(const Tensor& a, const Tensor& b) -> Tensor;

// Matrix operations
auto matmul(const Tensor& a, const Tensor& b) -> Tensor;
auto dot(const Tensor& a, const Tensor& b) -> Tensor;

// Power and exponential
auto pow(const Tensor& input, float exponent) -> Tensor;
auto exp(const Tensor& input) -> Tensor;
auto log(const Tensor& input) -> Tensor;
auto sqrt(const Tensor& input) -> Tensor;

// Trigonometric
auto sin(const Tensor& input) -> Tensor;
auto cos(const Tensor& input) -> Tensor;
auto tan(const Tensor& input) -> Tensor;

// Hyperbolic
auto sinh(const Tensor& input) -> Tensor;
auto cosh(const Tensor& input) -> Tensor;
auto tanh(const Tensor& input) -> Tensor;

// Element-wise operations
auto abs(const Tensor& input) -> Tensor;
auto neg(const Tensor& input) -> Tensor;
auto reciprocal(const Tensor& input) -> Tensor;
auto sign(const Tensor& input) -> Tensor;

// Rounding
auto floor(const Tensor& input) -> Tensor;
auto ceil(const Tensor& input) -> Tensor;
auto round(const Tensor& input) -> Tensor;

// Clamping
auto clamp(const Tensor& input, float min, float max) -> Tensor;
auto clamp_min(const Tensor& input, float min) -> Tensor;
auto clamp_max(const Tensor& input, float max) -> Tensor;

} // namespace tenzor
