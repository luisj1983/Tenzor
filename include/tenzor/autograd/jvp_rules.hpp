/**
 * @file jvp_rules.hpp
 * @brief JVP (Jacobian-Vector Product) rules for forward-mode AD
 *
 * Each rule takes dual inputs and returns a dual output, propagating
 * tangent vectors through the operation using the chain rule.
 */

#pragma once

#include "dual.hpp"
#include "../ops/op_id.hpp"
#include "../backend/op_attributes.hpp"
#include "../backend/backend.hpp"  // for `using OpAttributes = NewOpAttributes;`
#include <optional>
#include <vector>
#include <cstdint>
#include <span>

namespace tenzor {

/// @name Arithmetic JVP Rules
/// @{
auto jvp_add(const DualTensor& a, const DualTensor& b) -> DualTensor;
auto jvp_sub(const DualTensor& a, const DualTensor& b) -> DualTensor;
auto jvp_mul(const DualTensor& a, const DualTensor& b) -> DualTensor;
auto jvp_div(const DualTensor& a, const DualTensor& b) -> DualTensor;
auto jvp_neg(const DualTensor& a) -> DualTensor;
/// @}

/// @name Matrix JVP Rules
/// @{
auto jvp_matmul(const DualTensor& a, const DualTensor& b) -> DualTensor;
/// @}

/// @name Activation JVP Rules
/// @{
auto jvp_relu(const DualTensor& x) -> DualTensor;
auto jvp_sigmoid(const DualTensor& x) -> DualTensor;
auto jvp_tanh(const DualTensor& x) -> DualTensor;
auto jvp_gelu(const DualTensor& x) -> DualTensor;
auto jvp_leaky_relu(const DualTensor& x, float negative_slope = 0.01f) -> DualTensor;
auto jvp_elu(const DualTensor& x, float alpha = 1.0f) -> DualTensor;
auto jvp_selu(const DualTensor& x) -> DualTensor;
auto jvp_softplus(const DualTensor& x, float beta = 1.0f) -> DualTensor;
auto jvp_mish(const DualTensor& x) -> DualTensor;
/// @}

/// @name Math JVP Rules
/// @{
auto jvp_exp(const DualTensor& x) -> DualTensor;
auto jvp_log(const DualTensor& x) -> DualTensor;
auto jvp_sqrt(const DualTensor& x) -> DualTensor;
auto jvp_pow(const DualTensor& x, double exponent) -> DualTensor;
auto jvp_sin(const DualTensor& x) -> DualTensor;
auto jvp_cos(const DualTensor& x) -> DualTensor;
auto jvp_abs(const DualTensor& x) -> DualTensor;
auto jvp_tan(const DualTensor& x) -> DualTensor;
auto jvp_asin(const DualTensor& x) -> DualTensor;
auto jvp_acos(const DualTensor& x) -> DualTensor;
auto jvp_atan(const DualTensor& x) -> DualTensor;
auto jvp_sinh(const DualTensor& x) -> DualTensor;
auto jvp_cosh(const DualTensor& x) -> DualTensor;
auto jvp_log2(const DualTensor& x) -> DualTensor;
auto jvp_log10(const DualTensor& x) -> DualTensor;
auto jvp_log1p(const DualTensor& x) -> DualTensor;
auto jvp_exp2(const DualTensor& x) -> DualTensor;
auto jvp_expm1(const DualTensor& x) -> DualTensor;
auto jvp_reciprocal(const DualTensor& x) -> DualTensor;
auto jvp_sign(const DualTensor& x) -> DualTensor;
auto jvp_erf(const DualTensor& x) -> DualTensor;
auto jvp_erfc(const DualTensor& x) -> DualTensor;
auto jvp_clamp(const DualTensor& x, double min_val, double max_val) -> DualTensor;
/// @}

/// @name Reduction JVP Rules
/// @{
auto jvp_sum(const DualTensor& x,
             std::optional<int64_t> dim = std::nullopt,
             bool keepdim = false) -> DualTensor;
auto jvp_mean(const DualTensor& x,
              std::optional<int64_t> dim = std::nullopt,
              bool keepdim = false) -> DualTensor;
/// @}

/// @name Shape JVP Rules
/// @{
auto jvp_reshape(const DualTensor& x, std::vector<int64_t> shape) -> DualTensor;
auto jvp_transpose(const DualTensor& x, int64_t dim0, int64_t dim1) -> DualTensor;
auto jvp_permute(const DualTensor& x, std::vector<int64_t> dims) -> DualTensor;
auto jvp_squeeze(const DualTensor& x, std::optional<int64_t> dim = std::nullopt) -> DualTensor;
auto jvp_unsqueeze(const DualTensor& x, int64_t dim) -> DualTensor;
auto jvp_expand(const DualTensor& x, std::vector<int64_t> shape) -> DualTensor;
auto jvp_flatten(const DualTensor& x, int64_t start_dim = 0, int64_t end_dim = -1) -> DualTensor;
/// @}

/// @name Tensor Combination JVP Rules
/// @{
auto jvp_cat(std::span<const DualTensor> tensors, int64_t dim = 0) -> DualTensor;
auto jvp_stack(std::span<const DualTensor> tensors, int64_t dim = 0) -> DualTensor;
/// @}

/// @name Softmax JVP Rules
/// @{
auto jvp_softmax(const DualTensor& x, int64_t dim) -> DualTensor;
auto jvp_log_softmax(const DualTensor& x, int64_t dim) -> DualTensor;
/// @}

/// @name Linear Algebra JVP Rules
/// @{
auto jvp_linear(const DualTensor& input, const DualTensor& weight, const DualTensor& bias) -> DualTensor;
auto jvp_bmm(const DualTensor& a, const DualTensor& b) -> DualTensor;
auto jvp_inv(const DualTensor& a) -> DualTensor;
auto jvp_solve(const DualTensor& a, const DualTensor& b) -> DualTensor;
auto jvp_cholesky(const DualTensor& a, bool upper = false) -> DualTensor;
auto jvp_trace(const DualTensor& x) -> DualTensor;
auto jvp_det(const DualTensor& a) -> DualTensor;
/// @}

/// @name Conv JVP Rules
/// @{
/**
 * Convolution forward-mode JVP. Works for Conv{1,2,3}d and ConvTranspose{2,3}d
 * via the supplied OpId. JVP formula (linearity of conv in (x, w) + bias add):
 *     y  = conv(x, w) + b
 *     dy = conv(dx, w) + conv(x, dw) + db
 * `attrs` carries the same stride/padding/dilation/groups attributes that the
 * primal op consumes; both the primal and the two tangent passes reuse them.
 */
auto jvp_conv_forward(OpId op,
                      const DualTensor& input,
                      const DualTensor& weight,
                      const std::optional<DualTensor>& bias,
                      const OpAttributes& attrs) -> DualTensor;
/// @}

/// @name View / Shape long-tail JVP Rules
/// @{
auto jvp_repeat(const DualTensor& x, std::vector<int64_t> repeats) -> DualTensor;
auto jvp_tile(const DualTensor& x, std::vector<int64_t> reps) -> DualTensor;
auto jvp_diag(const DualTensor& x, int64_t diagonal = 0) -> DualTensor;
auto jvp_tril(const DualTensor& x, int64_t diagonal = 0) -> DualTensor;
auto jvp_triu(const DualTensor& x, int64_t diagonal = 0) -> DualTensor;
auto jvp_flip(const DualTensor& x, std::vector<int64_t> dims) -> DualTensor;
auto jvp_roll(const DualTensor& x, int64_t shifts, int64_t dim) -> DualTensor;
auto jvp_repeat_interleave(const DualTensor& x, int64_t repeats,
                           std::optional<int64_t> dim = std::nullopt) -> DualTensor;
auto jvp_take(const DualTensor& x, const Tensor& index) -> DualTensor;
auto jvp_take_along_dim(const DualTensor& x, const Tensor& indices, int64_t dim) -> DualTensor;
auto jvp_diagonal_scatter(const DualTensor& input, const DualTensor& src,
                          int64_t offset = 0, int64_t dim1 = 0, int64_t dim2 = 1) -> DualTensor;

/// @}

/// @name Element-wise math long-tail (Audit A.4 batch 5)
/// @{
/// Non-differentiable step/round functions: derivative is 0 almost everywhere.
auto jvp_floor(const DualTensor& x) -> DualTensor;
auto jvp_ceil(const DualTensor& x) -> DualTensor;
auto jvp_round(const DualTensor& x) -> DualTensor;
auto jvp_trunc(const DualTensor& x) -> DualTensor;
auto jvp_frac(const DualTensor& x) -> DualTensor;
auto jvp_heaviside(const DualTensor& x, const DualTensor& values) -> DualTensor;

/// d/dx atan2(y, x) = (x*dy - y*dx) / (x^2 + y^2)
auto jvp_atan2(const DualTensor& y, const DualTensor& x) -> DualTensor;
/// d/dx hypot(a, b) = (a*da + b*db) / hypot(a, b)
auto jvp_hypot(const DualTensor& a, const DualTensor& b) -> DualTensor;
/// d/d{a,b} logaddexp(a, b) = softmax_weighted(da, db) along {a,b}
auto jvp_logaddexp(const DualTensor& a, const DualTensor& b) -> DualTensor;

/// nan_to_num: derivative is 1 where x is finite, 0 at NaN/Inf positions.
auto jvp_nan_to_num(const DualTensor& x, double nan, double posinf, double neginf) -> DualTensor;
/// @}

/// @name Reductions long-tail (Audit A.4 batch 5)
/// @{
/// p-norm: tangent = sum(sign(x) * |x|^(p-1) * dx, dim) * norm^(1-p)
auto jvp_norm(const DualTensor& x, float p, std::optional<int64_t> dim, bool keepdim) -> DualTensor;
/// argmax/argmin/argsort produce integer indices: tangent is zero (not differentiable).
auto jvp_argmax(const DualTensor& x, std::optional<int64_t> dim, bool keepdim) -> DualTensor;
auto jvp_argmin(const DualTensor& x, std::optional<int64_t> dim, bool keepdim) -> DualTensor;
auto jvp_argsort(const DualTensor& x, int64_t dim, bool descending) -> DualTensor;
/// bucketize is non-differentiable (integer output).
auto jvp_bucketize(const DualTensor& x, const Tensor& boundaries, bool right) -> DualTensor;
/// @}

/// @name Index / scatter long-tail (Audit A.4 batch 5)
/// @{
/// index_add: y[index[i]] += source[i]; linear in both input and source.
auto jvp_index_add(const DualTensor& input, int64_t dim, const Tensor& index,
                   const DualTensor& source) -> DualTensor;
/// index_copy: y[index[i]] = source[i]; linear in source; passthrough for input (with
/// the indexed slots overwritten by the source tangent).
auto jvp_index_copy(const DualTensor& input, int64_t dim, const Tensor& index,
                    const DualTensor& source) -> DualTensor;
/// index_fill: y[index[i]] = constant; linear in input with the indexed slots zeroed.
auto jvp_index_fill(const DualTensor& input, int64_t dim, const Tensor& index,
                    float value) -> DualTensor;
/// select_scatter: linear in input and src.
auto jvp_select_scatter(const DualTensor& input, const DualTensor& src,
                        int64_t dim, int64_t index) -> DualTensor;
/// slice_scatter: linear in input and src.
auto jvp_slice_scatter(const DualTensor& input, const DualTensor& src,
                       int64_t dim, int64_t start, int64_t end, int64_t step) -> DualTensor;
/// unfold: linear shape operation.
auto jvp_unfold(const DualTensor& input, int64_t kernel_size, int64_t stride,
                int64_t padding, int64_t dilation) -> DualTensor;
/// @}

} // namespace tenzor
