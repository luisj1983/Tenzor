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
auto jvp_leaky_relu(const DualTensor& x, double negative_slope = 0.01) -> DualTensor;
auto jvp_elu(const DualTensor& x, double alpha = 1.0) -> DualTensor;
auto jvp_selu(const DualTensor& x) -> DualTensor;
auto jvp_softplus(const DualTensor& x, double beta = 1.0) -> DualTensor;
auto jvp_mish(const DualTensor& x) -> DualTensor;
auto jvp_hardswish(const DualTensor& x) -> DualTensor;
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

/// @name Sparse JVP Rules (Audit A.4 batch 7)
/// @{
/// SpMM: y = A_sparse @ B. Bilinear in A.values and B; sparse pattern (crow,
/// col) is constant (integer indices, non-differentiable). JVP:
///   dy = SpMM(crow, col, dA.values, B) + SpMM(crow, col, A.values, dB)
auto jvp_sparse_spmm(const Tensor& crow, const Tensor& col,
                     const DualTensor& values, const DualTensor& dense,
                     int64_t M, int64_t K) -> DualTensor;
/// SpMV: same as SpMM but B is a vector.
auto jvp_sparse_spmv(const Tensor& crow, const Tensor& col,
                     const DualTensor& values, const DualTensor& dense,
                     int64_t M, int64_t K) -> DualTensor;
/// SparseAdd: y = sparse(crow, col, values) + dense. Linear in values+dense.
///   dy = SparseAdd(crow, col, dvalues, ddense)  (== scatter-add+ddense)
auto jvp_sparse_add(const Tensor& crow, const Tensor& col,
                    const DualTensor& values, const DualTensor& dense,
                    int64_t M, int64_t K) -> DualTensor;
/// @}

/// @name RNN cell JVP Rules (Audit A.4 batch 7)
/// @{
/// GRU cell single-step forward. Re-derives the three gates in dual form:
///   gi = x@W_ih^T + b_ih   (split into [g_ir, g_iz, g_in])
///   gh = h@W_hh^T + b_hh   (split into [g_hr, g_hz, g_hn])
///   r  = sigmoid(g_ir + g_hr)
///   z  = sigmoid(g_iz + g_hz)
///   n  = tanh(g_in + r * g_hn)
///   hy = (1-z) * n + z * h
/// All four ops above are sigmoid/tanh (single-arg chain rule) and elementwise
/// mul/add; JVP follows directly from the existing jvp_sigmoid/jvp_tanh
/// formulas applied per-gate.
auto jvp_gru_cell_forward(const DualTensor& x, const DualTensor& h,
                          const DualTensor& W_ih, const DualTensor& W_hh,
                          const DualTensor& b_ih, const DualTensor& b_hh) -> DualTensor;
/// @}

/// @name Nested-tensor JVP Rules (Audit A.4 batch 7)
/// @{
/// NestedSum / NestedMean operate per-segment on packed values; they are
/// linear in `values`, so the tangent is the same op applied to dvalues.
/// Offsets are integer (non-differentiable).
auto jvp_nested_sum(const DualTensor& values, const Tensor& offsets,
                    int64_t dim, bool keepdim) -> DualTensor;
auto jvp_nested_mean(const DualTensor& values, const Tensor& offsets,
                     int64_t dim, bool keepdim) -> DualTensor;
/// NestedLinear: y = values @ W^T + b applied per-segment. Bilinear in
/// (values, W) + linear in b — same structure as dense Linear.
auto jvp_nested_linear(const DualTensor& values, const Tensor& offsets,
                       const DualTensor& weight,
                       const std::optional<DualTensor>& bias) -> DualTensor;
/// @}

/// @name Reduction long-tail JVP Rules (Audit A.4 batch 7)
/// @{
/// logcumsumexp along `dim`. Output l[t] = log(sum_{k<=t} exp(x[k])); the
/// forward-mode tangent is the softmax-weighted prefix sum:
///   dl[t] = sum_{k<=t} softmax_cum(x)[k,t] * dx[k]
///         = cumsum(exp(x - l_keepdim) * dx, dim) / exp(l - l_keepdim)
/// Equivalently (numerically stable, using cum LSE shift):
///   p[t,k] = exp(x[k] - l[t])  (for k <= t, else 0)
///   dl[t]  = sum_k p[t,k] * dx[k]
/// We implement the direct cumsum form: w = exp(x - l), dl = cumsum(w * dx, dim) / cumsum(w, dim).
auto jvp_logcumsumexp(const DualTensor& x, int64_t dim) -> DualTensor;
/// NumericalGradient (finite-difference along `dim`): linear convolution
/// with [-1/2, 0, +1/2] (interior) and one-sided differences at the edges.
/// Linear in input → tangent = same op applied to dx.
auto jvp_numerical_gradient(const DualTensor& x, int64_t dim, double spacing) -> DualTensor;
/// Trapezoid / CumulativeTrapezoid: linear in both y-values and optional
/// x-positions. JVP: same op applied to (dy, dx_positions).
auto jvp_trapezoid(const DualTensor& y, const std::optional<DualTensor>& x_pos,
                   int64_t dim, double dx_uniform) -> DualTensor;
auto jvp_cumulative_trapezoid(const DualTensor& y, const std::optional<DualTensor>& x_pos,
                              int64_t dim, double dx_uniform) -> DualTensor;
/// @}

/// @name Adaptive pooling JVP Rules (Audit A.4 batch 7)
/// @{
/// AdaptiveAvgPool{1,2,3}d: each output cell is an unweighted mean over the
/// corresponding (data-dependent but value-independent) input window. The
/// op is therefore linear in the input → tangent = same op applied to dx.
auto jvp_adaptive_avgpool_1d(const DualTensor& x, int64_t output_size) -> DualTensor;
auto jvp_adaptive_avgpool_2d(const DualTensor& x, int64_t output_h, int64_t output_w) -> DualTensor;
auto jvp_adaptive_avgpool_3d(const DualTensor& x, int64_t output_d, int64_t output_h, int64_t output_w) -> DualTensor;
/// @}

/// @name Attention JVP Rules (Audit A.4 batch 8)
/// @{
/// Scaled dot-product attention forward-mode JVP. Common JVP for the FlashAttention,
/// FusedAttention, FlexAttention (identity score-mod), NestedAttention families:
///   S    = (Q @ K^T) * scale     [+ causal_mask]
///   P    = softmax(S, dim=-1)
///   y    = P @ V
/// Tangent:
///   S_t  = ((dQ @ K^T) + (Q @ dK^T)) * scale
///   P_t  = P * (S_t - sum(P * S_t, dim=-1, keepdim=true))
///   y_t  = (P_t @ V) + (P @ dV)
/// The causal mask is constant — apply identically to S and S_t before the
/// softmax. Dropout (dropout_p > 0) makes the JVP discontinuous unless the
/// Bernoulli mask is saved; we refuse forward-mode AD in that case.
auto jvp_sdpa_forward(const DualTensor& Q,
                      const DualTensor& K,
                      const DualTensor& V,
                      double scale,
                      bool causal) -> DualTensor;
/// @}

/// @name Loss-function JVP Rules (Audit A.4 batch 8)
/// @{
/// FusedSoftmaxCrossEntropy(logits, targets, reduction="mean"|"sum"|"none"):
///   p          = softmax(logits, dim=-1)
///   per_loss[i] = logsumexp(logits[i]) - logits[i, target[i]]
/// Forward-mode tangent:
///   per_t[i]   = sum_j p[i,j] * dlogits[i,j] - dlogits[i, target[i]]
/// Then reduce: mean → mean(per_t); sum → sum(per_t); none → per_t.
/// `targets` is integer and non-differentiable.
auto jvp_fused_softmax_cross_entropy(const DualTensor& logits,
                                     const Tensor& targets,
                                     const std::string& reduction) -> DualTensor;
/// @}

} // namespace tenzor
