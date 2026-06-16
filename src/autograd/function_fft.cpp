#include "tenzor/autograd/function.hpp"
#include <cassert>
#include "tenzor/autograd/ops.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/fft.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/utils/safe_math.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <string>
#include <typeinfo>
#include <unordered_set>
#include <vector>
#ifdef __GNUC__
#include <cxxabi.h>
#include <cstdlib>
#endif

namespace tenzor {

// ============================================================================
// CumSum backward
// ============================================================================

auto CumSumBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CumSumBackward::forward should not be called");
}

auto CumSumBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // dL/dx = flip(cumsum(flip(grad, dim), dim), dim)
    auto flipped = flip(grad, {dim_});
    auto cum = cumsum(flipped, dim_);
    return {flip(cum, {dim_})};
}

auto CumSumBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // cumsum backward: flip(cumsum(flip(grad, dim), dim), dim)
    auto flipped = tenzor::flip(grad_outputs[0], {dim_});
    auto cum = tenzor::cumsum(flipped, dim_);
    return {tenzor::flip(cum, {dim_})};
}

// ============================================================================
// CumProd backward
// ============================================================================

auto CumProdBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CumProdBackward::forward should not be called");
}

auto CumProdBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];

    // dL/dx = flip(cumsum(flip(output * grad, dim), dim), dim) / input
    // With zero-safe division
    auto prod_grad = mul(output, grad);
    auto flipped = flip(prod_grad, {dim_});
    auto cum = cumsum(flipped, dim_);
    auto rev_cum = flip(cum, {dim_});

    // CC.3: zero-safe division without dtype-dependent eps.
    // The previous approach replaced zeros in the denominator with
    // `dtype_epsilon(input.dtype())`. For F16 that constant is ~9.8e-4 —
    // large enough to distort the gradient on the masked positions before
    // we zero them out. (rev_cum / eps explodes to ~1e3 for moderate rev_cum,
    // which feeds finite-magnitude garbage into intermediate buffers and
    // overflows when chained.) Replace zeros with ONES in the safe denominator
    // — the division result on those positions is then a bounded value that
    // we immediately overwrite with zero via the mask, with no eps-scale
    // intermediate.
    auto zero_t = zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                        input.dtype(), input.device());
    auto ones_t = ones(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                       input.dtype(), input.device());
    auto zero_mask = eq(input, zero_t);
    auto safe_input = where(zero_mask, ones_t, input);
    auto result = div(rev_cum, safe_input);

    // Zero out positions where input was zero
    return {where(zero_mask, zero_t, result)};
}

auto CumProdBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // cumprod backward: flip(cumsum(flip(output * grad, dim), dim), dim) / input
    const auto& input = saved_tensors_[0];
    // GG.1: recompute cumprod from saved input Variable on the higher-order
    // path so output_var carries grad_fn back through the upstream forward.
    // The saved output Tensor (saved_tensors_[1]) is still used by the
    // Tensor-only backward(); here we recompute to preserve the chain.
    Variable output_var;
    if (has_saved_variables()) {
        output_var = tenzor::cumprod(saved_variables_[0], dim_);
    } else {
        output_var = Variable(saved_tensors_[1], false);
    }

    auto prod_grad = output_var * grad_outputs[0];
    auto flipped = tenzor::flip(prod_grad, {dim_});
    auto cum = tenzor::cumsum(flipped, dim_);
    auto rev_cum = tenzor::flip(cum, {dim_});

    // CC.3: zero-safe division. Use ones (not eps) in the safe denominator —
    // eps is dtype-dependent and distorts F16 gradients before the mask zeroes
    // them out.
    // GG.1: on the higher-order path, build the safe denominator via Variable
    // ops sourced from saved_variables_[0] so the division's chain through
    // the input survives create_graph.
    auto zero_t = zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                        input.dtype(), input.device());
    auto ones_t = ones(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                       input.dtype(), input.device());
    auto zero_mask = eq(input, zero_t);  // mask is non-differentiable
    Variable mask_var(zero_mask, false);
    Variable ones_var(ones_t, false);
    Variable zero_var(zero_t, false);

    Variable safe_input_var;
    if (has_saved_variables()) {
        safe_input_var = tenzor::where(mask_var, ones_var, saved_variables_[0]);
    } else {
        auto safe_input = where(zero_mask, ones_t, input);
        safe_input_var = Variable(safe_input, false);
    }

    auto result = rev_cum / safe_input_var;

    // Zero out positions where input was zero
    return {tenzor::where(mask_var, zero_var, result)};
}

// ============================================================================
// TopK backward
// ============================================================================

auto TopKBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TopKBackward::forward should not be called");
}

auto TopKBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // saved_tensors_[0] = original shape as 1D Int64 tensor
    // saved_tensors_[1] = indices from topk
    const auto& shape_tensor = saved_tensors_[0];
    const auto& indices = saved_tensors_[1];

    auto shape_ptr = shape_tensor.data<int64_t>();
    auto orig_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    // Create zeros with original shape and scatter grad at index positions
    auto result = zeros(orig_shape, grad.dtype(), grad.device());
    return {scatter_add(result, dim_, indices, grad)};
}

auto TopKBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Indices and original shape are constants; scatter_add threads grad Variable
    // back to original positions preserving the graph for create_graph=true.
    const auto& shape_tensor = saved_tensors_[0];
    const auto& indices = saved_tensors_[1];
    const auto& grad_var = grad_outputs[0];

    auto shape_ptr = shape_tensor.data<int64_t>();
    auto orig_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    auto zeros_t = zeros(orig_shape, grad_var.tensor().dtype(), grad_var.tensor().device());
    Variable zeros_var(zeros_t, false);
    return {tenzor::scatter_add(zeros_var, dim_, indices, grad_var)};
}

// ============================================================================
// Sort backward
// ============================================================================

auto SortBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SortBackward::forward should not be called");
}

auto SortBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // saved_tensors_[0] = original shape as 1D Int64 tensor
    // saved_tensors_[1] = sort indices
    const auto& shape_tensor = saved_tensors_[0];
    const auto& indices = saved_tensors_[1];

    auto shape_ptr = shape_tensor.data<int64_t>();
    auto orig_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    // Scatter grad back using inverse permutation (same as scatter)
    auto result = zeros(orig_shape, grad.dtype(), grad.device());
    return {scatter(result, dim_, indices, grad)};
}

auto SortBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Sort indices are a constant permutation; scatter threads grad Variable
    // back through the inverse permutation preserving the graph.
    const auto& shape_tensor = saved_tensors_[0];
    const auto& indices = saved_tensors_[1];
    const auto& grad_var = grad_outputs[0];

    auto shape_ptr = shape_tensor.data<int64_t>();
    auto orig_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    auto zeros_t = zeros(orig_shape, grad_var.tensor().dtype(), grad_var.tensor().device());
    Variable zeros_var(zeros_t, false);
    return {tenzor::scatter(zeros_var, dim_, indices, grad_var)};
}

// ============================================================================
// Diag backward
// ============================================================================

auto DiagBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("DiagBackward::forward should not be called");
}

auto DiagBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // diag() is its own "transpose": applying diag to the grad reverses the operation
    return {diag(grad, diagonal_)};
}

auto DiagBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // diag backward: diag(grad, diagonal) reverses the operation
    return {tenzor::diag(grad_outputs[0], diagonal_)};
}

// ============================================================================
// Trace backward
// ============================================================================

auto TraceBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TraceBackward::forward should not be called");
}

auto TraceBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // saved_tensors_[0] holds dtype/device info from the original input
    const auto& input = saved_tensors_[0];
    // dL/dA = grad_scalar * eye(n)
    auto identity = eye(n_, std::nullopt, input.dtype(), input.device());
    // grad is scalar — expand it
    return {mul(identity, grad)};
}

auto TraceBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // trace backward: dL/dA = grad_scalar * eye(n)
    const auto& input = saved_tensors_[0];
    auto identity = eye(n_, std::nullopt, input.dtype(), input.device());
    Variable eye_var(identity, false);
    // grad is scalar; multiply broadcasts
    return {grad_outputs[0] * eye_var};
}

// ============================================================================
// Triu backward
// ============================================================================

auto TriuBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TriuBackward::forward should not be called");
}

auto TriuBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {triu(grad_outputs[0], diagonal_)};
}

auto TriuBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {tenzor::triu(grad_outputs[0], diagonal_)};
}

// ============================================================================
// Tril backward
// ============================================================================

auto TrilBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TrilBackward::forward should not be called");
}

auto TrilBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {tril(grad_outputs[0], diagonal_)};
}

auto TrilBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {tenzor::tril(grad_outputs[0], diagonal_)};
}

// ============================================================================
// FFT backward
// ============================================================================
// Normalization inversion. For y = FFT(x) with norm ν the Jacobian is
// the matrix A_ν; its Hermitian adjoint A_ν^H — which is what the chain
// rule needs — is the IFFT under the *opposite* scaling convention.
// Concretely: "backward" (unscaled fft, scaled ifft) pairs with "forward"
// (scaled fft, unscaled ifft); "ortho" is self-adjoint. Using the same
// norm on both branches (as the old code did) left grad_x short by a
// factor of N and broke numerical-gradient parity.
static std::string invert_fft_norm(const std::string& norm) {
    if (norm == "backward") return "forward";
    if (norm == "forward") return "backward";
    return norm;  // "ortho" is self-adjoint
}

// ---------------------------------------------------------------------------
// True linear adjoints for the real FFTs (rfft / irfft).
//
// The gradient of a linear operator is its Hermitian adjoint, NOT its inverse.
// rfft and irfft are inverses, but their adjoints differ from the inverse by
// the treatment of the conjugate-symmetric ("redundant") frequency bins:
//
//   * adjoint(rfft)(g)  = Re( ifft_full( zero-pad g to N bins ) )
//       — a plain complex inverse DFT of the one-sided spectrum padded with
//         zeros (NOT Hermitian-reconstructed/doubled as irfft does), real part.
//
//   * adjoint(irfft)(g) = fold( rfft(g) )
//       — the one-sided forward DFT with the interior bins doubled (DC and,
//         when N is even, the Nyquist bin are kept single). This is the
//         transpose of the Hermitian-symmetry embedding irfft applies.
//
// Using irfft as the rfft-adjoint (and bare rfft as the irfft-adjoint) leaves
// the off-DC/Nyquist bins mis-weighted by a factor of 2, which passes a
// grad-of-ones round-trip by luck but fails per-basis-vector gradcheck.
// The norm scaling is handled by routing the inner fft/ifft through the
// opposite normalization convention (invert_fft_norm).
// ---------------------------------------------------------------------------

static int64_t fft_norm_dim(int64_t dim, int64_t ndim) {
    return dim < 0 ? dim + ndim : dim;
}

// adjoint(rfft): complex one-sided grad (M bins) -> real grad (N samples).
static Tensor rfft_adjoint(const Tensor& grad_freq, int64_t signal_length,
                           int64_t dim, const std::string& norm) {
    const int64_t d = fft_norm_dim(dim, grad_freq.ndim());
    const int64_t M = grad_freq.shape()[d];
    const int64_t N = signal_length;

    Tensor full;
    if (M < N) {
        std::vector<int64_t> pad_shape(grad_freq.shape().begin(),
                                       grad_freq.shape().end());
        pad_shape[d] = N - M;
        Tensor zpad = tenzor::zeros(pad_shape, grad_freq.dtype(),
                                    grad_freq.device());
        full = tenzor::cat({grad_freq, zpad}, d);
    } else if (M > N) {
        full = tenzor::slice(grad_freq, d, 0, N);
    } else {
        full = grad_freq;
    }
    Tensor spatial = tenzor::fft::ifft(full, N, d, invert_fft_norm(norm));
    return tenzor::real(spatial);
}

// Per-bin Hermitian-fold weights as a complex tensor broadcastable along `dim`
// (length P there, 1 elsewhere): w[0]=1, interior=2, Nyquist=1 (when N even).
static Tensor hermitian_fold_weight(int64_t P, int64_t N, int64_t dim,
                                    int64_t ndim, DType complex_dtype,
                                    Device device) {
    const DType real_dtype = (complex_dtype == DType::Complex128)
        ? DType::Float64 : DType::Float32;
    std::vector<int64_t> wshape(static_cast<size_t>(ndim), 1);
    wshape[static_cast<size_t>(dim)] = P;

    std::vector<double> wv(static_cast<size_t>(P), 2.0);
    wv[0] = 1.0;                                   // DC bin
    if (N % 2 == 0 && P >= 1) wv[static_cast<size_t>(P - 1)] = 1.0;  // Nyquist

    Tensor w_real;
    if (real_dtype == DType::Float64) {
        w_real = tenzor::from_data(wv.data(), wshape, device);
    } else {
        std::vector<float> wf(wv.begin(), wv.end());
        w_real = tenzor::from_data(wf.data(), wshape, device);
    }
    Tensor w_imag = tenzor::zeros(wshape, real_dtype, device);
    return tenzor::complex(w_real, w_imag);
}

// adjoint(irfft): real grad (N samples) -> complex one-sided grad (M bins).
static Tensor irfft_adjoint(const Tensor& grad_spatial, int64_t bins_M,
                            int64_t dim, const std::string& norm) {
    const int64_t d = fft_norm_dim(dim, grad_spatial.ndim());
    const int64_t N = grad_spatial.shape()[d];

    Tensor freq = tenzor::fft::rfft(grad_spatial, std::nullopt, d,
                                    invert_fft_norm(norm));   // P = N/2+1 bins
    const int64_t P = freq.shape()[d];
    Tensor w = hermitian_fold_weight(P, N, d, freq.ndim(), freq.dtype(),
                                     freq.device());
    Tensor folded = tenzor::mul(freq, w);

    const int64_t M = (bins_M >= 0) ? bins_M : P;
    if (P > M) {
        folded = tenzor::slice(folded, d, 0, M);
    } else if (P < M) {
        std::vector<int64_t> pad_shape(folded.shape().begin(),
                                       folded.shape().end());
        pad_shape[d] = M - P;
        Tensor zpad = tenzor::zeros(pad_shape, folded.dtype(), folded.device());
        folded = tenzor::cat({folded, zpad}, d);
    }
    return folded;
}

auto FFTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("FFTBackward::forward should not be called");
}

auto FFTBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {fft::ifft(grad_outputs[0], n_, dim_, invert_fft_norm(norm_))};
}

auto FFTBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {fft_autograd::ifft(grad_outputs[0], n_, dim_, invert_fft_norm(norm_))};
}

// ============================================================================
// IFFT backward
// ============================================================================

auto IFFTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("IFFTBackward::forward should not be called");
}

auto IFFTBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {fft::fft(grad_outputs[0], n_, dim_, invert_fft_norm(norm_))};
}

auto IFFTBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {fft_autograd::fft(grad_outputs[0], n_, dim_, invert_fft_norm(norm_))};
}

// ============================================================================
// RFFT backward
// ============================================================================

auto RFFTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("RFFTBackward::forward should not be called");
}

auto RFFTBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {rfft_adjoint(grad_outputs[0], signal_length_, dim_, norm_)};
}

auto RFFTBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // True adjoint as autograd ops (so create_graph threads second order):
    //   Re( ifft_full( zero-pad(grad, N) ) ).
    const Variable& g = grad_outputs[0];
    const int64_t d = fft_norm_dim(dim_, g.tensor().ndim());
    const int64_t M = g.tensor().shape()[d];
    const int64_t N = signal_length_;

    Variable full = g;
    if (M < N) {
        std::vector<int64_t> pad_shape(g.tensor().shape().begin(),
                                       g.tensor().shape().end());
        pad_shape[d] = N - M;
        Variable zpad(tenzor::zeros(pad_shape, g.tensor().dtype(),
                                    g.tensor().device()), false);
        full = cat(std::vector<Variable>{g, zpad}, d);
    } else if (M > N) {
        full = slice(g, d, 0, N);
    }
    Variable spatial = fft_autograd::ifft(full, N, d, invert_fft_norm(norm_));
    // Real part: view_as_real appends a size-2 (re, im) trailing dim.
    Variable asreal = view_as_real(spatial);
    const int64_t last = asreal.tensor().ndim() - 1;
    Variable re = squeeze(slice(asreal, last, 0, 1), last);
    return {re};
}

// ============================================================================
// IRFFT backward
// ============================================================================

auto IRFFTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("IRFFTBackward::forward should not be called");
}

auto IRFFTBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // n_orig_ is the frequency-bin count (M) of the original irfft input, so
    // the adjoint produces exactly M bins even when the forward irfft used a
    // non-default time length.
    return {irfft_adjoint(grad_outputs[0], n_orig_, dim_, norm_)};
}

auto IRFFTBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // True adjoint as autograd ops: rfft(grad) with interior bins doubled
    // (DC and, for even N, Nyquist kept single), then fit to M bins.
    const Variable& g = grad_outputs[0];
    const int64_t d = fft_norm_dim(dim_, g.tensor().ndim());
    const int64_t N = g.tensor().shape()[d];

    Variable freq = fft_autograd::rfft(g, std::nullopt, d, invert_fft_norm(norm_));
    const int64_t P = freq.tensor().shape()[d];
    Tensor w = hermitian_fold_weight(P, N, d, freq.tensor().ndim(),
                                     freq.tensor().dtype(), freq.tensor().device());
    Variable folded = freq * Variable(w, false);

    const int64_t M = (n_orig_ >= 0) ? n_orig_ : P;
    if (P > M) {
        folded = slice(folded, d, 0, M);
    } else if (P < M) {
        std::vector<int64_t> pad_shape(folded.tensor().shape().begin(),
                                       folded.tensor().shape().end());
        pad_shape[d] = M - P;
        Variable zpad(tenzor::zeros(pad_shape, folded.tensor().dtype(),
                                    folded.tensor().device()), false);
        folded = cat(std::vector<Variable>{folded, zpad}, d);
    }
    return {folded};
}

// ============================================================================
// Phase A.3 / audit-6 AA.6 — STFT / ISTFT backward.
//
// PyTorch's autograd contract requires the **true linear adjoint** of each
// operator, not its inverse. STFT and ISTFT are inverses of each other only
// for tight-COLA windows; in general `ISTFT(STFT(x)) == x` but
// `ISTFT^H == STFT_with_renormalised_window`, and `STFT^H` is overlap-add
// of windowed IFFT frames **without** the window-sum normalisation that
// `ISTFT` applies. Routing the gradient through the inverse instead of the
// adjoint produces wrong gradients whenever the window is not tight-COLA.
//
// The audit-6 fix below computes both adjoints from first principles:
//
// * `stft_adjoint(grad)` — overlap-add of `window * IFFT(grad_frame)` into a
//   time-domain buffer of length `expected_length = n_fft + (T-1)*hop`,
//   then trim to `signal_length`. No window-sum division.
//
// * `istft_adjoint(grad)` — for each frame extract a windowed slice of the
//   incoming time-domain `grad`, divide pointwise by the same `window_sum`
//   that ISTFT used (the (window*window) overlap-add), then FFT (rfft if
//   onesided) per frame, stacking into the STFT-shaped output.
//
// Reference: PyTorch's `_stft_backward` / `_istft_backward` in
// `aten/src/ATen/native/SpectralOps.cpp`.
//
// The implementation is intentionally written in plain C++ over a Float32
// CPU buffer: backward passes are not the throughput bottleneck and we
// want strict bit-for-bit parity with the linear-adjoint formula. The
// inputs are cast to {Float32, Complex64} on entry and the original
// dtype/device is restored on return.
// ============================================================================

namespace {

// Materialise the window vector as a length-`n_fft` contiguous host buffer
// (scalar type `Real`) with `win_length` samples centered at
// `(n_fft - win_length) / 2`, matching the CPU stft / istft kernels. Returns
// the zero-padded window. Templated on the working precision so a Float64
// STFT computes its adjoint in double precision (see EE.4 / audit-2026-06-16).
template <typename Real, DType RealDT>
auto resolve_window(const Tensor& window, int64_t n_fft, int64_t win_length)
    -> std::vector<Real>
{
    std::vector<Real> win_data(static_cast<size_t>(n_fft), Real{0});
    int64_t win_offset = (n_fft - win_length) / 2;
    if (window.is_valid() && window.numel() > 0) {
        Tensor w_cpu = (window.device().type != Device::Type::CPU)
            ? window.to(Device::cpu())
            : window;
        Tensor w_r = (w_cpu.dtype() != RealDT)
            ? w_cpu.to(RealDT)
            : w_cpu;
        if (!w_r.is_contiguous()) {
            w_r = w_r.contiguous();
        }
        const Real* wp = w_r.data<Real>();
        for (int64_t i = 0; i < win_length; ++i) {
            win_data[static_cast<size_t>(win_offset + i)] = wp[i];
        }
    } else {
        for (int64_t i = 0; i < win_length; ++i) {
            win_data[static_cast<size_t>(win_offset + i)] = Real{1};
        }
    }
    return win_data;
}

// Per-batch trim helper: copy `[trim_start, trim_start + final_length)` of
// the expanded buffer into the output buffer for batch element `b`.
[[maybe_unused]] void copy_with_trim(const float* src, float* dst,
                    int64_t expected_length,
                    int64_t trim_start, int64_t final_length) {
    int64_t copy_len = std::min(final_length, expected_length - trim_start);
    if (copy_len > 0) {
        std::memcpy(dst, src + trim_start,
                    static_cast<size_t>(copy_len) * sizeof(float));
    }
    if (copy_len < final_length) {
        std::memset(dst + std::max<int64_t>(copy_len, 0), 0,
                    static_cast<size_t>(final_length -
                                        std::max<int64_t>(copy_len, 0)) *
                    sizeof(float));
    }
}

// True STFT linear adjoint: takes `grad` shape (..., freq_bins, num_frames)
// and returns shape (..., signal_length) by overlap-adding windowed IFFT
// frames. No window-sum normalisation.
template <typename Real, DType RealDT, DType CplxDT>
auto stft_adjoint_impl(const Tensor& grad,
                       int64_t n_fft,
                       int64_t hop_length,
                       int64_t win_length,
                       const Tensor& window,
                       bool center,
                       bool normalized,
                       bool onesided,
                       int64_t signal_length) -> Tensor
{
    using Cplx = std::complex<Real>;
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;

    // Move grad to CPU complex contiguous for direct pointer access. The
    // working complex dtype (Complex64 vs Complex128) follows the real input
    // dtype family so a Float64 STFT keeps double precision through the adjoint.
    Tensor g_cpu = (grad.device().type != Device::Type::CPU)
        ? grad.to(Device::cpu())
        : grad;
    Tensor g_c64 = (g_cpu.dtype() != CplxDT)
        ? g_cpu.to(CplxDT)
        : g_cpu;
    if (!g_c64.is_contiguous()) g_c64 = g_c64.contiguous();

    auto in_shape = g_c64.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (ndim < 2) {
        throw std::runtime_error("stft_adjoint: grad must have rank >= 2");
    }
    int64_t freq_bins = in_shape[ndim - 2];
    int64_t num_frames = in_shape[ndim - 1];
    int64_t batch_size = 1;
    for (int64_t d = 0; d < ndim - 2; ++d) batch_size *= in_shape[d];

    int64_t expected_length = n_fft + (num_frames - 1) * hop_length;

    auto win_data = resolve_window<Real, RealDT>(window, n_fft, win_length);

    std::vector<Real> output_data(
        static_cast<size_t>(batch_size * expected_length), Real{0});

    const auto* in_ptr =
        reinterpret_cast<const Cplx*>(g_c64.data_ptr());

    for (int64_t b = 0; b < batch_size; ++b) {
        Real* out = output_data.data() + b * expected_length;

        for (int64_t f = 0; f < num_frames; ++f) {
            Tensor frame_freq({freq_bins}, CplxDT, Device::cpu());
            auto* freq_data =
                reinterpret_cast<Cplx*>(frame_freq.data_ptr());
            for (int64_t k = 0; k < freq_bins; ++k) {
                freq_data[k] =
                    in_ptr[b * freq_bins * num_frames + k * num_frames + f];
            }

            // The adjoint of `Y = FFT(window .* x_frame)` w.r.t. the real
            // x_frame is `window .* adjoint(FFT)(Y)`. For the one-sided rfft
            // the adjoint is NOT `irfft` — it must zero-pad (not Hermitian-
            // double) the redundant bins, so reuse the proven `rfft_adjoint`
            // helper. Using `irfft * N` here double-counted the interior bins.
            Tensor real_frame;
            if (onesided) {
                real_frame = rfft_adjoint(frame_freq, n_fft, 0,
                                          normalized ? "ortho" : "backward");
            } else {
                // adjoint(fft_b) = N * ifft_b; take the real part since the
                // forward frame fed to the FFT was real-valued.
                Tensor time_frame = fft::ifft(frame_freq, n_fft, -1,
                                              normalized ? "ortho" : "backward");
                real_frame = tenzor::real(time_frame);
            }
            if (real_frame.dtype() != RealDT)
                real_frame = real_frame.to(RealDT);
            if (!real_frame.is_contiguous()) real_frame = real_frame.contiguous();
            const Real* frame_data = real_frame.data<Real>();

            // rfft_adjoint already encodes the norm; only the non-onesided
            // ifft path needs the forward-FFT `N` factor.
            const Real n_scale = (onesided || normalized)
                                      ? Real{1} : static_cast<Real>(n_fft);
            int64_t frame_offset = f * hop_length;
            for (int64_t i = 0; i < n_fft; ++i) {
                out[frame_offset + i] += frame_data[i] *
                                         win_data[static_cast<size_t>(i)] *
                                         n_scale;
            }
        }
    }

    // Undo the forward STFT's center padding. The forward pads by REFLECTION
    // (n_fft/2 each side), so the adjoint is NOT a plain trim: every reflected
    // sample's gradient must fold back onto the source index it was copied
    // from. (Treating it as zero-pad/trim left the first/last `pad` samples'
    // gradients wrong — the STFT round-trip gradcheck failure.)
    int64_t pad = center ? (n_fft / 2) : 0;
    int64_t final_length =
        (signal_length > 0) ? signal_length
                            : (expected_length - 2 * pad);

    std::vector<int64_t> final_shape;
    for (int64_t d = 0; d < ndim - 2; ++d) final_shape.push_back(in_shape[d]);
    final_shape.push_back(final_length);

    Tensor result(final_shape, RealDT, Device::cpu());
    Real* result_data = result.data<Real>();
    std::fill(result_data,
              result_data + static_cast<size_t>(batch_size * final_length),
              Real{0});
    for (int64_t b = 0; b < batch_size; ++b) {
        const Real* src = output_data.data() + b * expected_length;
        Real* dst = result_data + b * final_length;
        if (!center) {
            int64_t copy_len = std::min(final_length, expected_length);
            if (copy_len > 0)
                std::memcpy(dst, src,
                            static_cast<size_t>(copy_len) * sizeof(Real));
            continue;
        }
        // Identity (middle) copy: forward out[pad + j] = sig[j].
        for (int64_t j = 0; j < final_length; ++j) {
            int64_t s = pad + j;
            if (s >= 0 && s < expected_length) dst[j] += src[s];
        }
        // Left reflect: forward out[i] = sig[clamp(pad - i, .., L-1)].
        for (int64_t i = 0; i < pad && i < expected_length; ++i) {
            int64_t idx = pad - i;
            if (idx >= final_length) idx = final_length - 1;
            if (idx >= 0 && idx < final_length) dst[idx] += src[i];
        }
        // Right reflect: forward out[pad + L + i] = sig[clamp(L - 2 - i, 0, ..)].
        for (int64_t i = 0; i < pad; ++i) {
            int64_t s = pad + final_length + i;
            int64_t idx = final_length - 2 - i;
            if (idx < 0) idx = 0;
            if (idx >= 0 && idx < final_length && s < expected_length)
                dst[idx] += src[s];
        }
    }

    // Restore original dtype/device.
    Tensor out_dev = (grad.device().type != Device::Type::CPU)
        ? result.to(grad.device())
        : result;
    // grad is complex; the time-domain gradient is real and computed in the
    // working precision `Real` (Float32 for 16/32-bit families, Float64 for
    // Float64 inputs) so no precision is silently dropped.
    return out_dev;
}

// True ISTFT linear adjoint: takes `grad` shape (..., signal_length) and
// returns STFT shape (..., freq_bins, num_frames). Mirrors the ISTFT
// computation but divides each extracted frame by the per-sample
// `window_sum` ISTFT applied at forward time.
template <typename Real, DType RealDT, DType CplxDT>
auto istft_adjoint_impl(const Tensor& grad,
                        int64_t n_fft,
                        int64_t hop_length,
                        int64_t win_length,
                        const Tensor& window,
                        bool center,
                        bool normalized,
                        bool onesided) -> Tensor
{
    using Cplx = std::complex<Real>;
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;

    Tensor g_cpu = (grad.device().type != Device::Type::CPU)
        ? grad.to(Device::cpu())
        : grad;
    // Work in the real input dtype family (Float64 STFT keeps double precision).
    Tensor g_f32 = (g_cpu.dtype() != RealDT)
        ? g_cpu.to(RealDT)
        : g_cpu;
    if (!g_f32.is_contiguous()) g_f32 = g_f32.contiguous();

    auto in_shape = g_f32.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (ndim < 1) {
        throw std::runtime_error("istft_adjoint: grad must have rank >= 1");
    }
    int64_t signal_length = in_shape[ndim - 1];
    int64_t batch_size = 1;
    for (int64_t d = 0; d < ndim - 1; ++d) batch_size *= in_shape[d];

    // Re-derive the padded length / num_frames the forward ISTFT used.
    int64_t padded_length = signal_length;
    if (center) padded_length = signal_length + 2 * (n_fft / 2);
    int64_t num_frames = (padded_length - n_fft) / hop_length + 1;
    if (num_frames <= 0) {
        throw std::runtime_error(
            "istft_adjoint: signal too short for given n_fft/hop_length");
    }
    int64_t expected_length = n_fft + (num_frames - 1) * hop_length;

    auto win_data = resolve_window<Real, RealDT>(window, n_fft, win_length);

    // Compute window_sum used by ISTFT: sum_f window^2 placed at f*hop.
    std::vector<Real> window_sum_buf(
        static_cast<size_t>(expected_length), Real{0});
    for (int64_t f = 0; f < num_frames; ++f) {
        int64_t off = f * hop_length;
        for (int64_t i = 0; i < n_fft; ++i) {
            window_sum_buf[off + i] +=
                win_data[static_cast<size_t>(i)] *
                win_data[static_cast<size_t>(i)];
        }
    }

    // Build padded grad (reflect-pad mirrors the forward ISTFT trim).
    // Forward ISTFT only trims, so the adjoint pads with zeros (the time
    // samples discarded by trim contribute zero gradient).
    std::vector<Real> padded_grad(
        static_cast<size_t>(batch_size * expected_length), Real{0});
    int64_t trim_start = center ? (n_fft / 2) : 0;
    {
        const Real* gptr = g_f32.data<Real>();
        for (int64_t b = 0; b < batch_size; ++b) {
            Real* dst = padded_grad.data() + b * expected_length;
            int64_t copy_len = std::min(signal_length,
                                        expected_length - trim_start);
            if (copy_len > 0) {
                std::memcpy(dst + trim_start, gptr + b * signal_length,
                            static_cast<size_t>(copy_len) * sizeof(Real));
            }
        }
    }

    // Divide by window_sum (matches the forward ISTFT division) — entries
    // with near-zero window_sum mirror ISTFT's `if (wsum > 1e-10) out /= wsum`
    // guard so the adjoint is zero where the forward dropped the sample.
    for (int64_t b = 0; b < batch_size; ++b) {
        Real* row = padded_grad.data() + b * expected_length;
        for (int64_t i = 0; i < expected_length; ++i) {
            const Real w = window_sum_buf[static_cast<size_t>(i)];
            if (w > static_cast<Real>(1e-10)) {
                row[i] /= w;
            } else {
                row[i] = Real{0};
            }
        }
    }

    // For each frame, multiply by window and apply forward FFT.
    int64_t freq_bins = onesided ? (n_fft / 2 + 1) : n_fft;

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim - 1; ++d) out_shape.push_back(in_shape[d]);
    out_shape.push_back(freq_bins);
    out_shape.push_back(num_frames);

    Tensor result(out_shape, CplxDT, Device::cpu());
    auto* out_ptr =
        reinterpret_cast<Cplx*>(result.data_ptr());

    for (int64_t b = 0; b < batch_size; ++b) {
        const Real* row = padded_grad.data() + b * expected_length;
        for (int64_t f = 0; f < num_frames; ++f) {
            Tensor frame_t({n_fft}, RealDT, Device::cpu());
            Real* fd = frame_t.data<Real>();
            int64_t off = f * hop_length;
            for (int64_t i = 0; i < n_fft; ++i) {
                fd[i] = row[off + i] * win_data[static_cast<size_t>(i)];
            }

            // Forward ISTFT applied `irfft` per frame; its adjoint is
            // `irfft_adjoint` (Hermitian-fold of rfft, interior bins doubled),
            // NOT a bare `rfft`. The bare rfft left every off-DC/Nyquist bin
            // mis-scaled, breaking the round-trip gradcheck.
            Tensor freq;
            if (onesided) {
                freq = irfft_adjoint(frame_t, freq_bins, 0,
                                     normalized ? "ortho" : "backward");
            } else {
                // adjoint(ifft_b) = (1/N) * fft_b; the 1/N is applied below.
                freq = fft::fft(frame_t, n_fft, -1,
                                normalized ? "ortho" : "backward");
            }
            if (freq.dtype() != CplxDT) {
                freq = freq.to(CplxDT);
            }
            if (!freq.is_contiguous()) freq = freq.contiguous();

            const Real inv_scale = (onesided || normalized)
                ? Real{1} : (Real{1} / static_cast<Real>(n_fft));
            const auto* freq_ptr =
                reinterpret_cast<const Cplx*>(freq.data_ptr());
            for (int64_t k = 0; k < freq_bins; ++k) {
                out_ptr[b * freq_bins * num_frames + k * num_frames + f] =
                    freq_ptr[k] * inv_scale;
            }
        }
    }

    Tensor out_dev = (grad.device().type != Device::Type::CPU)
        ? result.to(grad.device())
        : result;
    return out_dev;
}

} // anonymous namespace

auto STFTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("STFTBackward::forward should not be called");
}

auto STFTBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // grad_outputs[0] has STFT shape (..., freq_bins, num_frames).
    // Use the true STFT linear adjoint (overlap-add of windowed IFFT frames
    // **without** ISTFT's window-sum normalisation). PyTorch's contract.
    //
    // Audit-7 EE.4 / audit-2026-06-16: select the adjoint working precision
    // from the forward input's real dtype. A Float64 STFT computes its
    // gradient in double precision (Float64 / Complex128 buffers); the 16/32-bit
    // families use Float32 / Complex64. Cast back to input_dtype_ on return.
    Tensor grad_t = (input_dtype_ == DType::Float64)
        ? stft_adjoint_impl<double, DType::Float64, DType::Complex128>(
              grad_outputs[0], n_fft_, hop_length_, win_length_,
              window_, center_, normalized_, onesided_, signal_length_)
        : stft_adjoint_impl<float, DType::Float32, DType::Complex64>(
              grad_outputs[0], n_fft_, hop_length_, win_length_,
              window_, center_, normalized_, onesided_, signal_length_);
    if (grad_t.dtype() != input_dtype_) {
        grad_t = grad_t.to(input_dtype_);
    }
    return {grad_t};
}

auto ISTFTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("ISTFTBackward::forward should not be called");
}

auto ISTFTBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // grad_outputs[0] has ISTFT output shape (..., signal_length).
    // True ISTFT linear adjoint = STFT with each frame divided by the same
    // window_sum the forward ISTFT applied.
    //
    // Audit-7 EE.4 / audit-2026-06-16: the forward ISTFT input is complex;
    // select the adjoint working precision from input_dtype_ so a Complex128
    // ISTFT computes its gradient in double precision (Float64 / Complex128
    // buffers) rather than silently in single precision.
    Tensor grad_t = (input_dtype_ == DType::Complex128)
        ? istft_adjoint_impl<double, DType::Float64, DType::Complex128>(
              grad_outputs[0], n_fft_, hop_length_, win_length_,
              window_, center_, normalized_, onesided_)
        : istft_adjoint_impl<float, DType::Float32, DType::Complex64>(
              grad_outputs[0], n_fft_, hop_length_, win_length_,
              window_, center_, normalized_, onesided_);
    if (grad_t.dtype() != input_dtype_) {
        grad_t = grad_t.to(input_dtype_);
    }
    return {grad_t};
}

} // namespace tenzor
