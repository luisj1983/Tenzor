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
    const auto& output = saved_tensors_[1];
    Variable output_var(output, false);

    auto prod_grad = output_var * grad_outputs[0];
    auto flipped = tenzor::flip(prod_grad, {dim_});
    auto cum = tenzor::cumsum(flipped, dim_);
    auto rev_cum = tenzor::flip(cum, {dim_});

    // CC.3: zero-safe division at Tensor level (input is constant).
    // Use ones (not eps) in the safe denominator — eps is dtype-dependent and
    // distorts F16 gradients before the mask zeroes them out. See the matching
    // comment in CumProdBackward::backward.
    auto zero_t = zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                        input.dtype(), input.device());
    auto ones_t = ones(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                       input.dtype(), input.device());
    auto zero_mask = eq(input, zero_t);
    auto safe_input = where(zero_mask, ones_t, input);
    Variable safe_input_var(safe_input, false);

    auto result = rev_cum / safe_input_var;

    // Zero out positions where input was zero
    Variable zero_var(zero_t, false);
    Variable mask_var(zero_mask, false);
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
    // irfft needs the original signal length to reconstruct
    return {fft::irfft(grad_outputs[0], signal_length_, dim_, invert_fft_norm(norm_))};
}

auto RFFTBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {fft_autograd::irfft(grad_outputs[0], signal_length_, dim_, invert_fft_norm(norm_))};
}

// ============================================================================
// IRFFT backward
// ============================================================================

auto IRFFTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("IRFFTBackward::forward should not be called");
}

auto IRFFTBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // R.7: Forward saved the original frequency-bin count so the adjoint
    // rfft reproduces that exact bin count instead of letting rfft infer
    // n from the time-domain shape (which differs whenever the forward
    // irfft used a non-default n).
    std::optional<int64_t> n_opt = (n_orig_ >= 0) ? std::optional<int64_t>(n_orig_) : std::nullopt;
    return {fft::rfft(grad_outputs[0], n_opt, dim_, invert_fft_norm(norm_))};
}

auto IRFFTBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    std::optional<int64_t> n_opt = (n_orig_ >= 0) ? std::optional<int64_t>(n_orig_) : std::nullopt;
    return {fft_autograd::rfft(grad_outputs[0], n_opt, dim_, invert_fft_norm(norm_))};
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

// Materialise the window vector as a length-`n_fft` Float32 contiguous
// host buffer with `win_length` samples centered at `(n_fft - win_length) / 2`,
// matching the CPU stft / istft kernels. Returns the zero-padded window.
auto resolve_window(const Tensor& window, int64_t n_fft, int64_t win_length)
    -> std::vector<float>
{
    std::vector<float> win_data(static_cast<size_t>(n_fft), 0.0f);
    int64_t win_offset = (n_fft - win_length) / 2;
    if (window.is_valid() && window.numel() > 0) {
        Tensor w_cpu = (window.device().type != Device::Type::CPU)
            ? window.to(Device::cpu())
            : window;
        Tensor w_f32 = (w_cpu.dtype() != DType::Float32)
            ? w_cpu.to(DType::Float32)
            : w_cpu;
        if (!w_f32.is_contiguous()) {
            w_f32 = w_f32.contiguous();
        }
        const float* wp = w_f32.data<float>();
        for (int64_t i = 0; i < win_length; ++i) {
            win_data[static_cast<size_t>(win_offset + i)] = wp[i];
        }
    } else {
        for (int64_t i = 0; i < win_length; ++i) {
            win_data[static_cast<size_t>(win_offset + i)] = 1.0f;
        }
    }
    return win_data;
}

// Per-batch trim helper: copy `[trim_start, trim_start + final_length)` of
// the expanded buffer into the output buffer for batch element `b`.
void copy_with_trim(const float* src, float* dst, int64_t expected_length,
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
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;

    // Move grad to CPU Complex64 contiguous for direct pointer access.
    Tensor g_cpu = (grad.device().type != Device::Type::CPU)
        ? grad.to(Device::cpu())
        : grad;
    Tensor g_c64 = (g_cpu.dtype() != DType::Complex64)
        ? g_cpu.to(DType::Complex64)
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

    auto win_data = resolve_window(window, n_fft, win_length);

    std::vector<float> output_data(
        static_cast<size_t>(batch_size * expected_length), 0.0f);

    const auto* in_ptr =
        reinterpret_cast<const std::complex<float>*>(g_c64.data_ptr());

    for (int64_t b = 0; b < batch_size; ++b) {
        float* out = output_data.data() + b * expected_length;

        for (int64_t f = 0; f < num_frames; ++f) {
            Tensor frame_freq({freq_bins}, DType::Complex64, Device::cpu());
            auto* freq_data =
                reinterpret_cast<std::complex<float>*>(frame_freq.data_ptr());
            for (int64_t k = 0; k < freq_bins; ++k) {
                freq_data[k] =
                    in_ptr[b * freq_bins * num_frames + k * num_frames + f];
            }

            // Inverse FFT — same norm as forward STFT used.
            Tensor time_frame;
            if (onesided) {
                time_frame = fft::irfft(frame_freq, n_fft, -1,
                                        normalized ? "ortho" : "backward");
            } else {
                time_frame = fft::ifft(frame_freq, n_fft, -1,
                                       normalized ? "ortho" : "backward");
            }

            // The adjoint of `Y = FFT(window .* x_frame)` w.r.t. x_frame is
            // `IFFT_adjoint(Y) .* window`. For backward-normalised FFT, the
            // adjoint of FFT equals N*IFFT — but combined with backward IFFT's
            // 1/N factor we get exactly `IFFT(Y)` * N, while combined with
            // ortho normalisation the adjoint equals the inverse. We compute
            // both consistently with the forward path's norm choice above and
            // then apply the (forward, not inverse) `N` scaling for the
            // backward-norm case so the per-sample relationship matches
            // `adjoint(FFT_b) = N * IFFT_b`.
            Tensor real_frame = (time_frame.dtype() == DType::Complex64 ||
                                 time_frame.dtype() == DType::Complex128)
                ? time_frame.to(DType::Float32)
                : time_frame;
            if (!real_frame.is_contiguous()) real_frame = real_frame.contiguous();
            const float* frame_data = real_frame.data<float>();

            const float n_scale = normalized ? 1.0f
                                             : static_cast<float>(n_fft);
            int64_t frame_offset = f * hop_length;
            for (int64_t i = 0; i < n_fft; ++i) {
                out[frame_offset + i] += frame_data[i] *
                                         win_data[static_cast<size_t>(i)] *
                                         n_scale;
            }
        }
    }

    // Trim center padding and resize to signal_length.
    int64_t trim_start = center ? (n_fft / 2) : 0;
    int64_t final_length =
        (signal_length > 0) ? signal_length
                            : (expected_length - 2 * trim_start);

    std::vector<int64_t> final_shape;
    for (int64_t d = 0; d < ndim - 2; ++d) final_shape.push_back(in_shape[d]);
    final_shape.push_back(final_length);

    Tensor result(final_shape, DType::Float32, Device::cpu());
    float* result_data = result.data<float>();
    for (int64_t b = 0; b < batch_size; ++b) {
        copy_with_trim(output_data.data() + b * expected_length,
                       result_data + b * final_length,
                       expected_length, trim_start, final_length);
    }

    // Restore original dtype/device.
    Tensor out_dev = (grad.device().type != Device::Type::CPU)
        ? result.to(grad.device())
        : result;
    // grad is complex; the time-domain gradient is real (float) — keep it as
    // Float32 to match the forward STFT input dtype family.
    return out_dev;
}

// True ISTFT linear adjoint: takes `grad` shape (..., signal_length) and
// returns STFT shape (..., freq_bins, num_frames). Mirrors the ISTFT
// computation but divides each extracted frame by the per-sample
// `window_sum` ISTFT applied at forward time.
auto istft_adjoint_impl(const Tensor& grad,
                        int64_t n_fft,
                        int64_t hop_length,
                        int64_t win_length,
                        const Tensor& window,
                        bool center,
                        bool normalized,
                        bool onesided) -> Tensor
{
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;

    Tensor g_cpu = (grad.device().type != Device::Type::CPU)
        ? grad.to(Device::cpu())
        : grad;
    Tensor g_f32 = (g_cpu.dtype() != DType::Float32)
        ? g_cpu.to(DType::Float32)
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

    auto win_data = resolve_window(window, n_fft, win_length);

    // Compute window_sum used by ISTFT: sum_f window^2 placed at f*hop.
    std::vector<float> window_sum_buf(
        static_cast<size_t>(expected_length), 0.0f);
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
    std::vector<float> padded_grad(
        static_cast<size_t>(batch_size * expected_length), 0.0f);
    int64_t trim_start = center ? (n_fft / 2) : 0;
    {
        const float* gptr = g_f32.data<float>();
        for (int64_t b = 0; b < batch_size; ++b) {
            float* dst = padded_grad.data() + b * expected_length;
            int64_t copy_len = std::min(signal_length,
                                        expected_length - trim_start);
            if (copy_len > 0) {
                std::memcpy(dst + trim_start, gptr + b * signal_length,
                            static_cast<size_t>(copy_len) * sizeof(float));
            }
        }
    }

    // Divide by window_sum (matches the forward ISTFT division) — entries
    // with near-zero window_sum mirror ISTFT's `if (wsum > 1e-10) out /= wsum`
    // guard so the adjoint is zero where the forward dropped the sample.
    for (int64_t b = 0; b < batch_size; ++b) {
        float* row = padded_grad.data() + b * expected_length;
        for (int64_t i = 0; i < expected_length; ++i) {
            const float w = window_sum_buf[static_cast<size_t>(i)];
            if (w > 1e-10f) {
                row[i] /= w;
            } else {
                row[i] = 0.0f;
            }
        }
    }

    // For each frame, multiply by window and apply forward FFT.
    int64_t freq_bins = onesided ? (n_fft / 2 + 1) : n_fft;

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim - 1; ++d) out_shape.push_back(in_shape[d]);
    out_shape.push_back(freq_bins);
    out_shape.push_back(num_frames);

    Tensor result(out_shape, DType::Complex64, Device::cpu());
    auto* out_ptr =
        reinterpret_cast<std::complex<float>*>(result.data_ptr());

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* row = padded_grad.data() + b * expected_length;
        for (int64_t f = 0; f < num_frames; ++f) {
            Tensor frame_t({n_fft}, DType::Float32, Device::cpu());
            float* fd = frame_t.data<float>();
            int64_t off = f * hop_length;
            for (int64_t i = 0; i < n_fft; ++i) {
                fd[i] = row[off + i] * win_data[static_cast<size_t>(i)];
            }

            Tensor freq;
            if (onesided) {
                freq = fft::rfft(frame_t, n_fft, -1,
                                 normalized ? "ortho" : "backward");
            } else {
                freq = fft::fft(frame_t, n_fft, -1,
                                normalized ? "ortho" : "backward");
            }
            if (freq.dtype() != DType::Complex64) {
                freq = freq.to(DType::Complex64);
            }
            if (!freq.is_contiguous()) freq = freq.contiguous();

            const auto* freq_ptr =
                reinterpret_cast<const std::complex<float>*>(freq.data_ptr());
            for (int64_t k = 0; k < freq_bins; ++k) {
                out_ptr[b * freq_bins * num_frames + k * num_frames + f] =
                    freq_ptr[k];
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
    return {stft_adjoint_impl(grad_outputs[0], n_fft_, hop_length_, win_length_,
                              window_, center_, normalized_, onesided_,
                              signal_length_)};
}

auto ISTFTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("ISTFTBackward::forward should not be called");
}

auto ISTFTBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // grad_outputs[0] has ISTFT output shape (..., signal_length).
    // True ISTFT linear adjoint = STFT with each frame divided by the same
    // window_sum the forward ISTFT applied.
    return {istft_adjoint_impl(grad_outputs[0], n_fft_, hop_length_, win_length_,
                               window_, center_, normalized_, onesided_)};
}

} // namespace tenzor
