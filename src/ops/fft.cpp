#include "tenzor/ops/fft.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/utils/widen_narrow.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tenzor {
namespace fft {

namespace {

// Validate input is a floating-point or complex dtype.  Half-precision
// inputs (Float16 / BFloat16) are widened to Float32 inside the CPU
// kernels via build_complex64_from_half (audit item F.10 / E.5), so we
// accept them here too.
void validate_fft_input(const Tensor& input, const char* op_name) {
    auto dt = input.dtype();
    if (dt != DType::Float32 && dt != DType::Float64 &&
        dt != DType::Float16 && dt != DType::BFloat16 &&
        dt != DType::Complex64 && dt != DType::Complex128) {
        throw std::runtime_error(
            std::string(op_name) + ": requires floating-point or complex input, got " +
            std::string(dtype_name(dt)));
    }
}

// Validate that an optional shape vector `s` matches the number of transform
// dimensions. PyTorch requires len(s) == len(dim); a mismatched length would
// otherwise index `(*s)[i]` past the vector end (OOB heap read) or silently
// drop trailing entries.
void validate_s_length(const std::optional<std::vector<int64_t>>& s,
                       size_t num_dims, const char* op_name) {
    if (s && s->size() != num_dims) {
        throw std::invalid_argument(
            std::string(op_name) + ": length of s (" + std::to_string(s->size()) +
            ") must equal the number of transformed dimensions (" +
            std::to_string(num_dims) + ")");
    }
}

// Get complex dtype corresponding to a real dtype.  Half-precision
// inputs widen to Complex64 (Float32 internal arithmetic per E.5).
DType to_complex_dtype(DType dt) {
    switch (dt) {
        case DType::Float32: return DType::Complex64;
        case DType::Float64: return DType::Complex128;
        case DType::Float16:
        case DType::BFloat16:
            return DType::Complex64;
        case DType::Complex64: return DType::Complex64;
        case DType::Complex128: return DType::Complex128;
        default: return DType::Complex64;
    }
}

// Normalize dim to positive
int64_t normalize_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("FFT: dimension out of range");
    }
    return dim;
}

} // anonymous namespace

auto fft(const Tensor& input, std::optional<int64_t> n, int64_t dim,
         const std::string& norm) -> Tensor {
    validate_fft_input(input, "fft");
    dim = normalize_dim(dim, input.ndim());

    int64_t signal_len = n.value_or(input.shape()[dim]);
    if (signal_len <= 0) {
        throw std::runtime_error("fft: n must be positive, got " + std::to_string(signal_len));
    }

    // Promote real to complex if needed
    // Real inputs widen to complex. Float16/BFloat16 must route through Float32
    // first (no direct half->complex cast); to_complex_dtype maps them to
    // Complex64. Without this the kernel received a half tensor and threw
    // "unsupported dtype (expected Complex64 or Complex128)".
    DType in_dt = input.dtype();
    Tensor inp =
        (in_dt == DType::Float16 || in_dt == DType::BFloat16)
            ? input.to(DType::Float32).to(to_complex_dtype(in_dt))
        : (in_dt == DType::Float32 || in_dt == DType::Float64)
            ? input.to(to_complex_dtype(in_dt))
            : input;

    std::array<Tensor, 1> inputs = {inp.contiguous()};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::N, signal_len);
    attrs.set(AttrKey::Norm, norm);
    return dispatch<OpId::FFT>(inputs, attrs)[0];
}

auto ifft(const Tensor& input, std::optional<int64_t> n, int64_t dim,
          const std::string& norm) -> Tensor {
    validate_fft_input(input, "ifft");
    dim = normalize_dim(dim, input.ndim());

    int64_t signal_len = n.value_or(input.shape()[dim]);
    if (signal_len <= 0) {
        throw std::runtime_error("ifft: n must be positive, got " + std::to_string(signal_len));
    }

    // Real inputs widen to complex. Float16/BFloat16 must route through Float32
    // first (no direct half->complex cast); to_complex_dtype maps them to
    // Complex64. Without this the kernel received a half tensor and threw
    // "unsupported dtype (expected Complex64 or Complex128)".
    DType in_dt = input.dtype();
    Tensor inp =
        (in_dt == DType::Float16 || in_dt == DType::BFloat16)
            ? input.to(DType::Float32).to(to_complex_dtype(in_dt))
        : (in_dt == DType::Float32 || in_dt == DType::Float64)
            ? input.to(to_complex_dtype(in_dt))
            : input;

    std::array<Tensor, 1> inputs = {inp.contiguous()};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::N, signal_len);
    attrs.set(AttrKey::Norm, norm);
    return dispatch<OpId::IFFT>(inputs, attrs)[0];
}

auto rfft(const Tensor& input, std::optional<int64_t> n, int64_t dim,
          const std::string& norm) -> Tensor {
    validate_fft_input(input, "rfft");
    dim = normalize_dim(dim, input.ndim());

    // Non-innermost transforms: decompose via transpose so every backend only needs
    // the last-dimension real transform (cuFFT/rocFFT only support last-dim rfft).
    // The reduced-size frequency axis is transposed back to `dim`.
    const int64_t last_dim_rfft = input.ndim() - 1;
    if (dim != last_dim_rfft) {
        Tensor t = transpose(input, dim, last_dim_rfft);
        Tensor r = rfft(t, n, last_dim_rfft, norm);
        return transpose(r, dim, last_dim_rfft);
    }

    int64_t signal_len = n.value_or(input.shape()[dim]);
    if (signal_len <= 0) {
        throw std::runtime_error("rfft: n must be positive, got " + std::to_string(signal_len));
    }

    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::N, signal_len);
    attrs.set(AttrKey::Norm, norm);
    return dispatch<OpId::RFFT>(inputs, attrs)[0];
}

auto irfft(const Tensor& input, std::optional<int64_t> n, int64_t dim,
           const std::string& norm) -> Tensor {
    validate_fft_input(input, "irfft");
    dim = normalize_dim(dim, input.ndim());

    // Non-innermost transforms: decompose via transpose so every backend only needs
    // the last-dimension inverse-real transform (cuFFT/rocFFT constraint).
    const int64_t last_dim_irfft = input.ndim() - 1;
    if (dim != last_dim_irfft) {
        Tensor t = transpose(input, dim, last_dim_irfft);
        Tensor r = irfft(t, n, last_dim_irfft, norm);
        return transpose(r, dim, last_dim_irfft);
    }

    // Default output length: 2 * (input_len - 1). For a 1-element spectrum
    // this default computes to 0; PyTorch's minimum valid real length is 2,
    // so fall back to 2 rather than throwing on a legitimate tiny spectrum.
    // An explicitly-supplied n is still validated for positivity.
    int64_t signal_len;
    if (n.has_value()) {
        signal_len = n.value();
    } else {
        signal_len = 2 * (input.shape()[dim] - 1);
        if (signal_len <= 0) signal_len = 2;
    }
    if (signal_len <= 0) {
        throw std::runtime_error("irfft: n must be positive, got " + std::to_string(signal_len));
    }

    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::N, signal_len);
    attrs.set(AttrKey::Norm, norm);
    return dispatch<OpId::IRFFT>(inputs, attrs)[0];
}

auto fft2(const Tensor& input, std::optional<std::vector<int64_t>> s,
          std::vector<int64_t> dim, const std::string& norm) -> Tensor {
    validate_s_length(s, dim.size(), "fft2");
    // Apply 1D FFT along each dimension sequentially
    Tensor result = input;
    for (size_t i = 0; i < dim.size(); ++i) {
        std::optional<int64_t> n_i = s ? std::make_optional((*s)[i]) : std::nullopt;
        result = fft(result, n_i, dim[i], norm);
    }
    return result;
}

auto ifft2(const Tensor& input, std::optional<std::vector<int64_t>> s,
           std::vector<int64_t> dim, const std::string& norm) -> Tensor {
    validate_s_length(s, dim.size(), "ifft2");
    Tensor result = input;
    for (size_t i = 0; i < dim.size(); ++i) {
        std::optional<int64_t> n_i = s ? std::make_optional((*s)[i]) : std::nullopt;
        result = ifft(result, n_i, dim[i], norm);
    }
    return result;
}

auto rfft2(const Tensor& input, std::optional<std::vector<int64_t>> s,
           std::vector<int64_t> dim, const std::string& norm) -> Tensor {
    validate_s_length(s, dim.size(), "rfft2");
    // Apply rfft along last dimension, then fft along remaining dimensions
    // dim should have exactly 2 elements, last one gets rfft
    std::optional<int64_t> n_last = s ? std::make_optional((*s)[dim.size() - 1]) : std::nullopt;
    Tensor result = rfft(input, n_last, dim.back(), norm);
    // Apply fft along all dimensions except the last
    for (size_t i = 0; i + 1 < dim.size(); ++i) {
        std::optional<int64_t> n_i = s ? std::make_optional((*s)[i]) : std::nullopt;
        result = fft(result, n_i, dim[i], norm);
    }
    return result;
}

auto irfft2(const Tensor& input, std::optional<std::vector<int64_t>> s,
            std::vector<int64_t> dim, const std::string& norm) -> Tensor {
    validate_s_length(s, dim.size(), "irfft2");
    // Inverse of rfft2: apply ifft along all dims except last, then irfft along last
    Tensor result = input;
    for (size_t i = 0; i + 1 < dim.size(); ++i) {
        std::optional<int64_t> n_i = s ? std::make_optional((*s)[i]) : std::nullopt;
        result = ifft(result, n_i, dim[i], norm);
    }
    std::optional<int64_t> n_last = s ? std::make_optional((*s)[dim.size() - 1]) : std::nullopt;
    result = irfft(result, n_last, dim.back(), norm);
    return result;
}

auto rfftn(const Tensor& input, std::optional<std::vector<int64_t>> s,
           std::optional<std::vector<int64_t>> dim, const std::string& norm) -> Tensor {
    std::vector<int64_t> dims;
    if (dim) {
        dims = *dim;
    } else {
        dims.resize(input.ndim());
        for (int64_t i = 0; i < input.ndim(); ++i) dims[i] = i;
    }
    validate_s_length(s, dims.size(), "rfftn");
    // Apply rfft along last dimension, then fft along remaining
    std::optional<int64_t> n_last = s ? std::make_optional((*s)[dims.size() - 1]) : std::nullopt;
    Tensor result = rfft(input, n_last, dims.back(), norm);
    for (size_t i = 0; i + 1 < dims.size(); ++i) {
        std::optional<int64_t> n_i = s ? std::make_optional((*s)[i]) : std::nullopt;
        result = fft(result, n_i, dims[i], norm);
    }
    return result;
}

auto irfftn(const Tensor& input, std::optional<std::vector<int64_t>> s,
            std::optional<std::vector<int64_t>> dim, const std::string& norm) -> Tensor {
    std::vector<int64_t> dims;
    if (dim) {
        dims = *dim;
    } else {
        dims.resize(input.ndim());
        for (int64_t i = 0; i < input.ndim(); ++i) dims[i] = i;
    }
    validate_s_length(s, dims.size(), "irfftn");
    // Apply ifft along all dims except last, then irfft along last
    Tensor result = input;
    for (size_t i = 0; i + 1 < dims.size(); ++i) {
        std::optional<int64_t> n_i = s ? std::make_optional((*s)[i]) : std::nullopt;
        result = ifft(result, n_i, dims[i], norm);
    }
    std::optional<int64_t> n_last = s ? std::make_optional((*s)[dims.size() - 1]) : std::nullopt;
    result = irfft(result, n_last, dims.back(), norm);
    return result;
}

auto fftn(const Tensor& input, std::optional<std::vector<int64_t>> s,
          std::optional<std::vector<int64_t>> dim, const std::string& norm) -> Tensor {
    std::vector<int64_t> dims;
    if (dim) {
        dims = *dim;
    } else {
        // Default: all dimensions
        dims.resize(input.ndim());
        for (int64_t i = 0; i < input.ndim(); ++i) dims[i] = i;
    }
    validate_s_length(s, dims.size(), "fftn");

    Tensor result = input;
    for (size_t i = 0; i < dims.size(); ++i) {
        std::optional<int64_t> n_i = s ? std::make_optional((*s)[i]) : std::nullopt;
        result = fft(result, n_i, dims[i], norm);
    }
    return result;
}

auto ifftn(const Tensor& input, std::optional<std::vector<int64_t>> s,
           std::optional<std::vector<int64_t>> dim, const std::string& norm) -> Tensor {
    std::vector<int64_t> dims;
    if (dim) {
        dims = *dim;
    } else {
        dims.resize(input.ndim());
        for (int64_t i = 0; i < input.ndim(); ++i) dims[i] = i;
    }
    validate_s_length(s, dims.size(), "ifftn");

    Tensor result = input;
    for (size_t i = 0; i < dims.size(); ++i) {
        std::optional<int64_t> n_i = s ? std::make_optional((*s)[i]) : std::nullopt;
        result = ifft(result, n_i, dims[i], norm);
    }
    return result;
}

auto stft(const Tensor& input, int64_t n_fft, int64_t hop_length, int64_t win_length,
          const Tensor& window, bool center, bool normalized, bool onesided) -> Tensor {
    if (n_fft <= 0) {
        throw std::runtime_error("stft: n_fft must be positive");
    }
    if (hop_length <= 0) hop_length = std::max<int64_t>(1, n_fft / 4);  // n_fft<4 would yield hop=0 -> div-by-zero in kernel
    if (win_length <= 0) win_length = n_fft;
    if (win_length > n_fft) {
        throw std::runtime_error("stft: win_length must be <= n_fft");
    }

    auto inp = input.contiguous();
    std::vector<Tensor> inputs = {inp};
    if (window.is_valid()) {
        inputs.push_back(window.contiguous());
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::NFft, n_fft);
    attrs.set(AttrKey::HopLength, hop_length);
    attrs.set(AttrKey::WinLength, win_length);
    attrs.set(AttrKey::Normalized, normalized ? int64_t{1} : int64_t{0});
    attrs.set(AttrKey::OnesidedAttr, onesided ? int64_t{1} : int64_t{0});
    attrs.set(AttrKey::Centered, center);

    return dispatch<OpId::STFT>(inputs, attrs)[0];
}

auto istft(const Tensor& input, int64_t n_fft, int64_t hop_length, int64_t win_length,
           const Tensor& window, bool center, bool normalized, bool onesided,
           std::optional<int64_t> length) -> Tensor {
    if (n_fft <= 0) {
        throw std::runtime_error("istft: n_fft must be positive");
    }
    if (hop_length <= 0) hop_length = std::max<int64_t>(1, n_fft / 4);  // n_fft<4 would yield hop=0 -> div-by-zero in kernel
    if (win_length <= 0) win_length = n_fft;
    if (win_length > n_fft) {
        throw std::runtime_error("istft: win_length must be <= n_fft");
    }

    auto inp = input.contiguous();
    std::vector<Tensor> inputs = {inp};
    if (window.is_valid()) {
        inputs.push_back(window.contiguous());
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::NFft, n_fft);
    attrs.set(AttrKey::HopLength, hop_length);
    attrs.set(AttrKey::WinLength, win_length);
    attrs.set(AttrKey::Normalized, normalized ? int64_t{1} : int64_t{0});
    attrs.set(AttrKey::OnesidedAttr, onesided ? int64_t{1} : int64_t{0});
    attrs.set(AttrKey::Centered, center);
    if (length.has_value()) {
        attrs.set(AttrKey::N, length.value());
    }

    return dispatch<OpId::ISTFT>(inputs, attrs)[0];
}

auto griffin_lim(const Tensor& magnitude,
                int64_t n_fft,
                int64_t hop_length,
                int64_t win_length,
                const Tensor& window,
                int64_t n_iter,
                double momentum) -> Tensor {
    // Canonical (Fast) Griffin-Lim phase reconstruction.
    //
    // Given a non-negative magnitude spectrogram |S|, recover a time-domain
    // signal whose STFT magnitude approximates |S|.  We use the complex-dtype
    // STFT/ISTFT (Complex64) directly rather than a real/imag last-axis
    // convention, since both round-trip ops here return / consume Complex64.
    //
    // Fast Griffin-Lim recurrence (Perraudin et al., 2013) - two-buffer form:
    //     T(s)    = STFT(ISTFT(s))                    // consistency projection
    //     P_C(s)  = polar(|S|, angle(s))              // magnitude projection
    //
    //     t_0   = polar(|S|, random_phase)
    //     y_{-1} = (unset; first iter uses momentum delta of 0)
    //     for n = 0..n_iter-1:
    //         y_n = P_C(T(t_n))
    //         if n == 0:
    //             t_{n+1} = y_n
    //         else:
    //             t_{n+1} = y_n + momentum * (y_n - y_{n-1})
    //         y_{n-1} <- y_n
    //     return ISTFT(t_{n_iter})
    //
    // momentum = 0 recovers original Griffin-Lim (1984):
    //     t_{n+1} = y_n = P_C(T(t_n)),
    // then x = ISTFT(t_{n_iter}).
    // momentum = 0.99 is the Fast Griffin-Lim default and typically converges
    // faster while remaining stable.
    if (hop_length < 0) hop_length = n_fft / 4;
    if (win_length < 0) win_length = n_fft;
    if (momentum < 0.0 || momentum >= 1.0) {
        throw std::runtime_error("griffin_lim: momentum must be in [0, 1)");
    }
    if (n_iter < 0) {
        throw std::runtime_error("griffin_lim: n_iter must be >= 0");
    }

    // Work in Float32 for the magnitude side; STFT/ISTFT operate on Float32 /
    // Complex64.  Higher-precision magnitudes are widened-narrowed.
    const DType mag_dtype = magnitude.dtype();
    Tensor mag = magnitude;
    if (mag.dtype() != DType::Float32) {
        mag = mag.to(DType::Float32);
    }
    mag = mag.contiguous();

    auto mag_shape = std::vector<int64_t>(mag.shape().begin(), mag.shape().end());

    // n_iter == 0 path: PyTorch returns ISTFT of |S| with zero phase.
    if (n_iter == 0) {
        auto zero_phase = ::tenzor::zeros(mag_shape, DType::Float32, mag.device());
        auto S0 = ::tenzor::polar(mag, zero_phase);
        auto out = istft(S0, n_fft, hop_length, win_length,
                         window, /*center=*/true, /*normalized=*/false,
                         /*onesided=*/true, std::nullopt);
        return (mag_dtype == DType::Float32) ? out : out.to(mag_dtype);
    }

    // Initial phase: uniform in [-pi, pi). Use the global generator so callers
    // that previously called manual_seed() get deterministic results.
    constexpr float k_two_pi = 6.283185307179586476925286766559f;
    constexpr float k_pi     = 3.141592653589793238462643383279f;
    auto phase0 = ::tenzor::rand(mag_shape, DType::Float32, mag.device());
    phase0 = phase0 * k_two_pi - k_pi;

    // Initial iterate t_0 = |S| * exp(i * phase0).
    Tensor t = ::tenzor::polar(mag, phase0);

    // Previous magnitude-projected spectrogram y_{n-1}.  Lazily set after the
    // first iteration so the first momentum delta is exactly zero.
    Tensor y_prev;
    bool have_y_prev = false;

    for (int64_t iter = 0; iter < n_iter; ++iter) {
        // T(t) = STFT(ISTFT(t)): project onto consistency manifold.
        auto x_hat = istft(t, n_fft, hop_length, win_length,
                           window, /*center=*/true, /*normalized=*/false,
                           /*onesided=*/true, std::nullopt);
        auto S_hat = stft(x_hat, n_fft, hop_length, win_length,
                          window, /*center=*/true, /*normalized=*/false,
                          /*onesided=*/true);
        // P_C(T(t)) = |S| * exp(i * angle(S_hat)).
        auto y = ::tenzor::polar(mag, ::tenzor::angle(S_hat));

        // Momentum update: t_{n+1} = y_n + momentum * (y_n - y_{n-1}).
        // First iteration has no y_{n-1}, so delta is zero -> t = y.
        if (!have_y_prev || momentum == 0.0) {
            t = y;
        } else {
            t = y + (y - y_prev) * momentum;
        }
        y_prev = y;
        have_y_prev = true;
    }

    // Final ISTFT of the last iterate.
    auto out = istft(t, n_fft, hop_length, win_length,
                     window, /*center=*/true, /*normalized=*/false,
                     /*onesided=*/true, std::nullopt);
    return (mag_dtype == DType::Float32) ? out : out.to(mag_dtype);
}

// ============================================================================
// fftshift / ifftshift — frequency-domain index rolling
// ============================================================================

namespace {

// Normalize a possibly-negative dim into [0, ndim).
int64_t resolve_dim(int64_t dim, int64_t ndim, const char* op_name) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument(std::string(op_name) + ": dim out of range");
    }
    return dim;
}

} // anonymous namespace

auto fftshift(const Tensor& input, std::vector<int64_t> dims) -> Tensor {
    const auto shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(shape.size());
    if (dims.empty()) {
        dims.reserve(ndim);
        for (int64_t i = 0; i < ndim; ++i) dims.push_back(i);
    }

    Tensor out = input;
    for (int64_t d : dims) {
        int64_t rd = resolve_dim(d, ndim, "fftshift");
        int64_t size = shape[rd];
        int64_t shift = size / 2;
        if (shift == 0) continue;
        out = tenzor::roll(out, shift, rd);
    }
    return out;
}

auto ifftshift(const Tensor& input, std::vector<int64_t> dims) -> Tensor {
    const auto shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(shape.size());
    if (dims.empty()) {
        dims.reserve(ndim);
        for (int64_t i = 0; i < ndim; ++i) dims.push_back(i);
    }

    Tensor out = input;
    for (int64_t d : dims) {
        int64_t rd = resolve_dim(d, ndim, "ifftshift");
        int64_t size = shape[rd];
        // For odd size, ifftshift shifts by (size + 1) / 2 to exactly invert
        // fftshift's shift of size / 2. For even sizes the two are equal.
        int64_t shift = (size + 1) / 2;
        if (shift == 0) continue;
        out = tenzor::roll(out, shift, rd);
    }
    return out;
}

// ============================================================================
// hfft / ihfft — Hermitian-symmetric transforms
// ============================================================================

auto hfft(const Tensor& input,
          std::optional<int64_t> n,
          int64_t dim,
          const std::string& norm) -> Tensor {
    // hfft(x) == irfft(conj(x))
    auto conj_x = tenzor::conj(input);
    return irfft(conj_x, n, dim, norm);
}

auto ihfft(const Tensor& input,
           std::optional<int64_t> n,
           int64_t dim,
           const std::string& norm) -> Tensor {
    // ihfft(x) == conj(rfft(x))
    auto r = rfft(input, n, dim, norm);
    return tenzor::conj(r);
}

auto fftfreq(int64_t n, double d, DType dtype, Device device) -> Tensor {
    // Returns: [0, 1, ..., n/2-1, -n/2, ..., -1] / (n * d)
    if (n < 0) {
        throw std::invalid_argument("fftfreq: n must be non-negative, got " +
                                    std::to_string(n));
    }
    if (n == 0) {
        // Empty result for n == 0 (avoids 1/(0*d) = inf -> NaN bins).
        return tenzor::empty({0}, dtype, device);
    }
    int64_t half = (n - 1) / 2 + 1;  // ceil(n/2)
    // Build the integer index ramp in Int64 (a float ramp loses exact integer
    // representation beyond 2^24) then scale by 1/(n*d) in double precision.
    double scale = 1.0 / (static_cast<double>(n) * d);
    auto pos = tenzor::arange(int64_t(0), half, int64_t(1), DType::Int64, device);
    // For n <= 1 the negative-frequency block is empty (n/2 == 0); arange would
    // throw on start==end, so emit only the single 0/(n*d) element.
    if (n / 2 == 0) {
        return tenzor::mul(pos.to(dtype),
                           tenzor::full({1}, scale, dtype, device));
    }
    auto neg = tenzor::arange(-(n / 2), int64_t(0), int64_t(1), DType::Int64, device);
    auto freqs = tenzor::cat({pos, neg}, 0).to(dtype);
    return tenzor::mul(freqs, tenzor::full({1}, scale, dtype, device));
}

auto rfftfreq(int64_t n, double d, DType dtype, Device device) -> Tensor {
    // Returns: [0, 1, ..., n/2] / (n * d)
    if (n < 0) {
        throw std::invalid_argument("rfftfreq: n must be non-negative, got " +
                                    std::to_string(n));
    }
    if (n == 0) {
        return tenzor::empty({0}, dtype, device);
    }
    int64_t half = n / 2 + 1;
    // Build the index ramp in Int64 for exact indices, then scale in double.
    double scale = 1.0 / (static_cast<double>(n) * d);
    auto freqs = tenzor::arange(int64_t(0), half, int64_t(1), DType::Int64, device);
    return tenzor::mul(freqs.to(dtype),
                       tenzor::full({1}, scale, dtype, device));
}

// ============================================================================
// DCT / IDCT — Discrete Cosine Transform via RFFT composition
// ============================================================================

namespace {

// Validate DCT type is 1-4
void validate_dct_type(int type, const char* op_name) {
    if (type < 1 || type > 4) {
        throw std::runtime_error(
            std::string(op_name) + ": type must be 1, 2, 3, or 4, got " + std::to_string(type));
    }
}

// DCT-II via RFFT: the standard "DCT" used in JPEG/MPEG.
//
// Algorithm:
//   1. Form y[k] = x[2k] for k=0..ceil(N/2)-1, x[2N-1-2k] for k=ceil(N/2)..N-1
//   2. Y = RFFT(y, N)
//   3. Multiply by twiddle factors: 2 * Re(Y[k] * exp(-j*pi*k/(2N)))
//
Tensor dct2_via_rfft(const Tensor& input, int64_t N, int64_t dim, const std::string& norm) {
    auto dtype = input.dtype();
    auto device = input.device();
    int64_t ndim = input.ndim();

    // Resolve dim
    int64_t d = dim < 0 ? dim + ndim : dim;

    // Step 1: Reorder input along dim
    // Build index tensor: [0, 2, 4, ..., 2*ceil(N/2)-2, 2*floor(N/2)-1, ..., 3, 1]
    // i.e., even indices ascending, then odd indices descending
    auto even_idx = tenzor::arange(0.0, static_cast<double>(N), 2.0, DType::Int64, device);
    // Odd indices descending: largest odd index <= N-1, then -2 each step.
    // For odd N, N-1 is EVEN, so start at N-2 (the largest odd index) instead;
    // otherwise the ramp re-emits even indices (e.g. N=3 -> [2,0] not [1]).
    const double odd_start = static_cast<double>((N % 2 == 0) ? (N - 1) : (N - 2));
    auto odd_idx = tenzor::arange(odd_start, 0.0 - 1.0, -2.0, DType::Int64, device);
    auto reorder_idx = tenzor::cat({even_idx, odd_idx}, 0);

    auto y = tenzor::index_select(input, d, reorder_idx);

    // Step 2: Full FFT of reordered signal (need all N coefficients)
    double pi = 3.14159265358979323846;
    auto Y_full = fft::fft(y, N, d, "backward");

    // Twiddle for all N coefficients
    auto k_full = tenzor::arange(0.0, static_cast<double>(N), 1.0, dtype, device);
    auto angle_full = tenzor::mul(k_full, static_cast<float>(-pi / (2.0 * N)));
    auto tw_cos_full = tenzor::cos(angle_full);
    auto tw_sin_full = tenzor::sin(angle_full);

    std::vector<int64_t> tw_full_shape(ndim, 1);
    tw_full_shape[d] = N;
    tw_cos_full = tw_cos_full.reshape(tw_full_shape);
    tw_sin_full = tw_sin_full.reshape(tw_full_shape);

    auto Yf_real = tenzor::real(Y_full);
    auto Yf_imag = tenzor::imag(Y_full);

    auto result = tenzor::sub(tenzor::mul(Yf_real, tw_cos_full),
                              tenzor::mul(Yf_imag, tw_sin_full));
    result = tenzor::mul(result, 2.0);

    // Apply normalization
    if (norm == "ortho") {
        // Ortho normalization: multiply by 1/sqrt(2*N) for all k,
        // then multiply k=0 by 1/sqrt(2) extra (or equivalently,
        // multiply all by sqrt(2/(N)) and k=0 by sqrt(1/(2*N))).
        //
        // Standard ortho: DCT[0] *= 1/sqrt(4*N), DCT[k>0] *= 1/sqrt(2*N)
        // But we already have 2* factor, so:
        // result already = 2*Re(Y*tw). Standard DCT-II is sum of x[n]*cos(pi*(2n+1)*k/(2N)).
        // With our reorder+FFT approach, the 2* gives us the right scale for "backward".
        // For "ortho": scale everything by sqrt(1/(2*N)), then k=0 by extra 1/sqrt(2).
        float scale_all = std::sqrt(1.0f / (2.0f * N));
        result = tenzor::mul(result, scale_all);

        // Scale k=0 by 1/sqrt(2)
        // Create a scale tensor: [1/sqrt(2), 1, 1, ..., 1]
        auto ortho_scale = tenzor::ones({N}, dtype, device);
        auto first_val = tenzor::full({1}, static_cast<float>(1.0 / std::sqrt(2.0)), dtype, device);
        auto rest = tenzor::ones({N - 1}, dtype, device);
        ortho_scale = tenzor::cat({first_val, rest}, 0);
        ortho_scale = ortho_scale.reshape(tw_full_shape);
        result = tenzor::mul(result, ortho_scale);
    } else if (norm == "forward") {
        // Forward normalization: divide by N
        result = tenzor::mul(result, static_cast<float>(1.0 / N));
    }
    // "backward" normalization: no additional scaling

    return result;
}

// DCT-III (inverse of DCT-II)
//
// The DCT-II backward computes:
//   C[k] = 2 * sum_{n=0}^{N-1} x[n] * cos(pi*(2n+1)*k/(2N))
//
// Its exact inverse is:
//   x[n] = (1/(2N)) * [C[0] + 2*sum_{k=1}^{N-1} C[k]*cos(pi*k*(2n+1)/(2N))]
//
// Algorithm using FFT:
//   1. Form modified coefficients: W[0] = C[0]/2, W[k] = C[k] for k>0
//      (This accounts for the half-weight on the DC term in the inverse formula.)
//   2. Multiply by twiddle: V[k] = W[k] * exp(+j*pi*k/(2N))
//   3. y_reordered = Re(IFFT(V)) * N   (IFFT backward = (1/N)*sum, multiply by N to undo)
//      This gives us (1/2)*[W[0] + 2*sum_{k=1} W[k]*cos(...)] with the *N canceling the 1/N.
//      Wait — let's be precise:
//      IFFT_backward(V)[n] = (1/N)*sum V[k]*exp(j*2*pi*k*n/N)
//      Re part = (1/N)*sum W[k]*cos(pi*k*(2n+1)/(2N))
//      Multiplied by N = sum W[k]*cos(pi*k*(2n+1)/(2N))
//        = C[0]/2 + sum_{k=1} C[k]*cos(pi*k*(2n+1)/(2N))
//      We want x[n] = (1/(2N))*[C[0] + 2*sum_{k=1} C[k]*cos(...)]
//        = (1/N)*[C[0]/2 + sum_{k=1} C[k]*cos(...)]
//      So we need result * (1/N) = IFFT_backward(V) after taking Re.
//      Actually IFFT already has the 1/N. So just take Re(IFFT(V)):
//      Re(IFFT_backward(V))[n] = (1/N)*sum W[k]*cos(pi*k*(2n+1)/(2N))
//        = (1/N)*[C[0]/2 + sum_{k=1} C[k]*cos(...)]
//        = x[n]  (for the backward norm case)
//
//   4. Un-reorder to get x from reorder(x).
//
Tensor dct3_via_rfft(const Tensor& input, int64_t N, int64_t dim, const std::string& norm) {
    auto dtype = input.dtype();
    auto device = input.device();
    int64_t ndim = input.ndim();
    int64_t d = dim < 0 ? dim + ndim : dim;
    double pi = 3.14159265358979323846;

    Tensor X = input;

    std::vector<int64_t> bcast_shape(ndim, 1);
    bcast_shape[d] = N;

    // For ortho norm: undo the ortho scaling that DCT-II ortho applied on the input,
    // converting the ortho-normalized coefficients back to "backward" norm coefficients.
    // DCT-II ortho applied: all *= sqrt(1/(2N)), k=0 *= extra 1/sqrt(2)
    // Undo: k=0 *= sqrt(2), then all /= sqrt(1/(2N)) = all *= sqrt(2N)
    if (norm == "ortho") {
        auto k0_scale = tenzor::full({1}, static_cast<float>(std::sqrt(2.0)), dtype, device);
        auto rest_ones = tenzor::ones({N - 1}, dtype, device);
        auto undo_k0 = tenzor::cat({k0_scale, rest_ones}, 0).reshape(bcast_shape);
        X = tenzor::mul(X, undo_k0);
        X = tenzor::mul(X, static_cast<float>(std::sqrt(2.0 * N)));
    }

    // Step 1: Half the DC term to account for the inverse formula weight
    // W[0] = X[0]/2, W[k>0] = X[k]
    auto dc_scale = tenzor::full({1}, 0.5f, dtype, device);
    auto rest_ones = tenzor::ones({N - 1}, dtype, device);
    auto w_scale = tenzor::cat({dc_scale, rest_ones}, 0).reshape(bcast_shape);
    auto W = tenzor::mul(X, w_scale);

    // Step 2: Twiddle factors: exp(+j*pi*k/(2N))
    auto k_full = tenzor::arange(0.0, static_cast<double>(N), 1.0, dtype, device);
    auto angle = tenzor::mul(k_full, static_cast<float>(pi / (2.0 * N)));
    auto tw_cos = tenzor::cos(angle).reshape(bcast_shape);
    auto tw_sin = tenzor::sin(angle).reshape(bcast_shape);

    // V = W * exp(+j*angle) = (W*cos(angle)) + j*(W*sin(angle))
    auto V_r = tenzor::mul(W, tw_cos);
    auto V_i = tenzor::mul(W, tw_sin);

    // Step 3: y_reordered = Re(IFFT(V))
    auto V_complex = tenzor::complex(V_r, V_i);
    auto y_complex = fft::ifft(V_complex, N, d, "backward");
    auto y_reordered = tenzor::real(y_complex);
    // y_reordered[n] = (1/N)*sum W[k]*cos(pi*k*(2n+1)/(2N))
    //                = (1/N)*[C[0]/2 + sum_{k=1} C[k]*cos(...)]
    //                = reorder(x)[n]  (for backward norm)

    // Step 4: Un-reorder
    // The DCT-II reorder: even original index i -> position i/2,
    //                      odd original index i -> position N-(i+1)/2
    // Inverse: for output position i, read from reordered position:
    //   i even -> i/2,   i odd -> N - (i+1)/2
    std::vector<int64_t> inv_idx_vec(N);
    for (int64_t i = 0; i < N; ++i) {
        if (i % 2 == 0) {
            inv_idx_vec[i] = i / 2;
        } else {
            inv_idx_vec[i] = N - (i + 1) / 2;
        }
    }
    auto inv_idx = Tensor::from_blob(inv_idx_vec.data(), {N}, DType::Int64, Device::cpu()).clone();
    if (device.type != Device::Type::CPU) {
        inv_idx = inv_idx.to(device);
    }

    auto result = tenzor::index_select(y_reordered, d, inv_idx);

    // Normalization adjustments
    if (norm == "backward") {
        // For backward norm: result already = (1/N)*[C[0]/2 + sum C[k]*cos(...)]
        // But the backward inverse of DCT-II backward needs the raw inverse:
        //   x[n] = (1/(2N))*[C[0] + 2*sum_{k=1} C[k]*cos(...)]
        //        = (1/N)*[C[0]/2 + sum_{k=1} C[k]*cos(...)]
        // This is exactly what we computed. No additional scaling.
    } else if (norm == "ortho") {
        // We already converted ortho coefficients to backward coefficients above.
        // The result is the original x. No additional scaling.
    } else {
        // "forward" norm
        result = tenzor::mul(result, static_cast<float>(N));
    }

    return result;
}

// DCT-I: X[k] = x[0] + (-1)^k * x[N-1] + 2 * sum_{n=1}^{N-2} x[n] * cos(pi*n*k/(N-1))
// This is just a scaled version of the DFT of a symmetrically extended signal.
Tensor dct1_via_rfft(const Tensor& input, int64_t N, int64_t dim, const std::string& norm) {
    if (N < 2) {
        throw std::runtime_error("dct type 1 requires signal length >= 2");
    }
    auto dtype = input.dtype();
    auto device = input.device();
    int64_t ndim = input.ndim();
    int64_t d = dim < 0 ? dim + ndim : dim;

    // DCT-I can be computed via RFFT of a mirrored signal of length 2*(N-1):
    // y = [x[0], x[1], ..., x[N-1], x[N-2], ..., x[1]]
    // DCT-I[k] = Re(FFT(y)[k]) for k=0..N-1

    // Build mirrored signal
    auto middle = tenzor::slice(input, d, 1, N - 1);  // x[1..N-2]
    auto middle_rev = tenzor::flip(middle, {d});       // x[N-2..1]
    auto mirrored = tenzor::cat({input, middle_rev}, d);

    int64_t M = 2 * (N - 1);  // length of mirrored signal

    // FFT of mirrored signal
    auto Y = fft::fft(mirrored, M, d, "backward");

    // Take first N real components
    auto result = tenzor::real(tenzor::slice(Y, d, 0, N));

    // Normalization
    if (norm == "ortho") {
        // Ortho: 1/sqrt(2*(N-1)), with endpoints scaled by 1/sqrt(2)
        float scale = std::sqrt(1.0f / (2.0f * (N - 1)));
        result = tenzor::mul(result, scale);

        std::vector<int64_t> s(ndim, 1);
        s[d] = N;
        auto edge_scale = tenzor::ones({N}, dtype, device);
        auto inv_sqrt2 = static_cast<float>(1.0 / std::sqrt(2.0));
        auto first = tenzor::full({1}, inv_sqrt2, dtype, device);
        auto last = tenzor::full({1}, inv_sqrt2, dtype, device);
        auto mid = tenzor::ones({N - 2}, dtype, device);
        edge_scale = tenzor::cat({first, mid, last}, 0).reshape(s);
        result = tenzor::mul(result, edge_scale);
    } else if (norm == "forward") {
        result = tenzor::mul(result, static_cast<float>(1.0 / (2.0 * (N - 1))));
    }

    return result;
}

// DCT-IV: X[k] = 2 * sum_{n=0}^{N-1} x[n] * cos(pi*(2n+1)*(2k+1)/(4N))
//
// Algorithm via 4N-point FFT:
//   1. Form u of length 4N: u[0]=x[0], u[4N-n]=x[n] for n=1..N-1, rest zero.
//      This time-reversal trick gives: DFT_{4N}(u)[j] = sum_n x[n]*exp(+j*pi*nj/(2N))
//   2. S[k] = DFT_{4N}(u)[2k+1] = sum_n x[n]*exp(j*pi*n*(2k+1)/(2N))
//   3. C[k] = 2*Re(exp(j*pi*(2k+1)/(4N)) * S[k])
//
// Self-inverse: DCT-IV applied twice = 2N * identity (backward norm).
// Ortho norm with sqrt(1/(2N)) scaling makes it self-inverse.
//
Tensor dct4_via_rfft(const Tensor& input, int64_t N, int64_t dim, const std::string& norm) {
    auto dtype = input.dtype();
    auto device = input.device();
    int64_t ndim = input.ndim();
    int64_t d = dim < 0 ? dim + ndim : dim;
    double pi = 3.14159265358979323846;

    int64_t M = 4 * N;

    // Step 1: Build u of length 4N.
    // u[0] = x[0], u[4N-n] = x[n] for n=1..N-1, rest zero.
    // Equivalently: u = [x[0], zeros(3N), x[1], x[2], ..., x[N-1]]
    //   where the last N-1 elements are x[1..N-1] at positions 3N+1..4N-1
    auto u_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    u_shape[d] = M;

    auto x_first = tenzor::slice(input, d, 0, 1);     // x[0], length 1

    if (N == 1) {
        // Special case: just x[0] followed by zeros
        auto zero_rest_shape = u_shape;
        zero_rest_shape[d] = M - 1;
        auto u = tenzor::cat({x_first, tenzor::zeros(zero_rest_shape, dtype, device)}, d);
        auto U = fft::fft(u, M, d, "backward");
        auto odd_idx = tenzor::arange(1.0, 2.0, 2.0, DType::Int64, device);
        auto U_odd = tenzor::index_select(U, d, odd_idx);
        auto tw_angle_val = static_cast<float>(pi / (4.0 * N));
        auto result = tenzor::mul(tenzor::real(U_odd), 2.0f * std::cos(tw_angle_val));
        if (norm == "ortho") {
            result = tenzor::mul(result, static_cast<float>(std::sqrt(1.0 / (2.0 * N))));
        } else if (norm == "forward") {
            result = tenzor::mul(result, static_cast<float>(1.0 / (2.0 * N)));
        }
        return result;
    }

    auto x_rest = tenzor::slice(input, d, 1, N);      // x[1..N-1], length N-1
    auto x_rest_rev = tenzor::flip(x_rest, {d});      // x[N-1..1], reversed

    auto zero_mid_shape = u_shape;
    zero_mid_shape[d] = 3 * N;
    auto zero_mid = tenzor::zeros(zero_mid_shape, dtype, device);

    // u = [x[0], zeros(3N), x[N-1], x[N-2], ..., x[1]]
    // so that u[4N-n] = x[n] for n=1..N-1
    auto u = tenzor::cat({x_first, zero_mid, x_rest_rev}, d);

    // Step 2: DFT of u (length 4N, backward = unnormalized)
    auto U = fft::fft(u, M, d, "backward");

    // Step 3: Extract odd DFT indices (1, 3, 5, ..., 2N-1) and apply twiddle
    // Split into real/imag before index_select (index_select may not support complex)
    auto U_real_all = tenzor::real(U);
    auto U_imag_all = tenzor::imag(U);
    auto odd_idx = tenzor::arange(1.0, static_cast<double>(2 * N), 2.0, DType::Int64, device);
    auto U_r = tenzor::index_select(U_real_all, d, odd_idx);
    auto U_i = tenzor::index_select(U_imag_all, d, odd_idx);

    // Twiddle: exp(+j*pi*(2k+1)/(4N)) for k=0..N-1
    auto k_vals = tenzor::arange(0.0, static_cast<double>(N), 1.0, dtype, device);
    // angle_k = pi*(2k+1)/(4N)
    auto tw_angle = tenzor::mul(
        tenzor::add(tenzor::mul(k_vals, 2.0f), tenzor::full({1}, 1.0f, dtype, device)),
        static_cast<float>(pi / (4.0 * N)));
    auto tw_cos = tenzor::cos(tw_angle);
    auto tw_sin = tenzor::sin(tw_angle);

    std::vector<int64_t> tw_shape(ndim, 1);
    tw_shape[d] = N;
    tw_cos = tw_cos.reshape(tw_shape);
    tw_sin = tw_sin.reshape(tw_shape);

    // (U_r + j*U_i) * (tw_cos + j*tw_sin) -> Real = U_r*tw_cos - U_i*tw_sin
    auto result = tenzor::sub(tenzor::mul(U_r, tw_cos), tenzor::mul(U_i, tw_sin));
    result = tenzor::mul(result, 2.0f);

    // Normalization
    if (norm == "ortho") {
        result = tenzor::mul(result, static_cast<float>(std::sqrt(1.0 / (2.0 * N))));
    } else if (norm == "forward") {
        result = tenzor::mul(result, static_cast<float>(1.0 / (2.0 * N)));
    }
    // backward: no scaling (raw DCT-IV with factor-2 convention)

    return result;
}

} // anonymous namespace

auto dct(const Tensor& input, int type, std::optional<int64_t> n, int64_t dim,
         const std::string& norm) -> Tensor {
    validate_fft_input(input, "dct");
    validate_dct_type(type, "dct");
    dim = normalize_dim(dim, input.ndim());

    int64_t N = n.value_or(input.shape()[dim]);
    if (N <= 0) {
        throw std::runtime_error("dct: n must be positive, got " + std::to_string(N));
    }

    // Composed RFFT-based implementation — works on all backends that support FFT.
    // Backend kernels (OpId::DCT) delegate here, so we do NOT dispatch to avoid recursion.

    // Pad or truncate to length N if needed
    Tensor x = input;
    if (input.shape()[dim] != N) {
        if (input.shape()[dim] > N) {
            x = tenzor::slice(input, dim, 0, N);
        } else {
            // Pad with zeros
            auto pad_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
            pad_shape[dim] = N - input.shape()[dim];
            auto pad = tenzor::zeros(pad_shape, input.dtype(), input.device());
            x = tenzor::cat({input, pad}, dim);
        }
    }

    switch (type) {
        case 1: return dct1_via_rfft(x, N, dim, norm);
        case 2: return dct2_via_rfft(x, N, dim, norm);
        case 3: return dct3_via_rfft(x, N, dim, norm);
        case 4: return dct4_via_rfft(x, N, dim, norm);
        default:
            throw std::runtime_error("dct: invalid type " + std::to_string(type));
    }
}

auto idct(const Tensor& input, int type, std::optional<int64_t> n, int64_t dim,
          const std::string& norm) -> Tensor {
    validate_fft_input(input, "idct");
    validate_dct_type(type, "idct");
    dim = normalize_dim(dim, input.ndim());

    int64_t N = n.value_or(input.shape()[dim]);
    if (N <= 0) {
        throw std::runtime_error("idct: n must be positive, got " + std::to_string(N));
    }

    // Composed implementation — backend kernels (OpId::IDCT) delegate here,
    // so we do NOT dispatch to avoid recursion.

    // IDCT is the inverse of DCT:
    // IDCT of type 2 = DCT of type 3 (and vice versa)
    // IDCT of type 1 = DCT of type 1 (self-inverse, up to scaling)
    // IDCT of type 4 = DCT of type 4 (self-inverse, up to scaling)

    // Pad or truncate to length N if needed
    Tensor x = input;
    if (input.shape()[dim] != N) {
        if (input.shape()[dim] > N) {
            x = tenzor::slice(input, dim, 0, N);
        } else {
            auto pad_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
            pad_shape[dim] = N - input.shape()[dim];
            auto pad = tenzor::zeros(pad_shape, input.dtype(), input.device());
            x = tenzor::cat({input, pad}, dim);
        }
    }

    // Map IDCT to the appropriate DCT type
    switch (type) {
        case 1: {
            // DCT-I is self-inverse up to scaling: T(T(x)) = 2*(N-1)*x
            if (norm == "backward") {
                auto result = dct1_via_rfft(x, N, dim, "backward");
                return tenzor::mul(result, static_cast<float>(1.0 / (2.0 * (N - 1))));
            } else if (norm == "ortho") {
                // Undo the ortho scaling on the input coefficients:
                //   ortho forward applied: all *= sqrt(1/(2*(N-1))), endpoints *= 1/sqrt(2)
                //   undo: endpoints *= sqrt(2), all *= sqrt(2*(N-1))
                int64_t dd = normalize_dim(dim, x.ndim());
                std::vector<int64_t> s(x.ndim(), 1);
                s[dd] = N;
                auto inv_sqrt2 = static_cast<float>(std::sqrt(2.0));
                auto first = tenzor::full({1}, inv_sqrt2, x.dtype(), x.device());
                auto last = tenzor::full({1}, inv_sqrt2, x.dtype(), x.device());
                auto mid = tenzor::ones({N - 2}, x.dtype(), x.device());
                auto undo_edge = tenzor::cat({first, mid, last}, 0).reshape(s);

                auto unscaled = tenzor::mul(x, undo_edge);
                unscaled = tenzor::mul(unscaled, static_cast<float>(std::sqrt(2.0 * (N - 1))));

                // Apply backward DCT-I, then scale by 1/(2*(N-1))
                auto result = dct1_via_rfft(unscaled, N, dim, "backward");
                result = tenzor::mul(result, static_cast<float>(1.0 / (2.0 * (N - 1))));
                return result;
            } else {
                // forward norm
                auto result = dct1_via_rfft(x, N, dim, "backward");
                return tenzor::mul(result, static_cast<float>(1.0 / (2.0 * (N - 1))));
            }
        }
        case 2:
            // IDCT-II = DCT-III (with appropriate normalization)
            if (norm == "ortho") {
                return dct3_via_rfft(x, N, dim, "ortho");
            } else if (norm == "backward") {
                // For "backward" DCT-II, the inverse needs 1/(2N) scaling
                return dct3_via_rfft(x, N, dim, "backward");
            } else {
                // forward
                auto result = dct3_via_rfft(x, N, dim, "backward");
                return tenzor::mul(result, static_cast<float>(1.0 / N));
            }
        case 3:
            // IDCT-III = DCT-II (with appropriate normalization)
            if (norm == "ortho") {
                return dct2_via_rfft(x, N, dim, "ortho");
            } else if (norm == "backward") {
                auto result = dct2_via_rfft(x, N, dim, "backward");
                return tenzor::mul(result, static_cast<float>(1.0 / (2.0 * N)));
            } else {
                auto result = dct2_via_rfft(x, N, dim, "backward");
                return tenzor::mul(result, static_cast<float>(1.0 / N));
            }
        case 4: {
            // DCT-IV is its own inverse (up to scaling by N/2)
            auto result = dct4_via_rfft(x, N, dim, norm);
            if (norm == "backward") {
                result = tenzor::mul(result, static_cast<float>(1.0 / (2.0 * N)));
            }
            return result;
        }
        default:
            throw std::runtime_error("idct: invalid type " + std::to_string(type));
    }
}

auto mel_scale(const Tensor& spectrogram, int64_t n_mels,
               double f_min, double f_max,
               int64_t sample_rate) -> Tensor {
    if (n_mels <= 0) {
        throw std::runtime_error("mel_scale: n_mels must be positive");
    }

    // Default f_max to Nyquist
    if (f_max <= 0.0) {
        f_max = static_cast<double>(sample_rate) / 2.0;
    }

    // Determine n_freqs from the spectrogram's second-to-last dimension
    int64_t ndim = spectrogram.ndim();
    if (ndim < 2) {
        throw std::runtime_error("mel_scale: spectrogram must have at least 2 dimensions");
    }
    int64_t n_freqs = spectrogram.shape()[ndim - 2];
    int64_t n_fft = (n_freqs - 1) * 2;

    // Hz-to-mel conversion (HTK formula)
    auto hz_to_mel = [](double hz) -> double {
        return 2595.0 * std::log10(1.0 + hz / 700.0);
    };
    auto mel_to_hz = [](double mel) -> double {
        return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
    };

    double mel_min = hz_to_mel(f_min);
    double mel_max = hz_to_mel(f_max);

    // Create n_mels+2 equally spaced mel points
    int64_t n_points = n_mels + 2;
    std::vector<double> mel_points(n_points);
    for (int64_t i = 0; i < n_points; ++i) {
        mel_points[i] = mel_min + i * (mel_max - mel_min) / (n_points - 1);
    }

    // Convert mel points to Hz and then to FFT bin indices
    std::vector<double> hz_points(n_points);
    std::vector<int64_t> bins(n_points);
    for (int64_t i = 0; i < n_points; ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
        bins[i] = static_cast<int64_t>(std::floor((n_fft + 1) * hz_points[i] / sample_rate));
        // Clamp FFT bin indices to the valid frequency range [0, n_freqs-1].
        // Without clamping, an f_max above the Nyquist frequency (or large
        // n_mels at low frequencies) produces bins that fall outside the
        // spectrogram's frequency axis. The triangular-slope loops below guard
        // writes with `k < n_freqs`, so overflowing bins silently skip their
        // slope and collapse the whole mel filter row to zero, dropping that
        // band entirely. Clamping (matching librosa/torchaudio) keeps the
        // bins in range so every filter remains a valid triangle.
        if (bins[i] < 0) {
            bins[i] = 0;
        } else if (bins[i] > n_freqs - 1) {
            bins[i] = n_freqs - 1;
        }
    }

    // Build triangular filterbank matrix (n_mels x n_freqs)
    std::vector<float> fb_data(n_mels * n_freqs, 0.0f);
    for (int64_t m = 0; m < n_mels; ++m) {
        int64_t left = bins[m];
        int64_t center = bins[m + 1];
        int64_t right = bins[m + 2];

        // Rising slope: left to center
        for (int64_t k = left; k < center && k < n_freqs; ++k) {
            if (k >= 0 && center != left) {
                fb_data[m * n_freqs + k] = static_cast<float>(
                    static_cast<double>(k - left) / static_cast<double>(center - left));
            }
        }
        // Falling slope: center to right
        for (int64_t k = center; k <= right && k < n_freqs; ++k) {
            if (k >= 0 && right != center) {
                fb_data[m * n_freqs + k] = static_cast<float>(
                    static_cast<double>(right - k) / static_cast<double>(right - center));
            }
        }
    }

    // Create filterbank tensor on CPU, then move to spectrogram's device
    auto filterbank = Tensor::from_blob(fb_data.data(), {n_mels, n_freqs},
                                        DType::Float32, Device::cpu()).clone();
    if (spectrogram.dtype() != DType::Float32) {
        filterbank = filterbank.to(spectrogram.device(), spectrogram.dtype());
    } else if (spectrogram.device().type != Device::Type::CPU) {
        filterbank = filterbank.to(spectrogram.device());
    }

    // Apply filterbank: result = filterbank @ spectrogram
    return tenzor::matmul(filterbank, spectrogram);
}

auto mfcc(const Tensor& waveform, int64_t sample_rate,
          int64_t n_mfcc, int64_t n_mels,
          int64_t n_fft, int64_t hop_length,
          double f_min, double f_max) -> Tensor {
    if (n_mfcc <= 0) {
        throw std::runtime_error("mfcc: n_mfcc must be positive");
    }
    if (n_mfcc > n_mels) {
        throw std::runtime_error("mfcc: n_mfcc must be <= n_mels");
    }

    // Step 1: Power spectrogram = |STFT(waveform)|^2
    auto stft_result = fft::stft(waveform, n_fft, hop_length);
    // Compute power from complex STFT: |z|^2 = real(z)^2 + imag(z)^2
    auto re = tenzor::real(stft_result);
    auto im = tenzor::imag(stft_result);
    auto power_spec = tenzor::add(tenzor::mul(re, re), tenzor::mul(im, im));

    // Step 2: Apply mel-scale filterbank
    auto mel_spec = fft::mel_scale(power_spec, n_mels, f_min, f_max, sample_rate);

    // Step 3: Log mel spectrogram (add epsilon for numerical stability)
    //
    // S13 dtype-preservation fix: for half-precision (Float16/BFloat16)
    // `mel_spec`, the epsilon constant 1e-10 rounds to zero in the input
    // dtype, so `mel_spec + 1e-10` is a no-op and `log(0)` becomes -inf
    // at frequency bins where the power is zero. Widen to Float32 around
    // the epsilon-shift + log so the constant remains representable,
    // then narrow the result back to the original dtype for the DCT step.
    auto log_mel = utils::widen_narrow_compute(
        mel_spec,
        [](const Tensor& ms) {
            return tenzor::log(tenzor::add(ms, 1e-10));
        });

    // Step 4: DCT (type 2, ortho normalization) along the mel dimension
    // log_mel shape: (..., n_mels, time_frames)
    // DCT along dim=-2 (the mel dimension)
    auto dct_result = fft::dct(log_mel, 2, std::nullopt, -2, "ortho");

    // Step 5: Truncate to n_mfcc coefficients along the mel dimension (dim=-2)
    int64_t ndim = dct_result.ndim();
    int64_t mel_dim = ndim - 2;
    auto result = tenzor::slice(dct_result, mel_dim, 0, n_mfcc);

    return result;
}

} // namespace fft
} // namespace tenzor
