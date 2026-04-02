#include "tenzor/ops/fft.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include <stdexcept>

namespace tenzor {
namespace fft {

namespace {

// Validate input is floating-point or complex
void validate_fft_input(const Tensor& input, const char* op_name) {
    auto dt = input.dtype();
    if (dt != DType::Float32 && dt != DType::Float64 &&
        dt != DType::Complex64 && dt != DType::Complex128) {
        throw std::runtime_error(
            std::string(op_name) + ": requires floating-point or complex input, got " +
            std::string(dtype_name(dt)));
    }
}

// Get complex dtype corresponding to a real dtype
DType to_complex_dtype(DType dt) {
    switch (dt) {
        case DType::Float32: return DType::Complex64;
        case DType::Float64: return DType::Complex128;
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
    Tensor inp = (input.dtype() == DType::Float32 || input.dtype() == DType::Float64)
                 ? input.to(to_complex_dtype(input.dtype())) : input;

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

    Tensor inp = (input.dtype() == DType::Float32 || input.dtype() == DType::Float64)
                 ? input.to(to_complex_dtype(input.dtype())) : input;

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

    // Default output length: 2 * (input_len - 1)
    int64_t signal_len = n.value_or(2 * (input.shape()[dim] - 1));
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
    Tensor result = input;
    for (size_t i = 0; i < dim.size(); ++i) {
        std::optional<int64_t> n_i = s ? std::make_optional((*s)[i]) : std::nullopt;
        result = ifft(result, n_i, dim[i], norm);
    }
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
    if (hop_length <= 0) hop_length = n_fft / 4;
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
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;

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

} // namespace fft
} // namespace tenzor
