#include "tenzor/ops/fft.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <stdexcept>
#include <cmath>
#include <complex>
#include <algorithm>

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

DType to_real_dtype(DType dt) {
    switch (dt) {
        case DType::Complex64: return DType::Float32;
        case DType::Complex128: return DType::Float64;
        case DType::Float32: return DType::Float32;
        case DType::Float64: return DType::Float64;
        default: return DType::Float32;
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

// Compute normalization scale factor
double get_norm_factor(int64_t n, const std::string& norm, bool is_forward) {
    if (norm == "ortho") {
        return 1.0 / std::sqrt(static_cast<double>(n));
    } else if ((norm == "forward" && is_forward) || (norm == "backward" && !is_forward)) {
        return 1.0 / static_cast<double>(n);
    }
    return 1.0;
}

// Compute dimension layout for operating along a single dimension
struct DimLayout {
    int64_t outer_size;   // product of dims before target dim
    int64_t inner_size;   // product of dims after target dim
};

DimLayout compute_dim_layout(std::span<const int64_t> shape, int64_t dim) {
    DimLayout layout{1, 1};
    for (int64_t i = 0; i < dim; ++i) layout.outer_size *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i)
        layout.inner_size *= shape[i];
    return layout;
}

// Naive O(n^2) 1D DFT along a single dimension of a contiguous complex tensor.
// sign = -1 for forward DFT, +1 for inverse DFT.
template<typename CType>
void naive_dft_1d(const CType* src, CType* dst,
                  int64_t N_in, int64_t N_out,
                  int64_t outer_size, int64_t inner_size,
                  double sign, double scale) {
    constexpr double TWO_PI = 2.0 * 3.14159265358979323846;
    int64_t sum_len = std::min(N_in, N_out);

    #pragma omp parallel for collapse(2) if (outer_size * inner_size * N_out > 256)
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            int64_t base_src = outer * N_in * inner_size + inner;
            int64_t base_dst = outer * N_out * inner_size + inner;

            for (int64_t k = 0; k < N_out; ++k) {
                double re = 0.0, im = 0.0;
                for (int64_t n = 0; n < sum_len; ++n) {
                    double angle = sign * TWO_PI * static_cast<double>(k) *
                                   static_cast<double>(n) / static_cast<double>(N_out);
                    double cos_a = std::cos(angle);
                    double sin_a = std::sin(angle);
                    auto& val = src[base_src + n * inner_size];
                    double x_re = static_cast<double>(val.real());
                    double x_im = static_cast<double>(val.imag());
                    re += x_re * cos_a - x_im * sin_a;
                    im += x_re * sin_a + x_im * cos_a;
                }
                using T = typename CType::value_type;
                dst[base_dst + k * inner_size] = CType(
                    static_cast<T>(re * scale), static_cast<T>(im * scale));
            }
        }
    }
}

// Naive O(n^2) IRFFT: Hermitian input (N/2+1 complex) -> real output (N elements).
// Reconstructs full spectrum from Hermitian symmetry and computes IDFT real part.
template<typename CType, typename RType>
void naive_irfft_1d(const CType* src, RType* dst,
                    int64_t N_in, int64_t N_out,
                    int64_t outer_size, int64_t inner_size,
                    double scale) {
    constexpr double TWO_PI = 2.0 * 3.14159265358979323846;

    #pragma omp parallel for collapse(2) if (outer_size * inner_size * N_out > 256)
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            int64_t base_src = outer * N_in * inner_size + inner;
            int64_t base_dst = outer * N_out * inner_size + inner;

            for (int64_t n = 0; n < N_out; ++n) {
                double re = 0.0;
                for (int64_t k = 0; k < N_out; ++k) {
                    // Reconstruct X[k] from Hermitian half
                    double xk_re, xk_im;
                    if (k < N_in) {
                        auto& val = src[base_src + k * inner_size];
                        xk_re = static_cast<double>(val.real());
                        xk_im = static_cast<double>(val.imag());
                    } else {
                        // Hermitian symmetry: X[k] = conj(X[N-k])
                        int64_t mirror = N_out - k;
                        auto& val = src[base_src + mirror * inner_size];
                        xk_re = static_cast<double>(val.real());
                        xk_im = -static_cast<double>(val.imag());
                    }

                    double angle = TWO_PI * static_cast<double>(k) *
                                   static_cast<double>(n) / static_cast<double>(N_out);
                    double cos_a = std::cos(angle);
                    double sin_a = std::sin(angle);
                    // Only need real part of X[k] * exp(+2πi*k*n/N)
                    re += xk_re * cos_a - xk_im * sin_a;
                }
                dst[base_dst + n * inner_size] = static_cast<RType>(re * scale);
            }
        }
    }
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
    DType out_dtype = to_complex_dtype(input.dtype());
    Tensor inp = (input.dtype() == DType::Float32 || input.dtype() == DType::Float64)
                 ? input.to(out_dtype) : input;

    // Try OpId dispatch
    try {
        std::array<Tensor, 1> inputs = {inp.contiguous()};
        NewOpAttributes attrs;
        attrs.set(AttrKey::Dim, dim);
        attrs.set(AttrKey::N, signal_len);
        attrs.set(AttrKey::Norm, norm);
        return dispatch<OpId::FFT>(inputs, attrs)[0];
    } catch (const std::runtime_error&) {
        // CPU inline fallback using naive DFT
    }

    // Naive O(n^2) DFT fallback
    auto cont = inp.contiguous();
    auto shape = cont.shape();
    int64_t N_in = shape[dim];
    int64_t N_out = signal_len;

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = N_out;
    auto result = Tensor(out_shape, out_dtype, input.device());

    auto layout = compute_dim_layout(shape, dim);
    double scale = get_norm_factor(N_out, norm, /*is_forward=*/true);

    if (out_dtype == DType::Complex64) {
        naive_dft_1d(cont.data<std::complex<float>>(),
                     result.data<std::complex<float>>(),
                     N_in, N_out, layout.outer_size, layout.inner_size,
                     -1.0, scale);
    } else {
        naive_dft_1d(cont.data<std::complex<double>>(),
                     result.data<std::complex<double>>(),
                     N_in, N_out, layout.outer_size, layout.inner_size,
                     -1.0, scale);
    }

    return result;
}

auto ifft(const Tensor& input, std::optional<int64_t> n, int64_t dim,
          const std::string& norm) -> Tensor {
    validate_fft_input(input, "ifft");
    dim = normalize_dim(dim, input.ndim());

    int64_t signal_len = n.value_or(input.shape()[dim]);
    if (signal_len <= 0) {
        throw std::runtime_error("ifft: n must be positive, got " + std::to_string(signal_len));
    }

    DType out_dtype = to_complex_dtype(input.dtype());
    Tensor inp = (input.dtype() == DType::Float32 || input.dtype() == DType::Float64)
                 ? input.to(out_dtype) : input;

    try {
        std::array<Tensor, 1> inputs = {inp.contiguous()};
        NewOpAttributes attrs;
        attrs.set(AttrKey::Dim, dim);
        attrs.set(AttrKey::N, signal_len);
        attrs.set(AttrKey::Norm, norm);
        return dispatch<OpId::IFFT>(inputs, attrs)[0];
    } catch (const std::runtime_error&) {
        // CPU inline fallback using naive IDFT
    }

    // Naive O(n^2) IDFT fallback (positive exponent)
    auto cont = inp.contiguous();
    auto shape = cont.shape();
    int64_t N_in = shape[dim];
    int64_t N_out = signal_len;

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = N_out;
    auto result = Tensor(out_shape, out_dtype, input.device());

    auto layout = compute_dim_layout(shape, dim);
    double scale = get_norm_factor(N_out, norm, /*is_forward=*/false);

    if (out_dtype == DType::Complex64) {
        naive_dft_1d(cont.data<std::complex<float>>(),
                     result.data<std::complex<float>>(),
                     N_in, N_out, layout.outer_size, layout.inner_size,
                     +1.0, scale);
    } else {
        naive_dft_1d(cont.data<std::complex<double>>(),
                     result.data<std::complex<double>>(),
                     N_in, N_out, layout.outer_size, layout.inner_size,
                     +1.0, scale);
    }

    return result;
}

auto rfft(const Tensor& input, std::optional<int64_t> n, int64_t dim,
          const std::string& norm) -> Tensor {
    validate_fft_input(input, "rfft");
    dim = normalize_dim(dim, input.ndim());

    int64_t signal_len = n.value_or(input.shape()[dim]);
    if (signal_len <= 0) {
        throw std::runtime_error("rfft: n must be positive, got " + std::to_string(signal_len));
    }

    try {
        std::array<Tensor, 1> inputs = {input.contiguous()};
        NewOpAttributes attrs;
        attrs.set(AttrKey::Dim, dim);
        attrs.set(AttrKey::N, signal_len);
        attrs.set(AttrKey::Norm, norm);
        return dispatch<OpId::RFFT>(inputs, attrs)[0];
    } catch (const std::runtime_error&) {
        // Fallback: compute full fft and take first n/2+1 elements
        auto full_result = fft(input, n, dim, norm);
        int64_t out_len = signal_len / 2 + 1;
        return full_result.slice(dim, 0, out_len);
    }
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

    try {
        std::array<Tensor, 1> inputs = {input.contiguous()};
        NewOpAttributes attrs;
        attrs.set(AttrKey::Dim, dim);
        attrs.set(AttrKey::N, signal_len);
        attrs.set(AttrKey::Norm, norm);
        return dispatch<OpId::IRFFT>(inputs, attrs)[0];
    } catch (const std::runtime_error&) {
        // CPU inline fallback using naive IRFFT with Hermitian reconstruction
    }

    // Naive O(n^2) IRFFT fallback: Hermitian input -> real output
    DType real_dtype = to_real_dtype(input.dtype());
    DType complex_dtype = to_complex_dtype(input.dtype());
    Tensor inp = (input.dtype() == DType::Float32 || input.dtype() == DType::Float64)
                 ? input.to(complex_dtype) : input;
    auto cont = inp.contiguous();
    auto shape = cont.shape();
    int64_t N_in = shape[dim];  // n/2+1 complex elements

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = signal_len;
    auto result = Tensor(out_shape, real_dtype, input.device());

    auto layout = compute_dim_layout(shape, dim);
    double scale = get_norm_factor(signal_len, norm, /*is_forward=*/false);

    if (complex_dtype == DType::Complex64) {
        naive_irfft_1d(cont.data<std::complex<float>>(),
                       result.data<float>(),
                       N_in, signal_len, layout.outer_size, layout.inner_size,
                       scale);
    } else {
        naive_irfft_1d(cont.data<std::complex<double>>(),
                       result.data<double>(),
                       N_in, signal_len, layout.outer_size, layout.inner_size,
                       scale);
    }

    return result;
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

} // namespace fft
} // namespace tenzor
