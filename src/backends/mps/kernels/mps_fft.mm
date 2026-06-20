/**
 * @file mps_fft.mm
 * @brief Native FFT kernels for MPS backend using Apple Accelerate (vDSP)
 *
 * On Apple Silicon, MPS tensors use MTLResourceStorageModeShared (unified
 * memory), so the CPU-side vDSP functions can operate directly on the tensor
 * data pointers with ZERO copies. This is NOT a CPU fallback -- Accelerate
 * on Apple Silicon is the optimal implementation for FFT on this hardware.
 *
 * Supported operations:
 *   - FFT/IFFT:   1D complex-to-complex forward/inverse
 *   - RFFT/IRFFT: 1D real-to-complex / complex-to-real
 *   - FFT2/IFFT2: 2D complex-to-complex forward/inverse
 *   - FFTN/IFFTN: N-D complex-to-complex forward/inverse
 *   - STFT/ISTFT: Short-time Fourier transform and inverse
 */

#import <Accelerate/Accelerate.h>

#include "../mps_backend.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"

#include <cmath>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor::mps {

namespace {

// ============================================================================
// Helpers
// ============================================================================

/// Normalize dim to positive
inline int64_t normalize_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("MPS FFT: dimension out of range");
    }
    return dim;
}

/// Compute outer/inner sizes for strided FFT along a dimension
struct DimLayout {
    int64_t outer_size;
    int64_t inner_size;
};

inline DimLayout compute_dim_layout(const std::vector<int64_t>& shape, int64_t dim) {
    DimLayout layout{1, 1};
    for (int64_t i = 0; i < dim; ++i) layout.outer_size *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i)
        layout.inner_size *= shape[i];
    return layout;
}

/// Compute normalization scale factor
inline float get_norm_factor(int64_t n, std::string_view norm, bool is_forward) {
    if (norm == "ortho") {
        return 1.0f / std::sqrt(static_cast<float>(n));
    } else if ((norm == "forward" && is_forward) || (norm == "backward" && !is_forward)) {
        return 1.0f / static_cast<float>(n);
    }
    return 1.0f;
}

/// Check if n is a power of 2
inline bool is_power_of_2(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/// Compute log2 of a power-of-2 number
inline int log2_int(int64_t n) {
    int result = 0;
    while (n > 1) { n >>= 1; ++result; }
    return result;
}

/// Next power of 2 >= n
inline int64_t next_pow2(int64_t n) {
    int64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// ============================================================================
// vDSP DFT-based 1D complex FFT (handles arbitrary sizes via DFT API)
// ============================================================================

/// Perform a single 1D complex FFT of length n on contiguous float data.
/// in_real/in_imag: input split-complex (length n)
/// out_real/out_imag: output split-complex (length n)
void vdsp_fft_1d_complex(const float* in_real, const float* in_imag,
                          float* out_real, float* out_imag,
                          int64_t n, bool is_forward) {
    vDSP_DFT_Direction dir = is_forward ? vDSP_DFT_FORWARD : vDSP_DFT_INVERSE;
    vDSP_DFT_Setup setup = vDSP_DFT_zop_CreateSetup(
        nullptr, static_cast<vDSP_Length>(n), dir);

    if (setup) {
        vDSP_DFT_Execute(setup, in_real, in_imag, out_real, out_imag);
        vDSP_DFT_DestroySetup(setup);
    } else {
        // Fallback: use vDSP_fft_zop, which supports power-of-2 sizes only.
        // Zero-padding a non-power-of-2 signal up to the next power of 2 and
        // taking that transform does NOT yield the DFT of the original signal
        // (the bin frequencies differ), so silently truncating the padded
        // result would return numerically wrong values. Fail loudly instead.
        if (!is_power_of_2(n)) {
            throw std::runtime_error(
                "MPS FFT: vDSP_DFT setup unavailable for non-power-of-2 length " +
                std::to_string(n) +
                " (vDSP_DFT_zop supports only sizes f*2^k, f in {1,3,5,15})");
        }
        int64_t n_padded = next_pow2(n);
        int log2n = log2_int(n_padded);

        FFTSetup fft_setup = vDSP_create_fftsetup(static_cast<vDSP_Length>(log2n), FFT_RADIX2);
        if (!fft_setup) {
            throw std::runtime_error("MPS FFT: failed to create FFTSetup");
        }

        // Pad input to power of 2 if needed
        std::vector<float> padded_real(static_cast<size_t>(n_padded), 0.0f);
        std::vector<float> padded_imag(static_cast<size_t>(n_padded), 0.0f);
        std::memcpy(padded_real.data(), in_real, static_cast<size_t>(n) * sizeof(float));
        std::memcpy(padded_imag.data(), in_imag, static_cast<size_t>(n) * sizeof(float));

        DSPSplitComplex input_sc = {padded_real.data(), padded_imag.data()};

        std::vector<float> result_real(static_cast<size_t>(n_padded));
        std::vector<float> result_imag(static_cast<size_t>(n_padded));
        DSPSplitComplex output_sc = {result_real.data(), result_imag.data()};

        FFTDirection fft_dir = is_forward ? FFT_FORWARD : FFT_INVERSE;
        vDSP_fft_zop(fft_setup, &input_sc, 1, &output_sc, 1,
                      static_cast<vDSP_Length>(log2n), fft_dir);

        std::memcpy(out_real, result_real.data(), static_cast<size_t>(n) * sizeof(float));
        std::memcpy(out_imag, result_imag.data(), static_cast<size_t>(n) * sizeof(float));

        vDSP_destroy_fftsetup(fft_setup);
    }
}

/// Perform a single 1D real-to-complex FFT.
/// in_real: input real data (length n)
/// out_real/out_imag: output split-complex (length n/2+1)
void vdsp_rfft_1d(const float* in_real,
                   float* out_real, float* out_imag,
                   int64_t n) {
    vDSP_DFT_Setup setup = vDSP_DFT_zrop_CreateSetup(
        nullptr, static_cast<vDSP_Length>(n), vDSP_DFT_FORWARD);

    if (setup) {
        // vDSP_DFT_zrop expects split-complex input where the even/odd
        // samples are packed: in_real[i] = x[2i], in_imag[i] = x[2i+1]
        int64_t half_n = n / 2;
        std::vector<float> even(static_cast<size_t>(half_n));
        std::vector<float> odd(static_cast<size_t>(half_n));
        for (int64_t i = 0; i < half_n; ++i) {
            even[static_cast<size_t>(i)] = in_real[2 * i];
            odd[static_cast<size_t>(i)] = in_real[2 * i + 1];
        }

        std::vector<float> res_real(static_cast<size_t>(half_n));
        std::vector<float> res_imag(static_cast<size_t>(half_n));
        vDSP_DFT_Execute(setup, even.data(), odd.data(),
                          res_real.data(), res_imag.data());
        vDSP_DFT_DestroySetup(setup);

        // Unpack: DC is (res_real[0], 0), Nyquist is (res_imag[0], 0)
        out_real[0] = res_real[0];
        out_imag[0] = 0.0f;
        for (int64_t i = 1; i < half_n; ++i) {
            out_real[i] = res_real[static_cast<size_t>(i)];
            out_imag[i] = res_imag[static_cast<size_t>(i)];
        }
        out_real[half_n] = res_imag[0]; // Nyquist
        out_imag[half_n] = 0.0f;
    } else {
        // Fallback: full complex FFT then truncate
        std::vector<float> imag_zeros(static_cast<size_t>(n), 0.0f);
        std::vector<float> full_real(static_cast<size_t>(n));
        std::vector<float> full_imag(static_cast<size_t>(n));
        vdsp_fft_1d_complex(in_real, imag_zeros.data(),
                             full_real.data(), full_imag.data(),
                             n, true);
        int64_t out_len = n / 2 + 1;
        std::memcpy(out_real, full_real.data(), static_cast<size_t>(out_len) * sizeof(float));
        std::memcpy(out_imag, full_imag.data(), static_cast<size_t>(out_len) * sizeof(float));
    }
}

/// Perform a single 1D complex-to-real IFFT.
/// in_real/in_imag: input split-complex (length n/2+1)
/// out_real: output real data (length n)
void vdsp_irfft_1d(const float* in_real, const float* in_imag,
                    float* out_real, int64_t n) {
    vDSP_DFT_Setup setup = vDSP_DFT_zrop_CreateSetup(
        nullptr, static_cast<vDSP_Length>(n), vDSP_DFT_INVERSE);

    if (setup) {
        int64_t half_n = n / 2;
        // Pack: DC in real[0], Nyquist in imag[0], rest in corresponding positions
        std::vector<float> packed_real(static_cast<size_t>(half_n));
        std::vector<float> packed_imag(static_cast<size_t>(half_n));
        packed_real[0] = in_real[0];
        packed_imag[0] = in_real[half_n]; // Nyquist
        for (int64_t i = 1; i < half_n; ++i) {
            packed_real[static_cast<size_t>(i)] = in_real[i];
            packed_imag[static_cast<size_t>(i)] = in_imag[i];
        }

        std::vector<float> res_even(static_cast<size_t>(half_n));
        std::vector<float> res_odd(static_cast<size_t>(half_n));
        vDSP_DFT_Execute(setup, packed_real.data(), packed_imag.data(),
                          res_even.data(), res_odd.data());
        vDSP_DFT_DestroySetup(setup);

        // Unpack interleaved even/odd samples and scale by 1/n
        float scale = 1.0f / static_cast<float>(n);
        for (int64_t i = 0; i < half_n; ++i) {
            out_real[2 * i] = res_even[static_cast<size_t>(i)] * scale;
            out_real[2 * i + 1] = res_odd[static_cast<size_t>(i)] * scale;
        }
    } else {
        // Fallback: reconstruct full spectrum and do complex IFFT
        std::vector<float> full_real(static_cast<size_t>(n));
        std::vector<float> full_imag(static_cast<size_t>(n));
        int64_t half_n = n / 2 + 1;
        for (int64_t i = 0; i < half_n; ++i) {
            full_real[static_cast<size_t>(i)] = in_real[i];
            full_imag[static_cast<size_t>(i)] = in_imag[i];
        }
        // Hermitian symmetry
        for (int64_t i = half_n; i < n; ++i) {
            full_real[static_cast<size_t>(i)] = full_real[static_cast<size_t>(n - i)];
            full_imag[static_cast<size_t>(i)] = -full_imag[static_cast<size_t>(n - i)];
        }
        std::vector<float> res_real(static_cast<size_t>(n));
        std::vector<float> res_imag(static_cast<size_t>(n));
        vdsp_fft_1d_complex(full_real.data(), full_imag.data(),
                             res_real.data(), res_imag.data(),
                             n, false);
        float scale = 1.0f / static_cast<float>(n);
        for (int64_t i = 0; i < n; ++i) {
            out_real[i] = res_real[static_cast<size_t>(i)] * scale;
        }
    }
}

/// Scale a float array by a constant
inline void scale_array(float* data, int64_t len, float scale) {
    if (scale != 1.0f) {
        vDSP_vsmul(data, 1, &scale, data, 1, static_cast<vDSP_Length>(len));
    }
}

} // anonymous namespace

// ============================================================================
// Public kernel functions
// ============================================================================

Tensor mps_fft_kernel(const Tensor& input, int64_t dim, int64_t signal_len,
                      std::string_view norm) {
    dim = normalize_dim(dim, input.ndim());

    // Convert real input to complex (interleaved re/im as Complex64)
    Tensor inp = input;
    if (inp.dtype() == DType::Float32) {
        inp = inp.to(DType::Complex64);
    } else if (inp.dtype() != DType::Complex64) {
        throw std::runtime_error("MPS FFT: only Float32 and Complex64 supported");
    }

    auto cont = inp.contiguous();
    auto shape = cont.shape();
    int64_t N_in = shape[dim];
    int64_t N_out = signal_len;

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = N_out;
    Tensor result(out_shape, DType::Complex64, input.device());

    auto layout = compute_dim_layout(shape, dim);
    float scale = get_norm_factor(N_out, norm, /*is_forward=*/true);

    // Complex64 is interleaved {re, im} pairs -- deinterleave for vDSP
    const auto* in_ptr = reinterpret_cast<const std::complex<float>*>(cont.data_ptr());
    auto* out_ptr = reinterpret_cast<std::complex<float>*>(result.data_ptr());

    // Temporary split-complex buffers
    int64_t max_n = std::max(N_in, N_out);
    std::vector<float> in_re(static_cast<size_t>(max_n), 0.0f);
    std::vector<float> in_im(static_cast<size_t>(max_n), 0.0f);
    std::vector<float> out_re(static_cast<size_t>(N_out));
    std::vector<float> out_im(static_cast<size_t>(N_out));

    for (int64_t outer = 0; outer < layout.outer_size; ++outer) {
        for (int64_t inner = 0; inner < layout.inner_size; ++inner) {
            // Gather strided complex data into contiguous split-complex
            for (int64_t i = 0; i < N_in; ++i) {
                int64_t idx = outer * N_in * layout.inner_size + i * layout.inner_size + inner;
                in_re[static_cast<size_t>(i)] = in_ptr[idx].real();
                in_im[static_cast<size_t>(i)] = in_ptr[idx].imag();
            }
            // Zero-pad if N_out > N_in
            for (int64_t i = N_in; i < N_out; ++i) {
                in_re[static_cast<size_t>(i)] = 0.0f;
                in_im[static_cast<size_t>(i)] = 0.0f;
            }

            vdsp_fft_1d_complex(in_re.data(), in_im.data(),
                                 out_re.data(), out_im.data(),
                                 N_out, true);

            // Apply normalization and scatter back
            scale_array(out_re.data(), N_out, scale);
            scale_array(out_im.data(), N_out, scale);

            for (int64_t i = 0; i < N_out; ++i) {
                int64_t idx = outer * N_out * layout.inner_size + i * layout.inner_size + inner;
                out_ptr[idx] = std::complex<float>(out_re[static_cast<size_t>(i)],
                                                    out_im[static_cast<size_t>(i)]);
            }
        }
    }

    return result;
}

Tensor mps_ifft_kernel(const Tensor& input, int64_t dim, int64_t signal_len,
                       std::string_view norm) {
    dim = normalize_dim(dim, input.ndim());

    Tensor inp = input;
    if (inp.dtype() == DType::Float32) {
        inp = inp.to(DType::Complex64);
    } else if (inp.dtype() != DType::Complex64) {
        throw std::runtime_error("MPS IFFT: only Float32 and Complex64 supported");
    }

    auto cont = inp.contiguous();
    auto shape = cont.shape();
    int64_t N_in = shape[dim];
    int64_t N_out = signal_len;

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = N_out;
    Tensor result(out_shape, DType::Complex64, input.device());

    auto layout = compute_dim_layout(shape, dim);
    float scale = get_norm_factor(N_out, norm, /*is_forward=*/false);

    const auto* in_ptr = reinterpret_cast<const std::complex<float>*>(cont.data_ptr());
    auto* out_ptr = reinterpret_cast<std::complex<float>*>(result.data_ptr());

    int64_t max_n = std::max(N_in, N_out);
    std::vector<float> in_re(static_cast<size_t>(max_n), 0.0f);
    std::vector<float> in_im(static_cast<size_t>(max_n), 0.0f);
    std::vector<float> out_re(static_cast<size_t>(N_out));
    std::vector<float> out_im(static_cast<size_t>(N_out));

    for (int64_t outer = 0; outer < layout.outer_size; ++outer) {
        for (int64_t inner = 0; inner < layout.inner_size; ++inner) {
            for (int64_t i = 0; i < N_in; ++i) {
                int64_t idx = outer * N_in * layout.inner_size + i * layout.inner_size + inner;
                in_re[static_cast<size_t>(i)] = in_ptr[idx].real();
                in_im[static_cast<size_t>(i)] = in_ptr[idx].imag();
            }
            for (int64_t i = N_in; i < N_out; ++i) {
                in_re[static_cast<size_t>(i)] = 0.0f;
                in_im[static_cast<size_t>(i)] = 0.0f;
            }

            vdsp_fft_1d_complex(in_re.data(), in_im.data(),
                                 out_re.data(), out_im.data(),
                                 N_out, false);

            // vDSP_DFT_INVERSE is unnormalized; get_norm_factor() already
            // encodes the full inverse normalization (backward -> 1/N,
            // ortho -> 1/sqrt(N), forward -> 1.0). Apply it exactly once,
            // mirroring the forward kernel which applies `scale` alone.
            scale_array(out_re.data(), N_out, scale);
            scale_array(out_im.data(), N_out, scale);

            for (int64_t i = 0; i < N_out; ++i) {
                int64_t idx = outer * N_out * layout.inner_size + i * layout.inner_size + inner;
                out_ptr[idx] = std::complex<float>(out_re[static_cast<size_t>(i)],
                                                    out_im[static_cast<size_t>(i)]);
            }
        }
    }

    return result;
}

Tensor mps_rfft_kernel(const Tensor& input, int64_t dim, int64_t signal_len,
                       std::string_view norm) {
    dim = normalize_dim(dim, input.ndim());

    if (input.dtype() != DType::Float32) {
        throw std::runtime_error("MPS RFFT: only Float32 input supported");
    }

    auto cont = input.contiguous();
    auto shape = cont.shape();
    int64_t N_in = shape[dim];
    int64_t N = signal_len;
    int64_t N_out = N / 2 + 1;

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = N_out;
    Tensor result(out_shape, DType::Complex64, input.device());

    auto layout = compute_dim_layout(shape, dim);
    float scale = get_norm_factor(N, norm, /*is_forward=*/true);

    const float* in_data = input.data<float>();
    auto* out_ptr = reinterpret_cast<std::complex<float>*>(result.data_ptr());

    std::vector<float> padded(static_cast<size_t>(N), 0.0f);
    std::vector<float> out_re(static_cast<size_t>(N_out));
    std::vector<float> out_im(static_cast<size_t>(N_out));

    for (int64_t outer = 0; outer < layout.outer_size; ++outer) {
        for (int64_t inner = 0; inner < layout.inner_size; ++inner) {
            // Gather real data
            for (int64_t i = 0; i < std::min(N_in, N); ++i) {
                int64_t idx = outer * N_in * layout.inner_size + i * layout.inner_size + inner;
                padded[static_cast<size_t>(i)] = in_data[idx];
            }
            for (int64_t i = N_in; i < N; ++i) {
                padded[static_cast<size_t>(i)] = 0.0f;
            }

            vdsp_rfft_1d(padded.data(), out_re.data(), out_im.data(), N);

            scale_array(out_re.data(), N_out, scale);
            scale_array(out_im.data(), N_out, scale);

            for (int64_t i = 0; i < N_out; ++i) {
                int64_t idx = outer * N_out * layout.inner_size + i * layout.inner_size + inner;
                out_ptr[idx] = std::complex<float>(out_re[static_cast<size_t>(i)],
                                                    out_im[static_cast<size_t>(i)]);
            }
        }
    }

    return result;
}

Tensor mps_irfft_kernel(const Tensor& input, int64_t dim, int64_t signal_len,
                        std::string_view norm) {
    dim = normalize_dim(dim, input.ndim());

    if (input.dtype() != DType::Complex64) {
        throw std::runtime_error("MPS IRFFT: only Complex64 input supported");
    }

    auto cont = input.contiguous();
    auto shape = cont.shape();
    int64_t N_freq = shape[dim]; // n/2 + 1
    int64_t N = signal_len;

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = N;
    Tensor result(out_shape, DType::Float32, input.device());

    auto layout = compute_dim_layout(shape, dim);
    float scale = get_norm_factor(N, norm, /*is_forward=*/false);

    const auto* in_ptr = reinterpret_cast<const std::complex<float>*>(cont.data_ptr());
    float* out_data = result.data<float>();

    std::vector<float> in_re(static_cast<size_t>(N_freq));
    std::vector<float> in_im(static_cast<size_t>(N_freq));
    std::vector<float> out_real(static_cast<size_t>(N));

    for (int64_t outer = 0; outer < layout.outer_size; ++outer) {
        for (int64_t inner = 0; inner < layout.inner_size; ++inner) {
            for (int64_t i = 0; i < N_freq; ++i) {
                int64_t idx = outer * N_freq * layout.inner_size + i * layout.inner_size + inner;
                in_re[static_cast<size_t>(i)] = in_ptr[idx].real();
                in_im[static_cast<size_t>(i)] = in_ptr[idx].imag();
            }

            vdsp_irfft_1d(in_re.data(), in_im.data(), out_real.data(), N);

            // vdsp_irfft_1d already applies 1/N; adjust for normalization mode
            // "backward" (default) => 1/N already applied, scale = 1
            // "ortho" => scale = sqrt(N)/N = 1/sqrt(N), but irfft did 1/N so
            //            multiply by sqrt(N)
            // "forward" => no scaling on inverse, so multiply by N
            float extra_scale = 1.0f;
            if (norm == "ortho") {
                extra_scale = std::sqrt(static_cast<float>(N));
            } else if (norm == "forward") {
                extra_scale = static_cast<float>(N);
            }
            // Note: scale from get_norm_factor gives 1/N for backward inverse,
            // but irfft already applied 1/N. So we just apply the extra factor.
            if (extra_scale != 1.0f) {
                scale_array(out_real.data(), N, extra_scale);
            }

            for (int64_t i = 0; i < N; ++i) {
                int64_t idx = outer * N * layout.inner_size + i * layout.inner_size + inner;
                out_data[idx] = out_real[static_cast<size_t>(i)];
            }
        }
    }

    return result;
}

Tensor mps_fft2_kernel(const Tensor& input,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& signal_lengths,
                       std::string_view norm) {
    // FFT2 = FFT along dim[0], then FFT along dim[1]
    Tensor result = mps_fft_kernel(input, dims[0], signal_lengths[0], norm);
    result = mps_fft_kernel(result, dims[1], signal_lengths[1], norm);
    return result;
}

Tensor mps_ifft2_kernel(const Tensor& input,
                        const std::vector<int64_t>& dims,
                        const std::vector<int64_t>& signal_lengths,
                        std::string_view norm) {
    Tensor result = mps_ifft_kernel(input, dims[0], signal_lengths[0], norm);
    result = mps_ifft_kernel(result, dims[1], signal_lengths[1], norm);
    return result;
}

Tensor mps_fftn_kernel(const Tensor& input,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& signal_lengths,
                       std::string_view norm) {
    Tensor result = input;
    for (size_t i = 0; i < dims.size(); ++i) {
        result = mps_fft_kernel(result, dims[i], signal_lengths[i], norm);
    }
    return result;
}

Tensor mps_ifftn_kernel(const Tensor& input,
                        const std::vector<int64_t>& dims,
                        const std::vector<int64_t>& signal_lengths,
                        std::string_view norm) {
    Tensor result = input;
    for (size_t i = 0; i < dims.size(); ++i) {
        result = mps_ifft_kernel(result, dims[i], signal_lengths[i], norm);
    }
    return result;
}

Tensor mps_stft_kernel(const Tensor& input, int64_t n_fft, int64_t hop_length,
                       int64_t win_length, const Tensor& window,
                       bool center, bool normalized, bool onesided) {
    if (n_fft <= 0) {
        throw std::runtime_error("MPS STFT: n_fft must be > 0");
    }
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;
    if (win_length > n_fft) {
        throw std::runtime_error("MPS STFT: win_length must be <= n_fft");
    }

    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (ndim < 1) {
        throw std::runtime_error("MPS STFT: input must have at least 1 dimension");
    }

    int64_t signal_length = in_shape[ndim - 1];
    int64_t batch_size = 1;
    for (int64_t d = 0; d < ndim - 1; ++d) {
        batch_size *= in_shape[d];
    }

    // Ensure Float32
    Tensor input_f32 = (input.dtype() != DType::Float32) ? input.to(DType::Float32) : input;
    auto input_cont = input_f32.contiguous();
    const float* in_data = input_cont.data<float>();

    // Center padding: reflect-pad by n_fft/2 on both sides
    int64_t padded_length = signal_length;
    std::vector<float> padded_data;
    const float* src_data = nullptr;

    if (center) {
        int64_t pad = n_fft / 2;
        padded_length = signal_length + 2 * pad;
        padded_data.resize(static_cast<size_t>(batch_size * padded_length));

        for (int64_t b = 0; b < batch_size; ++b) {
            const float* sig = in_data + b * signal_length;
            float* out = padded_data.data() + b * padded_length;
            for (int64_t i = 0; i < pad; ++i) {
                int64_t idx = pad - i;
                if (idx >= signal_length) idx = signal_length - 1;
                out[i] = sig[idx];
            }
            std::memcpy(out + pad, sig, static_cast<size_t>(signal_length) * sizeof(float));
            for (int64_t i = 0; i < pad; ++i) {
                int64_t idx = signal_length - 2 - i;
                if (idx < 0) idx = 0;
                out[pad + signal_length + i] = sig[idx];
            }
        }
        src_data = padded_data.data();
    } else {
        src_data = in_data;
    }

    int64_t num_frames = (padded_length - n_fft) / hop_length + 1;
    if (num_frames <= 0) {
        throw std::runtime_error("MPS STFT: signal too short for given n_fft and hop_length");
    }

    int64_t freq_bins = onesided ? (n_fft / 2 + 1) : n_fft;

    // Build window
    std::vector<float> win_data(static_cast<size_t>(n_fft), 0.0f);
    int64_t win_offset = (n_fft - win_length) / 2;
    if (window.is_valid() && window.numel() > 0) {
        Tensor win_f32 = (window.dtype() != DType::Float32) ? window.to(DType::Float32) : window;
        auto win_cont = win_f32.contiguous();
        const float* wp = win_cont.data<float>();
        for (int64_t i = 0; i < win_length; ++i) {
            win_data[static_cast<size_t>(win_offset + i)] = wp[i];
        }
    } else {
        for (int64_t i = 0; i < win_length; ++i) {
            win_data[static_cast<size_t>(win_offset + i)] = 1.0f;
        }
    }

    // Output shape: (..., freq_bins, num_frames)
    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim - 1; ++d) out_shape.push_back(in_shape[d]);
    out_shape.push_back(freq_bins);
    out_shape.push_back(num_frames);

    Tensor result(out_shape, DType::Complex64, input.device());
    auto* out_ptr = reinterpret_cast<std::complex<float>*>(result.data_ptr());

    // Temporary buffers for per-frame FFT
    std::vector<float> frame_buf(static_cast<size_t>(n_fft));
    std::vector<float> fft_re(static_cast<size_t>(onesided ? (n_fft / 2 + 1) : n_fft));
    std::vector<float> fft_im(static_cast<size_t>(onesided ? (n_fft / 2 + 1) : n_fft));

    std::string_view fft_norm = normalized ? "ortho" : "backward";

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* sig = src_data + b * padded_length;

        for (int64_t f = 0; f < num_frames; ++f) {
            const float* frame_start = sig + f * hop_length;

            // Apply window
            vDSP_vmul(frame_start, 1, win_data.data(), 1,
                       frame_buf.data(), 1, static_cast<vDSP_Length>(n_fft));

            if (onesided) {
                vdsp_rfft_1d(frame_buf.data(), fft_re.data(), fft_im.data(), n_fft);
            } else {
                std::vector<float> zero_im(static_cast<size_t>(n_fft), 0.0f);
                vdsp_fft_1d_complex(frame_buf.data(), zero_im.data(),
                                     fft_re.data(), fft_im.data(), n_fft, true);
            }

            if (normalized) {
                float s = 1.0f / std::sqrt(static_cast<float>(n_fft));
                scale_array(fft_re.data(), freq_bins, s);
                scale_array(fft_im.data(), freq_bins, s);
            }

            for (int64_t k = 0; k < freq_bins; ++k) {
                out_ptr[b * freq_bins * num_frames + k * num_frames + f] =
                    std::complex<float>(fft_re[static_cast<size_t>(k)],
                                        fft_im[static_cast<size_t>(k)]);
            }
        }
    }

    return result;
}

Tensor mps_istft_kernel(const Tensor& input, int64_t n_fft, int64_t hop_length,
                        int64_t win_length, const Tensor& window,
                        bool center, bool normalized, bool onesided,
                        int64_t length) {
    if (n_fft <= 0) {
        throw std::runtime_error("MPS ISTFT: n_fft must be > 0");
    }
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;

    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (ndim < 2) {
        throw std::runtime_error("MPS ISTFT: input must have at least 2 dimensions");
    }

    int64_t freq_bins = in_shape[ndim - 2];
    int64_t num_frames = in_shape[ndim - 1];
    int64_t batch_size = 1;
    for (int64_t d = 0; d < ndim - 2; ++d) batch_size *= in_shape[d];

    if (input.dtype() != DType::Complex64) {
        throw std::runtime_error("MPS ISTFT: only Complex64 input supported");
    }

    auto cont = input.contiguous();
    const auto* in_ptr = reinterpret_cast<const std::complex<float>*>(cont.data_ptr());

    // Build window
    std::vector<float> win_data(static_cast<size_t>(n_fft), 0.0f);
    int64_t win_offset = (n_fft - win_length) / 2;
    if (window.is_valid() && window.numel() > 0) {
        Tensor win_f32 = (window.dtype() != DType::Float32) ? window.to(DType::Float32) : window;
        auto win_cont = win_f32.contiguous();
        const float* wp = win_cont.data<float>();
        for (int64_t i = 0; i < win_length; ++i) {
            win_data[static_cast<size_t>(win_offset + i)] = wp[i];
        }
    } else {
        for (int64_t i = 0; i < win_length; ++i) {
            win_data[static_cast<size_t>(win_offset + i)] = 1.0f;
        }
    }

    // Reconstruct signal length
    int64_t expected_length = n_fft + (num_frames - 1) * hop_length;
    int64_t out_length = (length > 0) ? length : expected_length;
    if (center) {
        int64_t pad = n_fft / 2;
        if (length <= 0) out_length = expected_length - 2 * pad;
    }

    // Output shape: (..., out_length)
    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim - 2; ++d) out_shape.push_back(in_shape[d]);
    out_shape.push_back(out_length);

    Tensor result(out_shape, DType::Float32, input.device());
    float* out_data = result.data<float>();
    std::memset(out_data, 0, static_cast<size_t>(batch_size * out_length) * sizeof(float));

    // Overlap-add buffers
    std::vector<float> full_signal(static_cast<size_t>(batch_size * expected_length), 0.0f);
    std::vector<float> window_sum(static_cast<size_t>(expected_length), 0.0f);

    // Temporary per-frame buffers
    std::vector<float> frame_re(static_cast<size_t>(n_fft));
    std::vector<float> frame_im(static_cast<size_t>(n_fft));

    for (int64_t b = 0; b < batch_size; ++b) {
        std::fill(full_signal.begin() + static_cast<size_t>(b * expected_length),
                  full_signal.begin() + static_cast<size_t>((b + 1) * expected_length), 0.0f);
        if (b == 0) {
            std::fill(window_sum.begin(), window_sum.end(), 0.0f);
        }

        for (int64_t f = 0; f < num_frames; ++f) {
            // Extract per-frame spectrum
            if (onesided) {
                std::vector<float> freq_re(static_cast<size_t>(freq_bins));
                std::vector<float> freq_im(static_cast<size_t>(freq_bins));
                for (int64_t k = 0; k < freq_bins; ++k) {
                    auto c = in_ptr[b * freq_bins * num_frames + k * num_frames + f];
                    freq_re[static_cast<size_t>(k)] = c.real();
                    freq_im[static_cast<size_t>(k)] = c.imag();
                }

                if (normalized) {
                    float s = std::sqrt(static_cast<float>(n_fft));
                    scale_array(freq_re.data(), freq_bins, s);
                    scale_array(freq_im.data(), freq_bins, s);
                }

                vdsp_irfft_1d(freq_re.data(), freq_im.data(),
                               frame_re.data(), n_fft);
            } else {
                for (int64_t k = 0; k < n_fft; ++k) {
                    auto c = in_ptr[b * freq_bins * num_frames + k * num_frames + f];
                    frame_re[static_cast<size_t>(k)] = c.real();
                    frame_im[static_cast<size_t>(k)] = c.imag();
                }

                if (normalized) {
                    float s = std::sqrt(static_cast<float>(n_fft));
                    scale_array(frame_re.data(), n_fft, s);
                    scale_array(frame_im.data(), n_fft, s);
                }

                std::vector<float> ifft_re(static_cast<size_t>(n_fft));
                std::vector<float> ifft_im(static_cast<size_t>(n_fft));
                vdsp_fft_1d_complex(frame_re.data(), frame_im.data(),
                                     ifft_re.data(), ifft_im.data(),
                                     n_fft, false);
                float inv_n = 1.0f / static_cast<float>(n_fft);
                scale_array(ifft_re.data(), n_fft, inv_n);
                std::memcpy(frame_re.data(), ifft_re.data(),
                            static_cast<size_t>(n_fft) * sizeof(float));
            }

            // Overlap-add: window and accumulate
            int64_t start = f * hop_length;
            for (int64_t i = 0; i < n_fft; ++i) {
                full_signal[static_cast<size_t>(b * expected_length + start + i)] +=
                    frame_re[static_cast<size_t>(i)] * win_data[static_cast<size_t>(i)];
                if (b == 0) {
                    window_sum[static_cast<size_t>(start + i)] +=
                        win_data[static_cast<size_t>(i)] * win_data[static_cast<size_t>(i)];
                }
            }
        }

        // Normalize by window sum
        int64_t copy_start = center ? (n_fft / 2) : 0;
        for (int64_t i = 0; i < out_length; ++i) {
            int64_t src_idx = copy_start + i;
            float ws = window_sum[static_cast<size_t>(src_idx)];
            float val = full_signal[static_cast<size_t>(b * expected_length + src_idx)];
            out_data[b * out_length + i] = (ws > 1e-10f) ? (val / ws) : val;
        }
    }

    return result;
}

} // namespace tenzor::mps
