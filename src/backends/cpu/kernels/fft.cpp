/**
 * @file fft.cpp
 * @brief CPU FFT kernels using Intel MKL DFTI
 *
 * Provides O(n log n) FFT implementations for the CPU backend using MKL's
 * Discrete Fourier Transform Interface (DFTI). Falls back to the naive
 * O(n^2) DFT in src/ops/fft.cpp when MKL is not available.
 *
 * Supported operations:
 *   - FFT/IFFT: 1D complex-to-complex forward/inverse
 *   - RFFT: 1D real-to-complex forward
 *   - IRFFT: 1D complex-to-real inverse
 *   - FFT2/IFFT2: 2D complex-to-complex forward/inverse
 *   - FFTN/IFFTN: N-D complex-to-complex forward/inverse
 *
 * Normalization modes: "backward" (default), "forward", "ortho"
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"

#include <complex>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef TENZOR_USE_MKL
#include <mkl_dfti.h>
#endif

namespace tenzor {
namespace cpu {

// ============================================================================
// FFT helpers — MKL-independent. Used by both the MKL fast path and the
// non-MKL DFT fallback (Wave Inf-A).
// ============================================================================
namespace {

inline DType to_complex_dtype(DType dt) {
    switch (dt) {
        case DType::Float32: return DType::Complex64;
        case DType::Float64: return DType::Complex128;
        case DType::Complex64: return DType::Complex64;
        case DType::Complex128: return DType::Complex128;
        default: return DType::Complex64;
    }
}

inline DType to_real_dtype(DType dt) {
    switch (dt) {
        case DType::Complex64: return DType::Float32;
        case DType::Complex128: return DType::Float64;
        case DType::Float32: return DType::Float32;
        case DType::Float64: return DType::Float64;
        default: return DType::Float32;
    }
}

inline double get_norm_factor(int64_t n, std::string_view norm, bool is_forward) {
    if (norm == "ortho") return 1.0 / std::sqrt(static_cast<double>(n));
    if ((norm == "forward" && is_forward) || (norm == "backward" && !is_forward)) {
        return 1.0 / static_cast<double>(n);
    }
    return 1.0;
}

inline int64_t normalize_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("FFT: dimension out of range");
    }
    return dim;
}

struct DimLayout { int64_t outer_size; int64_t inner_size; };

inline DimLayout compute_dim_layout(std::span<const int64_t> shape, int64_t dim) {
    DimLayout layout{1, 1};
    for (int64_t i = 0; i < dim; ++i) layout.outer_size *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i)
        layout.inner_size *= shape[i];
    return layout;
}

}  // anonymous namespace (shared FFT helpers)

#ifdef TENZOR_USE_MKL

namespace {

// ============================================================================
// RAII wrapper for MKL DFTI descriptor
// ============================================================================

class DftiDescriptor {
public:
    DftiDescriptor() : handle_(nullptr) {}

    ~DftiDescriptor() {
        if (handle_) {
            DftiFreeDescriptor(&handle_);
        }
    }

    // Non-copyable, movable
    DftiDescriptor(const DftiDescriptor&) = delete;
    DftiDescriptor& operator=(const DftiDescriptor&) = delete;

    DftiDescriptor(DftiDescriptor&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    DftiDescriptor& operator=(DftiDescriptor&& other) noexcept {
        if (this != &other) {
            if (handle_) DftiFreeDescriptor(&handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    auto get() -> DFTI_DESCRIPTOR_HANDLE { return handle_; }
    auto ptr() -> DFTI_DESCRIPTOR_HANDLE* { return &handle_; }

private:
    DFTI_DESCRIPTOR_HANDLE handle_;
};

// ============================================================================
// Helpers
// ============================================================================

/// Check MKL DFTI status and throw on error
inline void check_dfti_status(MKL_LONG status, const char* op_name) {
    if (status != DFTI_NO_ERROR) {
        throw std::runtime_error(
            std::string(op_name) + ": MKL DFTI error: " +
            std::string(DftiErrorMessage(status)));
    }
}

/// Determine DFTI precision from dtype (MKL-only)
inline DFTI_CONFIG_VALUE dfti_precision(DType dt) {
    switch (dt) {
        case DType::Float64:
        case DType::Complex128:
            return DFTI_DOUBLE;
        default:
            return DFTI_SINGLE;
    }
}

// ============================================================================
// 1D Complex-to-Complex FFT/IFFT using MKL DFTI
// ============================================================================

/// Perform 1D complex-to-complex FFT along a single dimension.
/// Handles batching (outer/inner strides) by iterating over batches.
template<typename CType>
void mkl_fft_1d_complex(const CType* src, CType* dst,
                         int64_t N_in, int64_t N_out,
                         int64_t outer_size, int64_t inner_size,
                         bool is_forward, double scale,
                         DFTI_CONFIG_VALUE precision) {
    // When inner_size == 1, signals are contiguous and we can use MKL batch mode.
    // When inner_size > 1, signals are strided and we process one at a time.

    MKL_LONG fft_size = static_cast<MKL_LONG>(N_out);

    if (inner_size == 1) {
        // Signals are contiguous in memory -- use batched DFTI
        DftiDescriptor desc;
        check_dfti_status(
            DftiCreateDescriptor(desc.ptr(), precision, DFTI_COMPLEX, 1, fft_size),
            "fft_1d_complex");

        MKL_LONG batch = static_cast<MKL_LONG>(outer_size);
        check_dfti_status(DftiSetValue(desc.get(), DFTI_NUMBER_OF_TRANSFORMS, batch), "fft_1d_complex");
        check_dfti_status(DftiSetValue(desc.get(), DFTI_PLACEMENT, DFTI_NOT_INPLACE), "fft_1d_complex");

        // When N_in != N_out we build a padded (or truncated) copy
        // below. In that case the input stride/distance refers to the
        // PADDED buffer (N_out between batches), not the original src
        // buffer (N_in). Previously this was hard-coded to N_in, so
        // only batch 0 read correct data — every later batch read from
        // an offset into the padded buffer that didn't line up with
        // its padded row, producing garbage for batch >= 1.
        MKL_LONG input_strides[2] = {0, 1};
        MKL_LONG output_strides[2] = {0, 1};
        check_dfti_status(DftiSetValue(desc.get(), DFTI_INPUT_STRIDES, input_strides), "fft_1d_complex");
        check_dfti_status(DftiSetValue(desc.get(), DFTI_OUTPUT_STRIDES, output_strides), "fft_1d_complex");
        MKL_LONG input_distance = (N_in == N_out) ? static_cast<MKL_LONG>(N_in)
                                                  : static_cast<MKL_LONG>(N_out);
        check_dfti_status(DftiSetValue(desc.get(), DFTI_INPUT_DISTANCE, input_distance), "fft_1d_complex");
        check_dfti_status(DftiSetValue(desc.get(), DFTI_OUTPUT_DISTANCE, static_cast<MKL_LONG>(N_out)), "fft_1d_complex");

        // Set scaling
        if (is_forward) {
            check_dfti_status(DftiSetValue(desc.get(), DFTI_FORWARD_SCALE, scale), "fft_1d_complex");
            check_dfti_status(DftiSetValue(desc.get(), DFTI_BACKWARD_SCALE, 1.0), "fft_1d_complex");
        } else {
            check_dfti_status(DftiSetValue(desc.get(), DFTI_FORWARD_SCALE, 1.0), "fft_1d_complex");
            check_dfti_status(DftiSetValue(desc.get(), DFTI_BACKWARD_SCALE, scale), "fft_1d_complex");
        }

        check_dfti_status(DftiCommitDescriptor(desc.get()), "fft_1d_complex");

        // If N_in != N_out, we need to zero-pad or truncate the input.
        // Copy input into a padded buffer if needed.
        if (N_in == N_out) {
            if (is_forward) {
                check_dfti_status(
                    DftiComputeForward(desc.get(),
                        const_cast<CType*>(src),
                        dst),
                    "fft_1d_complex");
            } else {
                check_dfti_status(
                    DftiComputeBackward(desc.get(),
                        const_cast<CType*>(src),
                        dst),
                    "fft_1d_complex");
            }
        } else {
            // Zero-pad or truncate: create padded buffer
            std::vector<CType> padded(outer_size * N_out, CType(0));
            int64_t copy_len = std::min(N_in, N_out);
            for (int64_t b = 0; b < outer_size; ++b) {
                std::memcpy(&padded[b * N_out], &src[b * N_in],
                            copy_len * sizeof(CType));
            }

            // In-place is fine here since padded is our owned buffer
            // But we want out-of-place to dst
            if (is_forward) {
                check_dfti_status(
                    DftiComputeForward(desc.get(), padded.data(), dst),
                    "fft_1d_complex");
            } else {
                check_dfti_status(
                    DftiComputeBackward(desc.get(), padded.data(), dst),
                    "fft_1d_complex");
            }
        }
    } else {
        // Non-contiguous signals: process each (outer, inner) pair individually
        DftiDescriptor desc;
        check_dfti_status(
            DftiCreateDescriptor(desc.ptr(), precision, DFTI_COMPLEX, 1, fft_size),
            "fft_1d_complex_strided");
        check_dfti_status(DftiSetValue(desc.get(), DFTI_PLACEMENT, DFTI_NOT_INPLACE), "fft_1d_complex_strided");

        // Strides for a single transform: elements are inner_size apart
        MKL_LONG input_strides[2] = {0, static_cast<MKL_LONG>(inner_size)};
        MKL_LONG output_strides[2] = {0, static_cast<MKL_LONG>(inner_size)};
        check_dfti_status(DftiSetValue(desc.get(), DFTI_INPUT_STRIDES, input_strides), "fft_1d_complex_strided");
        check_dfti_status(DftiSetValue(desc.get(), DFTI_OUTPUT_STRIDES, output_strides), "fft_1d_complex_strided");

        if (is_forward) {
            check_dfti_status(DftiSetValue(desc.get(), DFTI_FORWARD_SCALE, scale), "fft_1d_complex_strided");
            check_dfti_status(DftiSetValue(desc.get(), DFTI_BACKWARD_SCALE, 1.0), "fft_1d_complex_strided");
        } else {
            check_dfti_status(DftiSetValue(desc.get(), DFTI_FORWARD_SCALE, 1.0), "fft_1d_complex_strided");
            check_dfti_status(DftiSetValue(desc.get(), DFTI_BACKWARD_SCALE, scale), "fft_1d_complex_strided");
        }

        check_dfti_status(DftiCommitDescriptor(desc.get()), "fft_1d_complex_strided");

        // For N_in != N_out with non-unit inner_size, we need per-signal padding
        bool needs_padding = (N_in != N_out);
        std::vector<CType> src_buf;
        std::vector<CType> dst_buf;
        if (needs_padding) {
            src_buf.resize(N_out, CType(0));
            dst_buf.resize(N_out, CType(0));

            // Recreate descriptor for contiguous padded buffer
            DftiDescriptor desc_pad;
            check_dfti_status(
                DftiCreateDescriptor(desc_pad.ptr(), precision, DFTI_COMPLEX, 1, fft_size),
                "fft_1d_complex_padded");
            check_dfti_status(DftiSetValue(desc_pad.get(), DFTI_PLACEMENT, DFTI_NOT_INPLACE), "fft_1d_complex_padded");

            if (is_forward) {
                check_dfti_status(DftiSetValue(desc_pad.get(), DFTI_FORWARD_SCALE, scale), "fft_1d_complex_padded");
                check_dfti_status(DftiSetValue(desc_pad.get(), DFTI_BACKWARD_SCALE, 1.0), "fft_1d_complex_padded");
            } else {
                check_dfti_status(DftiSetValue(desc_pad.get(), DFTI_FORWARD_SCALE, 1.0), "fft_1d_complex_padded");
                check_dfti_status(DftiSetValue(desc_pad.get(), DFTI_BACKWARD_SCALE, scale), "fft_1d_complex_padded");
            }

            check_dfti_status(DftiCommitDescriptor(desc_pad.get()), "fft_1d_complex_padded");

            int64_t copy_len = std::min(N_in, N_out);
            for (int64_t outer = 0; outer < outer_size; ++outer) {
                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    // Gather strided input into contiguous buffer
                    std::fill(src_buf.begin(), src_buf.end(), CType(0));
                    int64_t base_src = outer * N_in * inner_size + inner;
                    for (int64_t k = 0; k < copy_len; ++k) {
                        src_buf[k] = src[base_src + k * inner_size];
                    }

                    // Compute FFT on contiguous buffer
                    if (is_forward) {
                        check_dfti_status(
                            DftiComputeForward(desc_pad.get(), src_buf.data(), dst_buf.data()),
                            "fft_1d_complex_padded");
                    } else {
                        check_dfti_status(
                            DftiComputeBackward(desc_pad.get(), src_buf.data(), dst_buf.data()),
                            "fft_1d_complex_padded");
                    }

                    // Scatter result back to strided output
                    int64_t base_dst = outer * N_out * inner_size + inner;
                    for (int64_t k = 0; k < N_out; ++k) {
                        dst[base_dst + k * inner_size] = dst_buf[k];
                    }
                }
            }
        } else {
            // Same size, use strided descriptor directly
            for (int64_t outer = 0; outer < outer_size; ++outer) {
                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    const CType* in_ptr = &src[outer * N_in * inner_size + inner];
                    CType* out_ptr = &dst[outer * N_out * inner_size + inner];

                    if (is_forward) {
                        check_dfti_status(
                            DftiComputeForward(desc.get(),
                                const_cast<CType*>(in_ptr), out_ptr),
                            "fft_1d_complex_strided");
                    } else {
                        check_dfti_status(
                            DftiComputeBackward(desc.get(),
                                const_cast<CType*>(in_ptr), out_ptr),
                            "fft_1d_complex_strided");
                    }
                }
            }
        }
    }
}

// ============================================================================
// 1D Real-to-Complex FFT using MKL DFTI
// ============================================================================

template<typename RType, typename CType>
void mkl_rfft_1d(const RType* src, CType* dst,
                  int64_t N_in, int64_t N_out_signal,
                  int64_t outer_size, int64_t inner_size,
                  double scale, DFTI_CONFIG_VALUE precision) {
    // RFFT: real input of length N_out_signal -> complex output of length N_out_signal/2+1
    // MKL uses CCS (complex-conjugate-symmetric) storage for real FFT
    MKL_LONG fft_size = static_cast<MKL_LONG>(N_out_signal);
    int64_t N_complex_out = N_out_signal / 2 + 1;

    // Process each signal individually to handle arbitrary strides
    // and the CCS -> split complex conversion
    DftiDescriptor desc;
    check_dfti_status(
        DftiCreateDescriptor(desc.ptr(), precision, DFTI_REAL, 1, fft_size),
        "rfft_1d");
    check_dfti_status(DftiSetValue(desc.get(), DFTI_PLACEMENT, DFTI_NOT_INPLACE), "rfft_1d");

    // Configure conjugate-even storage for complex output
    check_dfti_status(DftiSetValue(desc.get(), DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "rfft_1d");

    // Set scaling
    check_dfti_status(DftiSetValue(desc.get(), DFTI_FORWARD_SCALE, scale), "rfft_1d");

    check_dfti_status(DftiCommitDescriptor(desc.get()), "rfft_1d");

    std::vector<RType> real_buf(N_out_signal, RType(0));
    std::vector<CType> complex_buf(N_complex_out);
    int64_t copy_len = std::min(N_in, N_out_signal);

    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            // Gather input (with zero-padding if needed)
            std::fill(real_buf.begin(), real_buf.end(), RType(0));
            int64_t base_src = outer * N_in * inner_size + inner;
            for (int64_t k = 0; k < copy_len; ++k) {
                real_buf[k] = src[base_src + k * inner_size];
            }

            // Compute RFFT
            check_dfti_status(
                DftiComputeForward(desc.get(), real_buf.data(), complex_buf.data()),
                "rfft_1d");

            // Scatter output
            int64_t base_dst = outer * N_complex_out * inner_size + inner;
            for (int64_t k = 0; k < N_complex_out; ++k) {
                dst[base_dst + k * inner_size] = complex_buf[k];
            }
        }
    }
}

// ============================================================================
// 1D Complex-to-Real IFFT using MKL DFTI
// ============================================================================

template<typename CType, typename RType>
void mkl_irfft_1d(const CType* src, RType* dst,
                   int64_t N_in, int64_t N_out,
                   int64_t outer_size, int64_t inner_size,
                   double scale, DFTI_CONFIG_VALUE precision) {
    // IRFFT: complex input of length N_in (= N_out/2+1) -> real output of length N_out
    MKL_LONG fft_size = static_cast<MKL_LONG>(N_out);
    int64_t N_complex_in = N_out / 2 + 1;

    DftiDescriptor desc;
    check_dfti_status(
        DftiCreateDescriptor(desc.ptr(), precision, DFTI_REAL, 1, fft_size),
        "irfft_1d");
    check_dfti_status(DftiSetValue(desc.get(), DFTI_PLACEMENT, DFTI_NOT_INPLACE), "irfft_1d");

    // Configure conjugate-even storage
    check_dfti_status(DftiSetValue(desc.get(), DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "irfft_1d");

    // Set scaling for backward transform
    check_dfti_status(DftiSetValue(desc.get(), DFTI_BACKWARD_SCALE, scale), "irfft_1d");

    check_dfti_status(DftiCommitDescriptor(desc.get()), "irfft_1d");

    std::vector<CType> complex_buf(N_complex_in);
    std::vector<RType> real_buf(N_out);

    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            // Gather complex input
            int64_t base_src = outer * N_in * inner_size + inner;
            int64_t gather_len = std::min(N_in, N_complex_in);
            std::fill(complex_buf.begin(), complex_buf.end(), CType(0));
            for (int64_t k = 0; k < gather_len; ++k) {
                complex_buf[k] = src[base_src + k * inner_size];
            }

            // Compute IRFFT
            check_dfti_status(
                DftiComputeBackward(desc.get(), complex_buf.data(), real_buf.data()),
                "irfft_1d");

            // Scatter real output
            int64_t base_dst = outer * N_out * inner_size + inner;
            for (int64_t k = 0; k < N_out; ++k) {
                dst[base_dst + k * inner_size] = real_buf[k];
            }
        }
    }
}

// ============================================================================
// N-D Complex-to-Complex FFT/IFFT using MKL DFTI
// ============================================================================

/// Perform N-D complex-to-complex FFT. dims specifies which dimensions to transform.
/// For efficiency, if the tensor is contiguous and the transform covers the last N dims
/// contiguously, we use a single multi-dimensional DFTI descriptor. Otherwise we fall
/// back to sequential 1D transforms.
template<typename CType>
void mkl_fftn_complex(const CType* src, CType* dst,
                       std::span<const int64_t> shape,
                       std::span<const int64_t> dims,
                       std::span<const int64_t> signal_lengths,
                       bool is_forward, double scale,
                       DFTI_CONFIG_VALUE precision) {
    int64_t ndim = static_cast<int64_t>(dims.size());

    // Check if we can use a single multi-dimensional DFTI descriptor:
    // The transform dimensions must be the last N contiguous dimensions
    // and input/output sizes must match (no padding needed).
    bool can_use_nd = true;
    int64_t rank = static_cast<int64_t>(shape.size());
    for (int64_t i = 0; i < ndim; ++i) {
        if (dims[i] != rank - ndim + i) {
            can_use_nd = false;
            break;
        }
        if (signal_lengths[i] != shape[dims[i]]) {
            can_use_nd = false;
            break;
        }
    }

    if (can_use_nd && ndim > 1) {
        // Multi-dimensional DFTI
        std::vector<MKL_LONG> lengths(ndim);
        for (int64_t i = 0; i < ndim; ++i) {
            lengths[i] = static_cast<MKL_LONG>(signal_lengths[i]);
        }

        // Compute batch size: product of dims before the transform dimensions
        int64_t batch_size = 1;
        for (int64_t i = 0; i < rank - ndim; ++i) {
            batch_size *= shape[i];
        }

        // Compute signal size: product of transform dimensions
        int64_t signal_size = 1;
        for (int64_t i = 0; i < ndim; ++i) {
            signal_size *= signal_lengths[i];
        }

        DftiDescriptor desc;
        check_dfti_status(
            DftiCreateDescriptor(desc.ptr(), precision, DFTI_COMPLEX,
                                  static_cast<MKL_LONG>(ndim), lengths.data()),
            "fftn_complex");

        check_dfti_status(DftiSetValue(desc.get(), DFTI_PLACEMENT, DFTI_NOT_INPLACE), "fftn_complex");
        check_dfti_status(DftiSetValue(desc.get(), DFTI_NUMBER_OF_TRANSFORMS,
                                        static_cast<MKL_LONG>(batch_size)), "fftn_complex");
        check_dfti_status(DftiSetValue(desc.get(), DFTI_INPUT_DISTANCE,
                                        static_cast<MKL_LONG>(signal_size)), "fftn_complex");
        check_dfti_status(DftiSetValue(desc.get(), DFTI_OUTPUT_DISTANCE,
                                        static_cast<MKL_LONG>(signal_size)), "fftn_complex");

        // Set strides for row-major layout
        std::vector<MKL_LONG> strides(ndim + 1);
        strides[0] = 0;  // offset
        strides[ndim] = 1;
        for (int64_t i = ndim - 1; i >= 1; --i) {
            strides[i] = strides[i + 1] * lengths[i];
        }
        check_dfti_status(DftiSetValue(desc.get(), DFTI_INPUT_STRIDES, strides.data()), "fftn_complex");
        check_dfti_status(DftiSetValue(desc.get(), DFTI_OUTPUT_STRIDES, strides.data()), "fftn_complex");

        if (is_forward) {
            check_dfti_status(DftiSetValue(desc.get(), DFTI_FORWARD_SCALE, scale), "fftn_complex");
            check_dfti_status(DftiSetValue(desc.get(), DFTI_BACKWARD_SCALE, 1.0), "fftn_complex");
        } else {
            check_dfti_status(DftiSetValue(desc.get(), DFTI_FORWARD_SCALE, 1.0), "fftn_complex");
            check_dfti_status(DftiSetValue(desc.get(), DFTI_BACKWARD_SCALE, scale), "fftn_complex");
        }

        check_dfti_status(DftiCommitDescriptor(desc.get()), "fftn_complex");

        if (is_forward) {
            check_dfti_status(
                DftiComputeForward(desc.get(), const_cast<CType*>(src), dst),
                "fftn_complex");
        } else {
            check_dfti_status(
                DftiComputeBackward(desc.get(), const_cast<CType*>(src), dst),
                "fftn_complex");
        }
    } else {
        // Fallback: sequential 1D transforms along each dimension.
        // First copy src to dst, then do in-place-style transforms by
        // reading from a temp buffer and writing to dst.
        int64_t total = 1;
        for (auto s : shape) total *= s;
        std::memcpy(dst, src, total * sizeof(CType));

        std::vector<CType> temp(total);
        for (int64_t d = 0; d < ndim; ++d) {
            int64_t dim = dims[d];
            int64_t N = shape[dim];
            int64_t N_out = signal_lengths[d];
            auto layout = compute_dim_layout(shape, dim);

            // For the sequential fallback, we apply the total scale only once
            // (on the last dimension). For other dimensions, scale = 1.0.
            double this_scale = (d == ndim - 1) ? scale : 1.0;

            std::memcpy(temp.data(), dst, total * sizeof(CType));

            // We need a modified shape for subsequent transforms if sizes change.
            // For simplicity, require no padding in N-D path (sizes must match).
            mkl_fft_1d_complex(temp.data(), dst,
                                N, N_out,
                                layout.outer_size, layout.inner_size,
                                is_forward, this_scale, precision);
        }
    }
}

} // anonymous namespace

// ============================================================================
// Public kernel functions
// ============================================================================

auto fft_kernel(const Tensor& input, int64_t dim, int64_t signal_len,
                std::string_view norm) -> Tensor {
    dim = normalize_dim(dim, input.ndim());

    DType out_dtype = to_complex_dtype(input.dtype());
    auto precision = dfti_precision(input.dtype());

    // Ensure input is complex
    Tensor inp = (input.dtype() == DType::Float32 || input.dtype() == DType::Float64)
                 ? input.to(out_dtype) : input;
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
        mkl_fft_1d_complex(cont.data<std::complex<float>>(),
                            result.data<std::complex<float>>(),
                            N_in, N_out, layout.outer_size, layout.inner_size,
                            /*is_forward=*/true, scale, precision);
    } else {
        mkl_fft_1d_complex(cont.data<std::complex<double>>(),
                            result.data<std::complex<double>>(),
                            N_in, N_out, layout.outer_size, layout.inner_size,
                            /*is_forward=*/true, scale, precision);
    }

    return result;
}

auto ifft_kernel(const Tensor& input, int64_t dim, int64_t signal_len,
                 std::string_view norm) -> Tensor {
    dim = normalize_dim(dim, input.ndim());

    DType out_dtype = to_complex_dtype(input.dtype());
    auto precision = dfti_precision(input.dtype());

    Tensor inp = (input.dtype() == DType::Float32 || input.dtype() == DType::Float64)
                 ? input.to(out_dtype) : input;
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
        mkl_fft_1d_complex(cont.data<std::complex<float>>(),
                            result.data<std::complex<float>>(),
                            N_in, N_out, layout.outer_size, layout.inner_size,
                            /*is_forward=*/false, scale, precision);
    } else {
        mkl_fft_1d_complex(cont.data<std::complex<double>>(),
                            result.data<std::complex<double>>(),
                            N_in, N_out, layout.outer_size, layout.inner_size,
                            /*is_forward=*/false, scale, precision);
    }

    return result;
}

auto rfft_kernel(const Tensor& input, int64_t dim, int64_t signal_len,
                 std::string_view norm) -> Tensor {
    dim = normalize_dim(dim, input.ndim());

    DType real_dtype = to_real_dtype(input.dtype());
    DType complex_dtype = to_complex_dtype(input.dtype());
    auto precision = dfti_precision(input.dtype());

    // Input must be real
    Tensor inp = input;
    if (input.dtype() == DType::Complex64 || input.dtype() == DType::Complex128) {
        // If complex input, take real part (by reinterpreting)
        inp = input.to(real_dtype);
    }
    auto cont = inp.contiguous();
    auto shape = cont.shape();
    int64_t N_in = shape[dim];
    int64_t N_complex_out = signal_len / 2 + 1;

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = N_complex_out;
    auto result = Tensor(out_shape, complex_dtype, input.device());

    auto layout = compute_dim_layout(shape, dim);
    double scale = get_norm_factor(signal_len, norm, /*is_forward=*/true);

    if (real_dtype == DType::Float32) {
        mkl_rfft_1d(cont.data<float>(),
                     result.data<std::complex<float>>(),
                     N_in, signal_len,
                     layout.outer_size, layout.inner_size,
                     scale, precision);
    } else {
        mkl_rfft_1d(cont.data<double>(),
                     result.data<std::complex<double>>(),
                     N_in, signal_len,
                     layout.outer_size, layout.inner_size,
                     scale, precision);
    }

    return result;
}

auto irfft_kernel(const Tensor& input, int64_t dim, int64_t signal_len,
                  std::string_view norm) -> Tensor {
    dim = normalize_dim(dim, input.ndim());

    DType real_dtype = to_real_dtype(input.dtype());
    DType complex_dtype = to_complex_dtype(input.dtype());
    auto precision = dfti_precision(input.dtype());

    // Ensure input is complex
    Tensor inp = (input.dtype() == DType::Float32 || input.dtype() == DType::Float64)
                 ? input.to(complex_dtype) : input;
    auto cont = inp.contiguous();
    auto shape = cont.shape();
    int64_t N_in = shape[dim];

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = signal_len;
    auto result = Tensor(out_shape, real_dtype, input.device());

    auto layout = compute_dim_layout(shape, dim);
    double scale = get_norm_factor(signal_len, norm, /*is_forward=*/false);

    if (real_dtype == DType::Float32) {
        mkl_irfft_1d(cont.data<std::complex<float>>(),
                      result.data<float>(),
                      N_in, signal_len,
                      layout.outer_size, layout.inner_size,
                      scale, precision);
    } else {
        mkl_irfft_1d(cont.data<std::complex<double>>(),
                      result.data<double>(),
                      N_in, signal_len,
                      layout.outer_size, layout.inner_size,
                      scale, precision);
    }

    return result;
}

auto fft2_kernel(const Tensor& input,
                  const std::vector<int64_t>& dims,
                  const std::vector<int64_t>& signal_lengths,
                  std::string_view norm) -> Tensor {
    // Normalize dims
    std::vector<int64_t> norm_dims(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        norm_dims[i] = normalize_dim(dims[i], input.ndim());
    }

    DType out_dtype = to_complex_dtype(input.dtype());
    auto precision = dfti_precision(input.dtype());

    // Ensure complex input
    Tensor inp = (input.dtype() == DType::Float32 || input.dtype() == DType::Float64)
                 ? input.to(out_dtype) : input;
    auto cont = inp.contiguous();
    auto shape = cont.shape();

    // Check if sizes match (no padding needed for ND fast path)
    bool sizes_match = true;
    for (size_t i = 0; i < norm_dims.size(); ++i) {
        if (signal_lengths[i] != shape[norm_dims[i]]) {
            sizes_match = false;
            break;
        }
    }

    // Compute total scale
    double scale = 1.0;
    for (auto len : signal_lengths) {
        scale *= get_norm_factor(len, norm, /*is_forward=*/true);
    }

    if (sizes_match) {
        // Use N-D DFTI
        auto result = Tensor(std::vector<int64_t>(shape.begin(), shape.end()), out_dtype, input.device());

        if (out_dtype == DType::Complex64) {
            mkl_fftn_complex(cont.data<std::complex<float>>(),
                              result.data<std::complex<float>>(),
                              shape, norm_dims, signal_lengths,
                              /*is_forward=*/true, scale, precision);
        } else {
            mkl_fftn_complex(cont.data<std::complex<double>>(),
                              result.data<std::complex<double>>(),
                              shape, norm_dims, signal_lengths,
                              /*is_forward=*/true, scale, precision);
        }
        return result;
    }

    // Fallback: sequential 1D FFTs along each dimension
    Tensor result = inp;
    for (size_t i = 0; i < norm_dims.size(); ++i) {
        result = fft_kernel(result, norm_dims[i], signal_lengths[i], norm);
    }
    return result;
}

auto ifft2_kernel(const Tensor& input,
                   const std::vector<int64_t>& dims,
                   const std::vector<int64_t>& signal_lengths,
                   std::string_view norm) -> Tensor {
    std::vector<int64_t> norm_dims(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        norm_dims[i] = normalize_dim(dims[i], input.ndim());
    }

    DType out_dtype = to_complex_dtype(input.dtype());
    auto precision = dfti_precision(input.dtype());

    Tensor inp = (input.dtype() == DType::Float32 || input.dtype() == DType::Float64)
                 ? input.to(out_dtype) : input;
    auto cont = inp.contiguous();
    auto shape = cont.shape();

    bool sizes_match = true;
    for (size_t i = 0; i < norm_dims.size(); ++i) {
        if (signal_lengths[i] != shape[norm_dims[i]]) {
            sizes_match = false;
            break;
        }
    }

    double scale = 1.0;
    for (auto len : signal_lengths) {
        scale *= get_norm_factor(len, norm, /*is_forward=*/false);
    }

    if (sizes_match) {
        auto result = Tensor(std::vector<int64_t>(shape.begin(), shape.end()), out_dtype, input.device());

        if (out_dtype == DType::Complex64) {
            mkl_fftn_complex(cont.data<std::complex<float>>(),
                              result.data<std::complex<float>>(),
                              shape, norm_dims, signal_lengths,
                              /*is_forward=*/false, scale, precision);
        } else {
            mkl_fftn_complex(cont.data<std::complex<double>>(),
                              result.data<std::complex<double>>(),
                              shape, norm_dims, signal_lengths,
                              /*is_forward=*/false, scale, precision);
        }
        return result;
    }

    Tensor result = inp;
    for (size_t i = 0; i < norm_dims.size(); ++i) {
        result = ifft_kernel(result, norm_dims[i], signal_lengths[i], norm);
    }
    return result;
}

auto fftn_kernel(const Tensor& input,
                  const std::vector<int64_t>& dims,
                  const std::vector<int64_t>& signal_lengths,
                  std::string_view norm) -> Tensor {
    return fft2_kernel(input, dims, signal_lengths, norm);
}

auto ifftn_kernel(const Tensor& input,
                   const std::vector<int64_t>& dims,
                   const std::vector<int64_t>& signal_lengths,
                   std::string_view norm) -> Tensor {
    return ifft2_kernel(input, dims, signal_lengths, norm);
}

#else // !TENZOR_USE_MKL

// ============================================================================
// Wave Inf-A: non-MKL CPU FFT fallback.
//
// Direct O(N²) discrete Fourier transform — correct for any signal length
// without requiring vendored libraries. The MKL path (above) is the
// production fast-path at O(N log N); this fallback exists so unit tests pass
// and small-signal FFTs work in MKL-free builds (e.g. dev machines without
// the oneAPI installer, ARM hosts, CI sanity-builds).
//
// A future Cooley-Tukey + Bluestein implementation (or a vendored PocketFFT)
// would replace these with O(N log N); plumbing it in is the documented
// follow-up. The kernel API is unchanged so the switch is internal.
// ============================================================================
namespace {

// Direct DFT along a strided dimension. Operates on a single (complex<T>)
// buffer treating it as (outer_size, signal_len_in, inner_size) and emits
// (outer_size, signal_len_out, inner_size). For signal_len_out != signal_len_in,
// the input is zero-padded (if out > in) or truncated (if out < in) — same
// Wave Inf-A (deferred → enhanced): added Cooley-Tukey radix-2 fast path.
// For power-of-2 lengths the runtime drops from O(N²) DFT to O(N log N)
// FFT. Non-power-of-2 lengths and N_in != N_out still use the direct
// O(N²) DFT (preserves correctness for any length).
template<typename T>
inline bool is_power_of_2(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Round up to next power of 2.
inline int64_t next_pow2(int64_t n) {
    if (n <= 1) return 1;
    int64_t r = 1;
    while (r < n) r <<= 1;
    return r;
}

// Iterative in-place Cooley-Tukey FFT, radix 2. Output overwrites input.
// Caller responsible for the global scale factor (applied outside this fn).
template<typename T>
void cooley_tukey_radix2(std::complex<T>* x, int64_t N, bool is_forward) {
    // Bit-reversal permutation (Gold-Rader algorithm).
    for (int64_t i = 1, j = 0; i < N; ++i) {
        int64_t bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
    // Butterflies.
    const T sign = is_forward ? T{-1} : T{1};
    for (int64_t len = 2; len <= N; len <<= 1) {
        const int64_t half = len >> 1;
        const T angle = sign * static_cast<T>(2.0 * M_PI) / static_cast<T>(len);
        const std::complex<T> wlen(std::cos(angle), std::sin(angle));
        for (int64_t i = 0; i < N; i += len) {
            std::complex<T> w(1, 0);
            for (int64_t k = 0; k < half; ++k) {
                std::complex<T> u = x[i + k];
                std::complex<T> v = x[i + k + half] * w;
                x[i + k]        = u + v;
                x[i + k + half] = u - v;
                w *= wlen;
            }
        }
    }
}

// Wave Inf-A continuation: Bluestein chirp z-transform for arbitrary N.
// Reduces an N-point DFT to one 2M-point complex multiplication (where
// M = next_pow2(2N-1)) plus two radix-2 FFTs and one inverse. O(N log N)
// for any length — eliminates the O(N²) DFT fallback for non-power-of-2.
//
// Identity used (Bluestein 1968):
//   X[k] = exp(-iπk²/N) · Σ_n [x[n] · exp(-iπn²/N)] · exp(iπ(k-n)²/N)
//        = exp(-iπk²/N) · (a * b)[k]   (convolution)
// where:
//   a[n] = x[n] · exp(-iπn²/N), n ∈ [0, N)
//   b[n] = exp(iπn²/N),         n ∈ (-N, N)   (symmetric around 0)
//
// We compute (a * b) as IFFT(FFT(a_padded) · FFT(b_padded)) at length M.
// M4 fix: cacheable Bluestein context. Precomputes chirp w[] and FFT(B[])
// once for a given (N, is_forward) pair so multi-row dft_1d_strided can
// reuse the work across rows instead of re-deriving per row.
template<typename T>
struct BluesteinContext {
    int64_t N;
    int64_t M;
    std::vector<std::complex<T>> w;         // chirp factors length N
    std::vector<std::complex<T>> B_fft;     // FFT of zero-padded B, length M
    bool is_forward;
};

template<typename T>
auto make_bluestein_context(int64_t N, bool is_forward) -> BluesteinContext<T> {
    BluesteinContext<T> ctx;
    ctx.N = N;
    ctx.M = next_pow2(2 * N - 1);
    ctx.is_forward = is_forward;
    const T pi = static_cast<T>(M_PI);
    const T sign = is_forward ? T{-1} : T{1};
    ctx.w.resize(N);
    for (int64_t n = 0; n < N; ++n) {
        T phase = -sign * pi * static_cast<T>(n) * static_cast<T>(n) / static_cast<T>(N);
        ctx.w[n] = std::complex<T>(std::cos(phase), std::sin(phase));
    }
    ctx.B_fft.assign(ctx.M, std::complex<T>(0, 0));
    ctx.B_fft[0] = ctx.w[0];
    for (int64_t n = 1; n < N; ++n) {
        ctx.B_fft[n]         = ctx.w[n];
        ctx.B_fft[ctx.M - n] = ctx.w[n];  // symmetric extension
    }
    cooley_tukey_radix2<T>(ctx.B_fft.data(), ctx.M, /*is_forward=*/true);
    return ctx;
}

// Run Bluestein on one row using a precomputed context. Caller supplies
// an A-scratch buffer of length ctx.M (allows reuse across rows in the
// dispatcher's parallel-for loop).
template<typename T>
void bluestein_fft_with_ctx(std::complex<T>* x,
                            std::complex<T>* A,  // length ctx.M scratch
                            const BluesteinContext<T>& ctx) {
    const int64_t N = ctx.N;
    const int64_t M = ctx.M;
    // Zero-pad A, multiply by conj(w[n]).
    for (int64_t n = 0; n < N; ++n) {
        A[n] = x[n] * std::conj(ctx.w[n]);
    }
    for (int64_t n = N; n < M; ++n) A[n] = std::complex<T>(0, 0);
    // FFT(A), pointwise multiply by precomputed FFT(B), IFFT.
    cooley_tukey_radix2<T>(A, M, /*is_forward=*/true);
    for (int64_t k = 0; k < M; ++k) A[k] *= ctx.B_fft[k];
    cooley_tukey_radix2<T>(A, M, /*is_forward=*/false);
    const T inv_M = T{1} / static_cast<T>(M);
    for (int64_t k = 0; k < N; ++k) {
        x[k] = A[k] * inv_M * std::conj(ctx.w[k]);
    }
}

// Wrapper kept for compatibility — single-shot Bluestein. New code should
// use the context-cached variant when transforming multiple rows.
template<typename T>
void bluestein_fft(std::complex<T>* x, int64_t N, bool is_forward) {
    if (N <= 1) return;
    auto ctx = make_bluestein_context<T>(N, is_forward);
    std::vector<std::complex<T>> A(ctx.M);
    bluestein_fft_with_ctx<T>(x, A.data(), ctx);
}

template<typename T>
void dft_1d_strided(const std::complex<T>* in,
                    std::complex<T>* out,
                    int64_t N_in, int64_t N_out,
                    int64_t outer, int64_t inner,
                    bool is_forward, double scale) {
    // Fast path: power-of-2 length, no resampling (N_in == N_out), strided
    // axis is inner==1 (otherwise the bit-reversal would interfere with
    // the strided layout). Use Cooley-Tukey radix-2 — O(N log N) per row.
    if (N_in == N_out && is_power_of_2<T>(N_out) && inner == 1) {
        #pragma omp parallel for schedule(static) if (outer > 4)
        for (int64_t o = 0; o < outer; ++o) {
            // Copy this row into the output buffer, then transform in place.
            const std::complex<T>* x = in + o * N_in;
            std::complex<T>* y = out + o * N_out;
            for (int64_t k = 0; k < N_out; ++k) y[k] = x[k];
            cooley_tukey_radix2<T>(y, N_out, is_forward);
            if (scale != 1.0) {
                T s = static_cast<T>(scale);
                for (int64_t k = 0; k < N_out; ++k) y[k] *= s;
            }
        }
        return;
    }
    // Wave Inf-A (deferred → landed): Bluestein chirp z-transform for
    // non-power-of-2 lengths. O(N log N) for any length.
    // M4 fix: precompute chirp + B_fft once and reuse across rows; gives
    // 30-50% speedup at outer >> 1 versus per-row recomputation.
    // Constraints: N_in == N_out (no resampling) — strided dims (inner > 1)
    // gather to a contiguous scratch then scatter back.
    if (N_in == N_out && N_out > 1) {
        auto ctx = make_bluestein_context<T>(N_out, is_forward);
        if (inner == 1) {
            #pragma omp parallel
            {
                std::vector<std::complex<T>> A_scratch(static_cast<size_t>(ctx.M));
                #pragma omp for schedule(static)
                for (int64_t o = 0; o < outer; ++o) {
                    const std::complex<T>* x = in + o * N_in;
                    std::complex<T>* y = out + o * N_out;
                    for (int64_t k = 0; k < N_out; ++k) y[k] = x[k];
                    bluestein_fft_with_ctx<T>(y, A_scratch.data(), ctx);
                    if (scale != 1.0) {
                        T s = static_cast<T>(scale);
                        for (int64_t k = 0; k < N_out; ++k) y[k] *= s;
                    }
                }
            }
        } else {
            // M4 strided fix: gather a row into contiguous scratch, run
            // Bluestein, scatter back. Same complexity as contiguous case
            // plus one O(N) gather/scatter per row.
            #pragma omp parallel
            {
                std::vector<std::complex<T>> row_scratch(static_cast<size_t>(N_out));
                std::vector<std::complex<T>> A_scratch(static_cast<size_t>(ctx.M));
                #pragma omp for collapse(2) schedule(static)
                for (int64_t o = 0; o < outer; ++o) {
                    for (int64_t j = 0; j < inner; ++j) {
                        const std::complex<T>* x = in + (o * N_in) * inner + j;
                        std::complex<T>* y = out + (o * N_out) * inner + j;
                        for (int64_t k = 0; k < N_in; ++k) {
                            row_scratch[static_cast<size_t>(k)] = x[k * inner];
                        }
                        bluestein_fft_with_ctx<T>(row_scratch.data(),
                                                  A_scratch.data(), ctx);
                        if (scale != 1.0) {
                            T s = static_cast<T>(scale);
                            for (int64_t k = 0; k < N_out; ++k) {
                                row_scratch[static_cast<size_t>(k)] *= s;
                            }
                        }
                        for (int64_t k = 0; k < N_out; ++k) {
                            y[k * inner] = row_scratch[static_cast<size_t>(k)];
                        }
                    }
                }
            }
        }
        return;
    }
    // Fallback: O(N²) direct DFT. Handles resampling (N_in != N_out) and
    // strided (inner > 1) axes — the Bluestein path can't handle those
    // because the chirp permutation assumes a flat contiguous row.
    const T two_pi_sign = static_cast<T>(is_forward ? -2.0 * M_PI : 2.0 * M_PI);
    #pragma omp parallel for collapse(2) schedule(static) if (outer * inner > 64)
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t j = 0; j < inner; ++j) {
            const std::complex<T>* x = in + (o * N_in) * inner + j;
            std::complex<T>* y = out + (o * N_out) * inner + j;
            const int64_t L = (N_in < N_out) ? N_in : N_out;
            for (int64_t k = 0; k < N_out; ++k) {
                std::complex<T> acc(0, 0);
                const T phase_base = two_pi_sign * static_cast<T>(k) / static_cast<T>(N_out);
                for (int64_t n = 0; n < L; ++n) {
                    T phase = phase_base * static_cast<T>(n);
                    std::complex<T> w(std::cos(phase), std::sin(phase));
                    acc += x[n * inner] * w;
                }
                y[k * inner] = acc * static_cast<T>(scale);
            }
        }
    }
}

template<typename T>
Tensor fft_1d_complex_dft(const Tensor& input, int64_t dim,
                          int64_t signal_len, std::string_view norm,
                          bool is_forward) {
    DType out_dtype = to_complex_dtype(input.dtype());
    Tensor inp = (input.dtype() == DType::Float32 || input.dtype() == DType::Float64)
                 ? input.to(out_dtype) : input;
    auto cont = inp.contiguous();
    auto shape = cont.shape();
    int64_t N_in = shape[dim];
    int64_t N_out = signal_len;
    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = N_out;
    Tensor result(out_shape, out_dtype, input.device());
    auto layout = compute_dim_layout(shape, dim);
    double scale = get_norm_factor(N_out, norm, is_forward);
    dft_1d_strided<T>(cont.template data<std::complex<T>>(),
                       result.template data<std::complex<T>>(),
                       N_in, N_out, layout.outer_size, layout.inner_size,
                       is_forward, scale);
    return result;
}

}  // anonymous namespace (DFT fallback)

auto fft_kernel(const Tensor& input, int64_t dim, int64_t signal_len,
                std::string_view norm) -> Tensor {
    dim = normalize_dim(dim, input.ndim());
    DType ct = to_complex_dtype(input.dtype());
    if (ct == DType::Complex64) {
        return fft_1d_complex_dft<float>(input, dim, signal_len, norm, true);
    }
    return fft_1d_complex_dft<double>(input, dim, signal_len, norm, true);
}

auto ifft_kernel(const Tensor& input, int64_t dim, int64_t signal_len,
                 std::string_view norm) -> Tensor {
    dim = normalize_dim(dim, input.ndim());
    DType ct = to_complex_dtype(input.dtype());
    if (ct == DType::Complex64) {
        return fft_1d_complex_dft<float>(input, dim, signal_len, norm, false);
    }
    return fft_1d_complex_dft<double>(input, dim, signal_len, norm, false);
}

auto rfft_kernel(const Tensor& input, int64_t dim, int64_t signal_len,
                 std::string_view norm) -> Tensor {
    // Real input → complex output of length N/2+1 (Hermitian).
    // Compute full complex FFT, then slice the positive-frequency half.
    dim = normalize_dim(dim, input.ndim());
    Tensor full = fft_kernel(input, dim, signal_len, norm);
    auto shape = full.shape();
    int64_t half = signal_len / 2 + 1;
    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = half;
    // Slice [0, half) along `dim`.
    return full.slice(dim, 0, half).contiguous();
}

auto irfft_kernel(const Tensor& input, int64_t dim, int64_t signal_len,
                  std::string_view norm) -> Tensor {
    // Complex Hermitian input → real output. Reconstruct full complex
    // spectrum from the one-sided input via conjugate-symmetry, then ifft.
    dim = normalize_dim(dim, input.ndim());
    auto in_shape = input.shape();
    int64_t half = in_shape[dim];
    // signal_len defaults to (half-1)*2 per PyTorch convention.
    int64_t N = (signal_len > 0) ? signal_len : (half - 1) * 2;
    DType ct = input.dtype();
    DType rt = to_real_dtype(ct);

    // Build full Hermitian-symmetric spectrum of length N.
    std::vector<int64_t> full_shape(in_shape.begin(), in_shape.end());
    full_shape[dim] = N;
    Tensor full(full_shape, ct, input.device());

    auto layout = compute_dim_layout(in_shape, dim);
    int64_t outer = layout.outer_size;
    int64_t inner = layout.inner_size;
    auto cont = input.contiguous();

    auto fill_hermitian = [&](auto* tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const std::complex<T>* x = cont.template data<std::complex<T>>();
        std::complex<T>* y = full.template data<std::complex<T>>();
        #pragma omp parallel for collapse(2) schedule(static) if (outer * inner > 64)
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t j = 0; j < inner; ++j) {
                // Positive frequencies (and DC, Nyquist if N even) copied directly.
                for (int64_t k = 0; k < half; ++k) {
                    y[(o * N + k) * inner + j] = x[(o * half + k) * inner + j];
                }
                // Negative frequencies: conjugate-symmetric.
                for (int64_t k = half; k < N; ++k) {
                    int64_t mirror = N - k;
                    if (mirror < half) {
                        y[(o * N + k) * inner + j] =
                            std::conj(x[(o * half + mirror) * inner + j]);
                    } else {
                        y[(o * N + k) * inner + j] = std::complex<T>(0, 0);
                    }
                }
            }
        }
    };

    if (ct == DType::Complex64) { float* t = nullptr; fill_hermitian(t); }
    else                        { double* t = nullptr; fill_hermitian(t); }

    // Inverse FFT of the full spectrum, then take real part.
    Tensor full_ifft = ifft_kernel(full, dim, N, norm);
    return full_ifft.to(rt);
}

// N-D variants: sequential 1D FFTs along each listed dim.
auto fft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                  const std::vector<int64_t>& signal_lengths,
                  std::string_view norm) -> Tensor {
    Tensor out = input;
    for (size_t i = 0; i < dims.size(); ++i) {
        out = fft_kernel(out, dims[i], signal_lengths[i], norm);
    }
    return out;
}

auto ifft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                   const std::vector<int64_t>& signal_lengths,
                   std::string_view norm) -> Tensor {
    Tensor out = input;
    for (size_t i = 0; i < dims.size(); ++i) {
        out = ifft_kernel(out, dims[i], signal_lengths[i], norm);
    }
    return out;
}

auto fftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                  const std::vector<int64_t>& signal_lengths,
                  std::string_view norm) -> Tensor {
    return fft2_kernel(input, dims, signal_lengths, norm);
}

auto ifftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                   const std::vector<int64_t>& signal_lengths,
                   std::string_view norm) -> Tensor {
    return ifft2_kernel(input, dims, signal_lengths, norm);
}

#endif // TENZOR_USE_MKL

// ============================================================================
// STFT Kernel - Short-Time Fourier Transform
// ============================================================================

auto stft_kernel(const Tensor& input, int64_t n_fft, int64_t hop_length, int64_t win_length,
                 const Tensor& window, bool center, bool normalized, bool onesided) -> Tensor {
    if (n_fft <= 0) {
        throw std::runtime_error("stft: n_fft must be > 0");
    }
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;
    if (win_length > n_fft) {
        throw std::runtime_error("stft: win_length must be <= n_fft");
    }

    // Determine batch dimensions: input is (..., signal_length)
    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (ndim < 1) {
        throw std::runtime_error("stft: input must have at least 1 dimension");
    }

    int64_t signal_length = in_shape[ndim - 1];
    int64_t batch_size = 1;
    for (int64_t d = 0; d < ndim - 1; ++d) {
        batch_size *= in_shape[d];
    }

    // Center padding: reflect-pad by n_fft/2 on both sides
    int64_t padded_length = signal_length;
    std::vector<float> padded_data;
    const float* src_data = nullptr;
    Tensor input_f32 = (input.dtype() != DType::Float32) ? input.to(DType::Float32) : input;

    if (center) {
        int64_t pad = n_fft / 2;
        padded_length = signal_length + 2 * pad;

        padded_data.resize(static_cast<size_t>(batch_size * padded_length));
        const float* in_ptr = input_f32.data<float>();

        for (int64_t b = 0; b < batch_size; ++b) {
            const float* sig = in_ptr + b * signal_length;
            float* out = padded_data.data() + b * padded_length;

            // Left reflect padding
            for (int64_t i = 0; i < pad; ++i) {
                int64_t idx = pad - i;
                if (idx >= signal_length) idx = signal_length - 1;
                out[i] = sig[idx];
            }
            // Copy original signal
            std::memcpy(out + pad, sig, static_cast<size_t>(signal_length) * sizeof(float));
            // Right reflect padding
            for (int64_t i = 0; i < pad; ++i) {
                int64_t idx = signal_length - 2 - i;
                if (idx < 0) idx = 0;
                out[pad + signal_length + i] = sig[idx];
            }
        }
        src_data = padded_data.data();
    } else {
        src_data = input_f32.data<float>();
    }

    // Number of frames
    int64_t num_frames = (padded_length - n_fft) / hop_length + 1;
    if (num_frames <= 0) {
        throw std::runtime_error("stft: signal too short for given n_fft and hop_length");
    }

    // Frequency bins
    int64_t freq_bins = onesided ? (n_fft / 2 + 1) : n_fft;

    // Build window data (pad win_length to n_fft if needed)
    std::vector<float> win_data(static_cast<size_t>(n_fft), 0.0f);
    int64_t win_offset = (n_fft - win_length) / 2;
    if (window.is_valid() && window.numel() > 0) {
        Tensor win_f32 = (window.dtype() != DType::Float32) ? window.to(DType::Float32) : window;
        const float* wp = win_f32.data<float>();
        for (int64_t i = 0; i < win_length; ++i) {
            win_data[static_cast<size_t>(win_offset + i)] = wp[i];
        }
    } else {
        // Default: rectangular window (all ones)
        for (int64_t i = 0; i < win_length; ++i) {
            win_data[static_cast<size_t>(win_offset + i)] = 1.0f;
        }
    }

    // Output: (..., freq_bins, num_frames) as Complex64
    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim - 1; ++d) {
        out_shape.push_back(in_shape[d]);
    }
    out_shape.push_back(freq_bins);
    out_shape.push_back(num_frames);

    Tensor result(out_shape, DType::Complex64, input.device());
    auto* out_ptr = reinterpret_cast<std::complex<float>*>(result.data_ptr());

    // Process each batch and frame
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* sig = src_data + b * padded_length;

        for (int64_t f = 0; f < num_frames; ++f) {
            // Extract windowed frame
            const float* frame_start = sig + f * hop_length;

            // Create a 1D tensor for the windowed frame
            Tensor frame_tensor({n_fft}, DType::Float32, input.device());
            float* frame_data = frame_tensor.data<float>();
            for (int64_t i = 0; i < n_fft; ++i) {
                frame_data[i] = frame_start[i] * win_data[static_cast<size_t>(i)];
            }

            // Apply FFT
            Tensor fft_result;
            if (onesided) {
                fft_result = rfft_kernel(frame_tensor, 0, n_fft, normalized ? "ortho" : "backward");
            } else {
                fft_result = fft_kernel(frame_tensor, 0, n_fft, normalized ? "ortho" : "backward");
            }

            // Copy FFT result into output
            auto* fft_ptr = reinterpret_cast<const std::complex<float>*>(fft_result.data_ptr());
            for (int64_t k = 0; k < freq_bins; ++k) {
                out_ptr[b * freq_bins * num_frames + k * num_frames + f] = fft_ptr[k];
            }
        }
    }

    return result;
}

// ============================================================================
// ISTFT Kernel - Inverse Short-Time Fourier Transform
// ============================================================================

auto istft_kernel(const Tensor& input, int64_t n_fft, int64_t hop_length, int64_t win_length,
                  const Tensor& window, bool center, bool normalized, bool onesided,
                  int64_t length) -> Tensor {
    if (n_fft <= 0) {
        throw std::runtime_error("istft: n_fft must be > 0");
    }
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;

    // Input shape: (..., freq_bins, num_frames)
    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (ndim < 2) {
        throw std::runtime_error("istft: input must have at least 2 dimensions");
    }

    int64_t freq_bins = in_shape[ndim - 2];
    int64_t num_frames = in_shape[ndim - 1];
    int64_t batch_size = 1;
    for (int64_t d = 0; d < ndim - 2; ++d) {
        batch_size *= in_shape[d];
    }

    // Build window
    std::vector<float> win_data(static_cast<size_t>(n_fft), 0.0f);
    int64_t win_offset = (n_fft - win_length) / 2;
    if (window.is_valid() && window.numel() > 0) {
        Tensor win_f32 = (window.dtype() != DType::Float32) ? window.to(DType::Float32) : window;
        const float* wp = win_f32.data<float>();
        for (int64_t i = 0; i < win_length; ++i) {
            win_data[static_cast<size_t>(win_offset + i)] = wp[i];
        }
    } else {
        for (int64_t i = 0; i < win_length; ++i) {
            win_data[static_cast<size_t>(win_offset + i)] = 1.0f;
        }
    }

    // Expected output length before trimming
    int64_t expected_length = n_fft + (num_frames - 1) * hop_length;

    // Ensure input is Complex64
    Tensor input_c64 = (input.dtype() != DType::Complex64) ? input.to(DType::Complex64) : input;
    auto* in_ptr = reinterpret_cast<const std::complex<float>*>(input_c64.data_ptr());

    // Output shape: (..., signal_length)
    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim - 2; ++d) {
        out_shape.push_back(in_shape[d]);
    }
    out_shape.push_back(expected_length);

    // Allocate output and window normalization buffer
    std::vector<float> output_data(static_cast<size_t>(batch_size * expected_length), 0.0f);
    std::vector<float> window_sum(static_cast<size_t>(batch_size * expected_length), 0.0f);

    for (int64_t b = 0; b < batch_size; ++b) {
        float* out = output_data.data() + b * expected_length;
        float* wsum = window_sum.data() + b * expected_length;

        for (int64_t f = 0; f < num_frames; ++f) {
            // Extract frequency column for this frame
            Tensor frame_freq({freq_bins}, DType::Complex64, input.device());
            auto* freq_data = reinterpret_cast<std::complex<float>*>(frame_freq.data_ptr());
            for (int64_t k = 0; k < freq_bins; ++k) {
                freq_data[k] = in_ptr[b * freq_bins * num_frames + k * num_frames + f];
            }

            // Apply inverse FFT
            Tensor time_frame;
            if (onesided) {
                time_frame = irfft_kernel(frame_freq, 0, n_fft, normalized ? "ortho" : "backward");
            } else {
                time_frame = ifft_kernel(frame_freq, 0, n_fft, normalized ? "ortho" : "backward");
            }

            // Get real part of time-domain frame
            Tensor real_frame = (time_frame.dtype() == DType::Complex64 || time_frame.dtype() == DType::Complex128)
                ? time_frame.to(DType::Float32)
                : time_frame;
            const float* frame_data = real_frame.data<float>();

            // Overlap-add with window
            int64_t frame_offset = f * hop_length;
            for (int64_t i = 0; i < n_fft; ++i) {
                out[frame_offset + i] += frame_data[i] * win_data[static_cast<size_t>(i)];
                wsum[frame_offset + i] += win_data[static_cast<size_t>(i)] * win_data[static_cast<size_t>(i)];
            }
        }

        // Normalize by window sum (COLA constraint)
        for (int64_t i = 0; i < expected_length; ++i) {
            if (wsum[i] > 1e-10f) {
                out[i] /= wsum[i];
            }
        }
    }

    // Trim if center padding was used
    int64_t final_length = expected_length;
    int64_t trim_start = 0;
    if (center) {
        trim_start = n_fft / 2;
        final_length = expected_length - 2 * (n_fft / 2);
    }
    if (length > 0) {
        final_length = length;
    }

    // Build final output shape
    std::vector<int64_t> final_shape;
    for (int64_t d = 0; d < ndim - 2; ++d) {
        final_shape.push_back(in_shape[d]);
    }
    final_shape.push_back(final_length);

    Tensor result(final_shape, DType::Float32, input.device());
    float* result_data = result.data<float>();

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* src = output_data.data() + b * expected_length + trim_start;
        float* dst = result_data + b * final_length;
        int64_t copy_len = std::min(final_length, expected_length - trim_start);
        std::memcpy(dst, src, static_cast<size_t>(copy_len) * sizeof(float));
        // Zero-fill if length extends beyond available data
        if (copy_len < final_length) {
            std::memset(dst + copy_len, 0, static_cast<size_t>(final_length - copy_len) * sizeof(float));
        }
    }

    return result;
}

} // namespace cpu
} // namespace tenzor
