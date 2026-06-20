/**
 * @file fft.cpp
 * @brief OneAPI/SYCL FFT kernels via oneMKL DFT
 *
 * Implements FFT, IFFT, RFFT, IRFFT, FFT2, IFFT2, FFTN, IFFTN
 * using oneMKL DFT (Discrete Fourier Transform) APIs when available.
 * Falls back to Cooley-Tukey (power-of-2) / Bluestein (general) when oneMKL is not present.
 *
 * Guarded by TENZOR_HAS_ONEMKL.
 */

#include "tenzor/core/tensor.hpp"
#include "oneapi_kernel_utils.hpp"
#include "tenzor/ops/fft.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cmath>

#ifdef TENZOR_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/dft.hpp>
#endif

namespace tenzor {
namespace oneapi {

template <typename T>
struct SyclDevicePtr {
    T* ptr = nullptr;
    sycl::queue& q;
    SyclDevicePtr(size_t count, sycl::queue& queue) : q(queue) {
        ptr = sycl::malloc_device<T>(count, queue);
    }
    ~SyclDevicePtr() { if (ptr) sycl::free(ptr, q); }
    operator T*() { return ptr; }
    T* get() { return ptr; }
    SyclDevicePtr(const SyclDevicePtr&) = delete;
    SyclDevicePtr& operator=(const SyclDevicePtr&) = delete;
};


auto clone_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;


// Device-side upcast: F16/BF16 -> F32 (zero host transfers)
inline void device_upcast_to_f32(const void* src_ptr, float* dst_ptr, int64_t numel,
                                  DType src_dtype, sycl::queue& queue) {
    if (src_dtype == DType::Float16) {
        const sycl::half* src = static_cast<const sycl::half*>(src_ptr);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            dst_ptr[i] = static_cast<float>(src[i]);
        }).wait();
    } else {
        // BFloat16 stored as uint16_t
        const uint16_t* src = static_cast<const uint16_t*>(src_ptr);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
            float val;
            __builtin_memcpy(&val, &bits, sizeof(float));
            dst_ptr[i] = val;
        }).wait();
    }
}

// Device-side downcast: F32 -> F16/BF16 (zero host transfers)
inline void device_downcast_from_f32(const float* src_ptr, void* dst_ptr, int64_t numel,
                                      DType dst_dtype, sycl::queue& queue) {
    if (dst_dtype == DType::Float16) {
        sycl::half* dst = static_cast<sycl::half*>(dst_ptr);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            dst[i] = sycl::half(src_ptr[i]);
        }).wait();
    } else {
        uint16_t* dst = static_cast<uint16_t*>(dst_ptr);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            uint32_t bits;
            __builtin_memcpy(&bits, &src_ptr[i], sizeof(uint32_t));
            // NaN: preserve a quiet-NaN bit pattern rather than letting the
            // rounding add below silently flip it into an infinity.
            if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
                dst[i] = static_cast<uint16_t>((bits >> 16) | 0x0040u);
            } else {
                // Round-to-nearest-even: add 0x7FFF + (lsb of the kept mantissa)
                // before truncating, matching the F16 sycl::half() path's rounding
                // so the BF16 downcast is not biased toward zero.
                uint32_t lsb = (bits >> 16) & 1u;
                uint32_t rounding_bias = 0x7FFFu + lsb;
                bits += rounding_bias;
                dst[i] = static_cast<uint16_t>(bits >> 16);
            }
        }).wait();
    }
}

#ifdef TENZOR_HAS_ONEMKL

namespace {

// ============================================================================
// Normalization helpers
// ============================================================================

/// Compute normalization scale factor for a single FFT dimension.
/// oneMKL DFT does NOT normalize by default (like cuFFT).
/// @param n      Transform size
/// @param norm   "backward" (default), "forward", or "ortho"
/// @param is_fwd True for forward transform, false for inverse
/// @return Scale factor to multiply output by
double get_norm_factor(int64_t n, const std::string& norm, bool is_fwd) {
    if (norm == "ortho") {
        return 1.0 / std::sqrt(static_cast<double>(n));
    } else if ((norm == "forward" && is_fwd) || (norm == "backward" && !is_fwd)) {
        return 1.0 / static_cast<double>(n);
    }
    return 1.0;
}

/// Apply in-place scaling to interleaved complex data (Float32 pairs).
/// Note: does NOT wait — caller must ensure completion before freeing buffers.
void apply_scale_f32(sycl::queue& queue, float* data, int64_t total_floats, float scale) {
    if (scale == 1.0f) return;
    queue.parallel_for(sycl::range<1>(total_floats), [=](sycl::id<1> idx) {
        data[idx] *= scale;
    });
}

/// Apply in-place scaling to interleaved complex data (Float64 pairs).
/// Note: does NOT wait — caller must ensure completion before freeing buffers.
void apply_scale_f64(sycl::queue& queue, double* data, int64_t total_doubles, double scale) {
    if (scale == 1.0) return;
    queue.parallel_for(sycl::range<1>(total_doubles), [=](sycl::id<1> idx) {
        data[idx] *= scale;
    });
}

/// Apply in-place scaling to real data (Float32).
/// Note: does NOT wait — caller must ensure completion before freeing buffers.
void apply_scale_real_f32(sycl::queue& queue, float* data, int64_t numel, float scale) {
    if (scale == 1.0f) return;
    queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
        data[idx] *= scale;
    });
}

/// Apply in-place scaling to real data (Float64).
/// Note: does NOT wait — caller must ensure completion before freeing buffers.
void apply_scale_real_f64(sycl::queue& queue, double* data, int64_t numel, double scale) {
    if (scale == 1.0) return;
    queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
        data[idx] *= scale;
    });
}

} // anonymous namespace

// ============================================================================
// FFT - 1D Complex-to-Complex Forward FFT
// ============================================================================
//
// Input: real tensor of shape (..., signal_len, ...) with dtype Float32/Float64
//        (treated as real-valued; output gets a trailing dim=2 for complex)
// Output: tensor of shape (..., signal_len, ..., 2)
//
// The current convention in this backend: complex is stored as interleaved
// float pairs in a trailing dimension of size 2.
//
// oneMKL DFT with domain::COMPLEX operates on interleaved complex data.
// Since the input is real, we create a complex buffer (zero imaginary),
// run the in-place C2C transform, then write to output.
// ============================================================================
auto fft_kernel(const Tensor& input, int64_t dim, int64_t n,
                const std::string& norm, sycl::queue& queue) -> Tensor {
    namespace dft = ::oneapi::mkl::dft;

    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    int64_t signal_len = shape[dim];

    // On-device pad/truncate when the requested length differs from the
    // input length along the FFT dimension. We build a new tensor with
    // size n along dim, copy the overlapping elements, and zero-fill any
    // padding — all on-device with no CPU round-trip.
    Tensor padded_input = input;  // default: no copy needed
    if (n != signal_len) {
        std::vector<int64_t> new_shape(shape.begin(), shape.end());
        new_shape[dim] = n;

        int64_t outer = 1, inner = 1;
        for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
        for (int64_t i = dim + 1; i < ndim; ++i) inner *= shape[i];

        int64_t copy_len = std::min(signal_len, n);

        padded_input = Tensor(new_shape, input.dtype(), input.device());
        // Zero the entire buffer so padding region is 0
        int64_t total_bytes = padded_input.numel() * padded_input.dtype_size();
        queue.memset(const_cast<void*>(padded_input.data_ptr()), 0,
                     static_cast<size_t>(total_bytes)).wait();

        // Copy the overlapping region element-by-element.
        // Each element is 2 floats (Complex64) or 2 doubles (Complex128).
        if (input.dtype() == DType::Complex64) {
            const float* src = reinterpret_cast<const float*>(input.data_ptr());
            float* dst = reinterpret_cast<float*>(const_cast<void*>(padded_input.data_ptr()));
            int64_t total_copies = outer * copy_len * inner;
            int64_t old_dim = signal_len;
            int64_t new_dim = n;
            queue.parallel_for(sycl::range<1>(total_copies), [=](sycl::id<1> idx_) {
                int64_t flat = idx_[0];
                int64_t o = flat / (copy_len * inner);
                int64_t rem = flat % (copy_len * inner);
                int64_t d = rem / inner;
                int64_t i = rem % inner;
                int64_t src_idx = (o * old_dim * inner + d * inner + i) * 2;
                int64_t dst_idx = (o * new_dim * inner + d * inner + i) * 2;
                dst[dst_idx]     = src[src_idx];
                dst[dst_idx + 1] = src[src_idx + 1];
            }).wait();
        } else if (input.dtype() == DType::Complex128) {
            const double* src = reinterpret_cast<const double*>(input.data_ptr());
            double* dst = reinterpret_cast<double*>(const_cast<void*>(padded_input.data_ptr()));
            int64_t total_copies = outer * copy_len * inner;
            int64_t old_dim = signal_len;
            int64_t new_dim = n;
            queue.parallel_for(sycl::range<1>(total_copies), [=](sycl::id<1> idx_) {
                int64_t flat = idx_[0];
                int64_t o = flat / (copy_len * inner);
                int64_t rem = flat % (copy_len * inner);
                int64_t d = rem / inner;
                int64_t i = rem % inner;
                int64_t src_idx = (o * old_dim * inner + d * inner + i) * 2;
                int64_t dst_idx = (o * new_dim * inner + d * inner + i) * 2;
                dst[dst_idx]     = src[src_idx];
                dst[dst_idx + 1] = src[src_idx + 1];
            }).wait();
        }

        // Update shape and signal_len for the rest of the function
        shape = new_shape;
        signal_len = n;
    }

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) batch_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    // Output shape: same shape as input, dtype Complex64/128. No trailing
    // 2 dim — the interleaved storage is the physical layout of Complex64
    // and we stay consistent with the CPU and CUDA backends.
    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

    // The public `tenzor::fft::fft` op promotes Float32/Float64 inputs to
    // Complex64/Complex128 before dispatch, so we only ever see a complex
    // dtype here. The old "Float32 + trailing 2" convention has been
    // retired to match the CPU kernel's output layout.
    if (padded_input.dtype() == DType::Complex64) {
        // Build interleaved complex buffer: (batch_size * signal_len * inner_size) complex values
        // stored as 2*N floats. We need contiguous layout along the FFT dimension for oneMKL.
        //
        // For the general strided case (dim != last), we process per-batch-per-inner
        // with NUMBER_OF_TRANSFORMS = 1 and manual offset computation.
        // For dim == last (most common), we can batch efficiently.

        int64_t total_transforms = batch_size * inner_size;
        int64_t complex_buf_floats = total_transforms * signal_len * 2;

        // Allocate device buffer for interleaved complex data
        SyclDevicePtr<float> complex_buf_owner(complex_buf_floats, queue);
        float* complex_buf = complex_buf_owner.get();

        // Read the interleaved (re, im) float pair storage of Complex64.
        const float* in_ptr = reinterpret_cast<const float*>(padded_input.data_ptr());

        if (dim == ndim - 1 && inner_size == 1) {
            // Fast path: FFT along last dim, contiguous complex per batch.
            queue.memcpy(complex_buf, in_ptr, complex_buf_floats * sizeof(float));

            dft::descriptor<dft::precision::SINGLE, dft::domain::COMPLEX> desc(signal_len);
            desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, batch_size);
            std::vector<std::int64_t> fwd_strides = {0, 1};
            desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
            desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);
            desc.set_value(dft::config_param::PLACEMENT, dft::config_value::INPLACE);
            desc.commit(queue);

            dft::compute_forward(desc, reinterpret_cast<std::complex<float>*>(complex_buf));
        } else {
            // General path: gather strided complex pairs into contiguous
            // complex buffer. Each complex element is 2 consecutive floats.
            queue.parallel_for(sycl::range<1>(total_transforms * signal_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / signal_len;
                    int64_t j = idx % signal_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t in_idx = (b * signal_len * inner_size + j * inner_size + inner) * 2;
                    complex_buf[2 * idx]     = in_ptr[in_idx];
                    complex_buf[2 * idx + 1] = in_ptr[in_idx + 1];
                });

            dft::descriptor<dft::precision::SINGLE, dft::domain::COMPLEX> desc(signal_len);
            desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, total_transforms);
            std::vector<std::int64_t> fwd_strides = {0, 1};
            desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
            desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);
            // For an in-place C2C transform the OUTPUT (backward domain) shares
            // the same gathered, contiguous-per-transform layout as the input.
            // BWD_DISTANCE defaults to 1, which would write the `total_transforms`
            // results with the wrong inter-transform spacing and clobber each
            // other (correct only when total_transforms == 1). Mirror the FWD
            // layout so every non-last-dim batched transform lands correctly.
            std::vector<std::int64_t> bwd_strides = {0, 1};
            desc.set_value(dft::config_param::BWD_STRIDES, bwd_strides);
            desc.set_value(dft::config_param::BWD_DISTANCE, signal_len);
            desc.set_value(dft::config_param::PLACEMENT, dft::config_value::INPLACE);
            desc.commit(queue);

            dft::compute_forward(desc, reinterpret_cast<std::complex<float>*>(complex_buf));
        }

        // Apply normalization
        double norm_factor = get_norm_factor(signal_len, norm, true);
        if (norm_factor != 1.0) {
            apply_scale_f32(queue, complex_buf, complex_buf_floats,
                            static_cast<float>(norm_factor));
        }

        // Scatter complex results back to output tensor. Output is Complex64
        // with the same shape as padded_input — no trailing 2 dim. The physical
        // storage is still interleaved (re, im) pairs.
        Tensor output(out_shape, DType::Complex64, padded_input.device());
        float* out_ptr = reinterpret_cast<float*>(const_cast<void*>(output.data_ptr()));

        if (dim == ndim - 1 && inner_size == 1) {
            queue.memcpy(out_ptr, complex_buf, complex_buf_floats * sizeof(float)).wait();
        } else {
            queue.parallel_for(sycl::range<1>(total_transforms * signal_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / signal_len;
                    int64_t k = idx % signal_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t out_idx = (b * signal_len * inner_size + k * inner_size + inner) * 2;
                    out_ptr[out_idx]     = complex_buf[2 * idx];
                    out_ptr[out_idx + 1] = complex_buf[2 * idx + 1];
                }).wait();
        }

        return output;

    } else if (padded_input.dtype() == DType::Complex128) {
        int64_t total_transforms = batch_size * inner_size;
        int64_t complex_buf_doubles = total_transforms * signal_len * 2;

        SyclDevicePtr<double> complex_buf_owner(complex_buf_doubles, queue);
        double* complex_buf = complex_buf_owner.get();
        const double* in_ptr = reinterpret_cast<const double*>(padded_input.data_ptr());

        if (dim == ndim - 1 && inner_size == 1) {
            queue.memcpy(complex_buf, in_ptr, complex_buf_doubles * sizeof(double));

            dft::descriptor<dft::precision::DOUBLE, dft::domain::COMPLEX> desc(signal_len);
            desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, batch_size);
            std::vector<std::int64_t> fwd_strides = {0, 1};
            desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
            desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);
            desc.set_value(dft::config_param::PLACEMENT, dft::config_value::INPLACE);
            desc.commit(queue);

            dft::compute_forward(desc, reinterpret_cast<std::complex<double>*>(complex_buf));
        } else {
            queue.parallel_for(sycl::range<1>(total_transforms * signal_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / signal_len;
                    int64_t j = idx % signal_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t in_idx = (b * signal_len * inner_size + j * inner_size + inner) * 2;
                    complex_buf[2 * idx]     = in_ptr[in_idx];
                    complex_buf[2 * idx + 1] = in_ptr[in_idx + 1];
                });

            dft::descriptor<dft::precision::DOUBLE, dft::domain::COMPLEX> desc(signal_len);
            desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, total_transforms);
            std::vector<std::int64_t> fwd_strides = {0, 1};
            desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
            desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);
            // In-place C2C: the backward (output) domain shares the gathered,
            // contiguous-per-transform layout. Without BWD_DISTANCE the batched
            // outputs overlap (defaults to 1). Mirror the FWD layout.
            std::vector<std::int64_t> bwd_strides = {0, 1};
            desc.set_value(dft::config_param::BWD_STRIDES, bwd_strides);
            desc.set_value(dft::config_param::BWD_DISTANCE, signal_len);
            desc.set_value(dft::config_param::PLACEMENT, dft::config_value::INPLACE);
            desc.commit(queue);

            dft::compute_forward(desc, reinterpret_cast<std::complex<double>*>(complex_buf));
        }

        double norm_factor = get_norm_factor(signal_len, norm, true);
        if (norm_factor != 1.0) {
            apply_scale_f64(queue, complex_buf, complex_buf_doubles, norm_factor);
        }

        Tensor output(out_shape, DType::Complex128, padded_input.device());
        double* out_ptr = reinterpret_cast<double*>(const_cast<void*>(output.data_ptr()));

        if (dim == ndim - 1 && inner_size == 1) {
            queue.memcpy(out_ptr, complex_buf, complex_buf_doubles * sizeof(double)).wait();
        } else {
            queue.parallel_for(sycl::range<1>(total_transforms * signal_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / signal_len;
                    int64_t k = idx % signal_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t out_idx = (b * signal_len * inner_size + k * inner_size + inner) * 2;
                    out_ptr[out_idx]     = complex_buf[2 * idx];
                    out_ptr[out_idx + 1] = complex_buf[2 * idx + 1];
                }).wait();
        }

        return output;

    } else {
        throw std::runtime_error(
            "fft_kernel: unsupported dtype (expected Complex64 or Complex128)");
    }
}

// ============================================================================
// IFFT - 1D Complex-to-Complex Inverse FFT
// ============================================================================
//
// Input: complex tensor stored as (..., signal_len, ..., 2) with Float32/Float64
// Output: same shape (complex output)
//
// oneMKL compute_backward performs unnormalized inverse DFT.
// ============================================================================
auto ifft_kernel(const Tensor& input, int64_t dim, int64_t n,
                 const std::string& norm, sycl::queue& queue) -> Tensor {
    namespace dft = ::oneapi::mkl::dft;

    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    // Expect Complex64/Complex128 input — matches the fft_kernel output
    // contract and the CPU backend.
    if (input.dtype() != DType::Complex64 && input.dtype() != DType::Complex128) {
        throw std::runtime_error(
            "ifft_kernel: expected Complex64 or Complex128 input");
    }

    int64_t signal_len = shape[dim];
    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) batch_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    // Output has same shape as input (complex, same dtype).
    std::vector<int64_t> out_shape(shape.begin(), shape.end());

    if (input.dtype() == DType::Complex64) {
        int64_t total_transforms = batch_size * inner_size;
        int64_t complex_buf_floats = total_transforms * signal_len * 2;

        SyclDevicePtr<float> complex_buf_owner(complex_buf_floats, queue);
        float* complex_buf = complex_buf_owner.get();
        const float* in_ptr = reinterpret_cast<const float*>(input.data_ptr());

        // Gather interleaved complex data into contiguous per-transform layout
        if (dim == ndim - 1 && inner_size == 1) {
            queue.memcpy(complex_buf, in_ptr, complex_buf_floats * sizeof(float));
        } else {
            queue.parallel_for(sycl::range<1>(total_transforms * signal_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / signal_len;
                    int64_t j = idx % signal_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t in_idx = (b * signal_len * inner_size + j * inner_size + inner) * 2;
                    complex_buf[2 * idx]     = in_ptr[in_idx];
                    complex_buf[2 * idx + 1] = in_ptr[in_idx + 1];
                });
        }

        // Run inverse DFT
        dft::descriptor<dft::precision::SINGLE, dft::domain::COMPLEX> desc(signal_len);
        desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, total_transforms);
        std::vector<std::int64_t> bwd_strides = {0, 1};
        desc.set_value(dft::config_param::BWD_STRIDES, bwd_strides);
        desc.set_value(dft::config_param::BWD_DISTANCE, signal_len);
        // In-place C2C inverse: the forward (output) domain shares the gathered,
        // contiguous-per-transform layout. FWD_DISTANCE defaults to 1, which
        // would overlap the batched outputs for total_transforms > 1 (non-last
        // dim). Mirror the BWD layout so each transform writes to its own slot.
        std::vector<std::int64_t> fwd_strides = {0, 1};
        desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
        desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);
        desc.set_value(dft::config_param::PLACEMENT, dft::config_value::INPLACE);
        desc.commit(queue);

        dft::compute_backward(desc, reinterpret_cast<std::complex<float>*>(complex_buf));

        // Apply normalization
        double norm_factor = get_norm_factor(signal_len, norm, false);
        if (norm_factor != 1.0) {
            apply_scale_f32(queue, complex_buf, complex_buf_floats,
                            static_cast<float>(norm_factor));
        }

        Tensor output(out_shape, DType::Complex64, input.device());
        float* out_ptr = reinterpret_cast<float*>(const_cast<void*>(output.data_ptr()));

        if (dim == ndim - 1 && inner_size == 1) {
            queue.memcpy(out_ptr, complex_buf, complex_buf_floats * sizeof(float)).wait();
        } else {
            queue.parallel_for(sycl::range<1>(total_transforms * signal_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / signal_len;
                    int64_t k = idx % signal_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t out_idx = (b * signal_len * inner_size + k * inner_size + inner) * 2;
                    out_ptr[out_idx]     = complex_buf[2 * idx];
                    out_ptr[out_idx + 1] = complex_buf[2 * idx + 1];
                }).wait();
        }

        return output;

    } else {  // Complex128
        int64_t total_transforms = batch_size * inner_size;
        int64_t complex_buf_doubles = total_transforms * signal_len * 2;

        SyclDevicePtr<double> complex_buf_owner(complex_buf_doubles, queue);
        double* complex_buf = complex_buf_owner.get();
        const double* in_ptr = reinterpret_cast<const double*>(input.data_ptr());

        if (dim == ndim - 1 && inner_size == 1) {
            queue.memcpy(complex_buf, in_ptr, complex_buf_doubles * sizeof(double));
        } else {
            queue.parallel_for(sycl::range<1>(total_transforms * signal_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / signal_len;
                    int64_t j = idx % signal_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t in_idx = (b * signal_len * inner_size + j * inner_size + inner) * 2;
                    complex_buf[2 * idx]     = in_ptr[in_idx];
                    complex_buf[2 * idx + 1] = in_ptr[in_idx + 1];
                });
        }

        dft::descriptor<dft::precision::DOUBLE, dft::domain::COMPLEX> desc(signal_len);
        desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, total_transforms);
        std::vector<std::int64_t> bwd_strides = {0, 1};
        desc.set_value(dft::config_param::BWD_STRIDES, bwd_strides);
        desc.set_value(dft::config_param::BWD_DISTANCE, signal_len);
        // In-place C2C inverse: mirror the layout to the forward (output) domain
        // so the batched outputs don't overlap (FWD_DISTANCE defaults to 1).
        std::vector<std::int64_t> fwd_strides = {0, 1};
        desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
        desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);
        desc.set_value(dft::config_param::PLACEMENT, dft::config_value::INPLACE);
        desc.commit(queue);

        dft::compute_backward(desc, reinterpret_cast<std::complex<double>*>(complex_buf));

        double norm_factor = get_norm_factor(signal_len, norm, false);
        if (norm_factor != 1.0) {
            apply_scale_f64(queue, complex_buf, complex_buf_doubles, norm_factor);
        }

        Tensor output(out_shape, DType::Complex128, input.device());
        double* out_ptr = reinterpret_cast<double*>(const_cast<void*>(output.data_ptr()));

        if (dim == ndim - 1 && inner_size == 1) {
            queue.memcpy(out_ptr, complex_buf, complex_buf_doubles * sizeof(double)).wait();
        } else {
            queue.parallel_for(sycl::range<1>(total_transforms * signal_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / signal_len;
                    int64_t k = idx % signal_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t out_idx = (b * signal_len * inner_size + k * inner_size + inner) * 2;
                    out_ptr[out_idx]     = complex_buf[2 * idx];
                    out_ptr[out_idx + 1] = complex_buf[2 * idx + 1];
                }).wait();
        }

        return output;
    }
}

// ============================================================================
// RFFT - 1D Real-to-Complex Forward FFT
// ============================================================================
//
// Input: real tensor of shape (..., signal_len, ...)
// Output: complex tensor of shape (..., signal_len/2+1, ...)
//         dtype Complex64 (or Complex128 for Float64 input)
//
// oneMKL DFT with domain::REAL performs R2C. The output is N/2+1 complex values.
// oneMKL stores R2C output in CCS (Complex-Conjugate-Symmetric) packed format
// by default. We use DFTI_NOT_INPLACE with separate real input and complex output
// buffers to get standard interleaved complex output.
//
// The tensor's physical storage is the same interleaved (real, imag) float
// pair layout used by std::complex<float>, so there is no conversion step
// beyond labelling the dtype correctly — we just avoid introducing a
// trailing length-2 dim that breaks backend parity with the CPU kernel.
// ============================================================================
auto rfft_kernel(const Tensor& input_raw, int64_t dim, int64_t n,
                 const std::string& norm, sycl::queue& queue) -> Tensor {
    namespace dft = ::oneapi::mkl::dft;

    auto shape_span = input_raw.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    int64_t signal_len = shape[dim];

    // On-device pad/truncate the REAL input to the requested transform length n
    // along `dim`, mirroring fft_kernel. PyTorch's rfft(x, n=K) zero-pads or
    // truncates the signal to K before transforming; without this the kernel
    // would transform the wrong length and use a wrong normalization factor.
    // Float16/BFloat16 are handled by the recursion below (which re-enters the
    // Float32 path with the same n), so we only pad the real F32/F64 buffers.
    Tensor input = input_raw;  // default: no copy needed
    if (n != signal_len &&
        (input_raw.dtype() == DType::Float32 || input_raw.dtype() == DType::Float64)) {
        std::vector<int64_t> new_shape(shape.begin(), shape.end());
        new_shape[dim] = n;

        int64_t outer = 1, inner = 1;
        for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
        for (int64_t i = dim + 1; i < ndim; ++i) inner *= shape[i];
        int64_t copy_len = std::min(signal_len, n);

        input = Tensor(new_shape, input_raw.dtype(), input_raw.device());
        int64_t total_bytes = input.numel() * input.dtype_size();
        queue.memset(const_cast<void*>(input.data_ptr()), 0,
                     static_cast<size_t>(total_bytes)).wait();

        int64_t old_dim = signal_len;
        int64_t new_dim = n;
        int64_t total_copies = outer * copy_len * inner;
        if (input_raw.dtype() == DType::Float32) {
            const float* src = get_data_ptr<const float>(input_raw);
            float* dst = reinterpret_cast<float*>(const_cast<void*>(input.data_ptr()));
            queue.parallel_for(sycl::range<1>(total_copies), [=](sycl::id<1> idx_) {
                int64_t flat = idx_[0];
                int64_t o = flat / (copy_len * inner);
                int64_t rem = flat % (copy_len * inner);
                int64_t d = rem / inner;
                int64_t i = rem % inner;
                dst[o * new_dim * inner + d * inner + i] = src[o * old_dim * inner + d * inner + i];
            }).wait();
        } else {  // Float64
            const double* src = get_data_ptr<const double>(input_raw);
            double* dst = reinterpret_cast<double*>(const_cast<void*>(input.data_ptr()));
            queue.parallel_for(sycl::range<1>(total_copies), [=](sycl::id<1> idx_) {
                int64_t flat = idx_[0];
                int64_t o = flat / (copy_len * inner);
                int64_t rem = flat % (copy_len * inner);
                int64_t d = rem / inner;
                int64_t i = rem % inner;
                dst[o * new_dim * inner + d * inner + i] = src[o * old_dim * inner + d * inner + i];
            }).wait();
        }

        shape = new_shape;
        signal_len = n;
    }

    int64_t out_len = signal_len / 2 + 1;

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) batch_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    // Output shape: replace dim with out_len. The dtype is Complex64/128,
    // so there is no trailing 2 dimension — each element is one complex
    // value of 8 or 16 bytes.
    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = out_len;
    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

    if (input.dtype() == DType::Float32) {
        int64_t total_transforms = batch_size * inner_size;

        // Allocate contiguous real input buffer and complex output buffer
        SyclDevicePtr<float> real_buf_owner(total_transforms * signal_len, queue);
        float* real_buf = real_buf_owner.get();
        SyclDevicePtr<float> complex_buf_owner(total_transforms * out_len * 2, queue);
        float* complex_buf = complex_buf_owner.get();

        const float* in_ptr = get_data_ptr<const float>(input);

        // Gather real input into contiguous per-transform layout
        if (dim == ndim - 1 && inner_size == 1) {
            queue.memcpy(real_buf, in_ptr, total_transforms * signal_len * sizeof(float));
        } else {
            queue.parallel_for(sycl::range<1>(total_transforms * signal_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / signal_len;
                    int64_t j = idx % signal_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t in_idx = b * signal_len * inner_size + j * inner_size + inner;
                    real_buf[idx] = in_ptr[in_idx];
                });
        }

        // oneMKL R2C descriptor
        // For out-of-place R2C, input is real (signal_len reals per transform),
        // output is complex (out_len complex values = out_len * 2 floats).
        // We need to set strides for both forward (real) and backward (complex) domains.
        dft::descriptor<dft::precision::SINGLE, dft::domain::REAL> desc(signal_len);
        desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, total_transforms);
        desc.set_value(dft::config_param::PLACEMENT, dft::config_value::NOT_INPLACE);

        // Forward (real) strides: contiguous reals
        std::vector<std::int64_t> fwd_strides = {0, 1};
        desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
        desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);

        // Backward (complex) strides: contiguous complex values
        // For R2C out-of-place, the output has out_len complex elements per transform
        std::vector<std::int64_t> bwd_strides = {0, 1};
        desc.set_value(dft::config_param::BWD_STRIDES, bwd_strides);
        desc.set_value(dft::config_param::BWD_DISTANCE, out_len);

        desc.commit(queue);

        dft::compute_forward(desc, real_buf,
                             reinterpret_cast<std::complex<float>*>(complex_buf));

        // Apply normalization
        double norm_factor = get_norm_factor(signal_len, norm, true);
        if (norm_factor != 1.0) {
            apply_scale_f32(queue, complex_buf, total_transforms * out_len * 2,
                            static_cast<float>(norm_factor));
        }

        // Scatter complex results to output tensor. Output is Complex64
        // (same interleaved (re, im) float storage as complex_buf).
        Tensor output(out_shape, DType::Complex64, input.device());
        float* out_ptr = reinterpret_cast<float*>(
            const_cast<void*>(output.data_ptr()));

        if (dim == ndim - 1 && inner_size == 1) {
            queue.memcpy(out_ptr, complex_buf,
                         total_transforms * out_len * 2 * sizeof(float)).wait();
        } else {
            queue.parallel_for(sycl::range<1>(total_transforms * out_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / out_len;
                    int64_t k = idx % out_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t out_idx = (b * out_len * inner_size + k * inner_size + inner) * 2;
                    out_ptr[out_idx] = complex_buf[2 * idx];
                    out_ptr[out_idx + 1] = complex_buf[2 * idx + 1];
                }).wait();
        }

        return output;

    } else if (input.dtype() == DType::Float64) {
        int64_t total_transforms = batch_size * inner_size;

        SyclDevicePtr<double> real_buf_owner(total_transforms * signal_len, queue);
        double* real_buf = real_buf_owner.get();
        SyclDevicePtr<double> complex_buf_owner(total_transforms * out_len * 2, queue);
        double* complex_buf = complex_buf_owner.get();

        const double* in_ptr = get_data_ptr<const double>(input);

        if (dim == ndim - 1 && inner_size == 1) {
            queue.memcpy(real_buf, in_ptr, total_transforms * signal_len * sizeof(double));
        } else {
            queue.parallel_for(sycl::range<1>(total_transforms * signal_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / signal_len;
                    int64_t j = idx % signal_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t in_idx = b * signal_len * inner_size + j * inner_size + inner;
                    real_buf[idx] = in_ptr[in_idx];
                });
        }

        dft::descriptor<dft::precision::DOUBLE, dft::domain::REAL> desc(signal_len);
        desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, total_transforms);
        desc.set_value(dft::config_param::PLACEMENT, dft::config_value::NOT_INPLACE);

        std::vector<std::int64_t> fwd_strides = {0, 1};
        desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
        desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);

        std::vector<std::int64_t> bwd_strides = {0, 1};
        desc.set_value(dft::config_param::BWD_STRIDES, bwd_strides);
        desc.set_value(dft::config_param::BWD_DISTANCE, out_len);

        desc.commit(queue);

        dft::compute_forward(desc, real_buf,
                             reinterpret_cast<std::complex<double>*>(complex_buf));

        double norm_factor = get_norm_factor(signal_len, norm, true);
        if (norm_factor != 1.0) {
            apply_scale_f64(queue, complex_buf, total_transforms * out_len * 2, norm_factor);
        }

        Tensor output(out_shape, DType::Complex128, input.device());
        double* out_ptr = reinterpret_cast<double*>(
            const_cast<void*>(output.data_ptr()));

        if (dim == ndim - 1 && inner_size == 1) {
            queue.memcpy(out_ptr, complex_buf,
                         total_transforms * out_len * 2 * sizeof(double)).wait();
        } else {
            queue.parallel_for(sycl::range<1>(total_transforms * out_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / out_len;
                    int64_t k = idx % out_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t out_idx = (b * out_len * inner_size + k * inner_size + inner) * 2;
                    out_ptr[out_idx] = complex_buf[2 * idx];
                    out_ptr[out_idx + 1] = complex_buf[2 * idx + 1];
                }).wait();
        }

        return output;

    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        int64_t numel = input.numel();

        Tensor f32_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         DType::Float32, input.device());
        device_upcast_to_f32(input.data_ptr(),
                             static_cast<float*>(const_cast<void*>(f32_input.data_ptr())),
                             numel, orig_dtype, queue);

        // There is no Complex16/Complex32 dtype, so half/bfloat input promotes
        // to Float32 and rfft returns Complex64 directly. This is the canonical,
        // build-independent contract: the non-oneMKL fallback rfft does exactly
        // the same (no downcast, no trailing-2), so an F16 rfft yields Complex64
        // under both builds.
        return rfft_kernel(f32_input, dim, n, norm, queue);

    } else {
        throw std::runtime_error("rfft_kernel: unsupported dtype (expected Float32 or Float64)");
    }
}

// ============================================================================
// IRFFT - 1D Complex-to-Real Inverse FFT
// ============================================================================
//
// Input: complex tensor shape (..., N/2+1, ...) with dtype Complex64/128
// Output: real tensor shape (..., n, ...) with dtype Float32/64
//
// oneMKL backward transform with domain::REAL performs C2R.
//
// The Complex64 physical layout is identical to the previous
// "Float32 + trailing 2" contract this kernel used to accept, so the
// per-element load/store in the scatter/gather loops is unchanged; only
// the shape-accounting and dtype checks move.
// ============================================================================
auto irfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                  const std::string& norm, sycl::queue& queue) -> Tensor {
    namespace dft = ::oneapi::mkl::dft;

    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    // Accept real (Float32/Float64) input by promoting to the matching
    // complex dtype with zero imaginary part — matches the CPU irfft kernel
    // contract. The top-level `tenzor::fft::irfft` does not promote, so the
    // backend has to.
    if (input.dtype() == DType::Float32) {
        return irfft_kernel(input.to(DType::Complex64), dim, n, norm, queue);
    } else if (input.dtype() == DType::Float64) {
        return irfft_kernel(input.to(DType::Complex128), dim, n, norm, queue);
    } else if (input.dtype() != DType::Complex64 && input.dtype() != DType::Complex128) {
        throw std::runtime_error(
            "irfft_kernel: expected real or complex input");
    }

    int64_t complex_len = shape[dim]; // N/2 + 1 (provided complex bins)
    int64_t output_len = n;           // Full real output length

    // oneMKL's C2R backward transform of size output_len reads exactly
    // output_len/2+1 complex inputs per transform. If the caller passes an
    // explicit n whose half-spectrum is LARGER than the provided bins
    // (output_len/2+1 > complex_len), we must zero-pad the frequency axis up to
    // expected_complex; otherwise oneMKL reads past the gathered data (OOB).
    // PyTorch zero-pads the frequency axis to n/2+1 in this case. The truncation
    // case (complex_len > expected_complex) is benign — we copy only the bins
    // the transform consumes.
    const int64_t expected_complex = output_len / 2 + 1;
    const int64_t gathered_len = std::max(complex_len, expected_complex);
    const int64_t copy_bins = std::min(complex_len, expected_complex);

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) batch_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    // Output shape: replace dim with output_len.
    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = output_len;

    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

    if (input.dtype() == DType::Complex64) {
        int64_t total_transforms = batch_size * inner_size;

        // Allocate contiguous complex input buffer (per-transform length
        // gathered_len so oneMKL can read its expected_complex bins) and real
        // output buffer. Zero-fill so any padding bins are 0.
        SyclDevicePtr<float> complex_buf_owner(total_transforms * gathered_len * 2, queue);
        float* complex_buf = complex_buf_owner.get();
        SyclDevicePtr<float> real_buf_owner(total_transforms * output_len, queue);
        float* real_buf = real_buf_owner.get();
        const int64_t g_len = gathered_len;
        const int64_t c_bins = copy_bins;

        queue.memset(complex_buf, 0,
                     total_transforms * gathered_len * 2 * sizeof(float)).wait();

        // Read Complex64 input as interleaved (re, im) floats.
        const float* in_ptr = reinterpret_cast<const float*>(input.data_ptr());

        // Gather copy_bins complex values per transform into the gathered_len-strided buffer.
        queue.parallel_for(sycl::range<1>(total_transforms * c_bins),
            [=](sycl::id<1> flat_idx) {
                int64_t idx = flat_idx[0];
                int64_t t = idx / c_bins;
                int64_t j = idx % c_bins;
                int64_t b = t / inner_size;
                int64_t inner = t % inner_size;
                int64_t in_idx = (b * complex_len * inner_size + j * inner_size + inner) * 2;
                int64_t buf_idx = t * g_len + j;
                complex_buf[2 * buf_idx]     = in_ptr[in_idx];
                complex_buf[2 * buf_idx + 1] = in_ptr[in_idx + 1];
            });

        // oneMKL C2R via backward transform with domain::REAL
        // The descriptor size is the REAL output length (output_len),
        // and the complex input has output_len/2+1 elements.
        dft::descriptor<dft::precision::SINGLE, dft::domain::REAL> desc(output_len);
        desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, total_transforms);
        desc.set_value(dft::config_param::PLACEMENT, dft::config_value::NOT_INPLACE);

        // Forward (real) strides — for the real output side
        std::vector<std::int64_t> fwd_strides = {0, 1};
        desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
        desc.set_value(dft::config_param::FWD_DISTANCE, output_len);

        // Backward (complex) strides — for the complex input side.
        // Per-transform distance is gathered_len (= expected_complex when padding,
        // matching the output_len/2+1 bins oneMKL consumes per transform).
        std::vector<std::int64_t> bwd_strides = {0, 1};
        desc.set_value(dft::config_param::BWD_STRIDES, bwd_strides);
        desc.set_value(dft::config_param::BWD_DISTANCE, gathered_len);

        desc.commit(queue);

        dft::compute_backward(desc,
                              reinterpret_cast<std::complex<float>*>(complex_buf),
                              real_buf);

        // Apply normalization
        double norm_factor = get_norm_factor(output_len, norm, false);
        if (norm_factor != 1.0) {
            apply_scale_real_f32(queue, real_buf, total_transforms * output_len,
                                 static_cast<float>(norm_factor));
        }

        // Scatter real results to output tensor
        Tensor output(out_shape, DType::Float32, input.device());
        float* out_ptr = get_data_ptr<float>(output);

        if (dim == ndim - 1 && inner_size == 1) {
            queue.memcpy(out_ptr, real_buf,
                         total_transforms * output_len * sizeof(float)).wait();
        } else {
            queue.parallel_for(sycl::range<1>(total_transforms * output_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / output_len;
                    int64_t j = idx % output_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t out_idx = b * output_len * inner_size + j * inner_size + inner;
                    out_ptr[out_idx] = real_buf[idx];
                }).wait();
        }

        return output;

    } else if (input.dtype() == DType::Complex128) {
        int64_t total_transforms = batch_size * inner_size;

        SyclDevicePtr<double> complex_buf_owner(total_transforms * gathered_len * 2, queue);
        double* complex_buf = complex_buf_owner.get();
        SyclDevicePtr<double> real_buf_owner(total_transforms * output_len, queue);
        double* real_buf = real_buf_owner.get();
        const int64_t g_len = gathered_len;
        const int64_t c_bins = copy_bins;

        queue.memset(complex_buf, 0,
                     total_transforms * gathered_len * 2 * sizeof(double)).wait();

        const double* in_ptr = reinterpret_cast<const double*>(input.data_ptr());

        // Gather copy_bins complex values per transform into the gathered_len-strided buffer.
        queue.parallel_for(sycl::range<1>(total_transforms * c_bins),
            [=](sycl::id<1> flat_idx) {
                int64_t idx = flat_idx[0];
                int64_t t = idx / c_bins;
                int64_t j = idx % c_bins;
                int64_t b = t / inner_size;
                int64_t inner = t % inner_size;
                int64_t in_idx = (b * complex_len * inner_size + j * inner_size + inner) * 2;
                int64_t buf_idx = t * g_len + j;
                complex_buf[2 * buf_idx]     = in_ptr[in_idx];
                complex_buf[2 * buf_idx + 1] = in_ptr[in_idx + 1];
            });

        dft::descriptor<dft::precision::DOUBLE, dft::domain::REAL> desc(output_len);
        desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, total_transforms);
        desc.set_value(dft::config_param::PLACEMENT, dft::config_value::NOT_INPLACE);

        std::vector<std::int64_t> fwd_strides = {0, 1};
        desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
        desc.set_value(dft::config_param::FWD_DISTANCE, output_len);

        std::vector<std::int64_t> bwd_strides = {0, 1};
        desc.set_value(dft::config_param::BWD_STRIDES, bwd_strides);
        desc.set_value(dft::config_param::BWD_DISTANCE, gathered_len);

        desc.commit(queue);

        dft::compute_backward(desc,
                              reinterpret_cast<std::complex<double>*>(complex_buf),
                              real_buf);

        double norm_factor = get_norm_factor(output_len, norm, false);
        if (norm_factor != 1.0) {
            apply_scale_real_f64(queue, real_buf, total_transforms * output_len, norm_factor);
        }

        Tensor output(out_shape, DType::Float64, input.device());
        double* out_ptr = get_data_ptr<double>(output);

        if (dim == ndim - 1 && inner_size == 1) {
            queue.memcpy(out_ptr, real_buf,
                         total_transforms * output_len * sizeof(double)).wait();
        } else {
            queue.parallel_for(sycl::range<1>(total_transforms * output_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / output_len;
                    int64_t j = idx % output_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t out_idx = b * output_len * inner_size + j * inner_size + inner;
                    out_ptr[out_idx] = real_buf[idx];
                }).wait();
        }

        return output;

    } else {
        throw std::runtime_error(
            "irfft_kernel: unsupported dtype (expected Complex64 or Complex128)");
    }
}

// ============================================================================
// FFT2 - 2D FFT (apply FFT along two dimensions)
// ============================================================================
auto fft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                 const std::vector<int64_t>& signal_lengths,
                 const std::string& norm, sycl::queue& queue) -> Tensor {
    // Apply FFT along each dimension sequentially
    Tensor result = fft_kernel(input, dims[0], signal_lengths[0], norm, queue);
    result = fft_kernel(result, dims[1], signal_lengths[1], norm, queue);
    return result;
}

// ============================================================================
// IFFT2 - 2D Inverse FFT
// ============================================================================
auto ifft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                  const std::vector<int64_t>& signal_lengths,
                  const std::string& norm, sycl::queue& queue) -> Tensor {
    Tensor result = ifft_kernel(input, dims[0], signal_lengths[0], norm, queue);
    result = ifft_kernel(result, dims[1], signal_lengths[1], norm, queue);
    return result;
}

// ============================================================================
// FFTN - N-dimensional FFT
// ============================================================================
auto fftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                 const std::vector<int64_t>& signal_lengths,
                 const std::string& norm, sycl::queue& queue) -> Tensor {
    Tensor result = clone_kernel(input, queue);
    for (size_t i = 0; i < dims.size(); ++i) {
        result = fft_kernel(result, dims[i], signal_lengths[i], norm, queue);
    }
    return result;
}

// ============================================================================
// IFFTN - N-dimensional Inverse FFT
// ============================================================================
auto ifftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                  const std::vector<int64_t>& signal_lengths,
                  const std::string& norm, sycl::queue& queue) -> Tensor {
    Tensor result = clone_kernel(input, queue);
    for (size_t i = 0; i < dims.size(); ++i) {
        result = ifft_kernel(result, dims[i], signal_lengths[i], norm, queue);
    }
    return result;
}

#else // !TENZOR_HAS_ONEMKL — Cooley-Tukey (power-of-2) + Bluestein (general) FFT fallback
#pragma message("WARNING: Building without oneMKL — using slower FFT fallback")

namespace {

// Check if n is a power of 2
inline bool is_power_of_2(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Pad (zero-fill) or truncate `input` along axis `dim` so that shape[dim] == n,
// entirely on-device. Matches the oneMKL path's pad/truncate semantics so the
// fallback honors a caller-requested transform length n != shape[dim]. Returns
// `input` unchanged when no resize is needed. Works for any dtype/element size:
// each (outer, dim-index) position owns `inner * dtype_size` contiguous bytes,
// where inner is the product of the dimensions after `dim`.
inline Tensor fft_pad_or_truncate(const Tensor& input, int64_t dim, int64_t n,
                                  sycl::queue& queue) {
    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    const int64_t ndim = static_cast<int64_t>(shape.size());
    const int64_t signal_len = shape[dim];
    if (n == signal_len) return input;

    std::vector<int64_t> new_shape = shape;
    new_shape[dim] = n;

    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner *= shape[i];

    const int64_t elem_bytes = static_cast<int64_t>(input.dtype_size());
    const int64_t slice_bytes = inner * elem_bytes;       // bytes per dim-index slice
    const int64_t copy_len = std::min(signal_len, n);

    Tensor out(new_shape, input.dtype(), input.device());
    // Zero the whole buffer so any padding region is zero-filled.
    queue.memset(const_cast<void*>(out.data_ptr()), 0,
                 static_cast<size_t>(out.numel() * elem_bytes)).wait();

    const char* src = static_cast<const char*>(input.data_ptr());
    char* dst = static_cast<char*>(const_cast<void*>(out.data_ptr()));
    const int64_t old_dim = signal_len;
    const int64_t new_dim = n;
    const int64_t total_bytes = outer * copy_len * slice_bytes;
    queue.parallel_for(sycl::range<1>(total_bytes), [=](sycl::id<1> idx) {
        int64_t b = idx[0];
        int64_t byte_in_slice = b % slice_bytes;
        int64_t pos = b / slice_bytes;          // (outer, dim-index) flattened
        int64_t d = pos % copy_len;
        int64_t o = pos / copy_len;
        int64_t src_off = (o * old_dim + d) * slice_bytes + byte_in_slice;
        int64_t dst_off = (o * new_dim + d) * slice_bytes + byte_in_slice;
        dst[dst_off] = src[src_off];
    }).wait();

    return out;
}

// Templated Cooley-Tukey FFT on device using SYCL parallel_for kernels.
// Operates on interleaved complex data: [re0, im0, re1, im1, ...].
// Supports batched execution: data contains batch_size independent FFTs,
// each of length N complex elements, separated by batch_stride floats.
// sign = -1.0 for forward FFT, +1.0 for inverse FFT.
template<typename T>
void cooley_tukey_fft_sycl(T* data, int64_t N, int64_t batch_size, int64_t batch_stride,
                           T sign, sycl::queue& queue) {
    int log2N = 0;
    { int64_t tmp = N; while (tmp > 1) { tmp >>= 1; log2N++; } }

    // Step 1: Bit-reverse permutation (on device) — all batches in one dispatch
    const int bits = log2N;
    int64_t total_items = N * batch_size;
    queue.parallel_for(sycl::range<1>(total_items), [=](sycl::id<1> idx) {
        int64_t global_id = idx[0];
        int64_t batch_idx = global_id / N;
        int64_t i = global_id % N;
        uint32_t rev = 0;
        uint32_t x = static_cast<uint32_t>(i);
        for (int b = 0; b < bits; ++b) {
            rev = (rev << 1) | (x & 1);
            x >>= 1;
        }
        int64_t j = static_cast<int64_t>(rev);
        if (i < j) {
            int64_t base = batch_idx * batch_stride;
            T tmp_re = data[base + 2 * i];
            T tmp_im = data[base + 2 * i + 1];
            data[base + 2 * i] = data[base + 2 * j];
            data[base + 2 * i + 1] = data[base + 2 * j + 1];
            data[base + 2 * j] = tmp_re;
            data[base + 2 * j + 1] = tmp_im;
        }
    }).wait();

    // Twiddle factors are computed in double precision regardless of T so the
    // radix-2 phase chain does not accumulate single-precision trig error for
    // T=float (it is narrowed to T only after cos/sin), matching oneMKL/CPU.
    constexpr double PI = 3.14159265358979323846;
    const double sign_d = static_cast<double>(sign);

    // Step 2: Butterfly stages — all batches in one dispatch per stage
    int64_t num_butterflies = N / 2;
    int64_t total_butterflies = num_butterflies * batch_size;
    for (int s = 1; s <= log2N; ++s) {
        int64_t stride = static_cast<int64_t>(1) << s;
        int64_t half = stride / 2;

        queue.parallel_for(sycl::range<1>(total_butterflies), [=](sycl::id<1> idx) {
            int64_t global_id = idx[0];
            int64_t batch_idx = global_id / num_butterflies;
            int64_t flat = global_id % num_butterflies;
            int64_t group = flat / half;
            int64_t k = flat % half;
            int64_t base_idx = group * stride;

            double angle = sign_d * 2.0 * PI * static_cast<double>(k) / static_cast<double>(stride);
            T w_re = static_cast<T>(sycl::cos(angle));
            T w_im = static_cast<T>(sycl::sin(angle));

            int64_t base = batch_idx * batch_stride;
            int64_t even_i = base_idx + k;
            int64_t odd_i = base_idx + k + half;

            T e_re = data[base + 2 * even_i];
            T e_im = data[base + 2 * even_i + 1];
            T o_re = data[base + 2 * odd_i];
            T o_im = data[base + 2 * odd_i + 1];

            T t_re = w_re * o_re - w_im * o_im;
            T t_im = w_re * o_im + w_im * o_re;

            data[base + 2 * even_i] = e_re + t_re;
            data[base + 2 * even_i + 1] = e_im + t_im;
            data[base + 2 * odd_i] = e_re - t_re;
            data[base + 2 * odd_i + 1] = e_im - t_im;
        }).wait();
    }
}

// Next power of 2 >= n
inline int64_t next_pow2(int64_t n) {
    int64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Bluestein FFT on device for non-power-of-2 sizes.
// Converts N-point DFT into a 2M-point circular convolution (M = next power of 2 >= 2N-1)
// using the Cooley-Tukey FFT for the power-of-2 convolution.
//
// Input:  d_in  — real input on device, layout [batch_size, signal_len, inner_size]
// Output: d_out — interleaved complex output on device,
//                 layout [batch_size, signal_len, inner_size, 2]
template<typename T>
void bluestein_fft_sycl(const T* d_in, T* d_out,
                        int64_t signal_len, int64_t batch_size, int64_t inner_size,
                        sycl::queue& queue) {
    const int64_t N = signal_len;
    const int64_t M = next_pow2(2 * N - 1);
    constexpr T PI = static_cast<T>(3.14159265358979323846);

    // Allocate device buffers:
    // chirp: interleaved complex [N, 2]
    // b_buf: interleaved complex [M, 2] — convolution kernel (shared across all batches/inner)
    // B_buf: FFT of b_buf [M, 2] — precomputed once
    // a_buf: interleaved complex [M, 2] — per-(batch, inner) working buffer
    SyclDevicePtr<T> chirp_owner(2 * N, queue);
    T* chirp   = chirp_owner.get();
    SyclDevicePtr<T> b_buf_owner(2 * M, queue);
    T* b_buf   = b_buf_owner.get();
    SyclDevicePtr<T> B_buf_owner(2 * M, queue);
    T* B_buf   = B_buf_owner.get();
    // Batched working buffer: one a_buf per (batch, inner) slice
    int64_t total_slices = batch_size * inner_size;
    SyclDevicePtr<T> a_buf_owner(2 * M * total_slices, queue);
    T* a_buf   = a_buf_owner.get();

    // Step 1: Generate chirp sequence: chirp[k] = exp(-j * pi * k^2 / N)
    queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        int64_t k = idx[0];
        T angle = -PI * static_cast<T>(k) * static_cast<T>(k) / static_cast<T>(N);
        chirp[2 * k]     = sycl::cos(angle);
        chirp[2 * k + 1] = sycl::sin(angle);
    }).wait();

    // Step 2: Build convolution kernel b[k] = conj(chirp[k]) for k=0..N-1,
    //         b[M-k] = conj(chirp[k]) for k=1..N-1, zeros elsewhere
    queue.memset(b_buf, 0, 2 * M * sizeof(T)).wait();
    queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        int64_t k = idx[0];
        b_buf[2 * k]     = chirp[2 * k];
        b_buf[2 * k + 1] = -chirp[2 * k + 1];
    }).wait();
    if (N > 1) {
        queue.parallel_for(sycl::range<1>(N - 1), [=](sycl::id<1> idx) {
            int64_t k = idx[0] + 1;
            int64_t m_idx = M - k;
            b_buf[2 * m_idx]     = chirp[2 * k];
            b_buf[2 * m_idx + 1] = -chirp[2 * k + 1];
        }).wait();
    }

    // Step 3: Precompute B = FFT(b) — power-of-2 size M (single, shared across all slices)
    queue.memcpy(B_buf, b_buf, 2 * M * sizeof(T)).wait();
    cooley_tukey_fft_sycl(B_buf, M, static_cast<int64_t>(1), static_cast<int64_t>(2 * M),
                          static_cast<T>(-1.0), queue);

    // Steps 4-8: Process all (batch, inner) slices in parallel

    // Step 4: Zero a_buf and build a[s][k] = x[b,k,inner] * chirp[k] for all slices
    queue.memset(a_buf, 0, 2 * M * total_slices * sizeof(T)).wait();
    queue.parallel_for(sycl::range<1>(N * total_slices), [=](sycl::id<1> idx) {
        int64_t global_id = idx[0];
        int64_t s = global_id / N;
        int64_t k = global_id % N;
        int64_t b = s / inner_size;
        int64_t inner = s % inner_size;
        int64_t in_idx = b * N * inner_size + k * inner_size + inner;
        T val = d_in[in_idx];
        int64_t a_base = s * 2 * M;
        a_buf[a_base + 2 * k]     = val * chirp[2 * k];
        a_buf[a_base + 2 * k + 1] = val * chirp[2 * k + 1];
    }).wait();

    // Step 5: A = FFT(a) — batched over all slices
    cooley_tukey_fft_sycl(a_buf, M, total_slices, static_cast<int64_t>(2 * M),
                          static_cast<T>(-1.0), queue);

    // Step 6: Pointwise multiply A[s][k] *= B[k] for all slices
    queue.parallel_for(sycl::range<1>(M * total_slices), [=](sycl::id<1> idx) {
        int64_t global_id = idx[0];
        int64_t s = global_id / M;
        int64_t k = global_id % M;
        int64_t a_base = s * 2 * M;
        T a_re = a_buf[a_base + 2 * k];
        T a_im = a_buf[a_base + 2 * k + 1];
        T b_re = B_buf[2 * k];
        T b_im = B_buf[2 * k + 1];
        a_buf[a_base + 2 * k]     = a_re * b_re - a_im * b_im;
        a_buf[a_base + 2 * k + 1] = a_re * b_im + a_im * b_re;
    }).wait();

    // Step 7: IFFT via forward FFT with sign=+1, divide by M — batched
    cooley_tukey_fft_sycl(a_buf, M, total_slices, static_cast<int64_t>(2 * M),
                          static_cast<T>(1.0), queue);
    T inv_M = static_cast<T>(1.0) / static_cast<T>(M);
    queue.parallel_for(sycl::range<1>(2 * M * total_slices), [=](sycl::id<1> idx) {
        a_buf[idx[0]] *= inv_M;
    }).wait();

    // Step 8: result[b,k,inner] = a[s][k] * conj(chirp[k]) for all slices
    queue.parallel_for(sycl::range<1>(N * total_slices), [=](sycl::id<1> idx) {
        int64_t global_id = idx[0];
        int64_t s = global_id / N;
        int64_t k = global_id % N;
        int64_t b = s / inner_size;
        int64_t inner = s % inner_size;
        int64_t a_base = s * 2 * M;
        T a_re = a_buf[a_base + 2 * k];
        T a_im = a_buf[a_base + 2 * k + 1];
        T c_re = chirp[2 * k];
        T c_im = -chirp[2 * k + 1]; // conj
        int64_t out_idx = (b * N * inner_size + k * inner_size + inner) * 2;
        d_out[out_idx]     = a_re * c_re - a_im * c_im;
        d_out[out_idx + 1] = a_re * c_im + a_im * c_re;
    }).wait();

}

// Bluestein FFT on device for non-power-of-2 sizes with COMPLEX input.
// Same algorithm as bluestein_fft_sycl but d_in is interleaved complex:
//   layout [batch_size, signal_len, inner_size, 2]
// sign = -1.0 for forward, +1.0 for inverse (before normalization).
// Output: d_out — interleaved complex, layout [batch_size, signal_len, inner_size, 2]
template<typename T>
void bluestein_fft_complex_sycl(const T* d_in, T* d_out,
                                int64_t signal_len, int64_t batch_size, int64_t inner_size,
                                T sign, sycl::queue& queue) {
    const int64_t N = signal_len;
    const int64_t M = next_pow2(2 * N - 1);
    constexpr T PI = static_cast<T>(3.14159265358979323846);

    SyclDevicePtr<T> chirp_owner(2 * N, queue);
    T* chirp   = chirp_owner.get();
    SyclDevicePtr<T> b_buf_owner(2 * M, queue);
    T* b_buf   = b_buf_owner.get();
    SyclDevicePtr<T> B_buf_owner(2 * M, queue);
    T* B_buf   = B_buf_owner.get();
    // Batched working buffer: one a_buf per (batch, inner) slice
    int64_t total_slices = batch_size * inner_size;
    SyclDevicePtr<T> a_buf_owner(2 * M * total_slices, queue);
    T* a_buf   = a_buf_owner.get();

    // Step 1: chirp[k] = exp(sign * j * pi * k^2 / N)
    queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        int64_t k = idx[0];
        T angle = sign * PI * static_cast<T>(k) * static_cast<T>(k) / static_cast<T>(N);
        chirp[2 * k]     = sycl::cos(angle);
        chirp[2 * k + 1] = sycl::sin(angle);
    }).wait();

    // Step 2: b[k] = conj(chirp[k])
    queue.memset(b_buf, 0, 2 * M * sizeof(T)).wait();
    queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        int64_t k = idx[0];
        b_buf[2 * k]     = chirp[2 * k];
        b_buf[2 * k + 1] = -chirp[2 * k + 1];
    }).wait();
    if (N > 1) {
        queue.parallel_for(sycl::range<1>(N - 1), [=](sycl::id<1> idx) {
            int64_t k = idx[0] + 1;
            int64_t m_idx = M - k;
            b_buf[2 * m_idx]     = chirp[2 * k];
            b_buf[2 * m_idx + 1] = -chirp[2 * k + 1];
        }).wait();
    }

    // Step 3: B = FFT(b) — single FFT, shared across all slices
    queue.memcpy(B_buf, b_buf, 2 * M * sizeof(T)).wait();
    cooley_tukey_fft_sycl(B_buf, M, static_cast<int64_t>(1), static_cast<int64_t>(2 * M),
                          static_cast<T>(-1.0), queue);

    // Steps 4-8: Process all (batch, inner) slices in parallel

    // Step 4: Zero a_buf and build a[s][k] = x[b,k,inner] * chirp[k] (complex multiply)
    queue.memset(a_buf, 0, 2 * M * total_slices * sizeof(T)).wait();
    queue.parallel_for(sycl::range<1>(N * total_slices), [=](sycl::id<1> idx) {
        int64_t global_id = idx[0];
        int64_t s = global_id / N;
        int64_t k = global_id % N;
        int64_t b = s / inner_size;
        int64_t inner = s % inner_size;
        int64_t in_idx = (b * N * inner_size + k * inner_size + inner) * 2;
        T x_re = d_in[in_idx];
        T x_im = d_in[in_idx + 1];
        T c_re = chirp[2 * k];
        T c_im = chirp[2 * k + 1];
        int64_t a_base = s * 2 * M;
        a_buf[a_base + 2 * k]     = x_re * c_re - x_im * c_im;
        a_buf[a_base + 2 * k + 1] = x_re * c_im + x_im * c_re;
    }).wait();

    // Step 5: A = FFT(a) — batched over all slices
    cooley_tukey_fft_sycl(a_buf, M, total_slices, static_cast<int64_t>(2 * M),
                          static_cast<T>(-1.0), queue);

    // Step 6: Pointwise multiply A[s][k] *= B[k] for all slices
    queue.parallel_for(sycl::range<1>(M * total_slices), [=](sycl::id<1> idx) {
        int64_t global_id = idx[0];
        int64_t s = global_id / M;
        int64_t k = global_id % M;
        int64_t a_base = s * 2 * M;
        T a_re = a_buf[a_base + 2 * k];
        T a_im = a_buf[a_base + 2 * k + 1];
        T b_re = B_buf[2 * k];
        T b_im = B_buf[2 * k + 1];
        a_buf[a_base + 2 * k]     = a_re * b_re - a_im * b_im;
        a_buf[a_base + 2 * k + 1] = a_re * b_im + a_im * b_re;
    }).wait();

    // Step 7: IFFT via forward FFT with sign=+1, divide by M — batched
    cooley_tukey_fft_sycl(a_buf, M, total_slices, static_cast<int64_t>(2 * M),
                          static_cast<T>(1.0), queue);
    T inv_M = static_cast<T>(1.0) / static_cast<T>(M);
    queue.parallel_for(sycl::range<1>(2 * M * total_slices), [=](sycl::id<1> idx) {
        a_buf[idx[0]] *= inv_M;
    }).wait();

    // Step 8: result[b,k,inner] = a[s][k] * conj(chirp[k]) for all slices
    queue.parallel_for(sycl::range<1>(N * total_slices), [=](sycl::id<1> idx) {
        int64_t global_id = idx[0];
        int64_t s = global_id / N;
        int64_t k = global_id % N;
        int64_t b = s / inner_size;
        int64_t inner = s % inner_size;
        int64_t a_base = s * 2 * M;
        T a_re = a_buf[a_base + 2 * k];
        T a_im = a_buf[a_base + 2 * k + 1];
        T c_re = chirp[2 * k];
        T c_im = -chirp[2 * k + 1]; // conj
        int64_t out_idx = (b * N * inner_size + k * inner_size + inner) * 2;
        d_out[out_idx]     = a_re * c_re - a_im * c_im;
        d_out[out_idx + 1] = a_re * c_im + a_im * c_re;
    }).wait();
}

} // anonymous namespace

// ============================================================================
// FFT - 1D Complex-to-Complex Forward FFT (Cooley-Tukey / Bluestein)
// ============================================================================
//
// Contract (build-independent, matches the oneMKL path and CPU/CUDA):
//   Input : Complex64 / Complex128 (interleaved re,im storage = physical layout)
//   Output: same shape and complex dtype, NO trailing dim of 2.
// The public `tenzor::fft::fft` op promotes real inputs to Complex64/Complex128
// before dispatch, so this kernel only ever receives a complex tensor.
//
// Internally a complex element is two consecutive floats/doubles, so numel() is
// the number of complex elements and the interleaved float/double count is 2*numel.
//
// `sign` selects forward (-1) vs inverse (+1); `inv_norm` selects the inverse
// 1/N normalization branch so fft_kernel and ifft_kernel can share this body.
namespace {
template <typename T>
Tensor fft_complex_fallback(const Tensor& input, int64_t dim, DType complex_dtype,
                            const std::string& norm, T sign, bool inv_norm,
                            sycl::queue& queue) {
    int64_t ndim = static_cast<int64_t>(input.shape().size());
    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t signal_len = shape[dim];

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) batch_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    bool use_cooley_tukey = is_power_of_2(signal_len) && inner_size == 1;

    // Output: identical shape and complex dtype (no trailing 2).
    std::vector<int64_t> out_shape = shape;
    int64_t out_complex_numel = input.numel();        // number of complex elements
    int64_t out_floats = out_complex_numel * 2;        // interleaved scalar count

    // Normalization scale (shared by forward "ortho"/"forward" and inverse).
    auto compute_scale = [&]() -> T {
        if (inv_norm) {
            if (norm == "ortho")   return static_cast<T>(1.0) / static_cast<T>(std::sqrt(static_cast<double>(signal_len)));
            if (norm == "forward") return static_cast<T>(1.0);
            return static_cast<T>(1.0) / static_cast<T>(signal_len);  // backward (default)
        } else {
            if (norm == "ortho")   return static_cast<T>(1.0) / static_cast<T>(std::sqrt(static_cast<double>(signal_len)));
            if (norm == "forward") return static_cast<T>(1.0) / static_cast<T>(signal_len);
            return static_cast<T>(1.0);  // backward (default): no scaling on forward
        }
    };
    bool need_scale = inv_norm || norm == "ortho" || norm == "forward";

    Tensor output(out_shape, complex_dtype, input.device());
    const T* d_in = static_cast<const T*>(input.data_ptr());
    T* d_outp = static_cast<T*>(const_cast<void*>(output.data_ptr()));

    if (use_cooley_tukey) {
        SyclDevicePtr<T> d_buf_owner(2 * signal_len * batch_size, queue);
        T* d_buf = d_buf_owner.get();
        // Input is already interleaved complex, contiguous per batch.
        queue.memcpy(d_buf, d_in, 2 * signal_len * batch_size * sizeof(T)).wait();

        cooley_tukey_fft_sycl(d_buf, signal_len, batch_size,
                              static_cast<int64_t>(2 * signal_len), sign, queue);

        if (need_scale) {
            T scale = compute_scale();
            queue.parallel_for(sycl::range<1>(2 * signal_len * batch_size), [=](sycl::id<1> idx) {
                d_buf[idx[0]] *= scale;
            }).wait();
        }
        queue.memcpy(d_outp, d_buf, out_floats * sizeof(T)).wait();
    } else {
        SyclDevicePtr<T> d_out_owner(out_floats, queue);
        T* d_out = d_out_owner.get();
        queue.memset(d_out, 0, out_floats * sizeof(T)).wait();
        bluestein_fft_complex_sycl(d_in, d_out, signal_len, batch_size, inner_size, sign, queue);

        if (need_scale) {
            T scale = compute_scale();
            queue.parallel_for(sycl::range<1>(out_floats), [=](sycl::id<1> idx) {
                d_out[idx[0]] *= scale;
            }).wait();
        }
        queue.memcpy(d_outp, d_out, out_floats * sizeof(T)).wait();
    }
    return output;
}
} // anonymous namespace

auto fft_kernel(const Tensor& input_arg, int64_t dim, int64_t n,
                const std::string& norm, sycl::queue& queue) -> Tensor {
    int64_t ndim = static_cast<int64_t>(input_arg.shape().size());
    if (dim < 0) dim += ndim;

    // Honor a caller-requested transform length: pad/truncate along dim to n,
    // matching the oneMKL path (and CPU/CUDA) instead of silently using shape[dim].
    Tensor input = fft_pad_or_truncate(input_arg, dim, n, queue);

    if (input.dtype() == DType::Complex64) {
        return fft_complex_fallback<float>(input, dim, DType::Complex64, norm,
                                           -1.0f, /*inv_norm=*/false, queue);
    } else if (input.dtype() == DType::Complex128) {
        return fft_complex_fallback<double>(input, dim, DType::Complex128, norm,
                                            -1.0, /*inv_norm=*/false, queue);
    } else {
        throw std::runtime_error(
            "fft_kernel: unsupported dtype (expected Complex64 or Complex128)");
    }
}

// ============================================================================
// IFFT - 1D Complex-to-Complex Inverse FFT (device-side Cooley-Tukey / Bluestein)
// ============================================================================
//
// Contract: Complex64/Complex128 in -> same shape and complex dtype out
// (no trailing 2). The public `tenzor::fft::ifft` op promotes real inputs to
// complex before dispatch, so this kernel only sees a complex tensor. Inverse
// transform = forward Cooley-Tukey/Bluestein with sign=+1 and 1/N normalization.
auto ifft_kernel(const Tensor& input_arg, int64_t dim, int64_t n,
                 const std::string& norm, sycl::queue& queue) -> Tensor {
    int64_t ndim = static_cast<int64_t>(input_arg.shape().size());
    if (dim < 0) dim += ndim;

    // Honor a caller-requested transform length: pad/truncate the complex signal
    // along dim to n. fft_pad_or_truncate works on whole elements (Complex64 is
    // 8 bytes, Complex128 16), so complex elements pad/truncate correctly.
    Tensor input = fft_pad_or_truncate(input_arg, dim, n, queue);

    if (input.dtype() == DType::Complex64) {
        return fft_complex_fallback<float>(input, dim, DType::Complex64, norm,
                                           +1.0f, /*inv_norm=*/true, queue);
    } else if (input.dtype() == DType::Complex128) {
        return fft_complex_fallback<double>(input, dim, DType::Complex128, norm,
                                            +1.0, /*inv_norm=*/true, queue);
    } else {
        throw std::runtime_error(
            "ifft_kernel: unsupported dtype (expected Complex64 or Complex128)");
    }
}

// ============================================================================
// RFFT - 1D Real-to-Complex Forward FFT (device-side Cooley-Tukey / Bluestein)
// ============================================================================
auto rfft_kernel(const Tensor& input_arg, int64_t dim, int64_t n,
                 const std::string& norm, sycl::queue& queue) -> Tensor {
    int64_t ndim = static_cast<int64_t>(input_arg.shape().size());
    if (dim < 0) dim += ndim;

    // Honor a caller-requested transform length: pad/truncate along dim to n so
    // out_len = n/2+1, matching the oneMKL path (and CPU/CUDA).
    Tensor input = fft_pad_or_truncate(input_arg, dim, n, queue);

    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

    int64_t signal_len = shape[dim];
    int64_t out_len = signal_len / 2 + 1;

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) batch_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    bool use_cooley_tukey = is_power_of_2(signal_len) && inner_size == 1;

    // Strategy: compute full N-point forward FFT on device, then truncate to N/2+1 bins.
    // Output contract matches the oneMKL path (and CPU/CUDA): Complex64/Complex128
    // with shape[dim] == N/2+1 and NO trailing 2 dim. The interleaved (re,im) float
    // storage of the complex tensor is the same physical layout the truncation loops
    // already write, so only the shape/dtype bookkeeping changes.
    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = out_len;

    if (input.dtype() == DType::Float32) {
        if (use_cooley_tukey) {
            // Pack real input into interleaved complex buffer [re, 0, re, 0, ...]
            SyclDevicePtr<float> d_buf_owner(2 * signal_len * batch_size, queue);
            float* d_buf = d_buf_owner.get();
            const float* d_in = static_cast<const float*>(input.data_ptr());
            int64_t total = signal_len * batch_size;
            queue.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
                int64_t i = idx[0];
                d_buf[2 * i] = d_in[i];
                d_buf[2 * i + 1] = 0.0f;
            }).wait();

            cooley_tukey_fft_sycl(d_buf, signal_len, batch_size,
                                  static_cast<int64_t>(2 * signal_len), -1.0f, queue);

            // Apply normalization
            if (norm == "ortho" || norm == "forward") {
                float scale = (norm == "ortho")
                    ? 1.0f / std::sqrt(static_cast<float>(signal_len))
                    : 1.0f / static_cast<float>(signal_len);
                queue.parallel_for(sycl::range<1>(2 * total), [=](sycl::id<1> idx) {
                    d_buf[idx[0]] *= scale;
                }).wait();
            }

            // Truncate: copy first out_len complex bins per batch to output
            Tensor output(out_shape, DType::Complex64, input.device());
            float* d_out = static_cast<float*>(const_cast<void*>(output.data_ptr()));
            queue.parallel_for(sycl::range<1>(batch_size * out_len), [=](sycl::id<1> idx) {
                int64_t flat = idx[0];
                int64_t b = flat / out_len;
                int64_t k = flat % out_len;
                int64_t src = b * 2 * signal_len + 2 * k;
                int64_t dst = (b * out_len + k) * 2;
                d_out[dst]     = d_buf[src];
                d_out[dst + 1] = d_buf[src + 1];
            }).wait();
            return output;
        } else {
            // Bluestein: compute full N-point FFT, then truncate
            const float* d_in = static_cast<const float*>(input.data_ptr());
            int64_t full_complex_numel = batch_size * signal_len * inner_size * 2;
            SyclDevicePtr<float> d_full_owner(full_complex_numel, queue);
            float* d_full = d_full_owner.get();
            queue.memset(d_full, 0, full_complex_numel * sizeof(float)).wait();
            bluestein_fft_sycl(d_in, d_full, signal_len, batch_size, inner_size, queue);

            // Apply normalization to full output
            if (norm == "ortho" || norm == "forward") {
                float scale = (norm == "ortho")
                    ? 1.0f / std::sqrt(static_cast<float>(signal_len))
                    : 1.0f / static_cast<float>(signal_len);
                queue.parallel_for(sycl::range<1>(full_complex_numel), [=](sycl::id<1> idx) {
                    d_full[idx[0]] *= scale;
                }).wait();
            }

            // Truncate: copy first out_len bins per (batch, inner)
            Tensor output(out_shape, DType::Complex64, input.device());
            float* d_out = static_cast<float*>(const_cast<void*>(output.data_ptr()));
            int64_t copy_count = batch_size * out_len * inner_size;
            queue.parallel_for(sycl::range<1>(copy_count), [=](sycl::id<1> idx) {
                int64_t flat = idx[0];
                int64_t b = flat / (out_len * inner_size);
                int64_t rem = flat % (out_len * inner_size);
                int64_t k = rem / inner_size;
                int64_t inner = rem % inner_size;
                int64_t src = (b * signal_len * inner_size + k * inner_size + inner) * 2;
                int64_t dst = (b * out_len * inner_size + k * inner_size + inner) * 2;
                d_out[dst]     = d_full[src];
                d_out[dst + 1] = d_full[src + 1];
            }).wait();
            return output;
        }
    } else if (input.dtype() == DType::Float64) {
        if (use_cooley_tukey) {
            SyclDevicePtr<double> d_buf_owner(2 * signal_len * batch_size, queue);
            double* d_buf = d_buf_owner.get();
            const double* d_in = static_cast<const double*>(input.data_ptr());
            int64_t total = signal_len * batch_size;
            queue.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
                int64_t i = idx[0];
                d_buf[2 * i] = d_in[i];
                d_buf[2 * i + 1] = 0.0;
            }).wait();

            cooley_tukey_fft_sycl(d_buf, signal_len, batch_size,
                                  static_cast<int64_t>(2 * signal_len), -1.0, queue);

            if (norm == "ortho" || norm == "forward") {
                double scale = (norm == "ortho")
                    ? 1.0 / std::sqrt(static_cast<double>(signal_len))
                    : 1.0 / static_cast<double>(signal_len);
                queue.parallel_for(sycl::range<1>(2 * total), [=](sycl::id<1> idx) {
                    d_buf[idx[0]] *= scale;
                }).wait();
            }

            Tensor output(out_shape, DType::Complex128, input.device());
            double* d_out = static_cast<double*>(const_cast<void*>(output.data_ptr()));
            queue.parallel_for(sycl::range<1>(batch_size * out_len), [=](sycl::id<1> idx) {
                int64_t flat = idx[0];
                int64_t b = flat / out_len;
                int64_t k = flat % out_len;
                int64_t src = b * 2 * signal_len + 2 * k;
                int64_t dst = (b * out_len + k) * 2;
                d_out[dst]     = d_buf[src];
                d_out[dst + 1] = d_buf[src + 1];
            }).wait();
            return output;
        } else {
            const double* d_in = static_cast<const double*>(input.data_ptr());
            int64_t full_complex_numel = batch_size * signal_len * inner_size * 2;
            SyclDevicePtr<double> d_full_owner(full_complex_numel, queue);
            double* d_full = d_full_owner.get();
            queue.memset(d_full, 0, full_complex_numel * sizeof(double)).wait();
            bluestein_fft_sycl(d_in, d_full, signal_len, batch_size, inner_size, queue);

            if (norm == "ortho" || norm == "forward") {
                double scale = (norm == "ortho")
                    ? 1.0 / std::sqrt(static_cast<double>(signal_len))
                    : 1.0 / static_cast<double>(signal_len);
                queue.parallel_for(sycl::range<1>(full_complex_numel), [=](sycl::id<1> idx) {
                    d_full[idx[0]] *= scale;
                }).wait();
            }

            Tensor output(out_shape, DType::Complex128, input.device());
            double* d_out = static_cast<double*>(const_cast<void*>(output.data_ptr()));
            int64_t copy_count = batch_size * out_len * inner_size;
            queue.parallel_for(sycl::range<1>(copy_count), [=](sycl::id<1> idx) {
                int64_t flat = idx[0];
                int64_t b = flat / (out_len * inner_size);
                int64_t rem = flat % (out_len * inner_size);
                int64_t k = rem / inner_size;
                int64_t inner = rem % inner_size;
                int64_t src = (b * signal_len * inner_size + k * inner_size + inner) * 2;
                int64_t dst = (b * out_len * inner_size + k * inner_size + inner) * 2;
                d_out[dst]     = d_full[src];
                d_out[dst + 1] = d_full[src + 1];
            }).wait();
            return output;
        }
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        // There is no Complex16/Complex32 dtype, so half/bfloat input promotes to
        // Float32 and returns Complex64 directly — matching the oneMKL rfft path
        // (which also returns Complex64 for half inputs, see finding 4). No
        // downcast and no trailing-2 dim, so the contract is build-independent.
        int64_t in_numel = input.numel();
        DType orig_dtype = input.dtype();

        Tensor f32_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         DType::Float32, input.device());
        device_upcast_to_f32(input.data_ptr(),
                             static_cast<float*>(const_cast<void*>(f32_input.data_ptr())),
                             in_numel, orig_dtype, queue);
        return rfft_kernel(f32_input, dim, n, norm, queue);
    } else {
        throw std::runtime_error("rfft_kernel: unsupported dtype (expected Float32 or Float64)");
    }
}

// ============================================================================
// IRFFT - 1D Complex-to-Real Inverse FFT (device-side Cooley-Tukey / Bluestein)
// ============================================================================
auto irfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                  const std::string& norm, sycl::queue& queue) -> Tensor {
    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    // Contract: complex input (Complex64/Complex128), real output (Float32/Float64),
    // shape[dim] is N/2+1 frequency bins, output dim is the requested length n. No
    // trailing-2 dim — the interleaved (re,im) storage IS the physical layout of the
    // complex tensor. The public `tenzor::fft::irfft` op does not promote, so real and
    // half inputs are widened to the matching complex dtype here (mirrors the oneMKL
    // path and the CPU kernel).
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        int64_t numel = input.numel();
        DType orig_dtype = input.dtype();
        Tensor f32_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         DType::Float32, input.device());
        device_upcast_to_f32(input.data_ptr(),
                             static_cast<float*>(const_cast<void*>(f32_input.data_ptr())),
                             numel, orig_dtype, queue);
        return irfft_kernel(f32_input.to(DType::Complex64), dim, n, norm, queue);
    } else if (input.dtype() == DType::Float32) {
        return irfft_kernel(input.to(DType::Complex64), dim, n, norm, queue);
    } else if (input.dtype() == DType::Float64) {
        return irfft_kernel(input.to(DType::Complex128), dim, n, norm, queue);
    } else if (input.dtype() != DType::Complex64 && input.dtype() != DType::Complex128) {
        throw std::runtime_error(
            "irfft_kernel: unsupported dtype (expected Complex64 or Complex128)");
    }

    int64_t complex_len = shape[dim];  // N/2+1 input bins
    int64_t output_len = n;            // N output points

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) batch_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    bool use_cooley_tukey = is_power_of_2(output_len) && inner_size == 1;

    // Build real output shape: replace dim with output_len (no trailing 2 dim).
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < dim; ++i) out_shape.push_back(shape[i]);
    out_shape.push_back(output_len);
    for (int64_t i = dim + 1; i < ndim; ++i) out_shape.push_back(shape[i]);
    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

    // Strategy: reconstruct full N-point complex spectrum from N/2+1 bins
    // using conjugate symmetry: X[k] = conj(X[N-k]) for k = complex_len..N-1
    // Then apply inverse FFT (sign=+1, normalize), take real part.
    // The complex input is interleaved (re,im); reinterpret as the scalar type.

    if (input.dtype() == DType::Complex64) {
        const float* d_in = static_cast<const float*>(input.data_ptr());

        if (use_cooley_tukey) {
            // Allocate full N-point interleaved complex buffer per batch
            SyclDevicePtr<float> d_buf_owner(2 * output_len * batch_size, queue);
            float* d_buf = d_buf_owner.get();

            // Reconstruct full spectrum on device
            queue.parallel_for(sycl::range<1>(batch_size * output_len), [=](sycl::id<1> idx) {
                int64_t flat = idx[0];
                int64_t b = flat / output_len;
                int64_t k = flat % output_len;
                int64_t dst = b * 2 * output_len + 2 * k;

                if (k < complex_len) {
                    // Direct copy from input
                    int64_t src = (b * complex_len + k) * 2;
                    d_buf[dst]     = d_in[src];
                    d_buf[dst + 1] = d_in[src + 1];
                } else {
                    // Conjugate symmetry: X[k] = conj(X[N-k]). When an explicit
                    // output length n >= 2*complex_len was requested, the mirror
                    // bin lands beyond the provided complex_len bins (zero-pad
                    // region); read only when mirror is in range, else write 0.
                    int64_t mirror = output_len - k;
                    if (mirror < complex_len) {
                        int64_t src = (b * complex_len + mirror) * 2;
                        d_buf[dst]     = d_in[src];
                        d_buf[dst + 1] = -d_in[src + 1];
                    } else {
                        d_buf[dst]     = 0.0f;
                        d_buf[dst + 1] = 0.0f;
                    }
                }
            }).wait();

            // Apply inverse FFT (sign=+1)
            cooley_tukey_fft_sycl(d_buf, output_len, batch_size,
                                  static_cast<int64_t>(2 * output_len), 1.0f, queue);

            // Normalize and extract real part
            float scale = 1.0f / static_cast<float>(output_len);
            if (norm == "ortho") {
                scale = 1.0f / std::sqrt(static_cast<float>(output_len));
            } else if (norm == "forward") {
                scale = 1.0f;
            }

            Tensor output(out_shape, DType::Float32, input.device());
            float* d_out = static_cast<float*>(const_cast<void*>(output.data_ptr()));
            queue.parallel_for(sycl::range<1>(batch_size * output_len), [=](sycl::id<1> idx) {
                int64_t flat = idx[0];
                int64_t b = flat / output_len;
                int64_t t = flat % output_len;
                d_out[b * output_len + t] = d_buf[b * 2 * output_len + 2 * t] * scale;
            }).wait();
            return output;
        } else {
            // Bluestein path: reconstruct full spectrum with inner_size support
            int64_t full_complex_numel = batch_size * output_len * inner_size * 2;
            SyclDevicePtr<float> d_full_owner(full_complex_numel, queue);
            float* d_full = d_full_owner.get();

            // Reconstruct full N-point spectrum on device
            int64_t total_entries = batch_size * output_len * inner_size;
            queue.parallel_for(sycl::range<1>(total_entries), [=](sycl::id<1> idx) {
                int64_t flat = idx[0];
                int64_t b = flat / (output_len * inner_size);
                int64_t rem = flat % (output_len * inner_size);
                int64_t k = rem / inner_size;
                int64_t inner = rem % inner_size;
                int64_t dst = (b * output_len * inner_size + k * inner_size + inner) * 2;

                if (k < complex_len) {
                    int64_t src = (b * complex_len * inner_size + k * inner_size + inner) * 2;
                    d_full[dst]     = d_in[src];
                    d_full[dst + 1] = d_in[src + 1];
                } else {
                    // Conjugate symmetry; zero-pad bins beyond complex_len when
                    // an explicit n >= 2*complex_len was requested.
                    int64_t mirror = output_len - k;
                    if (mirror < complex_len) {
                        int64_t src = (b * complex_len * inner_size + mirror * inner_size + inner) * 2;
                        d_full[dst]     = d_in[src];
                        d_full[dst + 1] = -d_in[src + 1];
                    } else {
                        d_full[dst]     = 0.0f;
                        d_full[dst + 1] = 0.0f;
                    }
                }
            }).wait();

            // Apply inverse Bluestein FFT (sign=+1)
            SyclDevicePtr<float> d_ifft_owner(full_complex_numel, queue);
            float* d_ifft = d_ifft_owner.get();
            queue.memset(d_ifft, 0, full_complex_numel * sizeof(float)).wait();
            bluestein_fft_complex_sycl(d_full, d_ifft, output_len, batch_size, inner_size, 1.0f, queue);

            // Normalize and extract real part
            float scale = 1.0f / static_cast<float>(output_len);
            if (norm == "ortho") {
                scale = 1.0f / std::sqrt(static_cast<float>(output_len));
            } else if (norm == "forward") {
                scale = 1.0f;
            }

            Tensor output(out_shape, DType::Float32, input.device());
            float* d_out = static_cast<float*>(const_cast<void*>(output.data_ptr()));
            queue.parallel_for(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
                int64_t flat = idx[0];
                int64_t b = flat / (output_len * inner_size);
                int64_t rem = flat % (output_len * inner_size);
                int64_t t = rem / inner_size;
                int64_t inner = rem % inner_size;
                int64_t src = (b * output_len * inner_size + t * inner_size + inner) * 2;
                d_out[flat] = d_ifft[src] * scale;
            }).wait();
            return output;
        }
    } else if (input.dtype() == DType::Complex128) {
        const double* d_in = static_cast<const double*>(input.data_ptr());

        if (use_cooley_tukey) {
            SyclDevicePtr<double> d_buf_owner(2 * output_len * batch_size, queue);
            double* d_buf = d_buf_owner.get();

            queue.parallel_for(sycl::range<1>(batch_size * output_len), [=](sycl::id<1> idx) {
                int64_t flat = idx[0];
                int64_t b = flat / output_len;
                int64_t k = flat % output_len;
                int64_t dst = b * 2 * output_len + 2 * k;

                if (k < complex_len) {
                    int64_t src = (b * complex_len + k) * 2;
                    d_buf[dst]     = d_in[src];
                    d_buf[dst + 1] = d_in[src + 1];
                } else {
                    // Conjugate symmetry; zero-pad bins beyond complex_len when
                    // an explicit n >= 2*complex_len was requested.
                    int64_t mirror = output_len - k;
                    if (mirror < complex_len) {
                        int64_t src = (b * complex_len + mirror) * 2;
                        d_buf[dst]     = d_in[src];
                        d_buf[dst + 1] = -d_in[src + 1];
                    } else {
                        d_buf[dst]     = 0.0;
                        d_buf[dst + 1] = 0.0;
                    }
                }
            }).wait();

            cooley_tukey_fft_sycl(d_buf, output_len, batch_size,
                                  static_cast<int64_t>(2 * output_len), 1.0, queue);

            double scale = 1.0 / static_cast<double>(output_len);
            if (norm == "ortho") {
                scale = 1.0 / std::sqrt(static_cast<double>(output_len));
            } else if (norm == "forward") {
                scale = 1.0;
            }

            Tensor output(out_shape, DType::Float64, input.device());
            double* d_out = static_cast<double*>(const_cast<void*>(output.data_ptr()));
            queue.parallel_for(sycl::range<1>(batch_size * output_len), [=](sycl::id<1> idx) {
                int64_t flat = idx[0];
                int64_t b = flat / output_len;
                int64_t t = flat % output_len;
                d_out[b * output_len + t] = d_buf[b * 2 * output_len + 2 * t] * scale;
            }).wait();
            return output;
        } else {
            int64_t full_complex_numel = batch_size * output_len * inner_size * 2;
            SyclDevicePtr<double> d_full_owner(full_complex_numel, queue);
            double* d_full = d_full_owner.get();

            int64_t total_entries = batch_size * output_len * inner_size;
            queue.parallel_for(sycl::range<1>(total_entries), [=](sycl::id<1> idx) {
                int64_t flat = idx[0];
                int64_t b = flat / (output_len * inner_size);
                int64_t rem = flat % (output_len * inner_size);
                int64_t k = rem / inner_size;
                int64_t inner = rem % inner_size;
                int64_t dst = (b * output_len * inner_size + k * inner_size + inner) * 2;

                if (k < complex_len) {
                    int64_t src = (b * complex_len * inner_size + k * inner_size + inner) * 2;
                    d_full[dst]     = d_in[src];
                    d_full[dst + 1] = d_in[src + 1];
                } else {
                    // Conjugate symmetry; zero-pad bins beyond complex_len when
                    // an explicit n >= 2*complex_len was requested.
                    int64_t mirror = output_len - k;
                    if (mirror < complex_len) {
                        int64_t src = (b * complex_len * inner_size + mirror * inner_size + inner) * 2;
                        d_full[dst]     = d_in[src];
                        d_full[dst + 1] = -d_in[src + 1];
                    } else {
                        d_full[dst]     = 0.0;
                        d_full[dst + 1] = 0.0;
                    }
                }
            }).wait();

            SyclDevicePtr<double> d_ifft_owner(full_complex_numel, queue);
            double* d_ifft = d_ifft_owner.get();
            queue.memset(d_ifft, 0, full_complex_numel * sizeof(double)).wait();
            bluestein_fft_complex_sycl(d_full, d_ifft, output_len, batch_size, inner_size, 1.0, queue);

            double scale = 1.0 / static_cast<double>(output_len);
            if (norm == "ortho") {
                scale = 1.0 / std::sqrt(static_cast<double>(output_len));
            } else if (norm == "forward") {
                scale = 1.0;
            }

            Tensor output(out_shape, DType::Float64, input.device());
            double* d_out = static_cast<double*>(const_cast<void*>(output.data_ptr()));
            queue.parallel_for(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
                int64_t flat = idx[0];
                int64_t b = flat / (output_len * inner_size);
                int64_t rem = flat % (output_len * inner_size);
                int64_t t = rem / inner_size;
                int64_t inner = rem % inner_size;
                int64_t src = (b * output_len * inner_size + t * inner_size + inner) * 2;
                d_out[flat] = d_ifft[src] * scale;
            }).wait();
            return output;
        }
    } else {
        throw std::runtime_error(
            "irfft_kernel: unsupported dtype (expected Complex64 or Complex128)");
    }
}

// ============================================================================
// FFT2 - 2D FFT (naive fallback)
// ============================================================================
auto fft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                 const std::vector<int64_t>& signal_lengths,
                 const std::string& norm, sycl::queue& queue) -> Tensor {
    Tensor result = fft_kernel(input, dims[0], signal_lengths[0], norm, queue);
    result = fft_kernel(result, dims[1], signal_lengths[1], norm, queue);
    return result;
}

// ============================================================================
// IFFT2 - 2D Inverse FFT (naive fallback)
// ============================================================================
auto ifft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                  const std::vector<int64_t>& signal_lengths,
                  const std::string& norm, sycl::queue& queue) -> Tensor {
    Tensor result = ifft_kernel(input, dims[0], signal_lengths[0], norm, queue);
    result = ifft_kernel(result, dims[1], signal_lengths[1], norm, queue);
    return result;
}

// ============================================================================
// FFTN - N-dimensional FFT (naive fallback)
// ============================================================================
auto fftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                 const std::vector<int64_t>& signal_lengths,
                 const std::string& norm, sycl::queue& queue) -> Tensor {
    Tensor result = clone_kernel(input, queue);
    for (size_t i = 0; i < dims.size(); ++i) {
        result = fft_kernel(result, dims[i], signal_lengths[i], norm, queue);
    }
    return result;
}

// ============================================================================
// IFFTN - N-dimensional Inverse FFT (naive fallback)
// ============================================================================
auto ifftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                  const std::vector<int64_t>& signal_lengths,
                  const std::string& norm, sycl::queue& queue) -> Tensor {
    Tensor result = clone_kernel(input, queue);
    for (size_t i = 0; i < dims.size(); ++i) {
        result = ifft_kernel(result, dims[i], signal_lengths[i], norm, queue);
    }
    return result;
}

#endif // TENZOR_HAS_ONEMKL

} // namespace oneapi
} // namespace tenzor
