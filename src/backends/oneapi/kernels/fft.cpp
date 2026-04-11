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
#include <sycl/sycl.hpp>
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

template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

auto clone_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;

// BFloat16 conversion helpers (BFloat16 stored as uint16_t)
inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}
inline uint16_t f32_to_bf16(float f32) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f32, sizeof(uint32_t));
    // Round to nearest even (banker's rounding) for BFloat16
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

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
            dst[i] = static_cast<uint16_t>(bits >> 16);
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
    if (input.dtype() == DType::Complex64) {
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
        const float* in_ptr = reinterpret_cast<const float*>(input.data_ptr());

        if (dim == ndim - 1 && inner_size == 1) {
            // Fast path: FFT along last dim, contiguous complex per batch.
            queue.memcpy(complex_buf, in_ptr, complex_buf_floats * sizeof(float));

            dft::descriptor<dft::precision::SINGLE, dft::domain::COMPLEX> desc(signal_len);
            desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, batch_size);
            std::int64_t fwd_strides[2] = {0, 1};
            desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
            desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);
            desc.set_value(dft::config_param::PLACEMENT, DFTI_INPLACE);
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
            std::int64_t fwd_strides[2] = {0, 1};
            desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
            desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);
            desc.set_value(dft::config_param::PLACEMENT, DFTI_INPLACE);
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
        // with the same shape as input — no trailing 2 dim. The physical
        // storage is still interleaved (re, im) pairs.
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

    } else if (input.dtype() == DType::Complex128) {
        int64_t total_transforms = batch_size * inner_size;
        int64_t complex_buf_doubles = total_transforms * signal_len * 2;

        SyclDevicePtr<double> complex_buf_owner(complex_buf_doubles, queue);
        double* complex_buf = complex_buf_owner.get();
        const double* in_ptr = reinterpret_cast<const double*>(input.data_ptr());

        if (dim == ndim - 1 && inner_size == 1) {
            queue.memcpy(complex_buf, in_ptr, complex_buf_doubles * sizeof(double));

            dft::descriptor<dft::precision::DOUBLE, dft::domain::COMPLEX> desc(signal_len);
            desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, batch_size);
            std::int64_t fwd_strides[2] = {0, 1};
            desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
            desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);
            desc.set_value(dft::config_param::PLACEMENT, DFTI_INPLACE);
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
            std::int64_t fwd_strides[2] = {0, 1};
            desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
            desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);
            desc.set_value(dft::config_param::PLACEMENT, DFTI_INPLACE);
            desc.commit(queue);

            dft::compute_forward(desc, reinterpret_cast<std::complex<double>*>(complex_buf));
        }

        double norm_factor = get_norm_factor(signal_len, norm, true);
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
        std::int64_t bwd_strides[2] = {0, 1};
        desc.set_value(dft::config_param::BWD_STRIDES, bwd_strides);
        desc.set_value(dft::config_param::BWD_DISTANCE, signal_len);
        desc.set_value(dft::config_param::PLACEMENT, DFTI_INPLACE);
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
        std::int64_t bwd_strides[2] = {0, 1};
        desc.set_value(dft::config_param::BWD_STRIDES, bwd_strides);
        desc.set_value(dft::config_param::BWD_DISTANCE, signal_len);
        desc.set_value(dft::config_param::PLACEMENT, DFTI_INPLACE);
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
auto rfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                 const std::string& norm, sycl::queue& queue) -> Tensor {
    namespace dft = ::oneapi::mkl::dft;

    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    int64_t signal_len = shape[dim];
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
        desc.set_value(dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);

        // Forward (real) strides: contiguous reals
        std::int64_t fwd_strides[2] = {0, 1};
        desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
        desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);

        // Backward (complex) strides: contiguous complex values
        // For R2C out-of-place, the output has out_len complex elements per transform
        std::int64_t bwd_strides[2] = {0, 1};
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
        desc.set_value(dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);

        std::int64_t fwd_strides[2] = {0, 1};
        desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
        desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);

        std::int64_t bwd_strides[2] = {0, 1};
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

        // There is no Complex16/Complex32 dtype, so half/bfloat input
        // promotes to Float32 and we return Complex64 directly — the
        // caller will downcast only if it has a real-valued target.
        (void)orig_dtype;
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

    // Input must be a complex dtype. Back-compat for Float32 + trailing-2
    // layouts is intentionally dropped — rfft_kernel now always returns
    // Complex64/Complex128, and callers that needed the old contract
    // will see a clear error.
    if (input.dtype() != DType::Complex64 && input.dtype() != DType::Complex128) {
        throw std::runtime_error(
            "irfft_kernel: expected Complex64 or Complex128 input");
    }

    int64_t complex_len = shape[dim]; // N/2 + 1
    int64_t output_len = n;           // Full real output length

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

        // Allocate contiguous complex input buffer and real output buffer
        SyclDevicePtr<float> complex_buf_owner(total_transforms * complex_len * 2, queue);
        float* complex_buf = complex_buf_owner.get();
        SyclDevicePtr<float> real_buf_owner(total_transforms * output_len, queue);
        float* real_buf = real_buf_owner.get();

        // Read Complex64 input as interleaved (re, im) floats.
        const float* in_ptr = reinterpret_cast<const float*>(input.data_ptr());

        // Gather complex input into contiguous per-transform layout
        if (dim == ndim - 1 && inner_size == 1) {
            queue.memcpy(complex_buf, in_ptr,
                         total_transforms * complex_len * 2 * sizeof(float));
        } else {
            queue.parallel_for(sycl::range<1>(total_transforms * complex_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / complex_len;
                    int64_t j = idx % complex_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t in_idx = (b * complex_len * inner_size + j * inner_size + inner) * 2;
                    complex_buf[2 * idx] = in_ptr[in_idx];
                    complex_buf[2 * idx + 1] = in_ptr[in_idx + 1];
                });
        }

        // oneMKL C2R via backward transform with domain::REAL
        // The descriptor size is the REAL output length (output_len),
        // and the complex input has output_len/2+1 elements.
        dft::descriptor<dft::precision::SINGLE, dft::domain::REAL> desc(output_len);
        desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, total_transforms);
        desc.set_value(dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);

        // Forward (real) strides — for the real output side
        std::int64_t fwd_strides[2] = {0, 1};
        desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
        desc.set_value(dft::config_param::FWD_DISTANCE, output_len);

        // Backward (complex) strides — for the complex input side
        std::int64_t bwd_strides[2] = {0, 1};
        desc.set_value(dft::config_param::BWD_STRIDES, bwd_strides);
        desc.set_value(dft::config_param::BWD_DISTANCE, complex_len);

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

        SyclDevicePtr<double> complex_buf_owner(total_transforms * complex_len * 2, queue);
        double* complex_buf = complex_buf_owner.get();
        SyclDevicePtr<double> real_buf_owner(total_transforms * output_len, queue);
        double* real_buf = real_buf_owner.get();

        const double* in_ptr = reinterpret_cast<const double*>(input.data_ptr());

        if (dim == ndim - 1 && inner_size == 1) {
            queue.memcpy(complex_buf, in_ptr,
                         total_transforms * complex_len * 2 * sizeof(double));
        } else {
            queue.parallel_for(sycl::range<1>(total_transforms * complex_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / complex_len;
                    int64_t j = idx % complex_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t in_idx = (b * complex_len * inner_size + j * inner_size + inner) * 2;
                    complex_buf[2 * idx] = in_ptr[in_idx];
                    complex_buf[2 * idx + 1] = in_ptr[in_idx + 1];
                });
        }

        dft::descriptor<dft::precision::DOUBLE, dft::domain::REAL> desc(output_len);
        desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, total_transforms);
        desc.set_value(dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);

        std::int64_t fwd_strides[2] = {0, 1};
        desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
        desc.set_value(dft::config_param::FWD_DISTANCE, output_len);

        std::int64_t bwd_strides[2] = {0, 1};
        desc.set_value(dft::config_param::BWD_STRIDES, bwd_strides);
        desc.set_value(dft::config_param::BWD_DISTANCE, complex_len);

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

    constexpr T PI = static_cast<T>(3.14159265358979323846);

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

            T angle = sign * static_cast<T>(2.0) * PI * static_cast<T>(k) / static_cast<T>(stride);
            T w_re = sycl::cos(angle);
            T w_im = sycl::sin(angle);

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
// FFT - 1D Forward FFT (Cooley-Tukey for power-of-2, Bluestein otherwise)
// ============================================================================
auto fft_kernel(const Tensor& input, int64_t dim, int64_t n,
                const std::string& norm, sycl::queue& queue) -> Tensor {
    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    int64_t signal_len = shape[dim];

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) batch_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    // Use on-device Cooley-Tukey when signal_len is power-of-2 and data is contiguous along dim
    bool use_cooley_tukey = is_power_of_2(signal_len) && inner_size == 1;

    if (input.dtype() == DType::Float32) {
        int64_t numel = input.numel();
        std::vector<int64_t> out_shape = shape;
        out_shape.push_back(2);
        int64_t out_numel = 1;
        for (auto s : out_shape) out_numel *= s;

        if (use_cooley_tukey) {
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

            if (norm == "ortho" || norm == "forward") {
                float scale = (norm == "ortho")
                    ? 1.0f / std::sqrt(static_cast<float>(signal_len))
                    : 1.0f / static_cast<float>(signal_len);
                queue.parallel_for(sycl::range<1>(2 * total), [=](sycl::id<1> idx) {
                    d_buf[idx[0]] *= scale;
                }).wait();
            }

            Tensor output(out_shape, DType::Float32, input.device());
            queue.memcpy(const_cast<void*>(output.data_ptr()), d_buf, out_numel * sizeof(float)).wait();
            return output;
        } else {
            // Bluestein FFT on device for non-power-of-2 sizes
            const float* d_in = static_cast<const float*>(input.data_ptr());
            SyclDevicePtr<float> d_out_owner(out_numel, queue);
            float* d_out = d_out_owner.get();
            queue.memset(d_out, 0, out_numel * sizeof(float)).wait();
            bluestein_fft_sycl(d_in, d_out, signal_len, batch_size, inner_size, queue);

            if (norm == "ortho" || norm == "forward") {
                float scale = (norm == "ortho")
                    ? 1.0f / std::sqrt(static_cast<float>(signal_len))
                    : 1.0f / static_cast<float>(signal_len);
                queue.parallel_for(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
                    d_out[idx[0]] *= scale;
                }).wait();
            }

            Tensor output(out_shape, DType::Float32, input.device());
            queue.memcpy(const_cast<void*>(output.data_ptr()), d_out, out_numel * sizeof(float)).wait();
            return output;
        }
    } else if (input.dtype() == DType::Float64) {
        int64_t numel = input.numel();
        std::vector<int64_t> out_shape = shape;
        out_shape.push_back(2);
        int64_t out_numel = 1;
        for (auto s : out_shape) out_numel *= s;

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

            Tensor output(out_shape, DType::Float64, input.device());
            queue.memcpy(const_cast<void*>(output.data_ptr()), d_buf, out_numel * sizeof(double)).wait();
            return output;
        } else {
            // Bluestein FFT on device for non-power-of-2 sizes
            const double* d_in = static_cast<const double*>(input.data_ptr());
            SyclDevicePtr<double> d_out_owner(out_numel, queue);
            double* d_out = d_out_owner.get();
            queue.memset(d_out, 0, out_numel * sizeof(double)).wait();
            bluestein_fft_sycl(d_in, d_out, signal_len, batch_size, inner_size, queue);

            if (norm == "ortho" || norm == "forward") {
                double scale = (norm == "ortho")
                    ? 1.0 / std::sqrt(static_cast<double>(signal_len))
                    : 1.0 / static_cast<double>(signal_len);
                queue.parallel_for(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
                    d_out[idx[0]] *= scale;
                }).wait();
            }

            Tensor output(out_shape, DType::Float64, input.device());
            queue.memcpy(const_cast<void*>(output.data_ptr()), d_out, out_numel * sizeof(double)).wait();
            return output;
        }
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        int64_t in_numel = input.numel();

        Tensor f32_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         DType::Float32, input.device());
        device_upcast_to_f32(input.data_ptr(),
                             static_cast<float*>(const_cast<void*>(f32_input.data_ptr())),
                             in_numel, orig_dtype, queue);
        Tensor f32_result = fft_kernel(f32_input, dim, n, norm, queue);

        int64_t out_numel = f32_result.numel();
        Tensor output(std::vector<int64_t>(f32_result.shape().begin(), f32_result.shape().end()),
                      orig_dtype, input.device());
        device_downcast_from_f32(static_cast<const float*>(f32_result.data_ptr()),
                                 const_cast<void*>(output.data_ptr()),
                                 out_numel, orig_dtype, queue);
        return output;
    } else {
        throw std::runtime_error("fft_kernel: unsupported dtype (expected Float32 or Float64)");
    }
}

// ============================================================================
// IFFT - 1D Complex-to-Complex Inverse FFT (device-side Cooley-Tukey / Bluestein)
// ============================================================================
auto ifft_kernel(const Tensor& input, int64_t dim, int64_t n,
                 const std::string& norm, sycl::queue& queue) -> Tensor {
    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    if (shape[ndim - 1] != 2) {
        throw std::runtime_error("ifft_kernel: expected complex input (last dim = 2)");
    }

    int64_t signal_len = shape[dim];
    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) batch_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim - 1; ++i) inner_size *= shape[i];

    bool use_cooley_tukey = is_power_of_2(signal_len) && inner_size == 1;

    if (input.dtype() == DType::Float32) {
        int64_t numel = input.numel();
        std::vector<int64_t> out_shape = shape;
        int64_t out_numel = numel;

        if (use_cooley_tukey) {
            // Input is [batch, signal_len, 2] — already interleaved complex
            // Copy to working buffer and apply Cooley-Tukey with sign=+1
            SyclDevicePtr<float> d_buf_owner(2 * signal_len * batch_size, queue);
            float* d_buf = d_buf_owner.get();
            const float* d_in = static_cast<const float*>(input.data_ptr());
            queue.memcpy(d_buf, d_in, 2 * signal_len * batch_size * sizeof(float)).wait();

            cooley_tukey_fft_sycl(d_buf, signal_len, batch_size,
                                  static_cast<int64_t>(2 * signal_len), 1.0f, queue);

            // Apply normalization: backward=1/N (default), ortho=1/sqrt(N), forward=1
            float scale = 1.0f / static_cast<float>(signal_len);
            if (norm == "ortho") {
                scale = 1.0f / std::sqrt(static_cast<float>(signal_len));
            } else if (norm == "forward") {
                scale = 1.0f;
            }
            int64_t total_elems = 2 * signal_len * batch_size;
            queue.parallel_for(sycl::range<1>(total_elems), [=](sycl::id<1> idx) {
                d_buf[idx[0]] *= scale;
            }).wait();

            Tensor output(out_shape, DType::Float32, input.device());
            queue.memcpy(const_cast<void*>(output.data_ptr()), d_buf, out_numel * sizeof(float)).wait();
            return output;
        } else {
            // Bluestein IFFT for non-power-of-2 (complex input, sign=+1)
            const float* d_in = static_cast<const float*>(input.data_ptr());
            SyclDevicePtr<float> d_out_owner(out_numel, queue);
            float* d_out = d_out_owner.get();
            queue.memset(d_out, 0, out_numel * sizeof(float)).wait();
            bluestein_fft_complex_sycl(d_in, d_out, signal_len, batch_size, inner_size, 1.0f, queue);

            // Apply normalization
            float scale = 1.0f / static_cast<float>(signal_len);
            if (norm == "ortho") {
                scale = 1.0f / std::sqrt(static_cast<float>(signal_len));
            } else if (norm == "forward") {
                scale = 1.0f;
            }
            queue.parallel_for(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
                d_out[idx[0]] *= scale;
            }).wait();

            Tensor output(out_shape, DType::Float32, input.device());
            queue.memcpy(const_cast<void*>(output.data_ptr()), d_out, out_numel * sizeof(float)).wait();
            return output;
        }
    } else if (input.dtype() == DType::Float64) {
        int64_t numel = input.numel();
        std::vector<int64_t> out_shape = shape;
        int64_t out_numel = numel;

        if (use_cooley_tukey) {
            SyclDevicePtr<double> d_buf_owner(2 * signal_len * batch_size, queue);
            double* d_buf = d_buf_owner.get();
            const double* d_in = static_cast<const double*>(input.data_ptr());
            queue.memcpy(d_buf, d_in, 2 * signal_len * batch_size * sizeof(double)).wait();

            cooley_tukey_fft_sycl(d_buf, signal_len, batch_size,
                                  static_cast<int64_t>(2 * signal_len), 1.0, queue);

            double scale = 1.0 / static_cast<double>(signal_len);
            if (norm == "ortho") {
                scale = 1.0 / std::sqrt(static_cast<double>(signal_len));
            } else if (norm == "forward") {
                scale = 1.0;
            }
            int64_t total_elems = 2 * signal_len * batch_size;
            queue.parallel_for(sycl::range<1>(total_elems), [=](sycl::id<1> idx) {
                d_buf[idx[0]] *= scale;
            }).wait();

            Tensor output(out_shape, DType::Float64, input.device());
            queue.memcpy(const_cast<void*>(output.data_ptr()), d_buf, out_numel * sizeof(double)).wait();
            return output;
        } else {
            const double* d_in = static_cast<const double*>(input.data_ptr());
            SyclDevicePtr<double> d_out_owner(out_numel, queue);
            double* d_out = d_out_owner.get();
            queue.memset(d_out, 0, out_numel * sizeof(double)).wait();
            bluestein_fft_complex_sycl(d_in, d_out, signal_len, batch_size, inner_size, 1.0, queue);

            double scale = 1.0 / static_cast<double>(signal_len);
            if (norm == "ortho") {
                scale = 1.0 / std::sqrt(static_cast<double>(signal_len));
            } else if (norm == "forward") {
                scale = 1.0;
            }
            queue.parallel_for(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
                d_out[idx[0]] *= scale;
            }).wait();

            Tensor output(out_shape, DType::Float64, input.device());
            queue.memcpy(const_cast<void*>(output.data_ptr()), d_out, out_numel * sizeof(double)).wait();
            return output;
        }
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        int64_t in_numel = input.numel();

        Tensor f32_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         DType::Float32, input.device());
        device_upcast_to_f32(input.data_ptr(),
                             static_cast<float*>(const_cast<void*>(f32_input.data_ptr())),
                             in_numel, orig_dtype, queue);
        Tensor f32_result = ifft_kernel(f32_input, dim, n, norm, queue);

        int64_t out_numel = f32_result.numel();
        Tensor output(std::vector<int64_t>(f32_result.shape().begin(), f32_result.shape().end()),
                      orig_dtype, input.device());
        device_downcast_from_f32(static_cast<const float*>(f32_result.data_ptr()),
                                 const_cast<void*>(output.data_ptr()),
                                 out_numel, orig_dtype, queue);
        return output;
    } else {
        throw std::runtime_error("ifft_kernel: unsupported dtype (expected Float32 or Float64)");
    }
}

// ============================================================================
// RFFT - 1D Real-to-Complex Forward FFT (device-side Cooley-Tukey / Bluestein)
// ============================================================================
auto rfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                 const std::string& norm, sycl::queue& queue) -> Tensor {
    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    int64_t signal_len = shape[dim];
    int64_t out_len = signal_len / 2 + 1;

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) batch_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    bool use_cooley_tukey = is_power_of_2(signal_len) && inner_size == 1;

    // Strategy: compute full N-point forward FFT on device, then truncate to N/2+1 bins
    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = out_len;
    out_shape.push_back(2);
    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

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
            Tensor output(out_shape, DType::Float32, input.device());
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
            Tensor output(out_shape, DType::Float32, input.device());
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

            Tensor output(out_shape, DType::Float64, input.device());
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

            Tensor output(out_shape, DType::Float64, input.device());
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
        DType orig_dtype = input.dtype();
        int64_t in_numel = input.numel();

        Tensor f32_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         DType::Float32, input.device());
        device_upcast_to_f32(input.data_ptr(),
                             static_cast<float*>(const_cast<void*>(f32_input.data_ptr())),
                             in_numel, orig_dtype, queue);
        Tensor f32_result = rfft_kernel(f32_input, dim, n, norm, queue);

        int64_t out_numel_f16 = f32_result.numel();
        Tensor output(std::vector<int64_t>(f32_result.shape().begin(), f32_result.shape().end()),
                      orig_dtype, input.device());
        device_downcast_from_f32(static_cast<const float*>(f32_result.data_ptr()),
                                 const_cast<void*>(output.data_ptr()),
                                 out_numel_f16, orig_dtype, queue);
        return output;
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

    if (shape[ndim - 1] != 2) {
        throw std::runtime_error("irfft_kernel: expected complex input (last dim = 2)");
    }

    int64_t complex_len = shape[dim];  // N/2+1 input bins
    int64_t output_len = n;            // N output points

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) batch_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim - 1; ++i) inner_size *= shape[i];

    bool use_cooley_tukey = is_power_of_2(output_len) && inner_size == 1;

    // Build real output shape (no trailing 2 dim)
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < dim; ++i) out_shape.push_back(shape[i]);
    out_shape.push_back(output_len);
    for (int64_t i = dim + 1; i < ndim - 1; ++i) out_shape.push_back(shape[i]);
    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

    // Strategy: reconstruct full N-point complex spectrum from N/2+1 bins
    // using conjugate symmetry: X[k] = conj(X[N-k]) for k = complex_len..N-1
    // Then apply inverse FFT (sign=+1, normalize), take real part.

    if (input.dtype() == DType::Float32) {
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
                    // Conjugate symmetry: X[k] = conj(X[N-k])
                    int64_t mirror = output_len - k;
                    int64_t src = (b * complex_len + mirror) * 2;
                    d_buf[dst]     = d_in[src];
                    d_buf[dst + 1] = -d_in[src + 1];
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
                    int64_t mirror = output_len - k;
                    int64_t src = (b * complex_len * inner_size + mirror * inner_size + inner) * 2;
                    d_full[dst]     = d_in[src];
                    d_full[dst + 1] = -d_in[src + 1];
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
    } else if (input.dtype() == DType::Float64) {
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
                    int64_t mirror = output_len - k;
                    int64_t src = (b * complex_len + mirror) * 2;
                    d_buf[dst]     = d_in[src];
                    d_buf[dst + 1] = -d_in[src + 1];
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
                    int64_t mirror = output_len - k;
                    int64_t src = (b * complex_len * inner_size + mirror * inner_size + inner) * 2;
                    d_full[dst]     = d_in[src];
                    d_full[dst + 1] = -d_in[src + 1];
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
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        int64_t numel = input.numel();

        Tensor f32_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         DType::Float32, input.device());
        device_upcast_to_f32(input.data_ptr(),
                             static_cast<float*>(const_cast<void*>(f32_input.data_ptr())),
                             numel, orig_dtype, queue);
        Tensor f32_result = irfft_kernel(f32_input, dim, n, norm, queue);

        int64_t out_numel_f16 = f32_result.numel();
        Tensor output(std::vector<int64_t>(f32_result.shape().begin(), f32_result.shape().end()),
                      orig_dtype, input.device());
        device_downcast_from_f32(static_cast<const float*>(f32_result.data_ptr()),
                                 const_cast<void*>(output.data_ptr()),
                                 out_numel_f16, orig_dtype, queue);
        return output;
    } else {
        throw std::runtime_error("irfft_kernel: unsupported dtype (expected Float32 or Float64)");
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
