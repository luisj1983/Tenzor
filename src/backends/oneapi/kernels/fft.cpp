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
    return static_cast<uint16_t>(bits >> 16);
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

    // Output shape: insert trailing dim=2 for complex
    std::vector<int64_t> out_shape = shape;
    out_shape.push_back(2);
    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

    if (input.dtype() == DType::Float32) {
        // Build interleaved complex buffer: (batch_size * signal_len * inner_size) complex values
        // stored as 2*N floats. We need contiguous layout along the FFT dimension for oneMKL.
        //
        // For the general strided case (dim != last), we process per-batch-per-inner
        // with NUMBER_OF_TRANSFORMS = 1 and manual offset computation.
        // For dim == last (most common), we can batch efficiently.

        int64_t total_transforms = batch_size * inner_size;
        int64_t complex_buf_floats = total_transforms * signal_len * 2;

        // Allocate device buffer for interleaved complex data
        float* complex_buf = sycl::malloc_device<float>(complex_buf_floats, queue);

        // Copy real input into complex buffer (real parts), zero imaginary parts
        const float* in_ptr = get_data_ptr<const float>(input);

        if (dim == ndim - 1 && inner_size == 1) {
            // Fast path: FFT along last dimension, data is contiguous per batch
            // Interleave: complex_buf[2*i] = in_ptr[i], complex_buf[2*i+1] = 0
            int64_t total_reals = batch_size * signal_len;
            queue.parallel_for(sycl::range<1>(total_reals), [=](sycl::id<1> idx) {
                complex_buf[2 * idx] = in_ptr[idx];
                complex_buf[2 * idx + 1] = 0.0f;
            });
            // No wait needed — in-order queue guarantees DFT sees completed gather

            // Create oneMKL DFT descriptor for batched 1D C2C
            dft::descriptor<dft::precision::SINGLE, dft::domain::COMPLEX> desc(signal_len);
            desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, batch_size);
            // Input/output strides: oneMKL uses {offset, stride} where offset=0
            // For interleaved complex, the "distance" between transforms is signal_len complex elements
            std::int64_t fwd_strides[2] = {0, 1};
            desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
            desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);
            desc.set_value(dft::config_param::PLACEMENT, DFTI_INPLACE);
            desc.commit(queue);

            dft::compute_forward(desc, reinterpret_cast<std::complex<float>*>(complex_buf));
            // No wait needed — in-order queue guarantees subsequent ops see completed DFT
        } else {
            // General path: gather strided real data into contiguous complex buffer
            // Layout: transform t = b * inner_size + inner
            // Input index for element j of transform t: b * signal_len * inner_size + j * inner_size + inner
            queue.parallel_for(sycl::range<1>(total_transforms * signal_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / signal_len;
                    int64_t j = idx % signal_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    int64_t in_idx = b * signal_len * inner_size + j * inner_size + inner;
                    complex_buf[2 * idx] = in_ptr[in_idx];
                    complex_buf[2 * idx + 1] = 0.0f;
                });
            // No wait needed — in-order queue guarantees DFT sees completed gather

            // Each transform is contiguous signal_len complex elements
            dft::descriptor<dft::precision::SINGLE, dft::domain::COMPLEX> desc(signal_len);
            desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, total_transforms);
            std::int64_t fwd_strides[2] = {0, 1};
            desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
            desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);
            desc.set_value(dft::config_param::PLACEMENT, DFTI_INPLACE);
            desc.commit(queue);

            dft::compute_forward(desc, reinterpret_cast<std::complex<float>*>(complex_buf));
            // No wait needed — in-order queue guarantees subsequent ops see completed DFT
        }

        // Apply normalization
        double norm_factor = get_norm_factor(signal_len, norm, true);
        if (norm_factor != 1.0) {
            apply_scale_f32(queue, complex_buf, complex_buf_floats,
                            static_cast<float>(norm_factor));
        }

        // Scatter complex results back to output tensor layout
        // Output shape: (..., signal_len, ..., 2)
        // out_idx for transform t, frequency k: (b * signal_len * inner_size + k * inner_size + inner) * 2
        Tensor output(out_shape, DType::Float32, input.device());
        float* out_ptr = get_data_ptr<float>(output);

        if (dim == ndim - 1 && inner_size == 1) {
            // Direct copy — complex_buf layout matches output layout
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
                    out_ptr[out_idx] = complex_buf[2 * idx];
                    out_ptr[out_idx + 1] = complex_buf[2 * idx + 1];
                }).wait();
        }

        sycl::free(complex_buf, queue);
        return output;

    } else if (input.dtype() == DType::Float64) {
        int64_t total_transforms = batch_size * inner_size;
        int64_t complex_buf_doubles = total_transforms * signal_len * 2;

        double* complex_buf = sycl::malloc_device<double>(complex_buf_doubles, queue);
        const double* in_ptr = get_data_ptr<const double>(input);

        if (dim == ndim - 1 && inner_size == 1) {
            int64_t total_reals = batch_size * signal_len;
            queue.parallel_for(sycl::range<1>(total_reals), [=](sycl::id<1> idx) {
                complex_buf[2 * idx] = in_ptr[idx];
                complex_buf[2 * idx + 1] = 0.0;
            });

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
                    int64_t in_idx = b * signal_len * inner_size + j * inner_size + inner;
                    complex_buf[2 * idx] = in_ptr[in_idx];
                    complex_buf[2 * idx + 1] = 0.0;
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

        Tensor output(out_shape, DType::Float64, input.device());
        double* out_ptr = get_data_ptr<double>(output);

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
                    out_ptr[out_idx] = complex_buf[2 * idx];
                    out_ptr[out_idx + 1] = complex_buf[2 * idx + 1];
                }).wait();
        }

        sycl::free(complex_buf, queue);
        return output;

    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        // Upcast to Float32, compute FFT, downcast result
        DType orig_dtype = input.dtype();
        int64_t numel = input.numel();

        // Copy input to host and convert to float32
        std::vector<float> host_f32(numel);
        if (orig_dtype == DType::Float16) {
            std::vector<sycl::half> host_in(numel);
            queue.memcpy(host_in.data(), input.data_ptr(), numel * sizeof(sycl::half)).wait();
            for (int64_t i = 0; i < numel; ++i) host_f32[i] = static_cast<float>(host_in[i]);
        } else {
            std::vector<uint16_t> host_in(numel);
            queue.memcpy(host_in.data(), input.data_ptr(), numel * sizeof(uint16_t)).wait();
            for (int64_t i = 0; i < numel; ++i) host_f32[i] = bf16_to_f32(host_in[i]);
        }

        // Create Float32 tensor, compute FFT
        Tensor f32_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         DType::Float32, input.device());
        queue.memcpy(const_cast<void*>(f32_input.data_ptr()), host_f32.data(), numel * sizeof(float)).wait();

        Tensor f32_result = fft_kernel(f32_input, dim, n, norm, queue);

        // Downcast result
        int64_t out_numel = f32_result.numel();
        Tensor output(std::vector<int64_t>(f32_result.shape().begin(), f32_result.shape().end()),
                      orig_dtype, input.device());

        std::vector<float> host_out(out_numel);
        queue.memcpy(host_out.data(), f32_result.data_ptr(), out_numel * sizeof(float)).wait();

        if (orig_dtype == DType::Float16) {
            std::vector<sycl::half> host_half(out_numel);
            for (int64_t i = 0; i < out_numel; ++i) host_half[i] = sycl::half(host_out[i]);
            queue.memcpy(const_cast<void*>(output.data_ptr()), host_half.data(), out_numel * sizeof(sycl::half)).wait();
        } else {
            std::vector<uint16_t> host_bf16(out_numel);
            for (int64_t i = 0; i < out_numel; ++i) host_bf16[i] = f32_to_bf16(host_out[i]);
            queue.memcpy(const_cast<void*>(output.data_ptr()), host_bf16.data(), out_numel * sizeof(uint16_t)).wait();
        }

        return output;

    } else {
        throw std::runtime_error("fft_kernel: unsupported dtype (expected Float32 or Float64)");
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

    // Expect complex input: last dim = 2
    if (shape[ndim - 1] != 2) {
        throw std::runtime_error("ifft_kernel: expected complex input (last dim = 2)");
    }

    int64_t signal_len = shape[dim];
    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) batch_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim - 1; ++i) inner_size *= shape[i];

    // Output has same shape as input (complex)
    std::vector<int64_t> out_shape = shape;
    int64_t out_numel = input.numel();

    if (input.dtype() == DType::Float32) {
        int64_t total_transforms = batch_size * inner_size;
        int64_t complex_buf_floats = total_transforms * signal_len * 2;

        float* complex_buf = sycl::malloc_device<float>(complex_buf_floats, queue);
        const float* in_ptr = get_data_ptr<const float>(input);

        // Gather interleaved complex data into contiguous per-transform layout
        if (dim == ndim - 2 && inner_size == 1) {
            // Fast path: dim is second-to-last, last dim is 2 (complex)
            // Input layout: batch * signal_len * 2 — already contiguous per transform
            queue.memcpy(complex_buf, in_ptr, complex_buf_floats * sizeof(float));
        } else {
            queue.parallel_for(sycl::range<1>(total_transforms * signal_len),
                [=](sycl::id<1> flat_idx) {
                    int64_t idx = flat_idx[0];
                    int64_t t = idx / signal_len;
                    int64_t j = idx % signal_len;
                    int64_t b = t / inner_size;
                    int64_t inner = t % inner_size;
                    // Input index: (b * signal_len * inner_size + j * inner_size + inner) * 2
                    int64_t in_idx = (b * signal_len * inner_size + j * inner_size + inner) * 2;
                    complex_buf[2 * idx] = in_ptr[in_idx];
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

        // Scatter back to output layout
        Tensor output(out_shape, DType::Float32, input.device());
        float* out_ptr = get_data_ptr<float>(output);

        if (dim == ndim - 2 && inner_size == 1) {
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
                    out_ptr[out_idx] = complex_buf[2 * idx];
                    out_ptr[out_idx + 1] = complex_buf[2 * idx + 1];
                }).wait();
        }

        sycl::free(complex_buf, queue);
        return output;

    } else if (input.dtype() == DType::Float64) {
        int64_t total_transforms = batch_size * inner_size;
        int64_t complex_buf_doubles = total_transforms * signal_len * 2;

        double* complex_buf = sycl::malloc_device<double>(complex_buf_doubles, queue);
        const double* in_ptr = get_data_ptr<const double>(input);

        if (dim == ndim - 2 && inner_size == 1) {
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
                    complex_buf[2 * idx] = in_ptr[in_idx];
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

        Tensor output(out_shape, DType::Float64, input.device());
        double* out_ptr = get_data_ptr<double>(output);

        if (dim == ndim - 2 && inner_size == 1) {
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
                    out_ptr[out_idx] = complex_buf[2 * idx];
                    out_ptr[out_idx + 1] = complex_buf[2 * idx + 1];
                }).wait();
        }

        sycl::free(complex_buf, queue);
        return output;

    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        int64_t numel = input.numel();

        std::vector<float> host_f32(numel);
        if (orig_dtype == DType::Float16) {
            std::vector<sycl::half> host_in(numel);
            queue.memcpy(host_in.data(), input.data_ptr(), numel * sizeof(sycl::half)).wait();
            for (int64_t i = 0; i < numel; ++i) host_f32[i] = static_cast<float>(host_in[i]);
        } else {
            std::vector<uint16_t> host_in(numel);
            queue.memcpy(host_in.data(), input.data_ptr(), numel * sizeof(uint16_t)).wait();
            for (int64_t i = 0; i < numel; ++i) host_f32[i] = bf16_to_f32(host_in[i]);
        }

        Tensor f32_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         DType::Float32, input.device());
        queue.memcpy(const_cast<void*>(f32_input.data_ptr()), host_f32.data(), numel * sizeof(float)).wait();

        Tensor f32_result = ifft_kernel(f32_input, dim, n, norm, queue);

        int64_t out_numel = f32_result.numel();
        Tensor output(std::vector<int64_t>(f32_result.shape().begin(), f32_result.shape().end()),
                      orig_dtype, input.device());

        std::vector<float> host_out(out_numel);
        queue.memcpy(host_out.data(), f32_result.data_ptr(), out_numel * sizeof(float)).wait();

        if (orig_dtype == DType::Float16) {
            std::vector<sycl::half> host_half(out_numel);
            for (int64_t i = 0; i < out_numel; ++i) host_half[i] = sycl::half(host_out[i]);
            queue.memcpy(const_cast<void*>(output.data_ptr()), host_half.data(), out_numel * sizeof(sycl::half)).wait();
        } else {
            std::vector<uint16_t> host_bf16(out_numel);
            for (int64_t i = 0; i < out_numel; ++i) host_bf16[i] = f32_to_bf16(host_out[i]);
            queue.memcpy(const_cast<void*>(output.data_ptr()), host_bf16.data(), out_numel * sizeof(uint16_t)).wait();
        }

        return output;

    } else {
        throw std::runtime_error("ifft_kernel: unsupported dtype (expected Float32 or Float64)");
    }
}

// ============================================================================
// RFFT - 1D Real-to-Complex Forward FFT
// ============================================================================
//
// Input: real tensor of shape (..., signal_len, ...)
// Output: complex tensor of shape (..., signal_len/2+1, ..., 2)
//
// oneMKL DFT with domain::REAL performs R2C. The output is N/2+1 complex values.
// oneMKL stores R2C output in CCS (Complex-Conjugate-Symmetric) packed format
// by default. We use DFTI_NOT_INPLACE with separate real input and complex output
// buffers to get standard interleaved complex output.
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

    // Output shape: replace dim with out_len, append trailing 2
    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = out_len;
    out_shape.push_back(2);
    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

    if (input.dtype() == DType::Float32) {
        int64_t total_transforms = batch_size * inner_size;

        // Allocate contiguous real input buffer and complex output buffer
        float* real_buf = sycl::malloc_device<float>(total_transforms * signal_len, queue);
        float* complex_buf = sycl::malloc_device<float>(total_transforms * out_len * 2, queue);

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

        // Scatter complex results to output tensor
        Tensor output(out_shape, DType::Float32, input.device());
        float* out_ptr = get_data_ptr<float>(output);

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

        sycl::free(real_buf, queue);
        sycl::free(complex_buf, queue);
        return output;

    } else if (input.dtype() == DType::Float64) {
        int64_t total_transforms = batch_size * inner_size;

        double* real_buf = sycl::malloc_device<double>(total_transforms * signal_len, queue);
        double* complex_buf = sycl::malloc_device<double>(total_transforms * out_len * 2, queue);

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

        Tensor output(out_shape, DType::Float64, input.device());
        double* out_ptr = get_data_ptr<double>(output);

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

        sycl::free(real_buf, queue);
        sycl::free(complex_buf, queue);
        return output;

    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        int64_t numel = input.numel();

        std::vector<float> host_f32(numel);
        if (orig_dtype == DType::Float16) {
            std::vector<sycl::half> host_in(numel);
            queue.memcpy(host_in.data(), input.data_ptr(), numel * sizeof(sycl::half)).wait();
            for (int64_t i = 0; i < numel; ++i) host_f32[i] = static_cast<float>(host_in[i]);
        } else {
            std::vector<uint16_t> host_in(numel);
            queue.memcpy(host_in.data(), input.data_ptr(), numel * sizeof(uint16_t)).wait();
            for (int64_t i = 0; i < numel; ++i) host_f32[i] = bf16_to_f32(host_in[i]);
        }

        Tensor f32_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         DType::Float32, input.device());
        queue.memcpy(const_cast<void*>(f32_input.data_ptr()), host_f32.data(), numel * sizeof(float)).wait();

        Tensor f32_result = rfft_kernel(f32_input, dim, n, norm, queue);

        int64_t out_numel = f32_result.numel();
        Tensor output(std::vector<int64_t>(f32_result.shape().begin(), f32_result.shape().end()),
                      orig_dtype, input.device());

        std::vector<float> host_out(out_numel);
        queue.memcpy(host_out.data(), f32_result.data_ptr(), out_numel * sizeof(float)).wait();

        if (orig_dtype == DType::Float16) {
            std::vector<sycl::half> host_half(out_numel);
            for (int64_t i = 0; i < out_numel; ++i) host_half[i] = sycl::half(host_out[i]);
            queue.memcpy(const_cast<void*>(output.data_ptr()), host_half.data(), out_numel * sizeof(sycl::half)).wait();
        } else {
            std::vector<uint16_t> host_bf16(out_numel);
            for (int64_t i = 0; i < out_numel; ++i) host_bf16[i] = f32_to_bf16(host_out[i]);
            queue.memcpy(const_cast<void*>(output.data_ptr()), host_bf16.data(), out_numel * sizeof(uint16_t)).wait();
        }

        return output;

    } else {
        throw std::runtime_error("rfft_kernel: unsupported dtype (expected Float32 or Float64)");
    }
}

// ============================================================================
// IRFFT - 1D Complex-to-Real Inverse FFT
// ============================================================================
//
// Input: complex tensor (..., N/2+1, ..., 2) with Float32/Float64
// Output: real tensor (..., n, ...)
//
// oneMKL backward transform with domain::REAL performs C2R.
// ============================================================================
auto irfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                  const std::string& norm, sycl::queue& queue) -> Tensor {
    namespace dft = ::oneapi::mkl::dft;

    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    if (shape[ndim - 1] != 2) {
        throw std::runtime_error("irfft_kernel: expected complex input (last dim = 2)");
    }

    int64_t complex_len = shape[dim]; // N/2 + 1
    int64_t output_len = n;           // Full real output length

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) batch_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim - 1; ++i) inner_size *= shape[i];

    // Output shape: real tensor without trailing 2
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < dim; ++i) out_shape.push_back(shape[i]);
    out_shape.push_back(output_len);
    for (int64_t i = dim + 1; i < ndim - 1; ++i) out_shape.push_back(shape[i]);

    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

    if (input.dtype() == DType::Float32) {
        int64_t total_transforms = batch_size * inner_size;

        // Allocate contiguous complex input buffer and real output buffer
        float* complex_buf = sycl::malloc_device<float>(total_transforms * complex_len * 2, queue);
        float* real_buf = sycl::malloc_device<float>(total_transforms * output_len, queue);

        const float* in_ptr = get_data_ptr<const float>(input);

        // Gather complex input into contiguous per-transform layout
        if (dim == ndim - 2 && inner_size == 1) {
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

        if (dim == ndim - 2 && inner_size == 1) {
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

        sycl::free(complex_buf, queue);
        sycl::free(real_buf, queue);
        return output;

    } else if (input.dtype() == DType::Float64) {
        int64_t total_transforms = batch_size * inner_size;

        double* complex_buf = sycl::malloc_device<double>(total_transforms * complex_len * 2, queue);
        double* real_buf = sycl::malloc_device<double>(total_transforms * output_len, queue);

        const double* in_ptr = get_data_ptr<const double>(input);

        if (dim == ndim - 2 && inner_size == 1) {
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

        if (dim == ndim - 2 && inner_size == 1) {
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

        sycl::free(complex_buf, queue);
        sycl::free(real_buf, queue);
        return output;

    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        int64_t numel = input.numel();

        std::vector<float> host_f32(numel);
        if (orig_dtype == DType::Float16) {
            std::vector<sycl::half> host_in(numel);
            queue.memcpy(host_in.data(), input.data_ptr(), numel * sizeof(sycl::half)).wait();
            for (int64_t i = 0; i < numel; ++i) host_f32[i] = static_cast<float>(host_in[i]);
        } else {
            std::vector<uint16_t> host_in(numel);
            queue.memcpy(host_in.data(), input.data_ptr(), numel * sizeof(uint16_t)).wait();
            for (int64_t i = 0; i < numel; ++i) host_f32[i] = bf16_to_f32(host_in[i]);
        }

        Tensor f32_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         DType::Float32, input.device());
        queue.memcpy(const_cast<void*>(f32_input.data_ptr()), host_f32.data(), numel * sizeof(float)).wait();

        Tensor f32_result = irfft_kernel(f32_input, dim, n, norm, queue);

        int64_t out_numel = f32_result.numel();
        Tensor output(std::vector<int64_t>(f32_result.shape().begin(), f32_result.shape().end()),
                      orig_dtype, input.device());

        std::vector<float> host_out(out_numel);
        queue.memcpy(host_out.data(), f32_result.data_ptr(), out_numel * sizeof(float)).wait();

        if (orig_dtype == DType::Float16) {
            std::vector<sycl::half> host_half(out_numel);
            for (int64_t i = 0; i < out_numel; ++i) host_half[i] = sycl::half(host_out[i]);
            queue.memcpy(const_cast<void*>(output.data_ptr()), host_half.data(), out_numel * sizeof(sycl::half)).wait();
        } else {
            std::vector<uint16_t> host_bf16(out_numel);
            for (int64_t i = 0; i < out_numel; ++i) host_bf16[i] = f32_to_bf16(host_out[i]);
            queue.memcpy(const_cast<void*>(output.data_ptr()), host_bf16.data(), out_numel * sizeof(uint16_t)).wait();
        }

        return output;

    } else {
        throw std::runtime_error("irfft_kernel: unsupported dtype (expected Float32 or Float64)");
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
// Operates on interleaved complex data: [re0, im0, re1, im1, ...] of length 2*N.
// sign = -1.0 for forward FFT, +1.0 for inverse FFT.
template<typename T>
void cooley_tukey_fft_sycl(T* data, int64_t N, T sign, sycl::queue& queue) {
    int log2N = 0;
    { int64_t tmp = N; while (tmp > 1) { tmp >>= 1; log2N++; } }

    // Step 1: Bit-reverse permutation (on device)
    const int bits = log2N;
    queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        uint32_t rev = 0;
        uint32_t x = static_cast<uint32_t>(i);
        for (int b = 0; b < bits; ++b) {
            rev = (rev << 1) | (x & 1);
            x >>= 1;
        }
        int64_t j = static_cast<int64_t>(rev);
        if (i < j) {
            T tmp_re = data[2 * i];
            T tmp_im = data[2 * i + 1];
            data[2 * i] = data[2 * j];
            data[2 * i + 1] = data[2 * j + 1];
            data[2 * j] = tmp_re;
            data[2 * j + 1] = tmp_im;
        }
    }).wait();

    constexpr T PI = static_cast<T>(3.14159265358979323846);

    // Step 2: Butterfly stages
    for (int s = 1; s <= log2N; ++s) {
        int64_t stride = static_cast<int64_t>(1) << s;
        int64_t half = stride / 2;
        int64_t num_butterflies = N / 2;

        queue.parallel_for(sycl::range<1>(num_butterflies), [=](sycl::id<1> idx) {
            int64_t flat = idx[0];
            int64_t group = flat / half;
            int64_t k = flat % half;
            int64_t base_idx = group * stride;

            T angle = sign * static_cast<T>(2.0) * PI * static_cast<T>(k) / static_cast<T>(stride);
            T w_re = sycl::cos(angle);
            T w_im = sycl::sin(angle);

            int64_t even_i = base_idx + k;
            int64_t odd_i = base_idx + k + half;

            T e_re = data[2 * even_i];
            T e_im = data[2 * even_i + 1];
            T o_re = data[2 * odd_i];
            T o_im = data[2 * odd_i + 1];

            T t_re = w_re * o_re - w_im * o_im;
            T t_im = w_re * o_im + w_im * o_re;

            data[2 * even_i] = e_re + t_re;
            data[2 * even_i + 1] = e_im + t_im;
            data[2 * odd_i] = e_re - t_re;
            data[2 * odd_i + 1] = e_im - t_im;
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
    T* chirp   = sycl::malloc_device<T>(2 * N, queue);
    T* b_buf   = sycl::malloc_device<T>(2 * M, queue);
    T* B_buf   = sycl::malloc_device<T>(2 * M, queue);
    T* a_buf   = sycl::malloc_device<T>(2 * M, queue);

    // Step 1: Generate chirp sequence: chirp[k] = exp(-j * pi * k^2 / N)
    queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        int64_t k = idx[0];
        T angle = -PI * static_cast<T>(k) * static_cast<T>(k) / static_cast<T>(N);
        chirp[2 * k]     = sycl::cos(angle);
        chirp[2 * k + 1] = sycl::sin(angle);
    }).wait();

    // Step 2: Build convolution kernel b[k] = conj(chirp[k]) for k=0..N-1,
    //         b[M-k] = conj(chirp[k]) for k=1..N-1, zeros elsewhere
    // First zero the entire buffer
    queue.memset(b_buf, 0, 2 * M * sizeof(T)).wait();
    // b[0..N-1] = conj(chirp[0..N-1])
    queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        int64_t k = idx[0];
        b_buf[2 * k]     = chirp[2 * k];       // real part (conj doesn't change)
        b_buf[2 * k + 1] = -chirp[2 * k + 1];  // negate imaginary
    }).wait();
    // b[M-k] = conj(chirp[k]) for k=1..N-1 (wrap-around for circular convolution)
    if (N > 1) {
        queue.parallel_for(sycl::range<1>(N - 1), [=](sycl::id<1> idx) {
            int64_t k = idx[0] + 1;
            int64_t m_idx = M - k;
            b_buf[2 * m_idx]     = chirp[2 * k];
            b_buf[2 * m_idx + 1] = -chirp[2 * k + 1];
        }).wait();
    }

    // Step 3: Precompute B = FFT(b) — power-of-2 size M
    queue.memcpy(B_buf, b_buf, 2 * M * sizeof(T)).wait();
    cooley_tukey_fft_sycl(B_buf, M, static_cast<T>(-1.0), queue);

    // Steps 4-8: Process each (batch, inner) slice
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            // Step 4: Build a[k] = x[k] * chirp[k] for k=0..N-1, zero-pad to M
            queue.memset(a_buf, 0, 2 * M * sizeof(T)).wait();
            queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                int64_t k = idx[0];
                int64_t in_idx = b * N * inner_size + k * inner_size + inner;
                T val = d_in[in_idx];
                // a[k] = val * chirp[k] (val is real, so: re = val*chirp_re, im = val*chirp_im)
                a_buf[2 * k]     = val * chirp[2 * k];
                a_buf[2 * k + 1] = val * chirp[2 * k + 1];
            }).wait();

            // Step 5: A = FFT(a)
            cooley_tukey_fft_sycl(a_buf, M, static_cast<T>(-1.0), queue);

            // Step 6: Pointwise multiply A[k] *= B[k] (complex multiply)
            queue.parallel_for(sycl::range<1>(M), [=](sycl::id<1> idx) {
                int64_t k = idx[0];
                T a_re = a_buf[2 * k];
                T a_im = a_buf[2 * k + 1];
                T b_re = B_buf[2 * k];
                T b_im = B_buf[2 * k + 1];
                a_buf[2 * k]     = a_re * b_re - a_im * b_im;
                a_buf[2 * k + 1] = a_re * b_im + a_im * b_re;
            }).wait();

            // Step 7: IFFT — forward FFT with sign=+1, then divide by M
            cooley_tukey_fft_sycl(a_buf, M, static_cast<T>(1.0), queue);
            T inv_M = static_cast<T>(1.0) / static_cast<T>(M);
            queue.parallel_for(sycl::range<1>(2 * M), [=](sycl::id<1> idx) {
                a_buf[idx[0]] *= inv_M;
            }).wait();

            // Step 8: result[k] = a[k] * conj(chirp[k]) for k=0..N-1
            queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                int64_t k = idx[0];
                T a_re = a_buf[2 * k];
                T a_im = a_buf[2 * k + 1];
                T c_re = chirp[2 * k];
                T c_im = -chirp[2 * k + 1]; // conj
                int64_t out_idx = (b * N * inner_size + k * inner_size + inner) * 2;
                d_out[out_idx]     = a_re * c_re - a_im * c_im;
                d_out[out_idx + 1] = a_re * c_im + a_im * c_re;
            }).wait();
        }
    }

    sycl::free(chirp, queue);
    sycl::free(b_buf, queue);
    sycl::free(B_buf, queue);
    sycl::free(a_buf, queue);
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
            float* d_buf = sycl::malloc_device<float>(2 * signal_len * batch_size, queue);
            const float* d_in = static_cast<const float*>(input.data_ptr());
            int64_t total = signal_len * batch_size;
            queue.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
                int64_t i = idx[0];
                d_buf[2 * i] = d_in[i];
                d_buf[2 * i + 1] = 0.0f;
            }).wait();

            for (int64_t b = 0; b < batch_size; ++b) {
                cooley_tukey_fft_sycl(d_buf + b * 2 * signal_len, signal_len, -1.0f, queue);
            }

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
            sycl::free(d_buf, queue);
            return output;
        } else {
            // Bluestein FFT on device for non-power-of-2 sizes
            const float* d_in = static_cast<const float*>(input.data_ptr());
            float* d_out = sycl::malloc_device<float>(out_numel, queue);
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
            sycl::free(d_out, queue);
            return output;
        }
    } else if (input.dtype() == DType::Float64) {
        int64_t numel = input.numel();
        std::vector<int64_t> out_shape = shape;
        out_shape.push_back(2);
        int64_t out_numel = 1;
        for (auto s : out_shape) out_numel *= s;

        if (use_cooley_tukey) {
            double* d_buf = sycl::malloc_device<double>(2 * signal_len * batch_size, queue);
            const double* d_in = static_cast<const double*>(input.data_ptr());
            int64_t total = signal_len * batch_size;
            queue.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
                int64_t i = idx[0];
                d_buf[2 * i] = d_in[i];
                d_buf[2 * i + 1] = 0.0;
            }).wait();

            for (int64_t b = 0; b < batch_size; ++b) {
                cooley_tukey_fft_sycl(d_buf + b * 2 * signal_len, signal_len, -1.0, queue);
            }

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
            sycl::free(d_buf, queue);
            return output;
        } else {
            // Bluestein FFT on device for non-power-of-2 sizes
            const double* d_in = static_cast<const double*>(input.data_ptr());
            double* d_out = sycl::malloc_device<double>(out_numel, queue);
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
            sycl::free(d_out, queue);
            return output;
        }
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        int64_t in_numel = input.numel();

        std::vector<float> host_f32(in_numel);
        if (orig_dtype == DType::Float16) {
            std::vector<sycl::half> host_in(in_numel);
            queue.memcpy(host_in.data(), input.data_ptr(), in_numel * sizeof(sycl::half)).wait();
            for (int64_t i = 0; i < in_numel; ++i) host_f32[i] = static_cast<float>(host_in[i]);
        } else {
            std::vector<uint16_t> host_in(in_numel);
            queue.memcpy(host_in.data(), input.data_ptr(), in_numel * sizeof(uint16_t)).wait();
            for (int64_t i = 0; i < in_numel; ++i) host_f32[i] = bf16_to_f32(host_in[i]);
        }

        Tensor f32_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         DType::Float32, input.device());
        queue.memcpy(const_cast<void*>(f32_input.data_ptr()), host_f32.data(), in_numel * sizeof(float)).wait();
        Tensor f32_result = fft_kernel(f32_input, dim, n, norm, queue);

        int64_t out_numel = f32_result.numel();
        Tensor output(std::vector<int64_t>(f32_result.shape().begin(), f32_result.shape().end()),
                      orig_dtype, input.device());
        std::vector<float> host_out(out_numel);
        queue.memcpy(host_out.data(), f32_result.data_ptr(), out_numel * sizeof(float)).wait();

        if (orig_dtype == DType::Float16) {
            std::vector<sycl::half> host_half(out_numel);
            for (int64_t i = 0; i < out_numel; ++i) host_half[i] = sycl::half(host_out[i]);
            queue.memcpy(const_cast<void*>(output.data_ptr()), host_half.data(), out_numel * sizeof(sycl::half)).wait();
        } else {
            std::vector<uint16_t> host_bf16(out_numel);
            for (int64_t i = 0; i < out_numel; ++i) host_bf16[i] = f32_to_bf16(host_out[i]);
            queue.memcpy(const_cast<void*>(output.data_ptr()), host_bf16.data(), out_numel * sizeof(uint16_t)).wait();
        }
        return output;
    } else {
        throw std::runtime_error("fft_kernel: unsupported dtype (expected Float32 or Float64)");
    }
}

// ============================================================================
// IFFT - 1D Complex-to-Complex Inverse FFT (naive fallback)
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

    if (input.dtype() == DType::Float32) {
        int64_t numel = input.numel();
        std::vector<float> h_in(numel);
        queue.memcpy(h_in.data(), input.data_ptr(), numel * sizeof(float)).wait();

        std::vector<int64_t> out_shape = shape;
        int64_t out_numel = numel;
        std::vector<float> h_out(out_numel, 0.0f);

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                for (int64_t k = 0; k < signal_len; ++k) {
                    float re_sum = 0.0f, im_sum = 0.0f;
                    for (int64_t t = 0; t < signal_len; ++t) {
                        int64_t in_idx = (b * signal_len * inner_size + t * inner_size + inner) * 2;
                        float re_in = h_in[in_idx];
                        float im_in = h_in[in_idx + 1];
                        float angle = 2.0f * 3.14159265358979323846f * k * t / signal_len;
                        re_sum += re_in * std::cos(angle) - im_in * std::sin(angle);
                        im_sum += re_in * std::sin(angle) + im_in * std::cos(angle);
                    }
                    float scale = 1.0f / static_cast<float>(signal_len);
                    if (norm == "ortho") {
                        scale = 1.0f / std::sqrt(static_cast<float>(signal_len));
                    } else if (norm == "forward") {
                        scale = 1.0f;
                    }
                    re_sum *= scale;
                    im_sum *= scale;
                    int64_t out_idx = (b * signal_len * inner_size + k * inner_size + inner) * 2;
                    h_out[out_idx] = re_sum;
                    h_out[out_idx + 1] = im_sum;
                }
            }
        }

        Tensor output(out_shape, DType::Float32, input.device());
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), out_numel * sizeof(float)).wait();
        return output;
    } else if (input.dtype() == DType::Float64) {
        int64_t numel = input.numel();
        std::vector<double> h_in(numel);
        queue.memcpy(h_in.data(), input.data_ptr(), numel * sizeof(double)).wait();

        std::vector<int64_t> out_shape = shape;
        int64_t out_numel = numel;
        std::vector<double> h_out(out_numel, 0.0);

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                for (int64_t k = 0; k < signal_len; ++k) {
                    double re_sum = 0.0, im_sum = 0.0;
                    for (int64_t t = 0; t < signal_len; ++t) {
                        int64_t in_idx = (b * signal_len * inner_size + t * inner_size + inner) * 2;
                        double re_in = h_in[in_idx];
                        double im_in = h_in[in_idx + 1];
                        double angle = 2.0 * 3.14159265358979323846 * k * t / signal_len;
                        re_sum += re_in * std::cos(angle) - im_in * std::sin(angle);
                        im_sum += re_in * std::sin(angle) + im_in * std::cos(angle);
                    }
                    double scale = 1.0 / static_cast<double>(signal_len);
                    if (norm == "ortho") {
                        scale = 1.0 / std::sqrt(static_cast<double>(signal_len));
                    } else if (norm == "forward") {
                        scale = 1.0;
                    }
                    re_sum *= scale;
                    im_sum *= scale;
                    int64_t out_idx = (b * signal_len * inner_size + k * inner_size + inner) * 2;
                    h_out[out_idx] = re_sum;
                    h_out[out_idx + 1] = im_sum;
                }
            }
        }

        Tensor output(out_shape, DType::Float64, input.device());
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), out_numel * sizeof(double)).wait();
        return output;
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        int64_t in_numel = input.numel();

        std::vector<float> host_f32(in_numel);
        if (orig_dtype == DType::Float16) {
            std::vector<sycl::half> host_in(in_numel);
            queue.memcpy(host_in.data(), input.data_ptr(), in_numel * sizeof(sycl::half)).wait();
            for (int64_t i = 0; i < in_numel; ++i) host_f32[i] = static_cast<float>(host_in[i]);
        } else {
            std::vector<uint16_t> host_in(in_numel);
            queue.memcpy(host_in.data(), input.data_ptr(), in_numel * sizeof(uint16_t)).wait();
            for (int64_t i = 0; i < in_numel; ++i) host_f32[i] = bf16_to_f32(host_in[i]);
        }

        Tensor f32_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         DType::Float32, input.device());
        queue.memcpy(const_cast<void*>(f32_input.data_ptr()), host_f32.data(), in_numel * sizeof(float)).wait();
        Tensor f32_result = ifft_kernel(f32_input, dim, n, norm, queue);

        int64_t out_numel = f32_result.numel();
        Tensor output(std::vector<int64_t>(f32_result.shape().begin(), f32_result.shape().end()),
                      orig_dtype, input.device());
        std::vector<float> host_out(out_numel);
        queue.memcpy(host_out.data(), f32_result.data_ptr(), out_numel * sizeof(float)).wait();

        if (orig_dtype == DType::Float16) {
            std::vector<sycl::half> host_half(out_numel);
            for (int64_t i = 0; i < out_numel; ++i) host_half[i] = sycl::half(host_out[i]);
            queue.memcpy(const_cast<void*>(output.data_ptr()), host_half.data(), out_numel * sizeof(sycl::half)).wait();
        } else {
            std::vector<uint16_t> host_bf16(out_numel);
            for (int64_t i = 0; i < out_numel; ++i) host_bf16[i] = f32_to_bf16(host_out[i]);
            queue.memcpy(const_cast<void*>(output.data_ptr()), host_bf16.data(), out_numel * sizeof(uint16_t)).wait();
        }
        return output;
    } else {
        throw std::runtime_error("ifft_kernel: unsupported dtype (expected Float32 or Float64)");
    }
}

// ============================================================================
// RFFT - 1D Real-to-Complex Forward FFT (naive fallback)
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

    if (input.dtype() == DType::Float32) {
        int64_t numel = input.numel();
        std::vector<float> h_in(numel);
        queue.memcpy(h_in.data(), input.data_ptr(), numel * sizeof(float)).wait();

        std::vector<int64_t> out_shape = shape;
        out_shape[dim] = out_len;
        out_shape.push_back(2);
        int64_t out_numel = 1;
        for (auto s : out_shape) out_numel *= s;
        std::vector<float> h_out(out_numel, 0.0f);

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                for (int64_t k = 0; k < out_len; ++k) {
                    float re_sum = 0.0f, im_sum = 0.0f;
                    for (int64_t t = 0; t < signal_len; ++t) {
                        int64_t in_idx = b * signal_len * inner_size + t * inner_size + inner;
                        float val = h_in[in_idx];
                        float angle = -2.0f * 3.14159265358979323846f * k * t / signal_len;
                        re_sum += val * std::cos(angle);
                        im_sum += val * std::sin(angle);
                    }
                    if (norm == "ortho") {
                        float scale = 1.0f / std::sqrt(static_cast<float>(signal_len));
                        re_sum *= scale;
                        im_sum *= scale;
                    } else if (norm == "forward") {
                        float scale = 1.0f / static_cast<float>(signal_len);
                        re_sum *= scale;
                        im_sum *= scale;
                    }
                    int64_t out_idx = (b * out_len * inner_size + k * inner_size + inner) * 2;
                    h_out[out_idx] = re_sum;
                    h_out[out_idx + 1] = im_sum;
                }
            }
        }

        Tensor output(out_shape, DType::Float32, input.device());
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), out_numel * sizeof(float)).wait();
        return output;
    } else if (input.dtype() == DType::Float64) {
        int64_t numel = input.numel();
        std::vector<double> h_in(numel);
        queue.memcpy(h_in.data(), input.data_ptr(), numel * sizeof(double)).wait();

        std::vector<int64_t> out_shape = shape;
        out_shape[dim] = out_len;
        out_shape.push_back(2);
        int64_t out_numel = 1;
        for (auto s : out_shape) out_numel *= s;
        std::vector<double> h_out(out_numel, 0.0);

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                for (int64_t k = 0; k < out_len; ++k) {
                    double re_sum = 0.0, im_sum = 0.0;
                    for (int64_t t = 0; t < signal_len; ++t) {
                        int64_t in_idx = b * signal_len * inner_size + t * inner_size + inner;
                        double val = h_in[in_idx];
                        double angle = -2.0 * 3.14159265358979323846 * k * t / signal_len;
                        re_sum += val * std::cos(angle);
                        im_sum += val * std::sin(angle);
                    }
                    if (norm == "ortho") {
                        double scale = 1.0 / std::sqrt(static_cast<double>(signal_len));
                        re_sum *= scale;
                        im_sum *= scale;
                    } else if (norm == "forward") {
                        double scale = 1.0 / static_cast<double>(signal_len);
                        re_sum *= scale;
                        im_sum *= scale;
                    }
                    int64_t out_idx = (b * out_len * inner_size + k * inner_size + inner) * 2;
                    h_out[out_idx] = re_sum;
                    h_out[out_idx + 1] = im_sum;
                }
            }
        }

        Tensor output(out_shape, DType::Float64, input.device());
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), out_numel * sizeof(double)).wait();
        return output;
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        int64_t in_numel = input.numel();

        std::vector<float> host_f32(in_numel);
        if (orig_dtype == DType::Float16) {
            std::vector<sycl::half> host_in(in_numel);
            queue.memcpy(host_in.data(), input.data_ptr(), in_numel * sizeof(sycl::half)).wait();
            for (int64_t i = 0; i < in_numel; ++i) host_f32[i] = static_cast<float>(host_in[i]);
        } else {
            std::vector<uint16_t> host_in(in_numel);
            queue.memcpy(host_in.data(), input.data_ptr(), in_numel * sizeof(uint16_t)).wait();
            for (int64_t i = 0; i < in_numel; ++i) host_f32[i] = bf16_to_f32(host_in[i]);
        }

        Tensor f32_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         DType::Float32, input.device());
        queue.memcpy(const_cast<void*>(f32_input.data_ptr()), host_f32.data(), in_numel * sizeof(float)).wait();
        Tensor f32_result = rfft_kernel(f32_input, dim, n, norm, queue);

        int64_t out_numel = f32_result.numel();
        Tensor output(std::vector<int64_t>(f32_result.shape().begin(), f32_result.shape().end()),
                      orig_dtype, input.device());
        std::vector<float> host_out(out_numel);
        queue.memcpy(host_out.data(), f32_result.data_ptr(), out_numel * sizeof(float)).wait();

        if (orig_dtype == DType::Float16) {
            std::vector<sycl::half> host_half(out_numel);
            for (int64_t i = 0; i < out_numel; ++i) host_half[i] = sycl::half(host_out[i]);
            queue.memcpy(const_cast<void*>(output.data_ptr()), host_half.data(), out_numel * sizeof(sycl::half)).wait();
        } else {
            std::vector<uint16_t> host_bf16(out_numel);
            for (int64_t i = 0; i < out_numel; ++i) host_bf16[i] = f32_to_bf16(host_out[i]);
            queue.memcpy(const_cast<void*>(output.data_ptr()), host_bf16.data(), out_numel * sizeof(uint16_t)).wait();
        }
        return output;
    } else {
        throw std::runtime_error("rfft_kernel: unsupported dtype (expected Float32 or Float64)");
    }
}

// ============================================================================
// IRFFT - 1D Complex-to-Real Inverse FFT (naive fallback)
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

    int64_t complex_len = shape[dim];
    int64_t output_len = n;

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) batch_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim - 1; ++i) inner_size *= shape[i];

    if (input.dtype() == DType::Float32) {
        int64_t numel = input.numel();
        std::vector<float> h_in(numel);
        queue.memcpy(h_in.data(), input.data_ptr(), numel * sizeof(float)).wait();

        std::vector<int64_t> out_shape;
        for (int64_t i = 0; i < dim; ++i) out_shape.push_back(shape[i]);
        out_shape.push_back(output_len);
        for (int64_t i = dim + 1; i < ndim - 1; ++i) out_shape.push_back(shape[i]);

        int64_t out_numel = 1;
        for (auto s : out_shape) out_numel *= s;
        std::vector<float> h_out(out_numel, 0.0f);

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                for (int64_t t = 0; t < output_len; ++t) {
                    float re_sum = 0.0f;
                    for (int64_t k = 0; k < complex_len; ++k) {
                        int64_t in_idx = (b * complex_len * inner_size + k * inner_size + inner) * 2;
                        float re_in = h_in[in_idx];
                        float im_in = h_in[in_idx + 1];
                        float angle = 2.0f * 3.14159265358979323846f * k * t / output_len;
                        float contribution = re_in * std::cos(angle) - im_in * std::sin(angle);
                        if (k > 0 && k < output_len - k) {
                            contribution *= 2.0f;
                        }
                        re_sum += contribution;
                    }
                    float scale = 1.0f / static_cast<float>(output_len);
                    if (norm == "ortho") {
                        scale = 1.0f / std::sqrt(static_cast<float>(output_len));
                    } else if (norm == "forward") {
                        scale = 1.0f;
                    }
                    re_sum *= scale;
                    int64_t out_idx = b * output_len * inner_size + t * inner_size + inner;
                    h_out[out_idx] = re_sum;
                }
            }
        }

        Tensor output(out_shape, DType::Float32, input.device());
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), out_numel * sizeof(float)).wait();
        return output;
    } else if (input.dtype() == DType::Float64) {
        int64_t numel = input.numel();
        std::vector<double> h_in(numel);
        queue.memcpy(h_in.data(), input.data_ptr(), numel * sizeof(double)).wait();

        std::vector<int64_t> out_shape;
        for (int64_t i = 0; i < dim; ++i) out_shape.push_back(shape[i]);
        out_shape.push_back(output_len);
        for (int64_t i = dim + 1; i < ndim - 1; ++i) out_shape.push_back(shape[i]);

        int64_t out_numel = 1;
        for (auto s : out_shape) out_numel *= s;
        std::vector<double> h_out(out_numel, 0.0);

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                for (int64_t t = 0; t < output_len; ++t) {
                    double re_sum = 0.0;
                    for (int64_t k = 0; k < complex_len; ++k) {
                        int64_t in_idx = (b * complex_len * inner_size + k * inner_size + inner) * 2;
                        double re_in = h_in[in_idx];
                        double im_in = h_in[in_idx + 1];
                        double angle = 2.0 * 3.14159265358979323846 * k * t / output_len;
                        double contribution = re_in * std::cos(angle) - im_in * std::sin(angle);
                        if (k > 0 && k < output_len - k) {
                            contribution *= 2.0;
                        }
                        re_sum += contribution;
                    }
                    double scale = 1.0 / static_cast<double>(output_len);
                    if (norm == "ortho") {
                        scale = 1.0 / std::sqrt(static_cast<double>(output_len));
                    } else if (norm == "forward") {
                        scale = 1.0;
                    }
                    re_sum *= scale;
                    int64_t out_idx = b * output_len * inner_size + t * inner_size + inner;
                    h_out[out_idx] = re_sum;
                }
            }
        }

        Tensor output(out_shape, DType::Float64, input.device());
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), out_numel * sizeof(double)).wait();
        return output;
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        // Upcast to Float32, compute, downcast
        DType orig_dtype = input.dtype();
        int64_t numel = input.numel();
        std::vector<float> host_f32(numel);
        if (orig_dtype == DType::Float16) {
            std::vector<sycl::half> host_in(numel);
            queue.memcpy(host_in.data(), input.data_ptr(), numel * sizeof(sycl::half)).wait();
            for (int64_t i = 0; i < numel; ++i) host_f32[i] = static_cast<float>(host_in[i]);
        } else {
            std::vector<uint16_t> host_in(numel);
            queue.memcpy(host_in.data(), input.data_ptr(), numel * sizeof(uint16_t)).wait();
            for (int64_t i = 0; i < numel; ++i) host_f32[i] = bf16_to_f32(host_in[i]);
        }
        Tensor f32_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         DType::Float32, input.device());
        queue.memcpy(const_cast<void*>(f32_input.data_ptr()), host_f32.data(), numel * sizeof(float)).wait();
        Tensor f32_result = irfft_kernel(f32_input, dim, n, norm, queue);

        // Downcast result to original dtype
        int64_t out_numel = f32_result.numel();
        std::vector<float> host_out(out_numel);
        queue.memcpy(host_out.data(), f32_result.data_ptr(), out_numel * sizeof(float)).wait();
        Tensor output(std::vector<int64_t>(f32_result.shape().begin(), f32_result.shape().end()),
                      orig_dtype, input.device());
        if (orig_dtype == DType::Float16) {
            std::vector<sycl::half> host_half(out_numel);
            for (int64_t i = 0; i < out_numel; ++i) host_half[i] = sycl::half(host_out[i]);
            queue.memcpy(const_cast<void*>(output.data_ptr()), host_half.data(), out_numel * sizeof(sycl::half)).wait();
        } else {
            std::vector<uint16_t> host_bf16(out_numel);
            for (int64_t i = 0; i < out_numel; ++i) host_bf16[i] = f32_to_bf16(host_out[i]);
            queue.memcpy(const_cast<void*>(output.data_ptr()), host_bf16.data(), out_numel * sizeof(uint16_t)).wait();
        }
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
