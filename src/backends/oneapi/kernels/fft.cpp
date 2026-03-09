/**
 * @file fft.cpp
 * @brief OneAPI/SYCL FFT kernels via oneMKL DFT
 *
 * Implements FFT, IFFT, RFFT, IRFFT, FFT2, IFFT2, FFTN, IFFTN
 * using oneMKL DFT (Discrete Fourier Transform) APIs when available.
 * Falls back to naive O(n^2) DFT when oneMKL is not present.
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
void apply_scale_f32(sycl::queue& queue, float* data, int64_t total_floats, float scale) {
    if (scale == 1.0f) return;
    queue.parallel_for(sycl::range<1>(total_floats), [=](sycl::id<1> idx) {
        data[idx] *= scale;
    }).wait();
}

/// Apply in-place scaling to interleaved complex data (Float64 pairs).
void apply_scale_f64(sycl::queue& queue, double* data, int64_t total_doubles, double scale) {
    if (scale == 1.0) return;
    queue.parallel_for(sycl::range<1>(total_doubles), [=](sycl::id<1> idx) {
        data[idx] *= scale;
    }).wait();
}

/// Apply in-place scaling to real data (Float32).
void apply_scale_real_f32(sycl::queue& queue, float* data, int64_t numel, float scale) {
    if (scale == 1.0f) return;
    queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
        data[idx] *= scale;
    }).wait();
}

/// Apply in-place scaling to real data (Float64).
void apply_scale_real_f64(sycl::queue& queue, double* data, int64_t numel, double scale) {
    if (scale == 1.0) return;
    queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
        data[idx] *= scale;
    }).wait();
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
            }).wait();

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
            queue.wait();
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
                }).wait();

            // Each transform is contiguous signal_len complex elements
            dft::descriptor<dft::precision::SINGLE, dft::domain::COMPLEX> desc(signal_len);
            desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, total_transforms);
            std::int64_t fwd_strides[2] = {0, 1};
            desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
            desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);
            desc.set_value(dft::config_param::PLACEMENT, DFTI_INPLACE);
            desc.commit(queue);

            dft::compute_forward(desc, reinterpret_cast<std::complex<float>*>(complex_buf));
            queue.wait();
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
            }).wait();

            dft::descriptor<dft::precision::DOUBLE, dft::domain::COMPLEX> desc(signal_len);
            desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, batch_size);
            std::int64_t fwd_strides[2] = {0, 1};
            desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
            desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);
            desc.set_value(dft::config_param::PLACEMENT, DFTI_INPLACE);
            desc.commit(queue);

            dft::compute_forward(desc, reinterpret_cast<std::complex<double>*>(complex_buf));
            queue.wait();
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
                }).wait();

            dft::descriptor<dft::precision::DOUBLE, dft::domain::COMPLEX> desc(signal_len);
            desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, total_transforms);
            std::int64_t fwd_strides[2] = {0, 1};
            desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
            desc.set_value(dft::config_param::FWD_DISTANCE, signal_len);
            desc.set_value(dft::config_param::PLACEMENT, DFTI_INPLACE);
            desc.commit(queue);

            dft::compute_forward(desc, reinterpret_cast<std::complex<double>*>(complex_buf));
            queue.wait();
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
            queue.memcpy(complex_buf, in_ptr, complex_buf_floats * sizeof(float)).wait();
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
                }).wait();
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
        queue.wait();

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
            queue.memcpy(complex_buf, in_ptr, complex_buf_doubles * sizeof(double)).wait();
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
                }).wait();
        }

        dft::descriptor<dft::precision::DOUBLE, dft::domain::COMPLEX> desc(signal_len);
        desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, total_transforms);
        std::int64_t bwd_strides[2] = {0, 1};
        desc.set_value(dft::config_param::BWD_STRIDES, bwd_strides);
        desc.set_value(dft::config_param::BWD_DISTANCE, signal_len);
        desc.set_value(dft::config_param::PLACEMENT, DFTI_INPLACE);
        desc.commit(queue);

        dft::compute_backward(desc, reinterpret_cast<std::complex<double>*>(complex_buf));
        queue.wait();

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
            queue.memcpy(real_buf, in_ptr, total_transforms * signal_len * sizeof(float)).wait();
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
                }).wait();
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
        queue.wait();

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
            queue.memcpy(real_buf, in_ptr, total_transforms * signal_len * sizeof(double)).wait();
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
                }).wait();
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
        queue.wait();

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
                         total_transforms * complex_len * 2 * sizeof(float)).wait();
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
                }).wait();
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
        queue.wait();

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
                         total_transforms * complex_len * 2 * sizeof(double)).wait();
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
                }).wait();
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
        queue.wait();

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

#else // !TENZOR_HAS_ONEMKL — Naive O(n^2) DFT fallback

// ============================================================================
// FFT - 1D Complex-to-Complex Forward FFT (naive fallback)
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

    if (input.dtype() == DType::Float32) {
        int64_t numel = input.numel();
        std::vector<float> h_in(numel);
        queue.memcpy(h_in.data(), input.data_ptr(), numel * sizeof(float)).wait();

        std::vector<int64_t> out_shape = shape;
        out_shape.push_back(2);
        int64_t out_numel = 1;
        for (auto s : out_shape) out_numel *= s;
        std::vector<float> h_out(out_numel, 0.0f);

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                for (int64_t k = 0; k < signal_len; ++k) {
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
        out_shape.push_back(2);
        int64_t out_numel = 1;
        for (auto s : out_shape) out_numel *= s;
        std::vector<double> h_out(out_numel, 0.0);

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                for (int64_t k = 0; k < signal_len; ++k) {
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
                    int64_t out_idx = (b * signal_len * inner_size + k * inner_size + inner) * 2;
                    h_out[out_idx] = re_sum;
                    h_out[out_idx + 1] = im_sum;
                }
            }
        }

        Tensor output(out_shape, DType::Float64, input.device());
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), out_numel * sizeof(double)).wait();
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
