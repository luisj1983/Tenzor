/**
 * @file fft.cu
 * @brief CUDA FFT kernels using cuFFT.
 *
 * Provides GPU-accelerated implementations of:
 * - fft  (1D complex-to-complex forward)
 * - ifft (1D complex-to-complex inverse)
 * - rfft (1D real-to-complex forward)
 * - irfft(1D complex-to-real inverse)
 * - fft2 (2D forward)
 * - ifft2(2D inverse)
 * - fftn (N-D forward)
 * - ifftn(N-D inverse)
 *
 * cuFFT does NOT normalize by default. Normalization ("backward", "forward",
 * "ortho") is applied manually via a post-transform CUDA kernel.
 */

#ifdef TENZOR_HAS_CUFFT

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "cuda_common.cuh"

#include <cufft.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>

namespace tenzor {
namespace cuda {

namespace {

// ============================================================================
// cuFFT error checking
// ============================================================================

#ifndef CUFFT_CHECK
#define CUFFT_CHECK(call)                                                      \
    do {                                                                        \
        cufftResult result = (call);                                           \
        if (result != CUFFT_SUCCESS) {                                         \
            throw std::runtime_error(                                          \
                std::string("cuFFT error at ") + __FILE__ + ":" +             \
                std::to_string(__LINE__) + " - error code " +                  \
                std::to_string(static_cast<int>(result)));                     \
        }                                                                      \
    } while (0)
#endif

// ============================================================================
// RAII wrapper for cufftHandle
// ============================================================================

struct CuFFTPlan {
    cufftHandle handle = 0;
    bool valid = false;
    cudaStream_t stream_ = nullptr;  ///< Stream bound to this plan via set_stream.

    CuFFTPlan() = default;
    ~CuFFTPlan() {
        if (valid) {
            // Audit QQ.7: synchronize the bound stream before destroying the
            // cuFFT handle. cuFFT plans own internal workspace memory still
            // referenced by inflight exec/normalization kernels; tearing the
            // plan down while those kernels are queued races and can corrupt
            // subsequent FFTs.
            if (stream_ != nullptr) {
                // Destructor must not throw — swallow errors.
                (void)cudaStreamSynchronize(stream_);
            }
            cufftDestroy(handle);
        }
    }

    CuFFTPlan(const CuFFTPlan&) = delete;
    CuFFTPlan& operator=(const CuFFTPlan&) = delete;

    CuFFTPlan(CuFFTPlan&& other) noexcept
        : handle(other.handle), valid(other.valid), stream_(other.stream_) {
        other.valid = false;
        other.stream_ = nullptr;
    }

    void create() {
        CUFFT_CHECK(cufftCreate(&handle));
        valid = true;
    }

    void set_stream(cudaStream_t stream) {
        CUFFT_CHECK(cufftSetStream(handle, stream));
        stream_ = stream;
    }

    operator cufftHandle() const { return handle; }
};

// ============================================================================
// Normalization kernels
// cuFFT does not normalize. We apply scaling as a post-processing step.
// ============================================================================

template<typename T>
__global__ void scale_kernel(T* data, int64_t numel, T scale) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] *= scale;
    }
}

// Scale complex data (2 floats per complex element)
template<typename RealT>
__global__ void scale_complex_kernel(RealT* data, int64_t numel_complex, RealT scale) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = numel_complex * 2;  // real + imag parts
    if (idx < total) {
        data[idx] *= scale;
    }
}

/// Compute normalization scale factor.
/// @param n The transform size along the FFT dimension.
/// @param norm Normalization mode: "backward" (default), "forward", "ortho".
/// @param is_forward True for forward FFT, false for inverse.
/// @return The scale factor to apply to the output.
double get_norm_factor(int64_t n, const std::string& norm, bool is_forward) {
    if (norm == "ortho") {
        return 1.0 / std::sqrt(static_cast<double>(n));
    } else if ((norm == "forward" && is_forward) || (norm == "backward" && !is_forward)) {
        return 1.0 / static_cast<double>(n);
    }
    // "backward" + forward, or "forward" + inverse: no scaling
    return 1.0;
}

/// Compute normalization scale factor for multi-dimensional FFTs.
/// The total normalization factor is the product over all FFT dimensions.
double get_norm_factor_nd(const std::vector<int64_t>& n_vec, const std::string& norm, bool is_forward) {
    double factor = 1.0;
    for (auto n : n_vec) {
        factor *= get_norm_factor(n, norm, is_forward);
    }
    return factor;
}

/// Apply scaling to a complex tensor (in-place).
void apply_normalization_complex(Tensor& output, double scale, bool is_float32, cudaStream_t stream) {
    if (scale == 1.0) return;

    int64_t numel = output.numel();
    int64_t total_reals = numel * 2;
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((total_reals + block_size - 1) / block_size);

    if (is_float32) {
        float s = static_cast<float>(scale);
        scale_complex_kernel<float><<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<float*>(output.data_ptr()), numel, s);
    } else {
        double s = scale;
        scale_complex_kernel<double><<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<double*>(output.data_ptr()), numel, s);
    }
    TENZOR_CUDA_CHECK(cudaGetLastError());
}

/// Apply scaling to a real tensor (in-place).
void apply_normalization_real(Tensor& output, double scale, bool is_float32, cudaStream_t stream) {
    if (scale == 1.0) return;

    int64_t numel = output.numel();
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);

    if (is_float32) {
        float s = static_cast<float>(scale);
        scale_kernel<float><<<grid_size, block_size, 0, stream>>>(
            output.data<float>(), numel, s);
    } else {
        double s = scale;
        scale_kernel<double><<<grid_size, block_size, 0, stream>>>(
            output.data<double>(), numel, s);
    }
    TENZOR_CUDA_CHECK(cudaGetLastError());
}

/// Check if data is contiguous along the FFT dimension (i.e., FFT dim is the last dim).
/// cuFFT works most naturally on the innermost dimension. If the FFT dimension is not
/// the last, we need to use advanced data layout (istride/ostride/idist/odist).

} // anonymous namespace

// ============================================================================
// 1D FFT: Complex-to-Complex forward
// ============================================================================

auto cuda_fft_kernel(const Tensor& input, int64_t dim, int64_t n,
                     const std::string& norm, cudaStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("FFT: dimension out of range");
    }
    bool is_float32 = (input.dtype() == DType::Complex64);

    // Determine transform parameters
    int64_t N_in = shape[dim];
    int64_t N_out = n;

    // Build output shape
    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = N_out;

    // Allocate output
    DType out_dtype = input.dtype();  // Complex64 or Complex128
    Tensor output(out_shape, out_dtype, input.device());

    // If input size along dim != n, we need to pad or truncate.
    // For simplicity, if they match, use in-place-style plan.
    // If they differ, we copy input data into output buffer (zero-padded or truncated).
    Tensor work_input = input;
    if (N_in != N_out) {
        // Create a zero-initialized buffer of the output shape and copy input data
        // For padding: copy N_in elements, rest stays zero
        // For truncation: copy N_out elements
        // We do this by allocating output as zeros and copying the min(N_in,N_out) slice
        TENZOR_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0,
            output.numel() * dtype_size(out_dtype), stream));

        // Compute strides for the copy
        int64_t copy_len = std::min(N_in, N_out);
        int64_t elem_size = dtype_size(out_dtype);

        // inner_size = product of dims after dim
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

        // outer_size = product of dims before dim
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];

        // Copy row by row
        for (int64_t outer = 0; outer < outer_size; ++outer) {
            const char* src = static_cast<const char*>(input.data_ptr())
                + outer * N_in * inner_size * elem_size;
            char* dst = static_cast<char*>(output.data_ptr())
                + outer * N_out * inner_size * elem_size;

            TENZOR_CUDA_CHECK(cudaMemcpyAsync(dst, src,
                copy_len * inner_size * elem_size,
                cudaMemcpyDeviceToDevice, stream));
        }

        // Use output as both input and output for the in-place FFT
        work_input = output;
    } else {
        // Copy input to output (cuFFT can do in-place)
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(output.data_ptr(), input.data_ptr(),
            input.numel() * dtype_size(out_dtype),
            cudaMemcpyDeviceToDevice, stream));
        work_input = output;
    }

    // Compute strides for cufftPlanMany
    // inner_size = product of dims after FFT dim
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= out_shape[i];

    // outer_size = product of dims before FFT dim
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= out_shape[i];

    int64_t batch = outer_size * inner_size;

    // Create cuFFT plan
    CuFFTPlan plan;
    plan.create();
    plan.set_stream(stream);

    int n_int = static_cast<int>(N_out);
    int inembed[] = {n_int};
    int onembed[] = {n_int};

    // audit-2026-05-03 — for FFT along a non-last dim with both outer
    // and inner extent > 1, cufftPlanMany cannot encode the layout
    // (stride between batches differs between within-outer-block and
    // across-outer-block). The previous code passed batch =
    // outer_size * inner_size with idist=1, which silently produced
    // wrong values for batches whose start offsets weren't consecutive.
    // Loop per outer block instead.
    bool last_dim = (dim == ndim - 1);
    cufftType fft_type = is_float32 ? CUFFT_C2C : CUFFT_Z2Z;

    if (last_dim) {
        int istride = 1;
        int ostride = 1;
        int idist = n_int;
        int odist = n_int;
        int plan_batch = static_cast<int>(outer_size);
        CUFFT_CHECK(cufftPlanMany(&plan.handle, 1, &n_int,
                                  inembed, istride, idist,
                                  onembed, ostride, odist,
                                  fft_type, plan_batch));
        CUFFT_CHECK(cufftSetStream(plan.handle, stream));

        if (is_float32) {
            CUFFT_CHECK(cufftExecC2C(plan,
                reinterpret_cast<cufftComplex*>(output.data_ptr()),
                reinterpret_cast<cufftComplex*>(output.data_ptr()),
                CUFFT_FORWARD));
        } else {
            CUFFT_CHECK(cufftExecZ2Z(plan,
                reinterpret_cast<cufftDoubleComplex*>(output.data_ptr()),
                reinterpret_cast<cufftDoubleComplex*>(output.data_ptr()),
                CUFFT_FORWARD));
        }
    } else {
        int istride = static_cast<int>(inner_size);
        int ostride = static_cast<int>(inner_size);
        int idist = 1;
        int odist = 1;
        int plan_batch = static_cast<int>(inner_size);  // batches per outer block
        CUFFT_CHECK(cufftPlanMany(&plan.handle, 1, &n_int,
                                  inembed, istride, idist,
                                  onembed, ostride, odist,
                                  fft_type, plan_batch));
        CUFFT_CHECK(cufftSetStream(plan.handle, stream));

        size_t outer_stride_elems = static_cast<size_t>(N_out) * static_cast<size_t>(inner_size);
        for (int64_t b = 0; b < outer_size; ++b) {
            void* base = static_cast<char*>(output.data_ptr())
                + b * outer_stride_elems * dtype_size(out_dtype);
            if (is_float32) {
                CUFFT_CHECK(cufftExecC2C(plan,
                    reinterpret_cast<cufftComplex*>(base),
                    reinterpret_cast<cufftComplex*>(base),
                    CUFFT_FORWARD));
            } else {
                CUFFT_CHECK(cufftExecZ2Z(plan,
                    reinterpret_cast<cufftDoubleComplex*>(base),
                    reinterpret_cast<cufftDoubleComplex*>(base),
                    CUFFT_FORWARD));
            }
        }
    }

    // Apply normalization
    double scale = get_norm_factor(N_out, norm, /*is_forward=*/true);
    apply_normalization_complex(output, scale, is_float32, stream);

    return output;
}

// ============================================================================
// 1D IFFT: Complex-to-Complex inverse
// ============================================================================

auto cuda_ifft_kernel(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm, cudaStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("IFFT: dimension out of range");
    }
    bool is_float32 = (input.dtype() == DType::Complex64);

    int64_t N_in = shape[dim];
    int64_t N_out = n;

    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = N_out;

    DType out_dtype = input.dtype();
    Tensor output(out_shape, out_dtype, input.device());

    // Handle padding/truncation
    if (N_in != N_out) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0,
            output.numel() * dtype_size(out_dtype), stream));

        int64_t copy_len = std::min(N_in, N_out);
        int64_t elem_size = dtype_size(out_dtype);
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            const char* src = static_cast<const char*>(input.data_ptr())
                + outer * N_in * inner_size * elem_size;
            char* dst = static_cast<char*>(output.data_ptr())
                + outer * N_out * inner_size * elem_size;
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(dst, src,
                copy_len * inner_size * elem_size,
                cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(output.data_ptr(), input.data_ptr(),
            input.numel() * dtype_size(out_dtype),
            cudaMemcpyDeviceToDevice, stream));
    }

    // Compute layout
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= out_shape[i];
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= out_shape[i];
    int64_t batch = outer_size * inner_size;

    CuFFTPlan plan;
    plan.create();
    plan.set_stream(stream);

    int n_int = static_cast<int>(N_out);
    int inembed[] = {n_int};
    int onembed[] = {n_int};
    bool last_dim = (dim == ndim - 1);
    cufftType fft_type = is_float32 ? CUFFT_C2C : CUFFT_Z2Z;

    if (last_dim) {
        int istride = 1;
        int ostride = 1;
        int idist = n_int;
        int odist = n_int;
        int plan_batch = static_cast<int>(outer_size);
        CUFFT_CHECK(cufftPlanMany(&plan.handle, 1, &n_int,
                                  inembed, istride, idist,
                                  onembed, ostride, odist,
                                  fft_type, plan_batch));
        CUFFT_CHECK(cufftSetStream(plan.handle, stream));

        if (is_float32) {
            CUFFT_CHECK(cufftExecC2C(plan,
                reinterpret_cast<cufftComplex*>(output.data_ptr()),
                reinterpret_cast<cufftComplex*>(output.data_ptr()),
                CUFFT_INVERSE));
        } else {
            CUFFT_CHECK(cufftExecZ2Z(plan,
                reinterpret_cast<cufftDoubleComplex*>(output.data_ptr()),
                reinterpret_cast<cufftDoubleComplex*>(output.data_ptr()),
                CUFFT_INVERSE));
        }
    } else {
        // audit-2026-05-03 — same outer-block loop fix as cuda_fft_kernel.
        int istride = static_cast<int>(inner_size);
        int ostride = static_cast<int>(inner_size);
        int idist = 1;
        int odist = 1;
        int plan_batch = static_cast<int>(inner_size);
        CUFFT_CHECK(cufftPlanMany(&plan.handle, 1, &n_int,
                                  inembed, istride, idist,
                                  onembed, ostride, odist,
                                  fft_type, plan_batch));
        CUFFT_CHECK(cufftSetStream(plan.handle, stream));

        size_t outer_stride_elems = static_cast<size_t>(N_out) * static_cast<size_t>(inner_size);
        for (int64_t b = 0; b < outer_size; ++b) {
            void* base = static_cast<char*>(output.data_ptr())
                + b * outer_stride_elems * dtype_size(out_dtype);
            if (is_float32) {
                CUFFT_CHECK(cufftExecC2C(plan,
                    reinterpret_cast<cufftComplex*>(base),
                    reinterpret_cast<cufftComplex*>(base),
                    CUFFT_INVERSE));
            } else {
                CUFFT_CHECK(cufftExecZ2Z(plan,
                    reinterpret_cast<cufftDoubleComplex*>(base),
                    reinterpret_cast<cufftDoubleComplex*>(base),
                    CUFFT_INVERSE));
            }
        }
    }

    // cuFFT inverse does NOT divide by N. Default "backward" norm = divide by N.
    double scale = get_norm_factor(N_out, norm, /*is_forward=*/false);
    apply_normalization_complex(output, scale, is_float32, stream);

    return output;
}

// ============================================================================
// 1D RFFT: Real-to-Complex forward
// ============================================================================

auto cuda_rfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm, cudaStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("RFFT: dimension out of range");
    }
    bool is_float32 = (input.dtype() == DType::Float32);

    int64_t N_in = shape[dim];
    int64_t N_out_complex = n / 2 + 1;

    // For R2C, cuFFT requires the FFT dimension to be the innermost (last) dimension
    // when using the simple plan interface. We handle only the last-dim case directly;
    // for other dims the dispatch layer decomposes into 1D FFTs.
    if (dim != ndim - 1) {
        throw std::runtime_error(
            "cuFFT rfft: only last-dimension FFT is supported on CUDA. "
            "The dispatch layer should decompose non-last-dim rfft.");
    }

    // Prepare real input buffer (padded or truncated to length n)
    std::vector<int64_t> real_shape = shape;
    real_shape[dim] = n;
    DType real_dtype = input.dtype();

    Tensor real_buf(real_shape, real_dtype, input.device());
    if (N_in != n) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(real_buf.data_ptr(), 0,
            real_buf.numel() * dtype_size(real_dtype), stream));
        int64_t copy_len = std::min(N_in, n);
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
        int64_t elem_size = dtype_size(real_dtype);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            const char* src = static_cast<const char*>(input.data_ptr())
                + outer * N_in * elem_size;
            char* dst = static_cast<char*>(real_buf.data_ptr())
                + outer * n * elem_size;
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(dst, src,
                copy_len * elem_size, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(real_buf.data_ptr(), input.data_ptr(),
            input.numel() * dtype_size(real_dtype),
            cudaMemcpyDeviceToDevice, stream));
    }

    // Output: complex, with shape[dim] = n/2 + 1
    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = N_out_complex;
    DType complex_dtype = is_float32 ? DType::Complex64 : DType::Complex128;
    Tensor output(out_shape, complex_dtype, input.device());

    // Batch = product of all dims except the last
    int64_t batch = 1;
    for (int64_t i = 0; i < dim; ++i) batch *= shape[i];

    CuFFTPlan plan;
    plan.create();
    plan.set_stream(stream);

    int n_int = static_cast<int>(n);
    cufftType fft_type = is_float32 ? CUFFT_R2C : CUFFT_D2Z;

    CUFFT_CHECK(cufftPlan1d(&plan.handle, n_int, fft_type, static_cast<int>(batch)));
    CUFFT_CHECK(cufftSetStream(plan.handle, stream));

    if (is_float32) {
        CUFFT_CHECK(cufftExecR2C(plan,
            reinterpret_cast<cufftReal*>(real_buf.data_ptr()),
            reinterpret_cast<cufftComplex*>(output.data_ptr())));
    } else {
        CUFFT_CHECK(cufftExecD2Z(plan,
            reinterpret_cast<cufftDoubleReal*>(real_buf.data_ptr()),
            reinterpret_cast<cufftDoubleComplex*>(output.data_ptr())));
    }

    // Apply normalization
    double scale = get_norm_factor(n, norm, /*is_forward=*/true);
    apply_normalization_complex(output, scale, is_float32, stream);

    return output;
}

// ============================================================================
// 1D IRFFT: Complex-to-Real inverse
// ============================================================================

auto cuda_irfft_kernel(const Tensor& input_raw, int64_t dim, int64_t n,
                       const std::string& norm, cudaStream_t stream) -> Tensor {
    // Accept real inputs (Float32 / Float64) by widening to the matching
    // complex dtype first — mirrors the CPU path (cpu/kernels/fft.cpp).
    Tensor input = input_raw;
    if (input.dtype() == DType::Float32) input = input.to(DType::Complex64);
    else if (input.dtype() == DType::Float64) input = input.to(DType::Complex128);

    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("IRFFT: dimension out of range");
    }
    bool is_float32 = (input.dtype() == DType::Complex64);

    if (dim != ndim - 1) {
        throw std::runtime_error(
            "cuFFT irfft: only last-dimension IRFFT is supported on CUDA. "
            "The dispatch layer should decompose non-last-dim irfft.");
    }

    int64_t N_in = shape[dim];  // n/2 + 1 complex elements
    int64_t expected_complex = n / 2 + 1;

    // Copy input (complex) into a work buffer sized for expected_complex
    std::vector<int64_t> complex_shape = shape;
    complex_shape[dim] = expected_complex;
    DType complex_dtype = input.dtype();
    Tensor complex_buf(complex_shape, complex_dtype, input.device());

    if (N_in != expected_complex) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(complex_buf.data_ptr(), 0,
            complex_buf.numel() * dtype_size(complex_dtype), stream));
        int64_t copy_len = std::min(N_in, expected_complex);
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
        int64_t elem_size = dtype_size(complex_dtype);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            const char* src = static_cast<const char*>(input.data_ptr())
                + outer * N_in * elem_size;
            char* dst = static_cast<char*>(complex_buf.data_ptr())
                + outer * expected_complex * elem_size;
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(dst, src,
                copy_len * elem_size, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(complex_buf.data_ptr(), input.data_ptr(),
            input.numel() * dtype_size(complex_dtype),
            cudaMemcpyDeviceToDevice, stream));
    }

    // Output: real tensor with shape[dim] = n
    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = n;
    DType real_dtype = is_float32 ? DType::Float32 : DType::Float64;
    Tensor output(out_shape, real_dtype, input.device());

    int64_t batch = 1;
    for (int64_t i = 0; i < dim; ++i) batch *= shape[i];

    CuFFTPlan plan;
    plan.create();
    plan.set_stream(stream);

    int n_int = static_cast<int>(n);
    cufftType fft_type = is_float32 ? CUFFT_C2R : CUFFT_Z2D;

    CUFFT_CHECK(cufftPlan1d(&plan.handle, n_int, fft_type, static_cast<int>(batch)));
    CUFFT_CHECK(cufftSetStream(plan.handle, stream));

    if (is_float32) {
        CUFFT_CHECK(cufftExecC2R(plan,
            reinterpret_cast<cufftComplex*>(complex_buf.data_ptr()),
            reinterpret_cast<cufftReal*>(output.data_ptr())));
    } else {
        CUFFT_CHECK(cufftExecZ2D(plan,
            reinterpret_cast<cufftDoubleComplex*>(complex_buf.data_ptr()),
            reinterpret_cast<cufftDoubleReal*>(output.data_ptr())));
    }

    // Apply normalization
    double scale = get_norm_factor(n, norm, /*is_forward=*/false);
    apply_normalization_real(output, scale, is_float32, stream);

    return output;
}

// ============================================================================
// 2D FFT: Complex-to-Complex forward
// ============================================================================

auto cuda_fft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                      const std::vector<int64_t>& n_vec,
                      const std::string& norm, cudaStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    bool is_float32 = (input.dtype() == DType::Complex64);

    // For 2D FFT, we require dims to be the last two dimensions for efficient cuFFT usage
    // The dispatch layer should have arranged this.
    if (dims.size() != 2) {
        throw std::runtime_error("cuFFT fft2: expected exactly 2 dimensions");
    }

    // Check if dims are the last two dimensions (most common case)
    bool last_two = (dims[0] == ndim - 2 && dims[1] == ndim - 1);
    if (!last_two) {
        // Fall back to sequential 1D FFTs for non-standard dimension ordering
        Tensor result = input;
        for (size_t i = 0; i < dims.size(); ++i) {
            result = cuda_fft_kernel(result, dims[i], n_vec[i], norm, stream);
        }
        return result;
    }

    int64_t N0 = n_vec[0];  // rows
    int64_t N1 = n_vec[1];  // cols

    // Handle padding/truncation: copy input into properly-sized buffer
    std::vector<int64_t> out_shape = shape;
    out_shape[dims[0]] = N0;
    out_shape[dims[1]] = N1;

    DType out_dtype = input.dtype();
    Tensor output(out_shape, out_dtype, input.device());
    TENZOR_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0,
        output.numel() * dtype_size(out_dtype), stream));

    // Copy input data into output buffer with proper padding/truncation
    int64_t batch = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch *= shape[i];

    int64_t copy_rows = std::min(shape[dims[0]], N0);
    int64_t copy_cols = std::min(shape[dims[1]], N1);
    int64_t elem_size = dtype_size(out_dtype);

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t r = 0; r < copy_rows; ++r) {
            const char* src = static_cast<const char*>(input.data_ptr())
                + (b * shape[dims[0]] * shape[dims[1]] + r * shape[dims[1]]) * elem_size;
            char* dst = static_cast<char*>(output.data_ptr())
                + (b * N0 * N1 + r * N1) * elem_size;
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(dst, src,
                copy_cols * elem_size, cudaMemcpyDeviceToDevice, stream));
        }
    }

    // Create 2D cuFFT plan
    CuFFTPlan plan;
    plan.create();
    plan.set_stream(stream);

    int n_arr[2] = {static_cast<int>(N0), static_cast<int>(N1)};
    cufftType fft_type = is_float32 ? CUFFT_C2C : CUFFT_Z2Z;

    CUFFT_CHECK(cufftPlanMany(&plan.handle, 2, n_arr,
                              nullptr, 1, static_cast<int>(N0 * N1),
                              nullptr, 1, static_cast<int>(N0 * N1),
                              fft_type, static_cast<int>(batch)));
    CUFFT_CHECK(cufftSetStream(plan.handle, stream));

    if (is_float32) {
        CUFFT_CHECK(cufftExecC2C(plan,
            reinterpret_cast<cufftComplex*>(output.data_ptr()),
            reinterpret_cast<cufftComplex*>(output.data_ptr()),
            CUFFT_FORWARD));
    } else {
        CUFFT_CHECK(cufftExecZ2Z(plan,
            reinterpret_cast<cufftDoubleComplex*>(output.data_ptr()),
            reinterpret_cast<cufftDoubleComplex*>(output.data_ptr()),
            CUFFT_FORWARD));
    }

    double scale = get_norm_factor_nd(n_vec, norm, /*is_forward=*/true);
    apply_normalization_complex(output, scale, is_float32, stream);

    return output;
}

// ============================================================================
// 2D IFFT: Complex-to-Complex inverse
// ============================================================================

auto cuda_ifft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& n_vec,
                       const std::string& norm, cudaStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    bool is_float32 = (input.dtype() == DType::Complex64);

    if (dims.size() != 2) {
        throw std::runtime_error("cuFFT ifft2: expected exactly 2 dimensions");
    }

    bool last_two = (dims[0] == ndim - 2 && dims[1] == ndim - 1);
    if (!last_two) {
        Tensor result = input;
        for (size_t i = 0; i < dims.size(); ++i) {
            result = cuda_ifft_kernel(result, dims[i], n_vec[i], norm, stream);
        }
        return result;
    }

    int64_t N0 = n_vec[0];
    int64_t N1 = n_vec[1];

    std::vector<int64_t> out_shape = shape;
    out_shape[dims[0]] = N0;
    out_shape[dims[1]] = N1;

    DType out_dtype = input.dtype();
    Tensor output(out_shape, out_dtype, input.device());
    TENZOR_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0,
        output.numel() * dtype_size(out_dtype), stream));

    int64_t batch = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch *= shape[i];

    int64_t copy_rows = std::min(shape[dims[0]], N0);
    int64_t copy_cols = std::min(shape[dims[1]], N1);
    int64_t elem_size = dtype_size(out_dtype);

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t r = 0; r < copy_rows; ++r) {
            const char* src = static_cast<const char*>(input.data_ptr())
                + (b * shape[dims[0]] * shape[dims[1]] + r * shape[dims[1]]) * elem_size;
            char* dst = static_cast<char*>(output.data_ptr())
                + (b * N0 * N1 + r * N1) * elem_size;
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(dst, src,
                copy_cols * elem_size, cudaMemcpyDeviceToDevice, stream));
        }
    }

    CuFFTPlan plan;
    plan.create();
    plan.set_stream(stream);

    int n_arr[2] = {static_cast<int>(N0), static_cast<int>(N1)};
    cufftType fft_type = is_float32 ? CUFFT_C2C : CUFFT_Z2Z;

    CUFFT_CHECK(cufftPlanMany(&plan.handle, 2, n_arr,
                              nullptr, 1, static_cast<int>(N0 * N1),
                              nullptr, 1, static_cast<int>(N0 * N1),
                              fft_type, static_cast<int>(batch)));
    CUFFT_CHECK(cufftSetStream(plan.handle, stream));

    if (is_float32) {
        CUFFT_CHECK(cufftExecC2C(plan,
            reinterpret_cast<cufftComplex*>(output.data_ptr()),
            reinterpret_cast<cufftComplex*>(output.data_ptr()),
            CUFFT_INVERSE));
    } else {
        CUFFT_CHECK(cufftExecZ2Z(plan,
            reinterpret_cast<cufftDoubleComplex*>(output.data_ptr()),
            reinterpret_cast<cufftDoubleComplex*>(output.data_ptr()),
            CUFFT_INVERSE));
    }

    double scale = get_norm_factor_nd(n_vec, norm, /*is_forward=*/false);
    apply_normalization_complex(output, scale, is_float32, stream);

    return output;
}

// ============================================================================
// N-D FFT: Complex-to-Complex forward
// ============================================================================

auto cuda_fftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                      const std::vector<int64_t>& n_vec,
                      const std::string& norm, cudaStream_t stream) -> Tensor {
    int64_t ndim = static_cast<int64_t>(input.shape().size());
    int64_t rank = static_cast<int64_t>(dims.size());

    // Check if dims are the last `rank` dimensions in order
    bool are_last_dims = true;
    for (int64_t i = 0; i < rank; ++i) {
        if (dims[i] != ndim - rank + i) {
            are_last_dims = false;
            break;
        }
    }

    if (!are_last_dims) {
        // Fallback: sequential 1D FFTs
        Tensor result = input;
        for (int64_t i = 0; i < rank; ++i) {
            result = cuda_fft_kernel(result, dims[i], n_vec[i], norm, stream);
        }
        return result;
    }

    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    bool is_float32 = (input.dtype() == DType::Complex64);

    // Build output shape
    std::vector<int64_t> out_shape = shape;
    for (int64_t i = 0; i < rank; ++i) {
        out_shape[dims[i]] = n_vec[i];
    }

    DType out_dtype = input.dtype();
    Tensor output(out_shape, out_dtype, input.device());
    TENZOR_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0,
        output.numel() * dtype_size(out_dtype), stream));

    // Compute batch size (product of dims before the FFT dims)
    int64_t batch = 1;
    for (int64_t i = 0; i < ndim - rank; ++i) batch *= shape[i];

    // Copy input data into output with proper padding
    int64_t in_fft_size = 1, out_fft_size = 1;
    for (int64_t i = 0; i < rank; ++i) {
        in_fft_size *= shape[dims[i]];
        out_fft_size *= n_vec[i];
    }

    // For N-D copy we need row-by-row copy across last rank dims.
    // Simplest approach: copy the minimum slice for each batch.
    int64_t elem_size = dtype_size(out_dtype);

    // For the common case where in and out FFT sizes match:
    if (in_fft_size == out_fft_size) {
        // Shapes match in FFT dimensions — direct batch copy
        bool shapes_match = true;
        for (int64_t i = 0; i < rank; ++i) {
            if (shape[dims[i]] != n_vec[i]) { shapes_match = false; break; }
        }
        if (shapes_match) {
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(output.data_ptr(), input.data_ptr(),
                input.numel() * elem_size, cudaMemcpyDeviceToDevice, stream));
        }
    }
    // General case: do sequential 1D FFTs (this is simpler and still uses cuFFT)
    if (in_fft_size != out_fft_size) {
        Tensor result = input;
        for (int64_t i = 0; i < rank; ++i) {
            result = cuda_fft_kernel(result, dims[i], n_vec[i], norm, stream);
        }
        return result;
    }

    // Create N-D cuFFT plan
    CuFFTPlan plan;
    plan.create();
    plan.set_stream(stream);

    std::vector<int> n_arr(rank);
    for (int64_t i = 0; i < rank; ++i) {
        n_arr[i] = static_cast<int>(n_vec[i]);
    }

    cufftType fft_type = is_float32 ? CUFFT_C2C : CUFFT_Z2Z;
    CUFFT_CHECK(cufftPlanMany(&plan.handle, static_cast<int>(rank), n_arr.data(),
                              nullptr, 1, static_cast<int>(out_fft_size),
                              nullptr, 1, static_cast<int>(out_fft_size),
                              fft_type, static_cast<int>(batch)));
    CUFFT_CHECK(cufftSetStream(plan.handle, stream));

    if (is_float32) {
        CUFFT_CHECK(cufftExecC2C(plan,
            reinterpret_cast<cufftComplex*>(output.data_ptr()),
            reinterpret_cast<cufftComplex*>(output.data_ptr()),
            CUFFT_FORWARD));
    } else {
        CUFFT_CHECK(cufftExecZ2Z(plan,
            reinterpret_cast<cufftDoubleComplex*>(output.data_ptr()),
            reinterpret_cast<cufftDoubleComplex*>(output.data_ptr()),
            CUFFT_FORWARD));
    }

    double scale = get_norm_factor_nd(n_vec, norm, /*is_forward=*/true);
    apply_normalization_complex(output, scale, is_float32, stream);

    return output;
}

// ============================================================================
// N-D IFFT: Complex-to-Complex inverse
// ============================================================================

auto cuda_ifftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& n_vec,
                       const std::string& norm, cudaStream_t stream) -> Tensor {
    int64_t ndim = static_cast<int64_t>(input.shape().size());
    int64_t rank = static_cast<int64_t>(dims.size());

    bool are_last_dims = true;
    for (int64_t i = 0; i < rank; ++i) {
        if (dims[i] != ndim - rank + i) {
            are_last_dims = false;
            break;
        }
    }

    if (!are_last_dims) {
        Tensor result = input;
        for (int64_t i = 0; i < rank; ++i) {
            result = cuda_ifft_kernel(result, dims[i], n_vec[i], norm, stream);
        }
        return result;
    }

    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    bool is_float32 = (input.dtype() == DType::Complex64);

    std::vector<int64_t> out_shape = shape;
    for (int64_t i = 0; i < rank; ++i) {
        out_shape[dims[i]] = n_vec[i];
    }

    DType out_dtype = input.dtype();
    Tensor output(out_shape, out_dtype, input.device());
    TENZOR_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0,
        output.numel() * dtype_size(out_dtype), stream));

    int64_t batch = 1;
    for (int64_t i = 0; i < ndim - rank; ++i) batch *= shape[i];

    int64_t in_fft_size = 1, out_fft_size = 1;
    for (int64_t i = 0; i < rank; ++i) {
        in_fft_size *= shape[dims[i]];
        out_fft_size *= n_vec[i];
    }

    int64_t elem_size = dtype_size(out_dtype);

    if (in_fft_size == out_fft_size) {
        bool shapes_match = true;
        for (int64_t i = 0; i < rank; ++i) {
            if (shape[dims[i]] != n_vec[i]) { shapes_match = false; break; }
        }
        if (shapes_match) {
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(output.data_ptr(), input.data_ptr(),
                input.numel() * elem_size, cudaMemcpyDeviceToDevice, stream));
        }
    }
    if (in_fft_size != out_fft_size) {
        Tensor result = input;
        for (int64_t i = 0; i < rank; ++i) {
            result = cuda_ifft_kernel(result, dims[i], n_vec[i], norm, stream);
        }
        return result;
    }

    CuFFTPlan plan;
    plan.create();
    plan.set_stream(stream);

    std::vector<int> n_arr(rank);
    for (int64_t i = 0; i < rank; ++i) {
        n_arr[i] = static_cast<int>(n_vec[i]);
    }

    cufftType fft_type = is_float32 ? CUFFT_C2C : CUFFT_Z2Z;
    CUFFT_CHECK(cufftPlanMany(&plan.handle, static_cast<int>(rank), n_arr.data(),
                              nullptr, 1, static_cast<int>(out_fft_size),
                              nullptr, 1, static_cast<int>(out_fft_size),
                              fft_type, static_cast<int>(batch)));
    CUFFT_CHECK(cufftSetStream(plan.handle, stream));

    if (is_float32) {
        CUFFT_CHECK(cufftExecC2C(plan,
            reinterpret_cast<cufftComplex*>(output.data_ptr()),
            reinterpret_cast<cufftComplex*>(output.data_ptr()),
            CUFFT_INVERSE));
    } else {
        CUFFT_CHECK(cufftExecZ2Z(plan,
            reinterpret_cast<cufftDoubleComplex*>(output.data_ptr()),
            reinterpret_cast<cufftDoubleComplex*>(output.data_ptr()),
            CUFFT_INVERSE));
    }

    double scale = get_norm_factor_nd(n_vec, norm, /*is_forward=*/false);
    apply_normalization_complex(output, scale, is_float32, stream);

    return output;
}

} // namespace cuda
} // namespace tenzor

#else // !TENZOR_HAS_CUFFT — Cooley-Tukey (power-of-2) + Bluestein (general) FFT fallback
#pragma message("WARNING: Building without cuFFT — using slower native CUDA FFT fallback")

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "cuda_common.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>

namespace tenzor {
namespace cuda {

namespace {

// ============================================================================
// Helper functions
// ============================================================================

inline bool is_power_of_2(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

inline int64_t next_pow2(int64_t n) {
    int64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// ============================================================================
// RAII wrapper for CUDA device memory
// ============================================================================

// HH.8: CudaDevicePtr now uses cudaMallocAsync/cudaFreeAsync against the
// captured stream so Bluestein workspaces don't force a device-wide sync
// per FFT call. Plain cudaFree implicitly synchronises the whole device,
// which defeats stream concurrency. Every FFT entry point (cuda_fft_kernel,
// cuda_ifft_kernel, cuda_rfft_kernel, cuda_irfft_kernel, and the
// bluestein_*_cuda helpers) plumbs the active stream through to here.
template<typename T>
struct CudaDevicePtr {
    T* ptr = nullptr;
    cudaStream_t stream = nullptr;

    CudaDevicePtr() = default;
    CudaDevicePtr(int64_t count, cudaStream_t s)
        : stream(s) {
        // cudaMallocAsync requires a stream-ordered memory pool. The default
        // pool is created on demand in driver runtime API >= 11.2.
        TENZOR_CUDA_CHECK(cudaMallocAsync(&ptr, count * sizeof(T), stream));
    }
    ~CudaDevicePtr() {
        if (ptr) {
            // cudaFreeAsync defers the free until prior work on `stream`
            // completes; no device-wide sync.
            cudaFreeAsync(ptr, stream);
        }
    }

    CudaDevicePtr(const CudaDevicePtr&) = delete;
    CudaDevicePtr& operator=(const CudaDevicePtr&) = delete;
    CudaDevicePtr(CudaDevicePtr&& o) noexcept
        : ptr(o.ptr), stream(o.stream) { o.ptr = nullptr; }

    T* get() const { return ptr; }
};

// ============================================================================
// Normalization helpers (same as cuFFT path)
// ============================================================================

double get_norm_factor(int64_t n, const std::string& norm, bool is_forward) {
    if (norm == "ortho") {
        return 1.0 / std::sqrt(static_cast<double>(n));
    } else if ((norm == "forward" && is_forward) || (norm == "backward" && !is_forward)) {
        return 1.0 / static_cast<double>(n);
    }
    return 1.0;
}

double get_norm_factor_nd(const std::vector<int64_t>& n_vec, const std::string& norm, bool is_forward) {
    double factor = 1.0;
    for (auto n : n_vec) {
        factor *= get_norm_factor(n, norm, is_forward);
    }
    return factor;
}

// ============================================================================
// Scale kernel
// ============================================================================

template<typename T>
__global__ void native_scale_kernel(T* data, int64_t numel, T scale) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] *= scale;
    }
}

template<typename T>
void launch_scale(T* data, int64_t numel, T scale, cudaStream_t stream) {
    if (numel == 0) return;
    constexpr int block = 256;
    int grid = static_cast<int>((numel + block - 1) / block);
    native_scale_kernel<T><<<grid, block, 0, stream>>>(data, numel, scale);
    TENZOR_CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Cooley-Tukey FFT CUDA kernels
// ============================================================================

/// Bit-reverse permutation kernel for batched interleaved complex data.
/// data layout: batch_size blocks of batch_stride floats each.
/// Each block contains N complex elements as [re0,im0,re1,im1,...].
template<typename T>
__global__ void bit_reverse_permutation_kernel(T* data, int64_t N, int64_t batch_size,
                                                int64_t batch_stride, int bits) {
    int64_t global_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_items = N * batch_size;
    if (global_id >= total_items) return;

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
        data[base + 2 * i]     = data[base + 2 * j];
        data[base + 2 * i + 1] = data[base + 2 * j + 1];
        data[base + 2 * j]     = tmp_re;
        data[base + 2 * j + 1] = tmp_im;
    }
}

/// Butterfly stage kernel. One thread per butterfly operation across all batches.
template<typename T>
__global__ void butterfly_stage_kernel(T* data, int64_t N, int64_t batch_size,
                                       int64_t batch_stride, int64_t stride, int64_t half,
                                       int64_t num_butterflies, T sign) {
    int64_t global_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = num_butterflies * batch_size;
    if (global_id >= total) return;

    int64_t batch_idx = global_id / num_butterflies;
    int64_t flat = global_id % num_butterflies;
    int64_t group = flat / half;
    int64_t k = flat % half;
    int64_t base_idx = group * stride;

    constexpr T PI = static_cast<T>(3.14159265358979323846);
    T angle = sign * static_cast<T>(2.0) * PI * static_cast<T>(k) / static_cast<T>(stride);
    T w_re, w_im;
    // Use sincos for efficiency; sincos is available as __sincosf/__sincos on device
    // but cos/sin work fine and the compiler will typically fuse them.
    w_re = cos(angle);
    w_im = sin(angle);

    int64_t base = batch_idx * batch_stride;
    int64_t even_i = base_idx + k;
    int64_t odd_i = base_idx + k + half;

    T e_re = data[base + 2 * even_i];
    T e_im = data[base + 2 * even_i + 1];
    T o_re = data[base + 2 * odd_i];
    T o_im = data[base + 2 * odd_i + 1];

    T t_re = w_re * o_re - w_im * o_im;
    T t_im = w_re * o_im + w_im * o_re;

    data[base + 2 * even_i]     = e_re + t_re;
    data[base + 2 * even_i + 1] = e_im + t_im;
    data[base + 2 * odd_i]      = e_re - t_re;
    data[base + 2 * odd_i + 1]  = e_im - t_im;
}

/// Host function: launch Cooley-Tukey FFT on interleaved complex data.
/// data: device pointer, batch_size independent FFTs, each N complex elements,
/// separated by batch_stride T values. sign = -1 for forward, +1 for inverse.
template<typename T>
void cooley_tukey_fft_cuda(T* data, int64_t N, int64_t batch_size, int64_t batch_stride,
                           T sign, cudaStream_t stream) {
    int log2N = 0;
    { int64_t tmp = N; while (tmp > 1) { tmp >>= 1; log2N++; } }

    constexpr int block = 256;

    // Step 1: Bit-reverse permutation
    {
        int64_t total = N * batch_size;
        int grid = static_cast<int>((total + block - 1) / block);
        bit_reverse_permutation_kernel<T><<<grid, block, 0, stream>>>(
            data, N, batch_size, batch_stride, log2N);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: Butterfly stages
    int64_t num_butterflies = N / 2;
    int64_t total_butterflies = num_butterflies * batch_size;
    int grid = static_cast<int>((total_butterflies + block - 1) / block);

    for (int s = 1; s <= log2N; ++s) {
        int64_t stride = static_cast<int64_t>(1) << s;
        int64_t half = stride / 2;

        butterfly_stage_kernel<T><<<grid, block, 0, stream>>>(
            data, N, batch_size, batch_stride, stride, half, num_butterflies, sign);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// Bluestein FFT helper kernels
// ============================================================================

/// Generate chirp: chirp[k] = exp(sign * j * pi * k^2 / N)
template<typename T>
__global__ void generate_chirp_kernel(T* chirp, int64_t N, T angle_sign) {
    int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= N) return;
    constexpr T PI = static_cast<T>(3.14159265358979323846);
    T angle = angle_sign * PI * static_cast<T>(k) * static_cast<T>(k) / static_cast<T>(N);
    chirp[2 * k]     = cos(angle);
    chirp[2 * k + 1] = sin(angle);
}

/// Build convolution kernel b: b[k] = conj(chirp[k]) for k=0..N-1
template<typename T>
__global__ void build_b_kernel(T* b_buf, const T* chirp, int64_t N) {
    int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= N) return;
    b_buf[2 * k]     = chirp[2 * k];
    b_buf[2 * k + 1] = -chirp[2 * k + 1];
}

/// Build wrap-around part: b[M-k] = conj(chirp[k]) for k=1..N-1
template<typename T>
__global__ void build_b_wrap_kernel(T* b_buf, const T* chirp, int64_t N, int64_t M) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N - 1) return;
    int64_t k = idx + 1;
    int64_t m_idx = M - k;
    b_buf[2 * m_idx]     = chirp[2 * k];
    b_buf[2 * m_idx + 1] = -chirp[2 * k + 1];
}

/// Build a_buf for real input Bluestein: a[s][k] = x[b,k,inner] * chirp[k]
template<typename T>
__global__ void bluestein_build_a_real_kernel(T* a_buf, const T* d_in, const T* chirp,
                                              int64_t N, int64_t M, int64_t total_slices,
                                              int64_t inner_size) {
    int64_t global_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = N * total_slices;
    if (global_id >= total) return;

    int64_t s = global_id / N;
    int64_t k = global_id % N;
    int64_t b = s / inner_size;
    int64_t inner = s % inner_size;
    int64_t in_idx = b * N * inner_size + k * inner_size + inner;
    T val = d_in[in_idx];
    int64_t a_base = s * 2 * M;
    a_buf[a_base + 2 * k]     = val * chirp[2 * k];
    a_buf[a_base + 2 * k + 1] = val * chirp[2 * k + 1];
}

/// Build a_buf for complex input Bluestein: a[s][k] = x[b,k,inner] * chirp[k] (complex mult)
template<typename T>
__global__ void bluestein_build_a_complex_kernel(T* a_buf, const T* d_in, const T* chirp,
                                                  int64_t N, int64_t M, int64_t total_slices,
                                                  int64_t inner_size) {
    int64_t global_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = N * total_slices;
    if (global_id >= total) return;

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
}

/// Pointwise complex multiply: A[s][k] *= B[k]
template<typename T>
__global__ void pointwise_complex_mul_kernel(T* a_buf, const T* B_buf,
                                              int64_t M, int64_t total_slices) {
    int64_t global_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = M * total_slices;
    if (global_id >= total) return;

    int64_t s = global_id / M;
    int64_t k = global_id % M;
    int64_t a_base = s * 2 * M;
    T a_re = a_buf[a_base + 2 * k];
    T a_im = a_buf[a_base + 2 * k + 1];
    T b_re = B_buf[2 * k];
    T b_im = B_buf[2 * k + 1];
    a_buf[a_base + 2 * k]     = a_re * b_re - a_im * b_im;
    a_buf[a_base + 2 * k + 1] = a_re * b_im + a_im * b_re;
}

/// Extract Bluestein result (real input): out[b,k,inner] = a[s][k] * conj(chirp[k])
template<typename T>
__global__ void bluestein_extract_real_kernel(T* d_out, const T* a_buf, const T* chirp,
                                              int64_t N, int64_t M, int64_t total_slices,
                                              int64_t inner_size) {
    int64_t global_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = N * total_slices;
    if (global_id >= total) return;

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
}

/// Extract Bluestein result (complex input): same layout
template<typename T>
__global__ void bluestein_extract_complex_kernel(T* d_out, const T* a_buf, const T* chirp,
                                                  int64_t N, int64_t M, int64_t total_slices,
                                                  int64_t inner_size) {
    int64_t global_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = N * total_slices;
    if (global_id >= total) return;

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
}

// ============================================================================
// Bluestein FFT — real input
// ============================================================================

/// Bluestein FFT for non-power-of-2 sizes with real input.
/// Input:  d_in  — real, layout [batch_size, signal_len, inner_size]
/// Output: d_out — interleaved complex, layout [batch_size, signal_len, inner_size, 2]
template<typename T>
void bluestein_fft_cuda(const T* d_in, T* d_out,
                        int64_t signal_len, int64_t batch_size, int64_t inner_size,
                        cudaStream_t stream) {
    const int64_t N = signal_len;
    const int64_t M = next_pow2(2 * N - 1);
    constexpr int block = 256;

    int64_t total_slices = batch_size * inner_size;

    // Allocate workspace
    CudaDevicePtr<T> chirp_owner(2 * N, stream);
    T* chirp = chirp_owner.get();
    CudaDevicePtr<T> b_buf_owner(2 * M, stream);
    T* b_buf = b_buf_owner.get();
    CudaDevicePtr<T> B_buf_owner(2 * M, stream);
    T* B_buf = B_buf_owner.get();
    CudaDevicePtr<T> a_buf_owner(2 * M * total_slices, stream);
    T* a_buf = a_buf_owner.get();

    // Step 1: Generate chirp: chirp[k] = exp(-j * pi * k^2 / N)
    {
        int grid = static_cast<int>((N + block - 1) / block);
        generate_chirp_kernel<T><<<grid, block, 0, stream>>>(chirp, N, static_cast<T>(-1.0));
        TENZOR_CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: Build convolution kernel b
    TENZOR_CUDA_CHECK(cudaMemsetAsync(b_buf, 0, 2 * M * sizeof(T), stream));
    {
        int grid = static_cast<int>((N + block - 1) / block);
        build_b_kernel<T><<<grid, block, 0, stream>>>(b_buf, chirp, N);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    }
    if (N > 1) {
        int grid = static_cast<int>((N - 1 + block - 1) / block);
        build_b_wrap_kernel<T><<<grid, block, 0, stream>>>(b_buf, chirp, N, M);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: B = FFT(b)
    TENZOR_CUDA_CHECK(cudaMemcpyAsync(B_buf, b_buf, 2 * M * sizeof(T),
                                       cudaMemcpyDeviceToDevice, stream));
    cooley_tukey_fft_cuda(B_buf, M, int64_t(1), int64_t(2 * M),
                          static_cast<T>(-1.0), stream);

    // Step 4: Build a_buf: a[s][k] = x[b,k,inner] * chirp[k]
    TENZOR_CUDA_CHECK(cudaMemsetAsync(a_buf, 0, 2 * M * total_slices * sizeof(T), stream));
    {
        int64_t total = N * total_slices;
        int grid = static_cast<int>((total + block - 1) / block);
        bluestein_build_a_real_kernel<T><<<grid, block, 0, stream>>>(
            a_buf, d_in, chirp, N, M, total_slices, inner_size);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    }

    // Step 5: A = FFT(a) — batched
    cooley_tukey_fft_cuda(a_buf, M, total_slices, int64_t(2 * M),
                          static_cast<T>(-1.0), stream);

    // Step 6: Pointwise multiply A *= B
    {
        int64_t total = M * total_slices;
        int grid = static_cast<int>((total + block - 1) / block);
        pointwise_complex_mul_kernel<T><<<grid, block, 0, stream>>>(a_buf, B_buf, M, total_slices);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    }

    // Step 7: IFFT via forward FFT with sign=+1, divide by M
    cooley_tukey_fft_cuda(a_buf, M, total_slices, int64_t(2 * M),
                          static_cast<T>(1.0), stream);
    {
        T inv_M = static_cast<T>(1.0) / static_cast<T>(M);
        int64_t total = 2 * M * total_slices;
        launch_scale(a_buf, total, inv_M, stream);
    }

    // Step 8: Extract result: out[b,k,inner] = a[s][k] * conj(chirp[k])
    {
        int64_t total = N * total_slices;
        int grid = static_cast<int>((total + block - 1) / block);
        bluestein_extract_real_kernel<T><<<grid, block, 0, stream>>>(
            d_out, a_buf, chirp, N, M, total_slices, inner_size);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    }
    // HH.8: no cudaStreamSynchronize — workspaces tear down via
    // cudaFreeAsync on `stream` once Step 8 retires.
}

// ============================================================================
// Bluestein FFT — complex input
// ============================================================================

/// Bluestein FFT for non-power-of-2 sizes with complex input.
/// Input:  d_in  — interleaved complex, layout [batch_size, signal_len, inner_size, 2]
/// Output: d_out — interleaved complex, layout [batch_size, signal_len, inner_size, 2]
/// sign = -1 for forward, +1 for inverse (before normalization).
template<typename T>
void bluestein_fft_complex_cuda(const T* d_in, T* d_out,
                                int64_t signal_len, int64_t batch_size, int64_t inner_size,
                                T sign, cudaStream_t stream) {
    const int64_t N = signal_len;
    const int64_t M = next_pow2(2 * N - 1);
    constexpr int block = 256;

    int64_t total_slices = batch_size * inner_size;

    CudaDevicePtr<T> chirp_owner(2 * N, stream);
    T* chirp = chirp_owner.get();
    CudaDevicePtr<T> b_buf_owner(2 * M, stream);
    T* b_buf = b_buf_owner.get();
    CudaDevicePtr<T> B_buf_owner(2 * M, stream);
    T* B_buf = B_buf_owner.get();
    CudaDevicePtr<T> a_buf_owner(2 * M * total_slices, stream);
    T* a_buf = a_buf_owner.get();

    // Step 1: chirp[k] = exp(sign * j * pi * k^2 / N)
    {
        int grid = static_cast<int>((N + block - 1) / block);
        generate_chirp_kernel<T><<<grid, block, 0, stream>>>(chirp, N, sign);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: b[k] = conj(chirp[k])
    TENZOR_CUDA_CHECK(cudaMemsetAsync(b_buf, 0, 2 * M * sizeof(T), stream));
    {
        int grid = static_cast<int>((N + block - 1) / block);
        build_b_kernel<T><<<grid, block, 0, stream>>>(b_buf, chirp, N);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    }
    if (N > 1) {
        int grid = static_cast<int>((N - 1 + block - 1) / block);
        build_b_wrap_kernel<T><<<grid, block, 0, stream>>>(b_buf, chirp, N, M);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: B = FFT(b)
    TENZOR_CUDA_CHECK(cudaMemcpyAsync(B_buf, b_buf, 2 * M * sizeof(T),
                                       cudaMemcpyDeviceToDevice, stream));
    cooley_tukey_fft_cuda(B_buf, M, int64_t(1), int64_t(2 * M),
                          static_cast<T>(-1.0), stream);

    // Step 4: a[s][k] = x[b,k,inner] * chirp[k] (complex multiply)
    TENZOR_CUDA_CHECK(cudaMemsetAsync(a_buf, 0, 2 * M * total_slices * sizeof(T), stream));
    {
        int64_t total = N * total_slices;
        int grid = static_cast<int>((total + block - 1) / block);
        bluestein_build_a_complex_kernel<T><<<grid, block, 0, stream>>>(
            a_buf, d_in, chirp, N, M, total_slices, inner_size);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    }

    // Step 5: A = FFT(a)
    cooley_tukey_fft_cuda(a_buf, M, total_slices, int64_t(2 * M),
                          static_cast<T>(-1.0), stream);

    // Step 6: Pointwise multiply A *= B
    {
        int64_t total = M * total_slices;
        int grid = static_cast<int>((total + block - 1) / block);
        pointwise_complex_mul_kernel<T><<<grid, block, 0, stream>>>(a_buf, B_buf, M, total_slices);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    }

    // Step 7: IFFT via forward FFT with sign=+1, divide by M
    cooley_tukey_fft_cuda(a_buf, M, total_slices, int64_t(2 * M),
                          static_cast<T>(1.0), stream);
    {
        T inv_M = static_cast<T>(1.0) / static_cast<T>(M);
        int64_t total = 2 * M * total_slices;
        launch_scale(a_buf, total, inv_M, stream);
    }

    // Step 8: Extract result
    {
        int64_t total = N * total_slices;
        int grid = static_cast<int>((total + block - 1) / block);
        bluestein_extract_complex_kernel<T><<<grid, block, 0, stream>>>(
            d_out, a_buf, chirp, N, M, total_slices, inner_size);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    }
    // HH.8: no cudaStreamSynchronize — workspaces tear down via
    // cudaFreeAsync on `stream` once Step 8 retires.
}

// ============================================================================
// Helper kernels for the wrapper functions
// ============================================================================

/// Pack real data into interleaved complex: d_buf[2*i] = d_in[i], d_buf[2*i+1] = 0
template<typename T>
__global__ void pack_real_to_complex_kernel(T* d_buf, const T* d_in, int64_t total) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    d_buf[2 * i]     = d_in[i];
    d_buf[2 * i + 1] = static_cast<T>(0);
}

/// Truncate: copy first out_len complex bins per batch from d_buf to d_out
template<typename T>
__global__ void truncate_rfft_kernel(T* d_out, const T* d_buf,
                                     int64_t out_len, int64_t signal_len,
                                     int64_t batch_size) {
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= batch_size * out_len) return;
    int64_t b = flat / out_len;
    int64_t k = flat % out_len;
    int64_t src = b * 2 * signal_len + 2 * k;
    int64_t dst = (b * out_len + k) * 2;
    d_out[dst]     = d_buf[src];
    d_out[dst + 1] = d_buf[src + 1];
}

/// Truncate for Bluestein rfft (with inner_size)
template<typename T>
__global__ void truncate_rfft_bluestein_kernel(T* d_out, const T* d_full,
                                                int64_t out_len, int64_t signal_len,
                                                int64_t batch_size, int64_t inner_size) {
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch_size * out_len * inner_size;
    if (flat >= total) return;
    int64_t b = flat / (out_len * inner_size);
    int64_t rem = flat % (out_len * inner_size);
    int64_t k = rem / inner_size;
    int64_t inner = rem % inner_size;
    int64_t src = (b * signal_len * inner_size + k * inner_size + inner) * 2;
    int64_t dst = (b * out_len * inner_size + k * inner_size + inner) * 2;
    d_out[dst]     = d_full[src];
    d_out[dst + 1] = d_full[src + 1];
}

/// Reconstruct full N-point spectrum from N/2+1 bins using conjugate symmetry (Cooley-Tukey path)
template<typename T>
__global__ void reconstruct_spectrum_kernel(T* d_buf, const T* d_in,
                                             int64_t output_len, int64_t complex_len,
                                             int64_t batch_size) {
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= batch_size * output_len) return;
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
}

/// Reconstruct full spectrum with inner_size support (Bluestein path)
template<typename T>
__global__ void reconstruct_spectrum_inner_kernel(T* d_full, const T* d_in,
                                                   int64_t output_len, int64_t complex_len,
                                                   int64_t batch_size, int64_t inner_size) {
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch_size * output_len * inner_size;
    if (flat >= total) return;
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
}

/// Extract real part with scaling (Cooley-Tukey irfft)
template<typename T>
__global__ void extract_real_scaled_kernel(T* d_out, const T* d_buf,
                                            int64_t output_len, int64_t batch_size, T scale) {
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= batch_size * output_len) return;
    int64_t b = flat / output_len;
    int64_t t = flat % output_len;
    d_out[b * output_len + t] = d_buf[b * 2 * output_len + 2 * t] * scale;
}

/// Extract real part with scaling and inner_size (Bluestein irfft)
template<typename T>
__global__ void extract_real_scaled_inner_kernel(T* d_out, const T* d_ifft,
                                                  int64_t output_len, int64_t batch_size,
                                                  int64_t inner_size, int64_t out_numel, T scale) {
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= out_numel) return;
    int64_t b = flat / (output_len * inner_size);
    int64_t rem = flat % (output_len * inner_size);
    int64_t t = rem / inner_size;
    int64_t inner = rem % inner_size;
    int64_t src = (b * output_len * inner_size + t * inner_size + inner) * 2;
    d_out[flat] = d_ifft[src] * scale;
}

} // anonymous namespace

// ============================================================================
// 1D FFT: Complex-to-Complex forward (native CUDA fallback)
//
// The cuFFT path uses Complex64/Complex128 dtypes where each element is
// internally stored as [re, im]. The native fallback must match this interface.
// ============================================================================

auto cuda_fft_kernel(const Tensor& input, int64_t dim, int64_t n,
                     const std::string& norm, cudaStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("FFT: dimension out of range");
    }
    bool is_float32 = (input.dtype() == DType::Complex64);

    int64_t N_in = shape[dim];
    int64_t N_out = n;

    // Build output shape
    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = N_out;

    DType out_dtype = input.dtype();
    Tensor output(out_shape, out_dtype, input.device());

    // Handle padding/truncation: copy input into output buffer
    Tensor work_input = input;
    if (N_in != N_out) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0,
            output.numel() * dtype_size(out_dtype), stream));

        int64_t copy_len = std::min(N_in, N_out);
        int64_t elem_size = dtype_size(out_dtype);
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            const char* src = static_cast<const char*>(input.data_ptr())
                + outer * N_in * inner_size * elem_size;
            char* dst = static_cast<char*>(output.data_ptr())
                + outer * N_out * inner_size * elem_size;
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(dst, src,
                copy_len * inner_size * elem_size,
                cudaMemcpyDeviceToDevice, stream));
        }
        work_input = output;
    } else {
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(output.data_ptr(), input.data_ptr(),
            input.numel() * dtype_size(out_dtype),
            cudaMemcpyDeviceToDevice, stream));
        work_input = output;
    }

    // Compute layout for FFT dispatch
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= out_shape[i];
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= out_shape[i];
    int64_t batch = outer_size * inner_size;

    // Complex64 stores [re,im] as 2 floats per element, Complex128 as 2 doubles.
    // The data is already interleaved in memory. We treat it as T* with stride 2 per complex.
    // For Cooley-Tukey: need N to be power of 2 and contiguous along dim (inner_size == 1).
    bool use_cooley_tukey = is_power_of_2(N_out) && inner_size == 1;

    if (use_cooley_tukey) {
        // Data is [outer_size, N_out, 2_reals] in memory when inner_size==1
        // batch_stride = 2 * N_out (floats per batch)
        if (is_float32) {
            cooley_tukey_fft_cuda(reinterpret_cast<float*>(output.data_ptr()),
                                  N_out, outer_size, int64_t(2 * N_out),
                                  -1.0f, stream);
        } else {
            cooley_tukey_fft_cuda(reinterpret_cast<double*>(output.data_ptr()),
                                  N_out, outer_size, int64_t(2 * N_out),
                                  -1.0, stream);
        }
    } else {
        // Bluestein for non-power-of-2 or non-contiguous. Use complex-input variant.
        // We need to treat the data as interleaved complex with inner_size stride.
        if (is_float32) {
            float* data_ptr = reinterpret_cast<float*>(output.data_ptr());
            CudaDevicePtr<float> tmp_out(output.numel() * 2, stream);
            bluestein_fft_complex_cuda(data_ptr, tmp_out.get(),
                                       N_out, outer_size, inner_size,
                                       -1.0f, stream);
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(data_ptr, tmp_out.get(),
                output.numel() * 2 * sizeof(float),
                cudaMemcpyDeviceToDevice, stream));
        } else {
            double* data_ptr = reinterpret_cast<double*>(output.data_ptr());
            CudaDevicePtr<double> tmp_out(output.numel() * 2, stream);
            bluestein_fft_complex_cuda(data_ptr, tmp_out.get(),
                                       N_out, outer_size, inner_size,
                                       -1.0, stream);
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(data_ptr, tmp_out.get(),
                output.numel() * 2 * sizeof(double),
                cudaMemcpyDeviceToDevice, stream));
        }
    }

    // Apply normalization
    double scale = get_norm_factor(N_out, norm, /*is_forward=*/true);
    if (scale != 1.0) {
        int64_t total_reals = output.numel() * 2;
        if (is_float32) {
            launch_scale(reinterpret_cast<float*>(output.data_ptr()),
                         total_reals, static_cast<float>(scale), stream);
        } else {
            launch_scale(reinterpret_cast<double*>(output.data_ptr()),
                         total_reals, scale, stream);
        }
    }

    return output;
}

// ============================================================================
// 1D IFFT: Complex-to-Complex inverse (native CUDA fallback)
// ============================================================================

auto cuda_ifft_kernel(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm, cudaStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("IFFT: dimension out of range");
    }
    bool is_float32 = (input.dtype() == DType::Complex64);

    int64_t N_in = shape[dim];
    int64_t N_out = n;

    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = N_out;

    DType out_dtype = input.dtype();
    Tensor output(out_shape, out_dtype, input.device());

    // Handle padding/truncation
    if (N_in != N_out) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0,
            output.numel() * dtype_size(out_dtype), stream));

        int64_t copy_len = std::min(N_in, N_out);
        int64_t elem_size = dtype_size(out_dtype);
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            const char* src = static_cast<const char*>(input.data_ptr())
                + outer * N_in * inner_size * elem_size;
            char* dst = static_cast<char*>(output.data_ptr())
                + outer * N_out * inner_size * elem_size;
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(dst, src,
                copy_len * inner_size * elem_size,
                cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(output.data_ptr(), input.data_ptr(),
            input.numel() * dtype_size(out_dtype),
            cudaMemcpyDeviceToDevice, stream));
    }

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= out_shape[i];
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= out_shape[i];

    bool use_cooley_tukey = is_power_of_2(N_out) && inner_size == 1;

    if (use_cooley_tukey) {
        if (is_float32) {
            cooley_tukey_fft_cuda(reinterpret_cast<float*>(output.data_ptr()),
                                  N_out, outer_size, int64_t(2 * N_out),
                                  1.0f, stream);
        } else {
            cooley_tukey_fft_cuda(reinterpret_cast<double*>(output.data_ptr()),
                                  N_out, outer_size, int64_t(2 * N_out),
                                  1.0, stream);
        }
    } else {
        if (is_float32) {
            float* data_ptr = reinterpret_cast<float*>(output.data_ptr());
            CudaDevicePtr<float> tmp_out(output.numel() * 2, stream);
            bluestein_fft_complex_cuda(data_ptr, tmp_out.get(),
                                       N_out, outer_size, inner_size,
                                       1.0f, stream);
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(data_ptr, tmp_out.get(),
                output.numel() * 2 * sizeof(float),
                cudaMemcpyDeviceToDevice, stream));
        } else {
            double* data_ptr = reinterpret_cast<double*>(output.data_ptr());
            CudaDevicePtr<double> tmp_out(output.numel() * 2, stream);
            bluestein_fft_complex_cuda(data_ptr, tmp_out.get(),
                                       N_out, outer_size, inner_size,
                                       1.0, stream);
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(data_ptr, tmp_out.get(),
                output.numel() * 2 * sizeof(double),
                cudaMemcpyDeviceToDevice, stream));
        }
    }

    // IFFT normalization: "backward" (default) = 1/N, "ortho" = 1/sqrt(N), "forward" = 1
    double scale = get_norm_factor(N_out, norm, /*is_forward=*/false);
    if (scale != 1.0) {
        int64_t total_reals = output.numel() * 2;
        if (is_float32) {
            launch_scale(reinterpret_cast<float*>(output.data_ptr()),
                         total_reals, static_cast<float>(scale), stream);
        } else {
            launch_scale(reinterpret_cast<double*>(output.data_ptr()),
                         total_reals, scale, stream);
        }
    }

    return output;
}

// ============================================================================
// 1D RFFT: Real-to-Complex forward (native CUDA fallback)
// ============================================================================

auto cuda_rfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm, cudaStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("RFFT: dimension out of range");
    }
    bool is_float32 = (input.dtype() == DType::Float32);

    int64_t N_in = shape[dim];
    int64_t N_out_complex = n / 2 + 1;

    if (dim != ndim - 1) {
        throw std::runtime_error(
            "cuda native rfft: only last-dimension FFT is supported. "
            "The dispatch layer should decompose non-last-dim rfft.");
    }

    // Prepare real input buffer (padded or truncated to length n)
    std::vector<int64_t> real_shape = shape;
    real_shape[dim] = n;
    DType real_dtype = input.dtype();

    Tensor real_buf(real_shape, real_dtype, input.device());
    if (N_in != n) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(real_buf.data_ptr(), 0,
            real_buf.numel() * dtype_size(real_dtype), stream));
        int64_t copy_len = std::min(N_in, n);
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
        int64_t elem_size = dtype_size(real_dtype);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            const char* src = static_cast<const char*>(input.data_ptr())
                + outer * N_in * elem_size;
            char* dst = static_cast<char*>(real_buf.data_ptr())
                + outer * n * elem_size;
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(dst, src,
                copy_len * elem_size, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(real_buf.data_ptr(), input.data_ptr(),
            input.numel() * dtype_size(real_dtype),
            cudaMemcpyDeviceToDevice, stream));
    }

    // Output: complex, with shape[dim] = n/2 + 1
    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = N_out_complex;
    DType complex_dtype = is_float32 ? DType::Complex64 : DType::Complex128;
    Tensor output(out_shape, complex_dtype, input.device());

    int64_t batch = 1;
    for (int64_t i = 0; i < dim; ++i) batch *= shape[i];

    // Strategy: compute full N-point FFT on real data, then truncate to N/2+1 bins.
    bool use_cooley_tukey = is_power_of_2(n);
    constexpr int block_size = 256;

    if (is_float32) {
        if (use_cooley_tukey) {
            // Pack real into interleaved complex
            CudaDevicePtr<float> d_buf(2 * n * batch, stream);
            int64_t total = n * batch;
            {
                int grid = static_cast<int>((total + block_size - 1) / block_size);
                pack_real_to_complex_kernel<float><<<grid, block_size, 0, stream>>>(
                    d_buf.get(), static_cast<const float*>(real_buf.data_ptr()), total);
                TENZOR_CUDA_CHECK(cudaGetLastError());
            }

            cooley_tukey_fft_cuda(d_buf.get(), n, batch, int64_t(2 * n), -1.0f, stream);

            // Apply normalization to full FFT before truncation
            double scale = get_norm_factor(n, norm, true);
            if (scale != 1.0) {
                launch_scale(d_buf.get(), 2 * total, static_cast<float>(scale), stream);
            }

            // Truncate to first N/2+1 bins
            {
                int64_t trunc_total = batch * N_out_complex;
                int grid = static_cast<int>((trunc_total + block_size - 1) / block_size);
                truncate_rfft_kernel<float><<<grid, block_size, 0, stream>>>(
                    reinterpret_cast<float*>(output.data_ptr()),
                    d_buf.get(), N_out_complex, n, batch);
                TENZOR_CUDA_CHECK(cudaGetLastError());
            }
        } else {
            // Bluestein: compute full N-point FFT on real data, then truncate
            int64_t full_complex_numel = batch * n * 2;  // inner_size=1 for last dim
            CudaDevicePtr<float> d_full(full_complex_numel, stream);
            TENZOR_CUDA_CHECK(cudaMemsetAsync(d_full.get(), 0,
                full_complex_numel * sizeof(float), stream));

            bluestein_fft_cuda(static_cast<const float*>(real_buf.data_ptr()),
                               d_full.get(), n, batch, int64_t(1), stream);

            double scale = get_norm_factor(n, norm, true);
            if (scale != 1.0) {
                launch_scale(d_full.get(), full_complex_numel, static_cast<float>(scale), stream);
            }

            // Truncate to N/2+1
            {
                int64_t trunc_total = batch * N_out_complex;
                int grid = static_cast<int>((trunc_total + block_size - 1) / block_size);
                truncate_rfft_kernel<float><<<grid, block_size, 0, stream>>>(
                    reinterpret_cast<float*>(output.data_ptr()),
                    d_full.get(), N_out_complex, n, batch);
                TENZOR_CUDA_CHECK(cudaGetLastError());
            }
        }
    } else {
        // Float64 path
        if (use_cooley_tukey) {
            CudaDevicePtr<double> d_buf(2 * n * batch, stream);
            int64_t total = n * batch;
            {
                int grid = static_cast<int>((total + block_size - 1) / block_size);
                pack_real_to_complex_kernel<double><<<grid, block_size, 0, stream>>>(
                    d_buf.get(), static_cast<const double*>(real_buf.data_ptr()), total);
                TENZOR_CUDA_CHECK(cudaGetLastError());
            }

            cooley_tukey_fft_cuda(d_buf.get(), n, batch, int64_t(2 * n), -1.0, stream);

            double scale = get_norm_factor(n, norm, true);
            if (scale != 1.0) {
                launch_scale(d_buf.get(), 2 * total, scale, stream);
            }

            {
                int64_t trunc_total = batch * N_out_complex;
                int grid = static_cast<int>((trunc_total + block_size - 1) / block_size);
                truncate_rfft_kernel<double><<<grid, block_size, 0, stream>>>(
                    reinterpret_cast<double*>(output.data_ptr()),
                    d_buf.get(), N_out_complex, n, batch);
                TENZOR_CUDA_CHECK(cudaGetLastError());
            }
        } else {
            int64_t full_complex_numel = batch * n * 2;
            CudaDevicePtr<double> d_full(full_complex_numel, stream);
            TENZOR_CUDA_CHECK(cudaMemsetAsync(d_full.get(), 0,
                full_complex_numel * sizeof(double), stream));

            bluestein_fft_cuda(static_cast<const double*>(real_buf.data_ptr()),
                               d_full.get(), n, batch, int64_t(1), stream);

            double scale = get_norm_factor(n, norm, true);
            if (scale != 1.0) {
                launch_scale(d_full.get(), full_complex_numel, scale, stream);
            }

            {
                int64_t trunc_total = batch * N_out_complex;
                int grid = static_cast<int>((trunc_total + block_size - 1) / block_size);
                truncate_rfft_kernel<double><<<grid, block_size, 0, stream>>>(
                    reinterpret_cast<double*>(output.data_ptr()),
                    d_full.get(), N_out_complex, n, batch);
                TENZOR_CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    return output;
}

// ============================================================================
// 1D IRFFT: Complex-to-Real inverse (native CUDA fallback)
// ============================================================================

auto cuda_irfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                       const std::string& norm, cudaStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("IRFFT: dimension out of range");
    }
    bool is_float32 = (input.dtype() == DType::Complex64);

    if (dim != ndim - 1) {
        throw std::runtime_error(
            "cuda native irfft: only last-dimension IRFFT is supported. "
            "The dispatch layer should decompose non-last-dim irfft.");
    }

    int64_t N_in = shape[dim];  // n/2 + 1 complex elements
    int64_t expected_complex = n / 2 + 1;

    // Copy input into work buffer sized for expected_complex
    std::vector<int64_t> complex_shape = shape;
    complex_shape[dim] = expected_complex;
    DType complex_dtype = input.dtype();
    Tensor complex_buf(complex_shape, complex_dtype, input.device());

    if (N_in != expected_complex) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(complex_buf.data_ptr(), 0,
            complex_buf.numel() * dtype_size(complex_dtype), stream));
        int64_t copy_len = std::min(N_in, expected_complex);
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
        int64_t elem_size = dtype_size(complex_dtype);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            const char* src = static_cast<const char*>(input.data_ptr())
                + outer * N_in * elem_size;
            char* dst = static_cast<char*>(complex_buf.data_ptr())
                + outer * expected_complex * elem_size;
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(dst, src,
                copy_len * elem_size, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(complex_buf.data_ptr(), input.data_ptr(),
            input.numel() * dtype_size(complex_dtype),
            cudaMemcpyDeviceToDevice, stream));
    }

    // Output: real tensor with shape[dim] = n
    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = n;
    DType real_dtype = is_float32 ? DType::Float32 : DType::Float64;
    Tensor output(out_shape, real_dtype, input.device());

    int64_t batch = 1;
    for (int64_t i = 0; i < dim; ++i) batch *= shape[i];

    // Strategy: reconstruct full N-point spectrum from N/2+1 bins using conjugate symmetry,
    // then apply inverse FFT, take real part.
    bool use_cooley_tukey = is_power_of_2(n);
    constexpr int block_size = 256;

    // Compute IFFT normalization
    double scale_d = get_norm_factor(n, norm, /*is_forward=*/false);

    if (is_float32) {
        const float* d_in = reinterpret_cast<const float*>(complex_buf.data_ptr());
        float scale = static_cast<float>(scale_d);

        if (use_cooley_tukey) {
            CudaDevicePtr<float> d_buf(2 * n * batch, stream);
            {
                int64_t total = batch * n;
                int grid = static_cast<int>((total + block_size - 1) / block_size);
                reconstruct_spectrum_kernel<float><<<grid, block_size, 0, stream>>>(
                    d_buf.get(), d_in, n, expected_complex, batch);
                TENZOR_CUDA_CHECK(cudaGetLastError());
            }

            cooley_tukey_fft_cuda(d_buf.get(), n, batch, int64_t(2 * n), 1.0f, stream);

            {
                int64_t total = batch * n;
                int grid = static_cast<int>((total + block_size - 1) / block_size);
                extract_real_scaled_kernel<float><<<grid, block_size, 0, stream>>>(
                    output.data<float>(), d_buf.get(), n, batch, scale);
                TENZOR_CUDA_CHECK(cudaGetLastError());
            }
        } else {
            // Bluestein: reconstruct full spectrum, inverse FFT, extract real
            int64_t full_complex_numel = batch * n * 2;
            CudaDevicePtr<float> d_full(full_complex_numel, stream);
            {
                int64_t total = batch * n;
                int grid = static_cast<int>((total + block_size - 1) / block_size);
                reconstruct_spectrum_kernel<float><<<grid, block_size, 0, stream>>>(
                    d_full.get(), d_in, n, expected_complex, batch);
                TENZOR_CUDA_CHECK(cudaGetLastError());
            }

            CudaDevicePtr<float> d_ifft(full_complex_numel, stream);
            TENZOR_CUDA_CHECK(cudaMemsetAsync(d_ifft.get(), 0,
                full_complex_numel * sizeof(float), stream));
            bluestein_fft_complex_cuda(d_full.get(), d_ifft.get(),
                                       n, batch, int64_t(1), 1.0f, stream);

            {
                int64_t total = batch * n;
                int grid = static_cast<int>((total + block_size - 1) / block_size);
                extract_real_scaled_kernel<float><<<grid, block_size, 0, stream>>>(
                    output.data<float>(), d_ifft.get(), n, batch, scale);
                TENZOR_CUDA_CHECK(cudaGetLastError());
            }
        }
    } else {
        const double* d_in = reinterpret_cast<const double*>(complex_buf.data_ptr());
        double scale = scale_d;

        if (use_cooley_tukey) {
            CudaDevicePtr<double> d_buf(2 * n * batch, stream);
            {
                int64_t total = batch * n;
                int grid = static_cast<int>((total + block_size - 1) / block_size);
                reconstruct_spectrum_kernel<double><<<grid, block_size, 0, stream>>>(
                    d_buf.get(), d_in, n, expected_complex, batch);
                TENZOR_CUDA_CHECK(cudaGetLastError());
            }

            cooley_tukey_fft_cuda(d_buf.get(), n, batch, int64_t(2 * n), 1.0, stream);

            {
                int64_t total = batch * n;
                int grid = static_cast<int>((total + block_size - 1) / block_size);
                extract_real_scaled_kernel<double><<<grid, block_size, 0, stream>>>(
                    output.data<double>(), d_buf.get(), n, batch, scale);
                TENZOR_CUDA_CHECK(cudaGetLastError());
            }
        } else {
            int64_t full_complex_numel = batch * n * 2;
            CudaDevicePtr<double> d_full(full_complex_numel, stream);
            {
                int64_t total = batch * n;
                int grid = static_cast<int>((total + block_size - 1) / block_size);
                reconstruct_spectrum_kernel<double><<<grid, block_size, 0, stream>>>(
                    d_full.get(), d_in, n, expected_complex, batch);
                TENZOR_CUDA_CHECK(cudaGetLastError());
            }

            CudaDevicePtr<double> d_ifft(full_complex_numel, stream);
            TENZOR_CUDA_CHECK(cudaMemsetAsync(d_ifft.get(), 0,
                full_complex_numel * sizeof(double), stream));
            bluestein_fft_complex_cuda(d_full.get(), d_ifft.get(),
                                       n, batch, int64_t(1), 1.0, stream);

            {
                int64_t total = batch * n;
                int grid = static_cast<int>((total + block_size - 1) / block_size);
                extract_real_scaled_kernel<double><<<grid, block_size, 0, stream>>>(
                    output.data<double>(), d_ifft.get(), n, batch, scale);
                TENZOR_CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    return output;
}

// ============================================================================
// 2D FFT: Complex-to-Complex forward (native CUDA fallback)
// ============================================================================

auto cuda_fft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                      const std::vector<int64_t>& n_vec,
                      const std::string& norm, cudaStream_t stream) -> Tensor {
    if (dims.size() != 2) {
        throw std::runtime_error("native cuda fft2: expected exactly 2 dimensions");
    }
    // Apply 1D FFT sequentially along each dimension
    Tensor result = input;
    for (size_t i = 0; i < dims.size(); ++i) {
        result = cuda_fft_kernel(result, dims[i], n_vec[i], norm, stream);
    }
    return result;
}

// ============================================================================
// 2D IFFT: Complex-to-Complex inverse (native CUDA fallback)
// ============================================================================

auto cuda_ifft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& n_vec,
                       const std::string& norm, cudaStream_t stream) -> Tensor {
    if (dims.size() != 2) {
        throw std::runtime_error("native cuda ifft2: expected exactly 2 dimensions");
    }
    Tensor result = input;
    for (size_t i = 0; i < dims.size(); ++i) {
        result = cuda_ifft_kernel(result, dims[i], n_vec[i], norm, stream);
    }
    return result;
}

// ============================================================================
// N-D FFT: Complex-to-Complex forward (native CUDA fallback)
// ============================================================================

auto cuda_fftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                      const std::vector<int64_t>& n_vec,
                      const std::string& norm, cudaStream_t stream) -> Tensor {
    Tensor result = input;
    for (size_t i = 0; i < dims.size(); ++i) {
        result = cuda_fft_kernel(result, dims[i], n_vec[i], norm, stream);
    }
    return result;
}

// ============================================================================
// N-D IFFT: Complex-to-Complex inverse (native CUDA fallback)
// ============================================================================

auto cuda_ifftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& n_vec,
                       const std::string& norm, cudaStream_t stream) -> Tensor {
    Tensor result = input;
    for (size_t i = 0; i < dims.size(); ++i) {
        result = cuda_ifft_kernel(result, dims[i], n_vec[i], norm, stream);
    }
    return result;
}

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_HAS_CUFFT
