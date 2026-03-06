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

    CuFFTPlan() = default;
    ~CuFFTPlan() { if (valid) cufftDestroy(handle); }

    CuFFTPlan(const CuFFTPlan&) = delete;
    CuFFTPlan& operator=(const CuFFTPlan&) = delete;

    CuFFTPlan(CuFFTPlan&& other) noexcept : handle(other.handle), valid(other.valid) {
        other.valid = false;
    }

    void create() {
        CUFFT_CHECK(cufftCreate(&handle));
        valid = true;
    }

    void set_stream(cudaStream_t stream) {
        CUFFT_CHECK(cufftSetStream(handle, stream));
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

/// Compute batch size: product of all dimensions except the FFT dimension(s).
int64_t compute_batch_size(const std::vector<int64_t>& shape, int64_t fft_dim) {
    int64_t batch = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
        if (i != fft_dim) batch *= shape[i];
    }
    return batch;
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
    int istride = static_cast<int>(inner_size);
    int ostride = static_cast<int>(inner_size);
    int idist, odist;

    if (dim == ndim - 1) {
        // FFT along last dimension: contiguous, batch = outer_size
        idist = n_int;
        odist = n_int;
        batch = outer_size;
    } else {
        // FFT along non-last dimension: strided access
        // Each "batch" element is separated by 1 in memory (inner_size batches
        // per outer block), and outer blocks are separated by N_out * inner_size.
        idist = 1;
        odist = 1;
        // We process inner_size batches per outer block, total = outer_size * inner_size
    }

    cufftType fft_type = is_float32 ? CUFFT_C2C : CUFFT_Z2Z;
    CUFFT_CHECK(cufftPlanMany(&plan.handle, 1, &n_int,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              fft_type, static_cast<int>(batch)));
    CUFFT_CHECK(cufftSetStream(plan.handle, stream));

    // Execute in-place
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
    int istride = static_cast<int>(inner_size);
    int ostride = static_cast<int>(inner_size);
    int idist, odist;

    if (dim == ndim - 1) {
        idist = n_int;
        odist = n_int;
        batch = outer_size;
    } else {
        idist = 1;
        odist = 1;
    }

    cufftType fft_type = is_float32 ? CUFFT_C2C : CUFFT_Z2Z;
    CUFFT_CHECK(cufftPlanMany(&plan.handle, 1, &n_int,
                              inembed, istride, idist,
                              onembed, ostride, odist,
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

auto cuda_irfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                       const std::string& norm, cudaStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
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

#endif // TENZOR_HAS_CUFFT
