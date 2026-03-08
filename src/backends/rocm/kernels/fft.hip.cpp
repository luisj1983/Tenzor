/**
 * @file fft.hip.cpp
 * @brief ROCm FFT kernels using rocFFT.
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
 * rocFFT does NOT normalize by default. Normalization ("backward", "forward",
 * "ortho") is applied manually via a post-transform HIP kernel.
 */

#ifdef TENZOR_HAS_ROCFFT

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"

#include <rocfft/rocfft.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>

namespace tenzor {
namespace rocm {

namespace {

// ============================================================================
// rocFFT error checking
// ============================================================================

#define ROCFFT_CHECK(call)                                                     \
    do {                                                                       \
        rocfft_status result = (call);                                         \
        if (result != rocfft_status_success) {                                 \
            throw std::runtime_error(                                          \
                std::string("rocFFT error at ") + __FILE__ + ":" +             \
                std::to_string(__LINE__) + " - error code " +                  \
                std::to_string(static_cast<int>(result)));                     \
        }                                                                      \
    } while (0)

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            throw std::runtime_error(                                          \
                std::string("HIP error: ") + hipGetErrorString(err));          \
        }                                                                      \
    } while (0)

// ============================================================================
// RAII wrappers for rocFFT objects
// ============================================================================

struct RocFFTPlan {
    rocfft_plan handle = nullptr;

    RocFFTPlan() = default;
    ~RocFFTPlan() { if (handle) rocfft_plan_destroy(handle); }

    RocFFTPlan(const RocFFTPlan&) = delete;
    RocFFTPlan& operator=(const RocFFTPlan&) = delete;

    RocFFTPlan(RocFFTPlan&& other) noexcept : handle(other.handle) {
        other.handle = nullptr;
    }
};

struct RocFFTDescription {
    rocfft_plan_description handle = nullptr;

    RocFFTDescription() { ROCFFT_CHECK(rocfft_plan_description_create(&handle)); }
    ~RocFFTDescription() { if (handle) rocfft_plan_description_destroy(handle); }

    RocFFTDescription(const RocFFTDescription&) = delete;
    RocFFTDescription& operator=(const RocFFTDescription&) = delete;
};

struct RocFFTExecutionInfo {
    rocfft_execution_info handle = nullptr;

    RocFFTExecutionInfo() { ROCFFT_CHECK(rocfft_execution_info_create(&handle)); }
    ~RocFFTExecutionInfo() { if (handle) rocfft_execution_info_destroy(handle); }

    RocFFTExecutionInfo(const RocFFTExecutionInfo&) = delete;
    RocFFTExecutionInfo& operator=(const RocFFTExecutionInfo&) = delete;
};

// ============================================================================
// Normalization kernels
// rocFFT does not normalize. We apply scaling as a post-processing step.
// ============================================================================

template<typename T>
__global__ void scale_kernel(T* data, int64_t numel, T scale) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] *= scale;
    }
}

// Scale complex data (2 floats/doubles per complex element)
template<typename RealT>
__global__ void scale_complex_kernel(RealT* data, int64_t numel_complex, RealT scale) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = numel_complex * 2;  // real + imag parts
    if (idx < total) {
        data[idx] *= scale;
    }
}

/// Compute normalization scale factor.
double get_norm_factor(int64_t n, const std::string& norm, bool is_forward) {
    if (norm == "ortho") {
        return 1.0 / std::sqrt(static_cast<double>(n));
    } else if ((norm == "forward" && is_forward) || (norm == "backward" && !is_forward)) {
        return 1.0 / static_cast<double>(n);
    }
    return 1.0;
}

/// Compute normalization scale factor for multi-dimensional FFTs.
double get_norm_factor_nd(const std::vector<int64_t>& n_vec, const std::string& norm, bool is_forward) {
    double factor = 1.0;
    for (auto n : n_vec) {
        factor *= get_norm_factor(n, norm, is_forward);
    }
    return factor;
}

/// Apply scaling to a complex tensor (in-place).
void apply_normalization_complex(Tensor& output, double scale, bool is_float32, hipStream_t stream) {
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
    HIP_CHECK(hipGetLastError());
}

/// Apply scaling to a real tensor (in-place).
void apply_normalization_real(Tensor& output, double scale, bool is_float32, hipStream_t stream) {
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
    HIP_CHECK(hipGetLastError());
}

/// Create a rocFFT plan for complex-to-complex transforms.
/// @param rank Number of FFT dimensions (1, 2, or N).
/// @param lengths Array of FFT sizes for each dimension.
/// @param batch Number of batches.
/// @param is_forward True for forward, false for inverse.
/// @param is_float32 True for single precision, false for double.
/// @param stride Stride between elements along FFT dim.
/// @param dist Distance between batches.
rocfft_plan create_c2c_plan(int rank, const size_t* lengths, size_t batch,
                            bool is_forward, bool is_float32,
                            const size_t* in_strides, size_t in_dist,
                            const size_t* out_strides, size_t out_dist) {
    auto transform_type = is_forward ? rocfft_transform_type_complex_forward
                                     : rocfft_transform_type_complex_inverse;
    auto precision = is_float32 ? rocfft_precision_single : rocfft_precision_double;

    RocFFTDescription desc;
    ROCFFT_CHECK(rocfft_plan_description_set_data_layout(
        desc.handle,
        rocfft_array_type_complex_interleaved,   // input layout
        rocfft_array_type_complex_interleaved,   // output layout
        nullptr, nullptr,                         // offsets (unused)
        static_cast<size_t>(rank), in_strides, in_dist,
        static_cast<size_t>(rank), out_strides, out_dist));

    rocfft_plan plan = nullptr;
    ROCFFT_CHECK(rocfft_plan_create(&plan, rocfft_placement_inplace,
                                    transform_type, precision,
                                    static_cast<size_t>(rank), lengths, batch,
                                    desc.handle));
    return plan;
}

/// Copy input data into output buffer with optional padding/truncation along dim.
void copy_with_padding(const Tensor& input, Tensor& output,
                       int64_t dim, int64_t N_in, int64_t N_out,
                       hipStream_t stream) {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    DType dtype = output.dtype();
    int64_t elem_size = dtype_size(dtype);

    if (N_in != N_out) {
        HIP_CHECK(hipMemsetAsync(output.data_ptr(), 0,
            output.numel() * elem_size, stream));

        int64_t copy_len = std::min(N_in, N_out);
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            const char* src = static_cast<const char*>(input.data_ptr())
                + outer * N_in * inner_size * elem_size;
            char* dst = static_cast<char*>(output.data_ptr())
                + outer * N_out * inner_size * elem_size;
            HIP_CHECK(hipMemcpyAsync(dst, src,
                copy_len * inner_size * elem_size,
                hipMemcpyDeviceToDevice, stream));
        }
    } else {
        HIP_CHECK(hipMemcpyAsync(output.data_ptr(), input.data_ptr(),
            input.numel() * elem_size,
            hipMemcpyDeviceToDevice, stream));
    }
}

/// Execute a rocFFT plan with work buffer management.
void execute_plan(rocfft_plan plan, void* in_buf, void* out_buf, hipStream_t stream) {
    RocFFTExecutionInfo info;
    ROCFFT_CHECK(rocfft_execution_info_set_stream(info.handle, stream));

    // Query and allocate work buffer
    size_t work_buf_size = 0;
    ROCFFT_CHECK(rocfft_plan_get_work_buffer_size(plan, &work_buf_size));

    void* work_buf = nullptr;
    if (work_buf_size > 0) {
        HIP_CHECK(hipMalloc(&work_buf, work_buf_size));
        ROCFFT_CHECK(rocfft_execution_info_set_work_buffer(info.handle, work_buf, work_buf_size));
    }

    void* in_buffers[1] = { in_buf };
    void* out_buffers[1] = { out_buf };

    // For in-place transforms, out_buffers should be nullptr
    rocfft_status status = rocfft_execute(plan, in_buffers,
                                          (in_buf == out_buf) ? nullptr : out_buffers,
                                          info.handle);

    // Free work buffer before checking status
    if (work_buf) {
        HIP_CHECK(hipFree(work_buf));
    }

    if (status != rocfft_status_success) {
        throw std::runtime_error(
            std::string("rocFFT execute failed - error code ") +
            std::to_string(static_cast<int>(status)));
    }
}

} // anonymous namespace

// ============================================================================
// 1D FFT: Complex-to-Complex forward
// ============================================================================

auto rocm_fft_kernel(const Tensor& input, int64_t dim, int64_t n,
                     const std::string& norm, hipStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    bool is_float32 = (input.dtype() == DType::Complex64);

    int64_t N_in = shape[dim];
    int64_t N_out = n;

    // Build output shape
    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = N_out;

    DType out_dtype = input.dtype();
    Tensor output(out_shape, out_dtype, input.device());

    // Copy input into output buffer (with padding/truncation if needed)
    copy_with_padding(input, output, dim, N_in, N_out, stream);

    // Compute strides for rocFFT
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= out_shape[i];
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= out_shape[i];
    int64_t batch = outer_size * inner_size;

    size_t lengths[1] = { static_cast<size_t>(N_out) };
    size_t istride[1], ostride[1];
    size_t idist, odist;

    istride[0] = static_cast<size_t>(inner_size);
    ostride[0] = static_cast<size_t>(inner_size);

    if (dim == ndim - 1) {
        idist = static_cast<size_t>(N_out);
        odist = static_cast<size_t>(N_out);
        batch = outer_size;
    } else {
        idist = 1;
        odist = 1;
    }

    // Create plan
    rocfft_plan plan = create_c2c_plan(1, lengths, static_cast<size_t>(batch),
                                       /*is_forward=*/true, is_float32,
                                       istride, idist, ostride, odist);
    RocFFTPlan plan_guard;
    plan_guard.handle = plan;

    // Execute in-place
    execute_plan(plan, output.data_ptr(), output.data_ptr(), stream);

    // Apply normalization
    double scale = get_norm_factor(N_out, norm, /*is_forward=*/true);
    apply_normalization_complex(output, scale, is_float32, stream);

    return output;
}

// ============================================================================
// 1D IFFT: Complex-to-Complex inverse
// ============================================================================

auto rocm_ifft_kernel(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm, hipStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    bool is_float32 = (input.dtype() == DType::Complex64);

    int64_t N_in = shape[dim];
    int64_t N_out = n;

    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = N_out;

    DType out_dtype = input.dtype();
    Tensor output(out_shape, out_dtype, input.device());

    copy_with_padding(input, output, dim, N_in, N_out, stream);

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= out_shape[i];
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= out_shape[i];
    int64_t batch = outer_size * inner_size;

    size_t lengths[1] = { static_cast<size_t>(N_out) };
    size_t istride[1] = { static_cast<size_t>(inner_size) };
    size_t ostride[1] = { static_cast<size_t>(inner_size) };
    size_t idist, odist;

    if (dim == ndim - 1) {
        idist = static_cast<size_t>(N_out);
        odist = static_cast<size_t>(N_out);
        batch = outer_size;
    } else {
        idist = 1;
        odist = 1;
    }

    rocfft_plan plan = create_c2c_plan(1, lengths, static_cast<size_t>(batch),
                                       /*is_forward=*/false, is_float32,
                                       istride, idist, ostride, odist);
    RocFFTPlan plan_guard;
    plan_guard.handle = plan;

    execute_plan(plan, output.data_ptr(), output.data_ptr(), stream);

    double scale = get_norm_factor(N_out, norm, /*is_forward=*/false);
    apply_normalization_complex(output, scale, is_float32, stream);

    return output;
}

// ============================================================================
// 1D RFFT: Real-to-Complex forward
// ============================================================================

auto rocm_rfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm, hipStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    bool is_float32 = (input.dtype() == DType::Float32);

    int64_t N_in = shape[dim];
    int64_t N_out_complex = n / 2 + 1;

    if (dim != ndim - 1) {
        throw std::runtime_error(
            "rocFFT rfft: only last-dimension FFT is supported on ROCm. "
            "The dispatch layer should decompose non-last-dim rfft.");
    }

    // Prepare real input buffer (padded or truncated to length n)
    std::vector<int64_t> real_shape = shape;
    real_shape[dim] = n;
    DType real_dtype = input.dtype();

    Tensor real_buf(real_shape, real_dtype, input.device());
    if (N_in != n) {
        HIP_CHECK(hipMemsetAsync(real_buf.data_ptr(), 0,
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
            HIP_CHECK(hipMemcpyAsync(dst, src,
                copy_len * elem_size, hipMemcpyDeviceToDevice, stream));
        }
    } else {
        HIP_CHECK(hipMemcpyAsync(real_buf.data_ptr(), input.data_ptr(),
            input.numel() * dtype_size(real_dtype),
            hipMemcpyDeviceToDevice, stream));
    }

    // Output: complex, with shape[dim] = n/2 + 1
    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = N_out_complex;
    DType complex_dtype = is_float32 ? DType::Complex64 : DType::Complex128;
    Tensor output(out_shape, complex_dtype, input.device());

    int64_t batch = 1;
    for (int64_t i = 0; i < dim; ++i) batch *= shape[i];

    auto precision = is_float32 ? rocfft_precision_single : rocfft_precision_double;
    size_t lengths[1] = { static_cast<size_t>(n) };

    // For R2C, use default strides (contiguous last-dim)
    RocFFTDescription desc;
    // Input: real, contiguous along last dim, batch stride = n
    size_t in_strides[1] = { 1 };
    size_t in_dist = static_cast<size_t>(n);
    // Output: complex interleaved, contiguous, batch stride = n/2+1
    size_t out_strides[1] = { 1 };
    size_t out_dist = static_cast<size_t>(N_out_complex);

    ROCFFT_CHECK(rocfft_plan_description_set_data_layout(
        desc.handle,
        rocfft_array_type_real,
        rocfft_array_type_complex_interleaved,
        nullptr, nullptr,
        1, in_strides, in_dist,
        1, out_strides, out_dist));

    rocfft_plan plan = nullptr;
    ROCFFT_CHECK(rocfft_plan_create(&plan, rocfft_placement_notinplace,
                                    rocfft_transform_type_real_forward,
                                    precision, 1, lengths,
                                    static_cast<size_t>(batch), desc.handle));
    RocFFTPlan plan_guard;
    plan_guard.handle = plan;

    // Execute out-of-place R2C
    execute_plan(plan, real_buf.data_ptr(), output.data_ptr(), stream);

    // Apply normalization
    double scale = get_norm_factor(n, norm, /*is_forward=*/true);
    apply_normalization_complex(output, scale, is_float32, stream);

    return output;
}

// ============================================================================
// 1D IRFFT: Complex-to-Real inverse
// ============================================================================

auto rocm_irfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                       const std::string& norm, hipStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    bool is_float32 = (input.dtype() == DType::Complex64);

    if (dim != ndim - 1) {
        throw std::runtime_error(
            "rocFFT irfft: only last-dimension IRFFT is supported on ROCm. "
            "The dispatch layer should decompose non-last-dim irfft.");
    }

    int64_t N_in = shape[dim];
    int64_t expected_complex = n / 2 + 1;

    // Copy input (complex) into a work buffer sized for expected_complex
    std::vector<int64_t> complex_shape = shape;
    complex_shape[dim] = expected_complex;
    DType complex_dtype = input.dtype();
    Tensor complex_buf(complex_shape, complex_dtype, input.device());

    if (N_in != expected_complex) {
        HIP_CHECK(hipMemsetAsync(complex_buf.data_ptr(), 0,
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
            HIP_CHECK(hipMemcpyAsync(dst, src,
                copy_len * elem_size, hipMemcpyDeviceToDevice, stream));
        }
    } else {
        HIP_CHECK(hipMemcpyAsync(complex_buf.data_ptr(), input.data_ptr(),
            input.numel() * dtype_size(complex_dtype),
            hipMemcpyDeviceToDevice, stream));
    }

    // Output: real tensor with shape[dim] = n
    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = n;
    DType real_dtype = is_float32 ? DType::Float32 : DType::Float64;
    Tensor output(out_shape, real_dtype, input.device());

    int64_t batch = 1;
    for (int64_t i = 0; i < dim; ++i) batch *= shape[i];

    auto precision = is_float32 ? rocfft_precision_single : rocfft_precision_double;
    size_t lengths[1] = { static_cast<size_t>(n) };

    RocFFTDescription desc;
    // Input: complex interleaved, batch stride = n/2+1
    size_t in_strides[1] = { 1 };
    size_t in_dist = static_cast<size_t>(expected_complex);
    // Output: real, batch stride = n
    size_t out_strides[1] = { 1 };
    size_t out_dist = static_cast<size_t>(n);

    ROCFFT_CHECK(rocfft_plan_description_set_data_layout(
        desc.handle,
        rocfft_array_type_complex_interleaved,
        rocfft_array_type_real,
        nullptr, nullptr,
        1, in_strides, in_dist,
        1, out_strides, out_dist));

    rocfft_plan plan = nullptr;
    ROCFFT_CHECK(rocfft_plan_create(&plan, rocfft_placement_notinplace,
                                    rocfft_transform_type_real_inverse,
                                    precision, 1, lengths,
                                    static_cast<size_t>(batch), desc.handle));
    RocFFTPlan plan_guard;
    plan_guard.handle = plan;

    // Execute out-of-place C2R
    execute_plan(plan, complex_buf.data_ptr(), output.data_ptr(), stream);

    // Apply normalization
    double scale = get_norm_factor(n, norm, /*is_forward=*/false);
    apply_normalization_real(output, scale, is_float32, stream);

    return output;
}

// ============================================================================
// 2D FFT: Complex-to-Complex forward
// ============================================================================

auto rocm_fft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                      const std::vector<int64_t>& n_vec,
                      const std::string& norm, hipStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    bool is_float32 = (input.dtype() == DType::Complex64);

    if (dims.size() != 2) {
        throw std::runtime_error("rocFFT fft2: expected exactly 2 dimensions");
    }

    // Check if dims are the last two dimensions
    bool last_two = (dims[0] == ndim - 2 && dims[1] == ndim - 1);
    if (!last_two) {
        // Fall back to sequential 1D FFTs
        Tensor result = input;
        for (size_t i = 0; i < dims.size(); ++i) {
            result = rocm_fft_kernel(result, dims[i], n_vec[i], norm, stream);
        }
        return result;
    }

    int64_t N0 = n_vec[0];  // rows
    int64_t N1 = n_vec[1];  // cols

    std::vector<int64_t> out_shape = shape;
    out_shape[dims[0]] = N0;
    out_shape[dims[1]] = N1;

    DType out_dtype = input.dtype();
    Tensor output(out_shape, out_dtype, input.device());
    HIP_CHECK(hipMemsetAsync(output.data_ptr(), 0,
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
            HIP_CHECK(hipMemcpyAsync(dst, src,
                copy_cols * elem_size, hipMemcpyDeviceToDevice, stream));
        }
    }

    // Create 2D rocFFT plan (contiguous last-two-dims)
    // rocFFT expects dimensions in row-major order (outermost first)
    size_t lengths[2] = { static_cast<size_t>(N0), static_cast<size_t>(N1) };
    size_t fft_size = static_cast<size_t>(N0 * N1);
    size_t istride[2] = { static_cast<size_t>(N1), 1 };
    size_t ostride[2] = { static_cast<size_t>(N1), 1 };

    rocfft_plan plan = create_c2c_plan(2, lengths, static_cast<size_t>(batch),
                                       /*is_forward=*/true, is_float32,
                                       istride, fft_size, ostride, fft_size);
    RocFFTPlan plan_guard;
    plan_guard.handle = plan;

    execute_plan(plan, output.data_ptr(), output.data_ptr(), stream);

    double scale = get_norm_factor_nd(n_vec, norm, /*is_forward=*/true);
    apply_normalization_complex(output, scale, is_float32, stream);

    return output;
}

// ============================================================================
// 2D IFFT: Complex-to-Complex inverse
// ============================================================================

auto rocm_ifft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& n_vec,
                       const std::string& norm, hipStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    bool is_float32 = (input.dtype() == DType::Complex64);

    if (dims.size() != 2) {
        throw std::runtime_error("rocFFT ifft2: expected exactly 2 dimensions");
    }

    bool last_two = (dims[0] == ndim - 2 && dims[1] == ndim - 1);
    if (!last_two) {
        Tensor result = input;
        for (size_t i = 0; i < dims.size(); ++i) {
            result = rocm_ifft_kernel(result, dims[i], n_vec[i], norm, stream);
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
    HIP_CHECK(hipMemsetAsync(output.data_ptr(), 0,
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
            HIP_CHECK(hipMemcpyAsync(dst, src,
                copy_cols * elem_size, hipMemcpyDeviceToDevice, stream));
        }
    }

    size_t lengths[2] = { static_cast<size_t>(N0), static_cast<size_t>(N1) };
    size_t fft_size = static_cast<size_t>(N0 * N1);
    size_t istride[2] = { static_cast<size_t>(N1), 1 };
    size_t ostride[2] = { static_cast<size_t>(N1), 1 };

    rocfft_plan plan = create_c2c_plan(2, lengths, static_cast<size_t>(batch),
                                       /*is_forward=*/false, is_float32,
                                       istride, fft_size, ostride, fft_size);
    RocFFTPlan plan_guard;
    plan_guard.handle = plan;

    execute_plan(plan, output.data_ptr(), output.data_ptr(), stream);

    double scale = get_norm_factor_nd(n_vec, norm, /*is_forward=*/false);
    apply_normalization_complex(output, scale, is_float32, stream);

    return output;
}

// ============================================================================
// N-D FFT: Complex-to-Complex forward
// ============================================================================

auto rocm_fftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                      const std::vector<int64_t>& n_vec,
                      const std::string& norm, hipStream_t stream) -> Tensor {
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
            result = rocm_fft_kernel(result, dims[i], n_vec[i], norm, stream);
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
    HIP_CHECK(hipMemsetAsync(output.data_ptr(), 0,
        output.numel() * dtype_size(out_dtype), stream));

    int64_t batch = 1;
    for (int64_t i = 0; i < ndim - rank; ++i) batch *= shape[i];

    int64_t in_fft_size = 1, out_fft_size = 1;
    for (int64_t i = 0; i < rank; ++i) {
        in_fft_size *= shape[dims[i]];
        out_fft_size *= n_vec[i];
    }

    int64_t elem_size = dtype_size(out_dtype);

    // If shapes match in FFT dimensions, direct copy
    if (in_fft_size == out_fft_size) {
        bool shapes_match = true;
        for (int64_t i = 0; i < rank; ++i) {
            if (shape[dims[i]] != n_vec[i]) { shapes_match = false; break; }
        }
        if (shapes_match) {
            HIP_CHECK(hipMemcpyAsync(output.data_ptr(), input.data_ptr(),
                input.numel() * elem_size, hipMemcpyDeviceToDevice, stream));
        }
    }
    // General case with different sizes: fall back to sequential 1D FFTs
    if (in_fft_size != out_fft_size) {
        Tensor result = input;
        for (int64_t i = 0; i < rank; ++i) {
            result = rocm_fft_kernel(result, dims[i], n_vec[i], norm, stream);
        }
        return result;
    }

    // Create N-D rocFFT plan
    std::vector<size_t> lengths(rank);
    for (int64_t i = 0; i < rank; ++i) {
        lengths[i] = static_cast<size_t>(n_vec[i]);
    }

    // Compute strides for contiguous last-rank-dims layout
    std::vector<size_t> strides(rank);
    strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * lengths[i + 1];
    }
    size_t dist = static_cast<size_t>(out_fft_size);

    rocfft_plan plan = create_c2c_plan(static_cast<int>(rank), lengths.data(),
                                       static_cast<size_t>(batch),
                                       /*is_forward=*/true, is_float32,
                                       strides.data(), dist, strides.data(), dist);
    RocFFTPlan plan_guard;
    plan_guard.handle = plan;

    execute_plan(plan, output.data_ptr(), output.data_ptr(), stream);

    double scale = get_norm_factor_nd(n_vec, norm, /*is_forward=*/true);
    apply_normalization_complex(output, scale, is_float32, stream);

    return output;
}

// ============================================================================
// N-D IFFT: Complex-to-Complex inverse
// ============================================================================

auto rocm_ifftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& n_vec,
                       const std::string& norm, hipStream_t stream) -> Tensor {
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
            result = rocm_ifft_kernel(result, dims[i], n_vec[i], norm, stream);
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
    HIP_CHECK(hipMemsetAsync(output.data_ptr(), 0,
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
            HIP_CHECK(hipMemcpyAsync(output.data_ptr(), input.data_ptr(),
                input.numel() * elem_size, hipMemcpyDeviceToDevice, stream));
        }
    }
    if (in_fft_size != out_fft_size) {
        Tensor result = input;
        for (int64_t i = 0; i < rank; ++i) {
            result = rocm_ifft_kernel(result, dims[i], n_vec[i], norm, stream);
        }
        return result;
    }

    std::vector<size_t> lengths(rank);
    for (int64_t i = 0; i < rank; ++i) {
        lengths[i] = static_cast<size_t>(n_vec[i]);
    }

    std::vector<size_t> strides(rank);
    strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * lengths[i + 1];
    }
    size_t dist = static_cast<size_t>(out_fft_size);

    rocfft_plan plan = create_c2c_plan(static_cast<int>(rank), lengths.data(),
                                       static_cast<size_t>(batch),
                                       /*is_forward=*/false, is_float32,
                                       strides.data(), dist, strides.data(), dist);
    RocFFTPlan plan_guard;
    plan_guard.handle = plan;

    execute_plan(plan, output.data_ptr(), output.data_ptr(), stream);

    double scale = get_norm_factor_nd(n_vec, norm, /*is_forward=*/false);
    apply_normalization_complex(output, scale, is_float32, stream);

    return output;
}

} // namespace rocm
} // namespace tenzor

#endif // TENZOR_HAS_ROCFFT
