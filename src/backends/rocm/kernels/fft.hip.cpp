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
#include "../hip_buffer.hpp"
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <mutex>
#include <memory>
#include <list>
#include <unordered_map>
#include <functional>

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

// HIP_CHECK is provided by rocm_error.hpp (via hip_buffer.hpp)

// ============================================================================
// RAII wrappers for rocFFT objects
// ============================================================================

/// RAII wrapper for a raw rocfft_plan handle.  Used both standalone (for
/// one-shot plans) and inside std::shared_ptr (for cached plans).
struct RocFFTPlanHandle {
    rocfft_plan handle = nullptr;

    RocFFTPlanHandle() = default;
    explicit RocFFTPlanHandle(rocfft_plan h) : handle(h) {}
    ~RocFFTPlanHandle() { if (handle) rocfft_plan_destroy(handle); }

    RocFFTPlanHandle(const RocFFTPlanHandle&) = delete;
    RocFFTPlanHandle& operator=(const RocFFTPlanHandle&) = delete;

    RocFFTPlanHandle(RocFFTPlanHandle&& other) noexcept : handle(other.handle) {
        other.handle = nullptr;
    }
};

// ============================================================================
// FFT Plan Cache — mutex-protected global LRU, max 32 entries
// ============================================================================

/// Transform categories for the cache key.
enum class FFTTransformKind : uint8_t {
    C2C_Forward,
    C2C_Inverse,
    R2C,
    C2R,
};

/// Cache key that uniquely identifies a rocFFT plan configuration.
struct PlanKey {
    FFTTransformKind kind;
    rocfft_precision precision;
    std::vector<size_t> lengths;
    size_t batch;
    // Stride / dist info (needed because identical lengths + batch can have
    // different memory layouts).
    std::vector<size_t> in_strides;
    size_t in_dist;
    std::vector<size_t> out_strides;
    size_t out_dist;
    rocfft_result_placement placement;

    bool operator==(const PlanKey& o) const {
        return kind == o.kind && precision == o.precision &&
               lengths == o.lengths && batch == o.batch &&
               in_strides == o.in_strides && in_dist == o.in_dist &&
               out_strides == o.out_strides && out_dist == o.out_dist &&
               placement == o.placement;
    }
};

struct PlanKeyHash {
    size_t operator()(const PlanKey& k) const {
        // FNV-1a style hash combining
        size_t h = 14695981039346656037ULL;
        auto mix = [&](size_t v) {
            h ^= v;
            h *= 1099511628211ULL;
        };
        mix(static_cast<size_t>(k.kind));
        mix(static_cast<size_t>(k.precision));
        mix(k.batch);
        mix(k.in_dist);
        mix(k.out_dist);
        mix(static_cast<size_t>(k.placement));
        for (auto v : k.lengths) mix(v);
        for (auto v : k.in_strides) mix(v);
        for (auto v : k.out_strides) mix(v);
        return h;
    }
};

class PlanCache {
public:
    static constexpr size_t kMaxEntries = 32;

    /// Return a cached plan or create one via \p create_fn.
    std::shared_ptr<RocFFTPlanHandle>
    get_or_create(const PlanKey& key,
                  const std::function<rocfft_plan()>& create_fn) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = map_.find(key);
        if (it != map_.end()) {
            // Move to front of LRU list (most recently used).
            lru_.splice(lru_.begin(), lru_, it->second.lru_it);
            return it->second.plan;
        }

        // Evict LRU entry if at capacity.
        if (map_.size() >= kMaxEntries) {
            const PlanKey& evict_key = lru_.back();
            map_.erase(evict_key);
            lru_.pop_back();
        }

        // Create the plan and insert.
        rocfft_plan raw = create_fn();
        auto sp = std::make_shared<RocFFTPlanHandle>(raw);
        lru_.push_front(key);
        map_[key] = CacheEntry{sp, lru_.begin()};
        return sp;
    }

private:
    struct CacheEntry {
        std::shared_ptr<RocFFTPlanHandle> plan;
        std::list<PlanKey>::iterator lru_it;
    };

    std::mutex mutex_;
    std::unordered_map<PlanKey, CacheEntry, PlanKeyHash> map_;
    std::list<PlanKey> lru_;  // front = most recently used
};

PlanCache& get_plan_cache() {
    static PlanCache cache;
    return cache;
}

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
    int64_t grid_size = (total_reals + block_size - 1) / block_size;
    if (grid_size > static_cast<int64_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("FFT normalization grid exceeds device limit");

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
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > static_cast<int64_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("FFT normalization grid exceeds device limit");

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

/// Create (or fetch from cache) a rocFFT plan for complex-to-complex transforms.
/// Returns a shared_ptr to a cached RocFFTPlanHandle.
std::shared_ptr<RocFFTPlanHandle>
create_c2c_plan(int rank, const size_t* lengths, size_t batch,
                bool is_forward, bool is_float32,
                const size_t* in_strides, size_t in_dist,
                const size_t* out_strides, size_t out_dist) {
    auto transform_type = is_forward ? rocfft_transform_type_complex_forward
                                     : rocfft_transform_type_complex_inverse;
    auto precision = is_float32 ? rocfft_precision_single : rocfft_precision_double;

    PlanKey key;
    key.kind = is_forward ? FFTTransformKind::C2C_Forward : FFTTransformKind::C2C_Inverse;
    key.precision = precision;
    key.lengths.assign(lengths, lengths + rank);
    key.batch = batch;
    key.in_strides.assign(in_strides, in_strides + rank);
    key.in_dist = in_dist;
    key.out_strides.assign(out_strides, out_strides + rank);
    key.out_dist = out_dist;
    key.placement = rocfft_placement_inplace;

    return get_plan_cache().get_or_create(key, [&]() -> rocfft_plan {
        RocFFTDescription desc;
        ROCFFT_CHECK(rocfft_plan_description_set_data_layout(
            desc.handle,
            rocfft_array_type_complex_interleaved,
            rocfft_array_type_complex_interleaved,
            nullptr, nullptr,
            static_cast<size_t>(rank), in_strides, in_dist,
            static_cast<size_t>(rank), out_strides, out_dist));

        rocfft_plan plan = nullptr;
        ROCFFT_CHECK(rocfft_plan_create(&plan, rocfft_placement_inplace,
                                        transform_type, precision,
                                        static_cast<size_t>(rank), lengths, batch,
                                        desc.handle));
        return plan;
    });
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

    // Query and allocate work buffer (RAII — freed automatically on scope exit)
    size_t work_buf_size = 0;
    ROCFFT_CHECK(rocfft_plan_get_work_buffer_size(plan, &work_buf_size));

    tenzor::rocm::HipBuffer work_buf(work_buf_size);
    if (work_buf_size > 0) {
        ROCFFT_CHECK(rocfft_execution_info_set_work_buffer(info.handle, work_buf.ptr, work_buf_size));
    }

    void* in_buffers[1] = { in_buf };
    void* out_buffers[1] = { out_buf };

    // For in-place transforms, out_buffers should be nullptr
    ROCFFT_CHECK(rocfft_execute(plan, in_buffers,
                                (in_buf == out_buf) ? nullptr : out_buffers,
                                info.handle));
}

} // anonymous namespace

// ============================================================================
// 1D FFT: Complex-to-Complex forward
// ============================================================================

auto rocm_fft_kernel(const Tensor& input, int64_t dim, int64_t n,
                     const std::string& norm, hipStream_t stream) -> Tensor {
    // BFloat16: upcast to Float32 (Complex64), compute, downcast
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = rocm_fft_kernel(input_f32, dim, n, norm, stream);
        return result_f32.to(DType::BFloat16);
    }

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

    bool last_dim = (dim == ndim - 1);
    if (last_dim) {
        istride[0] = 1;
        ostride[0] = 1;
        idist = static_cast<size_t>(N_out);
        odist = static_cast<size_t>(N_out);
        batch = outer_size;
    } else {
        idist = 1;
        odist = 1;
        // audit-2026-05-03 — for non-last dim with both outer + inner > 1,
        // a single rocFFT plan with idist=1 cannot encode the layout
        // (inter-batch offsets jump by N_out*inner_size at outer-block
        // boundaries, not by 1). Loop per outer block instead.
        batch = inner_size;
    }

    // Create or fetch cached plan
    auto cached_plan = create_c2c_plan(1, lengths, static_cast<size_t>(batch),
                                       /*is_forward=*/true, is_float32,
                                       istride, idist, ostride, odist);

    if (last_dim) {
        execute_plan(cached_plan->handle, output.data_ptr(), output.data_ptr(), stream);
    } else {
        size_t outer_stride_elems = static_cast<size_t>(N_out) * static_cast<size_t>(inner_size);
        size_t elem_size = dtype_size(out_dtype);
        for (int64_t b = 0; b < outer_size; ++b) {
            void* base = static_cast<char*>(output.data_ptr())
                + b * outer_stride_elems * elem_size;
            execute_plan(cached_plan->handle, base, base, stream);
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

auto rocm_ifft_kernel(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm, hipStream_t stream) -> Tensor {
    // BFloat16: upcast to Float32 (Complex64), compute, downcast
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = rocm_ifft_kernel(input_f32, dim, n, norm, stream);
        return result_f32.to(DType::BFloat16);
    }

    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("FFT: dimension out of range");
    }
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

    bool last_dim = (dim == ndim - 1);
    if (last_dim) {
        istride[0] = 1;
        ostride[0] = 1;
        idist = static_cast<size_t>(N_out);
        odist = static_cast<size_t>(N_out);
        batch = outer_size;
    } else {
        idist = 1;
        odist = 1;
        batch = inner_size;
    }

    auto cached_plan = create_c2c_plan(1, lengths, static_cast<size_t>(batch),
                                       /*is_forward=*/false, is_float32,
                                       istride, idist, ostride, odist);

    if (last_dim) {
        execute_plan(cached_plan->handle, output.data_ptr(), output.data_ptr(), stream);
    } else {
        size_t outer_stride_elems = static_cast<size_t>(N_out) * static_cast<size_t>(inner_size);
        size_t elem_size = dtype_size(out_dtype);
        for (int64_t b = 0; b < outer_size; ++b) {
            void* base = static_cast<char*>(output.data_ptr())
                + b * outer_stride_elems * elem_size;
            execute_plan(cached_plan->handle, base, base, stream);
        }
    }

    double scale = get_norm_factor(N_out, norm, /*is_forward=*/false);
    apply_normalization_complex(output, scale, is_float32, stream);

    return output;
}

// ============================================================================
// 1D RFFT: Real-to-Complex forward
// ============================================================================

auto rocm_rfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm, hipStream_t stream) -> Tensor {
    // Float16/BFloat16: upcast to Float32, compute rfft; result is Complex64
    // (no downcast needed — rfft always produces complex output). rocFFT has no
    // half-precision real transform, so a non-widened half input yields zeros.
    if (input.dtype() == DType::BFloat16 || input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        return rocm_rfft_kernel(input_f32, dim, n, norm, stream);
    }

    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("RFFT/IRFFT: dimension out of range");
    }
    bool is_float32 = (input.dtype() == DType::Float32);

    int64_t N_in = shape[dim];
    int64_t N_out_complex = n / 2 + 1;

    // rocFFT's simple plan interface requires the FFT dimension to be
    // innermost. Rather than relying solely on the op-layer's
    // transpose-before-dispatch trick (src/ops/fft.cpp rfft(), an unenforced,
    // comment-only contract that JIT-trace replay can bypass — see
    // src/jit/graph.cpp OpId::RFFT dispatch), honor arbitrary axes directly
    // here too: transpose the target axis to last, recurse, transpose back.
    // Matches CPU/CUDA/OneAPI/Vulkan's arbitrary-axis support.
    if (dim != ndim - 1) {
        Tensor input_t = input.transpose(dim, ndim - 1).contiguous();
        Tensor result_t = rocm_rfft_kernel(input_t, ndim - 1, n, norm, stream);
        return result_t.transpose(dim, ndim - 1).contiguous();
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
    // Input: real, contiguous along last dim, batch stride = n
    size_t in_strides[1] = { 1 };
    size_t in_dist = static_cast<size_t>(n);
    // Output: complex interleaved, contiguous, batch stride = n/2+1
    size_t out_strides[1] = { 1 };
    size_t out_dist = static_cast<size_t>(N_out_complex);

    // Build cache key for R2C plan
    PlanKey r2c_key;
    r2c_key.kind = FFTTransformKind::R2C;
    r2c_key.precision = precision;
    r2c_key.lengths.assign(lengths, lengths + 1);
    r2c_key.batch = static_cast<size_t>(batch);
    r2c_key.in_strides.assign(in_strides, in_strides + 1);
    r2c_key.in_dist = in_dist;
    r2c_key.out_strides.assign(out_strides, out_strides + 1);
    r2c_key.out_dist = out_dist;
    r2c_key.placement = rocfft_placement_notinplace;

    auto cached_plan = get_plan_cache().get_or_create(r2c_key, [&]() -> rocfft_plan {
        RocFFTDescription desc;
        // R2C forward: output must be hermitian_interleaved, not complex_interleaved.
        // rocFFT rejects complex_interleaved with rocfft_status_invalid_array_type (4).
        ROCFFT_CHECK(rocfft_plan_description_set_data_layout(
            desc.handle,
            rocfft_array_type_real,
            rocfft_array_type_hermitian_interleaved,
            nullptr, nullptr,
            1, in_strides, in_dist,
            1, out_strides, out_dist));

        rocfft_plan plan = nullptr;
        ROCFFT_CHECK(rocfft_plan_create(&plan, rocfft_placement_notinplace,
                                        rocfft_transform_type_real_forward,
                                        precision, 1, lengths,
                                        static_cast<size_t>(batch), desc.handle));
        return plan;
    });

    // Execute out-of-place R2C
    execute_plan(cached_plan->handle, real_buf.data_ptr(), output.data_ptr(), stream);

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
    // Float16/BFloat16: upcast to Float32 (-> Complex64), compute irfft, then
    // downcast the real result back to the half dtype.
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = rocm_irfft_kernel(input_f32, dim, n, norm, stream);
        return result_f32.to(DType::BFloat16);
    }
    if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = rocm_irfft_kernel(input_f32, dim, n, norm, stream);
        return result_f32.to(DType::Float16);
    }
    // Real-to-complex promotion. The CPU and CUDA irfft kernels accept Float32
    // or Float64 input by first casting to Complex64/Complex128 (imag = 0),
    // so mirror that here — autograd wires the Float32 forward path through
    // the same dispatch site.
    if (input.dtype() == DType::Float32) {
        auto input_c64 = input.to(DType::Complex64);
        return rocm_irfft_kernel(input_c64, dim, n, norm, stream);
    }
    if (input.dtype() == DType::Float64) {
        auto input_c128 = input.to(DType::Complex128);
        return rocm_irfft_kernel(input_c128, dim, n, norm, stream);
    }

    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    bool is_float32 = (input.dtype() == DType::Complex64);

    // rocFFT's simple plan interface requires the FFT dimension to be
    // innermost. Rather than relying solely on the op-layer's
    // transpose-before-dispatch trick (src/ops/fft.cpp irfft(), an unenforced,
    // comment-only contract that JIT-trace replay can bypass — see
    // src/jit/graph.cpp OpId::IRFFT dispatch), honor arbitrary axes directly
    // here too: transpose the target axis to last, recurse, transpose back.
    // Matches CPU/CUDA/OneAPI/Vulkan's arbitrary-axis support.
    if (dim != ndim - 1) {
        Tensor input_t = input.transpose(dim, ndim - 1).contiguous();
        Tensor result_t = rocm_irfft_kernel(input_t, ndim - 1, n, norm, stream);
        return result_t.transpose(dim, ndim - 1).contiguous();
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

    // Input: complex interleaved, batch stride = n/2+1
    size_t in_strides[1] = { 1 };
    size_t in_dist = static_cast<size_t>(expected_complex);
    // Output: real, batch stride = n
    size_t out_strides[1] = { 1 };
    size_t out_dist = static_cast<size_t>(n);

    // Build cache key for C2R plan
    PlanKey c2r_key;
    c2r_key.kind = FFTTransformKind::C2R;
    c2r_key.precision = precision;
    c2r_key.lengths.assign(lengths, lengths + 1);
    c2r_key.batch = static_cast<size_t>(batch);
    c2r_key.in_strides.assign(in_strides, in_strides + 1);
    c2r_key.in_dist = in_dist;
    c2r_key.out_strides.assign(out_strides, out_strides + 1);
    c2r_key.out_dist = out_dist;
    c2r_key.placement = rocfft_placement_notinplace;

    auto cached_plan = get_plan_cache().get_or_create(c2r_key, [&]() -> rocfft_plan {
        RocFFTDescription desc;
        // C2R inverse: input must be hermitian_interleaved, not complex_interleaved.
        ROCFFT_CHECK(rocfft_plan_description_set_data_layout(
            desc.handle,
            rocfft_array_type_hermitian_interleaved,
            rocfft_array_type_real,
            nullptr, nullptr,
            1, in_strides, in_dist,
            1, out_strides, out_dist));

        rocfft_plan plan = nullptr;
        ROCFFT_CHECK(rocfft_plan_create(&plan, rocfft_placement_notinplace,
                                        rocfft_transform_type_real_inverse,
                                        precision, 1, lengths,
                                        static_cast<size_t>(batch), desc.handle));
        return plan;
    });

    // Execute out-of-place C2R
    execute_plan(cached_plan->handle, complex_buf.data_ptr(), output.data_ptr(), stream);

    // Apply normalization
    double scale = get_norm_factor(n, norm, /*is_forward=*/false);
    apply_normalization_real(output, scale, is_float32, stream);

    return output;
}

// ============================================================================
// 2D FFT: Complex-to-Complex forward
//
// DEAD CODE (rocFFT-on variants): OpId::FFT2/IFFT2/FFTN/IFFTN are registered but
// the op layer (src/ops/fft.cpp) decomposes fft2/fftn into sequential 1D fft()
// calls that only dispatch OpId::FFT/IFFT, so this bespoke N-D logic is never
// reached or tested. It additionally has a latent silent-zeros bug: in the
// rocFFT direct-plan path the input is copied into the zeroed output only when
// in_fft_size==out_fft_size && shapes_match, and the 1D fallback only fires when
// the products differ — so a size-permuting n_vec (same product, different
// shape) skips BOTH and the plan runs on a still-zeroed buffer (all-zeros out).
// Fix when reviving: always take the 1D fallback unless shapes match exactly, or
// error out. The native (#else) variants correctly just loop 1D ffts.
// ============================================================================

auto rocm_fft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                      const std::vector<int64_t>& n_vec,
                      const std::string& norm, hipStream_t stream) -> Tensor {
    // BFloat16: upcast to Float32 (Complex64), compute, downcast
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = rocm_fft2_kernel(input_f32, dims, n_vec, norm, stream);
        return result_f32.to(DType::BFloat16);
    }

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

    auto cached_plan = create_c2c_plan(2, lengths, static_cast<size_t>(batch),
                                       /*is_forward=*/true, is_float32,
                                       istride, fft_size, ostride, fft_size);

    execute_plan(cached_plan->handle, output.data_ptr(), output.data_ptr(), stream);

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
    // BFloat16: upcast to Float32 (Complex64), compute, downcast
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = rocm_ifft2_kernel(input_f32, dims, n_vec, norm, stream);
        return result_f32.to(DType::BFloat16);
    }

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

    auto cached_plan = create_c2c_plan(2, lengths, static_cast<size_t>(batch),
                                       /*is_forward=*/false, is_float32,
                                       istride, fft_size, ostride, fft_size);

    execute_plan(cached_plan->handle, output.data_ptr(), output.data_ptr(), stream);

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
    // BFloat16: upcast to Float32 (Complex64), compute, downcast
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = rocm_fftn_kernel(input_f32, dims, n_vec, norm, stream);
        return result_f32.to(DType::BFloat16);
    }

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

    auto cached_plan = create_c2c_plan(static_cast<int>(rank), lengths.data(),
                                       static_cast<size_t>(batch),
                                       /*is_forward=*/true, is_float32,
                                       strides.data(), dist, strides.data(), dist);

    execute_plan(cached_plan->handle, output.data_ptr(), output.data_ptr(), stream);

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
    // BFloat16: upcast to Float32 (Complex64), compute, downcast
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = rocm_ifftn_kernel(input_f32, dims, n_vec, norm, stream);
        return result_f32.to(DType::BFloat16);
    }

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

    auto cached_plan = create_c2c_plan(static_cast<int>(rank), lengths.data(),
                                       static_cast<size_t>(batch),
                                       /*is_forward=*/false, is_float32,
                                       strides.data(), dist, strides.data(), dist);

    execute_plan(cached_plan->handle, output.data_ptr(), output.data_ptr(), stream);

    double scale = get_norm_factor_nd(n_vec, norm, /*is_forward=*/false);
    apply_normalization_complex(output, scale, is_float32, stream);

    return output;
}

} // namespace rocm
} // namespace tenzor

#else // !TENZOR_HAS_ROCFFT — Cooley-Tukey (power-of-2) + Bluestein (general) FFT fallback
#pragma message("WARNING: Building without rocFFT — using slower native HIP FFT fallback")

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "../rocm_error.hpp"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

namespace tenzor {
namespace rocm {

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

/// Compute a 1D kernel grid dimension (ceil(total/block)) and validate that it
/// fits in `int` before truncation. Mirrors the guard in apply_normalization_real
/// so an astronomically large FFT fails loudly instead of wrapping to a small or
/// negative grid that silently leaves part of the tensor untransformed.
inline int grid_dim_checked(int64_t total, int block) {
    int64_t grid = (total + block - 1) / block;
    if (grid > static_cast<int64_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("FFT kernel grid exceeds device limit");
    return static_cast<int>(grid);
}

// ============================================================================
// RAII wrapper for HIP device memory (typed)
// ============================================================================

template<typename T>
struct HipDevicePtr {
    T* ptr = nullptr;

    HipDevicePtr() = default;
    explicit HipDevicePtr(int64_t count) {
        HIP_CHECK(hipMalloc(&ptr, count * sizeof(T)));
    }
    ~HipDevicePtr() { if (ptr) hipFree(ptr); }

    HipDevicePtr(const HipDevicePtr&) = delete;
    HipDevicePtr& operator=(const HipDevicePtr&) = delete;
    HipDevicePtr(HipDevicePtr&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }

    T* get() const { return ptr; }
};

// ============================================================================
// Normalization helpers (same as rocFFT path)
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
void launch_scale(T* data, int64_t numel, T scale, hipStream_t stream) {
    if (numel == 0) return;
    constexpr int block = 256;
    int64_t grid = (numel + block - 1) / block;
    if (grid > static_cast<int64_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("FFT normalization grid exceeds device limit");
    native_scale_kernel<T><<<grid, block, 0, stream>>>(data, numel, scale);
    HIP_CHECK(hipGetLastError());
}

// ============================================================================
// Cooley-Tukey FFT HIP kernels
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
void cooley_tukey_fft_hip(T* data, int64_t N, int64_t batch_size, int64_t batch_stride,
                          T sign, hipStream_t stream) {
    int log2N = 0;
    { int64_t tmp = N; while (tmp > 1) { tmp >>= 1; log2N++; } }

    constexpr int block = 256;

    // Step 1: Bit-reverse permutation
    {
        int64_t total = N * batch_size;
        int grid = grid_dim_checked(total, block);
        bit_reverse_permutation_kernel<T><<<grid, block, 0, stream>>>(
            data, N, batch_size, batch_stride, log2N);
        HIP_CHECK(hipGetLastError());
    }

    // Step 2: Butterfly stages
    int64_t num_butterflies = N / 2;
    int64_t total_butterflies = num_butterflies * batch_size;
    int grid = grid_dim_checked(total_butterflies, block);

    for (int s = 1; s <= log2N; ++s) {
        int64_t stride = static_cast<int64_t>(1) << s;
        int64_t half = stride / 2;

        butterfly_stage_kernel<T><<<grid, block, 0, stream>>>(
            data, N, batch_size, batch_stride, stride, half, num_butterflies, sign);
        HIP_CHECK(hipGetLastError());
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

/// Extract Bluestein result (real input): out[b,k,inner] = a[s][k] * chirp[k]
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
    // Bluestein: X[k] = chirp[k] * (a conv b)[k] -- the extraction multiply
    // uses the plain chirp, not its conjugate. b_buf was already built from
    // conj(chirp) (build_b_kernel/build_b_wrap_kernel above), so conjugating
    // here too cancelled that and left a spurious exp(i*2*pi*k^2/N) phase on
    // every output bin. Matches the CUDA fix (cuda/kernels/fft.cu).
    T c_re = chirp[2 * k];
    T c_im = chirp[2 * k + 1];
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
    // Bluestein: X[k] = chirp[k] * (a conv b)[k] -- the extraction multiply
    // uses the plain chirp, not its conjugate. b_buf was already built from
    // conj(chirp) (build_b_kernel/build_b_wrap_kernel above), so conjugating
    // here too cancelled that and left a spurious exp(i*2*pi*k^2/N) phase on
    // every output bin. Matches the CUDA fix (cuda/kernels/fft.cu).
    T c_re = chirp[2 * k];
    T c_im = chirp[2 * k + 1];
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
void bluestein_fft_hip(const T* d_in, T* d_out,
                       int64_t signal_len, int64_t batch_size, int64_t inner_size,
                       hipStream_t stream) {
    const int64_t N = signal_len;
    const int64_t M = next_pow2(2 * N - 1);
    constexpr int block = 256;

    int64_t total_slices = batch_size * inner_size;

    // Allocate workspace
    HipDevicePtr<T> chirp_owner(2 * N);
    T* chirp = chirp_owner.get();
    HipDevicePtr<T> b_buf_owner(2 * M);
    T* b_buf = b_buf_owner.get();
    HipDevicePtr<T> B_buf_owner(2 * M);
    T* B_buf = B_buf_owner.get();
    HipDevicePtr<T> a_buf_owner(2 * M * total_slices);
    T* a_buf = a_buf_owner.get();

    // Step 1: Generate chirp: chirp[k] = exp(-j * pi * k^2 / N)
    {
        int grid = grid_dim_checked(N, block);
        generate_chirp_kernel<T><<<grid, block, 0, stream>>>(chirp, N, static_cast<T>(-1.0));
        HIP_CHECK(hipGetLastError());
    }

    // Step 2: Build convolution kernel b
    HIP_CHECK(hipMemsetAsync(b_buf, 0, 2 * M * sizeof(T), stream));
    {
        int grid = grid_dim_checked(N, block);
        build_b_kernel<T><<<grid, block, 0, stream>>>(b_buf, chirp, N);
        HIP_CHECK(hipGetLastError());
    }
    if (N > 1) {
        int grid = grid_dim_checked(N - 1, block);
        build_b_wrap_kernel<T><<<grid, block, 0, stream>>>(b_buf, chirp, N, M);
        HIP_CHECK(hipGetLastError());
    }

    // Step 3: B = FFT(b)
    HIP_CHECK(hipMemcpyAsync(B_buf, b_buf, 2 * M * sizeof(T),
                              hipMemcpyDeviceToDevice, stream));
    cooley_tukey_fft_hip(B_buf, M, int64_t(1), int64_t(2 * M),
                         static_cast<T>(-1.0), stream);

    // Step 4: Build a_buf: a[s][k] = x[b,k,inner] * chirp[k]
    HIP_CHECK(hipMemsetAsync(a_buf, 0, 2 * M * total_slices * sizeof(T), stream));
    {
        int64_t total = N * total_slices;
        int grid = grid_dim_checked(total, block);
        bluestein_build_a_real_kernel<T><<<grid, block, 0, stream>>>(
            a_buf, d_in, chirp, N, M, total_slices, inner_size);
        HIP_CHECK(hipGetLastError());
    }

    // Step 5: A = FFT(a) — batched
    cooley_tukey_fft_hip(a_buf, M, total_slices, int64_t(2 * M),
                         static_cast<T>(-1.0), stream);

    // Step 6: Pointwise multiply A *= B
    {
        int64_t total = M * total_slices;
        int grid = grid_dim_checked(total, block);
        pointwise_complex_mul_kernel<T><<<grid, block, 0, stream>>>(a_buf, B_buf, M, total_slices);
        HIP_CHECK(hipGetLastError());
    }

    // Step 7: IFFT via forward FFT with sign=+1, divide by M
    cooley_tukey_fft_hip(a_buf, M, total_slices, int64_t(2 * M),
                         static_cast<T>(1.0), stream);
    {
        T inv_M = static_cast<T>(1.0) / static_cast<T>(M);
        int64_t total = 2 * M * total_slices;
        launch_scale(a_buf, total, inv_M, stream);
    }

    // Step 8: Extract result: out[b,k,inner] = a[s][k] * conj(chirp[k])
    {
        int64_t total = N * total_slices;
        int grid = grid_dim_checked(total, block);
        bluestein_extract_real_kernel<T><<<grid, block, 0, stream>>>(
            d_out, a_buf, chirp, N, M, total_slices, inner_size);
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipStreamSynchronize(stream));
}

// ============================================================================
// Bluestein FFT — complex input
// ============================================================================

/// Bluestein FFT for non-power-of-2 sizes with complex input.
/// Input:  d_in  — interleaved complex, layout [batch_size, signal_len, inner_size, 2]
/// Output: d_out — interleaved complex, layout [batch_size, signal_len, inner_size, 2]
/// sign = -1 for forward, +1 for inverse (before normalization).
template<typename T>
void bluestein_fft_complex_hip(const T* d_in, T* d_out,
                               int64_t signal_len, int64_t batch_size, int64_t inner_size,
                               T sign, hipStream_t stream) {
    const int64_t N = signal_len;
    const int64_t M = next_pow2(2 * N - 1);
    constexpr int block = 256;

    int64_t total_slices = batch_size * inner_size;

    HipDevicePtr<T> chirp_owner(2 * N);
    T* chirp = chirp_owner.get();
    HipDevicePtr<T> b_buf_owner(2 * M);
    T* b_buf = b_buf_owner.get();
    HipDevicePtr<T> B_buf_owner(2 * M);
    T* B_buf = B_buf_owner.get();
    HipDevicePtr<T> a_buf_owner(2 * M * total_slices);
    T* a_buf = a_buf_owner.get();

    // Step 1: chirp[k] = exp(sign * j * pi * k^2 / N)
    {
        int grid = grid_dim_checked(N, block);
        generate_chirp_kernel<T><<<grid, block, 0, stream>>>(chirp, N, sign);
        HIP_CHECK(hipGetLastError());
    }

    // Step 2: b[k] = conj(chirp[k])
    HIP_CHECK(hipMemsetAsync(b_buf, 0, 2 * M * sizeof(T), stream));
    {
        int grid = grid_dim_checked(N, block);
        build_b_kernel<T><<<grid, block, 0, stream>>>(b_buf, chirp, N);
        HIP_CHECK(hipGetLastError());
    }
    if (N > 1) {
        int grid = grid_dim_checked(N - 1, block);
        build_b_wrap_kernel<T><<<grid, block, 0, stream>>>(b_buf, chirp, N, M);
        HIP_CHECK(hipGetLastError());
    }

    // Step 3: B = FFT(b)
    HIP_CHECK(hipMemcpyAsync(B_buf, b_buf, 2 * M * sizeof(T),
                              hipMemcpyDeviceToDevice, stream));
    cooley_tukey_fft_hip(B_buf, M, int64_t(1), int64_t(2 * M),
                         static_cast<T>(-1.0), stream);

    // Step 4: a[s][k] = x[b,k,inner] * chirp[k] (complex multiply)
    HIP_CHECK(hipMemsetAsync(a_buf, 0, 2 * M * total_slices * sizeof(T), stream));
    {
        int64_t total = N * total_slices;
        int grid = grid_dim_checked(total, block);
        bluestein_build_a_complex_kernel<T><<<grid, block, 0, stream>>>(
            a_buf, d_in, chirp, N, M, total_slices, inner_size);
        HIP_CHECK(hipGetLastError());
    }

    // Step 5: A = FFT(a)
    cooley_tukey_fft_hip(a_buf, M, total_slices, int64_t(2 * M),
                         static_cast<T>(-1.0), stream);

    // Step 6: Pointwise multiply A *= B
    {
        int64_t total = M * total_slices;
        int grid = grid_dim_checked(total, block);
        pointwise_complex_mul_kernel<T><<<grid, block, 0, stream>>>(a_buf, B_buf, M, total_slices);
        HIP_CHECK(hipGetLastError());
    }

    // Step 7: IFFT via forward FFT with sign=+1, divide by M
    cooley_tukey_fft_hip(a_buf, M, total_slices, int64_t(2 * M),
                         static_cast<T>(1.0), stream);
    {
        T inv_M = static_cast<T>(1.0) / static_cast<T>(M);
        int64_t total = 2 * M * total_slices;
        launch_scale(a_buf, total, inv_M, stream);
    }

    // Step 8: Extract result
    {
        int64_t total = N * total_slices;
        int grid = grid_dim_checked(total, block);
        bluestein_extract_complex_kernel<T><<<grid, block, 0, stream>>>(
            d_out, a_buf, chirp, N, M, total_slices, inner_size);
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipStreamSynchronize(stream));
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
    int64_t src_idx = (b * signal_len * inner_size + k * inner_size + inner) * 2;
    int64_t dst_idx = (b * out_len * inner_size + k * inner_size + inner) * 2;
    d_out[dst_idx]     = d_full[src_idx];
    d_out[dst_idx + 1] = d_full[src_idx + 1];
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
// 1D FFT: Complex-to-Complex forward (native HIP fallback)
// ============================================================================

auto rocm_fft_kernel(const Tensor& input, int64_t dim, int64_t n,
                     const std::string& norm, hipStream_t stream) -> Tensor {
    // Float16/BFloat16: widen to Complex64, compute, downcast (taking the real
    // part). The native fallback only operates on Complex64/Complex128 buffers,
    // classified by `is_float32 = (dtype == Complex64)`; a half input would
    // otherwise fall into the Complex128 branch and be reinterpret_cast as
    // Complex128 (16 bytes/element vs its 2-byte/element allocation), reading
    // far out of bounds. We widen to Complex64 (not Float32) so the recursion
    // lands in the complex branch; mirrors the rocFFT build's half handling and
    // the op-layer's real->complex promotion so direct/internal kernel calls
    // stay memory-safe and produce consistent results.
    if (input.dtype() == DType::BFloat16 || input.dtype() == DType::Float16) {
        auto input_c64 = input.to(DType::Float32).to(DType::Complex64);
        auto result_c64 = rocm_fft_kernel(input_c64, dim, n, norm, stream);
        return result_c64.to(input.dtype());
    }

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
        HIP_CHECK(hipMemsetAsync(output.data_ptr(), 0,
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
            HIP_CHECK(hipMemcpyAsync(dst, src,
                copy_len * inner_size * elem_size,
                hipMemcpyDeviceToDevice, stream));
        }
        work_input = output;
    } else {
        HIP_CHECK(hipMemcpyAsync(output.data_ptr(), input.data_ptr(),
            input.numel() * dtype_size(out_dtype),
            hipMemcpyDeviceToDevice, stream));
        work_input = output;
    }

    // Compute layout for FFT dispatch
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= out_shape[i];
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= out_shape[i];
    int64_t batch = outer_size * inner_size;

    bool use_cooley_tukey = is_power_of_2(N_out) && inner_size == 1;

    if (use_cooley_tukey) {
        if (is_float32) {
            cooley_tukey_fft_hip(reinterpret_cast<float*>(output.data_ptr()),
                                 N_out, outer_size, int64_t(2 * N_out),
                                 -1.0f, stream);
        } else {
            cooley_tukey_fft_hip(reinterpret_cast<double*>(output.data_ptr()),
                                 N_out, outer_size, int64_t(2 * N_out),
                                 -1.0, stream);
        }
    } else {
        if (is_float32) {
            float* data_ptr = reinterpret_cast<float*>(output.data_ptr());
            HipDevicePtr<float> tmp_out(output.numel() * 2);
            bluestein_fft_complex_hip(data_ptr, tmp_out.get(),
                                      N_out, outer_size, inner_size,
                                      -1.0f, stream);
            HIP_CHECK(hipMemcpyAsync(data_ptr, tmp_out.get(),
                output.numel() * 2 * sizeof(float),
                hipMemcpyDeviceToDevice, stream));
        } else {
            double* data_ptr = reinterpret_cast<double*>(output.data_ptr());
            HipDevicePtr<double> tmp_out(output.numel() * 2);
            bluestein_fft_complex_hip(data_ptr, tmp_out.get(),
                                      N_out, outer_size, inner_size,
                                      -1.0, stream);
            HIP_CHECK(hipMemcpyAsync(data_ptr, tmp_out.get(),
                output.numel() * 2 * sizeof(double),
                hipMemcpyDeviceToDevice, stream));
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
// 1D IFFT: Complex-to-Complex inverse (native HIP fallback)
// ============================================================================

auto rocm_ifft_kernel(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm, hipStream_t stream) -> Tensor {
    // Float16/BFloat16: widen to Complex64, compute, downcast (taking the real
    // part). The native fallback only operates on Complex64/Complex128 buffers,
    // classified by `is_float32 = (dtype == Complex64)`; a half input would
    // otherwise fall into the Complex128 branch and be reinterpret_cast as
    // Complex128 (16 bytes/element vs its 2-byte/element allocation), reading
    // far out of bounds. Mirrors the rocFFT build's half handling.
    if (input.dtype() == DType::BFloat16 || input.dtype() == DType::Float16) {
        auto input_c64 = input.to(DType::Float32).to(DType::Complex64);
        auto result_c64 = rocm_ifft_kernel(input_c64, dim, n, norm, stream);
        return result_c64.to(input.dtype());
    }

    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("FFT: dimension out of range");
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
        HIP_CHECK(hipMemsetAsync(output.data_ptr(), 0,
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
            HIP_CHECK(hipMemcpyAsync(dst, src,
                copy_len * inner_size * elem_size,
                hipMemcpyDeviceToDevice, stream));
        }
    } else {
        HIP_CHECK(hipMemcpyAsync(output.data_ptr(), input.data_ptr(),
            input.numel() * dtype_size(out_dtype),
            hipMemcpyDeviceToDevice, stream));
    }

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= out_shape[i];
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= out_shape[i];

    bool use_cooley_tukey = is_power_of_2(N_out) && inner_size == 1;

    if (use_cooley_tukey) {
        if (is_float32) {
            cooley_tukey_fft_hip(reinterpret_cast<float*>(output.data_ptr()),
                                 N_out, outer_size, int64_t(2 * N_out),
                                 1.0f, stream);
        } else {
            cooley_tukey_fft_hip(reinterpret_cast<double*>(output.data_ptr()),
                                 N_out, outer_size, int64_t(2 * N_out),
                                 1.0, stream);
        }
    } else {
        if (is_float32) {
            float* data_ptr = reinterpret_cast<float*>(output.data_ptr());
            HipDevicePtr<float> tmp_out(output.numel() * 2);
            bluestein_fft_complex_hip(data_ptr, tmp_out.get(),
                                      N_out, outer_size, inner_size,
                                      1.0f, stream);
            HIP_CHECK(hipMemcpyAsync(data_ptr, tmp_out.get(),
                output.numel() * 2 * sizeof(float),
                hipMemcpyDeviceToDevice, stream));
        } else {
            double* data_ptr = reinterpret_cast<double*>(output.data_ptr());
            HipDevicePtr<double> tmp_out(output.numel() * 2);
            bluestein_fft_complex_hip(data_ptr, tmp_out.get(),
                                      N_out, outer_size, inner_size,
                                      1.0, stream);
            HIP_CHECK(hipMemcpyAsync(data_ptr, tmp_out.get(),
                output.numel() * 2 * sizeof(double),
                hipMemcpyDeviceToDevice, stream));
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
// 1D RFFT: Real-to-Complex forward (native HIP fallback)
// ============================================================================

auto rocm_rfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm, hipStream_t stream) -> Tensor {
    // Float16/BFloat16: upcast to Float32, compute rfft; result is Complex64
    // (no downcast needed — rfft always produces complex output). The native
    // fallback classifies precision with `is_float32 = (dtype == Float32)`; a
    // half input would otherwise fall into the Float64 branch and be
    // reinterpret_cast as double (8 bytes/element vs its 2-byte/element
    // allocation), reading out of bounds. Mirrors the rocFFT build.
    if (input.dtype() == DType::BFloat16 || input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        return rocm_rfft_kernel(input_f32, dim, n, norm, stream);
    }

    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("RFFT/IRFFT: dimension out of range");
    }
    bool is_float32 = (input.dtype() == DType::Float32);

    int64_t N_in = shape[dim];
    int64_t N_out_complex = n / 2 + 1;

    // Honor arbitrary axes directly here too (mirrors the rocFFT build above
    // and CPU/CUDA/OneAPI/Vulkan's arbitrary-axis support): transpose the
    // target axis to last, recurse, transpose back.
    if (dim != ndim - 1) {
        Tensor input_t = input.transpose(dim, ndim - 1).contiguous();
        Tensor result_t = rocm_rfft_kernel(input_t, ndim - 1, n, norm, stream);
        return result_t.transpose(dim, ndim - 1).contiguous();
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

    // Strategy: compute full N-point FFT on real data, then truncate to N/2+1 bins.
    bool use_cooley_tukey = is_power_of_2(n);
    constexpr int block_size = 256;

    if (is_float32) {
        if (use_cooley_tukey) {
            // Pack real into interleaved complex
            HipDevicePtr<float> d_buf(2 * n * batch);
            int64_t total = n * batch;
            {
                int grid = grid_dim_checked(total, block_size);
                pack_real_to_complex_kernel<float><<<grid, block_size, 0, stream>>>(
                    d_buf.get(), static_cast<const float*>(real_buf.data_ptr()), total);
                HIP_CHECK(hipGetLastError());
            }

            cooley_tukey_fft_hip(d_buf.get(), n, batch, int64_t(2 * n), -1.0f, stream);

            // Apply normalization to full FFT before truncation
            double scale = get_norm_factor(n, norm, true);
            if (scale != 1.0) {
                launch_scale(d_buf.get(), 2 * total, static_cast<float>(scale), stream);
            }

            // Truncate to first N/2+1 bins
            {
                int64_t trunc_total = batch * N_out_complex;
                int grid = grid_dim_checked(trunc_total, block_size);
                truncate_rfft_kernel<float><<<grid, block_size, 0, stream>>>(
                    reinterpret_cast<float*>(output.data_ptr()),
                    d_buf.get(), N_out_complex, n, batch);
                HIP_CHECK(hipGetLastError());
            }
        } else {
            // Bluestein: compute full N-point FFT on real data, then truncate
            int64_t full_complex_numel = batch * n * 2;  // inner_size=1 for last dim
            HipDevicePtr<float> d_full(full_complex_numel);
            HIP_CHECK(hipMemsetAsync(d_full.get(), 0,
                full_complex_numel * sizeof(float), stream));

            bluestein_fft_hip(static_cast<const float*>(real_buf.data_ptr()),
                              d_full.get(), n, batch, int64_t(1), stream);

            double scale = get_norm_factor(n, norm, true);
            if (scale != 1.0) {
                launch_scale(d_full.get(), full_complex_numel, static_cast<float>(scale), stream);
            }

            // Truncate to N/2+1
            {
                int64_t trunc_total = batch * N_out_complex;
                int grid = grid_dim_checked(trunc_total, block_size);
                truncate_rfft_kernel<float><<<grid, block_size, 0, stream>>>(
                    reinterpret_cast<float*>(output.data_ptr()),
                    d_full.get(), N_out_complex, n, batch);
                HIP_CHECK(hipGetLastError());
            }
        }
    } else {
        // Float64 path
        if (use_cooley_tukey) {
            HipDevicePtr<double> d_buf(2 * n * batch);
            int64_t total = n * batch;
            {
                int grid = grid_dim_checked(total, block_size);
                pack_real_to_complex_kernel<double><<<grid, block_size, 0, stream>>>(
                    d_buf.get(), static_cast<const double*>(real_buf.data_ptr()), total);
                HIP_CHECK(hipGetLastError());
            }

            cooley_tukey_fft_hip(d_buf.get(), n, batch, int64_t(2 * n), -1.0, stream);

            double scale = get_norm_factor(n, norm, true);
            if (scale != 1.0) {
                launch_scale(d_buf.get(), 2 * total, scale, stream);
            }

            {
                int64_t trunc_total = batch * N_out_complex;
                int grid = grid_dim_checked(trunc_total, block_size);
                truncate_rfft_kernel<double><<<grid, block_size, 0, stream>>>(
                    reinterpret_cast<double*>(output.data_ptr()),
                    d_buf.get(), N_out_complex, n, batch);
                HIP_CHECK(hipGetLastError());
            }
        } else {
            int64_t full_complex_numel = batch * n * 2;
            HipDevicePtr<double> d_full(full_complex_numel);
            HIP_CHECK(hipMemsetAsync(d_full.get(), 0,
                full_complex_numel * sizeof(double), stream));

            bluestein_fft_hip(static_cast<const double*>(real_buf.data_ptr()),
                              d_full.get(), n, batch, int64_t(1), stream);

            double scale = get_norm_factor(n, norm, true);
            if (scale != 1.0) {
                launch_scale(d_full.get(), full_complex_numel, scale, stream);
            }

            {
                int64_t trunc_total = batch * N_out_complex;
                int grid = grid_dim_checked(trunc_total, block_size);
                truncate_rfft_kernel<double><<<grid, block_size, 0, stream>>>(
                    reinterpret_cast<double*>(output.data_ptr()),
                    d_full.get(), N_out_complex, n, batch);
                HIP_CHECK(hipGetLastError());
            }
        }
    }

    return output;
}

// ============================================================================
// 1D IRFFT: Complex-to-Real inverse (native HIP fallback)
// ============================================================================

auto rocm_irfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                       const std::string& norm, hipStream_t stream) -> Tensor {
    // Float16/BFloat16: upcast to Float32 (-> Complex64), compute irfft, then
    // downcast the real result back to the half dtype. The native fallback only
    // handles Complex64/Complex128 input (classified by `is_float32 = (dtype ==
    // Complex64)`); a half input would otherwise fall into the Complex128 branch
    // and be reinterpret_cast as Complex128, reading far out of bounds.
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = rocm_irfft_kernel(input_f32, dim, n, norm, stream);
        return result_f32.to(DType::BFloat16);
    }
    if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = rocm_irfft_kernel(input_f32, dim, n, norm, stream);
        return result_f32.to(DType::Float16);
    }
    // Real-to-complex promotion. The CPU and CUDA irfft kernels accept Float32
    // or Float64 input by first casting to Complex64/Complex128 (imag = 0), so
    // mirror that here (and the rocFFT build) — without this a real input is
    // misclassified as the wrong precision and the buffer is read at the wrong
    // element width.
    if (input.dtype() == DType::Float32) {
        auto input_c64 = input.to(DType::Complex64);
        return rocm_irfft_kernel(input_c64, dim, n, norm, stream);
    }
    if (input.dtype() == DType::Float64) {
        auto input_c128 = input.to(DType::Complex128);
        return rocm_irfft_kernel(input_c128, dim, n, norm, stream);
    }

    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    bool is_float32 = (input.dtype() == DType::Complex64);

    // Honor arbitrary axes directly here too (mirrors the rocFFT build above
    // and CPU/CUDA/OneAPI/Vulkan's arbitrary-axis support): transpose the
    // target axis to last, recurse, transpose back.
    if (dim != ndim - 1) {
        Tensor input_t = input.transpose(dim, ndim - 1).contiguous();
        Tensor result_t = rocm_irfft_kernel(input_t, ndim - 1, n, norm, stream);
        return result_t.transpose(dim, ndim - 1).contiguous();
    }

    int64_t N_in = shape[dim];  // n/2 + 1 complex elements
    int64_t expected_complex = n / 2 + 1;

    // Copy input into work buffer sized for expected_complex
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
            HipDevicePtr<float> d_buf(2 * n * batch);
            {
                int64_t total = batch * n;
                int grid = grid_dim_checked(total, block_size);
                reconstruct_spectrum_kernel<float><<<grid, block_size, 0, stream>>>(
                    d_buf.get(), d_in, n, expected_complex, batch);
                HIP_CHECK(hipGetLastError());
            }

            cooley_tukey_fft_hip(d_buf.get(), n, batch, int64_t(2 * n), 1.0f, stream);

            {
                int64_t total = batch * n;
                int grid = grid_dim_checked(total, block_size);
                extract_real_scaled_kernel<float><<<grid, block_size, 0, stream>>>(
                    output.data<float>(), d_buf.get(), n, batch, scale);
                HIP_CHECK(hipGetLastError());
            }
        } else {
            // Bluestein: reconstruct full spectrum, inverse FFT, extract real
            int64_t full_complex_numel = batch * n * 2;
            HipDevicePtr<float> d_full(full_complex_numel);
            {
                int64_t total = batch * n;
                int grid = grid_dim_checked(total, block_size);
                reconstruct_spectrum_kernel<float><<<grid, block_size, 0, stream>>>(
                    d_full.get(), d_in, n, expected_complex, batch);
                HIP_CHECK(hipGetLastError());
            }

            HipDevicePtr<float> d_ifft(full_complex_numel);
            HIP_CHECK(hipMemsetAsync(d_ifft.get(), 0,
                full_complex_numel * sizeof(float), stream));
            bluestein_fft_complex_hip(d_full.get(), d_ifft.get(),
                                      n, batch, int64_t(1), 1.0f, stream);

            {
                int64_t total = batch * n;
                int grid = grid_dim_checked(total, block_size);
                extract_real_scaled_kernel<float><<<grid, block_size, 0, stream>>>(
                    output.data<float>(), d_ifft.get(), n, batch, scale);
                HIP_CHECK(hipGetLastError());
            }
        }
    } else {
        const double* d_in = reinterpret_cast<const double*>(complex_buf.data_ptr());
        double scale = scale_d;

        if (use_cooley_tukey) {
            HipDevicePtr<double> d_buf(2 * n * batch);
            {
                int64_t total = batch * n;
                int grid = grid_dim_checked(total, block_size);
                reconstruct_spectrum_kernel<double><<<grid, block_size, 0, stream>>>(
                    d_buf.get(), d_in, n, expected_complex, batch);
                HIP_CHECK(hipGetLastError());
            }

            cooley_tukey_fft_hip(d_buf.get(), n, batch, int64_t(2 * n), 1.0, stream);

            {
                int64_t total = batch * n;
                int grid = grid_dim_checked(total, block_size);
                extract_real_scaled_kernel<double><<<grid, block_size, 0, stream>>>(
                    output.data<double>(), d_buf.get(), n, batch, scale);
                HIP_CHECK(hipGetLastError());
            }
        } else {
            int64_t full_complex_numel = batch * n * 2;
            HipDevicePtr<double> d_full(full_complex_numel);
            {
                int64_t total = batch * n;
                int grid = grid_dim_checked(total, block_size);
                reconstruct_spectrum_kernel<double><<<grid, block_size, 0, stream>>>(
                    d_full.get(), d_in, n, expected_complex, batch);
                HIP_CHECK(hipGetLastError());
            }

            HipDevicePtr<double> d_ifft(full_complex_numel);
            HIP_CHECK(hipMemsetAsync(d_ifft.get(), 0,
                full_complex_numel * sizeof(double), stream));
            bluestein_fft_complex_hip(d_full.get(), d_ifft.get(),
                                      n, batch, int64_t(1), 1.0, stream);

            {
                int64_t total = batch * n;
                int grid = grid_dim_checked(total, block_size);
                extract_real_scaled_kernel<double><<<grid, block_size, 0, stream>>>(
                    output.data<double>(), d_ifft.get(), n, batch, scale);
                HIP_CHECK(hipGetLastError());
            }
        }
    }

    return output;
}

// ============================================================================
// 2D FFT: Complex-to-Complex forward (native HIP fallback)
// ============================================================================

auto rocm_fft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                      const std::vector<int64_t>& n_vec,
                      const std::string& norm, hipStream_t stream) -> Tensor {
    if (dims.size() != 2) {
        throw std::runtime_error("native rocm fft2: expected exactly 2 dimensions");
    }
    // Apply 1D FFT sequentially along each dimension
    Tensor result = input;
    for (size_t i = 0; i < dims.size(); ++i) {
        result = rocm_fft_kernel(result, dims[i], n_vec[i], norm, stream);
    }
    return result;
}

// ============================================================================
// 2D IFFT: Complex-to-Complex inverse (native HIP fallback)
// ============================================================================

auto rocm_ifft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& n_vec,
                       const std::string& norm, hipStream_t stream) -> Tensor {
    if (dims.size() != 2) {
        throw std::runtime_error("native rocm ifft2: expected exactly 2 dimensions");
    }
    Tensor result = input;
    for (size_t i = 0; i < dims.size(); ++i) {
        result = rocm_ifft_kernel(result, dims[i], n_vec[i], norm, stream);
    }
    return result;
}

// ============================================================================
// N-D FFT: Complex-to-Complex forward (native HIP fallback)
// ============================================================================

auto rocm_fftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                      const std::vector<int64_t>& n_vec,
                      const std::string& norm, hipStream_t stream) -> Tensor {
    Tensor result = input;
    for (size_t i = 0; i < dims.size(); ++i) {
        result = rocm_fft_kernel(result, dims[i], n_vec[i], norm, stream);
    }
    return result;
}

// ============================================================================
// N-D IFFT: Complex-to-Complex inverse (native HIP fallback)
// ============================================================================

auto rocm_ifftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& n_vec,
                       const std::string& norm, hipStream_t stream) -> Tensor {
    Tensor result = input;
    for (size_t i = 0; i < dims.size(); ++i) {
        result = rocm_ifft_kernel(result, dims[i], n_vec[i], norm, stream);
    }
    return result;
}

} // namespace rocm
} // namespace tenzor

#endif // TENZOR_HAS_ROCFFT
