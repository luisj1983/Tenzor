#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/jit/tracer.hpp"
#ifdef TENZOR_USE_MKL
#include <mkl.h>
#include <mkl_lapacke.h>
#elif defined(TENZOR_USE_LAPACKE)
#include <lapacke.h>
#endif
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <vector>
#include <algorithm>

namespace tenzor::linalg {

namespace {

/// Try GPU dispatch for a single-output linalg op. Returns true and sets result on success.
/// CPU linalg ops are registered in the CPU dispatch table (cpu_kernel_registry.cpp) as
/// wrappers around the LAPACKE implementations below. GPU backends (CUDA/ROCm/Vulkan/OneAPI)
/// register native kernels.
/// Note: The CPU dispatch table wrappers call these functions, which check device==CPU
/// and skip try_gpu_dispatch, so there is no circular dispatch.
// Complex linalg (qr/svd/eig/det/inv/solve/cholesky/...) is, for most GPU backends
// (ROCm/Vulkan/OneAPI) and most ops, implemented only on the CPU LAPACK path; those
// GPU kernels handle real floats and throw for Complex64/Complex128. Rather than let
// that throw escape (which crashed lstsq and made every complex linalg op fail on GPU
// devices), route complex inputs through the CPU implementation and move the result
// back to the caller's device. This is transparent to callers and keeps device
// semantics.
//
// CUDA is the exception for a handful of ops: det, inv, solve, eig, and
// solve_triangular have genuine cuSOLVER/cuBLAS-backed Complex64/Complex128 kernels
// (src/backends/cuda/kernels/linalg.cu — cusolverDnCgetrf/Zgetrf, Xgeev, cublasCtrsm/
// Ztrsm). For those specific ops on a CUDA device, dispatch straight to the CUDA
// kernel instead of round-tripping through the CPU — see
// cuda_has_native_complex_kernel() below. All other ops (svd/qr/eigh/cholesky/lu/
// lu_solve/ldl_factor/ldl_solve/geqrf/ormqr/householder) have no native complex
// implementation on CUDA either, so they keep going through the CPU reroute.
static bool is_complex_dtype(DType dt) {
    return dt == DType::Complex64 || dt == DType::Complex128;
}

// Returns true for the small set of linalg ops that have a genuine, working
// Complex64/Complex128 CUDA implementation (verified against
// src/backends/cuda/kernels/linalg.cu). Keep in sync with that file: if a future
// change adds/removes native complex support for an op, update this list too.
static bool cuda_has_native_complex_kernel(OpId op) {
    switch (op) {
        case OpId::LinalgDet:
        case OpId::LinalgInv:
        case OpId::LinalgSolve:
        case OpId::LinalgEig:
        case OpId::SolveTriangular:
            return true;
        default:
            return false;
    }
}

bool try_gpu_dispatch(OpId op, std::span<const Tensor> inputs,
                      const OpAttributes& attrs, Tensor& result) {
    const Device dev = inputs[0].device();
    if (dev.type == Device::Type::CPU) return false;
    bool needs_cpu_reroute = is_complex_dtype(inputs[0].dtype()) &&
        !(dev.type == Device::Type::CUDA && cuda_has_native_complex_kernel(op));
    if (needs_cpu_reroute) {
        std::vector<Tensor> cpu_inputs;
        cpu_inputs.reserve(inputs.size());
        for (const auto& t : inputs) cpu_inputs.push_back(t.to(Device::cpu()));
        result = dispatch_single(op, cpu_inputs, attrs);
        if (result.device() != dev) result = result.to(dev);
        return true;
    }
    result = dispatch_single(op, inputs, attrs);
    return true;
}

/// Try GPU dispatch for a multi-output linalg op. Returns true and sets results on success.
bool try_gpu_dispatch_multi(OpId op, std::span<const Tensor> inputs,
                            const OpAttributes& attrs, std::vector<Tensor>& results) {
    const Device dev = inputs[0].device();
    if (dev.type == Device::Type::CPU) return false;
    bool needs_cpu_reroute = is_complex_dtype(inputs[0].dtype()) &&
        !(dev.type == Device::Type::CUDA && cuda_has_native_complex_kernel(op));
    if (needs_cpu_reroute) {
        std::vector<Tensor> cpu_inputs;
        cpu_inputs.reserve(inputs.size());
        for (const auto& t : inputs) cpu_inputs.push_back(t.to(Device::cpu()));
        results = dispatch(op, cpu_inputs, attrs);
        for (auto& r : results) if (r.device() != dev) r = r.to(dev);
        return true;
    }
    results = dispatch(op, inputs, attrs);
    return true;
}

// Check if dtype is a low-precision float that needs upcasting for LAPACK
bool needs_upcast(DType dt) {
    return dt == DType::Float16 || dt == DType::BFloat16;
}

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
[[noreturn]] void throw_no_lapack(const char* fn) {
    throw std::runtime_error(
        std::string("linalg::") + fn + ": requires MKL or LAPACKE. "
        "Build with TENZOR_USE_MKL or install liblapacke-dev.");
}
#endif

// Ensure tensor is contiguous Float32/Float64/Complex64/Complex128 on CPU,
// return a working copy. Float16 and BFloat16 are upcast to Float32.
// Only called for CPU tensors — GPU tensors are handled by try_gpu_dispatch.
auto prepare_matrix(const Tensor& A) -> Tensor {
    if (A.device().type != Device::Type::CPU) {
        throw std::logic_error("prepare_matrix: expected CPU tensor, got GPU tensor. "
                               "GPU linalg ops should go through backend dispatch.");
    }
    auto dt = A.dtype();
    if (dt != DType::Float32 && dt != DType::Float64 &&
        dt != DType::Float16 && dt != DType::BFloat16 &&
        dt != DType::Complex64 && dt != DType::Complex128) {
        throw std::runtime_error("linalg: only Float32, Float64, Float16, BFloat16, "
                                 "Complex64, and Complex128 are supported");
    }
    // Upcast low-precision floats to Float32 for LAPACK compatibility
    auto cpu_tensor = A;
    if (needs_upcast(cpu_tensor.dtype())) {
        cpu_tensor = cpu_tensor.to(DType::Float32);
    }
    return cpu_tensor.contiguous().clone();
}

// Downcast result tensor back to original dtype if it was upcast
auto maybe_downcast(const Tensor& result, DType original_dtype) -> Tensor {
    if (needs_upcast(original_dtype) && result.dtype() != original_dtype) {
        return result.to(original_dtype);
    }
    return result;
}

#if defined(TENZOR_USE_MKL) || defined(TENZOR_USE_LAPACKE)
// MKL_Complex8/MKL_Complex16 have the same binary layout as
// std::complex<float>/std::complex<double> (two consecutive floats/doubles).
// These helpers let us obtain a lapack_complex_float* from a Complex64 tensor
// without copying.
inline lapack_complex_float* c64_ptr(Tensor& t) {
    return reinterpret_cast<lapack_complex_float*>(t.data<std::complex<float>>());
}
inline const lapack_complex_float* c64_cptr(const Tensor& t) {
    return reinterpret_cast<const lapack_complex_float*>(t.data<std::complex<float>>());
}
inline lapack_complex_double* c128_ptr(Tensor& t) {
    return reinterpret_cast<lapack_complex_double*>(t.data<std::complex<double>>());
}
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE

auto check_square(const Tensor& A) -> std::pair<int64_t, int64_t> {
    auto shape = A.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) {
        throw std::invalid_argument("linalg: input must be at least 2D");
    }
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];
    if (m != n) {
        throw std::invalid_argument("linalg: expected square matrix, got " +
            std::to_string(m) + "x" + std::to_string(n));
    }
    return {m, ndim};
}

// Validate that a matrix dimension fits in lapack_int (32-bit under the LP64
// LAPACK ABI) before it is passed to a LAPACKE call. The surrounding C++
// offset math uses true int64, so a silent 32-bit truncation here would cause
// LAPACK to operate on a wrong/under-sized extent. Mirrors the sparse layer's
// mkl_index_fits guard. Returns the value cast to lapack_int.
#if defined(TENZOR_USE_MKL) || defined(TENZOR_USE_LAPACKE)
[[maybe_unused]] inline lapack_int to_lapack_int(int64_t v, const char* what) {
    if (v < std::numeric_limits<lapack_int>::min() ||
        v > std::numeric_limits<lapack_int>::max()) {
        throw std::overflow_error(
            std::string("linalg: dimension ") + what + " (" + std::to_string(v) +
            ") exceeds the LAPACK 32-bit index range");
    }
    return static_cast<lapack_int>(v);
}
#endif

// Compute batch size (product of all dims except last two)
auto batch_size(const Tensor& A) -> int64_t {
    auto shape = A.shape();
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        batch *= shape[i];
    }
    return batch;
}

} // anonymous namespace

// ============================================================================
// JIT tracer visibility for CPU LAPACK-backed linalg ops (findings.txt JIT-R133)
// ============================================================================
// The GPU/complex path already reaches the interceptor via try_gpu_dispatch's
// dispatch_single()/dispatch() call, so once the corresponding OpId::Linalg*
// is mapped in tracing_interceptor.cpp's opid_to_optype, that path is traced
// automatically. The real-dtype CPU path below never calls dispatch() at all
// (try_gpu_dispatch returns false for CPU tensors and execution falls straight
// through to the hand-written LAPACKE code), so it is completely invisible to
// the tracer: a traced call silently bakes its real, input-dependent result as
// a trace-time constant (severing the graph's dependence on the runtime
// input), exactly the mechanism already fixed elsewhere for RMSNorm/Embedding/
// detection ops. These helpers record the correct (input -> output) node right
// after that CPU computation completes, mirroring jit_record_rms_norm
// (src/nn/layers/normalization.cpp) and detection.cpp's manual recorders.
namespace {

inline auto jit_record_linalg1(::tenzor::jit::OpType type, const Tensor& a,
                                const Tensor& out) -> void {
    auto& tracer = ::tenzor::jit::Tracer::get_instance();
    if (!tracer.is_tracing()) return;
    auto a_id = tracer.register_tensor(a);
    auto o_id = tracer.register_new_tensor(out);
    ::tenzor::jit::TracedOp op(type, {a_id}, {o_id});
    tracer.record_op(std::move(op));
}

inline auto jit_record_linalg1_bool_attr(::tenzor::jit::OpType type, const Tensor& a,
                                          const Tensor& out, const char* attr_name,
                                          bool attr_val) -> void {
    auto& tracer = ::tenzor::jit::Tracer::get_instance();
    if (!tracer.is_tracing()) return;
    auto a_id = tracer.register_tensor(a);
    auto o_id = tracer.register_new_tensor(out);
    ::tenzor::jit::TracedOp op(type, {a_id}, {o_id});
    op.bool_attrs[attr_name] = attr_val;
    tracer.record_op(std::move(op));
}

inline auto jit_record_linalg2to1(::tenzor::jit::OpType type, const Tensor& a,
                                   const Tensor& b, const Tensor& out) -> void {
    auto& tracer = ::tenzor::jit::Tracer::get_instance();
    if (!tracer.is_tracing()) return;
    auto a_id = tracer.register_tensor(a);
    auto b_id = tracer.register_tensor(b);
    auto o_id = tracer.register_new_tensor(out);
    ::tenzor::jit::TracedOp op(type, {a_id, b_id}, {o_id});
    tracer.record_op(std::move(op));
}

inline auto jit_record_linalg1to2(::tenzor::jit::OpType type, const Tensor& a,
                                   const Tensor& out0, const Tensor& out1) -> void {
    auto& tracer = ::tenzor::jit::Tracer::get_instance();
    if (!tracer.is_tracing()) return;
    auto a_id = tracer.register_tensor(a);
    auto o0_id = tracer.register_new_tensor(out0);
    auto o1_id = tracer.register_new_tensor(out1);
    ::tenzor::jit::TracedOp op(type, {a_id}, {o0_id, o1_id});
    tracer.record_op(std::move(op));
}

inline auto jit_record_linalg1to3_bool_attr(::tenzor::jit::OpType type, const Tensor& a,
                                             const Tensor& out0, const Tensor& out1,
                                             const Tensor& out2, const char* attr_name,
                                             bool attr_val) -> void {
    auto& tracer = ::tenzor::jit::Tracer::get_instance();
    if (!tracer.is_tracing()) return;
    auto a_id = tracer.register_tensor(a);
    auto o0_id = tracer.register_new_tensor(out0);
    auto o1_id = tracer.register_new_tensor(out1);
    auto o2_id = tracer.register_new_tensor(out2);
    ::tenzor::jit::TracedOp op(type, {a_id}, {o0_id, o1_id, o2_id});
    op.bool_attrs[attr_name] = attr_val;
    tracer.record_op(std::move(op));
}

// JIT-R110: added alongside wiring solve_triangular/slogdet/eigvalsh/eig/lu/
// lu_solve/householder_product/ldl_factor/ldl_solve/ormqr/geqrf's CPU
// LAPACKE paths into the tracer -- these were the remaining input/output
// arities among that group not already covered above.
inline auto jit_record_linalg2to1_2bool_attr(::tenzor::jit::OpType type, const Tensor& a,
                                              const Tensor& b, const Tensor& out,
                                              const char* attr0_name, bool attr0_val,
                                              const char* attr1_name, bool attr1_val)
    -> void {
    auto& tracer = ::tenzor::jit::Tracer::get_instance();
    if (!tracer.is_tracing()) return;
    auto a_id = tracer.register_tensor(a);
    auto b_id = tracer.register_tensor(b);
    auto o_id = tracer.register_new_tensor(out);
    ::tenzor::jit::TracedOp op(type, {a_id, b_id}, {o_id});
    op.bool_attrs[attr0_name] = attr0_val;
    op.bool_attrs[attr1_name] = attr1_val;
    tracer.record_op(std::move(op));
}

inline auto jit_record_linalg1to3(::tenzor::jit::OpType type, const Tensor& a,
                                   const Tensor& out0, const Tensor& out1,
                                   const Tensor& out2) -> void {
    auto& tracer = ::tenzor::jit::Tracer::get_instance();
    if (!tracer.is_tracing()) return;
    auto a_id = tracer.register_tensor(a);
    auto o0_id = tracer.register_new_tensor(out0);
    auto o1_id = tracer.register_new_tensor(out1);
    auto o2_id = tracer.register_new_tensor(out2);
    ::tenzor::jit::TracedOp op(type, {a_id}, {o0_id, o1_id, o2_id});
    tracer.record_op(std::move(op));
}

inline auto jit_record_linalg3to1(::tenzor::jit::OpType type, const Tensor& a,
                                   const Tensor& b, const Tensor& c,
                                   const Tensor& out) -> void {
    auto& tracer = ::tenzor::jit::Tracer::get_instance();
    if (!tracer.is_tracing()) return;
    auto a_id = tracer.register_tensor(a);
    auto b_id = tracer.register_tensor(b);
    auto c_id = tracer.register_tensor(c);
    auto o_id = tracer.register_new_tensor(out);
    ::tenzor::jit::TracedOp op(type, {a_id, b_id, c_id}, {o_id});
    tracer.record_op(std::move(op));
}

inline auto jit_record_linalg3to1_2bool_attr(::tenzor::jit::OpType type, const Tensor& a,
                                              const Tensor& b, const Tensor& c,
                                              const Tensor& out,
                                              const char* attr0_name, bool attr0_val,
                                              const char* attr1_name, bool attr1_val)
    -> void {
    auto& tracer = ::tenzor::jit::Tracer::get_instance();
    if (!tracer.is_tracing()) return;
    auto a_id = tracer.register_tensor(a);
    auto b_id = tracer.register_tensor(b);
    auto c_id = tracer.register_tensor(c);
    auto o_id = tracer.register_new_tensor(out);
    ::tenzor::jit::TracedOp op(type, {a_id, b_id, c_id}, {o_id});
    op.bool_attrs[attr0_name] = attr0_val;
    op.bool_attrs[attr1_name] = attr1_val;
    tracer.record_op(std::move(op));
}

} // namespace

namespace {
auto det_cpu_impl(const Tensor& A) -> Tensor {
#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("det");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    // Output shape: all dims except last two. For a plain 2D input the
    // result is a scalar (shape {}), matching torch.linalg.det semantics —
    // a previous version force-padded to {1}, which surfaced as a "size-1
    // vs size-0 shape" mismatch in DetShape / SLogDet tests.
    std::vector<int64_t> out_shape;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        out_shape.push_back(shape[i]);
    }
    // out_shape empty ⇒ scalar result; preserve that by passing an empty
    // shape to zeros() (which materialises a 0-D tensor with numel==1).
    auto result = zeros(out_shape, work.dtype(), Device::cpu());

    std::vector<lapack_int> ipiv(n);

    if (work.dtype() == DType::Float32) {
        float* data = work.data<float>();
        float* res_data = result.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = data + b * n * n;
            lapack_int info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR,
                static_cast<lapack_int>(n), static_cast<lapack_int>(n),
                mat, static_cast<lapack_int>(n), ipiv.data());

            if (info < 0) throw std::runtime_error("linalg::det: invalid argument");

            float d = 1.0f;
            for (int64_t i = 0; i < n; ++i) {
                d *= mat[i * n + i];
                if (ipiv[i] != static_cast<lapack_int>(i + 1)) d = -d;
            }
            res_data[b] = d;
        }
    } else if (work.dtype() == DType::Complex64) {
        auto result_c = zeros(out_shape, DType::Complex64, Device::cpu());
        auto* rd = result_c.data<std::complex<float>>();

        for (int64_t b = 0; b < nbatch; ++b) {
            lapack_complex_float* mat = c64_ptr(work) + b * n * n;
            auto ln = static_cast<lapack_int>(n);
            lapack_int info = LAPACKE_cgetrf(LAPACK_ROW_MAJOR, ln, ln, mat, ln, ipiv.data());
            if (info < 0) throw std::runtime_error("linalg::det: invalid argument (complex64)");

            std::complex<float> d{1.f, 0.f};
            for (int64_t i = 0; i < n; ++i) {
                // mat is lapack_complex_float* but same layout as std::complex<float>
                const auto* row = reinterpret_cast<const std::complex<float>*>(mat);
                d *= row[i * n + i];
                if (ipiv[i] != static_cast<lapack_int>(i + 1)) d = -d;
            }
            rd[b] = d;
        }
        return result_c;
    } else if (work.dtype() == DType::Complex128) {
        auto result_c = zeros(out_shape, DType::Complex128, Device::cpu());
        auto* rd = result_c.data<std::complex<double>>();

        for (int64_t b = 0; b < nbatch; ++b) {
            lapack_complex_double* mat = c128_ptr(work) + b * n * n;
            auto ln = static_cast<lapack_int>(n);
            lapack_int info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, ln, ln, mat, ln, ipiv.data());
            if (info < 0) throw std::runtime_error("linalg::det: invalid argument (complex128)");

            std::complex<double> d{1., 0.};
            for (int64_t i = 0; i < n; ++i) {
                const auto* row = reinterpret_cast<const std::complex<double>*>(mat);
                d *= row[i * n + i];
                if (ipiv[i] != static_cast<lapack_int>(i + 1)) d = -d;
            }
            rd[b] = d;
        }
        return result_c;
    } else {
        double* data = work.data<double>();
        double* res_data = result.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = data + b * n * n;
            lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,
                static_cast<lapack_int>(n), static_cast<lapack_int>(n),
                mat, static_cast<lapack_int>(n), ipiv.data());

            if (info < 0) throw std::runtime_error("linalg::det: invalid argument");

            double d = 1.0;
            for (int64_t i = 0; i < n; ++i) {
                d *= mat[i * n + i];
                if (ipiv[i] != static_cast<lapack_int>(i + 1)) d = -d;
            }
            res_data[b] = d;
        }
    }

    return maybe_downcast(result, original_dtype);
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}
} // namespace

auto det(const Tensor& A) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch(OpId::LinalgDet, inputs, {}, result)) return result;
    }
    Tensor result = det_cpu_impl(A);
    jit_record_linalg1(::tenzor::jit::OpType::Det, A, result);
    return result;
}

namespace {
auto inv_cpu_impl(const Tensor& A) -> Tensor {
#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("inv");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<lapack_int> ipiv(n);

    if (work.dtype() == DType::Float32) {
        float* data = work.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = data + b * n * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, ln, ln, mat, ln, ipiv.data());
            if (info != 0) throw std::runtime_error("linalg::inv: LU factorization failed (singular matrix)");

            info = LAPACKE_sgetri(LAPACK_ROW_MAJOR, ln, mat, ln, ipiv.data());
            if (info != 0) throw std::runtime_error("linalg::inv: inversion failed");
        }
    } else if (work.dtype() == DType::Complex64) {
        for (int64_t b = 0; b < nbatch; ++b) {
            lapack_complex_float* mat = c64_ptr(work) + b * n * n;
            auto ln = static_cast<lapack_int>(n);
            lapack_int info = LAPACKE_cgetrf(LAPACK_ROW_MAJOR, ln, ln, mat, ln, ipiv.data());
            if (info != 0) throw std::runtime_error("linalg::inv: LU factorization failed (complex64, singular?)");
            info = LAPACKE_cgetri(LAPACK_ROW_MAJOR, ln, mat, ln, ipiv.data());
            if (info != 0) throw std::runtime_error("linalg::inv: inversion failed (complex64)");
        }
    } else if (work.dtype() == DType::Complex128) {
        for (int64_t b = 0; b < nbatch; ++b) {
            lapack_complex_double* mat = c128_ptr(work) + b * n * n;
            auto ln = static_cast<lapack_int>(n);
            lapack_int info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, ln, ln, mat, ln, ipiv.data());
            if (info != 0) throw std::runtime_error("linalg::inv: LU factorization failed (complex128, singular?)");
            info = LAPACKE_zgetri(LAPACK_ROW_MAJOR, ln, mat, ln, ipiv.data());
            if (info != 0) throw std::runtime_error("linalg::inv: inversion failed (complex128)");
        }
    } else {
        double* data = work.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = data + b * n * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, ln, ln, mat, ln, ipiv.data());
            if (info != 0) throw std::runtime_error("linalg::inv: LU factorization failed (singular matrix)");

            info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, ln, mat, ln, ipiv.data());
            if (info != 0) throw std::runtime_error("linalg::inv: inversion failed");
        }
    }

    return maybe_downcast(work, original_dtype);
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}
} // namespace

auto inv(const Tensor& A) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch(OpId::LinalgInv, inputs, {}, result)) return result;
    }
    Tensor result = inv_cpu_impl(A);
    jit_record_linalg1(::tenzor::jit::OpType::Inv, A, result);
    return result;
}

namespace {
auto solve_cpu_impl(const Tensor& A, const Tensor& B) -> Tensor {
#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("solve");
#else
    auto original_dtype = A.dtype();
    auto work_a = prepare_matrix(A);
    auto work_b = prepare_matrix(B);
    auto [n, ndim_a] = check_square(work_a);

    auto b_shape = B.shape();
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    if (b_ndim < 1) throw std::invalid_argument("linalg::solve: B must be at least 1D");

    // Validate B's row dimension matches A's order n. LAPACKE_?gesv writes
    // n*nrhs elements per batch into work_b; a B with fewer rows than n
    // would otherwise produce an out-of-bounds heap write.
    int64_t b_rows = (b_ndim >= 2) ? b_shape[b_ndim - 2] : b_shape[0];
    if (b_rows != n) {
        throw std::invalid_argument(
            "linalg::solve: B row dimension (" + std::to_string(b_rows) +
            ") must match A's size n (" + std::to_string(n) + ")");
    }

    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;
    int64_t nbatch = batch_size(work_a);
    // B must have the same batch count as A: the per-batch loop indexes B as
    // b_data + b*n*nrhs and LAPACKE overwrites it in place, so a batched A with a
    // non-batched (or differently-batched) B reads/writes past the end of work_b.
    if (batch_size(work_b) != nbatch) {
        throw std::invalid_argument(
            "linalg::solve: batch count of A (" + std::to_string(nbatch) +
            ") does not match batch count of B (" + std::to_string(batch_size(work_b)) + ")");
    }

    // Reject dims that would truncate in the 32-bit LAPACK index ABI.
    (void)to_lapack_int(n, "n");
    (void)to_lapack_int(nrhs, "nrhs");

    std::vector<lapack_int> ipiv(n);

    if (work_a.dtype() == DType::Float32) {
        float* a_data = work_a.data<float>();
        float* b_data = work_b.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* a_mat = a_data + b * n * n;
            float* b_mat = b_data + b * n * nrhs;
            auto ln = static_cast<lapack_int>(n);
            auto lnrhs = static_cast<lapack_int>(nrhs);

            lapack_int info = LAPACKE_sgesv(LAPACK_ROW_MAJOR, ln, lnrhs,
                a_mat, ln, ipiv.data(), b_mat, lnrhs);
            if (info != 0) throw std::runtime_error("linalg::solve: solution failed");
        }
    } else if (work_a.dtype() == DType::Complex64) {
        for (int64_t b = 0; b < nbatch; ++b) {
            lapack_complex_float* a_mat = c64_ptr(work_a) + b * n * n;
            lapack_complex_float* b_mat = c64_ptr(work_b) + b * n * nrhs;
            auto ln = static_cast<lapack_int>(n);
            auto lnrhs = static_cast<lapack_int>(nrhs);
            lapack_int info = LAPACKE_cgesv(LAPACK_ROW_MAJOR, ln, lnrhs,
                a_mat, ln, ipiv.data(), b_mat, lnrhs);
            if (info != 0) throw std::runtime_error("linalg::solve: solution failed (complex64)");
        }
    } else if (work_a.dtype() == DType::Complex128) {
        for (int64_t b = 0; b < nbatch; ++b) {
            lapack_complex_double* a_mat = c128_ptr(work_a) + b * n * n;
            lapack_complex_double* b_mat = c128_ptr(work_b) + b * n * nrhs;
            auto ln = static_cast<lapack_int>(n);
            auto lnrhs = static_cast<lapack_int>(nrhs);
            lapack_int info = LAPACKE_zgesv(LAPACK_ROW_MAJOR, ln, lnrhs,
                a_mat, ln, ipiv.data(), b_mat, lnrhs);
            if (info != 0) throw std::runtime_error("linalg::solve: solution failed (complex128)");
        }
    } else {
        double* a_data = work_a.data<double>();
        double* b_data = work_b.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* a_mat = a_data + b * n * n;
            double* b_mat = b_data + b * n * nrhs;
            auto ln = static_cast<lapack_int>(n);
            auto lnrhs = static_cast<lapack_int>(nrhs);

            lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, ln, lnrhs,
                a_mat, ln, ipiv.data(), b_mat, lnrhs);
            if (info != 0) throw std::runtime_error("linalg::solve: solution failed");
        }
    }

    return maybe_downcast(work_b, original_dtype);
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}
} // namespace

auto solve(const Tensor& A, const Tensor& B) -> Tensor {
    // Try GPU dispatch first. The GPU solve kernels expect a 2D RHS (they
    // transpose B for the column-major cuSOLVER call, which fails on a 1D
    // vector); promote a 1D B to (n, 1) and squeeze the solution back.
    {
        const bool b_was_1d = (B.shape().size() == 1);
        Tensor B2 = b_was_1d ? tenzor::reshape(B, {B.shape()[0], 1}) : B;
        Tensor result;
        std::array<Tensor, 2> inputs = {A, B2};
        if (try_gpu_dispatch(OpId::LinalgSolve, inputs, {}, result)) {
            return b_was_1d ? tenzor::reshape(result, {B.shape()[0]}) : result;
        }
    }
    Tensor result = solve_cpu_impl(A, B);
    jit_record_linalg2to1(::tenzor::jit::OpType::Solve, A, B, result);
    return result;
}

auto solve_triangular(const Tensor& A, const Tensor& B, bool upper, bool unitriangular) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 2> inputs = {A, B};
        OpAttributes attrs;
        attrs.set(AttrKey::Upper, upper);
        attrs.set(AttrKey::UnitTriangular, unitriangular);
        if (try_gpu_dispatch(OpId::SolveTriangular, inputs, attrs, result)) return result;
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    // Fallback: forward/back substitution without LAPACK
    auto original_dtype = A.dtype();
    auto work_a = prepare_matrix(A);
    auto work_b = prepare_matrix(B);
    auto [n, ndim_a] = check_square(work_a);

    auto b_shape = B.shape();
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    if (b_ndim < 1) throw std::invalid_argument("linalg::solve_triangular: B must be at least 1D");

    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;
    int64_t nbatch = batch_size(work_a);

    // For a non-unitriangular solve the substitution divides by each diagonal
    // entry. A zero diagonal would silently yield inf/NaN; reject it up front
    // with a diagnostic (matching sparse_triangular_solve's behaviour).
    if (!unitriangular) {
        auto check_zero_diag = [&](auto* a_data) {
            for (int64_t batch = 0; batch < nbatch; ++batch) {
                const auto* A_mat = a_data + batch * n * n;
                for (int64_t i = 0; i < n; ++i) {
                    if (A_mat[i * n + i] == std::remove_const_t<
                            std::remove_pointer_t<decltype(a_data)>>(0)) {
                        throw std::runtime_error(
                            "linalg::solve_triangular: zero diagonal element at row " +
                            std::to_string(i));
                    }
                }
            }
        };
        if (work_a.dtype() == DType::Float32) {
            check_zero_diag(work_a.data<float>());
        } else if (work_a.dtype() == DType::Complex64) {
            check_zero_diag(work_a.data<std::complex<float>>());
        } else if (work_a.dtype() == DType::Complex128) {
            check_zero_diag(work_a.data<std::complex<double>>());
        } else {
            check_zero_diag(work_a.data<double>());
        }
    }

    if (work_a.dtype() == DType::Float32) {
        const float* a_data = work_a.data<float>();
        float* b_data = work_b.data<float>();
        for (int64_t batch = 0; batch < nbatch; ++batch) {
            const float* A_mat = a_data + batch * n * n;
            float* X_mat = b_data + batch * n * nrhs;
            if (upper) {
                for (int64_t i = n - 1; i >= 0; --i) {
                    for (int64_t j = 0; j < nrhs; ++j) {
                        float sum = X_mat[i * nrhs + j];
                        for (int64_t k = i + 1; k < n; ++k)
                            sum -= A_mat[i * n + k] * X_mat[k * nrhs + j];
                        X_mat[i * nrhs + j] = unitriangular ? sum : sum / A_mat[i * n + i];
                    }
                }
            } else {
                for (int64_t i = 0; i < n; ++i) {
                    for (int64_t j = 0; j < nrhs; ++j) {
                        float sum = X_mat[i * nrhs + j];
                        for (int64_t k = 0; k < i; ++k)
                            sum -= A_mat[i * n + k] * X_mat[k * nrhs + j];
                        X_mat[i * nrhs + j] = unitriangular ? sum : sum / A_mat[i * n + i];
                    }
                }
            }
        }
    } else if (work_a.dtype() == DType::Complex64) {
        const auto* a_data = work_a.data<std::complex<float>>();
        auto* b_data = work_b.data<std::complex<float>>();
        for (int64_t batch = 0; batch < nbatch; ++batch) {
            const std::complex<float>* A_mat = a_data + batch * n * n;
            std::complex<float>* X_mat = b_data + batch * n * nrhs;
            if (upper) {
                for (int64_t i = n - 1; i >= 0; --i) {
                    for (int64_t j = 0; j < nrhs; ++j) {
                        std::complex<float> sum = X_mat[i * nrhs + j];
                        for (int64_t k = i + 1; k < n; ++k)
                            sum -= A_mat[i * n + k] * X_mat[k * nrhs + j];
                        X_mat[i * nrhs + j] = unitriangular ? sum : sum / A_mat[i * n + i];
                    }
                }
            } else {
                for (int64_t i = 0; i < n; ++i) {
                    for (int64_t j = 0; j < nrhs; ++j) {
                        std::complex<float> sum = X_mat[i * nrhs + j];
                        for (int64_t k = 0; k < i; ++k)
                            sum -= A_mat[i * n + k] * X_mat[k * nrhs + j];
                        X_mat[i * nrhs + j] = unitriangular ? sum : sum / A_mat[i * n + i];
                    }
                }
            }
        }
    } else if (work_a.dtype() == DType::Complex128) {
        const auto* a_data = work_a.data<std::complex<double>>();
        auto* b_data = work_b.data<std::complex<double>>();
        for (int64_t batch = 0; batch < nbatch; ++batch) {
            const std::complex<double>* A_mat = a_data + batch * n * n;
            std::complex<double>* X_mat = b_data + batch * n * nrhs;
            if (upper) {
                for (int64_t i = n - 1; i >= 0; --i) {
                    for (int64_t j = 0; j < nrhs; ++j) {
                        std::complex<double> sum = X_mat[i * nrhs + j];
                        for (int64_t k = i + 1; k < n; ++k)
                            sum -= A_mat[i * n + k] * X_mat[k * nrhs + j];
                        X_mat[i * nrhs + j] = unitriangular ? sum : sum / A_mat[i * n + i];
                    }
                }
            } else {
                for (int64_t i = 0; i < n; ++i) {
                    for (int64_t j = 0; j < nrhs; ++j) {
                        std::complex<double> sum = X_mat[i * nrhs + j];
                        for (int64_t k = 0; k < i; ++k)
                            sum -= A_mat[i * n + k] * X_mat[k * nrhs + j];
                        X_mat[i * nrhs + j] = unitriangular ? sum : sum / A_mat[i * n + i];
                    }
                }
            }
        }
    } else {
        const double* a_data = work_a.data<double>();
        double* b_data = work_b.data<double>();
        for (int64_t batch = 0; batch < nbatch; ++batch) {
            const double* A_mat = a_data + batch * n * n;
            double* X_mat = b_data + batch * n * nrhs;
            if (upper) {
                for (int64_t i = n - 1; i >= 0; --i) {
                    for (int64_t j = 0; j < nrhs; ++j) {
                        double sum = X_mat[i * nrhs + j];
                        for (int64_t k = i + 1; k < n; ++k)
                            sum -= A_mat[i * n + k] * X_mat[k * nrhs + j];
                        X_mat[i * nrhs + j] = unitriangular ? sum : sum / A_mat[i * n + i];
                    }
                }
            } else {
                for (int64_t i = 0; i < n; ++i) {
                    for (int64_t j = 0; j < nrhs; ++j) {
                        double sum = X_mat[i * nrhs + j];
                        for (int64_t k = 0; k < i; ++k)
                            sum -= A_mat[i * n + k] * X_mat[k * nrhs + j];
                        X_mat[i * nrhs + j] = unitriangular ? sum : sum / A_mat[i * n + i];
                    }
                }
            }
        }
    }
    {
        Tensor result = maybe_downcast(work_b, original_dtype);
        jit_record_linalg2to1_2bool_attr(::tenzor::jit::OpType::SolveTriangular, A, B,
                                          result, "upper", upper, "unitriangular",
                                          unitriangular);
        return result;
    }
#else
    // Use CBLAS trsm for triangular solve
    auto original_dtype = A.dtype();
    auto work_a = prepare_matrix(A);
    auto work_b = prepare_matrix(B);
    auto [n, ndim_a] = check_square(work_a);

    auto b_shape = B.shape();
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    if (b_ndim < 1) throw std::invalid_argument("linalg::solve_triangular: B must be at least 1D");

    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;
    int64_t nbatch = batch_size(work_a);

    // For a non-unitriangular solve, cblas_trsm divides by each diagonal
    // entry internally with no error reporting -- a zero diagonal silently
    // yields Inf/NaN instead of throwing. Reject it up front with the same
    // diagnostic the no-LAPACK fallback branch above already uses, so this
    // MKL/LAPACKE build path (the codebase's default/primary CPU build)
    // matches inv/solve/cholesky/svd/qr/eigh/eig/lu_solve/ldl_solve, all of
    // which check for a singular input and throw.
    if (!unitriangular) {
        auto check_zero_diag = [&](auto* a_data) {
            for (int64_t batch = 0; batch < nbatch; ++batch) {
                const auto* A_mat = a_data + batch * n * n;
                for (int64_t i = 0; i < n; ++i) {
                    if (A_mat[i * n + i] == std::remove_const_t<
                            std::remove_pointer_t<decltype(a_data)>>(0)) {
                        throw std::runtime_error(
                            "linalg::solve_triangular: zero diagonal element at row " +
                            std::to_string(i));
                    }
                }
            }
        };
        if (work_a.dtype() == DType::Float32) {
            check_zero_diag(work_a.data<float>());
        } else if (work_a.dtype() == DType::Complex64) {
            check_zero_diag(work_a.data<std::complex<float>>());
        } else if (work_a.dtype() == DType::Complex128) {
            check_zero_diag(work_a.data<std::complex<double>>());
        } else {
            check_zero_diag(work_a.data<double>());
        }
    }

    auto uplo = upper ? CblasUpper : CblasLower;
    auto diag = unitriangular ? CblasUnit : CblasNonUnit;
    auto ln = static_cast<int>(n);
    auto lnrhs = static_cast<int>(nrhs);

    if (work_a.dtype() == DType::Float32) {
        const float* a_data = work_a.data<float>();
        float* b_data = work_b.data<float>();
        for (int64_t batch = 0; batch < nbatch; ++batch) {
            cblas_strsm(CblasRowMajor, CblasLeft, uplo, CblasNoTrans, diag,
                        ln, lnrhs, 1.0f,
                        a_data + batch * n * n, ln,
                        b_data + batch * n * nrhs, lnrhs);
        }
    } else if (work_a.dtype() == DType::Complex64) {
        const auto* a_data = work_a.data<std::complex<float>>();
        auto* b_data = work_b.data<std::complex<float>>();
        const std::complex<float> one(1.0f, 0.0f);
        for (int64_t batch = 0; batch < nbatch; ++batch) {
            cblas_ctrsm(CblasRowMajor, CblasLeft, uplo, CblasNoTrans, diag,
                        ln, lnrhs, &one,
                        a_data + batch * n * n, ln,
                        b_data + batch * n * nrhs, lnrhs);
        }
    } else if (work_a.dtype() == DType::Complex128) {
        const auto* a_data = work_a.data<std::complex<double>>();
        auto* b_data = work_b.data<std::complex<double>>();
        const std::complex<double> one(1.0, 0.0);
        for (int64_t batch = 0; batch < nbatch; ++batch) {
            cblas_ztrsm(CblasRowMajor, CblasLeft, uplo, CblasNoTrans, diag,
                        ln, lnrhs, &one,
                        a_data + batch * n * n, ln,
                        b_data + batch * n * nrhs, lnrhs);
        }
    } else {
        const double* a_data = work_a.data<double>();
        double* b_data = work_b.data<double>();
        for (int64_t batch = 0; batch < nbatch; ++batch) {
            cblas_dtrsm(CblasRowMajor, CblasLeft, uplo, CblasNoTrans, diag,
                        ln, lnrhs, 1.0,
                        a_data + batch * n * n, ln,
                        b_data + batch * n * nrhs, lnrhs);
        }
    }

    {
        Tensor result = maybe_downcast(work_b, original_dtype);
        jit_record_linalg2to1_2bool_attr(::tenzor::jit::OpType::SolveTriangular, A, B,
                                          result, "upper", upper, "unitriangular",
                                          unitriangular);
        return result;
    }
#endif
}

namespace {
auto cholesky_cpu_impl(const Tensor& A, bool upper) -> Tensor {
#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("cholesky");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    char uplo = upper ? 'U' : 'L';

    if (work.dtype() == DType::Float32) {
        float* data = work.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = data + b * n * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, uplo, ln, mat, ln);
            if (info != 0) throw std::runtime_error("linalg::cholesky: factorization failed (not positive definite)");

            // Zero out the other triangle
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    if (upper ? (i > j) : (i < j)) {
                        mat[i * n + j] = 0.0f;
                    }
                }
            }
        }
    } else if (work.dtype() == DType::Complex64) {
        for (int64_t b = 0; b < nbatch; ++b) {
            lapack_complex_float* mat = c64_ptr(work) + b * n * n;
            auto ln = static_cast<lapack_int>(n);
            lapack_int info = LAPACKE_cpotrf(LAPACK_ROW_MAJOR, uplo, ln, mat, ln);
            if (info != 0) throw std::runtime_error("linalg::cholesky: factorization failed (complex64, not HPD)");
            // Zero out the other triangle
            auto* cmat = reinterpret_cast<std::complex<float>*>(mat);
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = 0; j < n; ++j)
                    if (upper ? (i > j) : (i < j))
                        cmat[i * n + j] = {0.f, 0.f};
        }
    } else if (work.dtype() == DType::Complex128) {
        for (int64_t b = 0; b < nbatch; ++b) {
            lapack_complex_double* mat = c128_ptr(work) + b * n * n;
            auto ln = static_cast<lapack_int>(n);
            lapack_int info = LAPACKE_zpotrf(LAPACK_ROW_MAJOR, uplo, ln, mat, ln);
            if (info != 0) throw std::runtime_error("linalg::cholesky: factorization failed (complex128, not HPD)");
            auto* cmat = reinterpret_cast<std::complex<double>*>(mat);
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = 0; j < n; ++j)
                    if (upper ? (i > j) : (i < j))
                        cmat[i * n + j] = {0., 0.};
        }
    } else {
        double* data = work.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = data + b * n * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, uplo, ln, mat, ln);
            if (info != 0) throw std::runtime_error("linalg::cholesky: factorization failed (not positive definite)");

            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    if (upper ? (i > j) : (i < j)) {
                        mat[i * n + j] = 0.0;
                    }
                }
            }
        }
    }

    return maybe_downcast(work, original_dtype);
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}
} // namespace

auto cholesky(const Tensor& A, bool upper) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 1> inputs = {A};
        OpAttributes attrs;
        attrs.set(AttrKey::Upper, upper);
        if (try_gpu_dispatch(OpId::LinalgCholesky, inputs, attrs, result)) return result;
    }
    Tensor result = cholesky_cpu_impl(A, upper);
    jit_record_linalg1_bool_attr(::tenzor::jit::OpType::Cholesky, A, result, "upper", upper);
    return result;
}

auto norm(const Tensor& A, const std::string& ord) -> Tensor {
    if (ord == "fro") {
        // Frobenius norm: sqrt(sum(|x|^2)). Expressed in pure tensor ops so it
        // runs device-agnostically (CPU / CUDA / etc.) — no dedicated kernel
        // needed. For complex inputs use the elementwise magnitude so we sum
        // |z|^2 (z*conj(z)) rather than z^2.
        Tensor mag = (A.dtype() == DType::Complex64 || A.dtype() == DType::Complex128)
            ? tenzor::abs(A) : A;
        auto flat = mag.contiguous().reshape({mag.numel()});
        auto sq = flat * flat;
        auto s = tenzor::sum(sq);
        return tenzor::sqrt(s);
    }

    if (ord == "nuc") {
        // Nuclear norm: sum of singular values. Requires SVD which currently
        // dispatches to LAPACKE on CPU (Float32/Float64). svdvals() already
        // validates the at-least-2-D requirement.
        auto S = svdvals(A);
        // Reduce over the singular-value axis (last); leave any leading batch
        // dims intact so the result shape matches matrix_norm's contract.
        return tenzor::sum(S, /*dim=*/-1, /*keepdim=*/false);
    }

    // Induced p-norms: parse and delegate to matrix_norm, which already
    // handles all of ±1 / ±2 / ±inf.
    double ord_double = 0.0;
    bool parsed = false;
    if (ord == "1") { ord_double = 1.0;  parsed = true; }
    else if (ord == "-1")   { ord_double = -1.0; parsed = true; }
    else if (ord == "2")    { ord_double = 2.0;  parsed = true; }
    else if (ord == "-2")   { ord_double = -2.0; parsed = true; }
    else if (ord == "inf")  { ord_double = std::numeric_limits<double>::infinity();  parsed = true; }
    else if (ord == "-inf") { ord_double = -std::numeric_limits<double>::infinity(); parsed = true; }

    if (parsed) {
        return matrix_norm(A, ord_double);
    }

    throw std::runtime_error(
        "linalg::norm: unsupported norm order '" + ord +
        "' (supported: 'fro', 'nuc', '1', '-1', '2', '-2', 'inf', '-inf')");
}

auto slogdet(const Tensor& A) -> std::tuple<Tensor, Tensor> {
    // GPU path: derive slogdet from det (which dispatches to backend kernels).
    // sign = sign(det), logabsdet = log(|det|). This avoids a dedicated
    // slogdet OpId + per-backend kernels while still running on-device.
    if (A.device().type != Device::Type::CPU) {
        Tensor d = tenzor::linalg::det(A);
        Tensor sign_t = tenzor::sign(d);
        Tensor logabsdet_t = tenzor::log(tenzor::abs(d));
        return {sign_t, logabsdet_t};
    }
#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("slogdet");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    // 2D input ⇒ scalar outputs (shape {}). See det() for rationale; the
    // same force-to-{1} padding broke SLogDetReturnsSignAndLogabs.
    std::vector<int64_t> out_shape;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        out_shape.push_back(shape[i]);
    }

    // For complex input the sign is complex (det/|det|) while logabsdet is the
    // real log|det|, matching torch.linalg.slogdet and the GPU path above
    // (sign(d) is complex, log(abs(d)) is real). Compute via LU like det().
    if (work.dtype() == DType::Complex64 || work.dtype() == DType::Complex128) {
        std::vector<lapack_int> ipiv_c(n);
        if (work.dtype() == DType::Complex64) {
            auto sign_c = zeros(out_shape, DType::Complex64, Device::cpu());
            auto logabs_r = zeros(out_shape, DType::Float32, Device::cpu());
            auto* sign_data = sign_c.data<std::complex<float>>();
            auto* logabs_data = logabs_r.data<float>();
            for (int64_t b = 0; b < nbatch; ++b) {
                lapack_complex_float* mat = c64_ptr(work) + b * n * n;
                auto ln = static_cast<lapack_int>(n);
                lapack_int info = LAPACKE_cgetrf(LAPACK_ROW_MAJOR, ln, ln, mat, ln, ipiv_c.data());
                if (info < 0) throw std::runtime_error("linalg::slogdet: invalid argument (complex64)");
                std::complex<float> d{1.f, 0.f};
                const auto* row = reinterpret_cast<const std::complex<float>*>(mat);
                for (int64_t i = 0; i < n; ++i) {
                    d *= row[i * n + i];
                    if (ipiv_c[i] != static_cast<lapack_int>(i + 1)) d = -d;
                }
                float mag = std::abs(d);
                if (mag == 0.0f) {
                    sign_data[b] = std::complex<float>{0.f, 0.f};
                    logabs_data[b] = -std::numeric_limits<float>::infinity();
                } else {
                    sign_data[b] = d / mag;
                    logabs_data[b] = std::log(mag);
                }
            }
            jit_record_linalg1to2(::tenzor::jit::OpType::Slogdet, A, sign_c, logabs_r);
            return {sign_c, logabs_r};
        } else {
            auto sign_c = zeros(out_shape, DType::Complex128, Device::cpu());
            auto logabs_r = zeros(out_shape, DType::Float64, Device::cpu());
            auto* sign_data = sign_c.data<std::complex<double>>();
            auto* logabs_data = logabs_r.data<double>();
            for (int64_t b = 0; b < nbatch; ++b) {
                lapack_complex_double* mat = c128_ptr(work) + b * n * n;
                auto ln = static_cast<lapack_int>(n);
                lapack_int info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, ln, ln, mat, ln, ipiv_c.data());
                if (info < 0) throw std::runtime_error("linalg::slogdet: invalid argument (complex128)");
                std::complex<double> d{1., 0.};
                const auto* row = reinterpret_cast<const std::complex<double>*>(mat);
                for (int64_t i = 0; i < n; ++i) {
                    d *= row[i * n + i];
                    if (ipiv_c[i] != static_cast<lapack_int>(i + 1)) d = -d;
                }
                double mag = std::abs(d);
                if (mag == 0.0) {
                    sign_data[b] = std::complex<double>{0., 0.};
                    logabs_data[b] = -std::numeric_limits<double>::infinity();
                } else {
                    sign_data[b] = d / mag;
                    logabs_data[b] = std::log(mag);
                }
            }
            jit_record_linalg1to2(::tenzor::jit::OpType::Slogdet, A, sign_c, logabs_r);
            return {sign_c, logabs_r};
        }
    }

    auto sign_result = zeros(out_shape, work.dtype(), Device::cpu());
    auto logabsdet_result = zeros(out_shape, work.dtype(), Device::cpu());

    std::vector<lapack_int> ipiv(n);

    if (work.dtype() == DType::Float32) {
        float* data = work.data<float>();
        float* sign_data = sign_result.data<float>();
        float* logabs_data = logabsdet_result.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = data + b * n * n;
            lapack_int info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR,
                static_cast<lapack_int>(n), static_cast<lapack_int>(n),
                mat, static_cast<lapack_int>(n), ipiv.data());

            if (info < 0) throw std::runtime_error("linalg::slogdet: invalid argument");

            float sign = 1.0f;
            float logabsdet = 0.0f;
            bool is_zero = false;
            for (int64_t i = 0; i < n; ++i) {
                float diag = mat[i * n + i];
                if (diag == 0.0f) { is_zero = true; break; }
                if (diag < 0.0f) { sign = -sign; diag = -diag; }
                logabsdet += std::log(diag);
                if (ipiv[i] != static_cast<lapack_int>(i + 1)) sign = -sign;
            }
            if (is_zero) {
                sign_data[b] = 0.0f;
                logabs_data[b] = -std::numeric_limits<float>::infinity();
            } else {
                sign_data[b] = sign;
                logabs_data[b] = logabsdet;
            }
        }
    } else {
        double* data = work.data<double>();
        double* sign_data = sign_result.data<double>();
        double* logabs_data = logabsdet_result.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = data + b * n * n;
            lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,
                static_cast<lapack_int>(n), static_cast<lapack_int>(n),
                mat, static_cast<lapack_int>(n), ipiv.data());

            if (info < 0) throw std::runtime_error("linalg::slogdet: invalid argument");

            double sign = 1.0;
            double logabsdet = 0.0;
            bool is_zero = false;
            for (int64_t i = 0; i < n; ++i) {
                double diag = mat[i * n + i];
                if (diag == 0.0) { is_zero = true; break; }
                if (diag < 0.0) { sign = -sign; diag = -diag; }
                logabsdet += std::log(diag);
                if (ipiv[i] != static_cast<lapack_int>(i + 1)) sign = -sign;
            }
            if (is_zero) {
                sign_data[b] = 0.0;
                logabs_data[b] = -std::numeric_limits<double>::infinity();
            } else {
                sign_data[b] = sign;
                logabs_data[b] = logabsdet;
            }
        }
    }

    {
        Tensor sign_out = maybe_downcast(sign_result, original_dtype);
        Tensor logabsdet_out = maybe_downcast(logabsdet_result, original_dtype);
        jit_record_linalg1to2(::tenzor::jit::OpType::Slogdet, A, sign_out, logabsdet_out);
        return {sign_out, logabsdet_out};
    }
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

namespace {

// ---------------------------------------------------------------------------
// Deterministic vector sign/phase canonicalization (findings F016, F015).
//
// SVD (U/Vh columns) and symmetric/Hermitian eigh (eigenvectors) are defined
// only up to a per-vector sign (real) or unit phase (complex). LAPACK (CPU) and
// cuSOLVER/rocSOLVER/etc. (GPU) legitimately pick opposite signs, so raw
// element-wise cross-backend comparison diverges. We fix a single convention on
// EVERY backend: scale each vector so its largest-magnitude entry is real and
// positive. Reconstruction is preserved: for SVD the paired Vh row is scaled by
// the inverse phase, so U·diag(S)·Vh is unchanged; for eigh V·diag(W)·Vᴴ is
// unchanged (a column phase cancels in the outer product).
// ---------------------------------------------------------------------------
template <typename R>
void canon_real_cols(R* U, R* Vt, int64_t nbatch, int64_t rows,
                     int64_t ncolsU, int64_t ncolsVt, int64_t num_cols) {
    for (int64_t b = 0; b < nbatch; ++b) {
        R* Ub = U + b * rows * ncolsU;
        R* Vtb = Vt ? Vt + b * num_cols * ncolsVt : nullptr;
        for (int64_t j = 0; j < num_cols; ++j) {
            int64_t istar = 0;
            R best = R(-1);
            for (int64_t i = 0; i < rows; ++i) {
                R a = std::abs(Ub[i * ncolsU + j]);
                if (a > best) { best = a; istar = i; }
            }
            if (Ub[istar * ncolsU + j] < R(0)) {
                for (int64_t i = 0; i < rows; ++i)
                    Ub[i * ncolsU + j] = -Ub[i * ncolsU + j];
                if (Vtb)
                    for (int64_t c = 0; c < ncolsVt; ++c)
                        Vtb[j * ncolsVt + c] = -Vtb[j * ncolsVt + c];
            }
        }
    }
}

template <typename R>
void canon_complex_cols(std::complex<R>* U, std::complex<R>* Vt, int64_t nbatch,
                        int64_t rows, int64_t ncolsU, int64_t ncolsVt,
                        int64_t num_cols) {
    for (int64_t b = 0; b < nbatch; ++b) {
        std::complex<R>* Ub = U + b * rows * ncolsU;
        std::complex<R>* Vtb = Vt ? Vt + b * num_cols * ncolsVt : nullptr;
        for (int64_t j = 0; j < num_cols; ++j) {
            int64_t istar = 0;
            R best = R(-1);
            for (int64_t i = 0; i < rows; ++i) {
                R a = std::abs(Ub[i * ncolsU + j]);
                if (a > best) { best = a; istar = i; }
            }
            std::complex<R> val = Ub[istar * ncolsU + j];
            R mag = std::abs(val);
            if (mag == R(0)) continue;
            std::complex<R> c = std::conj(val) / mag;  // U[istar,j] -> real, +|val|
            std::complex<R> d = val / mag;             // inverse phase for paired Vh
            for (int64_t i = 0; i < rows; ++i) Ub[i * ncolsU + j] *= c;
            if (Vtb)
                for (int64_t cc = 0; cc < ncolsVt; ++cc)
                    Vtb[j * ncolsVt + cc] *= d;
        }
    }
}

// Canonicalize the first `num_cols` columns of U (and, if have_vt, the paired
// rows of Vt) on host copies, returning device-resident results. Float16/
// BFloat16 are widened to Float32 for the decision and narrowed back (sign
// flips are exact, but keeps the code single-path).
std::pair<Tensor, Tensor> canonicalize_uv(const Tensor& U_in, const Tensor& Vt_in,
                                          int64_t num_cols, bool have_vt) {
    Device dev = U_in.device();
    DType orig = U_in.dtype();
    bool half = (orig == DType::Float16 || orig == DType::BFloat16);
    Tensor U = (half ? U_in.to(DType::Float32) : U_in).cpu().contiguous();
    Tensor Vt;
    if (have_vt)
        Vt = (half ? Vt_in.to(DType::Float32) : Vt_in).cpu().contiguous();

    auto ush = U.shape();
    int64_t un = static_cast<int64_t>(ush.size());
    int64_t rows = ush[un - 2];
    int64_t ncolsU = ush[un - 1];
    int64_t nbatch = 1;
    for (int64_t i = 0; i + 2 < un; ++i) nbatch *= ush[i];
    int64_t ncolsVt = 0;
    if (have_vt) { auto vsh = Vt.shape(); ncolsVt = vsh[vsh.size() - 1]; }

    switch (U.dtype()) {
        case DType::Float32:
            canon_real_cols<float>(U.data<float>(),
                                   have_vt ? Vt.data<float>() : nullptr,
                                   nbatch, rows, ncolsU, ncolsVt, num_cols);
            break;
        case DType::Float64:
            canon_real_cols<double>(U.data<double>(),
                                    have_vt ? Vt.data<double>() : nullptr,
                                    nbatch, rows, ncolsU, ncolsVt, num_cols);
            break;
        case DType::Complex64:
            canon_complex_cols<float>(U.data<std::complex<float>>(),
                                      have_vt ? Vt.data<std::complex<float>>() : nullptr,
                                      nbatch, rows, ncolsU, ncolsVt, num_cols);
            break;
        case DType::Complex128:
            canon_complex_cols<double>(U.data<std::complex<double>>(),
                                       have_vt ? Vt.data<std::complex<double>>() : nullptr,
                                       nbatch, rows, ncolsU, ncolsVt, num_cols);
            break;
        default:
            break;
    }

    Tensor U_out = half ? U.to(orig).to(dev) : U.to(dev);
    Tensor Vt_out = Vt_in;
    if (have_vt) Vt_out = half ? Vt.to(orig).to(dev) : Vt.to(dev);
    return {U_out, Vt_out};
}

// ---------------------------------------------------------------------------
// eig canonicalization (findings F014 order, F015 sign).
//
// General eig eigenvalue ORDER and eigenvector sign/phase are algorithm-defined,
// so CPU (LAPACK geev) and the GPU solvers diverge. We impose a single canonical
// form on every backend WHERE IT IS WELL-DEFINED:
//   * complex-input eig (V complex, no conjugate-pair packing): sort by
//     (Re λ, Im λ) ascending and phase-fix each eigenvector.
//   * real-input eig with an entirely REAL spectrum (the symmetric-matrix case
//     of F014, and any real-eigenvalue non-symmetric matrix): sort ascending by
//     eigenvalue and sign-fix each real eigenvector.
//   * real-input eig with complex conjugate pairs: phase-fix each pair's
//     eigenvector IN PLACE (no reorder). Reordering the LAPACK real-packed
//     (Re,Im) column pairs is convention-laden and would risk splitting a pair,
//     so order is left to the (residual-invariant) comparison there.
// Reconstruction A·v = λ·v is preserved in every branch.
// ---------------------------------------------------------------------------
template <typename R>
void canon_eig_real_packed(R* wr, R* wi, R* V, int64_t n) {
    bool all_real = true;
    for (int64_t i = 0; i < n; ++i)
        if (wi[i] != R(0)) { all_real = false; break; }

    if (all_real) {
        std::vector<int64_t> idx(n);
        for (int64_t i = 0; i < n; ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(),
                         [&](int64_t a, int64_t b) { return wr[a] < wr[b]; });
        std::vector<R> nwr(n);
        std::vector<R> newV(static_cast<size_t>(n) * n);
        for (int64_t c = 0; c < n; ++c) {
            int64_t src = idx[c];
            nwr[c] = wr[src];
            int64_t istar = 0;
            R best = R(-1);
            for (int64_t i = 0; i < n; ++i) {
                R a = std::abs(V[i * n + src]);
                if (a > best) { best = a; istar = i; }
            }
            R s = (V[istar * n + src] < R(0)) ? R(-1) : R(1);
            for (int64_t i = 0; i < n; ++i) newV[i * n + c] = s * V[i * n + src];
        }
        for (int64_t i = 0; i < n; ++i) { wr[i] = nwr[i]; wi[i] = R(0); }
        for (size_t i = 0; i < static_cast<size_t>(n) * n; ++i) V[i] = newV[i];
        return;
    }

    // Mixed / complex spectrum: phase-fix in place, no reorder.
    for (int64_t j = 0; j < n;) {
        if (wi[j] == R(0)) {
            int64_t istar = 0;
            R best = R(-1);
            for (int64_t i = 0; i < n; ++i) {
                R a = std::abs(V[i * n + j]);
                if (a > best) { best = a; istar = i; }
            }
            if (V[istar * n + j] < R(0))
                for (int64_t i = 0; i < n; ++i) V[i * n + j] = -V[i * n + j];
            j += 1;
        } else {
            int64_t istar = 0;
            R best = R(-1);
            for (int64_t i = 0; i < n; ++i) {
                R a = std::hypot(V[i * n + j], V[i * n + j + 1]);
                if (a > best) { best = a; istar = i; }
            }
            std::complex<R> val(V[istar * n + j], V[istar * n + j + 1]);
            R mag = std::abs(val);
            std::complex<R> cph = (mag > R(0)) ? std::conj(val) / mag
                                               : std::complex<R>(R(1), R(0));
            for (int64_t i = 0; i < n; ++i) {
                std::complex<R> cv = std::complex<R>(V[i * n + j], V[i * n + j + 1]) * cph;
                V[i * n + j] = cv.real();
                V[i * n + j + 1] = cv.imag();
            }
            j += 2;
        }
    }
}

template <typename R>
void canon_eig_complex(R* wr, R* wi, std::complex<R>* V, int64_t n) {
    std::vector<int64_t> idx(n);
    for (int64_t i = 0; i < n; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
        if (wr[a] != wr[b]) return wr[a] < wr[b];
        return wi[a] < wi[b];
    });
    std::vector<R> nwr(n), nwi(n);
    std::vector<std::complex<R>> newV(static_cast<size_t>(n) * n);
    for (int64_t c = 0; c < n; ++c) {
        int64_t src = idx[c];
        nwr[c] = wr[src];
        nwi[c] = wi[src];
        int64_t istar = 0;
        R best = R(-1);
        for (int64_t i = 0; i < n; ++i) {
            R a = std::abs(V[i * n + src]);
            if (a > best) { best = a; istar = i; }
        }
        std::complex<R> val = V[istar * n + src];
        R mag = std::abs(val);
        std::complex<R> cph = (mag > R(0)) ? std::conj(val) / mag
                                           : std::complex<R>(R(1), R(0));
        for (int64_t i = 0; i < n; ++i) newV[i * n + c] = V[i * n + src] * cph;
    }
    for (int64_t i = 0; i < n; ++i) { wr[i] = nwr[i]; wi[i] = nwi[i]; }
    for (size_t i = 0; i < static_cast<size_t>(n) * n; ++i) V[i] = newV[i];
}

std::tuple<Tensor, Tensor, Tensor> canonicalize_eig(const Tensor& Wr_in,
                                                    const Tensor& Wi_in,
                                                    const Tensor& V_in) {
    Device dev = V_in.device();
    DType vdt = V_in.dtype();
    bool is_complex = (vdt == DType::Complex64 || vdt == DType::Complex128);
    DType worig = Wr_in.dtype();
    bool half = (worig == DType::Float16 || worig == DType::BFloat16);

    Tensor Wr = (half ? Wr_in.to(DType::Float32) : Wr_in).cpu().contiguous();
    Tensor Wi = (half ? Wi_in.to(DType::Float32) : Wi_in).cpu().contiguous();
    Tensor V = (half ? V_in.to(DType::Float32) : V_in).cpu().contiguous();

    auto vsh = V.shape();
    int64_t vn = static_cast<int64_t>(vsh.size());
    int64_t n = vsh[vn - 1];
    int64_t nbatch = 1;
    for (int64_t i = 0; i + 2 < vn; ++i) nbatch *= vsh[i];

    if (is_complex) {
        if (Wr.dtype() == DType::Float32) {
            auto* wr = Wr.data<float>(); auto* wi = Wi.data<float>();
            auto* v = V.data<std::complex<float>>();
            for (int64_t b = 0; b < nbatch; ++b)
                canon_eig_complex<float>(wr + b * n, wi + b * n, v + b * n * n, n);
        } else {
            auto* wr = Wr.data<double>(); auto* wi = Wi.data<double>();
            auto* v = V.data<std::complex<double>>();
            for (int64_t b = 0; b < nbatch; ++b)
                canon_eig_complex<double>(wr + b * n, wi + b * n, v + b * n * n, n);
        }
    } else {
        if (V.dtype() == DType::Float32) {
            auto* wr = Wr.data<float>(); auto* wi = Wi.data<float>();
            auto* v = V.data<float>();
            for (int64_t b = 0; b < nbatch; ++b)
                canon_eig_real_packed<float>(wr + b * n, wi + b * n, v + b * n * n, n);
        } else {
            auto* wr = Wr.data<double>(); auto* wi = Wi.data<double>();
            auto* v = V.data<double>();
            for (int64_t b = 0; b < nbatch; ++b)
                canon_eig_real_packed<double>(wr + b * n, wi + b * n, v + b * n * n, n);
        }
    }

    Tensor Wr_out = half ? Wr.to(worig).to(dev) : Wr.to(dev);
    Tensor Wi_out = half ? Wi.to(worig).to(dev) : Wi.to(dev);
    Tensor V_out = half ? V.to(vdt).to(dev) : V.to(dev);
    return {Wr_out, Wi_out, V_out};
}

} // namespace

namespace {
auto svd_cpu_impl(const Tensor& A, bool full_matrices) -> std::tuple<Tensor, Tensor, Tensor> {
#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("svd");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto shape = A.shape();
    auto a_ndim = static_cast<int64_t>(shape.size());
    if (a_ndim < 2) throw std::invalid_argument("linalg::svd: input must be at least 2D");

    int64_t m = shape[a_ndim - 2];
    int64_t n_cols = shape[a_ndim - 1];
    int64_t k = std::min(m, n_cols);
    int64_t nbatch = batch_size(work);

    // Batch dims
    std::vector<int64_t> batch_dims;
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);

    // Output shapes
    std::vector<int64_t> u_shape = batch_dims;
    std::vector<int64_t> s_shape = batch_dims;
    std::vector<int64_t> vt_shape = batch_dims;

    s_shape.push_back(k);
    if (full_matrices) {
        u_shape.push_back(m); u_shape.push_back(m);
        vt_shape.push_back(n_cols); vt_shape.push_back(n_cols);
    } else {
        u_shape.push_back(m); u_shape.push_back(k);
        vt_shape.push_back(k); vt_shape.push_back(n_cols);
    }

    // For complex inputs: U and Vt stay complex, S is real (Float32 or Float64)
    bool is_complex = (work.dtype() == DType::Complex64 || work.dtype() == DType::Complex128);
    DType s_dtype = work.dtype();
    if (work.dtype() == DType::Complex64)  s_dtype = DType::Float32;
    if (work.dtype() == DType::Complex128) s_dtype = DType::Float64;

    auto U  = zeros(u_shape,  work.dtype(), Device::cpu());
    auto S  = zeros(s_shape,  s_dtype,      Device::cpu());
    auto Vt = zeros(vt_shape, work.dtype(), Device::cpu());

    char jobz = full_matrices ? 'A' : 'S';

    // superb array for ?gesvd
    int64_t superb_size = k - 1;
    if (superb_size < 1) superb_size = 1;

    if (work.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* u_data = U.data<float>();
        float* s_data = S.data<float>();
        float* vt_data = Vt.data<float>();
        std::vector<float> superb(superb_size);

        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);
        // Row-major leading dimensions: ldu = #cols of U, ldvt = #cols of Vt
        auto ldu = full_matrices ? lm : static_cast<lapack_int>(k);
        auto ldvt = full_matrices ? ln : ln;
        int64_t u_stride = full_matrices ? m * m : m * k;
        int64_t vt_stride = full_matrices ? n_cols * n_cols : k * n_cols;

        for (int64_t b = 0; b < nbatch; ++b) {
            float* a_mat = a_data + b * m * n_cols;
            float* u_mat = u_data + b * u_stride;
            float* s_vec = s_data + b * k;
            float* vt_mat = vt_data + b * vt_stride;

            lapack_int info = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, jobz, jobz,
                lm, ln, a_mat, ln, s_vec, u_mat, ldu, vt_mat, ldvt, superb.data());
            if (info != 0) throw std::runtime_error("linalg::svd: computation failed (info=" + std::to_string(info) + ")");
        }
    } else if (work.dtype() == DType::Complex64) {
        float* s_data = S.data<float>();
        std::vector<float> superb(superb_size);
        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);
        auto ldu  = full_matrices ? lm : static_cast<lapack_int>(k);
        auto ldvt = full_matrices ? ln : ln;
        int64_t u_stride  = full_matrices ? m * m : m * k;
        int64_t vt_stride = full_matrices ? n_cols * n_cols : k * n_cols;

        for (int64_t b = 0; b < nbatch; ++b) {
            lapack_complex_float* a_mat = c64_ptr(work) + b * m * n_cols;
            lapack_complex_float* u_mat = c64_ptr(U)    + b * u_stride;
            lapack_complex_float* vt_mat= c64_ptr(Vt)   + b * vt_stride;
            float* s_vec = s_data + b * k;
            lapack_int info = LAPACKE_cgesvd(LAPACK_ROW_MAJOR, jobz, jobz,
                lm, ln, a_mat, ln, s_vec, u_mat, ldu, vt_mat, ldvt, superb.data());
            if (info != 0) throw std::runtime_error("linalg::svd: cgesvd failed (info=" + std::to_string(info) + ")");
        }
    } else if (work.dtype() == DType::Complex128) {
        double* s_data = S.data<double>();
        std::vector<double> superb(superb_size);
        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);
        auto ldu  = full_matrices ? lm : static_cast<lapack_int>(k);
        auto ldvt = full_matrices ? ln : ln;
        int64_t u_stride  = full_matrices ? m * m : m * k;
        int64_t vt_stride = full_matrices ? n_cols * n_cols : k * n_cols;

        for (int64_t b = 0; b < nbatch; ++b) {
            lapack_complex_double* a_mat = c128_ptr(work) + b * m * n_cols;
            lapack_complex_double* u_mat = c128_ptr(U)    + b * u_stride;
            lapack_complex_double* vt_mat= c128_ptr(Vt)   + b * vt_stride;
            double* s_vec = s_data + b * k;
            lapack_int info = LAPACKE_zgesvd(LAPACK_ROW_MAJOR, jobz, jobz,
                lm, ln, a_mat, ln, s_vec, u_mat, ldu, vt_mat, ldvt, superb.data());
            if (info != 0) throw std::runtime_error("linalg::svd: zgesvd failed (info=" + std::to_string(info) + ")");
        }
    } else {
        double* a_data = work.data<double>();
        double* u_data = U.data<double>();
        double* s_data = S.data<double>();
        double* vt_data = Vt.data<double>();
        std::vector<double> superb(superb_size);

        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);
        auto ldu = full_matrices ? lm : static_cast<lapack_int>(k);
        auto ldvt = full_matrices ? ln : ln;
        int64_t u_stride = full_matrices ? m * m : m * k;
        int64_t vt_stride = full_matrices ? n_cols * n_cols : k * n_cols;

        for (int64_t b = 0; b < nbatch; ++b) {
            double* a_mat = a_data + b * m * n_cols;
            double* u_mat = u_data + b * u_stride;
            double* s_vec = s_data + b * k;
            double* vt_mat = vt_data + b * vt_stride;

            lapack_int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, jobz, jobz,
                lm, ln, a_mat, ln, s_vec, u_mat, ldu, vt_mat, ldvt, superb.data());
            if (info != 0) throw std::runtime_error("linalg::svd: computation failed (info=" + std::to_string(info) + ")");
        }
    }

    if (is_complex) {
        // S is already real (Float32 or Float64); U and Vt keep complex dtype
        auto [Uc, Vtc] = canonicalize_uv(U, Vt, k, /*have_vt=*/true);
        return {Uc, S, Vtc};
    }
    {
        auto Ud = maybe_downcast(U, original_dtype);
        auto Vtd = maybe_downcast(Vt, original_dtype);
        auto [Uc, Vtc] = canonicalize_uv(Ud, Vtd, k, /*have_vt=*/true);
        return {Uc, maybe_downcast(S, original_dtype), Vtc};
    }
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}
} // namespace

auto svd(const Tensor& A, bool full_matrices) -> std::tuple<Tensor, Tensor, Tensor> {
    // Try GPU dispatch first
    {
        std::vector<Tensor> results;
        std::array<Tensor, 1> inputs = {A};
        OpAttributes attrs;
        attrs.set(AttrKey::FullMatrices, full_matrices);
        if (try_gpu_dispatch_multi(OpId::LinalgSVD, inputs, attrs, results)) {
            int64_t k = results[1].shape().back();
            auto [Uc, Vtc] = canonicalize_uv(results[0], results[2], k, /*have_vt=*/true);
            return {Uc, results[1], Vtc};
        }
    }
    auto [U, S, Vt] = svd_cpu_impl(A, full_matrices);
    jit_record_linalg1to3_bool_attr(::tenzor::jit::OpType::Svd, A, U, S, Vt,
                                     "full_matrices", full_matrices);
    return {U, S, Vt};
}

namespace {
auto qr_cpu_impl(const Tensor& A) -> std::tuple<Tensor, Tensor> {
#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("qr");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto shape = A.shape();
    auto a_ndim = static_cast<int64_t>(shape.size());
    if (a_ndim < 2) throw std::invalid_argument("linalg::qr: input must be at least 2D");

    int64_t m = shape[a_ndim - 2];
    int64_t n_cols = shape[a_ndim - 1];
    int64_t k = std::min(m, n_cols);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);

    // Q is (m, k), R is (k, n)
    std::vector<int64_t> q_shape = batch_dims;
    q_shape.push_back(m); q_shape.push_back(k);
    std::vector<int64_t> r_shape = batch_dims;
    r_shape.push_back(k); r_shape.push_back(n_cols);

    auto Q = zeros(q_shape, work.dtype(), Device::cpu());
    auto R = zeros(r_shape, work.dtype(), Device::cpu());

    if (work.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* q_data = Q.data<float>();
        float* r_data = R.data<float>();
        std::vector<float> tau(k);

        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);

        for (int64_t b = 0; b < nbatch; ++b) {
            float* a_mat = a_data + b * m * n_cols;
            float* q_mat = q_data + b * m * k;
            float* r_mat = r_data + b * k * n_cols;

            // Compute QR factorization in-place
            lapack_int info = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, lm, ln, a_mat, ln, tau.data());
            if (info != 0) throw std::runtime_error("linalg::qr: factorization failed");

            // Extract R (upper triangle of A)
            for (int64_t i = 0; i < k; ++i) {
                for (int64_t j = 0; j < n_cols; ++j) {
                    r_mat[i * n_cols + j] = (j >= i) ? a_mat[i * n_cols + j] : 0.0f;
                }
            }

            // Generate Q from Householder reflectors
            info = LAPACKE_sorgqr(LAPACK_ROW_MAJOR, lm, static_cast<lapack_int>(k),
                static_cast<lapack_int>(k), a_mat, ln, tau.data());
            if (info != 0) throw std::runtime_error("linalg::qr: Q generation failed");

            // Copy Q columns
            for (int64_t i = 0; i < m; ++i) {
                for (int64_t j = 0; j < k; ++j) {
                    q_mat[i * k + j] = a_mat[i * n_cols + j];
                }
            }
        }
    } else if (work.dtype() == DType::Complex64) {
        std::vector<lapack_complex_float> tau(k);
        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);
        auto lk = static_cast<lapack_int>(k);

        for (int64_t b = 0; b < nbatch; ++b) {
            lapack_complex_float* a_mat = c64_ptr(work) + b * m * n_cols;
            lapack_complex_float* q_mat = c64_ptr(Q)    + b * m * k;
            auto* r_mat = reinterpret_cast<std::complex<float>*>(c64_ptr(R) + b * k * n_cols);
            auto* a_std = reinterpret_cast<const std::complex<float>*>(a_mat);

            lapack_int info = LAPACKE_cgeqrf(LAPACK_ROW_MAJOR, lm, ln, a_mat, ln, tau.data());
            if (info != 0) throw std::runtime_error("linalg::qr: cgeqrf failed");

            // Extract R (upper triangle)
            for (int64_t i = 0; i < k; ++i)
                for (int64_t j = 0; j < n_cols; ++j)
                    r_mat[i * n_cols + j] = (j >= i) ? a_std[i * n_cols + j] : std::complex<float>{0.f, 0.f};

            // Generate unitary Q via cungqr
            info = LAPACKE_cungqr(LAPACK_ROW_MAJOR, lm, lk, lk, a_mat, ln, tau.data());
            if (info != 0) throw std::runtime_error("linalg::qr: cungqr failed");

            // Copy Q columns
            auto* a_std2 = reinterpret_cast<const std::complex<float>*>(a_mat);
            auto* q_std  = reinterpret_cast<std::complex<float>*>(q_mat);
            for (int64_t i = 0; i < m; ++i)
                for (int64_t j = 0; j < k; ++j)
                    q_std[i * k + j] = a_std2[i * n_cols + j];
        }
    } else if (work.dtype() == DType::Complex128) {
        std::vector<lapack_complex_double> tau(k);
        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);
        auto lk = static_cast<lapack_int>(k);

        for (int64_t b = 0; b < nbatch; ++b) {
            lapack_complex_double* a_mat = c128_ptr(work) + b * m * n_cols;
            lapack_complex_double* q_mat = c128_ptr(Q)    + b * m * k;
            auto* r_mat = reinterpret_cast<std::complex<double>*>(c128_ptr(R) + b * k * n_cols);
            auto* a_std = reinterpret_cast<const std::complex<double>*>(a_mat);

            lapack_int info = LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, lm, ln, a_mat, ln, tau.data());
            if (info != 0) throw std::runtime_error("linalg::qr: zgeqrf failed");

            for (int64_t i = 0; i < k; ++i)
                for (int64_t j = 0; j < n_cols; ++j)
                    r_mat[i * n_cols + j] = (j >= i) ? a_std[i * n_cols + j] : std::complex<double>{0., 0.};

            info = LAPACKE_zungqr(LAPACK_ROW_MAJOR, lm, lk, lk, a_mat, ln, tau.data());
            if (info != 0) throw std::runtime_error("linalg::qr: zungqr failed");

            auto* a_std2 = reinterpret_cast<const std::complex<double>*>(a_mat);
            auto* q_std  = reinterpret_cast<std::complex<double>*>(q_mat);
            for (int64_t i = 0; i < m; ++i)
                for (int64_t j = 0; j < k; ++j)
                    q_std[i * k + j] = a_std2[i * n_cols + j];
        }
    } else {
        double* a_data = work.data<double>();
        double* q_data = Q.data<double>();
        double* r_data = R.data<double>();
        std::vector<double> tau(k);

        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);

        for (int64_t b = 0; b < nbatch; ++b) {
            double* a_mat = a_data + b * m * n_cols;
            double* q_mat = q_data + b * m * k;
            double* r_mat = r_data + b * k * n_cols;

            lapack_int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, lm, ln, a_mat, ln, tau.data());
            if (info != 0) throw std::runtime_error("linalg::qr: factorization failed");

            for (int64_t i = 0; i < k; ++i) {
                for (int64_t j = 0; j < n_cols; ++j) {
                    r_mat[i * n_cols + j] = (j >= i) ? a_mat[i * n_cols + j] : 0.0;
                }
            }

            info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, lm, static_cast<lapack_int>(k),
                static_cast<lapack_int>(k), a_mat, ln, tau.data());
            if (info != 0) throw std::runtime_error("linalg::qr: Q generation failed");

            for (int64_t i = 0; i < m; ++i) {
                for (int64_t j = 0; j < k; ++j) {
                    q_mat[i * k + j] = a_mat[i * n_cols + j];
                }
            }
        }
    }

    if (original_dtype == DType::Complex64 || original_dtype == DType::Complex128) {
        return {Q, R};  // already the correct complex dtype
    }
    return {maybe_downcast(Q, original_dtype), maybe_downcast(R, original_dtype)};
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}
} // namespace

auto qr(const Tensor& A) -> std::tuple<Tensor, Tensor> {
    // Try GPU dispatch first
    {
        std::vector<Tensor> results;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch_multi(OpId::LinalgQR, inputs, {}, results)) {
            return {results[0], results[1]};
        }
    }
    auto [Q, R] = qr_cpu_impl(A);
    jit_record_linalg1to2(::tenzor::jit::OpType::Qr, A, Q, R);
    return {Q, R};
}

namespace {
auto eigh_cpu_impl(const Tensor& A) -> std::tuple<Tensor, Tensor> {
#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("eigh");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);

    // Eigenvalues shape: (..., N)
    // For complex Hermitian input the eigenvalues are real: Float32 for Complex64,
    // Float64 for Complex128. For real symmetric input the dtype matches input.
    std::vector<int64_t> w_shape = batch_dims;
    w_shape.push_back(n);
    DType w_dtype = work.dtype();
    if (w_dtype == DType::Complex64)  w_dtype = DType::Float32;
    if (w_dtype == DType::Complex128) w_dtype = DType::Float64;
    auto W = zeros(w_shape, w_dtype, Device::cpu());

    // Eigenvectors are stored in work (overwritten by dsyev/ssyev/cheev/zheev)

    if (work.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* w_data = W.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = a_data + b * n * n;
            float* w_vec = w_data + b * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', 'U', ln, mat, ln, w_vec);
            if (info != 0) throw std::runtime_error("linalg::eigh: computation failed (info=" + std::to_string(info) + ")");
        }
    } else if (work.dtype() == DType::Complex64) {
        // cheev overwrites work with eigenvectors (columns, complex); eigenvalues are real floats
        float* w_data = W.data<float>();
        for (int64_t b = 0; b < nbatch; ++b) {
            lapack_complex_float* mat = c64_ptr(work) + b * n * n;
            float* w_vec = w_data + b * n;
            auto ln = static_cast<lapack_int>(n);
            lapack_int info = LAPACKE_cheev(LAPACK_ROW_MAJOR, 'V', 'U', ln, mat, ln, w_vec);
            if (info != 0) throw std::runtime_error("linalg::eigh: cheev failed (info=" + std::to_string(info) + ")");
        }
    } else if (work.dtype() == DType::Complex128) {
        double* w_data = W.data<double>();
        for (int64_t b = 0; b < nbatch; ++b) {
            lapack_complex_double* mat = c128_ptr(work) + b * n * n;
            double* w_vec = w_data + b * n;
            auto ln = static_cast<lapack_int>(n);
            lapack_int info = LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', ln, mat, ln, w_vec);
            if (info != 0) throw std::runtime_error("linalg::eigh: zheev failed (info=" + std::to_string(info) + ")");
        }
    } else {
        double* a_data = work.data<double>();
        double* w_data = W.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = a_data + b * n * n;
            double* w_vec = w_data + b * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', ln, mat, ln, w_vec);
            if (info != 0) throw std::runtime_error("linalg::eigh: computation failed (info=" + std::to_string(info) + ")");
        }
    }

    // work now contains eigenvectors (columns of orthogonal/unitary matrix)
    // For complex input, work still has the correct complex dtype; no downcast needed.
    if (original_dtype == DType::Complex64 || original_dtype == DType::Complex128) {
        // W is already the correct real dtype; work holds complex eigenvectors.
        auto [Vc, unused] = canonicalize_uv(work, Tensor(), n, /*have_vt=*/false);
        (void)unused;
        return {W, Vc};
    }
    {
        auto Wd = maybe_downcast(W, original_dtype);
        auto Vd = maybe_downcast(work, original_dtype);
        auto [Vc, unused] = canonicalize_uv(Vd, Tensor(), n, /*have_vt=*/false);
        (void)unused;
        return {Wd, Vc};
    }
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}
} // namespace

auto eigh(const Tensor& A) -> std::tuple<Tensor, Tensor> {
    // Try GPU dispatch first
    {
        std::vector<Tensor> results;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch_multi(OpId::LinalgEigh, inputs, {}, results)) {
            int64_t nn = results[0].shape().back();
            auto [Vc, unused] = canonicalize_uv(results[1], Tensor(), nn, /*have_vt=*/false);
            (void)unused;
            return {results[0], Vc};
        }
    }
    auto [W, V] = eigh_cpu_impl(A);
    jit_record_linalg1to2(::tenzor::jit::OpType::Eigh, A, W, V);
    return {W, V};
}

auto eigvalsh(const Tensor& A) -> Tensor {
    // Try GPU dispatch via eigh (compute eigenvalues + eigenvectors, return only eigenvalues)
    {
        std::vector<Tensor> results;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch_multi(OpId::LinalgEigh, inputs, {}, results)) {
            return results[0];  // eigenvalues
        }
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("eigvalsh");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);

    std::vector<int64_t> w_shape = batch_dims;
    w_shape.push_back(n);
    // For complex Hermitian input the eigenvalues are real: Float32 for
    // Complex64, Float64 for Complex128 (mirrors eigh()).
    DType w_dtype = work.dtype();
    if (w_dtype == DType::Complex64)  w_dtype = DType::Float32;
    if (w_dtype == DType::Complex128) w_dtype = DType::Float64;
    auto W = zeros(w_shape, w_dtype, Device::cpu());

    if (work.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* w_data = W.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = a_data + b * n * n;
            float* w_vec = w_data + b * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'N', 'U', ln, mat, ln, w_vec);
            if (info != 0) throw std::runtime_error("linalg::eigvalsh: computation failed");
        }
    } else if (work.dtype() == DType::Complex64) {
        // cheev with jobz='N' computes only the (real) eigenvalues of a
        // Hermitian matrix; mat is overwritten with intermediate data.
        float* w_data = W.data<float>();
        for (int64_t b = 0; b < nbatch; ++b) {
            lapack_complex_float* mat = c64_ptr(work) + b * n * n;
            float* w_vec = w_data + b * n;
            auto ln = static_cast<lapack_int>(n);
            lapack_int info = LAPACKE_cheev(LAPACK_ROW_MAJOR, 'N', 'U', ln, mat, ln, w_vec);
            if (info != 0) throw std::runtime_error("linalg::eigvalsh: cheev failed (info=" + std::to_string(info) + ")");
        }
    } else if (work.dtype() == DType::Complex128) {
        double* w_data = W.data<double>();
        for (int64_t b = 0; b < nbatch; ++b) {
            lapack_complex_double* mat = c128_ptr(work) + b * n * n;
            double* w_vec = w_data + b * n;
            auto ln = static_cast<lapack_int>(n);
            lapack_int info = LAPACKE_zheev(LAPACK_ROW_MAJOR, 'N', 'U', ln, mat, ln, w_vec);
            if (info != 0) throw std::runtime_error("linalg::eigvalsh: zheev failed (info=" + std::to_string(info) + ")");
        }
    } else {
        double* a_data = work.data<double>();
        double* w_data = W.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = a_data + b * n * n;
            double* w_vec = w_data + b * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', ln, mat, ln, w_vec);
            if (info != 0) throw std::runtime_error("linalg::eigvalsh: computation failed");
        }
    }

    // For complex input W already has the correct real dtype; no downcast.
    if (original_dtype == DType::Complex64 || original_dtype == DType::Complex128) {
        jit_record_linalg1(::tenzor::jit::OpType::Eigvalsh, A, W);
        return W;
    }
    {
        Tensor result = maybe_downcast(W, original_dtype);
        jit_record_linalg1(::tenzor::jit::OpType::Eigvalsh, A, result);
        return result;
    }
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto eig(const Tensor& A) -> std::tuple<Tensor, Tensor, Tensor> {
    // Try GPU dispatch first
    {
        std::vector<Tensor> results;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch_multi(OpId::LinalgEig, inputs, {}, results)) {
            return canonicalize_eig(results[0], results[1], results[2]);
        }
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("eig");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);

    // Eigenvalues shape: (..., N) — real and imaginary parts separately
    std::vector<int64_t> w_shape = batch_dims;
    w_shape.push_back(n);

    // Eigenvectors shape: (..., N, N)
    std::vector<int64_t> v_shape = batch_dims;
    v_shape.push_back(n);
    v_shape.push_back(n);

    // For a genuinely complex matrix use cgeev/zgeev, which return complex
    // eigenvalues (single array) and complex right eigenvectors. To keep the
    // {W_real, W_imag, V} contract used by callers (autograd EigBackward,
    // CPU kernel registry), split the complex eigenvalues into real/imag
    // real-typed tensors and return complex eigenvectors in V.
    if (work.dtype() == DType::Complex64 || work.dtype() == DType::Complex128) {
        DType real_dtype = (work.dtype() == DType::Complex64) ? DType::Float32 : DType::Float64;
        auto Wr = zeros(w_shape, real_dtype, Device::cpu());
        auto Wi = zeros(w_shape, real_dtype, Device::cpu());
        auto V  = zeros(v_shape, work.dtype(), Device::cpu());

        if (work.dtype() == DType::Complex64) {
            std::vector<std::complex<float>> w_buf(n);
            float* wr_data = Wr.data<float>();
            float* wi_data = Wi.data<float>();
            for (int64_t b = 0; b < nbatch; ++b) {
                lapack_complex_float* mat = c64_ptr(work) + b * n * n;
                lapack_complex_float* v_mat = c64_ptr(V) + b * n * n;
                auto* w_ptr = reinterpret_cast<lapack_complex_float*>(w_buf.data());
                auto ln = static_cast<lapack_int>(n);
                lapack_int info = LAPACKE_cgeev(LAPACK_ROW_MAJOR, 'N', 'V',
                    ln, mat, ln, w_ptr, nullptr, ln, v_mat, ln);
                if (info != 0) {
                    throw std::runtime_error("linalg::eig: cgeev failed (info=" +
                        std::to_string(info) + ")");
                }
                for (int64_t i = 0; i < n; ++i) {
                    wr_data[b * n + i] = w_buf[i].real();
                    wi_data[b * n + i] = w_buf[i].imag();
                }
            }
        } else {
            std::vector<std::complex<double>> w_buf(n);
            double* wr_data = Wr.data<double>();
            double* wi_data = Wi.data<double>();
            for (int64_t b = 0; b < nbatch; ++b) {
                lapack_complex_double* mat = c128_ptr(work) + b * n * n;
                lapack_complex_double* v_mat = c128_ptr(V) + b * n * n;
                auto* w_ptr = reinterpret_cast<lapack_complex_double*>(w_buf.data());
                auto ln = static_cast<lapack_int>(n);
                lapack_int info = LAPACKE_zgeev(LAPACK_ROW_MAJOR, 'N', 'V',
                    ln, mat, ln, w_ptr, nullptr, ln, v_mat, ln);
                if (info != 0) {
                    throw std::runtime_error("linalg::eig: zgeev failed (info=" +
                        std::to_string(info) + ")");
                }
                for (int64_t i = 0; i < n; ++i) {
                    wr_data[b * n + i] = w_buf[i].real();
                    wi_data[b * n + i] = w_buf[i].imag();
                }
            }
        }
        auto [Wr_out, Wi_out, V_out] = canonicalize_eig(Wr, Wi, V);
        jit_record_linalg1to3(::tenzor::jit::OpType::LinalgEig, A, Wr_out, Wi_out, V_out);
        return {Wr_out, Wi_out, V_out};
    }

    auto Wr = zeros(w_shape, work.dtype(), Device::cpu());  // real part of eigenvalues
    auto Wi = zeros(w_shape, work.dtype(), Device::cpu());  // imaginary part of eigenvalues
    auto Vr = zeros(v_shape, work.dtype(), Device::cpu());  // right eigenvectors

    if (work.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* wr_data = Wr.data<float>();
        float* wi_data = Wi.data<float>();
        float* vr_data = Vr.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = a_data + b * n * n;
            float* wr_vec = wr_data + b * n;
            float* wi_vec = wi_data + b * n;
            float* vr_mat = vr_data + b * n * n;
            auto ln = static_cast<lapack_int>(n);

            // Compute eigenvalues and right eigenvectors (no left eigenvectors)
            lapack_int info = LAPACKE_sgeev(LAPACK_ROW_MAJOR, 'N', 'V',
                ln, mat, ln, wr_vec, wi_vec,
                nullptr, ln, vr_mat, ln);
            if (info != 0) {
                throw std::runtime_error("linalg::eig: computation failed (info=" +
                    std::to_string(info) + ")");
            }
        }
    } else {
        double* a_data = work.data<double>();
        double* wr_data = Wr.data<double>();
        double* wi_data = Wi.data<double>();
        double* vr_data = Vr.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = a_data + b * n * n;
            double* wr_vec = wr_data + b * n;
            double* wi_vec = wi_data + b * n;
            double* vr_mat = vr_data + b * n * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V',
                ln, mat, ln, wr_vec, wi_vec,
                nullptr, ln, vr_mat, ln);
            if (info != 0) {
                throw std::runtime_error("linalg::eig: computation failed (info=" +
                    std::to_string(info) + ")");
            }
        }
    }

    {
        auto [Wr_out, Wi_out, V_out] = canonicalize_eig(
            maybe_downcast(Wr, original_dtype), maybe_downcast(Wi, original_dtype),
            maybe_downcast(Vr, original_dtype));
        jit_record_linalg1to3(::tenzor::jit::OpType::LinalgEig, A, Wr_out, Wi_out, V_out);
        return {Wr_out, Wi_out, V_out};
    }
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto matrix_power(const Tensor& A, int64_t n) -> Tensor {
    if (A.ndim() < 2) {
        throw std::invalid_argument("linalg::matrix_power: expected at least 2D tensor");
    }
    auto shape = A.shape();
    int64_t rows = shape[A.ndim() - 2];
    int64_t cols = shape[A.ndim() - 1];
    if (rows != cols) {
        throw std::invalid_argument("linalg::matrix_power: matrix must be square, got " +
            std::to_string(rows) + "x" + std::to_string(cols));
    }

    if (n == 0) {
        // A^0 = identity matrix. For batched input (ndim > 2) the identity
        // must be broadcast across every leading batch dimension so the
        // output shape matches A, not a bare (rows, rows).
        auto eye2d = tenzor::eye(rows, std::nullopt, A.dtype(), A.device());
        if (A.ndim() == 2) {
            return eye2d;
        }
        std::vector<int64_t> out_shape(shape.begin(), shape.end());
        // Reshape the 2-D identity to (1,...,1,rows,rows) then expand.
        std::vector<int64_t> id_shape(A.ndim(), 1);
        id_shape[A.ndim() - 2] = rows;
        id_shape[A.ndim() - 1] = cols;
        return tenzor::expand(tenzor::reshape(eye2d, id_shape), out_shape).contiguous();
    }

    // For negative exponents, invert first then exponentiate. Compute the
    // magnitude unsigned: std::abs(INT64_MIN) is undefined behavior since its
    // magnitude is not representable as int64_t. The n==0 case returned above.
    Tensor base = (n < 0) ? inv(A) : A;
    uint64_t exp = (n < 0) ? (~static_cast<uint64_t>(n) + 1) : static_cast<uint64_t>(n);

    // Binary exponentiation: O(log n) matmuls
    Tensor result = base;
    exp--;
    while (exp > 0) {
        if (exp & 1u) {
            result = tenzor::matmul(result, base);
        }
        base = tenzor::matmul(base, base);
        exp >>= 1;
    }

    return result;
}

auto lu(const Tensor& A) -> std::tuple<Tensor, Tensor, Tensor> {
    // Try GPU dispatch first
    {
        std::vector<Tensor> results;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch_multi(OpId::LinalgLU, inputs, {}, results)) {
            return {results[0], results[1], results[2]};
        }
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("lu");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    if (is_complex_dtype(work.dtype())) {
        throw std::invalid_argument(
            "linalg::lu: complex (Complex64/Complex128) is not supported");
    }
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    auto shape = A.shape();
    std::vector<int64_t> batch_dims;
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);

    std::vector<int64_t> mat_shape = batch_dims;
    mat_shape.push_back(n); mat_shape.push_back(n);

    std::vector<int64_t> piv_shape = batch_dims;
    piv_shape.push_back(n);

    auto L = zeros(mat_shape, work.dtype(), Device::cpu());
    auto U = zeros(mat_shape, work.dtype(), Device::cpu());
    auto pivots_out = zeros(piv_shape, DType::Int32, Device::cpu());

    if (work.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* l_data = L.data<float>();
        float* u_data = U.data<float>();
        int32_t* piv_data = pivots_out.data<int32_t>();
        std::vector<lapack_int> ipiv(n);

        for (int64_t b = 0; b < nbatch; ++b) {
            float* a_mat = a_data + b * n * n;
            float* l_mat = l_data + b * n * n;
            float* u_mat = u_data + b * n * n;
            int32_t* piv_mat = piv_data + b * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, ln, ln, a_mat, ln, ipiv.data());
            if (info < 0) throw std::runtime_error("linalg::lu: invalid argument " + std::to_string(-info));

            // Extract L (unit lower triangular) and U (upper triangular) from packed result
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    if (i > j) {
                        l_mat[i * n + j] = a_mat[i * n + j];
                        u_mat[i * n + j] = 0.0f;
                    } else if (i == j) {
                        l_mat[i * n + j] = 1.0f;
                        u_mat[i * n + j] = a_mat[i * n + j];
                    } else {
                        l_mat[i * n + j] = 0.0f;
                        u_mat[i * n + j] = a_mat[i * n + j];
                    }
                }
                piv_mat[i] = static_cast<int32_t>(ipiv[i]);
            }
        }
    } else {
        double* a_data = work.data<double>();
        double* l_data = L.data<double>();
        double* u_data = U.data<double>();
        int32_t* piv_data = pivots_out.data<int32_t>();
        std::vector<lapack_int> ipiv(n);

        for (int64_t b = 0; b < nbatch; ++b) {
            double* a_mat = a_data + b * n * n;
            double* l_mat = l_data + b * n * n;
            double* u_mat = u_data + b * n * n;
            int32_t* piv_mat = piv_data + b * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, ln, ln, a_mat, ln, ipiv.data());
            if (info < 0) throw std::runtime_error("linalg::lu: invalid argument " + std::to_string(-info));

            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    if (i > j) {
                        l_mat[i * n + j] = a_mat[i * n + j];
                        u_mat[i * n + j] = 0.0;
                    } else if (i == j) {
                        l_mat[i * n + j] = 1.0;
                        u_mat[i * n + j] = a_mat[i * n + j];
                    } else {
                        l_mat[i * n + j] = 0.0;
                        u_mat[i * n + j] = a_mat[i * n + j];
                    }
                }
                piv_mat[i] = static_cast<int32_t>(ipiv[i]);
            }
        }
    }

    {
        Tensor L_out = maybe_downcast(L, original_dtype);
        Tensor U_out = maybe_downcast(U, original_dtype);
        jit_record_linalg1to3(::tenzor::jit::OpType::LinalgLU, A, L_out, U_out, pivots_out);
        return {L_out, U_out, pivots_out};
    }
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto lu_solve(const Tensor& LU_data, const Tensor& pivots,
              const Tensor& B) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 3> inputs = {LU_data, pivots, B};
        if (try_gpu_dispatch(OpId::LinalgLUSolve, inputs, {}, result)) return result;
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("lu_solve");
#else
    auto original_dtype = B.dtype();
    auto work_lu = prepare_matrix(LU_data);
    auto work_b = prepare_matrix(B);
    if (is_complex_dtype(work_lu.dtype())) {
        throw std::invalid_argument(
            "linalg::lu_solve: complex (Complex64/Complex128) is not supported");
    }

    auto lu_shape = LU_data.shape();
    auto b_shape = B.shape();
    auto lu_ndim = static_cast<int64_t>(lu_shape.size());
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    if (lu_ndim < 2 || b_ndim < 2)
        throw std::invalid_argument("linalg::lu_solve: inputs must be at least 2D");

    int64_t n = lu_shape[lu_ndim - 1];
    // LAPACKE_?getrs indexes lu_mat = lu_ptr + b*n*n and reads n*n elements
    // per batch; a non-square LU factor would read past the buffer.
    if (lu_shape[lu_ndim - 2] != n) {
        throw std::invalid_argument(
            "linalg::lu_solve: LU factor must be square, got " +
            std::to_string(lu_shape[lu_ndim - 2]) + "x" + std::to_string(n));
    }
    if (b_shape[b_ndim - 2] != n) {
        throw std::invalid_argument(
            "linalg::lu_solve: B row dimension (" +
            std::to_string(b_shape[b_ndim - 2]) +
            ") must match LU order n (" + std::to_string(n) + ")");
    }
    int64_t nrhs = b_shape[b_ndim - 1];
    int64_t nbatch = batch_size(work_lu);
    // B is overwritten in place per batch (b_ptr + b*n*nrhs); require matching
    // batch count to avoid out-of-bounds read/write when B is not batched like LU.
    if (batch_size(work_b) != nbatch) {
        throw std::invalid_argument(
            "linalg::lu_solve: batch count of LU (" + std::to_string(nbatch) +
            ") does not match batch count of B (" + std::to_string(batch_size(work_b)) + ")");
    }
    (void)to_lapack_int(n, "n");
    (void)to_lapack_int(nrhs, "nrhs");

    auto piv_cpu = pivots.to(Device::cpu()).contiguous();

    if (work_lu.dtype() == DType::Float32) {
        float* lu_ptr = work_lu.data<float>();
        float* b_ptr = work_b.data<float>();
        auto* piv_ptr = piv_cpu.data<int32_t>();
        std::vector<lapack_int> ipiv(n);

        for (int64_t b = 0; b < nbatch; ++b) {
            float* lu_mat = lu_ptr + b * n * n;
            float* b_mat = b_ptr + b * n * nrhs;
            int32_t* piv_mat = piv_ptr + b * n;
            for (int64_t i = 0; i < n; ++i) ipiv[i] = static_cast<lapack_int>(piv_mat[i]);

            auto ln = static_cast<lapack_int>(n);
            auto lnrhs = static_cast<lapack_int>(nrhs);
            lapack_int info = LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', ln, lnrhs,
                lu_mat, ln, ipiv.data(), b_mat, lnrhs);
            if (info != 0) throw std::runtime_error("linalg::lu_solve: solve failed");
        }
    } else {
        double* lu_ptr = work_lu.data<double>();
        double* b_ptr = work_b.data<double>();
        auto* piv_ptr = piv_cpu.data<int32_t>();
        std::vector<lapack_int> ipiv(n);

        for (int64_t b = 0; b < nbatch; ++b) {
            double* lu_mat = lu_ptr + b * n * n;
            double* b_mat = b_ptr + b * n * nrhs;
            int32_t* piv_mat = piv_ptr + b * n;
            for (int64_t i = 0; i < n; ++i) ipiv[i] = static_cast<lapack_int>(piv_mat[i]);

            auto ln = static_cast<lapack_int>(n);
            auto lnrhs = static_cast<lapack_int>(nrhs);
            lapack_int info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', ln, lnrhs,
                lu_mat, ln, ipiv.data(), b_mat, lnrhs);
            if (info != 0) throw std::runtime_error("linalg::lu_solve: solve failed");
        }
    }

    {
        Tensor result = maybe_downcast(work_b, original_dtype);
        jit_record_linalg3to1(::tenzor::jit::OpType::LinalgLUSolve, LU_data, pivots, B,
                               result);
        return result;
    }
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

// ============================================================================
// lstsq / pinv / matrix_exp — higher-level linalg routines composed on top of
// the LAPACKE primitives above. These run on CPU only; per project policy
// (no silent CPU fallbacks) they throw rather than migrate GPU tensors to
// host invisibly.
// ============================================================================

auto lstsq(const Tensor& A, const Tensor& B) -> std::tuple<Tensor, Tensor> {
    // Least-squares solver for A @ x = B with m >= n full-rank A, expressed
    // entirely as a composition over the registered linalg primitives:
    //   Q, R = qr(A, mode='reduced')      // Q: (m, n), R: (n, n)
    //   x    = solve_triangular(R, Qᵀ B, upper=true)
    //   res  = sum((A x - B)², dim=0)     // squared residual per column
    //
    // Every op in that chain has a per-backend kernel (cuSOLVER / rocSOLVER
    // / Vulkan compute / oneMKL), so the routine runs entirely on whatever
    // device A is on — no LAPACK round-trip, no host pointer loops.
    //
    // Under-determined systems (m < n) need a min-norm solution; the QR
    // path can't produce that, so we throw and direct the caller to pinv.
    auto a_shape = A.shape();
    auto b_shape = B.shape();
    if (a_shape.size() != 2) {
        throw std::invalid_argument("linalg::lstsq: A must be 2D (got ndim=" +
            std::to_string(a_shape.size()) + ")");
    }
    if (b_shape.size() != 1 && b_shape.size() != 2) {
        throw std::invalid_argument("linalg::lstsq: B must be 1D or 2D");
    }
    const int64_t m = a_shape[0];
    const int64_t n = a_shape[1];
    if (b_shape[0] != m) {
        throw std::invalid_argument("linalg::lstsq: A rows (" + std::to_string(m) +
            ") must match B rows (" + std::to_string(b_shape[0]) + ")");
    }
    const auto original_dtype = A.dtype();
    const auto orig_device    = A.device();
    const bool b_was_1d       = (b_shape.size() == 1);
    const int64_t nrhs        = b_was_1d ? 1 : b_shape[1];

    // Widen reduced-precision floats so the QR / SVD recurrences don't lose
    // significance on accumulated rank-1 updates.
    DType compute_dtype = original_dtype;
    if (needs_upcast(compute_dtype)) compute_dtype = DType::Float32;

    // Complex linalg primitives (qr / svd / triangular-solve) are implemented
    // only on the CPU LAPACK path in this codebase — the GPU kernels handle
    // Float32/Float64 and throw for complex, which qr() catches and silently
    // services on CPU. Composing lstsq from those ops on a GPU device therefore
    // mixed CPU results (from qr) with GPU operands (B), producing NaN. Run the
    // whole complex composition on CPU for device consistency and move the
    // result back to the caller's device at the end.
    const bool is_complex_dt = (compute_dtype == DType::Complex64 ||
                                compute_dtype == DType::Complex128);
    const Device compute_device = is_complex_dt ? Device::cpu() : orig_device;

    Tensor A_c = A.contiguous();
    if (A_c.dtype() != compute_dtype) A_c = A_c.to(compute_dtype);
    if (A_c.device() != compute_device) A_c = A_c.to(compute_device);
    Tensor B_c = B.contiguous();
    if (B_c.dtype() != compute_dtype) B_c = B_c.to(compute_dtype);
    if (B_c.device() != compute_device) B_c = B_c.to(compute_device);

    // Promote 1D B to (m, 1) so the composition is uniform; we'll squeeze
    // the solution back at the end.
    if (b_was_1d) B_c = tenzor::reshape(B_c, {m, 1});

    Tensor x;
    Tensor residuals;

    if (m < n) {
        // Under-determined: there are infinitely many exact solutions; we
        // return the minimum-norm one via the SVD-based Moore-Penrose
        // pseudo-inverse:   x = pinv(A) @ B.
        // pinv itself is composed from svd / transpose / where / mul /
        // matmul, all of which dispatch per-backend, so this path runs on
        // whatever device A lives on. Residuals are conventionally empty
        // for under-determined systems (A·x = B is satisfied within the
        // pseudoinverse cutoff) — matches numpy.linalg.lstsq semantics.
        Tensor pinvA = tenzor::linalg::pinv(A_c);                // (n, m)
        x = tenzor::matmul(pinvA, B_c);                          // (n, nrhs)
        residuals = tenzor::zeros({0}, compute_dtype, orig_device);
    } else {
        // Over-determined or square: QR gives the least-squares solution
        // when A has full column rank. Reduced QR returns Q (m, n), R (n, n).
        auto [Q, R] = tenzor::linalg::qr(A_c);
        // Complex least squares uses the conjugate (Hermitian) transpose:
        // x = R^{-1} Q^H b. Conjugate the contiguous Q first, THEN transpose
        // (conjugating the transposed view can be mishandled). For real inputs
        // Q^H == Q^T.
        const bool is_complex = (compute_dtype == DType::Complex64 ||
                                 compute_dtype == DType::Complex128);
        Tensor QT = is_complex
            ? tenzor::transpose(tenzor::conj(Q), -2, -1)         // Q^H
            : tenzor::transpose(Q, -2, -1);                      // Q^T
        Tensor QTB = tenzor::matmul(QT, B_c);                    // (n, nrhs)
        x = tenzor::linalg::solve_triangular(
            R, QTB, /*upper=*/true, /*unitriangular=*/false);    // (n, nrhs)

        // Residual per column: sum over m of |A x − B|². Empty for m == n.
        // numpy.linalg.lstsq returns REAL residuals = sum(|A x - B|^2). For
        // complex inputs |z|^2 = z·conj(z); sum(z^2) would yield a complex,
        // wrong-magnitude value. real(diff·conj(diff)) gives the squared
        // magnitude and a real-typed result.
        if (m > n) {
            Tensor Ax    = tenzor::matmul(A_c, x);
            Tensor diff  = tenzor::sub(Ax, B_c);
            Tensor sq;
            if (is_complex) {
                Tensor mag2 = tenzor::real(tenzor::mul(diff, tenzor::conj(diff)));
                sq = mag2;
            } else {
                sq = tenzor::mul(diff, diff);
            }
            Tensor sumsq = tenzor::sum(sq, /*dim=*/0, /*keepdim=*/false);
            residuals = tenzor::reshape(sumsq, {nrhs});
        } else {
            residuals = tenzor::zeros({0}, compute_dtype, orig_device);
        }
    }

    // Squeeze solution back to 1D if B was 1D.
    if (b_was_1d) x = tenzor::reshape(x, {n});

    Tensor x_out = maybe_downcast(x, original_dtype);
    Tensor res_out = maybe_downcast(residuals, original_dtype);
    if (x_out.device() != orig_device) x_out = x_out.to(orig_device);
    if (res_out.device() != orig_device) res_out = res_out.to(orig_device);
    return {x_out, res_out};
}

auto pinv(const Tensor& A, double rcond) -> Tensor {
    // pinv(A) = V @ diag(s_inv) @ U^T, with s_inv[i] = 1/s[i] when
    // s[i] > rcond * max(s), else 0.
    //
    // Implementation: pure op composition over the tensor primitives that
    // already have per-backend kernels (svd, max, gt, reciprocal, where,
    // transpose, mul, matmul). No host scalar work, no D2H/H2D round-trip,
    // and the entire thing runs on whatever device A is on.
    auto a_shape = A.shape();
    if (a_shape.size() != 2) {
        throw std::invalid_argument("linalg::pinv: A must be 2D (got ndim=" +
            std::to_string(a_shape.size()) + ")");
    }

    auto original_dtype = A.dtype();
    auto original_device = A.device();

    // Reduced SVD: U is (M, K), S is (K,), Vt is (K, N).
    auto [U, S, Vt] = svd(A, /*full_matrices=*/false);

    const int64_t k = S.numel();
    if (k == 0) {
        return zeros({a_shape[1], a_shape[0]}, original_dtype, original_device);
    }

    // SVD widens reduced-precision inputs to F32/F64 internally; keep that
    // dtype for the pseudoinverse arithmetic and downcast at the end.
    auto compute_dtype = S.dtype();

    // cutoff = rcond * max(S), as a 0-D tensor on the same device as S.
    Tensor max_s = tenzor::max(S);
    Tensor rcond_t = tenzor::full({}, rcond, compute_dtype, S.device());
    Tensor cutoff = tenzor::mul(max_s, rcond_t);

    // s_inv[i] = (S[i] > cutoff) ? 1/S[i] : 0.
    // tenzor::where selects element-wise — the reciprocal of any zero
    // singular value produces +inf in the unselected slot but is masked out,
    // so no NaN escapes.
    Tensor mask = tenzor::gt(S, cutoff);
    Tensor s_recip = tenzor::reciprocal(S);
    Tensor zero_vec = tenzor::zeros({k}, compute_dtype, S.device());
    Tensor s_inv = tenzor::where(mask, s_recip, zero_vec);  // (K,)

    // pinv = (Vt^T * s_inv_row) @ U^T
    // Vt: (K, N) → V = Vt^T: (N, K). Broadcast s_inv (1, K) along the last
    // dim to scale columns of V.
    // For a complex A = U S Vʰ the pseudoinverse is V S⁺ Uʰ, i.e. the
    // conjugate transpose of U and Vt — a plain transpose omits the
    // conjugation and yields a wrong result for complex inputs.
    const bool is_complex = (U.dtype() == DType::Complex64 ||
                             U.dtype() == DType::Complex128);
    Tensor V = tenzor::transpose(Vt, -2, -1);            // (N, K)
    if (is_complex) V = tenzor::conj(V);
    Tensor s_inv_row = tenzor::reshape(s_inv, {1, k});   // (1, K), broadcasts
    Tensor V_scaled = tenzor::mul(V, s_inv_row);         // (N, K)
    Tensor UT = tenzor::transpose(U, -2, -1);            // (K, M)
    if (is_complex) UT = tenzor::conj(UT);
    Tensor result = tenzor::matmul(V_scaled, UT);        // (N, M)

    return maybe_downcast(result, original_dtype);
}

auto matrix_exp(const Tensor& A) -> Tensor {
    // Scaling-and-squaring with Padé-13 approximation (Higham 2005).
    // Thresholds chosen to match scipy.linalg.expm; see table on p.16 of the
    // paper for the justification of theta_13 ≈ 5.37.
    //
    // The algorithm is expressed in tensor ops (matmul / add / mul / solve)
    // that all dispatch per-device, so this runs on CPU and GPU alike.
    auto shape = A.shape();
    if (shape.size() != 2 || shape[0] != shape[1]) {
        throw std::invalid_argument("linalg::matrix_exp: A must be a square matrix");
    }
    const int64_t n = shape[0];

    auto original_dtype = A.dtype();
    const Device dev = A.device();

    // Upcast Float16 / BFloat16 to Float32 for numerical stability; keep
    // Float64 as Float64.
    DType compute_dtype = original_dtype;
    if (needs_upcast(compute_dtype)) compute_dtype = DType::Float32;

    auto work = A.contiguous();
    if (work.dtype() != compute_dtype) work = work.to(compute_dtype);

    // Compute ||A||_1 = max over j of sum_i |A[i,j]| using tensor ops so we
    // stay on-device. Result is a scalar tensor; move to CPU for the branch.
    Tensor col_sums = tenzor::sum(tenzor::abs(work), /*dim=*/0);
    Tensor one_norm_t = tenzor::max(col_sums);
    double one_norm = static_cast<double>(
        one_norm_t.to(DType::Float64).to(Device::cpu()).data<double>()[0]);

    // Choose scaling factor s such that ||A / 2^s||_1 <= theta_13.
    constexpr double theta_13 = 5.371920351148152;
    int s = 0;
    if (one_norm > theta_13) {
        s = static_cast<int>(std::ceil(std::log2(one_norm / theta_13)));
        if (s < 0) s = 0;
    }
    const double scale = std::ldexp(1.0, -s);

    // Padé-13 coefficients.
    static const double b[] = {
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0,  129060195264000.0,   10559470521600.0,
        670442572800.0,      33522128640.0,       1323241920.0,
        40840800.0,          960960.0,            16380.0,
        182.0,               1.0
    };

    auto scalar_ct = [&](double v) {
        return full({1}, v, compute_dtype, dev);
    };

    // Build A_scaled = A * scale, identity I on the compute device.
    auto A_scaled = mul(work, scalar_ct(scale));
    auto I = eye(n, std::nullopt, compute_dtype, dev);

    // Power series: compute A^2, A^4, A^6 once.
    auto A2 = matmul(A_scaled, A_scaled);
    auto A4 = matmul(A2, A2);
    auto A6 = matmul(A4, A2);

    // U = A * (A6*(b13 A6 + b11 A4 + b9 A2) + b7 A6 + b5 A4 + b3 A2 + b1 I)
    auto inner_u = A6 * scalar_ct(b[13]) + A4 * scalar_ct(b[11]) + A2 * scalar_ct(b[9]);
    auto outer_u = matmul(A6, inner_u) + A6 * scalar_ct(b[7]) + A4 * scalar_ct(b[5]) +
                   A2 * scalar_ct(b[3]) + I * scalar_ct(b[1]);
    auto U = matmul(A_scaled, outer_u);

    // V = A6*(b12 A6 + b10 A4 + b8 A2) + b6 A6 + b4 A4 + b2 A2 + b0 I
    auto inner_v = A6 * scalar_ct(b[12]) + A4 * scalar_ct(b[10]) + A2 * scalar_ct(b[8]);
    auto V = matmul(A6, inner_v) + A6 * scalar_ct(b[6]) + A4 * scalar_ct(b[4]) +
             A2 * scalar_ct(b[2]) + I * scalar_ct(b[0]);

    // R = (V - U)^{-1} @ (V + U)
    auto P = V + U;
    auto Q = V - U;
    auto R = solve(Q, P);

    // Square s times: R = R @ R.
    for (int i = 0; i < s; ++i) {
        R = matmul(R, R);
    }

    if (R.dtype() != original_dtype) R = R.to(original_dtype);
    return R;
}

// =========================================================================
// New linear algebra operations for PyTorch parity (compositions)
// =========================================================================

auto outer(const Tensor& a, const Tensor& b) -> Tensor {
    // a[:, None] * b[None, :] → (M, N) from (M,) and (N,)
    auto a2 = a.reshape({a.numel(), 1});
    auto b2 = b.reshape({1, b.numel()});
    return tenzor::mul(a2, b2);
}

auto inner(const Tensor& a, const Tensor& b) -> Tensor {
    // For 1D: dot product. For ND: sum product over last dims
    if (a.ndim() == 1 && b.ndim() == 1) {
        return tenzor::dot(a, b);
    }
    // General: contract last dim of a with last dim of b
    return tenzor::matmul(a, b);
}

auto vdot(const Tensor& a, const Tensor& b) -> Tensor {
    // Conjugate dot: conj(a) . b
    auto a_flat = a.reshape({-1});
    auto b_flat = b.reshape({-1});
    auto a_conj = tenzor::conj(a_flat);
    return tenzor::dot(a_conj, b_flat);
}

auto svdvals(const Tensor& A) -> Tensor {
    if (A.ndim() < 2) {
        throw std::invalid_argument("linalg::svdvals: input must be at least 2-D, got " +
                                    std::to_string(A.ndim()) + "-D");
    }
    auto [U, S, Vh] = svd(A, /*full_matrices=*/false);
    return S;
}

auto eigvals(const Tensor& A) -> std::tuple<Tensor, Tensor> {
    if (A.ndim() < 2) {
        throw std::invalid_argument("linalg::eigvals: input must be at least 2-D, got " +
                                    std::to_string(A.ndim()) + "-D");
    }
    auto shape = A.shape();
    if (shape[shape.size() - 2] != shape[shape.size() - 1]) {
        throw std::invalid_argument("linalg::eigvals: input must be a square matrix, got (" +
                                    std::to_string(shape[shape.size() - 2]) + ", " +
                                    std::to_string(shape[shape.size() - 1]) + ")");
    }
    auto [vals_real, vals_imag, vecs] = eig(A);
    return {vals_real, vals_imag};
}

auto cond(const Tensor& A, const std::string& p) -> Tensor {
    if (A.ndim() < 2) {
        throw std::invalid_argument("linalg::cond: input must be at least 2-D, got " +
                                    std::to_string(A.ndim()) + "-D");
    }
    auto shape = A.shape();
    if (shape[shape.size() - 2] != shape[shape.size() - 1]) {
        throw std::invalid_argument("linalg::cond: input must be a square matrix for p=\"" +
                                    p + "\", got (" + std::to_string(shape[shape.size() - 2]) +
                                    ", " + std::to_string(shape[shape.size() - 1]) + ")");
    }

    if (p == "2") {
        // Per-matrix condition number: reduce singular values over the last
        // dim only so a batched input (..., N, N) yields a (...) result rather
        // than a single global scalar mixing all batches.
        auto S = svdvals(A);
        auto s_max = tenzor::max(S, -1, /*keepdim=*/false);
        auto s_min = tenzor::min(S, -1, /*keepdim=*/false);
        return tenzor::div(s_max, s_min);
    } else if (p == "fro") {
        // Batch-aware Frobenius norm over the last two dims (vector_norm
        // normalizes the negative dims), so cond is computed per matrix.
        auto norm_A = vector_norm(A, 2.0, {-2, -1}, /*keepdim=*/false);
        auto A_inv = inv(A);
        auto norm_inv = vector_norm(A_inv, 2.0, {-2, -1}, /*keepdim=*/false);
        return tenzor::mul(norm_A, norm_inv);
    } else {
        throw std::invalid_argument("linalg::cond: unsupported norm order \"" + p +
                                    "\", expected \"2\" or \"fro\"");
    }
}

auto matrix_rank(const Tensor& A, double tol) -> Tensor {
    if (A.ndim() < 2) {
        throw std::invalid_argument("linalg::matrix_rank: input must be at least 2-D, got " +
                                    std::to_string(A.ndim()) + "-D");
    }
    auto S = svdvals(A);  // (..., K)
    Tensor threshold;
    if (tol < 0) {
        // Default tolerance: max(M,N) * max(S) * eps. Reduce max over the last
        // dim only (keepdim) so a batched input keeps a per-matrix threshold of
        // shape (..., 1) that broadcasts against S; a global max would mix all
        // batches into a single scalar.
        auto s_max = tenzor::max(S, -1, /*keepdim=*/true);
        auto shape = A.shape();
        double mn = static_cast<double>(std::max(shape[shape.size()-2], shape[shape.size()-1]));
        // Use the true machine epsilon of the compute dtype svdvals returned
        // (Float16/BFloat16 are widened to Float32), matching
        // torch.linalg.matrix_rank's default threshold of max(M,N)*eps*max(S).
        double eps = (S.dtype() == DType::Float64)
                         ? static_cast<double>(std::numeric_limits<double>::epsilon())
                         : static_cast<double>(std::numeric_limits<float>::epsilon());
        threshold = tenzor::mul(s_max, tenzor::full({1}, mn * eps, S.dtype(), S.device()));
    } else {
        threshold = tenzor::full({1}, tol, S.dtype(), S.device());
    }
    auto mask = tenzor::gt(S, threshold);
    // Count singular values above threshold per matrix (reduce the last dim
    // only) so a batched (..., M, N) input returns a (...) tensor of ranks;
    // a global sum would collapse all batches into one number.
    return tenzor::sum(mask.to(DType::Int64), /*dim=*/-1, /*keepdim=*/false);
}

auto multi_dot(const std::vector<Tensor>& tensors) -> Tensor {
    if (tensors.size() < 2) {
        throw std::invalid_argument("linalg::multi_dot: need at least 2 tensors, got " +
                                    std::to_string(tensors.size()));
    }

    // Validate shapes: all must be 1-D or 2-D, inner dimensions must match
    const size_t n = tensors.size();
    std::vector<int64_t> rows(n), cols(n);

    for (size_t i = 0; i < n; ++i) {
        if (tensors[i].ndim() == 1) {
            // 1-D tensors: first one treated as row vector, last as column vector
            if (i == 0) {
                rows[i] = 1;
                cols[i] = tensors[i].shape()[0];
            } else if (i == n - 1) {
                rows[i] = tensors[i].shape()[0];
                cols[i] = 1;
            } else {
                throw std::invalid_argument(
                    "linalg::multi_dot: inner tensors (index " + std::to_string(i) +
                    ") must be 2-D, got 1-D");
            }
        } else if (tensors[i].ndim() == 2) {
            rows[i] = tensors[i].shape()[0];
            cols[i] = tensors[i].shape()[1];
        } else {
            throw std::invalid_argument(
                "linalg::multi_dot: tensors must be 1-D or 2-D, tensor at index " +
                std::to_string(i) + " is " + std::to_string(tensors[i].ndim()) + "-D");
        }

        if (i > 0 && cols[i - 1] != rows[i]) {
            throw std::invalid_argument(
                "linalg::multi_dot: shape mismatch between tensors " +
                std::to_string(i - 1) + " and " + std::to_string(i) +
                ": (" + std::to_string(rows[i - 1]) + ", " + std::to_string(cols[i - 1]) +
                ") vs (" + std::to_string(rows[i]) + ", " + std::to_string(cols[i]) + ")");
        }
    }

    // Trivial case: 2 matrices
    if (n == 2) {
        return tenzor::matmul(tensors[0], tensors[1]);
    }

    // Dynamic programming for optimal parenthesization (MCM algorithm)
    // cost[i][j] = minimum scalar multiplications to compute product of tensors[i..j]
    // split[i][j] = optimal split point k such that we multiply (i..k) x (k+1..j)
    std::vector<std::vector<int64_t>> cost(n, std::vector<int64_t>(n, 0));
    std::vector<std::vector<size_t>> split(n, std::vector<size_t>(n, 0));

    // dims: chain of dimensions. For matrices A0(r0 x c0), A1(r1 x c1), ...,
    // the dimension array is [r0, c0, c1, c2, ...] (since c_{i-1} == r_i)
    std::vector<int64_t> dims(n + 1);
    dims[0] = rows[0];
    for (size_t i = 0; i < n; ++i) {
        dims[i + 1] = cols[i];
    }

    // Fill DP table: chain length l from 2 to n
    for (size_t l = 2; l <= n; ++l) {
        for (size_t i = 0; i <= n - l; ++i) {
            size_t j = i + l - 1;
            cost[i][j] = std::numeric_limits<int64_t>::max();
            for (size_t k = i; k < j; ++k) {
                int64_t c = cost[i][k] + cost[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1];
                if (c < cost[i][j]) {
                    cost[i][j] = c;
                    split[i][j] = k;
                }
            }
        }
    }

    // Recursively execute the optimal parenthesization
    std::function<Tensor(size_t, size_t)> execute = [&](size_t i, size_t j) -> Tensor {
        if (i == j) {
            return tensors[i];
        }
        auto left = execute(i, split[i][j]);
        auto right = execute(split[i][j] + 1, j);
        return tenzor::matmul(left, right);
    };

    return execute(0, n - 1);
}

auto diag_embed(const Tensor& input, int64_t offset, int64_t dim1, int64_t dim2) -> Tensor {
    // Route through the registered OpId::DiagEmbed backend kernel, which is
    // batch-aware (CPU/CUDA/ROCm/Vulkan/OneAPI all register it and fill the
    // diagonal for both 1D and (..., N) inputs). The previous inline path only
    // filled data for 1D input and returned an all-zeros tensor for any
    // ndim>1 (batched) case.
    //
    // Backend kernels only correctly honor the *default* placement, where the
    // two new size-N matrix axes are the trailing two axes of the output
    // (dim1 = -2, dim2 = -1). Some backends (e.g. CUDA) ignore the dim1/dim2
    // attributes entirely and always emit the matrix on the trailing axes,
    // while the CPU kernel builds a dim1/dim2-aware shape but fills as if the
    // matrix were trailing — so non-default placements diverged across
    // backends. To guarantee identical results everywhere, always dispatch the
    // kernel with the default trailing-axis placement and rearrange the two
    // matrix axes into the requested dim1/dim2 positions here in the op layer.
    OpAttributes attrs;
    attrs.set(AttrKey::Diagonal, offset);
    // Request the default trailing-axis placement from every backend.
    attrs.set(AttrKey::Dim0, static_cast<int64_t>(-2));
    attrs.set(AttrKey::Dim1, static_cast<int64_t>(-1));

    std::array<Tensor, 1> inputs = {input};
    Tensor result = dispatch_single(OpId::DiagEmbed, inputs, attrs);

    // Output rank = input rank + 1. Normalize the requested dim1/dim2 against
    // the output rank, then move the trailing two axes (where the kernel placed
    // the N x N matrix) to those positions.
    const int64_t out_ndim = static_cast<int64_t>(result.shape().size());
    int64_t d1 = dim1 < 0 ? dim1 + out_ndim : dim1;
    int64_t d2 = dim2 < 0 ? dim2 + out_ndim : dim2;

    const int64_t last0 = out_ndim - 2;  // first matrix axis as emitted
    const int64_t last1 = out_ndim - 1;  // second matrix axis as emitted

    // Already in the requested (default) placement: nothing to rearrange.
    if (d1 == last0 && d2 == last1) {
        return result;
    }

    // Move the two trailing matrix axes to the requested destinations. movedim
    // handles all distinct ordered placements (including d1 > d2) correctly.
    return tenzor::movedim(result, {last0, last1}, {d1, d2});
}

auto diagflat(const Tensor& input, int64_t offset) -> Tensor {
    // Flatten input then create diagonal matrix using existing diag()
    auto flat = input.reshape({-1});
    return tenzor::diag(flat, offset);
}

// ============================================================================
// householder_product / ldl_factor / ldl_solve / vector_norm / matrix_norm /
// vecdot — additional linalg routines for PyTorch parity.
// ============================================================================

auto householder_product(const Tensor& input, const Tensor& tau) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 2> inputs = {input, tau};
        if (try_gpu_dispatch(OpId::LinalgHouseholder, inputs, {}, result)) return result;
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("householder_product");
#else
    auto original_dtype = input.dtype();
    auto work = prepare_matrix(input);
    auto tau_work = prepare_matrix(tau);
    if (is_complex_dtype(work.dtype())) {
        throw std::invalid_argument(
            "linalg::householder_product: complex (Complex64/Complex128) is not supported");
    }

    auto shape = input.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2)
        throw std::invalid_argument("linalg::householder_product: input must be at least 2D");

    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];

    auto tau_shape = tau.shape();
    int64_t k = tau_shape.back();
    int64_t nbatch = batch_size(work);

    // LAPACK ?orgqr requires m >= n >= k >= 0. Out-of-range values read past
    // the tau / work buffers. Also require the tau batch count to match the
    // input batch count, since tau_ptr = tau_data + b*k assumes that layout.
    if (k < 0 || n < k || m < n) {
        throw std::invalid_argument(
            "linalg::householder_product: requires m >= n >= k >= 0, got m=" +
            std::to_string(m) + ", n=" + std::to_string(n) + ", k=" +
            std::to_string(k));
    }
    {
        int64_t tau_batch = 1;
        for (int64_t i = 0; i + 1 < static_cast<int64_t>(tau_shape.size()); ++i)
            tau_batch *= tau_shape[i];
        if (tau_batch != nbatch) {
            throw std::invalid_argument(
                "linalg::householder_product: tau batch count (" +
                std::to_string(tau_batch) + ") must match input batch count (" +
                std::to_string(nbatch) + ")");
        }
    }
    (void)to_lapack_int(m, "m");
    (void)to_lapack_int(n, "n");
    (void)to_lapack_int(k, "k");

    if (work.dtype() == DType::Float32) {
        float* data = work.data<float>();
        float* tau_data = tau_work.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = data + b * m * n;
            float* tau_ptr = tau_data + b * k;
            auto lm = static_cast<lapack_int>(m);
            auto ln = static_cast<lapack_int>(n);
            auto lk = static_cast<lapack_int>(k);

            lapack_int info = LAPACKE_sorgqr(LAPACK_ROW_MAJOR, lm, ln, lk,
                mat, ln, tau_ptr);
            if (info != 0)
                throw std::runtime_error("linalg::householder_product: sorgqr failed (info=" +
                    std::to_string(info) + ")");
        }
    } else {
        double* data = work.data<double>();
        double* tau_data = tau_work.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = data + b * m * n;
            double* tau_ptr = tau_data + b * k;
            auto lm = static_cast<lapack_int>(m);
            auto ln = static_cast<lapack_int>(n);
            auto lk = static_cast<lapack_int>(k);

            lapack_int info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, lm, ln, lk,
                mat, ln, tau_ptr);
            if (info != 0)
                throw std::runtime_error("linalg::householder_product: dorgqr failed (info=" +
                    std::to_string(info) + ")");
        }
    }

    {
        Tensor result = maybe_downcast(work, original_dtype);
        jit_record_linalg2to1(::tenzor::jit::OpType::LinalgHouseholder, input, tau, result);
        return result;
    }
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto ldl_factor(const Tensor& A) -> std::tuple<Tensor, Tensor> {
    // Try GPU dispatch first
    {
        std::vector<Tensor> results;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch_multi(OpId::LinalgLDLFactor, inputs, {}, results)) {
            return {results[0], results[1]};
        }
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("ldl_factor");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    if (is_complex_dtype(work.dtype())) {
        throw std::invalid_argument(
            "linalg::ldl_factor: complex (Complex64/Complex128) is not supported");
    }
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    auto shape = A.shape();
    std::vector<int64_t> batch_dims;
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);

    std::vector<int64_t> piv_shape = batch_dims;
    piv_shape.push_back(n);

    auto pivots_out = zeros(piv_shape, DType::Int32, Device::cpu());

    if (work.dtype() == DType::Float32) {
        float* data = work.data<float>();
        int32_t* piv_data = pivots_out.data<int32_t>();
        std::vector<lapack_int> ipiv(n);

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = data + b * n * n;
            int32_t* piv_mat = piv_data + b * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_ssytrf(LAPACK_ROW_MAJOR, 'L', ln, mat, ln, ipiv.data());
            if (info < 0)
                throw std::runtime_error("linalg::ldl_factor: invalid argument " +
                    std::to_string(-info));
            // info > 0 means D has zeros on diagonal (singular), but factorization completed

            for (int64_t i = 0; i < n; ++i)
                piv_mat[i] = static_cast<int32_t>(ipiv[i]);
        }
    } else {
        double* data = work.data<double>();
        int32_t* piv_data = pivots_out.data<int32_t>();
        std::vector<lapack_int> ipiv(n);

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = data + b * n * n;
            int32_t* piv_mat = piv_data + b * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_dsytrf(LAPACK_ROW_MAJOR, 'L', ln, mat, ln, ipiv.data());
            if (info < 0)
                throw std::runtime_error("linalg::ldl_factor: invalid argument " +
                    std::to_string(-info));

            for (int64_t i = 0; i < n; ++i)
                piv_mat[i] = static_cast<int32_t>(ipiv[i]);
        }
    }

    {
        Tensor LD_out = maybe_downcast(work, original_dtype);
        jit_record_linalg1to2(::tenzor::jit::OpType::LinalgLDLFactor, A, LD_out, pivots_out);
        return {LD_out, pivots_out};
    }
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

#if defined(TENZOR_USE_MKL) || defined(TENZOR_USE_LAPACKE)
// LAPACKE_?sytrs (the SOLVE step) trusts its LD/pivots inputs and does not
// re-check for a zero pivot the way LAPACKE_?sytrf (the FACTOR step) does --
// its own `info` only reports illegal-argument errors, never singularity.
// If LD/pivots came from a validated ldl_factor() call this is unreachable
// (factor's own info check already rejects a singular matrix), but a
// hand-crafted or corrupted LD/pivots pair would silently divide by zero
// inside LAPACK, propagating Inf/NaN with no diagnostic. Mirrors the same
// pivot-block decoding ldl_factor's info convention describes, applied
// host-side before ever calling into LAPACK.
template<typename T>
static bool ldl_has_zero_pivot(const T* ld_mat, const lapack_int* ipiv, int64_t n) {
    for (int64_t k = 0; k < n; ) {
        lapack_int p = ipiv[k];
        if (p > 0) {
            if (ld_mat[k * n + k] == T(0)) return true;
            k++;
        } else {
            T d11 = ld_mat[k * n + k];
            T d21 = ld_mat[(k + 1) * n + k];
            T d22 = ld_mat[(k + 1) * n + (k + 1)];
            if (d11 * d22 - d21 * d21 == T(0)) return true;
            k += 2;
        }
    }
    return false;
}
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE

auto ldl_solve(const Tensor& LD, const Tensor& pivots,
               const Tensor& B) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 3> inputs = {LD, pivots, B};
        if (try_gpu_dispatch(OpId::LinalgLDLSolve, inputs, {}, result)) return result;
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("ldl_solve");
#else
    auto original_dtype = B.dtype();
    auto work_ld = prepare_matrix(LD);
    auto work_b = prepare_matrix(B);
    if (is_complex_dtype(work_ld.dtype())) {
        throw std::invalid_argument(
            "linalg::ldl_solve: complex (Complex64/Complex128) is not supported");
    }

    auto ld_shape = LD.shape();
    auto b_shape = B.shape();
    auto ld_ndim = static_cast<int64_t>(ld_shape.size());
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    if (ld_ndim < 2 || b_ndim < 2)
        throw std::invalid_argument("linalg::ldl_solve: inputs must be at least 2D");

    int64_t n = ld_shape[ld_ndim - 1];
    // LAPACKE_?sytrs indexes ld_mat = ld_ptr + b*n*n and reads n*n elements
    // per batch; a non-square LD factor would read past the buffer.
    if (ld_shape[ld_ndim - 2] != n) {
        throw std::invalid_argument(
            "linalg::ldl_solve: LD factor must be square, got " +
            std::to_string(ld_shape[ld_ndim - 2]) + "x" + std::to_string(n));
    }
    if (b_shape[b_ndim - 2] != n) {
        throw std::invalid_argument(
            "linalg::ldl_solve: B row dimension (" +
            std::to_string(b_shape[b_ndim - 2]) +
            ") must match LD order n (" + std::to_string(n) + ")");
    }
    int64_t nrhs = b_shape[b_ndim - 1];
    int64_t nbatch = batch_size(work_ld);
    // B is overwritten in place per batch (b_ptr + b*n*nrhs); require matching
    // batch count to avoid out-of-bounds read/write when B is not batched like LD.
    if (batch_size(work_b) != nbatch) {
        throw std::invalid_argument(
            "linalg::ldl_solve: batch count of LD (" + std::to_string(nbatch) +
            ") does not match batch count of B (" + std::to_string(batch_size(work_b)) + ")");
    }
    (void)to_lapack_int(n, "n");
    (void)to_lapack_int(nrhs, "nrhs");

    auto piv_cpu = pivots.to(Device::cpu()).contiguous();

    if (work_ld.dtype() == DType::Float32) {
        float* ld_ptr = work_ld.data<float>();
        float* b_ptr = work_b.data<float>();
        auto* piv_ptr = piv_cpu.data<int32_t>();
        std::vector<lapack_int> ipiv(n);

        for (int64_t b = 0; b < nbatch; ++b) {
            float* ld_mat = ld_ptr + b * n * n;
            float* b_mat = b_ptr + b * n * nrhs;
            int32_t* piv_mat = piv_ptr + b * n;
            for (int64_t i = 0; i < n; ++i) ipiv[i] = static_cast<lapack_int>(piv_mat[i]);

            if (ldl_has_zero_pivot(ld_mat, ipiv.data(), n)) {
                throw std::runtime_error(
                    "linalg::ldl_solve: singular LDL^T factor (zero pivot)");
            }

            auto ln = static_cast<lapack_int>(n);
            auto lnrhs = static_cast<lapack_int>(nrhs);
            lapack_int info = LAPACKE_ssytrs(LAPACK_ROW_MAJOR, 'L', ln, lnrhs,
                ld_mat, ln, ipiv.data(), b_mat, lnrhs);
            if (info != 0)
                throw std::runtime_error("linalg::ldl_solve: solve failed (info=" +
                    std::to_string(info) + ")");
        }
    } else {
        double* ld_ptr = work_ld.data<double>();
        double* b_ptr = work_b.data<double>();
        auto* piv_ptr = piv_cpu.data<int32_t>();
        std::vector<lapack_int> ipiv(n);

        for (int64_t b = 0; b < nbatch; ++b) {
            double* ld_mat = ld_ptr + b * n * n;
            double* b_mat = b_ptr + b * n * nrhs;
            int32_t* piv_mat = piv_ptr + b * n;
            for (int64_t i = 0; i < n; ++i) ipiv[i] = static_cast<lapack_int>(piv_mat[i]);

            if (ldl_has_zero_pivot(ld_mat, ipiv.data(), n)) {
                throw std::runtime_error(
                    "linalg::ldl_solve: singular LDL^T factor (zero pivot)");
            }

            auto ln = static_cast<lapack_int>(n);
            auto lnrhs = static_cast<lapack_int>(nrhs);
            lapack_int info = LAPACKE_dsytrs(LAPACK_ROW_MAJOR, 'L', ln, lnrhs,
                ld_mat, ln, ipiv.data(), b_mat, lnrhs);
            if (info != 0)
                throw std::runtime_error("linalg::ldl_solve: solve failed (info=" +
                    std::to_string(info) + ")");
        }
    }

    {
        Tensor result = maybe_downcast(work_b, original_dtype);
        jit_record_linalg3to1(::tenzor::jit::OpType::LinalgLDLSolve, LD, pivots, B, result);
        return result;
    }
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto vector_norm(const Tensor& input, double ord,
                 std::vector<int64_t> dim, bool keepdim) -> Tensor {
    // No LAPACKE needed — implemented with tensor ops.
    // GPU tensors use the same tensor-op path (ops dispatch per-element).

    auto x = input;
    // Complex vector norm = norm of elementwise magnitudes (PyTorch semantics);
    // reduce to the real magnitude up front so the downstream real-valued ops
    // (pow / sum / max) compute the correct result on every backend.
    if (input.dtype() == DType::Complex64 || input.dtype() == DType::Complex128) {
        x = tenzor::abs(input);
    }

    // Normalize negative dims against the input rank up front. The reduce
    // helpers sort dims in descending order and reduce sequentially with the
    // caller's keepdim; if a mixed/negative multi-dim set (e.g. {-2,-1}) were
    // left un-normalized, the raw-value sort ([-1,-2]) plus keepdim=false would
    // shift the remaining axes and reduce the wrong dimensions.
    const int64_t nd = input.ndim();
    for (auto& d : dim) {
        if (d < 0) d += nd;
        if (d < 0 || d >= nd) {
            throw std::out_of_range(
                "linalg::vector_norm: dim out of range for input of rank " +
                std::to_string(nd));
        }
    }

    // Helper: reduce over a single dim or all dims
    auto reduce_sum = [&](const Tensor& t, const std::vector<int64_t>& dims,
                          bool kd) -> Tensor {
        if (dims.empty()) {
            return tenzor::sum(t, std::nullopt, kd);
        }
        // Reduce dims from highest to lowest to keep indices valid
        auto sorted_dims = dims;
        std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<int64_t>());
        Tensor result = t;
        for (auto d : sorted_dims) {
            result = tenzor::sum(result, d, kd);
        }
        return result;
    };

    auto reduce_max = [&](const Tensor& t, const std::vector<int64_t>& dims,
                          bool kd) -> Tensor {
        if (dims.empty()) {
            return tenzor::max(t, std::nullopt, kd);
        }
        auto sorted_dims = dims;
        std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<int64_t>());
        Tensor result = t;
        for (auto d : sorted_dims) {
            result = tenzor::max(result, d, kd);
        }
        return result;
    };

    auto reduce_min = [&](const Tensor& t, const std::vector<int64_t>& dims,
                          bool kd) -> Tensor {
        if (dims.empty()) {
            return tenzor::min(t, std::nullopt, kd);
        }
        auto sorted_dims = dims;
        std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<int64_t>());
        Tensor result = t;
        for (auto d : sorted_dims) {
            result = tenzor::min(result, d, kd);
        }
        return result;
    };

    if (std::isinf(ord) && ord > 0) {
        // ord = +inf: max(abs(x))
        return reduce_max(tenzor::abs(x), dim, keepdim);
    } else if (std::isinf(ord) && ord < 0) {
        // ord = -inf: min(abs(x))
        return reduce_min(tenzor::abs(x), dim, keepdim);
    } else if (ord == 0.0) {
        // ord = 0: count of nonzero entries (the L0 "norm", which isn't a
        // norm but is universally documented under this name).
        // Build an exact 0/1 indicator via element-wise inequality with 0
        // (returns Bool), cast to the input dtype, then sum-reduce. This
        // replaces the older `ax / (ax + 1e-38)` trick whose asymptote was
        // never exactly 1 — it biased every nonzero entry slightly low and
        // could even mis-count finite-but-tiny values as fractional.
        auto zero_tensor = tenzor::full({}, 0.0, x.dtype(), x.device());
        auto nz_mask = tenzor::ne(x, zero_tensor);              // Bool
        auto nz_count = nz_mask.to(x.dtype());                  // 0/1 in input dtype
        return reduce_sum(nz_count, dim, keepdim);
    } else if (ord == 1.0) {
        return reduce_sum(tenzor::abs(x), dim, keepdim);
    } else if (ord == 2.0) {
        // Euclidean norm: sqrt(sum(x^2))
        return tenzor::sqrt(reduce_sum(tenzor::mul(x, x), dim, keepdim));
    } else {
        // General p-norm: sum(abs(x)^p)^(1/p)
        auto abs_x = tenzor::abs(x);
        auto powered = tenzor::pow(abs_x, static_cast<float>(ord));
        auto summed = reduce_sum(powered, dim, keepdim);
        return tenzor::pow(summed, static_cast<float>(1.0 / ord));
    }
}

auto matrix_norm(const Tensor& input, double ord) -> Tensor {
    // No LAPACKE needed for most cases — implemented with tensor ops.
    auto shape = input.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2)
        throw std::invalid_argument("linalg::matrix_norm: input must be at least 2D");

    if (ord == 2.0) {
        // Spectral norm: largest singular value
        auto sv = svdvals(input);
        return tenzor::max(sv, -1, false);
    } else if (ord == -2.0) {
        // Smallest singular value
        auto sv = svdvals(input);
        return tenzor::min(sv, -1, false);
    } else if (ord == 1.0) {
        // Max absolute column sum: max_j sum_i |a_ij|
        auto abs_input = tenzor::abs(input);
        auto col_sums = tenzor::sum(abs_input, ndim - 2, false);  // sum over rows
        return tenzor::max(col_sums, -1, false);
    } else if (ord == -1.0) {
        // Min absolute column sum
        auto abs_input = tenzor::abs(input);
        auto col_sums = tenzor::sum(abs_input, ndim - 2, false);
        return tenzor::min(col_sums, -1, false);
    } else if (std::isinf(ord) && ord > 0) {
        // Max absolute row sum: max_i sum_j |a_ij|
        auto abs_input = tenzor::abs(input);
        auto row_sums = tenzor::sum(abs_input, ndim - 1, false);  // sum over cols
        return tenzor::max(row_sums, -1, false);
    } else if (std::isinf(ord) && ord < 0) {
        // Min absolute row sum
        auto abs_input = tenzor::abs(input);
        auto row_sums = tenzor::sum(abs_input, ndim - 1, false);
        return tenzor::min(row_sums, -1, false);
    } else {
        throw std::invalid_argument(
            "linalg::matrix_norm: unsupported ord=" + std::to_string(ord) +
            ". Supported: 1, -1, 2, -2, inf, -inf");
    }
}

auto vecdot(const Tensor& a, const Tensor& b, int64_t dim) -> Tensor {
    // PyTorch convention: vecdot(a, b) = sum(conj(a) * b, dim).
    // For real-valued inputs conj is identity, so the math is unchanged.
    if (a.is_complex()) {
        auto product = tenzor::mul(tenzor::conj(a), b);
        return tenzor::sum(product, dim, false);
    }
    auto product = tenzor::mul(a, b);
    return tenzor::sum(product, dim, false);
}

// ============================================================================
// cholesky_inverse / tensorinv / tensorsolve / ormqr / geqrf
// ============================================================================

auto cholesky_inverse(const Tensor& L, bool upper) -> Tensor {
    // Compose from existing ops: given Cholesky factor L (lower) or U (upper),
    // compute A^{-1} by solving the triangular systems:
    //   If lower: A^{-1} = solve_triangular(L^T, solve_triangular(L, I), upper=true)
    //   If upper: A^{-1} = solve_triangular(U, solve_triangular(U^T, I), upper=true)
    auto shape = L.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) throw std::invalid_argument("linalg::cholesky_inverse: input must be at least 2D");
    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n)
        throw std::invalid_argument("linalg::cholesky_inverse: expected square matrix");

    auto I = tenzor::eye(n, std::nullopt, L.dtype(), L.device());

    // Broadcast I to match batch dimensions if needed
    if (ndim > 2) {
        std::vector<int64_t> eye_shape(shape.begin(), shape.end());
        I = tenzor::expand(I, std::move(eye_shape));
        I = I.contiguous();
    }

    // For a complex Hermitian factor the adjoint is the conjugate transpose L^H,
    // not the plain transpose L^T. conj is identity for real dtypes.
    Tensor L_h = tenzor::transpose(L.is_complex() ? tenzor::conj(L) : L, ndim - 2, ndim - 1);
    if (!upper) {
        // L is lower triangular: solve L @ Y = I, then L^H @ X = Y
        auto Y = solve_triangular(L, I, /*upper=*/false);
        return solve_triangular(L_h, Y, /*upper=*/true);
    } else {
        // U is upper triangular: solve U^H @ Y = I, then U @ X = Y
        auto Y = solve_triangular(L_h, I, /*upper=*/false);
        return solve_triangular(L, Y, /*upper=*/true);
    }
}

auto cholesky_solve(const Tensor& B, const Tensor& L, bool upper) -> Tensor {
    auto shape = L.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) throw std::invalid_argument("linalg::cholesky_solve: L must be at least 2D");
    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n)
        throw std::invalid_argument("linalg::cholesky_solve: expected square Cholesky factor");

    // Compose from solve_triangular (which handles GPU dispatch internally)
    // A = L @ L^T, solve A @ X = B
    // Step 1: L @ Y = B  (forward substitution)
    // Step 2: L^T @ X = Y (back substitution)
    // Complex Hermitian factors require the conjugate transpose L^H (conj is
    // identity for real dtypes).
    Tensor L_h = tenzor::transpose(L.is_complex() ? tenzor::conj(L) : L, ndim - 2, ndim - 1).contiguous();
    if (!upper) {
        // A = L @ L^H: solve L @ Y = B, then L^H @ X = Y
        auto Y = solve_triangular(L.contiguous(), B.contiguous(), /*upper=*/false);
        return solve_triangular(L_h, Y.contiguous(), /*upper=*/true);
    } else {
        // A = U^H @ U: solve U^H @ Y = B, then U @ X = Y
        auto Y = solve_triangular(L_h, B.contiguous(), /*upper=*/false);
        return solve_triangular(L, Y, /*upper=*/true);
    }
}

auto tensorinv(const Tensor& input, int64_t ind) -> Tensor {
    auto shape = input.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ind <= 0 || ind > ndim)
        throw std::invalid_argument("linalg::tensorinv: ind must be in [1, ndim]");

    // Compute the product of the first `ind` dimensions and the remaining dimensions
    int64_t rows = 1;
    for (int64_t i = 0; i < ind; ++i) rows *= shape[i];
    int64_t cols = 1;
    for (int64_t i = ind; i < ndim; ++i) cols *= shape[i];

    if (rows != cols) {
        throw std::invalid_argument(
            "linalg::tensorinv: product of first " + std::to_string(ind) +
            " dims (" + std::to_string(rows) + ") must equal product of remaining dims (" +
            std::to_string(cols) + ")");
    }

    // Reshape to 2D, invert, reshape back
    auto mat = tenzor::reshape(input, {rows, cols});
    auto inv_mat = inv(mat);

    // Output shape: trailing dims followed by leading dims
    std::vector<int64_t> out_shape;
    for (int64_t i = ind; i < ndim; ++i) out_shape.push_back(shape[i]);
    for (int64_t i = 0; i < ind; ++i) out_shape.push_back(shape[i]);

    return tenzor::reshape(inv_mat, out_shape);
}

auto tensorsolve(const Tensor& A, const Tensor& B) -> Tensor {
    auto a_shape = A.shape();
    auto b_shape = B.shape();
    auto a_ndim = static_cast<int64_t>(a_shape.size());
    auto b_ndim = static_cast<int64_t>(b_shape.size());

    // The number of "B dimensions" is b_ndim. The A tensor must have shape
    // (*B.shape, *x_shape) where prod(x_shape) == prod(B.shape).
    // We flatten A into a 2D matrix [prod(B.shape), prod(x_shape)] and solve.
    int64_t rhs_size = 1;
    for (int64_t i = 0; i < b_ndim; ++i) rhs_size *= b_shape[i];

    int64_t total = 1;
    for (int64_t i = 0; i < a_ndim; ++i) total *= a_shape[i];
    int64_t lhs_size = total / rhs_size;

    if (rhs_size != lhs_size) {
        throw std::invalid_argument(
            "linalg::tensorsolve: the dimensions of A do not form a square system "
            "(rhs_size=" + std::to_string(rhs_size) +
            ", lhs_size=" + std::to_string(lhs_size) + ")");
    }

    auto A_2d = tenzor::reshape(A, {rhs_size, lhs_size});
    auto B_flat = tenzor::reshape(B, {rhs_size});
    // solve expects (..., N, K) for B, so unsqueeze to column vector
    auto B_col = tenzor::unsqueeze(B_flat, -1);
    auto X_col = solve(A_2d, B_col);
    auto X_flat = tenzor::squeeze(X_col, -1);

    // Reshape X back to x_shape (the trailing dims of A after b_ndim dims)
    std::vector<int64_t> x_shape;
    for (int64_t i = b_ndim; i < a_ndim; ++i) x_shape.push_back(a_shape[i]);
    if (x_shape.empty()) {
        return X_flat;
    }
    return tenzor::reshape(X_flat, x_shape);
}

auto ormqr(const Tensor& input, const Tensor& tau, const Tensor& other,
           bool left, bool transpose) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 3> inputs = {input, tau, other};
        OpAttributes attrs;
        attrs.set(AttrKey::Left, left);
        attrs.set(AttrKey::TransposeQ, transpose);
        if (try_gpu_dispatch(OpId::Ormqr, inputs, attrs, result)) return result;
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("ormqr");
#else
    auto original_dtype = input.dtype();
    auto work_input = prepare_matrix(input);
    auto work_tau = (needs_upcast(tau.dtype()) ? tau.to(DType::Float32) : tau).contiguous();
    auto work_other = prepare_matrix(other);
    if (is_complex_dtype(work_input.dtype())) {
        throw std::invalid_argument(
            "linalg::ormqr: complex (Complex64/Complex128) is not supported");
    }

    auto in_shape = input.shape();
    auto other_shape = other.shape();
    auto in_ndim = static_cast<int64_t>(in_shape.size());
    auto other_ndim = static_cast<int64_t>(other_shape.size());
    if (in_ndim < 2)
        throw std::invalid_argument("linalg::ormqr: input must be at least 2D");
    if (other_ndim < 2)
        throw std::invalid_argument("linalg::ormqr: other must be at least 2D");

    int64_t m = other_shape[other_ndim - 2];
    int64_t n_cols = other_shape[other_ndim - 1];
    int64_t k = tau.shape()[static_cast<int64_t>(tau.shape().size()) - 1];
    int64_t nbatch = batch_size(work_other);

    // LAPACK ?ormqr requires 0 <= k <= order, where order = m for side 'L'
    // and n_cols for side 'R'. The k reflectors live in input's columns, so
    // input must have at least k columns. Out-of-range values read past the
    // input / tau buffers. The tau batch count must also match other's.
    {
        int64_t order = left ? m : n_cols;
        int64_t in_cols = in_shape[in_ndim - 1];
        if (k < 0 || k > order || in_cols < k) {
            throw std::invalid_argument(
                "linalg::ormqr: requires 0 <= k <= " +
                std::string(left ? "rows(other)" : "cols(other)") +
                " and cols(input) >= k, got k=" + std::to_string(k) +
                ", order=" + std::to_string(order) +
                ", cols(input)=" + std::to_string(in_cols));
        }
        int64_t tau_batch = 1;
        auto ts = tau.shape();
        for (int64_t i = 0; i + 1 < static_cast<int64_t>(ts.size()); ++i)
            tau_batch *= ts[i];
        if (tau_batch != nbatch) {
            throw std::invalid_argument(
                "linalg::ormqr: tau batch count (" + std::to_string(tau_batch) +
                ") must match other batch count (" + std::to_string(nbatch) + ")");
        }
    }
    (void)to_lapack_int(m, "m");
    (void)to_lapack_int(n_cols, "n");
    (void)to_lapack_int(k, "k");
    char side = left ? 'L' : 'R';
    char trans = transpose ? 'T' : 'N';

    if (work_input.dtype() == DType::Float32) {
        const float* in_data = work_input.data<float>();
        const float* tau_data = work_tau.data<float>();
        float* other_data = work_other.data<float>();

        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);
        auto lk = static_cast<lapack_int>(k);

        int64_t in_m = in_shape[in_ndim - 2];
        int64_t in_n = in_shape[in_ndim - 1];

        for (int64_t b = 0; b < nbatch; ++b) {
            const float* in_mat = in_data + b * in_m * in_n;
            const float* tau_ptr = tau_data + b * k;
            float* other_mat = other_data + b * m * n_cols;

            // LAPACKE_sormqr modifies other in-place
            lapack_int lda = static_cast<lapack_int>(in_n);
            lapack_int ldc = static_cast<lapack_int>(n_cols);
            lapack_int info = LAPACKE_sormqr(LAPACK_ROW_MAJOR, side, trans,
                lm, ln, lk,
                const_cast<float*>(in_mat), lda,
                const_cast<float*>(tau_ptr),
                other_mat, ldc);
            if (info != 0)
                throw std::runtime_error("linalg::ormqr: sormqr failed (info=" +
                    std::to_string(info) + ")");
        }
    } else {
        const double* in_data = work_input.data<double>();
        const double* tau_data = work_tau.data<double>();
        double* other_data = work_other.data<double>();

        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);
        auto lk = static_cast<lapack_int>(k);

        int64_t in_m = in_shape[in_ndim - 2];
        int64_t in_n = in_shape[in_ndim - 1];

        for (int64_t b = 0; b < nbatch; ++b) {
            const double* in_mat = in_data + b * in_m * in_n;
            const double* tau_ptr = tau_data + b * k;
            double* other_mat = other_data + b * m * n_cols;

            lapack_int lda = static_cast<lapack_int>(in_n);
            lapack_int ldc = static_cast<lapack_int>(n_cols);
            lapack_int info = LAPACKE_dormqr(LAPACK_ROW_MAJOR, side, trans,
                lm, ln, lk,
                const_cast<double*>(in_mat), lda,
                const_cast<double*>(tau_ptr),
                other_mat, ldc);
            if (info != 0)
                throw std::runtime_error("linalg::ormqr: dormqr failed (info=" +
                    std::to_string(info) + ")");
        }
    }

    {
        Tensor result = maybe_downcast(work_other, original_dtype);
        jit_record_linalg3to1_2bool_attr(::tenzor::jit::OpType::Ormqr, input, tau, other,
                                          result, "left", left, "transpose", transpose);
        return result;
    }
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto geqrf(const Tensor& input) -> std::tuple<Tensor, Tensor> {
    // Try GPU dispatch first
    {
        std::vector<Tensor> results;
        std::array<Tensor, 1> inputs = {input};
        if (try_gpu_dispatch_multi(OpId::Geqrf, inputs, {}, results)) {
            return {results[0], results[1]};
        }
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("geqrf");
#else
    auto original_dtype = input.dtype();
    auto work = prepare_matrix(input);
    if (is_complex_dtype(work.dtype())) {
        throw std::invalid_argument(
            "linalg::geqrf: complex (Complex64/Complex128) is not supported");
    }
    auto shape = input.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2)
        throw std::invalid_argument("linalg::geqrf: input must be at least 2D");

    int64_t m = shape[ndim - 2];
    int64_t n_cols = shape[ndim - 1];
    int64_t k = std::min(m, n_cols);
    int64_t nbatch = batch_size(work);

    // tau shape: batch_dims + {k}
    std::vector<int64_t> batch_dims;
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);
    std::vector<int64_t> tau_shape = batch_dims;
    tau_shape.push_back(k);

    auto tau_result = tenzor::zeros(tau_shape, work.dtype(), Device::cpu());

    if (work.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* tau_data = tau_result.data<float>();

        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);

        for (int64_t b = 0; b < nbatch; ++b) {
            float* a_mat = a_data + b * m * n_cols;
            float* tau_ptr = tau_data + b * k;

            lapack_int info = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, lm, ln, a_mat, ln, tau_ptr);
            if (info != 0)
                throw std::runtime_error("linalg::geqrf: sgeqrf failed (info=" +
                    std::to_string(info) + ")");
        }
    } else {
        double* a_data = work.data<double>();
        double* tau_data = tau_result.data<double>();

        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);

        for (int64_t b = 0; b < nbatch; ++b) {
            double* a_mat = a_data + b * m * n_cols;
            double* tau_ptr = tau_data + b * k;

            lapack_int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, lm, ln, a_mat, ln, tau_ptr);
            if (info != 0)
                throw std::runtime_error("linalg::geqrf: dgeqrf failed (info=" +
                    std::to_string(info) + ")");
        }
    }

    {
        Tensor A_factored = maybe_downcast(work, original_dtype);
        Tensor tau_out = maybe_downcast(tau_result, original_dtype);
        jit_record_linalg1to2(::tenzor::jit::OpType::Geqrf, input, A_factored, tau_out);
        return {A_factored, tau_out};
    }
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

// ============================================================================
// LOBPCG — Locally Optimal Block Preconditioned Conjugate Gradient
// ============================================================================

auto lobpcg(const Tensor& A, const Tensor& X0, int64_t k,
            const Tensor& B, int64_t max_iter, double tol)
    -> std::pair<Tensor, Tensor>
{
    // Validate inputs
    if (A.ndim() != 2 || A.shape()[0] != A.shape()[1])
        throw std::invalid_argument("lobpcg: A must be a square 2D matrix");
    int64_t N = A.shape()[0];
    if (X0.ndim() != 2 || X0.shape()[0] != N || X0.shape()[1] < k)
        throw std::invalid_argument("lobpcg: X0 must have shape (N, k) with k <= columns");
    if (k < 1 || k > N)
        throw std::invalid_argument("lobpcg: k must be in [1, N]");

    auto dtype = A.dtype();
    auto dev = A.device();

    // GPU dispatch: this is a composed algorithm using existing ops,
    // so it works on all backends. No special GPU dispatch needed.

    // Helper: orthonormalize columns via QR
    auto orthonormalize = [](const Tensor& M) -> Tensor {
        auto [Q, R] = qr(M.contiguous());
        return Q;
    };

    // Helper: regularize a gram matrix by adding eps * I to the diagonal
    auto regularize_gram = [&](const Tensor& G, int64_t sz) -> Tensor {
        double eps = (dtype == DType::Float32) ? 1e-6 : 1e-12;
        auto reg = tenzor::mul(
            eye(sz, std::nullopt, dtype, dev),
            tenzor::full({1}, eps, dtype, dev));
        return tenzor::add(G, reg);
    };

    // Step 1: orthonormalize initial guess, take first k columns
    Tensor X = orthonormalize(
        tenzor::slice(X0, 1, 0, k).contiguous());

    Tensor eigenvalues;
    Tensor P;  // conjugate directions, empty initially
    bool have_P = false;

    for (int64_t iter = 0; iter < max_iter; ++iter) {
        // Compute A @ X
        auto AX = tenzor::matmul(A.contiguous(), X.contiguous());

        // Rayleigh quotient: eigenvalues = diag(X^T A X)
        auto XtAX = tenzor::matmul(
            tenzor::transpose(X, 0, 1).contiguous(), AX.contiguous());

        // Solve small eigenvalue problem for the current subspace X
        auto [evals_x, evecs_x] = eigh(XtAX.contiguous());
        eigenvalues = evals_x;

        // Reorder X to align with eigenvectors of XtAX
        X = tenzor::matmul(X.contiguous(), evecs_x.contiguous());
        AX = tenzor::matmul(AX.contiguous(), evecs_x.contiguous());

        // Compute residual: G = AX - X * diag(eigenvalues)
        // Broadcast eigenvalues (k,) across rows of X (N, k)
        auto evals_row = tenzor::reshape(eigenvalues, {1, k});
        auto G = tenzor::sub(AX, tenzor::mul(X, evals_row));

        // Check convergence: max absolute residual
        auto max_residual = tenzor::max(tenzor::abs(G));
        // Move to CPU for comparison
        auto max_res_cpu = max_residual.to(Device::cpu()).contiguous();
        double residual_val = 0.0;
        if (dtype == DType::Float32) {
            residual_val = static_cast<double>(*max_res_cpu.data<float>());
        } else if (dtype == DType::Float64) {
            residual_val = *max_res_cpu.data<double>();
        } else {
            // For Float16/BFloat16, cast to float32
            auto as_f32 = max_res_cpu.to(DType::Float32);
            residual_val = static_cast<double>(*as_f32.data<float>());
        }

        if (residual_val < tol) break;

        // Apply preconditioner if provided
        Tensor W = G;
        if (B.is_valid() && B.numel() > 0) {
            W = tenzor::matmul(B.contiguous(), G.contiguous());
        }

        // Orthogonalize W against X
        auto XtW = tenzor::matmul(
            tenzor::transpose(X, 0, 1).contiguous(), W.contiguous());
        W = tenzor::sub(W, tenzor::matmul(X.contiguous(), XtW.contiguous()));

        // QR orthonormalize W
        W = orthonormalize(W);

        // Build search subspace S and A@S
        Tensor S, AS;
        if (!have_P) {
            // First iteration: S = [X, W]
            S = tenzor::cat({X, W}, /*dim=*/1);
            auto AW = tenzor::matmul(A.contiguous(), W.contiguous());
            AS = tenzor::cat({AX, AW}, /*dim=*/1);
        } else {
            // Subsequent iterations: S = [X, W, P]
            S = tenzor::cat({X, W, P}, /*dim=*/1);
            auto AW = tenzor::matmul(A.contiguous(), W.contiguous());
            auto AP = tenzor::matmul(A.contiguous(), P.contiguous());
            AS = tenzor::cat({AX, AW, AP}, /*dim=*/1);
        }

        int64_t subspace_dim = S.shape()[1];

        // Solve projected generalized eigenproblem:
        // S^T A S c = lambda * S^T S c
        auto St = tenzor::transpose(S, 0, 1).contiguous();
        auto gram_A = tenzor::matmul(St, AS.contiguous());
        auto gram_S = tenzor::matmul(St, S.contiguous());

        // Regularize gram_S for numerical stability
        gram_S = regularize_gram(gram_S, subspace_dim);

        // Solve the generalized eigenvalue problem:
        //   gram_A @ c = lambda * gram_S @ c
        // via Cholesky reduction: gram_S = L @ L^T
        //   L^{-1} @ gram_A @ L^{-T} @ c' = lambda * c'
        // where c = L^{-T} @ c'
        //
        // This preserves symmetry so eigh gives correct results.
        Tensor sub_evals, sub_evecs;
        try {
            auto L_sub = tenzor::linalg::cholesky(gram_S.contiguous(), false);
            // M = L^{-1} @ gram_A @ L^{-T}
            auto temp = solve_triangular(L_sub, gram_A.contiguous(), false);
            auto L_sub_t = tenzor::transpose(L_sub, 0, 1).contiguous();
            auto M = solve_triangular(L_sub_t, tenzor::transpose(temp, 0, 1).contiguous(), true);
            M = tenzor::transpose(M, 0, 1).contiguous();

            // Symmetrize for numerical stability
            auto Mt = tenzor::transpose(M, 0, 1).contiguous();
            M = tenzor::mul(tenzor::add(M, Mt), tenzor::full({1}, 0.5f, dtype, dev));

            auto [evals_sub, evecs_sub] = eigh(M.contiguous());
            sub_evals = evals_sub;
            // Transform back: c = L^{-T} @ c'
            sub_evecs = solve_triangular(L_sub_t, evecs_sub.contiguous(), true);
        } catch (...) {
            // Fallback: use pinv if Cholesky fails (gram_S near-singular)
            auto M = tenzor::matmul(pinv(gram_S.contiguous()), gram_A.contiguous());
            auto Mt_f = tenzor::transpose(M, 0, 1).contiguous();
            M = tenzor::mul(tenzor::add(M, Mt_f), tenzor::full({1}, 0.5f, dtype, dev));
            auto [evals_f, evecs_f] = eigh(M.contiguous());
            sub_evals = evals_f;
            sub_evecs = evecs_f;
        }

        // Select the k smallest eigenvalues
        eigenvalues = tenzor::slice(sub_evals, 0, 0, k);
        auto coeffs_k = tenzor::slice(sub_evecs, 1, 0, k).contiguous();

        // Update X = S @ coeffs_k
        X = tenzor::matmul(S.contiguous(), coeffs_k);
        X = orthonormalize(X);

        // Update conjugate directions P
        if (subspace_dim > 2 * k) {
            // P = S @ coeffs for directions k..2k
            auto coeffs_p = tenzor::slice(sub_evecs, 1, k,
                                          std::min(2 * k, subspace_dim)).contiguous();
            P = tenzor::matmul(S.contiguous(), coeffs_p);
            P = orthonormalize(P);
        } else {
            // Not enough directions for P, use W
            P = W;
        }
        have_P = true;
    }

    return {eigenvalues, X};
}

} // namespace tenzor::linalg
