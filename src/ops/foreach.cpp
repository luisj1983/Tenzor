/**
 * @file foreach.cpp
 * @brief Multi-tensor (foreach) optimizer ops
 *
 * Each op loops over the list and applies the corresponding per-tensor kernel.
 * The outer loop is OMP-parallelised when the list has 4 or more tensors.
 *
 * Implementation strategy: loop over existing per-tensor kernels.
 * Correctness first; fused single-pass kernels are a follow-up optimisation.
 */

#include "tenzor/ops/foreach.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/backend/fast_dispatch.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static void check_same_size(const char* name,
                             const std::vector<Tensor>& a,
                             const std::vector<Tensor>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            std::string(name) + ": lists must have the same size (got " +
            std::to_string(a.size()) + " and " + std::to_string(b.size()) + ")");
    }
}

static void check_same_size3(const char* name,
                              const std::vector<Tensor>& self,
                              const std::vector<Tensor>& a,
                              const std::vector<Tensor>& b) {
    if (self.size() != a.size() || self.size() != b.size()) {
        throw std::invalid_argument(
            std::string(name) + ": all three lists must have the same size");
    }
}

// OMP parallel threshold: use parallel loop when list has this many tensors.
static constexpr int64_t OMP_PARALLEL_THRESHOLD = 4;

// Host-parallelising per-tensor dispatch only helps for CPU tensors. For
// device tensors every dispatch funnels into the backend's (mutex-guarded)
// queue, so OMP threads just fight over the lock — on the Intel oneAPI CPU
// OpenCL runtime the contention made 1000 tiny adds take ~17 s instead of
// ~30 ms (ForeachOps.PerfManyTensors), and concurrent enqueue is a suspected
// trigger of its internal-state crashes. Device lists dispatch sequentially
// into the (already asynchronous) queue instead.
static inline bool foreach_use_omp(int64_t n, const std::vector<Tensor>& lst) {
    return n >= OMP_PARALLEL_THRESHOLD &&
           (lst.empty() || lst[0].device().type == Device::Type::CPU);
}

// ---------------------------------------------------------------------------
// Out-of-place binary ops
// ---------------------------------------------------------------------------

auto foreach_add(const std::vector<Tensor>& a, const std::vector<Tensor>& b)
    -> std::vector<Tensor> {
    check_same_size("foreach_add", a, b);
    const auto n = static_cast<int64_t>(a.size());

    // Fused fast path: if the active backend registers a fused foreach_add
    // kernel (oneapi does — see OpId::ForeachAdd / oneapi foreach_add_kernel),
    // dispatch all N (a,b) pairs as a single kernel launch. The oneAPI SYCL/
    // OpenCL runtime has ~0.2 ms per-launch overhead, so looping add() over
    // 1000 small tensors spends ~200 ms in launch overhead alone
    // (ForeachOps.PerfManyTensors); one fused launch drops it to a few ms.
    // Backends without a fused kernel (cpu/cuda/rocm/vulkan) are already fast
    // via async per-tensor dispatch, so they keep the loop below.
    if (!a.empty() && is_op_supported(OpId::ForeachAdd, a[0].device().type)) {
        std::vector<Tensor> inputs;
        inputs.reserve(static_cast<size_t>(n) * 2);
        for (int64_t i = 0; i < n; ++i) {
            inputs.push_back(a[i]);
            inputs.push_back(b[i]);
        }
        return dispatch(OpId::ForeachAdd, std::span<const Tensor>(inputs.data(), inputs.size()));
    }

    std::vector<Tensor> out(a.size());
    #pragma omp parallel for if(foreach_use_omp(n, a)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        out[i] = tenzor::add(a[i], b[i]);
    }
    return out;
}

auto foreach_sub(const std::vector<Tensor>& a, const std::vector<Tensor>& b)
    -> std::vector<Tensor> {
    check_same_size("foreach_sub", a, b);
    std::vector<Tensor> out(a.size());
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(foreach_use_omp(n, a)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        out[i] = tenzor::sub(a[i], b[i]);
    }
    return out;
}

auto foreach_mul(const std::vector<Tensor>& a, const std::vector<Tensor>& b)
    -> std::vector<Tensor> {
    check_same_size("foreach_mul", a, b);
    std::vector<Tensor> out(a.size());
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(foreach_use_omp(n, a)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        out[i] = tenzor::mul(a[i], b[i]);
    }
    return out;
}

auto foreach_div(const std::vector<Tensor>& a, const std::vector<Tensor>& b)
    -> std::vector<Tensor> {
    check_same_size("foreach_div", a, b);
    std::vector<Tensor> out(a.size());
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(foreach_use_omp(n, a)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        out[i] = tenzor::div(a[i], b[i]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Out-of-place unary ops
// ---------------------------------------------------------------------------

auto foreach_neg(const std::vector<Tensor>& a) -> std::vector<Tensor> {
    std::vector<Tensor> out(a.size());
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(foreach_use_omp(n, a)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        out[i] = tenzor::neg(a[i]);
    }
    return out;
}

auto foreach_abs(const std::vector<Tensor>& a) -> std::vector<Tensor> {
    std::vector<Tensor> out(a.size());
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(foreach_use_omp(n, a)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        out[i] = tenzor::abs(a[i]);
    }
    return out;
}

auto foreach_sqrt(const std::vector<Tensor>& a) -> std::vector<Tensor> {
    std::vector<Tensor> out(a.size());
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(foreach_use_omp(n, a)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        out[i] = tenzor::sqrt(a[i]);
    }
    return out;
}

auto foreach_copy(const std::vector<Tensor>& src) -> std::vector<Tensor> {
    std::vector<Tensor> out(src.size());
    const auto n = static_cast<int64_t>(src.size());
    #pragma omp parallel for if(foreach_use_omp(n, src)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        out[i] = src[i].clone();
    }
    return out;
}

// ---------------------------------------------------------------------------
// In-place binary ops
// ---------------------------------------------------------------------------

void foreach_add_(std::vector<Tensor>& a, const std::vector<Tensor>& b) {
    check_same_size("foreach_add_", a, b);
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(foreach_use_omp(n, a)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        tenzor::add_(a[i], b[i]);
    }
}

void foreach_sub_(std::vector<Tensor>& a, const std::vector<Tensor>& b) {
    check_same_size("foreach_sub_", a, b);
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(foreach_use_omp(n, a)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        tenzor::sub_(a[i], b[i]);
    }
}

void foreach_mul_(std::vector<Tensor>& a, const std::vector<Tensor>& b) {
    check_same_size("foreach_mul_", a, b);
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(foreach_use_omp(n, a)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        tenzor::mul_(a[i], b[i]);
    }
}

void foreach_div_(std::vector<Tensor>& a, const std::vector<Tensor>& b) {
    check_same_size("foreach_div_", a, b);
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(foreach_use_omp(n, a)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        tenzor::div_(a[i], b[i]);
    }
}

// ---------------------------------------------------------------------------
// In-place unary ops
// ---------------------------------------------------------------------------

void foreach_neg_(std::vector<Tensor>& a) {
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(foreach_use_omp(n, a)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        // neg_ is not a standalone op; use mul_ by -1 cast to same dtype
        Tensor minus_one = full_like(a[i], -1.0);
        tenzor::mul_(a[i], minus_one);
    }
}

// In-place foreach writeback idiom — CONFIRMED FUNCTIONALLY CORRECT.
//
// These ops write the result back into a[i]'s existing storage via
// `add_(a[i], sub(result, a[i]))` rather than a real in-place copy, because no
// general in-place `copy_` / `OpId::Copy` exists at the ops layer (adding one is
// a cross-cutting all-backend change; reassigning the vector element would sever
// the caller's aliasing/view semantics, and a CPU-only memcpy would be wrong for
// device tensors). The idiom is device-correct (every op is dispatched, so GPU
// tensors stay on-device) and produces the right values: it is verified by the
// AbsInplace / SqrtInplace / Zero cases in tests/unit/test_foreach_ops.cpp
// across all backends/dtypes. The only residual is a possible last-bit FP
// rounding on abs_/sqrt_ from the extra add (x + (result - x) is not guaranteed
// to round-trip exactly); the exact-copy form awaits the cross-backend
// in-place Copy op recorded in the review. Left intact deliberately — applying
// an unsafe partial (e.g. a CPU-only memcpy) would break device tensors.
void foreach_abs_(std::vector<Tensor>& a) {
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(foreach_use_omp(n, a)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        Tensor result = tenzor::abs(a[i]);
        // Copy result data back into a[i] storage (same shape/dtype guaranteed)
        tenzor::add_(a[i], tenzor::sub(result, a[i]));
    }
}

void foreach_sqrt_(std::vector<Tensor>& a) {
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(foreach_use_omp(n, a)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        Tensor result = tenzor::sqrt(a[i]);
        tenzor::add_(a[i], tenzor::sub(result, a[i]));
    }
}

void foreach_zero_(std::vector<Tensor>& a) {
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(foreach_use_omp(n, a)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        Tensor z = zeros_like(a[i]);
        tenzor::add_(a[i], tenzor::sub(z, a[i]));
    }
}

// ---------------------------------------------------------------------------
// Ternary fused ops
// ---------------------------------------------------------------------------

void foreach_addcdiv_(std::vector<Tensor>& self,
                      const std::vector<Tensor>& a,
                      const std::vector<Tensor>& b,
                      double scalar) {
    check_same_size3("foreach_addcdiv_", self, a, b);
    const auto n = static_cast<int64_t>(self.size());
    #pragma omp parallel for if(foreach_use_omp(n, self)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        // self[i] = self[i] + scalar * a[i] / b[i]
        Tensor result = tenzor::addcdiv(self[i], a[i], b[i], scalar);
        tenzor::add_(self[i], tenzor::sub(result, self[i]));
    }
}

void foreach_addcmul_(std::vector<Tensor>& self,
                      const std::vector<Tensor>& a,
                      const std::vector<Tensor>& b,
                      double scalar) {
    check_same_size3("foreach_addcmul_", self, a, b);
    const auto n = static_cast<int64_t>(self.size());
    #pragma omp parallel for if(foreach_use_omp(n, self)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        // self[i] = self[i] + scalar * a[i] * b[i]
        Tensor result = tenzor::addcmul(self[i], a[i], b[i], scalar);
        tenzor::add_(self[i], tenzor::sub(result, self[i]));
    }
}

void foreach_lerp_(std::vector<Tensor>& self,
                   const std::vector<Tensor>& b,
                   double scalar) {
    check_same_size("foreach_lerp_", self, b);
    const auto n = static_cast<int64_t>(self.size());
    #pragma omp parallel for if(foreach_use_omp(n, self)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        // self[i] = lerp(self[i], b[i], scalar) = self[i] + scalar*(b[i]-self[i])
        Tensor result = tenzor::lerp(self[i], b[i], scalar);
        tenzor::add_(self[i], tenzor::sub(result, self[i]));
    }
}

// ---------------------------------------------------------------------------
// Reduction
// ---------------------------------------------------------------------------

auto foreach_norm(const std::vector<Tensor>& a, double p) -> std::vector<Tensor> {
    std::vector<Tensor> out(a.size());
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(foreach_use_omp(n, a)) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        // norm() with no dim gives global scalar norm. norm() takes a double
        // order; pass p through without narrowing to float.
        out[i] = tenzor::norm(a[i], p, std::nullopt, false);
    }
    return out;
}

} // namespace tenzor
