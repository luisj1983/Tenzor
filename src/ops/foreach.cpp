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

// ---------------------------------------------------------------------------
// Out-of-place binary ops
// ---------------------------------------------------------------------------

auto foreach_add(const std::vector<Tensor>& a, const std::vector<Tensor>& b)
    -> std::vector<Tensor> {
    check_same_size("foreach_add", a, b);
    std::vector<Tensor> out(a.size());
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
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
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
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
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
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
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
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
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        out[i] = tenzor::neg(a[i]);
    }
    return out;
}

auto foreach_abs(const std::vector<Tensor>& a) -> std::vector<Tensor> {
    std::vector<Tensor> out(a.size());
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        out[i] = tenzor::abs(a[i]);
    }
    return out;
}

auto foreach_sqrt(const std::vector<Tensor>& a) -> std::vector<Tensor> {
    std::vector<Tensor> out(a.size());
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        out[i] = tenzor::sqrt(a[i]);
    }
    return out;
}

auto foreach_copy(const std::vector<Tensor>& src) -> std::vector<Tensor> {
    std::vector<Tensor> out(src.size());
    const auto n = static_cast<int64_t>(src.size());
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
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
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        tenzor::add_(a[i], b[i]);
    }
}

void foreach_sub_(std::vector<Tensor>& a, const std::vector<Tensor>& b) {
    check_same_size("foreach_sub_", a, b);
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        tenzor::sub_(a[i], b[i]);
    }
}

void foreach_mul_(std::vector<Tensor>& a, const std::vector<Tensor>& b) {
    check_same_size("foreach_mul_", a, b);
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        tenzor::mul_(a[i], b[i]);
    }
}

void foreach_div_(std::vector<Tensor>& a, const std::vector<Tensor>& b) {
    check_same_size("foreach_div_", a, b);
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        tenzor::div_(a[i], b[i]);
    }
}

// ---------------------------------------------------------------------------
// In-place unary ops
// ---------------------------------------------------------------------------

void foreach_neg_(std::vector<Tensor>& a) {
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        // neg_ is not a standalone op; use mul_ by -1 cast to same dtype
        Tensor minus_one = full_like(a[i], -1.0);
        tenzor::mul_(a[i], minus_one);
    }
}

void foreach_abs_(std::vector<Tensor>& a) {
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        Tensor result = tenzor::abs(a[i]);
        // Copy result data back into a[i] storage (same shape/dtype guaranteed)
        tenzor::add_(a[i], tenzor::sub(result, a[i]));
    }
}

void foreach_sqrt_(std::vector<Tensor>& a) {
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        Tensor result = tenzor::sqrt(a[i]);
        tenzor::add_(a[i], tenzor::sub(result, a[i]));
    }
}

void foreach_zero_(std::vector<Tensor>& a) {
    const auto n = static_cast<int64_t>(a.size());
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
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
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
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
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
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
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
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
    const auto p_f = static_cast<float>(p);
    #pragma omp parallel for if(n >= OMP_PARALLEL_THRESHOLD) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        // norm() with no dim gives global scalar norm
        out[i] = tenzor::norm(a[i], p_f, std::nullopt, false);
    }
    return out;
}

} // namespace tenzor
