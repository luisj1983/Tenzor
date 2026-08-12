/**
 * @file pointwise_kernel.hpp
 * @brief Generic binary pointwise kernel that eliminates dtype dispatch
 *        boilerplate from individual kernels.
 *
 * Handles: output allocation, broadcasting, dtype dispatch, SIMD selection,
 * and OpenMP parallelization. Individual ops only need to provide:
 * 1. A SIMD dispatch function for the contiguous fast path
 * 2. A scalar lambda for the broadcasting path
 *
 * This replaces ~200 lines of per-kernel boilerplate with ~5 lines per op.
 */

#pragma once

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/dtype_dispatch.hpp"
#include "tenzor/backend/omp_thresholds.hpp"
#include "broadcast.hpp"
#include "fp8_emulation.hpp"
#include <algorithm>
#include <cstddef>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "simd_traits.hpp"

namespace tenzor::cpu {

namespace detail {
// Caps the OMP thread count used to chunk a pointwise op so tiny tensors
// just above the threshold don't get split into more chunks than there is
// useful work per chunk (each chunk still needs to clear the
// SimdTrait::apply call's own fixed overhead). Mirrors the existing
// ACTIVATION_OMP_THRESHOLD-family convention of gating parallelism by size,
// just applied per-chunk instead of per-element.
inline int pointwise_num_threads(size_t n) {
    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    // At least ~65536 elements per thread so each chunk's SIMD loop
    // amortizes its own fixed costs; never fewer than 1 thread.
    int by_work = static_cast<int>(std::max<size_t>(1, n / 65536));
    return std::max(1, std::min(max_threads, by_work));
}

// Chunked-parallel dispatch for the same-shape fast path. Pulled out into its
// own (non-macro-expanded) function template so the `#pragma omp` directive
// isn't embedded inside TENZOR_DISPATCH_ALL_TYPES_AND_COMPLEX's macro
// argument list -- GCC accepts directives-in-macro-args syntactically
// (C++23) but this build fails to parse it correctly when nested inside the
// dispatch macro's generated switch/lambda, so keep the pragma in ordinary
// code instead.
template<typename Op, typename scalar_t>
void pointwise_apply_parallel(const scalar_t* a_data, const scalar_t* b_data,
                               scalar_t* c_data, size_t n) {
    if (static_cast<int64_t>(n) >= ::tenzor::OmpThresholds::simple()) {
        const int num_threads = pointwise_num_threads(n);
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int t = 0; t < num_threads; ++t) {
            const size_t chunk = (n + static_cast<size_t>(num_threads) - 1)
                                 / static_cast<size_t>(num_threads);
            const size_t start = static_cast<size_t>(t) * chunk;
            const size_t end = std::min(start + chunk, n);
            if (start < end) {
                SimdTrait<Op, scalar_t>::apply(a_data + start, b_data + start,
                                                c_data + start, end - start);
            }
        }
    } else {
        SimdTrait<Op, scalar_t>::apply(a_data, b_data, c_data, n);
    }
}
} // namespace detail

// ============================================================================
// binary_pointwise_kernel — the main entry point
// ============================================================================

/**
 * @brief Generic binary pointwise kernel.
 *
 * Handles FP8 emulation, output allocation, broadcasting, dtype dispatch,
 * and SIMD selection. Replaces ~200 lines of boilerplate per binary op.
 *
 * @tparam Op Operation tag type (AddOp, SubOp, MulOp, DivOp)
 * @tparam FP8Emulate FP8 emulation function (nullptr-like to skip)
 * @param a First input tensor
 * @param b Second input tensor
 * @param fp8_emulate Optional FP8 emulation function
 * @return Result tensor
 */
template<typename Op>
auto binary_pointwise_kernel(
    const Tensor& a_raw, const Tensor& b_raw,
    Tensor (*fp8_emulate)(const Tensor&, const Tensor&) = nullptr) -> Tensor
{
    // Validation is done by the caller (validate_elementwise in math.cpp)

    // audit-2026-05-03 bug #15 mirror: ensure both inputs are contiguous.
    // The same-shape fast path below reads via flat pointer; for non-contiguous
    // slice/expand views this would skip logical elements (the same root cause
    // as the sum/sigmoid/tanh kernel fixes).
    auto a = a_raw.contiguous();
    auto b = b_raw.contiguous();

    // FP8 emulation
    if (fp8_emulate && is_fp8(a.dtype()) && is_fp8(b.dtype())) {
        return fp8_emulate(a, b);
    }

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result = Tensor::empty_uninitialized(output_shape, a.dtype(), a.device());

    // Early return for zero-element outputs (e.g. broadcasting with empty tensors)
    if (result.numel() == 0) {
        return result;
    }

    bool same_shape = (shape_a_vec == shape_b_vec);
    if (same_shape) {
        // Contiguous fast path with SIMD, parallelized over chunks. Each
        // chunk still runs through the existing, unmodified SimdTrait::apply
        // (itself vectorized/ISA-dispatched) -- this parallelizes over
        // *chunks* rather than touching the ~10 leaf SIMD functions
        // individually, avoiding any risk of nested parallel regions inside
        // dispatch::g_dispatch's function-pointer chain.
        size_t n = static_cast<size_t>(a.numel());

        TENZOR_DISPATCH_ALL_TYPES_AND_COMPLEX(a.dtype(), "binary_pointwise", [&] {
            const scalar_t* a_data = a.data<scalar_t>();
            const scalar_t* b_data = b.data<scalar_t>();
            scalar_t* c_data = result.data<scalar_t>();
            detail::pointwise_apply_parallel<Op, scalar_t>(a_data, b_data, c_data, n);
        });
    } else {
        // Broadcasting path
        TENZOR_DISPATCH_ALL_TYPES_AND_COMPLEX(a.dtype(), "binary_pointwise_broadcast", [&] {
            const scalar_t* a_data = a.data<scalar_t>();
            const scalar_t* b_data = b.data<scalar_t>();
            scalar_t* c_data = result.data<scalar_t>();
            detail::broadcast_op(a_data, b_data, c_data,
                                 shape_a_vec, shape_b_vec, output_shape,
                                 [](scalar_t x, scalar_t y) {
                                     return Op::template scalar<scalar_t>(x, y);
                                 });
        });
    }

    return result;
}

} // namespace tenzor::cpu
