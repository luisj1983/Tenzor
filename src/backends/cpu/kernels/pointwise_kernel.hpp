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
#include "broadcast.hpp"
#include "fp8_emulation.hpp"
#include <cstddef>
#include <complex>

#include "simd_traits.hpp"


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
    const Tensor& a, const Tensor& b,
    Tensor (*fp8_emulate)(const Tensor&, const Tensor&) = nullptr) -> Tensor
{
    // Validation is done by the caller (validate_elementwise in math.cpp)

    // FP8 emulation
    if (fp8_emulate && is_fp8(a.dtype()) && is_fp8(b.dtype())) {
        return fp8_emulate(a, b);
    }

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    bool same_shape = (shape_a_vec == shape_b_vec);
    if (same_shape) {
        // Contiguous fast path with SIMD
        size_t n = static_cast<size_t>(a.numel());

        TENZOR_DISPATCH_ALL_TYPES_AND_COMPLEX(a.dtype(), "binary_pointwise", [&] {
            const scalar_t* a_data = a.data<scalar_t>();
            const scalar_t* b_data = b.data<scalar_t>();
            scalar_t* c_data = result.data<scalar_t>();
            SimdTrait<Op, scalar_t>::apply(a_data, b_data, c_data, n);
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
