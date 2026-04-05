/**
 * @file lite_ops.hpp
 * @brief Operator registry for the lite runtime
 *
 * Each platform (ARM NEON, x86 SSE/AVX, reference C++) registers its
 * kernel implementations into the global LiteOpsRegistry. The graph
 * executor looks up kernels by LiteOpType at execution time.
 */

#pragma once

#include "lite_graph.hpp"

namespace tenzor::lite {

// Kernel function signature: takes an array of input tensors, their count,
// and the node's attribute block; returns one output tensor.
using LiteKernelFn = LiteTensor (*)(const LiteTensor* inputs, int num_inputs,
                                    const LiteAttributes& attrs);

class LiteOpsRegistry {
public:
    /** Get the process-wide singleton registry. */
    static auto instance() -> LiteOpsRegistry&;

    /** Register a kernel for the given op type. */
    auto register_op(LiteOpType op, LiteKernelFn fn) -> void;

    /** Look up the kernel for the given op type. Returns nullptr if unregistered. */
    auto get_op(LiteOpType op) const -> LiteKernelFn;

private:
    LiteKernelFn table_[64]{};
};

}  // namespace tenzor::lite
