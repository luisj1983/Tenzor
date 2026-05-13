/**
 * @file dispatch_bridge.cpp
 * @brief Phase 1 implementation of the LiteAttributes -> OpAttributes bridge.
 *
 * The positional attribute encoding per supported OpId is documented inline
 * below. Each new op added to the Lite-supported set extends `build_attrs`.
 */

#include "dispatch_bridge.hpp"

#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"

namespace tenzor::lite {

namespace {

// Translate the positional LiteAttributes into the typed OpAttributes map for
// a given OpId. Unsupported (or attribute-less) ops fall through to an empty
// map; the dispatched kernel will surface any missing-attribute errors itself.
auto build_attrs(LiteOpType op, const LiteAttributes& la) -> OpAttributes {
    OpAttributes oa;
    switch (op) {
        // Element-wise binary / unary ops have no attributes in Phase 1.
        case OpId::Add:
        case OpId::Sub:
        case OpId::Mul:
        case OpId::Div:
        case OpId::MatMul:
        case OpId::ReLU:
        case OpId::Sigmoid:
        case OpId::Tanh:
            break;

        // Softmax: dim = i[0].
        case OpId::Softmax:
            oa.set(AttrKey::Dim, la.i[0]);
            break;

        default:
            // No mapping yet — pass attrs through as empty. Kernels that
            // *require* an attribute will throw with a clear message. Later
            // phases extend this switch as more ops join the Lite set.
            break;
    }
    return oa;
}

}  // namespace

auto run_op(LiteOpType op,
            std::span<const Tensor> inputs,
            const LiteAttributes& attrs) -> std::vector<Tensor> {
    return ::tenzor::dispatch(op, inputs, build_attrs(op, attrs));
}

}  // namespace tenzor::lite
