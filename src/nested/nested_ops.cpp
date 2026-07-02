/**
 * @file nested_ops.cpp
 * @brief NestedTensor operation implementations
 *
 * Element-wise ops work on the contiguous values buffer directly.
 * Offset-aware ops use the dispatch system for backend-specific kernels
 * or fall back to per-segment processing.
 */

#include "tenzor/nested/nested_ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tenzor {

// =========================================================================
// Validation Helpers
// =========================================================================

namespace {

auto verify_same_structure(const NestedTensor& a, const NestedTensor& b) -> void {
    if (a.batch_size() != b.batch_size()) {
        throw std::runtime_error(
            "nested op: batch size mismatch (" + std::to_string(a.batch_size()) +
            " vs " + std::to_string(b.batch_size()) + ")");
    }
    if (a.values().numel() != b.values().numel()) {
        throw std::runtime_error(
            "nested op: values numel mismatch (" +
            std::to_string(a.values().numel()) + " vs " +
            std::to_string(b.values().numel()) + ")");
    }
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("nested op: dtype mismatch");
    }
    if (a.device() != b.device()) {
        throw std::runtime_error("nested op: device mismatch");
    }
    // Compare the full offsets tensors element-wise. Equal batch_size and equal
    // total values numel are NOT sufficient: two NestedTensors can share both
    // yet still partition their packed rows into different per-element lengths
    // (e.g. a.offsets=[0,2,5], b.offsets=[0,3,5]). An element-wise op would then
    // run positionally on mismatched jagged structures and silently tag the
    // result with a's offsets. Copy to host (offsets are small, B+1) and verify.
    const auto& a_off = a.offsets();
    const auto& b_off = b.offsets();
    if (a_off.numel() != b_off.numel()) {
        throw std::runtime_error("nested op: offsets length mismatch (" +
            std::to_string(a_off.numel()) + " vs " +
            std::to_string(b_off.numel()) + ")");
    }
    auto a_off_cpu = (a_off.device().type != Device::Type::CPU)
        ? a_off.to(Device::cpu()) : a_off;
    auto b_off_cpu = (b_off.device().type != Device::Type::CPU)
        ? b_off.to(Device::cpu()) : b_off;
    const auto* ap = a_off_cpu.data<int64_t>();
    const auto* bp = b_off_cpu.data<int64_t>();
    for (int64_t i = 0; i < a_off_cpu.numel(); ++i) {
        if (ap[i] != bp[i]) {
            throw std::runtime_error(
                "nested op: offsets mismatch (different jagged structure)");
        }
    }
}

/// Verify two NestedTensors share the same ragged structure (batch size, the
/// full offsets tensor, and device) WITHOUT requiring equal values numel. Used
/// where only the per-element sequence partitioning must agree but the trailing
/// regular feature dimension may differ — e.g. attention's value projection,
/// whose feature dim Dv need not equal the query/key feature dim Dq.
auto verify_same_offsets(const NestedTensor& a, const NestedTensor& b) -> void {
    if (a.batch_size() != b.batch_size()) {
        throw std::runtime_error(
            "nested op: batch size mismatch (" + std::to_string(a.batch_size()) +
            " vs " + std::to_string(b.batch_size()) + ")");
    }
    if (a.device() != b.device()) {
        throw std::runtime_error("nested op: device mismatch");
    }
    const auto& a_off = a.offsets();
    const auto& b_off = b.offsets();
    if (a_off.numel() != b_off.numel()) {
        throw std::runtime_error("nested op: offsets length mismatch (" +
            std::to_string(a_off.numel()) + " vs " +
            std::to_string(b_off.numel()) + ")");
    }
    auto a_off_cpu = (a_off.device().type != Device::Type::CPU)
        ? a_off.to(Device::cpu()) : a_off;
    auto b_off_cpu = (b_off.device().type != Device::Type::CPU)
        ? b_off.to(Device::cpu()) : b_off;
    const auto* ap = a_off_cpu.data<int64_t>();
    const auto* bp = b_off_cpu.data<int64_t>();
    for (int64_t i = 0; i < a_off_cpu.numel(); ++i) {
        if (ap[i] != bp[i]) {
            throw std::runtime_error(
                "nested op: offsets mismatch (different jagged structure)");
        }
    }
}

/// Build the contiguous offsets [0, 1, ..., B] for a NestedTensor whose every
/// batch element holds exactly one row (the result of a ragged-dim reduction:
/// each segment collapses to a single row, so values has B rows total).
/// Offsets are constructed on the HOST — data<int64_t>() returns a device
/// pointer on GPU backends, so host writes there are UB — then moved to the
/// target device before from_jagged.
auto make_unit_offsets(int64_t B, Device device) -> Tensor {
    auto offsets = tenzor::zeros({B + 1}, DType::Int64, Device::cpu());
    auto* ptr = offsets.data<int64_t>();
    for (int64_t i = 0; i <= B; ++i) {
        ptr[i] = i;
    }
    if (device.type != Device::Type::CPU) {
        offsets = offsets.to(device);
    }
    return offsets;
}

/// Helper for segmented per-element operation with a unary function on segments.
auto segmented_unary(const NestedTensor& input,
                     auto&& segment_fn) -> NestedTensor {
    auto offsets_cpu = (input.offsets().device().type != Device::Type::CPU)
        ? input.offsets().to(Device::cpu()) : input.offsets();
    const auto* off_ptr = offsets_cpu.data<int64_t>();
    int64_t B = input.batch_size();

    std::vector<Tensor> results;
    results.reserve(B);
    for (int64_t i = 0; i < B; ++i) {
        auto segment = input.values().slice(0, off_ptr[i], off_ptr[i + 1]);
        results.push_back(segment_fn(segment));
    }

    if (results.empty()) {
        return NestedTensor::from_jagged(
            tenzor::zeros({0}, input.dtype(), input.device()),
            input.offsets(), input.ragged_dim());
    }

    return NestedTensor::from_jagged(
        tenzor::cat(results, 0), input.offsets(), input.ragged_dim());
}

} // anonymous namespace

// =========================================================================
// Element-wise Binary Operations
// =========================================================================

auto nested_add(const NestedTensor& a, const NestedTensor& b) -> NestedTensor {
    verify_same_structure(a, b);
    return NestedTensor::from_jagged(
        a.values() + b.values(), a.offsets(), a.ragged_dim());
}

auto nested_sub(const NestedTensor& a, const NestedTensor& b) -> NestedTensor {
    verify_same_structure(a, b);
    return NestedTensor::from_jagged(
        a.values() - b.values(), a.offsets(), a.ragged_dim());
}

auto nested_mul(const NestedTensor& a, const NestedTensor& b) -> NestedTensor {
    verify_same_structure(a, b);
    return NestedTensor::from_jagged(
        a.values() * b.values(), a.offsets(), a.ragged_dim());
}

auto nested_div(const NestedTensor& a, const NestedTensor& b) -> NestedTensor {
    verify_same_structure(a, b);
    return NestedTensor::from_jagged(
        a.values() / b.values(), a.offsets(), a.ragged_dim());
}

// =========================================================================
// Element-wise Unary Operations
// =========================================================================

auto nested_neg(const NestedTensor& a) -> NestedTensor {
    return NestedTensor::from_jagged(
        tenzor::neg(a.values()), a.offsets(), a.ragged_dim());
}

auto nested_relu(const NestedTensor& a) -> NestedTensor {
    std::vector<Tensor> inputs = {a.values()};
    return NestedTensor::from_jagged(
        dispatch_single(OpId::ReLU, inputs),
        a.offsets(), a.ragged_dim());
}

auto nested_gelu(const NestedTensor& a) -> NestedTensor {
    std::vector<Tensor> inputs = {a.values()};
    return NestedTensor::from_jagged(
        dispatch_single(OpId::Gelu, inputs),
        a.offsets(), a.ragged_dim());
}

auto nested_sigmoid(const NestedTensor& a) -> NestedTensor {
    return NestedTensor::from_jagged(
        tenzor::sigmoid(a.values()), a.offsets(), a.ragged_dim());
}

auto nested_tanh(const NestedTensor& a) -> NestedTensor {
    return NestedTensor::from_jagged(
        tenzor::tanh(a.values()), a.offsets(), a.ragged_dim());
}

auto nested_abs(const NestedTensor& a) -> NestedTensor {
    return NestedTensor::from_jagged(
        tenzor::abs(a.values()), a.offsets(), a.ragged_dim());
}

// =========================================================================
// Scalar Operations
// =========================================================================

auto nested_add_scalar(const NestedTensor& a, double scalar) -> NestedTensor {
    return NestedTensor::from_jagged(
        tenzor::add(a.values(), scalar), a.offsets(), a.ragged_dim());
}

auto nested_mul_scalar(const NestedTensor& a, double scalar) -> NestedTensor {
    return NestedTensor::from_jagged(
        tenzor::mul(a.values(), scalar), a.offsets(), a.ragged_dim());
}

// =========================================================================
// Offset-aware Operations
// =========================================================================

auto nested_softmax(const NestedTensor& input, int64_t dim) -> NestedTensor {
    if (dim < 0) dim += input.ndim();
    if (dim != input.ragged_dim()) {
        // Softmax along a regular dim — operate on values directly. The values
        // tensor has the leading batch dim (0) collapsed into the ragged axis,
        // so a regular nested dim maps to values_dim = dim - 1. dim==0 here
        // means the batch dim (and it is not the ragged dim), which is not a
        // valid per-element softmax axis — reject it instead of silently
        // computing values_dim = -1 (the last feature axis).
        if (dim == 0) {
            throw std::runtime_error(
                "nested_softmax: softmax over the batch dim (dim=0) is not supported");
        }
        int64_t values_dim = dim - 1;
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, values_dim);
        std::vector<Tensor> inputs = {input.values()};
        auto result = dispatch_single(OpId::Softmax, inputs, attrs);
        return NestedTensor::from_jagged(result, input.offsets(),
                                         input.ragged_dim());
    }

    // Softmax along ragged dim — dispatch to native NestedSoftmax kernel.
    // B.4: every shipped backend (CPU/CUDA/ROCm/OneAPI/Vulkan) registers
    // OpId::NestedSoftmax, so the previous try/catch around dispatch was a
    // workaround for partial coverage that no longer applies. The catch is
    // removed so a real kernel failure surfaces to the caller instead of
    // being masked by the silently-slower segmented fallback.
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    std::vector<Tensor> inputs = {input.values(), input.offsets()};
    auto result = dispatch<OpId::NestedSoftmax>(inputs, attrs);
    return NestedTensor::from_jagged(result[0], input.offsets(),
                                     input.ragged_dim());
}

auto nested_log_softmax(const NestedTensor& input, int64_t dim) -> NestedTensor {
    if (dim < 0) dim += input.ndim();
    if (dim != input.ragged_dim()) {
        // See nested_softmax: dim==0 in the regular-dim path is the batch dim
        // (not the ragged dim) and is not a valid per-element axis; reject it
        // rather than silently computing values_dim = -1.
        if (dim == 0) {
            throw std::runtime_error(
                "nested_log_softmax: log_softmax over the batch dim (dim=0) is not supported");
        }
        int64_t values_dim = dim - 1;
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, values_dim);
        std::vector<Tensor> inputs = {input.values()};
        auto result = dispatch_single(OpId::LogSoftmax, inputs, attrs);
        return NestedTensor::from_jagged(result, input.offsets(),
                                         input.ragged_dim());
    }

    // B.4: native NestedLogSoftmax kernel is registered in every shipped
    // backend; the previous try/catch fallback is removed so real failures
    // surface.
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    std::vector<Tensor> inputs = {input.values(), input.offsets()};
    auto result = dispatch<OpId::NestedLogSoftmax>(inputs, attrs);
    return NestedTensor::from_jagged(result[0], input.offsets(),
                                     input.ragged_dim());
}

auto nested_layer_norm(const NestedTensor& input, const Tensor& weight,
                       const Tensor& bias, double eps) -> NestedTensor {
    // B.4: native NestedLayerNorm kernel is registered in every shipped
    // backend; the previous try/catch + segmented fallback is removed so
    // real kernel failures surface to the caller.
    OpAttributes attrs;
    attrs.set(AttrKey::Eps, eps);
    std::vector<Tensor> inputs = {input.values(), input.offsets(), weight, bias};
    auto result = dispatch<OpId::NestedLayerNorm>(inputs, attrs);
    return NestedTensor::from_jagged(result[0], input.offsets(),
                                     input.ragged_dim());
}

auto nested_sum(const NestedTensor& input, int64_t dim,
                bool keepdim) -> NestedTensor {
    if (dim < 0) dim += input.ndim();
    if (dim == 0) {
        // Layout is [batch(0), ragged(1), regular(2..)]; the batch axis is
        // interleaved into values dim 0 and cannot be reduced by the regular-dim
        // mapping (which would compute values_dim = -1 and silently reduce the
        // last regular dim instead). Reject rather than return a wrong result.
        throw std::runtime_error(
            "nested_sum: reduction over the batch dim (dim=0) is not supported");
    }
    if (dim != input.ragged_dim()) {
        // values_dim maps an outer dim onto the packed values tensor; only valid
        // for dims beyond the ragged axis (dim > ragged_dim, i.e. dim >= 2).
        int64_t values_dim = dim - 1;
        auto result = tenzor::sum(input.values(), values_dim, keepdim);
        return NestedTensor::from_jagged(result, input.offsets(),
                                         input.ragged_dim());
    }

    // Sum along ragged dim — dispatch to the native NestedSum kernel rather than
    // looping B small reductions on the host (the kernel is the same one the
    // autograd forward path uses, see nested_autograd_ops). The kernel returns
    // one row per batch element ([B, *regular]); each batch element then holds a
    // single row, so offsets are the unit sequence [0, 1, ..., B].
    int64_t B = input.batch_size();

    if (B == 0) {
        std::vector<int64_t> shape = {0};
        shape.insert(shape.end(), input.regular_shape().begin(),
                     input.regular_shape().end());
        return NestedTensor::from_jagged(
            tenzor::zeros(shape, input.dtype(), input.device()),
            input.offsets(), input.ragged_dim());
    }

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input.values(), input.offsets()};
    auto reduced = dispatch<OpId::NestedSum>(inputs, attrs)[0];  // dense [B, *regular]
    // Reducing over the ragged dim collapses each variable-length segment to a
    // single [*regular] row, so the result is a DENSE [B, *regular] tensor with
    // no remaining ragged axis. `keepdim` selects whether the collapsed ragged
    // dim is retained as a size-1 axis:
    //   keepdim=false -> drop the ragged axis entirely: dense values [B, *regular].
    //   keepdim=true  -> retain it as size 1:            values [B, 1, *regular].
    // Previously BOTH branches wrapped [B, *regular] in an identical unit-offset
    // NestedTensor, so keepdim was silently ignored. The dense result is
    // surfaced as the returned NestedTensor's values(); its per-batch unit
    // offsets encode the collapsed (one-row-per-element) structure.
    // NOTE: the return type is fixed to NestedTensor by out-of-scope callers
    // (python bindings and tests using .values()/.batch_size()), so a bare
    // dense Tensor cannot be returned here; keepdim is honored in values shape.
    if (keepdim) {
        reduced = tenzor::unsqueeze(reduced, /*dim=*/1);  // [B, 1, *regular]
    }
    auto new_offsets = make_unit_offsets(B, input.device());
    return NestedTensor::from_jagged(reduced, new_offsets,
                                     input.ragged_dim());
}

auto nested_mean(const NestedTensor& input, int64_t dim,
                 bool keepdim) -> NestedTensor {
    if (dim < 0) dim += input.ndim();
    if (dim == 0) {
        // See nested_sum: the batch axis cannot be reduced via the regular-dim
        // mapping; reject instead of silently reducing the last regular dim.
        throw std::runtime_error(
            "nested_mean: reduction over the batch dim (dim=0) is not supported");
    }
    if (dim != input.ragged_dim()) {
        int64_t values_dim = dim - 1;
        auto result = tenzor::mean(input.values(), values_dim, keepdim);
        return NestedTensor::from_jagged(result, input.offsets(),
                                         input.ragged_dim());
    }

    // Mean along ragged dim — dispatch to the native NestedMean kernel (same
    // kernel used by the autograd forward path) instead of B host-side slices +
    // reductions. The kernel returns [B, *regular]; each batch element holds one
    // row, so offsets are the unit sequence [0, 1, ..., B].
    int64_t B = input.batch_size();

    if (B == 0) {
        std::vector<int64_t> shape = {0};
        shape.insert(shape.end(), input.regular_shape().begin(),
                     input.regular_shape().end());
        return NestedTensor::from_jagged(
            tenzor::zeros(shape, input.dtype(), input.device()),
            input.offsets(), input.ragged_dim());
    }

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input.values(), input.offsets()};
    auto reduced = dispatch<OpId::NestedMean>(inputs, attrs)[0];  // dense [B, *regular]
    // See nested_sum: reducing over the ragged dim yields a DENSE [B, *regular]
    // tensor (no ragged axis remains). keepdim selects whether the collapsed
    // ragged dim is retained as a size-1 axis:
    //   keepdim=false -> drop it:           dense values [B, *regular].
    //   keepdim=true  -> retain as size 1:  values [B, 1, *regular].
    // Previously both branches produced an identical unit-offset NestedTensor,
    // so keepdim was silently ignored; it is now honored in the values shape.
    // (Return type is fixed to NestedTensor by out-of-scope callers, so the
    // dense result is surfaced via values() rather than as a bare Tensor.)
    if (keepdim) {
        reduced = tenzor::unsqueeze(reduced, /*dim=*/1);  // [B, 1, *regular]
    }
    auto new_offsets = make_unit_offsets(B, input.device());
    return NestedTensor::from_jagged(reduced, new_offsets,
                                     input.ragged_dim());
}

// =========================================================================
// Compound Operations
// =========================================================================

auto nested_linear(const NestedTensor& input, const Tensor& weight,
                   const Tensor* bias) -> NestedTensor {
    auto result_values = tenzor::matmul(
        input.values(), tenzor::transpose(weight, -2, -1));
    if (bias) {
        result_values = result_values + *bias;
    }
    return NestedTensor::from_jagged(result_values, input.offsets(),
                                     input.ragged_dim());
}

auto nested_matmul(const NestedTensor& a, const Tensor& b) -> NestedTensor {
    auto result_values = tenzor::matmul(a.values(), b);
    return NestedTensor::from_jagged(result_values, a.offsets(),
                                     a.ragged_dim());
}

auto nested_attention(const NestedTensor& query, const NestedTensor& key,
                      const NestedTensor& value, double scale,
                      bool causal) -> NestedTensor {
    // Key must match query in full (feature dim Dk == Dq is required by the
    // Q @ K^T score matmul). Value only needs the SAME ragged structure as the
    // query/key: the attn_weights @ V matmul accepts a distinct value feature
    // dim Dv, so requiring equal values numel here would wrongly reject valid
    // value projections to a different dimension.
    verify_same_structure(query, key);
    verify_same_offsets(query, value);

    auto offsets_cpu = (query.offsets().device().type != Device::Type::CPU)
        ? query.offsets().to(Device::cpu()) : query.offsets();
    const auto* off_ptr = offsets_cpu.data<int64_t>();
    int64_t B = query.batch_size();

    double actual_scale = scale;
    if (actual_scale < 0.0) {
        int64_t head_dim = query.values().shape().back();
        actual_scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    }

    // Fast path: when the value feature dim Dv equals the query/key head dim,
    // dispatch to the native NestedAttention kernel (same kernel the autograd
    // forward path uses) instead of issuing B host-side slice+matmul+softmax
    // ops. The kernel assumes Dv == Dq == Dk, so a distinct value feature dim
    // (permitted by verify_same_offsets) falls through to the segmented path
    // below, which handles arbitrary Dv via the explicit attn_weights @ V matmul.
    if (B > 0 &&
        value.values().shape().back() == query.values().shape().back()) {
        OpAttributes attrs;
        attrs.set(AttrKey::Scale, actual_scale);
        attrs.set(AttrKey::Causal, causal);
        std::vector<Tensor> inputs = {query.values(), key.values(),
                                      value.values(), query.offsets(),
                                      query.offsets()};
        auto result = dispatch<OpId::NestedAttention>(inputs, attrs)[0];
        return NestedTensor::from_jagged(result, query.offsets(),
                                         query.ragged_dim());
    }

    std::vector<Tensor> results;
    results.reserve(B);

    for (int64_t i = 0; i < B; ++i) {
        auto q = query.values().slice(0, off_ptr[i], off_ptr[i + 1]);
        auto k = key.values().slice(0, off_ptr[i], off_ptr[i + 1]);
        auto v = value.values().slice(0, off_ptr[i], off_ptr[i + 1]);

        // scores = Q @ K^T * scale
        auto scores = tenzor::matmul(q, tenzor::transpose(k, -2, -1));
        scores = tenzor::mul(scores, actual_scale);

        if (causal) {
            int64_t seq_len = off_ptr[i + 1] - off_ptr[i];
            auto mask = tenzor::ones({seq_len, seq_len}, scores.dtype(),
                                     scores.device());
            // Create lower-triangular mask via dispatch
            std::vector<Tensor> tril_inputs = {mask};
            OpAttributes tril_attrs;
            auto tril_mask = dispatch_single(OpId::Tril, tril_inputs, tril_attrs);
            // Mask out the upper triangle by *adding* a large negative bias to
            // disallowed positions. Two correctness requirements:
            //  1. Use a dtype-aware FINITE large-negative value. -1e9 overflows
            //     the Float16 range and the float->half conversion yields -inf,
            //     not a saturated finite value.
            //  2. Apply the bias additively (scores + (1 - tril) * neg) rather
            //     than multiplicatively (scores * tril + neg * (1 - tril)). The
            //     multiplicative form computes neg * 0 at KEPT positions, which
            //     is -inf * 0 = NaN for fp16 and poisons the whole softmax row.
            double neg_bias = -1e9;
            if (scores.dtype() == DType::Float16) {
                neg_bias = -3e4;  // safely within fp16 finite range (~-65504)
            } else if (scores.dtype() == DType::BFloat16) {
                neg_bias = -1e9;  // bf16 has fp32-like exponent range; finite
            }
            auto ones_sq = tenzor::ones({seq_len, seq_len},
                                         scores.dtype(), scores.device());
            auto neg_inf_tensor = tenzor::full({seq_len, seq_len}, neg_bias,
                                                scores.dtype(), scores.device());
            scores = scores + neg_inf_tensor * (ones_sq - tril_mask);
        }

        // attn_weights = softmax(scores, dim=-1)
        OpAttributes sm_attrs;
        sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
        std::vector<Tensor> sm_inputs = {scores};
        auto attn_weights = dispatch_single(OpId::Softmax, sm_inputs, sm_attrs);

        // output = attn_weights @ V
        results.push_back(tenzor::matmul(attn_weights, v));
    }

    if (results.empty()) {
        return NestedTensor::from_jagged(
            tenzor::zeros({0}, query.dtype(), query.device()),
            query.offsets(), query.ragged_dim());
    }

    return NestedTensor::from_jagged(
        tenzor::cat(results, 0), query.offsets(), query.ragged_dim());
}

// =========================================================================
// Manipulation
// =========================================================================

auto nested_cat(std::span<const NestedTensor> tensors,
                int64_t dim) -> NestedTensor {
    if (tensors.empty()) {
        throw std::runtime_error("nested_cat: empty input");
    }
    if (dim != 0) {
        throw std::runtime_error(
            "nested_cat: only batch dimension (dim=0) concatenation supported");
    }

    auto ref_dtype = tensors[0].dtype();
    auto ref_device = tensors[0].device();
    auto ref_ragged_dim = tensors[0].ragged_dim();

    std::vector<Tensor> all_values;
    int64_t total_batch = 0;

    for (const auto& nt : tensors) {
        if (nt.dtype() != ref_dtype || nt.device() != ref_device) {
            throw std::runtime_error("nested_cat: dtype/device mismatch");
        }
        all_values.push_back(nt.values());
        total_batch += nt.batch_size();
    }

    auto cat_values = tenzor::cat(all_values, 0);

    // Build offsets on the HOST: data<int64_t>() returns a device pointer when
    // ref_device is a GPU, so the host writes below would be UB (segfault or a
    // silent no-op leaving offsets all-zero). Mirror nested_sum/nested_mean:
    // fill on CPU, then move to ref_device before from_jagged.
    auto new_offsets = tenzor::zeros({total_batch + 1}, DType::Int64, Device::cpu());
    auto* noff_ptr = new_offsets.data<int64_t>();
    int64_t idx = 0;
    noff_ptr[0] = 0;

    for (const auto& nt : tensors) {
        auto nt_offsets_cpu = (nt.offsets().device().type != Device::Type::CPU)
            ? nt.offsets().to(Device::cpu()) : nt.offsets();
        const auto* nt_off_ptr = nt_offsets_cpu.data<int64_t>();

        for (int64_t i = 0; i < nt.batch_size(); ++i) {
            int64_t len = nt_off_ptr[i + 1] - nt_off_ptr[i];
            noff_ptr[idx + 1] = noff_ptr[idx] + len;
            ++idx;
        }
    }

    if (ref_device.type != Device::Type::CPU) {
        new_offsets = new_offsets.to(ref_device);
    }
    return NestedTensor::from_jagged(cat_values, new_offsets, ref_ragged_dim);
}

auto nested_dropout(const NestedTensor& input, double p,
                    bool training) -> NestedTensor {
    if (!training || p == 0.0) {
        return input;
    }

    // Apply dropout to the entire values buffer
    OpAttributes attrs;
    // The Dropout kernel reads AttrKey::P (not DropoutP); using the wrong key
    // silently dropped the requested probability and defaulted to ~0.5.
    attrs.set(AttrKey::P, p);
    attrs.set(AttrKey::Training, training);
    std::vector<Tensor> inputs = {input.values()};
    auto dropped = dispatch_single(OpId::Dropout, inputs, attrs);
    return NestedTensor::from_jagged(dropped, input.offsets(),
                                     input.ragged_dim());
}

} // namespace tenzor
