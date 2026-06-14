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
    if (dim != input.ragged_dim()) {
        // Softmax along a regular dim — operate on values directly.
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
    if (dim != input.ragged_dim()) {
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
    if (dim != input.ragged_dim()) {
        int64_t values_dim = dim - 1;
        auto result = tenzor::sum(input.values(), values_dim, keepdim);
        return NestedTensor::from_jagged(result, input.offsets(),
                                         input.ragged_dim());
    }

    // Sum along ragged dim — per segment
    auto offsets_cpu = (input.offsets().device().type != Device::Type::CPU)
        ? input.offsets().to(Device::cpu()) : input.offsets();
    const auto* off_ptr = offsets_cpu.data<int64_t>();
    int64_t B = input.batch_size();

    std::vector<Tensor> results;
    results.reserve(B);
    for (int64_t i = 0; i < B; ++i) {
        auto seg = input.values().slice(0, off_ptr[i], off_ptr[i + 1]);
        results.push_back(tenzor::sum(seg, 0, keepdim));
    }

    if (results.empty()) {
        std::vector<int64_t> shape = {0};
        shape.insert(shape.end(), input.regular_shape().begin(),
                     input.regular_shape().end());
        return NestedTensor::from_jagged(
            tenzor::zeros(shape, input.dtype(), input.device()),
            input.offsets(), input.ragged_dim());
    }

    auto new_values = tenzor::cat(results, 0);
    // Build offsets on CPU (host writes) then move to the input's device.
    // Writing through .data<int64_t>() on a device-resident tensor is a
    // host-side dereference of GPU memory and segfaults on Vulkan/CUDA/etc.
    auto new_offsets = tenzor::zeros({B + 1}, DType::Int64, Device::cpu());
    auto* noff_ptr = new_offsets.data<int64_t>();
    for (int64_t i = 0; i <= B; ++i) {
        noff_ptr[i] = i;
    }
    if (input.device().type != Device::Type::CPU) {
        new_offsets = new_offsets.to(input.device());
    }
    return NestedTensor::from_jagged(new_values, new_offsets,
                                     input.ragged_dim());
}

auto nested_mean(const NestedTensor& input, int64_t dim,
                 bool keepdim) -> NestedTensor {
    if (dim != input.ragged_dim()) {
        int64_t values_dim = dim - 1;
        auto result = tenzor::mean(input.values(), values_dim, keepdim);
        return NestedTensor::from_jagged(result, input.offsets(),
                                         input.ragged_dim());
    }

    auto offsets_cpu = (input.offsets().device().type != Device::Type::CPU)
        ? input.offsets().to(Device::cpu()) : input.offsets();
    const auto* off_ptr = offsets_cpu.data<int64_t>();
    int64_t B = input.batch_size();

    std::vector<Tensor> results;
    results.reserve(B);
    for (int64_t i = 0; i < B; ++i) {
        auto seg = input.values().slice(0, off_ptr[i], off_ptr[i + 1]);
        results.push_back(tenzor::mean(seg, 0, keepdim));
    }

    if (results.empty()) {
        std::vector<int64_t> shape = {0};
        shape.insert(shape.end(), input.regular_shape().begin(),
                     input.regular_shape().end());
        return NestedTensor::from_jagged(
            tenzor::zeros(shape, input.dtype(), input.device()),
            input.offsets(), input.ragged_dim());
    }

    auto new_values = tenzor::cat(results, 0);
    // Build offsets on CPU (host writes) then move to the input's device.
    // Writing through .data<int64_t>() on a device-resident tensor is a
    // host-side dereference of GPU memory and segfaults on Vulkan/CUDA/etc.
    auto new_offsets = tenzor::zeros({B + 1}, DType::Int64, Device::cpu());
    auto* noff_ptr = new_offsets.data<int64_t>();
    for (int64_t i = 0; i <= B; ++i) {
        noff_ptr[i] = i;
    }
    if (input.device().type != Device::Type::CPU) {
        new_offsets = new_offsets.to(input.device());
    }
    return NestedTensor::from_jagged(new_values, new_offsets,
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
    verify_same_structure(query, key);
    verify_same_structure(query, value);

    auto offsets_cpu = (query.offsets().device().type != Device::Type::CPU)
        ? query.offsets().to(Device::cpu()) : query.offsets();
    const auto* off_ptr = offsets_cpu.data<int64_t>();
    int64_t B = query.batch_size();

    double actual_scale = scale;
    if (actual_scale < 0.0) {
        int64_t head_dim = query.values().shape().back();
        actual_scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
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
            // scores = where(tril_mask > 0, scores, -inf)
            auto neg_inf_tensor = tenzor::full({seq_len, seq_len}, -1e9,
                                                scores.dtype(), scores.device());
            auto ones_sq = tenzor::ones({seq_len, seq_len},
                                         scores.dtype(), scores.device());
            scores = scores * tril_mask + neg_inf_tensor * (ones_sq - tril_mask);
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
    attrs.set(AttrKey::DropoutP, p);
    attrs.set(AttrKey::Training, training);
    std::vector<Tensor> inputs = {input.values()};
    auto dropped = dispatch_single(OpId::Dropout, inputs, attrs);
    return NestedTensor::from_jagged(dropped, input.offsets(),
                                     input.ragged_dim());
}

} // namespace tenzor
