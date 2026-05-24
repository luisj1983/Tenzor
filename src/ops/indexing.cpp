#include "tenzor/ops/indexing.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/shape.hpp"
#include <cstdint>
#include <optional>
#include <variant>

namespace tenzor {

auto slice(const Tensor& input, int64_t dim, int64_t start,
          int64_t end, int64_t step) -> Tensor {
    return input.slice(dim, start, end, step);
}

auto index_select(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    std::vector<Tensor> inputs = {input, index};
    return dispatch(OpId::IndexSelect, inputs, attrs)[0];
}

auto gather(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    std::vector<Tensor> inputs = {input, index};
    return dispatch(OpId::Gather, inputs, attrs)[0];
}

auto scatter(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    std::vector<Tensor> inputs = {input, index, src};
    return dispatch(OpId::Scatter, inputs, attrs)[0];
}

auto scatter_add(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    std::vector<Tensor> inputs = {input, index, src};
    return dispatch(OpId::ScatterAdd, inputs, attrs)[0];
}

auto scatter_reduce(const Tensor& input, int64_t dim, const Tensor& index,
                    const Tensor& src, const std::string& reduce,
                    bool include_self) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Reduction, reduce);
    attrs.set(AttrKey::IncludeSelf, include_self);
    std::vector<Tensor> inputs = {input.contiguous(), index.contiguous(), src.contiguous()};
    return dispatch(OpId::ScatterReduce, inputs, attrs)[0];
}

auto masked_select(const Tensor& input, const Tensor& mask) -> Tensor {
    std::vector<Tensor> inputs = {input, mask};
    return dispatch(OpId::MaskedSelect, inputs)[0];
}

auto masked_fill(const Tensor& input, const Tensor& mask, double value) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Value, value);
    std::vector<Tensor> inputs = {input, mask};
    return dispatch(OpId::MaskedFill, inputs, attrs)[0];
}

// Deprecated `float` overload (audit-5 Y.3): forward to the `double` overload
// so the user-provided scalar travels at double precision into `AttrKey::Value`.
auto masked_fill(const Tensor& input, const Tensor& mask, float value) -> Tensor {
    return masked_fill(input, mask, static_cast<double>(value));
}

auto where(const Tensor& condition, const Tensor& x, const Tensor& y) -> Tensor {
    // Backend kernels historically require all three tensors to have
    // identical shape (the CPU path throws outright, Vulkan rejects
    // dim-count mismatch). Broadcasting — e.g. condition[32,1] with
    // x,y[32,32] — therefore failed everywhere. Materialize the common
    // broadcast shape here so each backend still sees equal-shape
    // inputs but callers don't have to pre-broadcast manually.
    auto cs = condition.shape();
    auto xs = x.shape();
    auto ys = y.shape();
    bool same_shape =
        cs.size() == xs.size() && cs.size() == ys.size() &&
        std::equal(cs.begin(), cs.end(), xs.begin()) &&
        std::equal(cs.begin(), cs.end(), ys.begin());

    if (same_shape) {
        std::vector<Tensor> inputs = {condition, x, y};
        return dispatch(OpId::Where, inputs)[0];
    }

    // NumPy/PyTorch broadcasting rules, right-aligned: the common rank
    // is max(rank(cond), rank(x), rank(y)); each axis is the max of
    // the (1-padded) extents, and size-1 is broadcastable.
    int64_t ndim = static_cast<int64_t>(
        std::max({cs.size(), xs.size(), ys.size()}));
    auto extent_at = [&](auto s, int64_t axis) -> int64_t {
        int64_t offset = ndim - static_cast<int64_t>(s.size());
        return (axis < offset) ? int64_t{1} : s[axis - offset];
    };
    std::vector<int64_t> out_shape(ndim);
    for (int64_t a = 0; a < ndim; ++a) {
        int64_t ec = extent_at(cs, a);
        int64_t ex = extent_at(xs, a);
        int64_t ey = extent_at(ys, a);
        int64_t want = std::max({ec, ex, ey});
        auto ok = [&](int64_t e) { return e == want || e == 1; };
        if (!ok(ec) || !ok(ex) || !ok(ey)) {
            throw tenzor::ValueError(
                "where: shapes are not broadcastable");
        }
        out_shape[a] = want;
    }
    auto shape_eq = [&](auto s, const std::vector<int64_t>& target) {
        return s.size() == target.size() &&
               std::equal(s.begin(), s.end(), target.begin());
    };
    Tensor cond_b = shape_eq(cs, out_shape) ? condition : broadcast_to(condition, out_shape);
    Tensor x_b    = shape_eq(xs, out_shape) ? x         : broadcast_to(x, out_shape);
    Tensor y_b    = shape_eq(ys, out_shape) ? y         : broadcast_to(y, out_shape);
    // broadcast_to may return a non-contiguous view; backend kernels
    // like the CPU where generally assume contiguous strides.
    if (!cond_b.is_contiguous()) cond_b = contiguous(cond_b);
    if (!x_b.is_contiguous())    x_b    = contiguous(x_b);
    if (!y_b.is_contiguous())    y_b    = contiguous(y_b);
    std::vector<Tensor> inputs = {cond_b, x_b, y_b};
    return dispatch(OpId::Where, inputs)[0];
}

auto take(const Tensor& input, const Tensor& index) -> Tensor {
    std::vector<Tensor> inputs = {input, index};
    return dispatch(OpId::Take, inputs)[0];
}

auto put(const Tensor& input, const Tensor& index, const Tensor& source) -> Tensor {
    std::vector<Tensor> inputs = {input, index, source};
    return dispatch(OpId::Put, inputs)[0];
}

auto nonzero(const Tensor& input) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    // Use backend dispatch for non-CPU devices (avoids GPU→CPU→GPU round-trip)
    if (input.device().type != Device::Type::CPU) {
        OpAttributes attrs;
        // Pass shape info so the backend can compute multi-dimensional indices
        std::string shape_str;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) shape_str += ",";
            shape_str += std::to_string(shape[i]);
        }
        attrs.set(AttrKey::Shape, shape_str);
        attrs.set(AttrKey::Dim, ndim);

        std::vector<Tensor> inputs = {input};
        auto results = dispatch(OpId::Nonzero, inputs, attrs);
        return results[0];
    }

    // CPU fallback — dtype-aware nonzero check
    auto input_cpu = input.to(Device::cpu());
    int64_t numel = input.numel();

    // Lambda to check if element at flat_idx is nonzero, dispatched by dtype
    auto is_nonzero = [&](int64_t flat_idx) -> bool {
        switch (input_cpu.dtype()) {
            case DType::Float32: return static_cast<const float*>(input_cpu.data_ptr())[flat_idx] != 0.0f;
            case DType::Float64: return static_cast<const double*>(input_cpu.data_ptr())[flat_idx] != 0.0;
            case DType::Int32:   return static_cast<const int32_t*>(input_cpu.data_ptr())[flat_idx] != 0;
            case DType::Int64:   return static_cast<const int64_t*>(input_cpu.data_ptr())[flat_idx] != 0;
            case DType::Int16:   return static_cast<const int16_t*>(input_cpu.data_ptr())[flat_idx] != 0;
            case DType::Int8:    return static_cast<const int8_t*>(input_cpu.data_ptr())[flat_idx] != 0;
            case DType::UInt8:   return static_cast<const uint8_t*>(input_cpu.data_ptr())[flat_idx] != 0;
            case DType::Bool:    return static_cast<const bool*>(input_cpu.data_ptr())[flat_idx];
            case DType::Float16: return static_cast<float>(static_cast<const Float16*>(input_cpu.data_ptr())[flat_idx]) != 0.0f;
            case DType::BFloat16: return static_cast<float>(static_cast<const BFloat16*>(input_cpu.data_ptr())[flat_idx]) != 0.0f;
            default: return static_cast<const float*>(input_cpu.data_ptr())[flat_idx] != 0.0f;
        }
    };

    std::vector<std::vector<int64_t>> indices;

    for (int64_t flat_idx = 0; flat_idx < numel; ++flat_idx) {
        if (is_nonzero(flat_idx)) {
            std::vector<int64_t> multi_idx(ndim);
            int64_t remaining = flat_idx;
            for (int64_t d = ndim - 1; d >= 0; --d) {
                multi_idx[d] = remaining % shape[d];
                remaining /= shape[d];
            }
            indices.push_back(multi_idx);
        }
    }

    int64_t num_nonzero = indices.size();
    if (num_nonzero == 0) {
        return tenzor::empty({0, ndim}, DType::Int64, input.device());
    }

    std::vector<int64_t> result_data(num_nonzero * ndim);
    for (int64_t i = 0; i < num_nonzero; ++i) {
        for (int64_t d = 0; d < ndim; ++d) {
            result_data[i * ndim + d] = indices[i][d];
        }
    }

    auto result = tenzor::from_data(result_data.data(), {num_nonzero, ndim}, Device::cpu());
    return result;
}

auto select(const Tensor& input, int64_t dim, int64_t index) -> Tensor {
    // Normalize dimension
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw tenzor::IndexError("Dimension out of range for select");
    }

    // Normalize and validate index
    int64_t dim_size = input.shape()[dim];
    if (index < 0) index += dim_size;
    if (index < 0 || index >= dim_size) {
        throw tenzor::IndexError("Index out of range for select: index=" + std::to_string(index) +
                               ", dimension size=" + std::to_string(dim_size));
    }

    // Use slice to get the single element, then squeeze to remove the dimension
    auto sliced = input.slice(dim, index, index + 1, 1);
    return sliced.squeeze(dim);
}

auto narrow(const Tensor& input, int64_t dim, int64_t start, int64_t length) -> Tensor {
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw tenzor::IndexError("Dimension out of range for narrow");
    }
    if (start < 0 || length < 0 || start + length > input.shape()[dim]) {
        throw tenzor::IndexError("narrow: start (" + std::to_string(start) + ") + length (" +
                               std::to_string(length) + ") exceeds dimension size (" +
                               std::to_string(input.shape()[dim]) + ")");
    }
    return input.slice(dim, start, start + length, 1);
}

auto index(const Tensor& input,
           const std::vector<std::optional<Tensor>>& indices) -> Tensor {
    if (indices.empty()) {
        throw std::runtime_error("index: at least one index tensor must be provided");
    }
    if (static_cast<int64_t>(indices.size()) > input.ndim()) {
        throw std::runtime_error("index: too many index tensors (" +
                                 std::to_string(indices.size()) + ") for tensor with " +
                                 std::to_string(input.ndim()) + " dimensions");
    }

    // Validate index dtypes and collect non-null indices for broadcasting
    std::vector<size_t> non_null_dims;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i].has_value()) {
            const auto& idx = indices[i].value();
            if (idx.dtype() != DType::Int32 && idx.dtype() != DType::Int64) {
                throw std::runtime_error(
                    "index: index tensors must be Int32 or Int64, got index at dim " +
                    std::to_string(i));
            }
            non_null_dims.push_back(i);
        }
    }

    if (non_null_dims.empty()) {
        throw std::runtime_error("index: at least one non-null index tensor is required");
    }

    // Broadcast all non-null index tensors to a common shape
    auto first_shape = indices[non_null_dims[0]].value().shape();
    std::vector<int64_t> broadcast_shape(first_shape.begin(), first_shape.end());
    for (size_t k = 1; k < non_null_dims.size(); ++k) {
        broadcast_shape = broadcast_shapes(
            broadcast_shape, indices[non_null_dims[k]].value().shape());
    }

    // Pack index tensors (expanded to broadcast shape, cast to Int64) into inputs vector
    // Convention: inputs[0] = source tensor, inputs[1..N] = index tensors (one per indexed dim)
    // Non-null indices are passed as actual tensors; null positions are represented by
    // a 0-element Int64 sentinel tensor so the kernel knows which dims are indexed.
    std::vector<Tensor> dispatch_inputs;
    dispatch_inputs.reserve(1 + indices.size());
    dispatch_inputs.push_back(input);

    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i].has_value()) {
            auto idx = indices[i].value();
            if (idx.dtype() == DType::Int32) {
                // Cast to Int64 for uniform handling
                NewOpAttributes cast_attrs;
                cast_attrs.set(AttrKey::TargetDtype, static_cast<int64_t>(DType::Int64));
                std::vector<Tensor> cast_inputs = {idx};
                idx = dispatch_single(OpId::Cast, cast_inputs, cast_attrs);
            }
            // Expand to broadcast shape
            idx = idx.expand(broadcast_shape);
            dispatch_inputs.push_back(std::move(idx));
        } else {
            // Sentinel: 0-element tensor signals "full slice on this dimension"
            dispatch_inputs.push_back(empty({0}, DType::Int64, input.device()));
        }
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::NumIndices, static_cast<int64_t>(indices.size()));

    return dispatch_single(OpId::AdvancedIndex, dispatch_inputs, attrs);
}

void index_put(Tensor& input,
               const std::vector<std::optional<Tensor>>& indices,
               const Tensor& values) {
    if (indices.empty()) {
        throw std::runtime_error("index_put: at least one index tensor must be provided");
    }
    if (static_cast<int64_t>(indices.size()) > input.ndim()) {
        throw std::runtime_error("index_put: too many index tensors (" +
                                 std::to_string(indices.size()) + ") for tensor with " +
                                 std::to_string(input.ndim()) + " dimensions");
    }

    // Validate and collect non-null indices
    std::vector<size_t> non_null_dims;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i].has_value()) {
            const auto& idx = indices[i].value();
            if (idx.dtype() != DType::Int32 && idx.dtype() != DType::Int64) {
                throw std::runtime_error(
                    "index_put: index tensors must be Int32 or Int64, got index at dim " +
                    std::to_string(i));
            }
            non_null_dims.push_back(i);
        }
    }

    if (non_null_dims.empty()) {
        throw std::runtime_error("index_put: at least one non-null index tensor is required");
    }

    // Broadcast non-null index tensors
    auto first_shape_put = indices[non_null_dims[0]].value().shape();
    std::vector<int64_t> broadcast_shape(first_shape_put.begin(), first_shape_put.end());
    for (size_t k = 1; k < non_null_dims.size(); ++k) {
        broadcast_shape = broadcast_shapes(
            broadcast_shape, indices[non_null_dims[k]].value().shape());
    }

    // Pack inputs: [input, values, idx0, idx1, ...]
    std::vector<Tensor> dispatch_inputs;
    dispatch_inputs.reserve(2 + indices.size());
    dispatch_inputs.push_back(input);
    dispatch_inputs.push_back(values);

    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i].has_value()) {
            auto idx = indices[i].value();
            if (idx.dtype() == DType::Int32) {
                NewOpAttributes cast_attrs;
                cast_attrs.set(AttrKey::TargetDtype, static_cast<int64_t>(DType::Int64));
                std::vector<Tensor> cast_inputs = {idx};
                idx = dispatch_single(OpId::Cast, cast_inputs, cast_attrs);
            }
            idx = idx.expand(broadcast_shape);
            dispatch_inputs.push_back(std::move(idx));
        } else {
            dispatch_inputs.push_back(empty({0}, DType::Int64, input.device()));
        }
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::NumIndices, static_cast<int64_t>(indices.size()));

    auto result = dispatch_single(OpId::AdvancedIndexPut, dispatch_inputs, attrs);

    // Copy result back into input (in-place semantics)
    // The kernel returns the modified tensor; overwrite input's data
    input = result;
}

// ============================================================================
// Extended indexing with Ellipsis, NewAxis, and boolean mask support
// ============================================================================

auto index_extended(const Tensor& input,
                    const std::vector<IndexElement>& indices) -> Tensor {
    int64_t ndim = input.dim();

    // Step 1: Count explicit dimensions and validate
    int64_t num_ellipsis = 0;
    int64_t num_explicit_dims = 0;
    std::vector<int64_t> newaxis_positions;

    for (size_t i = 0; i < indices.size(); ++i) {
        if (std::holds_alternative<Ellipsis>(indices[i])) {
            ++num_ellipsis;
        } else if (std::holds_alternative<NewAxis>(indices[i])) {
            newaxis_positions.push_back(static_cast<int64_t>(i));
        } else if (std::holds_alternative<std::nullopt_t>(indices[i])) {
            ++num_explicit_dims;
        } else if (std::holds_alternative<Tensor>(indices[i])) {
            ++num_explicit_dims;
        }
    }

    if (num_ellipsis > 1) {
        throw std::runtime_error("index_extended: at most one Ellipsis allowed");
    }

    // Step 2: Expand ellipsis to fill remaining dims
    int64_t ellipsis_ndims = ndim - num_explicit_dims;
    if (ellipsis_ndims < 0) {
        throw std::runtime_error("index_extended: too many index dimensions");
    }

    // Build expanded optional index list
    std::vector<std::optional<Tensor>> expanded;
    expanded.reserve(ndim + newaxis_positions.size());

    for (const auto& elem : indices) {
        if (std::holds_alternative<Ellipsis>(elem)) {
            // Insert ellipsis_ndims full-slices
            for (int64_t j = 0; j < ellipsis_ndims; ++j) {
                expanded.push_back(std::nullopt);
            }
        } else if (std::holds_alternative<NewAxis>(elem)) {
            // Skip — handled after indexing
            continue;
        } else if (std::holds_alternative<std::nullopt_t>(elem)) {
            expanded.push_back(std::nullopt);
        } else if (std::holds_alternative<Tensor>(elem)) {
            const auto& t = std::get<Tensor>(elem);
            // Boolean mask: convert to integer indices via nonzero
            if (t.dtype() == DType::Bool) {
                auto nz = nonzero(t);
                expanded.push_back(nz);
            } else {
                expanded.push_back(t);
            }
        }
    }

    // If no ellipsis was present, pad remaining dims with nullopt
    while (static_cast<int64_t>(expanded.size()) < ndim) {
        expanded.push_back(std::nullopt);
    }

    // Step 3: Perform the indexing
    auto result = index(input, expanded);

    // Step 4: Insert new dimensions for NewAxis positions
    // Process in reverse order to keep positions valid
    for (auto it = newaxis_positions.rbegin(); it != newaxis_positions.rend(); ++it) {
        result = result.unsqueeze(*it);
    }

    return result;
}

auto one_hot(const Tensor& input, int64_t num_classes) -> Tensor {
    if (num_classes < 0) {
        // Infer from max value — need to compute on CPU
        auto input_cpu = input.to(Device::cpu()).contiguous();
        int64_t max_val = 0;
        auto numel = input_cpu.numel();
        switch (input_cpu.dtype()) {
            case DType::Int64: {
                auto* ptr = static_cast<const int64_t*>(input_cpu.data_ptr());
                for (int64_t i = 0; i < numel; ++i) max_val = std::max(max_val, ptr[i]);
                break;
            }
            case DType::Int32: {
                auto* ptr = static_cast<const int32_t*>(input_cpu.data_ptr());
                for (int64_t i = 0; i < numel; ++i) max_val = std::max(max_val, static_cast<int64_t>(ptr[i]));
                break;
            }
            default: {
                auto converted = input.to(DType::Int64).to(Device::cpu()).contiguous();
                auto* ptr = static_cast<const int64_t*>(converted.data_ptr());
                for (int64_t i = 0; i < numel; ++i) max_val = std::max(max_val, ptr[i]);
                break;
            }
        }
        num_classes = max_val + 1;
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::NumClasses, num_classes);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::OneHot, inputs, attrs)[0];
}

auto index_add(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& source) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    std::vector<Tensor> inputs = {input.contiguous(), index.contiguous(), source.contiguous()};
    return dispatch(OpId::IndexAdd, inputs, attrs)[0];
}

auto index_copy(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& source) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    std::vector<Tensor> inputs = {input.contiguous(), index.contiguous(), source.contiguous()};
    return dispatch(OpId::IndexCopy, inputs, attrs)[0];
}

auto index_fill(const Tensor& input, int64_t dim, const Tensor& index, double value) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Value, value);
    std::vector<Tensor> inputs = {input.contiguous(), index.contiguous()};
    return dispatch(OpId::IndexFill, inputs, attrs)[0];
}

// Deprecated `float` overload (audit-5 Y.3): forward to the `double` overload
// so the user-provided scalar travels at double precision into `AttrKey::Value`.
auto index_fill(const Tensor& input, int64_t dim, const Tensor& index, float value) -> Tensor {
    return index_fill(input, dim, index, static_cast<double>(value));
}

auto bincount(const Tensor& input,
              const std::optional<Tensor>& weights,
              int64_t minlength) -> Tensor {
    if (input.ndim() != 1) {
        throw std::runtime_error("bincount: input must be 1-dimensional");
    }
    if (minlength < 0) {
        throw std::runtime_error("bincount: minlength must be non-negative");
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::Minlength, minlength);

    std::vector<Tensor> inputs;
    inputs.push_back(input.contiguous());
    if (weights.has_value()) {
        if (weights->numel() != input.numel()) {
            throw std::runtime_error("bincount: weights must have same length as input");
        }
        inputs.push_back(weights->contiguous());
    }

    return dispatch(OpId::Bincount, inputs, attrs)[0];
}

auto index_reduce(const Tensor& input, int64_t dim, const Tensor& index,
                  const Tensor& source, const std::string& reduce,
                  bool include_self) -> Tensor {
    // index_reduce is a thin wrapper around scatter_reduce.
    // The difference from scatter_reduce is argument naming/ordering:
    //   scatter_reduce(input, dim, index, src, reduce, include_self)
    //   index_reduce(input, dim, index, source, reduce, include_self)
    // Both use the same OpId::ScatterReduce dispatch.
    return scatter_reduce(input, dim, index, source, reduce, include_self);
}

auto take_along_dim(const Tensor& input, const Tensor& indices, int64_t dim) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    std::vector<Tensor> inputs = {input.contiguous(), indices.contiguous()};
    return dispatch(OpId::TakeAlongDim, inputs, attrs)[0];
}

auto masked_scatter(const Tensor& input, const Tensor& mask, const Tensor& source) -> Tensor {
    std::vector<Tensor> inputs = {input.contiguous(), mask.contiguous(), source.contiguous()};
    return dispatch(OpId::MaskedScatter, inputs)[0];
}

auto select_scatter(const Tensor& input, const Tensor& src, int64_t dim, int64_t index) -> Tensor {
    // audit-5 Y.6: normalise negative dim at the dispatcher so backends that
    // index `shape[dim]` without their own normalisation (CUDA/ROCm/OneAPI
    // per audit-3 Q.1 / audit-4 V.14) don't underflow.  Mirrors `cat` /
    // `slice`.
    const int64_t ndim = static_cast<int64_t>(input.shape().size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("select_scatter: dim out of range");
    }
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Index, index);
    std::vector<Tensor> inputs = {input.contiguous(), src.contiguous()};
    return dispatch(OpId::SelectScatter, inputs, attrs)[0];
}

auto slice_scatter(const Tensor& input, const Tensor& src, int64_t dim,
                   int64_t start, int64_t end, int64_t step) -> Tensor {
    // audit-5 Y.6: same negative-dim normalisation as select_scatter above.
    const int64_t ndim = static_cast<int64_t>(input.shape().size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("slice_scatter: dim out of range");
    }
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Start, start);
    attrs.set(AttrKey::End, end);
    attrs.set(AttrKey::Step, step);
    std::vector<Tensor> inputs = {input.contiguous(), src.contiguous()};
    return dispatch(OpId::SliceScatter, inputs, attrs)[0];
}

auto diagonal_scatter(const Tensor& input, const Tensor& src, int64_t offset,
                      int64_t dim1, int64_t dim2) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Diagonal, offset);
    attrs.set(AttrKey::Dim1, dim1);
    attrs.set(AttrKey::Dim2, dim2);
    std::vector<Tensor> inputs = {input.contiguous(), src.contiguous()};
    return dispatch(OpId::DiagonalScatter, inputs, attrs)[0];
}

} // namespace tenzor
