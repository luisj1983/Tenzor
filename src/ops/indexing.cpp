#include "tenzor/ops/indexing.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/dtype.hpp"
#include <cstdint>

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

auto masked_select(const Tensor& input, const Tensor& mask) -> Tensor {
    std::vector<Tensor> inputs = {input, mask};
    return dispatch(OpId::MaskedSelect, inputs)[0];
}

auto masked_fill(const Tensor& input, const Tensor& mask, float value) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Value, static_cast<double>(value));
    std::vector<Tensor> inputs = {input, mask};
    return dispatch(OpId::MaskedFill, inputs, attrs)[0];
}

auto where(const Tensor& condition, const Tensor& x, const Tensor& y) -> Tensor {
    std::vector<Tensor> inputs = {condition, x, y};
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
        throw std::out_of_range("Dimension out of range for select");
    }

    // Normalize and validate index
    int64_t dim_size = input.shape()[dim];
    if (index < 0) index += dim_size;
    if (index < 0 || index >= dim_size) {
        throw std::out_of_range("Index out of range for select: index=" + std::to_string(index) +
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
        throw std::out_of_range("Dimension out of range for narrow");
    }
    if (start < 0 || length < 0 || start + length > input.shape()[dim]) {
        throw std::out_of_range("narrow: start (" + std::to_string(start) + ") + length (" +
                               std::to_string(length) + ") exceeds dimension size (" +
                               std::to_string(input.shape()[dim]) + ")");
    }
    return input.slice(dim, start, start + length, 1);
}

} // namespace tenzor
