#include "tenzor/ops/indexing.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/creation.hpp"
#include <iostream>

namespace tenzor {

auto slice(const Tensor& input, int64_t dim, int64_t start,
          int64_t end, int64_t step) -> Tensor {
    return input.slice(dim, start, end, step);
}

auto index_select(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor {
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim);
    std::vector<Tensor> inputs = {input, index};
    return dispatch(OpId::IndexSelect, inputs, attrs)[0];
}

auto gather(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor {
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim);
    std::vector<Tensor> inputs = {input, index};
    return dispatch(OpId::Gather, inputs, attrs)[0];
}

auto scatter(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) -> Tensor {
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim);
    std::vector<Tensor> inputs = {input, index, src};
    return dispatch(OpId::Scatter, inputs, attrs)[0];
}

auto masked_select(const Tensor& input, const Tensor& mask) -> Tensor {
    std::vector<Tensor> inputs = {input, mask};
    return dispatch(OpId::MaskedSelect, inputs)[0];
}

auto masked_fill(const Tensor& input, const Tensor& mask, float value) -> Tensor {
    OpAttributes attrs;
    char value_buf[32];
    snprintf(value_buf, sizeof(value_buf), "%.9e", value);
    attrs["value"] = std::string(value_buf);
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
        attrs["shape"] = shape_str;
        attrs["ndim"] = std::to_string(ndim);

        std::vector<Tensor> inputs = {input};
        auto results = dispatch(OpId::Nonzero, inputs, attrs);
        return results[0];
    }

    // CPU fallback
    auto input_cpu = input.to(Device::cpu());
    const float* data = static_cast<const float*>(input_cpu.data_ptr());
    int64_t numel = input.numel();

    std::vector<std::vector<int64_t>> indices;

    for (int64_t flat_idx = 0; flat_idx < numel; ++flat_idx) {
        if (std::abs(data[flat_idx]) > 1e-10f) {
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

} // namespace tenzor
