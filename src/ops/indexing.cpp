#include "tenzor/ops/indexing.hpp"
#include "tenzor/backend/dispatch.hpp"
#include "tenzor/ops/creation.hpp"

namespace tenzor {

auto slice(const Tensor& input, int64_t dim, int64_t start,
          int64_t end, int64_t step) -> Tensor {
    return input.slice(dim, start, end, step);
}

auto index_select(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor {
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim);
    std::vector<Tensor> inputs = {input, index};
    return Dispatcher::dispatch("index_select", inputs, attrs)[0];
}

auto gather(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor {
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim);
    std::vector<Tensor> inputs = {input, index};
    return Dispatcher::dispatch("gather", inputs, attrs)[0];
}

auto scatter(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) -> Tensor {
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim);
    std::vector<Tensor> inputs = {input, index, src};
    return Dispatcher::dispatch("scatter", inputs, attrs)[0];
}

auto masked_select(const Tensor& input, const Tensor& mask) -> Tensor {
    std::vector<Tensor> inputs = {input, mask};
    return Dispatcher::dispatch("masked_select", inputs)[0];
}

auto masked_fill(const Tensor& input, const Tensor& mask, float value) -> Tensor {
    OpAttributes attrs;
    char value_buf[32];
    snprintf(value_buf, sizeof(value_buf), "%.9e", value);
    attrs["value"] = std::string(value_buf);
    std::vector<Tensor> inputs = {input, mask};
    return Dispatcher::dispatch("masked_fill", inputs, attrs)[0];
}

auto where(const Tensor& condition, const Tensor& x, const Tensor& y) -> Tensor {
    std::vector<Tensor> inputs = {condition, x, y};
    return Dispatcher::dispatch("where", inputs)[0];
}

auto take(const Tensor& input, const Tensor& index) -> Tensor {
    std::vector<Tensor> inputs = {input, index};
    return Dispatcher::dispatch("take", inputs)[0];
}

auto put(const Tensor& input, const Tensor& index, const Tensor& source) -> Tensor {
    std::vector<Tensor> inputs = {input, index, source};
    return Dispatcher::dispatch("put", inputs)[0];
}

auto nonzero(const Tensor& input) -> Tensor {
    // Simple CPU implementation - find all non-zero elements
    auto input_cpu = input.to(Device::cpu());
    const float* data = static_cast<const float*>(input_cpu.data_ptr());

    auto shape = input.shape();
    int64_t ndim = shape.size();
    int64_t numel = input.numel();

    // First pass: count non-zero elements
    std::vector<std::vector<int64_t>> indices;

    for (int64_t flat_idx = 0; flat_idx < numel; ++flat_idx) {
        if (std::abs(data[flat_idx]) > 1e-10f) {
            // Convert flat index to multi-dimensional indices
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

    // Create result tensor
    std::vector<int64_t> result_data(num_nonzero * ndim);
    for (int64_t i = 0; i < num_nonzero; ++i) {
        for (int64_t d = 0; d < ndim; ++d) {
            result_data[i * ndim + d] = indices[i][d];
        }
    }

    auto result = tenzor::from_data(result_data.data(), {num_nonzero, ndim}, Device::cpu());
    return result.to(input.device());
}

auto select(const Tensor& input, int64_t dim, int64_t index) -> Tensor {
    // Use slice to get the single element, then squeeze to remove the dimension
    auto sliced = input.slice(dim, index, index + 1, 1);
    return sliced.squeeze(dim);
}

} // namespace tenzor
