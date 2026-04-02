#include "tenzor/autograd/vmap.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/transform.hpp"
#include <vector>

namespace tenzor {

auto vmap(std::function<Variable(const Variable&)> func,
          const Variable& batched_input,
          int64_t batch_dim) -> Variable {
    auto input_tensor = batched_input.tensor();
    auto shape = input_tensor.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    // Normalize negative batch_dim
    if (batch_dim < 0) {
        batch_dim += ndim;
    }

    int64_t batch_size = shape[batch_dim];

    // Apply func to each slice along batch_dim
    std::vector<Tensor> results;
    results.reserve(batch_size);

    for (int64_t i = 0; i < batch_size; ++i) {
        // Select the i-th slice along batch_dim
        auto slice = tenzor::select(input_tensor, batch_dim, i);
        Variable slice_var(slice, batched_input.requires_grad());

        auto output = func(slice_var);
        results.push_back(output.tensor());
    }

    // Stack results along batch_dim
    auto stacked = tenzor::stack(std::span<const Tensor>(results), batch_dim);

    // Wrap in Variable - gradient tracking follows from inputs
    return Variable(stacked, batched_input.requires_grad());
}

} // namespace tenzor
