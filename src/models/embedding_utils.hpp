#pragma once

#include <algorithm>
#include <vector>

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"

namespace tenzor {
namespace models {

// Build sequential position ids [0, 1, ..., seq_len-1] broadcast across the
// batch: result shape [batch_size, seq_len], Int64, on input_ids' device.
// Shared by the BERT and ALBERT embedding layers.
inline auto make_sequential_position_ids(const Tensor& input_ids) -> Tensor {
    auto shape = input_ids.shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[1];
    auto target_device = input_ids.device();

    // Create position IDs on CPU first: [0, 1, 2, ..., seq_len-1]
    std::vector<int64_t> pos_data(seq_len);
    for (int64_t i = 0; i < seq_len; ++i) {
        pos_data[i] = i;
    }

    Tensor position_ids_cpu(std::vector<int64_t>{seq_len}, DType::Int64, Device::cpu());
    std::copy(pos_data.begin(), pos_data.end(), position_ids_cpu.data<int64_t>());

    // Expand to [batch_size, seq_len] on CPU
    position_ids_cpu = position_ids_cpu.unsqueeze(0);  // [1, seq_len]
    Tensor expanded_cpu(std::vector<int64_t>{batch_size, seq_len}, DType::Int64, Device::cpu());
    auto pos_data_ptr = position_ids_cpu.data<int64_t>();
    auto expanded_ptr = expanded_cpu.data<int64_t>();
    for (int64_t b = 0; b < batch_size; ++b) {
        std::copy(pos_data_ptr, pos_data_ptr + seq_len, expanded_ptr + b * seq_len);
    }

    // Move to target device
    return (target_device == Device::cpu()) ? expanded_cpu : expanded_cpu.to(target_device);
}

}  // namespace models
}  // namespace tenzor
