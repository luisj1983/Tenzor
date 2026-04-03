/**
 * @file fuzz_tensor_creation.cpp
 * @brief libFuzzer target for tensor creation with untrusted shapes
 *
 * Interprets fuzzer bytes as shape dimensions and attempts to create
 * tensors, catching integer overflow or excessive allocation issues.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <cstdint>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 2) return 0;

    // First byte: number of dimensions (capped at 8)
    uint8_t ndim = data[0] % 8 + 1;
    size_t offset = 1;

    // Remaining bytes: shape dimensions (2 bytes each, capped at 1024)
    std::vector<int64_t> shape;
    for (uint8_t i = 0; i < ndim && offset + 1 < size; ++i) {
        uint16_t dim = static_cast<uint16_t>(data[offset]) |
                       (static_cast<uint16_t>(data[offset + 1]) << 8);
        // Cap at 1024 to avoid OOM from legitimate allocations
        shape.push_back(dim % 1024 + 1);
        offset += 2;
    }

    if (shape.empty()) return 0;

    try {
        auto tensor = tenzor::zeros(shape, tenzor::DType::Float32, tenzor::Device::cpu());
        (void)tensor.numel();
        (void)tensor.is_contiguous();
    } catch (...) {
        // Expected for invalid shapes
    }

    return 0;
}
