/**
 * @file fuzz_reshape.cpp
 * @brief libFuzzer target for tensor reshape with untrusted dimensions
 *
 * Creates a small tensor and attempts to reshape it using fuzzer-provided
 * dimensions, testing for crashes in shape validation logic.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include <cstdint>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 3) return 0;

    // Create a small source tensor (numel from first 2 bytes, capped)
    uint16_t src_size = (static_cast<uint16_t>(data[0]) |
                         (static_cast<uint16_t>(data[1]) << 8));
    src_size = (src_size % 256) + 1; // 1-256 elements

    auto tensor = tenzor::ones({static_cast<int64_t>(src_size)},
                                tenzor::DType::Float32, tenzor::Device::cpu());

    // Remaining bytes: target shape (2 bytes per dim, signed to allow -1)
    size_t offset = 2;
    uint8_t ndim = data[offset++] % 6 + 1; // 1-6 dims

    std::vector<int64_t> target_shape;
    for (uint8_t i = 0; i < ndim && offset + 1 < size; ++i) {
        int16_t dim = static_cast<int16_t>(data[offset]) |
                      (static_cast<int16_t>(data[offset + 1]) << 8);
        // Allow -1 (infer) and small positive values
        if (dim == -1) {
            target_shape.push_back(-1);
        } else {
            target_shape.push_back(std::abs(dim) % 512 + 1);
        }
        offset += 2;
    }

    if (target_shape.empty()) return 0;

    try {
        auto reshaped = tenzor::reshape(tensor, target_shape);
        (void)reshaped.numel();
    } catch (...) {
        // Expected for incompatible shapes
    }

    return 0;
}
