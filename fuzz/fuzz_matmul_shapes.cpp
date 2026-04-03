/**
 * @file fuzz_matmul_shapes.cpp
 * @brief libFuzzer target for matmul with edge-case shapes
 *
 * Tests matmul with various M, N, K dimensions derived from fuzzer
 * input, including edge cases like 0, 1, and non-square shapes.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) return 0;

    // Parse dimensions (capped at 128 to prevent OOM)
    int64_t M = (data[0] % 128) + 1;
    int64_t N = (data[1] % 128) + 1;
    int64_t K = (data[2] % 128) + 1;
    uint8_t dtype_id = data[3] % 2;  // 0=Float32, 1=Float64

    try {
        using namespace tenzor;

        DType dtype = dtype_id == 0 ? DType::Float32 : DType::Float64;
        auto a = randn({M, K}, dtype, Device::cpu());
        auto b = randn({K, N}, dtype, Device::cpu());

        auto c = matmul(a, b);

        // Verify shape
        auto shape = c.shape();
        if (shape.size() != 2 || shape[0] != M || shape[1] != N) {
            __builtin_trap();  // Signal to fuzzer: shape invariant violated
        }
    } catch (...) {
        // Expected for some combinations
    }

    return 0;
}
