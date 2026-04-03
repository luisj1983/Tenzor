/**
 * @file fuzz_type_promotion.cpp
 * @brief libFuzzer target for type promotion in binary operations
 *
 * Creates tensors with different dtypes and applies binary operations,
 * testing that type promotion rules don't crash.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) return 0;

    using namespace tenzor;

    // Map fuzzer bytes to dtypes (subset that supports arithmetic)
    static constexpr DType dtypes[] = {
        DType::Float32, DType::Float64,
        DType::Int32, DType::Int64,
        DType::Float16, DType::BFloat16,
    };
    constexpr size_t num_dtypes = sizeof(dtypes) / sizeof(dtypes[0]);

    DType dtype_a = dtypes[data[0] % num_dtypes];
    DType dtype_b = dtypes[data[1] % num_dtypes];
    uint8_t op_id = data[2] % 4;
    int64_t dim = (data[3] % 16) + 1;

    try {
        auto a = ones({dim, dim}, dtype_a, Device::cpu());
        auto b = ones({dim, dim}, dtype_b, Device::cpu());

        Tensor result;
        switch (op_id) {
        case 0: result = add(a, b); break;
        case 1: result = sub(a, b); break;
        case 2: result = mul(a, b); break;
        case 3: result = div(a, b); break;
        }

        // Result should exist and have correct shape
        if (result.numel() != dim * dim) {
            __builtin_trap();
        }
    } catch (...) {
        // Some type combinations may not be supported
    }

    return 0;
}
