/**
 * @file fuzz_autograd_backward.cpp
 * @brief libFuzzer target for autograd backward pass
 *
 * Constructs small random computation graphs from fuzzer input,
 * runs forward + backward, and catches crashes in gradient computation.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include <cstdint>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) return 0;

    // Parse tensor dimensions from first 2 bytes (capped at 32)
    int64_t rows = (data[0] % 32) + 1;
    int64_t cols = (data[1] % 32) + 1;
    uint8_t num_ops = (data[2] % 8) + 1;  // 1-8 operations
    size_t offset = 3;

    try {
        using namespace tenzor;
        using namespace tenzor::autograd;

        auto x = Variable(randn({rows, cols}, DType::Float32, Device::cpu()), true);
        auto y = Variable(randn({cols, rows}, DType::Float32, Device::cpu()), true);

        Variable result = x;

        for (uint8_t i = 0; i < num_ops && offset < size; ++i) {
            uint8_t op = data[offset++] % 8;
            switch (op) {
            case 0: result = add(result, result); break;
            case 1: result = mul(result, result); break;
            case 2: result = sub(result, result); break;
            case 3: result = autograd::relu(result); break;
            case 4: result = autograd::sigmoid(result); break;
            case 5: result = autograd::tanh(result); break;
            case 6: result = autograd::sum(result); break;
            case 7: result = autograd::neg(result); break;
            }
        }

        // Sum to scalar if needed, then backward
        if (result.tensor().numel() > 1) {
            result = autograd::sum(result);
        }

        result.backward();

        // Access gradients to trigger any lazy computation
        if (x.grad().has_value()) {
            (void)x.grad().value().numel();
        }
    } catch (...) {
        // Expected for some op combinations
    }

    return 0;
}
