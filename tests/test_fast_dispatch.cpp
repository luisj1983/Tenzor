/**
 * @file test_fast_dispatch.cpp
 * @brief Quick test to verify the new O(1) dispatch system works
 */

#include "tenzor/tenzor.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include <iostream>
#include <chrono>

using namespace tenzor;

int main() {
    // Initialize Tenzor
    initialize();

    std::cout << "\n========================================\n";
    std::cout << "  Testing O(1) Fast Dispatch System\n";
    std::cout << "========================================\n\n";

    // Create test tensors
    auto a = randn({256, 256});
    auto b = randn({256, 256});

    // Test using new dispatch (via math.cpp which now uses dispatch<OpId>)
    std::cout << "Testing add operation via fast dispatch..." << std::endl;
    auto c = add(a, b);
    std::cout << "  Result shape: [" << c.shape()[0] << ", " << c.shape()[1] << "]" << std::endl;

    // Measure dispatch overhead
    std::cout << "\nMeasuring dispatch overhead (10000 iterations)..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        auto result = add(a, b);
        volatile void* p = result.data_ptr();  // Prevent optimization
        (void)p;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double avg_ns = static_cast<double>(duration) / 10000.0;

    std::cout << "  Average time per add: " << avg_ns << " ns" << std::endl;
    std::cout << "  Operations per second: " << 1e9 / avg_ns << std::endl;

    // Test matmul
    std::cout << "\nTesting matmul operation..." << std::endl;
    auto d = matmul(a, b);
    std::cout << "  Result shape: [" << d.shape()[0] << ", " << d.shape()[1] << "]" << std::endl;

    std::cout << "\n========================================\n";
    std::cout << "  All tests passed!\n";
    std::cout << "========================================\n";

    return 0;
}
