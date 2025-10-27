#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include <iostream>
#include <iomanip>

using namespace tenzor;

TEST(TensorLifetimeTest, ContiguousDataPointerStability) {
    std::cout << std::fixed << std::setprecision(6);

    // Create base tensor
    auto boxes = zeros({3, 4}, DType::Float32, Device::cpu());
    auto* data = boxes.data<float>();

    for (int i = 0; i < 3; i++) {
        data[i * 4 + 0] = static_cast<float>(i * 2);
        data[i * 4 + 1] = static_cast<float>(i * 2 + 1);
        data[i * 4 + 2] = static_cast<float>(i * 2 + 10);
        data[i * 4 + 3] = static_cast<float>(i * 2 + 11);
    }

    std::cout << "\n=== STEP 1: Create slice ===\n";
    auto x2_slice = boxes.slice(1, 2, 3);
    std::cout << "x2_slice is_contiguous: " << x2_slice.is_contiguous() << "\n";
    std::cout << "x2_slice data ptr: " << (void*)x2_slice.data<float>() << "\n";
    std::cout << "x2_slice values: [" << x2_slice.data<float>()[0] << ", "
              << x2_slice.data<float>()[1] << ", " << x2_slice.data<float>()[2] << "]\n";

    std::cout << "\n=== STEP 2: Call .contiguous() ===\n";
    auto x2_contig = x2_slice.contiguous();
    std::cout << "x2_contig is_contiguous: " << x2_contig.is_contiguous() << "\n";
    std::cout << "x2_contig data ptr: " << (void*)x2_contig.data<float>() << "\n";
    std::cout << "x2_contig values: [" << x2_contig.data<float>()[0] << ", "
              << x2_contig.data<float>()[1] << ", " << x2_contig.data<float>()[2] << "]\n";

    std::cout << "\n=== STEP 3: Create x1_contig ===\n";
    auto x1_contig = boxes.slice(1, 0, 1).contiguous();
    std::cout << "x1_contig is_contiguous: " << x1_contig.is_contiguous() << "\n";
    std::cout << "x1_contig data ptr: " << (void*)x1_contig.data<float>() << "\n";
    std::cout << "x1_contig values: [" << x1_contig.data<float>()[0] << ", "
              << x1_contig.data<float>()[1] << ", " << x1_contig.data<float>()[2] << "]\n";

    std::cout << "\n=== STEP 4: Check x2_contig data BEFORE subtraction ===\n";
    std::cout << "x2_contig data ptr: " << (void*)x2_contig.data<float>() << "\n";
    std::cout << "x2_contig values: [" << x2_contig.data<float>()[0] << ", "
              << x2_contig.data<float>()[1] << ", " << x2_contig.data<float>()[2] << "]\n";

    std::cout << "\n=== STEP 5: Perform subtraction ===\n";
    auto widths = x2_contig - x1_contig;

    std::cout << "\n=== STEP 6: Check x2_contig data AFTER subtraction ===\n";
    std::cout << "x2_contig data ptr: " << (void*)x2_contig.data<float>() << "\n";
    std::cout << "x2_contig values: [" << x2_contig.data<float>()[0] << ", "
              << x2_contig.data<float>()[1] << ", " << x2_contig.data<float>()[2] << "]\n";

    std::cout << "\n=== STEP 7: Check result ===\n";
    std::cout << "widths data ptr: " << (void*)widths.data<float>() << "\n";
    std::cout << "widths values: [" << widths.data<float>()[0] << ", "
              << widths.data<float>()[1] << ", " << widths.data<float>()[2] << "]\n";

    std::cout << "\n=== VERIFICATION ===\n";
    std::cout << "Expected widths: [10, 10, 10]\n";
    std::cout << "Actual widths:   [" << widths.data<float>()[0] << ", "
              << widths.data<float>()[1] << ", " << widths.data<float>()[2] << "]\n";

    // Test expectations
    EXPECT_NEAR(x2_contig.data<float>()[0], 10.0f, 1e-5f) << "x2_contig[0] should be 10";
    EXPECT_NEAR(x2_contig.data<float>()[1], 12.0f, 1e-5f) << "x2_contig[1] should be 12";
    EXPECT_NEAR(x2_contig.data<float>()[2], 14.0f, 1e-5f) << "x2_contig[2] should be 14";

    EXPECT_NEAR(widths.data<float>()[0], 10.0f, 1e-5f) << "widths[0] should be 10";
    EXPECT_NEAR(widths.data<float>()[1], 10.0f, 1e-5f) << "widths[1] should be 10";
    EXPECT_NEAR(widths.data<float>()[2], 10.0f, 1e-5f) << "widths[2] should be 10";
}

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
