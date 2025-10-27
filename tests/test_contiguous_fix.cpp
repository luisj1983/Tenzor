#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include <iostream>

using namespace tenzor;

TEST(ContiguousFixTest, SliceWithContiguous) {
    // Create test tensor
    auto boxes = zeros({3, 4}, DType::Float32, Device::cpu());
    auto* data = boxes.data<float>();

    for (int i = 0; i < 3; i++) {
        data[i * 4 + 0] = static_cast<float>(i * 2);
        data[i * 4 + 1] = static_cast<float>(i * 2 + 1);
        data[i * 4 + 2] = static_cast<float>(i * 2 + 10);
        data[i * 4 + 3] = static_cast<float>(i * 2 + 11);
    }

    std::cout << "\n=== WITHOUT .contiguous() ===\n";
    auto x1_view = boxes.slice(1, 0, 1);
    std::cout << "x1 is_contiguous: " << x1_view.is_contiguous() << "\n";
    auto* x1_data = x1_view.data<float>();
    std::cout << "x1 values: [" << x1_data[0] << ", " << x1_data[1] << ", " << x1_data[2] << "]\n";
    std::cout << "Expected:  [0, 2, 4]\n";

    std::cout << "\n=== WITH .contiguous() ===\n";
    auto x1_contig = boxes.slice(1, 0, 1).contiguous();
    std::cout << "x1_contig is_contiguous: " << x1_contig.is_contiguous() << "\n";
    auto* x1_contig_data = x1_contig.data<float>();
    std::cout << "x1_contig values: [" << x1_contig_data[0] << ", " << x1_contig_data[1] << ", " << x1_contig_data[2] << "]\n";
    std::cout << "Expected:         [0, 2, 4]\n";

    // Test subtraction with contiguous
    auto x2_contig = boxes.slice(1, 2, 3).contiguous();
    auto widths = x2_contig - x1_contig;
    std::cout << "\n=== Subtraction with contiguous ===\n";
    auto* widths_data = widths.data<float>();
    std::cout << "widths: [" << widths_data[0] << ", " << widths_data[1] << ", " << widths_data[2] << "]\n";
    std::cout << "Expected: [10, 10, 10]\n";

    // Verify results
    EXPECT_NEAR(x1_contig_data[0], 0.0f, 1e-5f);
    EXPECT_NEAR(x1_contig_data[1], 2.0f, 1e-5f);
    EXPECT_NEAR(x1_contig_data[2], 4.0f, 1e-5f);

    EXPECT_NEAR(widths_data[0], 10.0f, 1e-5f);
    EXPECT_NEAR(widths_data[1], 10.0f, 1e-5f);
    EXPECT_NEAR(widths_data[2], 10.0f, 1e-5f);
}

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
