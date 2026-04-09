#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include <iostream>

using namespace tenzor;

TEST(SliceDebugTest, InspectSlicedTensors) {
    // Create a simple test case
    auto boxes = zeros({3, 4}, DType::Float32, Device::cpu());
    auto* data = boxes.data<float>();

    // Fill with known values:
    // Box 0: [0, 1, 10, 11] -> width=10, height=10, area=100
    // Box 1: [2, 3, 12, 13] -> width=10, height=10, area=100
    // Box 2: [4, 5, 14, 15] -> width=10, height=10, area=100
    for (int i = 0; i < 3; i++) {
        data[i * 4 + 0] = static_cast<float>(i * 2);      // x1
        data[i * 4 + 1] = static_cast<float>(i * 2 + 1);  // y1
        data[i * 4 + 2] = static_cast<float>(i * 2 + 10); // x2
        data[i * 4 + 3] = static_cast<float>(i * 2 + 11); // y2
    }

    std::cout << "Original tensor shape: [" << boxes.shape()[0] << ", " << boxes.shape()[1] << "]\n";
    std::cout << "Original data:\n";
    for (int i = 0; i < 3; i++) {
        std::cout << "  Box " << i << ": ["
                  << data[i*4+0] << ", " << data[i*4+1] << ", "
                  << data[i*4+2] << ", " << data[i*4+3] << "]\n";
    }

    // Try slicing column 0 (x1 values)
    auto x1 = boxes.slice(1, 0, 1);
    std::cout << "\nSlice(1, 0, 1) - x1 column:\n";
    std::cout << "  Shape: [" << x1.shape()[0] << ", " << x1.shape()[1] << "]\n";
    std::cout << "  Strides: [" << x1.strides()[0] << ", " << x1.strides()[1] << "]\n";
    std::cout << "  Expected values: [0, 2, 4]\n";

    auto* x1_data = x1.data<float>();
    std::cout << "  Actual values via data<float>():\n";
    for (int64_t i = 0; i < x1.shape()[0]; i++) {
        std::cout << "    [" << i << "]: " << x1_data[i] << "\n";
    }

    // Try slicing column 2 (x2 values)
    auto x2 = boxes.slice(1, 2, 3);
    std::cout << "\nSlice(1, 2, 3) - x2 column:\n";
    std::cout << "  Shape: [" << x2.shape()[0] << ", " << x2.shape()[1] << "]\n";
    std::cout << "  Strides: [" << x2.strides()[0] << ", " << x2.strides()[1] << "]\n";
    std::cout << "  Expected values: [10, 12, 14]\n";

    auto* x2_data = x2.data<float>();
    std::cout << "  Actual values via data<float>():\n";
    for (int64_t i = 0; i < x2.shape()[0]; i++) {
        std::cout << "    [" << i << "]: " << x2_data[i] << "\n";
    }

    // Try the subtraction
    std::cout << "\nSubtraction x2 - x1:\n";
    auto widths = x2 - x1;
    std::cout << "  Shape: [" << widths.shape()[0];
    if (widths.ndim() > 1) {
        std::cout << ", " << widths.shape()[1];
    }
    std::cout << "]\n";
    std::cout << "  Expected values: [10, 10, 10]\n";

    auto* widths_data = widths.data<float>();
    std::cout << "  Actual values:\n";
    for (int64_t i = 0; i < std::min(widths.numel(), int64_t(10)); i++) {
        std::cout << "    [" << i << "]: " << widths_data[i] << "\n";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (!::testing::GTEST_FLAG(list_tests)) {
        tenzor::initialize();
    }
    return RUN_ALL_TESTS();
}
