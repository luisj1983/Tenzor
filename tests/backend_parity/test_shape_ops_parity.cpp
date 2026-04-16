/**
 * @file test_shape_ops_parity.cpp
 * @brief Shape/view operation parity tests across backends
 *
 * Tests reshape, transpose, permute, squeeze, unsqueeze, flatten, stack,
 * split, chunk, cat, repeat, tile, and expand operations. These are pure
 * data movement ops so exact match (rtol=0, atol=0) is used throughout.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Shape/View Operations Parity Tests (16 operations, exact match)
// ============================================================================

TEST(ShapeOpsParity, Reshape) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return reshape(inputs[0], {16, 64});
    }, {a}, 0, 0, "Reshape");
}

TEST(ShapeOpsParity, Transpose2D) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return transpose(inputs[0], 0, 1);
    }, {a}, 0, 0, "Transpose2D");
}

TEST(ShapeOpsParity, Permute3D) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({4, 8, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return permute(inputs[0], {2, 0, 1});
    }, {a}, 0, 0, "Permute3D");
}

TEST(ShapeOpsParity, Permute4D) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({2, 3, 4, 5}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return permute(inputs[0], {3, 1, 0, 2});
    }, {a}, 0, 0, "Permute4D");
}

TEST(ShapeOpsParity, Squeeze) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({1, 32, 1, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return squeeze(inputs[0]);
    }, {a}, 0, 0, "Squeeze");
}

TEST(ShapeOpsParity, SqueezeDim) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 1, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return squeeze(inputs[0], 1);
    }, {a}, 0, 0, "SqueezeDim");
}

TEST(ShapeOpsParity, Unsqueeze) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return unsqueeze(inputs[0], 0);
    }, {a}, 0, 0, "Unsqueeze");
}

TEST(ShapeOpsParity, Flatten) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({2, 4, 8, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return flatten(inputs[0], 1, 2);
    }, {a}, 0, 0, "Flatten");
}

TEST(ShapeOpsParity, StackDim0) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> to_stack = {inputs[0], inputs[1]};
        return stack(to_stack, 0);
    }, {a, b}, 0, 0, "Stack_Dim0");
}

TEST(ShapeOpsParity, StackDim1) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> to_stack = {inputs[0], inputs[1]};
        return stack(to_stack, 1);
    }, {a, b}, 0, 0, "Stack_Dim1");
}

TEST(ShapeOpsParity, Split) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto parts = split(inputs[0], 8, 0);
        return parts[0];
    }, {a}, 0, 0, "Split");
}

TEST(ShapeOpsParity, Chunk) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto parts = chunk(inputs[0], 4, 1);
        return parts[2];
    }, {a}, 0, 0, "Chunk");
}

TEST(ShapeOpsParity, CatDim0) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> v = {inputs[0], inputs[1]};
        return cat(v, 0);
    }, {a, b}, 0, 0, "Cat_Dim0");
}

TEST(ShapeOpsParity, CatDim1) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> v = {inputs[0], inputs[1]};
        return cat(v, 1);
    }, {a, b}, 0, 0, "Cat_Dim1");
}

TEST(ShapeOpsParity, Repeat) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return repeat(inputs[0], {2, 3});
    }, {a}, 0, 0, "Repeat");
}

TEST(ShapeOpsParity, Expand) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({1, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return expand(inputs[0], {16, 32});
    }, {a}, 0, 0, "Expand");
}



int main(int argc, char** argv) {
    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
    }
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    try {
        tenzor::finalize();
    } catch (...) {}
    return result;
}
