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
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class ShapeOpsParity : public BackendTest {};
// ============================================================================
// Shape/View Operations Parity Tests (16 operations, exact match)
// ============================================================================

TEST_P(ShapeOpsParity, Reshape) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return reshape(inputs[0], {16, 64});
    }, {a}, device, 0, 0, "Reshape");
}

TEST_P(ShapeOpsParity, Transpose2D) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return transpose(inputs[0], 0, 1);
    }, {a}, device, 0, 0, "Transpose2D");
}

TEST_P(ShapeOpsParity, Permute3D) {

    auto a = randn({4, 8, 16}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return permute(inputs[0], {2, 0, 1});
    }, {a}, device, 0, 0, "Permute3D");
}

TEST_P(ShapeOpsParity, Permute4D) {

    auto a = randn({2, 3, 4, 5}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return permute(inputs[0], {3, 1, 0, 2});
    }, {a}, device, 0, 0, "Permute4D");
}

TEST_P(ShapeOpsParity, Squeeze) {

    auto a = randn({1, 32, 1, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return squeeze(inputs[0]);
    }, {a}, device, 0, 0, "Squeeze");
}

TEST_P(ShapeOpsParity, SqueezeDim) {

    auto a = randn({32, 1, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return squeeze(inputs[0], 1);
    }, {a}, device, 0, 0, "SqueezeDim");
}

TEST_P(ShapeOpsParity, Unsqueeze) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return unsqueeze(inputs[0], 0);
    }, {a}, device, 0, 0, "Unsqueeze");
}

TEST_P(ShapeOpsParity, Flatten) {

    auto a = randn({2, 4, 8, 16}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return flatten(inputs[0], 1, 2);
    }, {a}, device, 0, 0, "Flatten");
}

TEST_P(ShapeOpsParity, StackDim0) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> to_stack = {inputs[0], inputs[1]};
        return stack(to_stack, 0);
    }, {a, b}, device, 0, 0, "Stack_Dim0");
}

TEST_P(ShapeOpsParity, StackDim1) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> to_stack = {inputs[0], inputs[1]};
        return stack(to_stack, 1);
    }, {a, b}, device, 0, 0, "Stack_Dim1");
}

TEST_P(ShapeOpsParity, Split) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto parts = split(inputs[0], 8, 0);
        return parts[0];
    }, {a}, device, 0, 0, "Split");
}

TEST_P(ShapeOpsParity, Chunk) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto parts = chunk(inputs[0], 4, 1);
        return parts[2];
    }, {a}, device, 0, 0, "Chunk");
}

TEST_P(ShapeOpsParity, CatDim0) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> v = {inputs[0], inputs[1]};
        return cat(v, 0);
    }, {a, b}, device, 0, 0, "Cat_Dim0");
}

TEST_P(ShapeOpsParity, CatDim1) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> v = {inputs[0], inputs[1]};
        return cat(v, 1);
    }, {a, b}, device, 0, 0, "Cat_Dim1");
}

TEST_P(ShapeOpsParity, Repeat) {

    auto a = randn({8, 8}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return repeat(inputs[0], {2, 3});
    }, {a}, device, 0, 0, "Repeat");
}

TEST_P(ShapeOpsParity, Expand) {

    auto a = randn({1, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return expand(inputs[0], {16, 32});
    }, {a}, device, 0, 0, "Expand");
}

// Phase 6-followup #27: gradient parity for shape ops. Backward of these
// is essentially a reverse view operation; backend kernels for the
// metadata-only path are simple but a stride bug would still surface.
TEST_P(ShapeOpsParity, Reshape_GradientParity) {
    auto a = randn({4, 6}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            return reshape(in[0], std::vector<int64_t>{2, 12});
        },
        {a}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "Reshape_Grad");
}

TEST_P(ShapeOpsParity, Transpose_GradientParity) {
    auto a = randn({4, 6}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            return transpose(in[0], 0, 1);
        },
        {a}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "Transpose_Grad");
}

TEST_P(ShapeOpsParity, Permute_GradientParity) {
    auto a = randn({2, 3, 4}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            return permute(in[0], std::vector<int64_t>{2, 0, 1});
        },
        {a}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "Permute_Grad");
}

TEST_P(ShapeOpsParity, Flatten_GradientParity) {
    auto a = randn({2, 3, 4}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable { return flatten(in[0]); },
        {a}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "Flatten_Grad");
}

TEST_P(ShapeOpsParity, Unsqueeze_GradientParity) {
    auto a = randn({4, 6}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            return unsqueeze(in[0], 1);
        },
        {a}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "Unsqueeze_Grad");
}

INSTANTIATE_BACKEND_TESTS(ShapeOpsParity);




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
