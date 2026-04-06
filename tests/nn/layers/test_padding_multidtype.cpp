/**
 * @file test_padding_multidtype.cpp
 * @brief Multi-dtype tests for all 9 padding layers
 *
 * Tests ConstantPad1d, ConstantPad2d, ConstantPad3d,
 *       ReflectionPad1d, ReflectionPad2d,
 *       ReplicationPad1d, ReplicationPad2d, ReplicationPad3d,
 *       ZeroPad2d
 * with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// ============================================================================
// Test Fixture
// ============================================================================

class PaddingMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// ConstantPad1d
// ============================================================================

TEST_P(PaddingMultiDTypeTest, ConstantPad1d_OutputShape) {
    auto pad = ConstantPad1d(2, 3, 1.0);
    convert_model(pad);

    Variable input = createInput({2, 3, 10}, true);
    auto output = pad.forward(input);

    // W: 10 + 2 + 3 = 15
    expectShape(output.tensor(), {2, 3, 15});
    expectDType(output.tensor());
}

TEST_P(PaddingMultiDTypeTest, ConstantPad1d_PaddingValues) {
    double fill_val = 5.0;
    auto pad = ConstantPad1d(2, 3, fill_val);
    convert_model(pad);

    // Use zeros so we can distinguish padded regions clearly
    auto input_tensor = createZeros({1, 1, 4});
    // Set interior to ones
    auto ones_t = createOnes({1, 1, 4});
    auto input = Variable(ones_t, false);
    auto output = pad.forward(input);

    // Output shape: (1, 1, 9)
    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* data = out_cpu.data<float>();

    // Left padding (indices 0, 1) should be fill_val
    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(data[i], static_cast<float>(fill_val), atol())
            << "Left pad index " << i;
    }
    // Interior (indices 2..5) should be 1.0
    for (int i = 2; i < 6; ++i) {
        EXPECT_NEAR(data[i], 1.0f, atol())
            << "Interior index " << i;
    }
    // Right padding (indices 6, 7, 8) should be fill_val
    for (int i = 6; i < 9; ++i) {
        EXPECT_NEAR(data[i], static_cast<float>(fill_val), atol())
            << "Right pad index " << i;
    }
}

TEST_P(PaddingMultiDTypeTest, ConstantPad1d_GradientFlow) {
    auto pad = ConstantPad1d(static_cast<int64_t>(2), static_cast<int64_t>(3), 0.0);
    convert_model(pad);

    Variable input = createInput({2, 3, 10}, true);
    auto output = pad.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 3);
    EXPECT_EQ(grad.shape()[2], 10);
    EXPECT_EQ(grad.dtype(), dtype());
}

// ============================================================================
// ConstantPad2d
// ============================================================================

TEST_P(PaddingMultiDTypeTest, ConstantPad2d_OutputShape) {
    auto pad = ConstantPad2d(1, 2, 3, 4, 0.0);
    convert_model(pad);

    Variable input = createInput({2, 3, 8, 8}, true);
    auto output = pad.forward(input);

    // H: 8 + 3 + 4 = 15, W: 8 + 1 + 2 = 11
    expectShape(output.tensor(), {2, 3, 15, 11});
    expectDType(output.tensor());
}

TEST_P(PaddingMultiDTypeTest, ConstantPad2d_PaddingValues) {
    double fill_val = 7.0;
    // Symmetric padding of 1
    auto pad = ConstantPad2d(1, fill_val);
    convert_model(pad);

    auto input = Variable(createOnes({1, 1, 2, 2}), false);
    auto output = pad.forward(input);

    // Output shape: (1, 1, 4, 4)
    expectShape(output.tensor(), {1, 1, 4, 4});

    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* data = out_cpu.data<float>();

    // Top-left corner should be fill_val
    EXPECT_NEAR(data[0], static_cast<float>(fill_val), atol());
    // Interior (row 1, col 1) should be 1.0
    // Row 1, col 1 -> index 1*4 + 1 = 5
    EXPECT_NEAR(data[5], 1.0f, atol());
    // Bottom-right corner should be fill_val
    EXPECT_NEAR(data[15], static_cast<float>(fill_val), atol());
}

TEST_P(PaddingMultiDTypeTest, ConstantPad2d_GradientFlow) {
    auto pad = ConstantPad2d(1, 0.0);
    convert_model(pad);

    Variable input = createInput({2, 3, 8, 8}, true);
    auto output = pad.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 3);
    EXPECT_EQ(grad.shape()[2], 8);
    EXPECT_EQ(grad.shape()[3], 8);
    EXPECT_EQ(grad.dtype(), dtype());
}

// ============================================================================
// ConstantPad3d
// ============================================================================

TEST_P(PaddingMultiDTypeTest, ConstantPad3d_OutputShape) {
    // Symmetric padding of 1 on all 6 sides
    auto pad = ConstantPad3d(1, 0.0);
    convert_model(pad);

    Variable input = createInput({2, 3, 4, 4, 4}, true);
    auto output = pad.forward(input);

    // D: 4+2=6, H: 4+2=6, W: 4+2=6
    expectShape(output.tensor(), {2, 3, 6, 6, 6});
    expectDType(output.tensor());
}

TEST_P(PaddingMultiDTypeTest, ConstantPad3d_PaddingValues) {
    double fill_val = 3.0;
    auto pad = ConstantPad3d(1, fill_val);
    convert_model(pad);

    auto input = Variable(createOnes({1, 1, 2, 2, 2}), false);
    auto output = pad.forward(input);

    // Output: (1, 1, 4, 4, 4)
    expectShape(output.tensor(), {1, 1, 4, 4, 4});

    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* data = out_cpu.data<float>();

    // Corner (0,0,0) should be fill_val
    EXPECT_NEAR(data[0], static_cast<float>(fill_val), atol());
    // Interior (1,1,1) -> index 1*16 + 1*4 + 1 = 21
    EXPECT_NEAR(data[21], 1.0f, atol());
}

TEST_P(PaddingMultiDTypeTest, ConstantPad3d_AsymmetricShape) {
    // left=1, right=2, top=0, bottom=1, front=2, back=0
    auto pad = ConstantPad3d({1, 2, 0, 1, 2, 0}, 0.0);
    convert_model(pad);

    Variable input = createInput({2, 3, 4, 4, 4}, true);
    auto output = pad.forward(input);

    // D: 4+2+0=6, H: 4+0+1=5, W: 4+1+2=7
    expectShape(output.tensor(), {2, 3, 6, 5, 7});
    expectDType(output.tensor());
}

TEST_P(PaddingMultiDTypeTest, ConstantPad3d_GradientFlow) {
    auto pad = ConstantPad3d(1, 0.0);
    convert_model(pad);

    Variable input = createInput({2, 3, 4, 4, 4}, true);
    auto output = pad.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 3);
    EXPECT_EQ(grad.shape()[2], 4);
    EXPECT_EQ(grad.shape()[3], 4);
    EXPECT_EQ(grad.shape()[4], 4);
    EXPECT_EQ(grad.dtype(), dtype());
}

// ============================================================================
// ReflectionPad1d
// ============================================================================

TEST_P(PaddingMultiDTypeTest, ReflectionPad1d_OutputShape) {
    auto pad = ReflectionPad1d(2, 3);
    convert_model(pad);

    Variable input = createInput({2, 3, 10}, true);
    auto output = pad.forward(input);

    // W: 10 + 2 + 3 = 15
    expectShape(output.tensor(), {2, 3, 15});
    expectDType(output.tensor());
}

TEST_P(PaddingMultiDTypeTest, ReflectionPad1d_PaddingValues) {
    auto pad = ReflectionPad1d(2, 2);
    convert_model(pad);

    // Create known input: [0, 1, 2, 3, 4]
    auto input_cpu = tenzor::arange(0.0f, 5.0f, 1.0f, DType::Float32, Device::cpu())
                         .reshape({1, 1, 5});
    if (dtype() != DType::Float32) {
        input_cpu = input_cpu.to(dtype());
    }
    auto input_dev = (device() == Device::cpu()) ? input_cpu : input_cpu.to(device());
    auto input = Variable(input_dev, false);

    auto output = pad.forward(input);

    // Reflected: [2, 1, 0, 1, 2, 3, 4, 3, 2]
    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* data = out_cpu.data<float>();

    float expected[] = {2.0f, 1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 2.0f};
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(data[i], expected[i], atol()) << "Index " << i;
    }
}

TEST_P(PaddingMultiDTypeTest, ReflectionPad1d_GradientFlow) {
    auto pad = ReflectionPad1d(2);
    convert_model(pad);

    Variable input = createInput({2, 3, 10}, true);
    auto output = pad.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 3);
    EXPECT_EQ(grad.shape()[2], 10);
    EXPECT_EQ(grad.dtype(), dtype());
}

// ============================================================================
// ReflectionPad2d
// ============================================================================

TEST_P(PaddingMultiDTypeTest, ReflectionPad2d_OutputShape) {
    auto pad = ReflectionPad2d(1, 2, 3, 2);
    convert_model(pad);

    Variable input = createInput({2, 3, 8, 8}, true);
    auto output = pad.forward(input);

    // H: 8 + 3 + 2 = 13, W: 8 + 1 + 2 = 11
    expectShape(output.tensor(), {2, 3, 13, 11});
    expectDType(output.tensor());
}

TEST_P(PaddingMultiDTypeTest, ReflectionPad2d_PaddingValues) {
    auto pad = ReflectionPad2d(1);
    convert_model(pad);

    // 2x2 input with known values
    // [[1, 2],
    //  [3, 4]]
    auto input_cpu = tenzor::arange(1.0f, 5.0f, 1.0f, DType::Float32, Device::cpu())
                         .reshape({1, 1, 2, 2});
    if (dtype() != DType::Float32) {
        input_cpu = input_cpu.to(dtype());
    }
    auto input_dev = (device() == Device::cpu()) ? input_cpu : input_cpu.to(device());
    auto input = Variable(input_dev, false);

    auto output = pad.forward(input);

    // Output: 4x4, reflection of 2x2 with pad=1
    // [[4, 3, 4, 3],
    //  [2, 1, 2, 1],
    //  [4, 3, 4, 3],
    //  [2, 1, 2, 1]]
    expectShape(output.tensor(), {1, 1, 4, 4});

    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* data = out_cpu.data<float>();

    float expected[] = {4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1};
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(data[i], expected[i], atol()) << "Index " << i;
    }
}

TEST_P(PaddingMultiDTypeTest, ReflectionPad2d_GradientFlow) {
    auto pad = ReflectionPad2d(2);
    convert_model(pad);

    Variable input = createInput({2, 3, 8, 8}, true);
    auto output = pad.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 3);
    EXPECT_EQ(grad.shape()[2], 8);
    EXPECT_EQ(grad.shape()[3], 8);
    EXPECT_EQ(grad.dtype(), dtype());
}

// ============================================================================
// ReplicationPad1d
// ============================================================================

TEST_P(PaddingMultiDTypeTest, ReplicationPad1d_OutputShape) {
    auto pad = ReplicationPad1d(2, 3);
    convert_model(pad);

    Variable input = createInput({2, 3, 10}, true);
    auto output = pad.forward(input);

    // W: 10 + 2 + 3 = 15
    expectShape(output.tensor(), {2, 3, 15});
    expectDType(output.tensor());
}

TEST_P(PaddingMultiDTypeTest, ReplicationPad1d_PaddingValues) {
    auto pad = ReplicationPad1d(2, 3);
    convert_model(pad);

    // Input: [0, 1, 2, 3]
    auto input_cpu = tenzor::arange(0.0f, 4.0f, 1.0f, DType::Float32, Device::cpu())
                         .reshape({1, 1, 4});
    if (dtype() != DType::Float32) {
        input_cpu = input_cpu.to(dtype());
    }
    auto input_dev = (device() == Device::cpu()) ? input_cpu : input_cpu.to(device());
    auto input = Variable(input_dev, false);

    auto output = pad.forward(input);

    // Replicated: [0, 0, 0, 1, 2, 3, 3, 3, 3]
    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* data = out_cpu.data<float>();

    float expected[] = {0, 0, 0, 1, 2, 3, 3, 3, 3};
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(data[i], expected[i], atol()) << "Index " << i;
    }
}

TEST_P(PaddingMultiDTypeTest, ReplicationPad1d_GradientFlow) {
    auto pad = ReplicationPad1d(2);
    convert_model(pad);

    Variable input = createInput({2, 3, 10}, true);
    auto output = pad.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 3);
    EXPECT_EQ(grad.shape()[2], 10);
    EXPECT_EQ(grad.dtype(), dtype());
}

// ============================================================================
// ReplicationPad2d
// ============================================================================

TEST_P(PaddingMultiDTypeTest, ReplicationPad2d_OutputShape) {
    auto pad = ReplicationPad2d(1, 2, 3, 4);
    convert_model(pad);

    Variable input = createInput({2, 3, 8, 8}, true);
    auto output = pad.forward(input);

    // H: 8 + 3 + 4 = 15, W: 8 + 1 + 2 = 11
    expectShape(output.tensor(), {2, 3, 15, 11});
    expectDType(output.tensor());
}

TEST_P(PaddingMultiDTypeTest, ReplicationPad2d_PaddingValues) {
    auto pad = ReplicationPad2d(1);
    convert_model(pad);

    // 2x2 input: [[1, 2], [3, 4]]
    auto input_cpu = tenzor::arange(1.0f, 5.0f, 1.0f, DType::Float32, Device::cpu())
                         .reshape({1, 1, 2, 2});
    if (dtype() != DType::Float32) {
        input_cpu = input_cpu.to(dtype());
    }
    auto input_dev = (device() == Device::cpu()) ? input_cpu : input_cpu.to(device());
    auto input = Variable(input_dev, false);

    auto output = pad.forward(input);

    // Replicated edges:
    // [[1, 1, 2, 2],
    //  [1, 1, 2, 2],
    //  [3, 3, 4, 4],
    //  [3, 3, 4, 4]]
    expectShape(output.tensor(), {1, 1, 4, 4});

    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* data = out_cpu.data<float>();

    float expected[] = {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4};
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(data[i], expected[i], atol()) << "Index " << i;
    }
}

TEST_P(PaddingMultiDTypeTest, ReplicationPad2d_GradientFlow) {
    auto pad = ReplicationPad2d(2);
    convert_model(pad);

    Variable input = createInput({2, 3, 8, 8}, true);
    auto output = pad.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 3);
    EXPECT_EQ(grad.shape()[2], 8);
    EXPECT_EQ(grad.shape()[3], 8);
    EXPECT_EQ(grad.dtype(), dtype());
}

// ============================================================================
// ReplicationPad3d
// ============================================================================

TEST_P(PaddingMultiDTypeTest, ReplicationPad3d_OutputShape) {
    auto pad = ReplicationPad3d(1);
    convert_model(pad);

    Variable input = createInput({2, 3, 4, 4, 4}, true);
    auto output = pad.forward(input);

    // D: 4+2=6, H: 4+2=6, W: 4+2=6
    expectShape(output.tensor(), {2, 3, 6, 6, 6});
    expectDType(output.tensor());
}

TEST_P(PaddingMultiDTypeTest, ReplicationPad3d_PaddingValues) {
    auto pad = ReplicationPad3d(1);
    convert_model(pad);

    auto input = Variable(createOnes({1, 1, 2, 2, 2}), false);
    auto output = pad.forward(input);

    // All values should be 1.0 since edge replication of all-ones is all-ones
    expectShape(output.tensor(), {1, 1, 4, 4, 4});

    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* data = out_cpu.data<float>();
    for (int i = 0; i < 64; ++i) {
        EXPECT_NEAR(data[i], 1.0f, atol()) << "Index " << i;
    }
}

TEST_P(PaddingMultiDTypeTest, ReplicationPad3d_AsymmetricShape) {
    // left=1, right=2, top=0, bottom=1, front=2, back=0
    auto pad = ReplicationPad3d({1, 2, 0, 1, 2, 0});
    convert_model(pad);

    Variable input = createInput({2, 3, 4, 4, 4}, true);
    auto output = pad.forward(input);

    // D: 4+2+0=6, H: 4+0+1=5, W: 4+1+2=7
    expectShape(output.tensor(), {2, 3, 6, 5, 7});
    expectDType(output.tensor());
}

TEST_P(PaddingMultiDTypeTest, ReplicationPad3d_GradientFlow) {
    auto pad = ReplicationPad3d(1);
    convert_model(pad);

    Variable input = createInput({2, 3, 4, 4, 4}, true);
    auto output = pad.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 3);
    EXPECT_EQ(grad.shape()[2], 4);
    EXPECT_EQ(grad.shape()[3], 4);
    EXPECT_EQ(grad.shape()[4], 4);
    EXPECT_EQ(grad.dtype(), dtype());
}

// ============================================================================
// ZeroPad2d
// ============================================================================

TEST_P(PaddingMultiDTypeTest, ZeroPad2d_OutputShape) {
    auto pad = ZeroPad2d(1, 2, 3, 4);
    convert_model(pad);

    Variable input = createInput({2, 3, 8, 8}, true);
    auto output = pad.forward(input);

    // H: 8 + 3 + 4 = 15, W: 8 + 1 + 2 = 11
    expectShape(output.tensor(), {2, 3, 15, 11});
    expectDType(output.tensor());
}

TEST_P(PaddingMultiDTypeTest, ZeroPad2d_PaddingValues) {
    auto pad = ZeroPad2d(1);
    convert_model(pad);

    auto input = Variable(createOnes({1, 1, 2, 2}), false);
    auto output = pad.forward(input);

    // Output: (1, 1, 4, 4)
    expectShape(output.tensor(), {1, 1, 4, 4});

    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* data = out_cpu.data<float>();

    // Row 0: all zeros (top padding)
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(data[i], 0.0f, atol()) << "Top pad index " << i;
    }
    // Row 1: [0, 1, 1, 0]
    EXPECT_NEAR(data[4], 0.0f, atol());
    EXPECT_NEAR(data[5], 1.0f, atol());
    EXPECT_NEAR(data[6], 1.0f, atol());
    EXPECT_NEAR(data[7], 0.0f, atol());
    // Row 2: [0, 1, 1, 0]
    EXPECT_NEAR(data[8], 0.0f, atol());
    EXPECT_NEAR(data[9], 1.0f, atol());
    EXPECT_NEAR(data[10], 1.0f, atol());
    EXPECT_NEAR(data[11], 0.0f, atol());
    // Row 3: all zeros (bottom padding)
    for (int i = 12; i < 16; ++i) {
        EXPECT_NEAR(data[i], 0.0f, atol()) << "Bottom pad index " << i;
    }
}

TEST_P(PaddingMultiDTypeTest, ZeroPad2d_GradientFlow) {
    auto pad = ZeroPad2d(2);
    convert_model(pad);

    Variable input = createInput({2, 3, 8, 8}, true);
    auto output = pad.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 3);
    EXPECT_EQ(grad.shape()[2], 8);
    EXPECT_EQ(grad.shape()[3], 8);
    EXPECT_EQ(grad.dtype(), dtype());
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(PaddingMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Padding Layers Tested: 9
 *   ConstantPad1d, ConstantPad2d, ConstantPad3d,
 *   ReflectionPad1d, ReflectionPad2d,
 *   ReplicationPad1d, ReplicationPad2d, ReplicationPad3d,
 *   ZeroPad2d
 *
 * Tests per layer: 3 (OutputShape, PaddingValues, GradientFlow)
 *   + 1 extra AsymmetricShape for ConstantPad3d and ReplicationPad3d
 *
 * Total Test Cases: 29
 * DTypes Tested: Float32, Float64, Float16
 * Backends Tested: CPU, CUDA, OneAPI, Vulkan, ROCm
 * Total Scenarios: 29 tests x 3 dtypes x 5 backends = 435 test scenarios
 */
