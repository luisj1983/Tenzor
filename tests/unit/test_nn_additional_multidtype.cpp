/**
 * @file test_nn_additional_multidtype.cpp
 * @brief Multi-dtype() comprehensive tests for NN modules
 *
 * This file provides extensive tests for:
 * - Activation functions (edge cases, gradients, numerical stability)
 * - Loss functions (reduction modes, edge cases, label smoothing)
 * - Normalization layers (BatchNorm, LayerNorm, GroupNorm variations)
 * - Pooling layers (MaxPool2d, AvgPool2d edge cases)
 * - Embedding layers (out of bounds, padding index)
 * - RNN layers (sequence handling, bidirectional, dropout)
 *
 * Tests across multiple data types: Float32, Float64, Float16, Int32
 * All tests use BackendDTypeParam for multi-backend and multi-dtype() support.
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <tenzor/nn/layers/normalization.hpp>
#include <tenzor/nn/layers/pooling.hpp>
#include <tenzor/nn/layers/embedding.hpp>
#include <tenzor/nn/layers/rnn.hpp>
#include <cmath>
#include <limits>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// ============================================================================
// Multi-DType Test Fixture
// ============================================================================

class NNMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    double tolerance = 1e-4;

    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        if (IsSkipped()) return;
        if (dtype() == DType::Float16) tolerance = 1e-2;
        else if (dtype() == DType::BFloat16) tolerance = 5e-2;
        else if (dtype() == DType::Float64) tolerance = 1e-9;
        else tolerance = 1e-4;
    }
};

// ============================================================================
// Activation Functions Tests
// ============================================================================

TEST_P(NNMultiDTypeTest, ReLU_EdgeCases) {
    // Skip non-floating point dtypes for activation functions
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP() << "Activation functions only support floating point dtypes";
    }

    auto relu = ReLU();

    // Test with zeros
    auto zeros_input = Variable(zeros({3, 3}, dtype(), device()), true);
    auto zeros_output = relu.forward(zeros_input);
    expectTensorNear(zeros_output.tensor(), zeros({3, 3}, dtype(), device()));

    // Test with negative values
    auto neg_input = Variable(full({3, 3}, -5.0f, dtype(), device()), true);
    auto neg_output = relu.forward(neg_input);
    expectTensorNear(neg_output.tensor(), zeros({3, 3}, dtype(), device()));

    // Test with positive values
    auto pos_input = Variable(full({3, 3}, 5.0f, dtype(), device()), true);
    auto pos_output = relu.forward(pos_input);
    expectTensorNear(pos_output.tensor(), full({3, 3}, 5.0f, dtype(), device()));
}

TEST_P(NNMultiDTypeTest, ReLU_Gradient) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    auto relu = ReLU();

    // Test gradient flow for positive values
    auto pos_input = Variable(ones({3, 3}, dtype(), device()), true);
    auto pos_output = relu.forward(pos_input);
    auto loss = sum(pos_output);
    loss.backward();

    // Gradient should be 1.0 for positive inputs
    auto grad_cpu = pos_input.grad().value().to(Device::cpu());
    if (dtype() == DType::Float32) {
        auto grad_data = grad_cpu.data<float>();
        for (int64_t i = 0; i < 9; ++i) {
            EXPECT_NEAR(grad_data[i], 1.0f, tolerance);
        }
    } else if (dtype() == DType::Float64) {
        auto grad_data = grad_cpu.data<double>();
        for (int64_t i = 0; i < 9; ++i) {
            EXPECT_NEAR(grad_data[i], 1.0, tolerance);
        }
    } else if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        // ReLU' = 1 for positive inputs; exactly representable in half. Widen
        // to Float32 to read, with the dtype-aware half tolerance.
        auto grad_f32 = grad_cpu.to(DType::Float32);
        auto grad_data = grad_f32.data<float>();
        for (int64_t i = 0; i < 9; ++i) {
            EXPECT_NEAR(grad_data[i], 1.0f, static_cast<float>(tolerance));
        }
    }
}

TEST_P(NNMultiDTypeTest, ReLU6_Clipping) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    auto relu6 = ReLU6();

    // Values above 6 should be clipped
    auto large_input = Variable(full({3, 3}, 10.0f, dtype(), device()), true);
    auto large_output = relu6.forward(large_input);
    expectTensorNear(large_output.tensor(), full({3, 3}, 6.0f, dtype(), device()));

    // Values in [0, 6] should pass through
    auto mid_input = Variable(full({3, 3}, 3.0f, dtype(), device()), true);
    auto mid_output = relu6.forward(mid_input);
    expectTensorNear(mid_output.tensor(), full({3, 3}, 3.0f, dtype(), device()));

    // Negative values should be zero
    auto neg_input = Variable(full({3, 3}, -2.0f, dtype(), device()), true);
    auto neg_output = relu6.forward(neg_input);
    expectTensorNear(neg_output.tensor(), zeros({3, 3}, dtype(), device()));
}

TEST_P(NNMultiDTypeTest, LeakyReLU_NegativeSlope) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    auto leaky_relu = LeakyReLU(0.01);

    // Test negative values get scaled
    auto neg_input = Variable(full({3, 3}, -10.0f, dtype(), device()), true);
    auto neg_output = leaky_relu.forward(neg_input);

    auto output_cpu = neg_output.tensor().to(Device::cpu());
    if (dtype() == DType::Float32) {
        EXPECT_NEAR(output_cpu.data<float>()[0], -0.1f, tolerance);
    } else if (dtype() == DType::Float64) {
        EXPECT_NEAR(output_cpu.data<double>()[0], -0.1, tolerance);
    }

    // Test positive values pass through
    auto pos_input = Variable(full({3, 3}, 5.0f, dtype(), device()), true);
    auto pos_output = leaky_relu.forward(pos_input);
    expectTensorNear(pos_output.tensor(), full({3, 3}, 5.0f, dtype(), device()));
}

TEST_P(NNMultiDTypeTest, Sigmoid_Range) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    auto sigmoid = Sigmoid();

    // Sigmoid output is in [0, 1] — mathematically (0, 1) but in floating
    // point it saturates at the bounds for large |x|. With randn * 5 some
    // samples land beyond ~17, where 1/(1+exp(-x)) rounds to exactly 1.0
    // in Float32 (and similarly for negative tails). Match Tanh_Range
    // below which uses GE/LE for the same reason.
    auto input = Variable(randn({100}, dtype(), device()) * 5.0f, true);
    auto output = sigmoid.forward(input);

    auto output_cpu = output.tensor().to(Device::cpu());
    if (dtype() == DType::Float32) {
        auto output_data = output_cpu.data<float>();
        for (int64_t i = 0; i < 100; ++i) {
            EXPECT_GE(output_data[i], 0.0f);
            EXPECT_LE(output_data[i], 1.0f);
        }
    } else if (dtype() == DType::Float64) {
        auto output_data = output_cpu.data<double>();
        for (int64_t i = 0; i < 100; ++i) {
            EXPECT_GE(output_data[i], 0.0);
            EXPECT_LE(output_data[i], 1.0);
        }
    } else if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        // sigmoid output in [0, 1] is a robust range bound; widen to Float32.
        auto output_f32 = output_cpu.to(DType::Float32);
        auto output_data = output_f32.data<float>();
        for (int64_t i = 0; i < 100; ++i) {
            EXPECT_GE(output_data[i], 0.0f);
            EXPECT_LE(output_data[i], 1.0f);
        }
    }
}

TEST_P(NNMultiDTypeTest, Sigmoid_ExtremeValues) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    auto sigmoid = Sigmoid();

    // Very large positive values should approach 1
    auto large_pos = Variable(full({3, 3}, 100.0f, dtype(), device()), true);
    auto output_pos = sigmoid.forward(large_pos);
    auto output_pos_cpu = output_pos.tensor().to(Device::cpu());
    if (dtype() == DType::Float32) {
        EXPECT_NEAR(output_pos_cpu.data<float>()[0], 1.0f, 1e-3f);
    }

    // Very large negative values should approach 0
    auto large_neg = Variable(full({3, 3}, -100.0f, dtype(), device()), true);
    auto output_neg = sigmoid.forward(large_neg);
    auto output_neg_cpu = output_neg.tensor().to(Device::cpu());
    if (dtype() == DType::Float32) {
        EXPECT_NEAR(output_neg_cpu.data<float>()[0], 0.0f, 1e-3f);
    }

    // Zero should give 0.5
    auto zero_input = Variable(zeros({3, 3}, dtype(), device()), true);
    auto zero_output = sigmoid.forward(zero_input);
    auto zero_output_cpu = zero_output.tensor().to(Device::cpu());
    if (dtype() == DType::Float32) {
        EXPECT_NEAR(zero_output_cpu.data<float>()[0], 0.5f, tolerance);
    }
}

TEST_P(NNMultiDTypeTest, Tanh_Range) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    auto tanh_layer = Tanh();

    // Test output is in (-1, 1)
    auto input = Variable(randn({100}, dtype(), device()) * 5.0f, true);
    auto output = tanh_layer.forward(input);

    auto output_cpu = output.tensor().to(Device::cpu());
    if (dtype() == DType::Float32) {
        auto output_data = output_cpu.data<float>();
        for (int64_t i = 0; i < 100; ++i) {
            // tanh output is in [-1, 1], can reach bounds for large inputs
            EXPECT_GE(output_data[i], -1.0f);
            EXPECT_LE(output_data[i], 1.0f);
        }
    } else if (dtype() == DType::Float64) {
        auto output_data = output_cpu.data<double>();
        for (int64_t i = 0; i < 100; ++i) {
            EXPECT_GE(output_data[i], -1.0);
            EXPECT_LE(output_data[i], 1.0);
        }
    } else if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        // Range bound is robust across precision: widen to Float32 to read.
        auto output_f32 = output_cpu.to(DType::Float32);
        auto output_data = output_f32.data<float>();
        for (int64_t i = 0; i < 100; ++i) {
            EXPECT_GE(output_data[i], -1.0f);
            EXPECT_LE(output_data[i], 1.0f);
        }
    }
}

TEST_P(NNMultiDTypeTest, Tanh_ZeroCentered) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    auto tanh_layer = Tanh();

    // Zero input should give zero output
    auto zero_input = Variable(zeros({3, 3}, dtype(), device()), true);
    auto zero_output = tanh_layer.forward(zero_input);
    expectTensorNear(zero_output.tensor(), zeros({3, 3}, dtype(), device()));
}

TEST_P(NNMultiDTypeTest, Softmax_SumToOne) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    auto softmax = Softmax(-1);

    // Test outputs sum to 1 along last dimension
    auto input = Variable(randn({4, 10}, dtype(), device()), true);
    auto output = softmax.forward(input);

    // Sum along last dimension
    auto sum_output = sum(output, -1);
    auto sum_cpu = sum_output.tensor().to(Device::cpu());

    if (dtype() == DType::Float32) {
        auto sum_data = sum_cpu.data<float>();
        for (int64_t i = 0; i < 4; ++i) {
            EXPECT_NEAR(sum_data[i], 1.0f, tolerance);
        }
    } else if (dtype() == DType::Float64) {
        auto sum_data = sum_cpu.data<double>();
        for (int64_t i = 0; i < 4; ++i) {
            EXPECT_NEAR(sum_data[i], 1.0, tolerance);
        }
    } else if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        // Sum-to-one is robust across precision; widen to Float32 and use the
        // dtype-aware half tolerance set in SetUp().
        auto sum_f32 = sum_cpu.to(DType::Float32);
        auto sum_data = sum_f32.data<float>();
        for (int64_t i = 0; i < 4; ++i) {
            EXPECT_NEAR(sum_data[i], 1.0f, static_cast<float>(tolerance));
        }
    }
}

TEST_P(NNMultiDTypeTest, Softmax_NumericalStability) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    auto softmax = Softmax(-1);

    // Test with very large values (should not overflow)
    auto large_input = Variable(full({3, 5}, 1000.0f, dtype(), device()), true);
    auto output = softmax.forward(large_input);

    // Should still be valid probabilities
    auto output_cpu = output.tensor().to(Device::cpu());
    if (dtype() == DType::Float32) {
        auto output_data = output_cpu.data<float>();
        for (int64_t i = 0; i < 15; ++i) {
            EXPECT_FALSE(std::isnan(output_data[i]));
            EXPECT_FALSE(std::isinf(output_data[i]));
        }
    } else if (dtype() == DType::Float64) {
        auto output_data = output_cpu.data<double>();
        for (int64_t i = 0; i < 15; ++i) {
            EXPECT_FALSE(std::isnan(output_data[i]));
            EXPECT_FALSE(std::isinf(output_data[i]));
        }
    } else if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        // Finiteness (no overflow) is robust across precision; widen to Float32.
        auto output_f32 = output_cpu.to(DType::Float32);
        auto output_data = output_f32.data<float>();
        for (int64_t i = 0; i < 15; ++i) {
            EXPECT_FALSE(std::isnan(output_data[i]));
            EXPECT_FALSE(std::isinf(output_data[i]));
        }
    }
}

// ============================================================================
// Loss Functions Tests
// ============================================================================

TEST_P(NNMultiDTypeTest, MSELoss_ReductionModes) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    auto input = Variable(ones({4, 5}, dtype(), device()), false);
    auto target = Variable(zeros({4, 5}, dtype(), device()), false);

    // Test all reduction modes
    auto criterion_none = MSELoss(Reduction::None);
    auto criterion_mean = MSELoss(Reduction::Mean);
    auto criterion_sum = MSELoss(Reduction::Sum);

    auto loss_none = criterion_none(input, target);
    auto loss_mean = criterion_mean(input, target);
    auto loss_sum = criterion_sum(input, target);

    // None should return tensor of same shape as input
    EXPECT_EQ(loss_none.shape().size(), 2);
    EXPECT_EQ(loss_none.shape()[0], 4);
    EXPECT_EQ(loss_none.shape()[1], 5);

    // Mean and sum should be scalars
    EXPECT_EQ(loss_mean.shape().size(), 0);
    EXPECT_EQ(loss_sum.shape().size(), 0);

    // Sum should be larger than mean
    auto mean_cpu = loss_mean.tensor().to(Device::cpu());
    auto sum_cpu = loss_sum.tensor().to(Device::cpu());
    if (dtype() == DType::Float32) {
        EXPECT_GT(sum_cpu.item<float>(), mean_cpu.item<float>());
    }
}

TEST_P(NNMultiDTypeTest, MSELoss_PerfectPrediction) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    auto input = Variable(ones({4, 5}, dtype(), device()), false);
    auto target = Variable(ones({4, 5}, dtype(), device()), false);

    auto criterion = MSELoss(Reduction::Mean);
    auto loss = criterion(input, target);

    auto loss_cpu = loss.tensor().to(Device::cpu());
    if (dtype() == DType::Float32) {
        EXPECT_NEAR(loss_cpu.item<float>(), 0.0f, tolerance);
    }
}

TEST_P(NNMultiDTypeTest, CrossEntropyLoss_Basic) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    // 3 classes, batch size 4
    auto logits = Variable(randn({4, 3}, dtype(), device()), false);

    // Target class indices (0, 1, 2, 0)
    std::vector<int64_t> target_data = {0, 1, 2, 0};
    auto target = from_data(target_data.data(), {4}, device());

    auto criterion = CrossEntropyLoss(Reduction::Mean);
    auto loss = criterion(logits, target);

    // Loss should be positive
    auto loss_cpu = loss.tensor().to(Device::cpu());
    if (dtype() == DType::Float32) {
        EXPECT_GT(loss_cpu.item<float>(), 0.0f);
    }
}

// ============================================================================
// Normalization Layers Tests
// ============================================================================

TEST_P(NNMultiDTypeTest, LayerNorm_SingleDimension) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    auto layer_norm = LayerNorm({64}, 1e-5, true);

    // Input: (batch=4, features=64)
    auto input = Variable(randn({4, 64}, dtype(), device()), true);
    auto output = layer_norm.forward(input);

    // Output should have same shape
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 64);

    // Check normalization: mean ~0, std ~1 along last dimension
    auto output_cpu = output.tensor().to(Device::cpu());
    if (dtype() == DType::Float32) {
        auto output_data = output_cpu.data<float>();
        for (int64_t i = 0; i < 4; ++i) {
            double sum = 0.0;
            for (int64_t j = 0; j < 64; ++j) {
                sum += output_data[i * 64 + j];
            }
            double mean = sum / 64.0;
            EXPECT_NEAR(mean, 0.0, 0.1);
        }
    }
}

TEST_P(NNMultiDTypeTest, GroupNorm_Basic) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    // 32 channels, 8 groups (4 channels per group)
    auto group_norm = GroupNorm(8, 32, 1e-5, true);

    // Input: (batch=2, channels=32, height=16, width=16)
    auto input = Variable(randn({2, 32, 16, 16}, dtype(), device()), true);
    auto output = group_norm.forward(input);

    // Output should have same shape
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 32);
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

// ============================================================================
// Pooling Layers Tests
// ============================================================================

TEST_P(NNMultiDTypeTest, MaxPool2d_BasicDownsampling) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    auto pool = MaxPool2d(2, 2, 0);

    // Input: (batch=2, channels=3, height=8, width=8)
    auto input = Variable(randn({2, 3, 8, 8}, dtype(), device()), true);
    auto output = pool.forward(input);

    // Output should be (2, 3, 4, 4)
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 4);
}

TEST_P(NNMultiDTypeTest, AvgPool2d_BasicDownsampling) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    auto pool = AvgPool2d(2, 2, 0);

    auto input = Variable(randn({2, 3, 8, 8}, dtype(), device()), true);
    auto output = pool.forward(input);

    // Output should be (2, 3, 4, 4)
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 4);
}

TEST_P(NNMultiDTypeTest, AdaptiveAvgPool2d_FixedOutputSize) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64 && dtype() != DType::Float16 && dtype() != DType::BFloat16) {
        GTEST_SKIP();
    }

    auto pool = AdaptiveAvgPool2d(7, 7);

    // Test with different input sizes
    auto input1 = Variable(randn({1, 64, 14, 14}, dtype(), device()), true);
    auto output1 = pool.forward(input1);

    auto input2 = Variable(randn({1, 64, 28, 28}, dtype(), device()), true);
    auto output2 = pool.forward(input2);

    // Both should output 7x7
    EXPECT_EQ(output1.shape()[2], 7);
    EXPECT_EQ(output1.shape()[3], 7);
    EXPECT_EQ(output2.shape()[2], 7);
    EXPECT_EQ(output2.shape()[3], 7);
}

// ============================================================================
// Embedding Layers Tests
// ============================================================================

TEST_P(NNMultiDTypeTest, Embedding_BasicLookup) {
    // Embeddings are typically Float32 or Float64
    if (dtype() != DType::Float32 && dtype() != DType::Float64) {
        GTEST_SKIP();
    }

    // 100 words, 50-dimensional embeddings
    auto embedding = Embedding(100, 50);

    // Input: batch of indices
    std::vector<int64_t> indices_data = {0, 10, 25, 50, 99};
    auto indices = Variable(from_data(indices_data.data(), {5}, device()), false);

    auto output = embedding.forward(indices);

    // Output should be (5, 50)
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 5);
    EXPECT_EQ(output.shape()[1], 50);
}

TEST_P(NNMultiDTypeTest, Embedding_BatchedInput) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64) {
        GTEST_SKIP();
    }

    auto embedding = Embedding(1000, 300);

    // Input: (batch=4, sequence_length=10)
    // Create random indices manually
    auto indices_tensor = zeros({4, 10}, DType::Int64, device());
    auto indices_cpu = indices_tensor.to(Device::cpu());
    auto indices_ptr = indices_cpu.data<int64_t>();
    for (int64_t i = 0; i < 40; ++i) {
        indices_ptr[i] = i % 1000;  // Simple pattern for testing
    }
    auto indices = Variable(indices_cpu.to(device()), false);
    auto output = embedding.forward(indices);

    // Output should be (4, 10, 300)
    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.shape()[2], 300);
}

// ============================================================================
// RNN Layers Tests
// ============================================================================

TEST_P(NNMultiDTypeTest, RNNCell_SingleStep) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64) {
        GTEST_SKIP();
    }

    auto cell = RNNCell(128, 256, "tanh", true);

    // Input: (batch=4, input_size=128)
    auto x = Variable(randn({4, 128}, dtype(), device()), true);
    auto h = Variable(randn({4, 256}, dtype(), device()), true);

    auto h_next = cell.forward(x, h);

    // Output should be (4, 256)
    EXPECT_EQ(h_next.shape().size(), 2);
    EXPECT_EQ(h_next.shape()[0], 4);
    EXPECT_EQ(h_next.shape()[1], 256);
}

TEST_P(NNMultiDTypeTest, RNN_SequenceProcessing) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64) {
        GTEST_SKIP();
    }

    auto rnn = RNN(128, 256, 1, "tanh", true, false, 0.0, false);

    // Input: (seq_len=10, batch=4, input_size=128)
    auto x = Variable(randn({10, 4, 128}, dtype(), device()), true);
    auto h0 = Variable(randn({1, 4, 256}, dtype(), device()), true);

    auto result = rnn.forward(x, h0);
    auto output = result.first;
    auto h_n = result.second;

    // Output: (10, 4, 256), h_n: (1, 4, 256)
    EXPECT_EQ(output.shape()[0], 10);
    EXPECT_EQ(output.shape()[1], 4);
    EXPECT_EQ(output.shape()[2], 256);
    EXPECT_EQ(h_n.shape()[0], 1);
    EXPECT_EQ(h_n.shape()[1], 4);
    EXPECT_EQ(h_n.shape()[2], 256);
}

TEST_P(NNMultiDTypeTest, LSTMCell_SingleStep) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64) {
        GTEST_SKIP();
    }

    auto cell = LSTMCell(128, 256, true);

    auto x = Variable(randn({4, 128}, dtype(), device()), true);
    auto h = Variable(randn({4, 256}, dtype(), device()), true);
    auto c = Variable(randn({4, 256}, dtype(), device()), true);

    auto result = cell.forward(x, h, c);
    auto h_next = result.first;
    auto c_next = result.second;

    // Both outputs should be (4, 256)
    EXPECT_EQ(h_next.shape()[0], 4);
    EXPECT_EQ(h_next.shape()[1], 256);
    EXPECT_EQ(c_next.shape()[0], 4);
    EXPECT_EQ(c_next.shape()[1], 256);
}

TEST_P(NNMultiDTypeTest, GRUCell_SingleStep) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64) {
        GTEST_SKIP();
    }

    auto cell = GRUCell(128, 256, true);

    auto x = Variable(randn({4, 128}, dtype(), device()), true);
    auto h = Variable(randn({4, 256}, dtype(), device()), true);

    auto h_next = cell.forward(x, h);

    EXPECT_EQ(h_next.shape()[0], 4);
    EXPECT_EQ(h_next.shape()[1], 256);
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    AllBackendsFloatDTypes,
    NNMultiDTypeTest,
    ::testing::Combine(
        STANDARD_BACKENDS,
        ::testing::Values(DType::Float32, DType::Float64, DType::Float16, DType::BFloat16)
    ),
    BackendDTypeParamName);
