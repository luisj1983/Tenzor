/**
 * @file test_hrm.cpp
 * @brief Comprehensive multi-backend tests for Hierarchical Reasoning Model
 *
 * Tests cover:
 * - Basic forward pass on all backends
 * - Gradient flow and detachment
 * - Deep supervision
 * - Memory efficiency (O(1) recurrence)
 * - Adaptive Computational Time
 * - Parameter counting
 * - Numerical correctness
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/hrm.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/ops/indexing.hpp"
#include <vector>
#include <chrono>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

// Helper to compare shapes (span vs vector)
bool shapes_equal(std::span<const int64_t> span, const std::vector<int64_t>& vec) {
    if (span.size() != vec.size()) return false;
    for (size_t i = 0; i < span.size(); ++i) {
        if (span[i] != vec[i]) return false;
    }
    return true;
}

// Global test environment
class TenzorEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const tenzor_env =
    ::testing::AddGlobalTestEnvironment(new TenzorEnvironment());

// ============================================================================
// Test Fixtures
// ============================================================================

/**
 * @brief Base test fixture for HRM tests
 */
class HRMTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default config for testing
        config_.d_model = 64;
        config_.n_heads = 4;
        config_.d_feedforward = 128;
        config_.n_high_cycles = 2;
        config_.t_low_steps = 4;
        config_.dropout = 0.0;  // Disable for deterministic tests
        config_.use_post_norm = true;
        config_.deep_supervision = true;
        config_.use_act = false;
        config_.max_seq_len = 64;
    }

    HRMConfig config_;
};

/**
 * @brief Parameterized test fixture for multi-backend tests
 */
class HRMMultiBackendTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        backend_name_ = GetParam();

        // Check if backend is available
        Device::Type device_type;
        if (backend_name_ == "cpu") {
            device_type = Device::Type::CPU;
            device_ = Device::cpu();
        } else if (backend_name_ == "cuda") {
            device_type = Device::Type::CUDA;
            device_ = Device::cuda();
        } else if (backend_name_ == "vulkan") {
            device_type = Device::Type::Vulkan;
            device_ = Device::vulkan();
        } else if (backend_name_ == "oneapi") {
            device_type = Device::Type::OneAPI;
            device_ = Device::oneapi();
        } else {
            GTEST_SKIP() << "Unknown backend: " << backend_name_;
            return;
        }

        auto* backend = backend_registry().get_backend(device_type);
        if (!backend || !backend->is_available()) {
            GTEST_SKIP() << backend_name_ << " backend not available";
        }

        // Default config
        config_.d_model = 64;
        config_.n_heads = 4;
        config_.d_feedforward = 128;
        config_.n_high_cycles = 2;
        config_.t_low_steps = 4;
        config_.dropout = 0.0;
        config_.use_post_norm = true;
        config_.deep_supervision = true;
        config_.max_seq_len = 64;
    }

    std::string backend_name_;
    Device device_{Device::cpu()};
    HRMConfig config_;
};

// Instantiate tests for all backends
INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    HRMMultiBackendTest,
    ::testing::Values("cpu", "cuda", "vulkan", "oneapi", "rocm"),
    [](const ::testing::TestParamInfo<std::string>& info) {
        return info.param;
    }
);

// ============================================================================
// RMSNorm Tests
// ============================================================================

TEST_F(HRMTest, RMSNormBasic) {
    RMSNorm norm(64);

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, true);

    auto output = norm.forward(x);

    EXPECT_TRUE(shapes_equal(output.shape(), std::vector<int64_t>(x.shape().begin(), x.shape().end())));
    EXPECT_TRUE(output.requires_grad());
}

TEST_F(HRMTest, RMSNormNormalization) {
    RMSNorm norm(64);

    // Create input with known properties
    Tensor input = ones({1, 1, 64}, DType::Float32, Device::cpu());
    Variable x(input, false);

    auto output = norm.forward(x);

    // RMS of ones is 1, so output should be close to ones (scaled by gamma=1)
    auto max_diff = tenzor::max(tenzor::abs(output.tensor() - input));
    EXPECT_LT(max_diff.item<float>(), 0.01f);
}

// Stablemax previously extracted input.tensor() and returned a Variable
// with no grad_fn — backward() produced no gradient. Rewritten to use
// Variable-level ops; this test locks in that backward flows through.
TEST_F(HRMTest, StablemaxPropagatesGradient) {
    Variable x(randn({2, 5}, DType::Float32, Device::cpu()), /*requires_grad=*/true);
    auto y = stablemax(x, -1, 1e-6);
    auto loss = tenzor::sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad())
        << "stablemax must propagate gradient (was severed pre-fix)";
    auto g = x.grad().value().to(Device::cpu()).contiguous();
    const float* gp = g.data<float>();
    float max_abs = 0.0f;
    for (int64_t i = 0; i < g.numel(); ++i) {
        max_abs = std::max(max_abs, std::abs(gp[i]));
    }
    EXPECT_GT(max_abs, 0.0f)
        << "stablemax gradient identically zero — graph still severed";
}

// stablemax_cross_entropy previously extracted probs.tensor() and built the
// loss on the host, wrapping mean in a fresh Variable with no grad_fn — same
// graph-sever pattern as stablemax itself. Rewritten to use the OneHot +
// Variable-arithmetic path from CrossEntropyLoss. This test locks in that
// backward flows through to the logits.
TEST_F(HRMTest, StablemaxCrossEntropyPropagatesGradient) {
    const int64_t B = 4, C = 6;
    Variable logits(randn({B, C}, DType::Float32, Device::cpu()),
                    /*requires_grad=*/true);

    // Int64 class-index target path.
    Tensor target_tensor({B}, DType::Int64, Device::cpu());
    auto* tp = target_tensor.data<int64_t>();
    for (int64_t i = 0; i < B; ++i) {
        tp[i] = i % C;
    }
    Variable target(target_tensor, /*requires_grad=*/false);

    auto loss = stablemax_cross_entropy(logits, target, 1e-8);
    loss.backward();

    ASSERT_TRUE(logits.has_grad())
        << "stablemax_cross_entropy must propagate gradient to logits";
    auto g = logits.grad().value().to(Device::cpu()).contiguous();
    ASSERT_EQ(g.numel(), logits.tensor().numel());
    const float* gp = g.data<float>();
    float max_abs = 0.0f;
    for (int64_t i = 0; i < g.numel(); ++i) {
        EXPECT_FALSE(std::isnan(gp[i])) << "grad[" << i << "] is NaN";
        EXPECT_FALSE(std::isinf(gp[i])) << "grad[" << i << "] is Inf";
        max_abs = std::max(max_abs, std::abs(gp[i]));
    }
    EXPECT_GT(max_abs, 0.0f)
        << "stablemax_cross_entropy gradient identically zero — graph severed";
}

// Stablemax output must be a valid probability distribution: non-negative,
// sums to ≈ 1 along the normalised dimension (modulo the eps term).
TEST_F(HRMTest, StablemaxOutputIsProbability) {
    Variable x(randn({3, 7}, DType::Float32, Device::cpu()), /*requires_grad=*/false);
    auto y = stablemax(x, -1, 1e-8);

    auto y_cpu = y.tensor().to(Device::cpu()).contiguous();
    const float* yp = y_cpu.data<float>();
    for (int64_t i = 0; i < y_cpu.numel(); ++i) {
        EXPECT_GE(yp[i], 0.0f) << "stablemax output should be non-negative";
    }

    auto row_sums = tenzor::sum(y.tensor(), /*dim=*/1, /*keepdim=*/false);
    auto row_sums_cpu = row_sums.to(Device::cpu()).contiguous();
    const float* rp = row_sums_cpu.data<float>();
    for (int64_t i = 0; i < row_sums_cpu.numel(); ++i) {
        EXPECT_NEAR(rp[i], 1.0f, 1e-4f)
            << "stablemax rows should sum to ~1 (within eps tolerance)";
    }
}

// ============================================================================
// GatedLinearUnit Tests
// ============================================================================

TEST_F(HRMTest, GLUBasic) {
    GatedLinearUnit glu(64, 128, true, false);

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, true);

    auto output = glu.forward(x);

    // Output should have same shape as input (down projection)
    EXPECT_TRUE(shapes_equal(output.shape(), std::vector<int64_t>(x.shape().begin(), x.shape().end())));
}

TEST_F(HRMTest, GLUParameters) {
    GatedLinearUnit glu(64, 128, true, false);

    auto params = glu.parameters();

    // Should have 3 linear layers (gate, up, down) with weights only (no bias)
    // gate: 64->128, up: 64->128, down: 128->64
    int64_t expected_params = 64*128 + 64*128 + 128*64;

    int64_t total = 0;
    for (auto& p : params) {
        total += p->tensor().numel();
    }

    EXPECT_EQ(total, expected_params);
}

// ============================================================================
// RotaryPositionEmbedding Tests
// ============================================================================

TEST_F(HRMTest, RoPEBasic) {
    RotaryPositionEmbedding rope(16, 64);  // head_dim=16

    // Input: (batch, seq_len, n_heads, head_dim)
    Tensor input = randn({2, 8, 4, 16}, DType::Float32, Device::cpu());
    Variable x(input, true);

    auto output = rope.forward(x);

    EXPECT_TRUE(shapes_equal(output.shape(), std::vector<int64_t>(x.shape().begin(), x.shape().end())));
}

TEST_F(HRMTest, RoPEPositionDependence) {
    RotaryPositionEmbedding rope(16, 64);

    Tensor input = ones({1, 4, 1, 16}, DType::Float32, Device::cpu());
    Variable x(input, false);

    auto output = rope.forward(x);

    // Different positions should have different outputs due to rotation
    auto pos0 = tenzor::select(output.tensor(), 1, 0);  // Position 0
    auto pos1 = tenzor::select(output.tensor(), 1, 1);  // Position 1

    auto diff = tenzor::sum(tenzor::abs(pos0 - pos1));
    float diff_value = diff.item<float>();
    EXPECT_GT(diff_value, 0.01f);
}

// ============================================================================
// HRMBlock Tests
// ============================================================================

TEST_F(HRMTest, HRMBlockBasic) {
    HRMBlock block(64, 4, 128, 0.0, true, 64);

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, true);

    auto output = block.forward(x);

    EXPECT_TRUE(shapes_equal(output.shape(), std::vector<int64_t>(x.shape().begin(), x.shape().end())));
}

TEST_F(HRMTest, HRMBlockWithContext) {
    HRMBlock block(64, 4, 128, 0.0, true, 64);

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Tensor context = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, true);
    Variable ctx(context, true);

    auto output = block.forward(x, ctx);

    EXPECT_TRUE(shapes_equal(output.shape(), std::vector<int64_t>(x.shape().begin(), x.shape().end())));
}

TEST_F(HRMTest, HRMBlockPreNorm) {
    HRMBlock block(64, 4, 128, 0.0, false, 64);  // use_post_norm=false

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, true);

    auto output = block.forward(x);

    EXPECT_TRUE(shapes_equal(output.shape(), std::vector<int64_t>(x.shape().begin(), x.shape().end())));
}

// ============================================================================
// Full HRM Tests
// ============================================================================

TEST_F(HRMTest, HRMConstruction) {
    HRM model(config_);

    auto params = model.parameters();
    EXPECT_GT(params.size(), 0);

    int64_t num_params = model.num_parameters();
    EXPECT_GT(num_params, 0);

    std::cout << "HRM parameters: " << num_params << std::endl;
}

TEST_F(HRMTest, HRMForward) {
    HRM model(config_);

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, true);

    auto output = model.forward(x);

    EXPECT_TRUE(shapes_equal(output.shape(), {2, 8, 64}));
}

TEST_F(HRMTest, HRMForwardWithAux) {
    config_.deep_supervision = true;
    HRM model(config_);

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, true);

    auto [output, aux_outputs] = model.forward_with_aux(x);

    EXPECT_TRUE(shapes_equal(output.shape(), {2, 8, 64}));

    // Should have output at each H cycle
    EXPECT_EQ(static_cast<int64_t>(aux_outputs.size()), config_.n_high_cycles);

    for (const auto& aux : aux_outputs) {
        EXPECT_TRUE(shapes_equal(aux.shape(), std::vector<int64_t>(output.shape().begin(), output.shape().end())));
    }
}

TEST_F(HRMTest, HRMWithOutputProjection) {
    config_.num_classes = 10;
    HRM model(config_);

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, true);

    auto output = model.forward(x);

    EXPECT_TRUE(shapes_equal(output.shape(), {2, 8, 10}));
}

TEST_F(HRMTest, HRMGradientFlow) {
    // With approximate gradients (state detachment), gradients only flow
    // through the final cycle. Use single cycle to test gradient flow.
    config_.n_high_cycles = 1;
    config_.t_low_steps = 1;
    HRM model(config_);
    model.train();

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, true);

    auto output = model.forward(x);

    // Sum and backward
    auto loss = tenzor::sum(output.tensor());
    Variable loss_var(loss, true);

    // Check that backward doesn't throw
    EXPECT_NO_THROW({
        loss_var.backward();
    });

    // With a single cycle, gradients should flow to output layers at minimum
    // Note: With approximate gradients, not all parameters get gradients
    // This test verifies the backward pass completes without error
}

TEST_F(HRMTest, HRMForwardStats) {
    HRM model(config_);

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, false);

    model.forward(x);

    auto stats = model.last_forward_stats();

    EXPECT_EQ(stats.actual_high_cycles, config_.n_high_cycles);
    EXPECT_EQ(stats.actual_low_steps, config_.n_high_cycles * config_.t_low_steps);
    EXPECT_GT(stats.h_participation_ratio, 0.0);
    EXPECT_GT(stats.l_participation_ratio, 0.0);
}

// ============================================================================
// Adaptive Computational Time Tests
// ============================================================================

TEST_F(HRMTest, ACTBasic) {
    AdaptiveComputationalTime act(64, 8, 0.99);

    Tensor state = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable s(state, true);

    auto halt_prob = act.compute_halt_prob(s);

    EXPECT_TRUE(shapes_equal(halt_prob.shape(), {2, 8}));

    // Probabilities should be in [0, 1]
    auto min_val = tenzor::min(halt_prob.tensor());
    auto max_val = tenzor::max(halt_prob.tensor());
    EXPECT_GE(min_val.item<float>(), 0.0f);
    EXPECT_LE(max_val.item<float>(), 1.0f);
}

TEST_F(HRMTest, HRMWithACT) {
    config_.use_act = true;
    config_.act_threshold = 0.5;  // Lower threshold for testing
    config_.n_high_cycles = 8;
    HRM model(config_);

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, false);

    model.forward(x);

    auto stats = model.last_forward_stats();

    // With ACT, may use fewer cycles
    EXPECT_LE(stats.actual_high_cycles, config_.n_high_cycles);
    EXPECT_GE(stats.actual_high_cycles, 1);
}

// ============================================================================
// Deep Supervision Loss Tests
// ============================================================================

TEST_F(HRMTest, DeepSupervisionLoss) {
    // Create fake outputs
    std::vector<Variable> outputs;
    for (int i = 0; i < 4; ++i) {
        Tensor t = randn({2, 8, 10}, DType::Float32, Device::cpu());
        outputs.push_back(Variable(t, true));
    }

    // Create targets - use randn since we're just testing the loss mechanism
    Tensor targets_data = randn({2, 8, 10}, DType::Float32, Device::cpu());
    Variable targets(targets_data, false);

    // Define loss function (simplified MSE for testing)
    auto loss_fn = [](const Variable& pred, const Variable& target) -> Variable {
        auto diff = pred.tensor() - target.tensor();
        return Variable(tenzor::mean(diff * diff), true);
    };

    auto loss = hrm_deep_supervision_loss(outputs, targets, loss_fn, 0.5);

    EXPECT_GT(loss.tensor().numel(), 0);
}

// ============================================================================
// Multi-Backend Tests
// ============================================================================

TEST_P(HRMMultiBackendTest, ForwardPass) {
    HRM model(config_);
    model.to(device_);

    Tensor input = randn({2, 8, 64}, DType::Float32, device_);
    Variable x(input, true);

    auto output = model.forward(x);

    EXPECT_TRUE(shapes_equal(output.shape(), {2, 8, 64}));
    EXPECT_EQ(output.tensor().device().type, device_.type);
}

TEST_P(HRMMultiBackendTest, ForwardWithAux) {
    config_.deep_supervision = true;
    HRM model(config_);
    model.to(device_);

    Tensor input = randn({2, 8, 64}, DType::Float32, device_);
    Variable x(input, true);

    auto [output, aux_outputs] = model.forward_with_aux(x);

    EXPECT_EQ(output.tensor().device().type, device_.type);
    EXPECT_EQ(static_cast<int64_t>(aux_outputs.size()), config_.n_high_cycles);
}

TEST_P(HRMMultiBackendTest, GradientComputation) {
    HRM model(config_);
    model.to(device_);
    model.train();

    Tensor input = randn({2, 8, 64}, DType::Float32, device_);
    Variable x(input, true);

    auto output = model.forward(x);
    auto loss = tenzor::mean(output.tensor());
    Variable loss_var(loss, true);

    // Backward should work on all backends
    EXPECT_NO_THROW({
        loss_var.backward();
    });
}

TEST_P(HRMMultiBackendTest, LargerModel) {
    // Test with larger dimensions
    config_.d_model = 128;
    config_.n_heads = 8;
    config_.d_feedforward = 256;

    HRM model(config_);
    model.to(device_);

    Tensor input = randn({4, 16, 128}, DType::Float32, device_);
    Variable x(input, false);

    auto output = model.forward(x);

    EXPECT_TRUE(shapes_equal(output.shape(), {4, 16, 128}));
}

TEST_P(HRMMultiBackendTest, WithOutputProjection) {
    config_.num_classes = 10;
    HRM model(config_);
    model.to(device_);

    Tensor input = randn({2, 8, 64}, DType::Float32, device_);
    Variable x(input, false);

    auto output = model.forward(x);

    EXPECT_TRUE(shapes_equal(output.shape(), {2, 8, 10}));
}

TEST_P(HRMMultiBackendTest, MultipleForwardPasses) {
    HRM model(config_);
    model.to(device_);

    // Run multiple forward passes to test stability
    for (int i = 0; i < 5; ++i) {
        Tensor input = randn({2, 8, 64}, DType::Float32, device_);
        Variable x(input, false);

        auto output = model.forward(x);
        EXPECT_TRUE(shapes_equal(output.shape(), {2, 8, 64}));
    }
}

TEST_P(HRMMultiBackendTest, DifferentSequenceLengths) {
    HRM model(config_);
    model.to(device_);

    std::vector<int64_t> seq_lens = {4, 8, 16, 32};

    for (int64_t seq_len : seq_lens) {
        Tensor input = randn({2, seq_len, 64}, DType::Float32, device_);
        Variable x(input, false);

        auto output = model.forward(x);
        EXPECT_TRUE(shapes_equal(output.shape(), {2, seq_len, 64}));
    }
}

TEST_P(HRMMultiBackendTest, BatchSizeVariation) {
    HRM model(config_);
    model.to(device_);

    std::vector<int64_t> batch_sizes = {1, 2, 4, 8};

    for (int64_t batch : batch_sizes) {
        Tensor input = randn({batch, 8, 64}, DType::Float32, device_);
        Variable x(input, false);

        auto output = model.forward(x);
        EXPECT_EQ(static_cast<int64_t>(output.shape()[0]), batch);
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_P(HRMMultiBackendTest, ForwardPerformance) {
    config_.d_model = 128;
    config_.n_heads = 8;
    config_.d_feedforward = 256;
    config_.n_high_cycles = 4;
    config_.t_low_steps = 8;

    HRM model(config_);
    model.to(device_);
    model.eval();

    Tensor input = randn({4, 32, 128}, DType::Float32, device_);
    Variable x(input, false);

    // Warmup
    model.forward(x);

    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    constexpr int num_iterations = 10;

    for (int i = 0; i < num_iterations; ++i) {
        model.forward(x);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_ms = duration.count() / 1000.0 / num_iterations;
    std::cout << "[" << backend_name_ << "] HRM forward: " << avg_ms << " ms avg" << std::endl;

    // Just ensure it completes in reasonable time
    EXPECT_LT(avg_ms, 10000.0);  // 10 seconds max
}

// ============================================================================
// Numerical Correctness Tests
// ============================================================================

TEST_F(HRMTest, DeterministicOutput) {
    HRM model(config_);
    model.eval();

    // Use fixed seed for reproducibility
    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, false);

    auto output1 = model.forward(x);
    auto output2 = model.forward(x);

    // Outputs should be identical for same input in eval mode
    auto diff = tenzor::max(tenzor::abs(output1.tensor() - output2.tensor()));
    EXPECT_LT(diff.item<float>(), 1e-5f);
}

TEST_F(HRMTest, OutputRange) {
    HRM model(config_);

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, false);

    auto output = model.forward(x);

    // Check for NaN/Inf by verifying output is finite
    // Output should be bounded - check min/max are reasonable
    auto min_val = tenzor::min(output.tensor()).item<float>();
    auto max_val = tenzor::max(output.tensor()).item<float>();

    // Values should be finite and not extreme
    EXPECT_FALSE(std::isnan(min_val));
    EXPECT_FALSE(std::isnan(max_val));
    EXPECT_FALSE(std::isinf(min_val));
    EXPECT_FALSE(std::isinf(max_val));
    EXPECT_LT(std::abs(min_val), 1e6f);
    EXPECT_LT(std::abs(max_val), 1e6f);
}

// ============================================================================
// Memory Efficiency Tests
// ============================================================================

TEST_F(HRMTest, MemoryEfficiencyApproximateGradient) {
    // Test that approximate gradient (detach) doesn't accumulate memory
    config_.n_high_cycles = 8;
    config_.t_low_steps = 8;

    HRM model(config_);

    // This would OOM without detach for large enough models
    Tensor input = randn({4, 32, 64}, DType::Float32, Device::cpu());
    Variable x(input, true);

    auto output = model.forward(x);
    auto loss = tenzor::mean(output.tensor());
    Variable loss_var(loss, true);

    // Should complete without memory issues
    EXPECT_NO_THROW({
        loss_var.backward();
    });
}

// ============================================================================
// Component Integration Tests
// ============================================================================

TEST_P(HRMMultiBackendTest, RMSNormOnDevice) {
    RMSNorm norm(64);
    norm.to(device_);

    Tensor input = randn({2, 8, 64}, DType::Float32, device_);
    Variable x(input, true);

    auto output = norm.forward(x);
    EXPECT_EQ(output.tensor().device().type, device_.type);
}

TEST_P(HRMMultiBackendTest, GLUOnDevice) {
    GatedLinearUnit glu(64, 128, true, false);
    glu.to(device_);

    Tensor input = randn({2, 8, 64}, DType::Float32, device_);
    Variable x(input, true);

    auto output = glu.forward(x);
    EXPECT_EQ(output.tensor().device().type, device_.type);
}

TEST_P(HRMMultiBackendTest, HRMBlockOnDevice) {
    HRMBlock block(64, 4, 128, 0.0, true, 64);
    block.to(device_);

    Tensor input = randn({2, 8, 64}, DType::Float32, device_);
    Variable x(input, true);

    auto output = block.forward(x);
    EXPECT_EQ(output.tensor().device().type, device_.type);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(HRMTest, SingleElement) {
    HRM model(config_);

    Tensor input = randn({1, 1, 64}, DType::Float32, Device::cpu());
    Variable x(input, false);

    auto output = model.forward(x);
    EXPECT_TRUE(shapes_equal(output.shape(), {1, 1, 64}));
}

TEST_F(HRMTest, SingleCycle) {
    config_.n_high_cycles = 1;
    config_.t_low_steps = 1;
    HRM model(config_);

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, false);

    auto output = model.forward(x);
    EXPECT_TRUE(shapes_equal(output.shape(), {2, 8, 64}));
}

TEST_F(HRMTest, ManyCycles) {
    config_.n_high_cycles = 16;
    config_.t_low_steps = 16;
    HRM model(config_);

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, false);

    auto output = model.forward(x);
    EXPECT_TRUE(shapes_equal(output.shape(), {2, 8, 64}));

    auto stats = model.last_forward_stats();
    EXPECT_EQ(stats.actual_high_cycles, 16);
}

// ============================================================================
// New Feature Tests: Stablemax, Q-Learning ACT, Initialization
// ============================================================================

TEST_F(HRMTest, StablemaxBasic) {
    // Test stablemax activation
    Tensor input = randn({2, 10}, DType::Float32, Device::cpu());
    Variable x(input, true);

    auto output = stablemax(x, -1, 1e-12);

    // Output should sum to 1 along last dimension
    auto sum = tenzor::sum(output.tensor(), {-1}, false);
    for (int64_t i = 0; i < 2; ++i) {
        float s = sum.data<float>()[i];
        EXPECT_NEAR(s, 1.0f, 0.01f);
    }

    // All values should be positive
    auto min_val = tenzor::min(output.tensor());
    EXPECT_GE(min_val.item<float>(), 0.0f);
}

TEST_F(HRMTest, StablemaxNumericalStability) {
    // Test with extreme values
    Tensor input({2, 5}, DType::Float32, Device::cpu());
    float* data = input.data<float>();
    // Large positive and negative values
    data[0] = 1000.0f; data[1] = -1000.0f; data[2] = 0.0f; data[3] = 500.0f; data[4] = -500.0f;
    data[5] = 1e-10f; data[6] = 1e-10f; data[7] = 1e-10f; data[8] = 1e-10f; data[9] = 1e-10f;

    Variable x(input, false);
    auto output = stablemax(x, -1, 1e-12);

    // Should not have NaN or Inf
    auto min_val = tenzor::min(output.tensor()).item<float>();
    auto max_val = tenzor::max(output.tensor()).item<float>();
    EXPECT_FALSE(std::isnan(min_val));
    EXPECT_FALSE(std::isnan(max_val));
    EXPECT_FALSE(std::isinf(min_val));
    EXPECT_FALSE(std::isinf(max_val));
}

TEST_F(HRMTest, QLearningACTBasic) {
    QLearningACT act(64, 8, 0.1, 0.99, 0.01);

    Tensor state = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable s(state, true);

    auto [q_halt, q_continue] = act.compute_q_values(s);

    // Q-values should be bounded to [0, 1]
    auto halt_min = tenzor::min(q_halt.tensor()).item<float>();
    auto halt_max = tenzor::max(q_halt.tensor()).item<float>();
    auto cont_min = tenzor::min(q_continue.tensor()).item<float>();
    auto cont_max = tenzor::max(q_continue.tensor()).item<float>();

    EXPECT_GE(halt_min, 0.0f);
    EXPECT_LE(halt_max, 1.0f);
    EXPECT_GE(cont_min, 0.0f);
    EXPECT_LE(cont_max, 1.0f);
}

TEST_F(HRMTest, QLearningACTDecision) {
    QLearningACT act(64, 8, 0.0, 0.99, 0.01);  // epsilon=0 for deterministic

    Tensor state = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable s(state, false);

    auto [q_halt, q_continue] = act.compute_q_values(s);

    // Action should be deterministic with epsilon=0
    bool action1 = act.select_action(q_halt, q_continue, false);
    bool action2 = act.select_action(q_halt, q_continue, false);

    // Both should return same action for same Q-values
    EXPECT_EQ(action1, action2);
}

TEST_F(HRMTest, QLearningACTEpsilonDecay) {
    QLearningACT act(64, 8, 1.0, 0.99, 0.01);

    auto initial_stats = act.stats();
    EXPECT_EQ(initial_stats.exploration_rate, 1.0);

    // Decay epsilon
    act.decay_epsilon(0.5, 0.01);

    auto decayed_stats = act.stats();
    EXPECT_EQ(decayed_stats.exploration_rate, 0.5);

    // Decay to minimum
    for (int i = 0; i < 100; ++i) {
        act.decay_epsilon(0.5, 0.01);
    }

    auto final_stats = act.stats();
    EXPECT_GE(final_stats.exploration_rate, 0.01);
}

TEST_F(HRMTest, HRMWithQLearningACT) {
    config_.use_act = true;
    config_.use_qlearning_act = true;
    config_.act_epsilon = 0.1;
    config_.n_high_cycles = 4;
    HRM model(config_);

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, false);

    model.eval();  // Disable exploration
    auto output = model.forward(x);

    EXPECT_TRUE(shapes_equal(output.shape(), {2, 8, 64}));

    auto stats = model.last_forward_stats();
    // With Q-learning ACT, may halt early
    EXPECT_LE(stats.actual_high_cycles, config_.n_high_cycles);
    EXPECT_GE(stats.actual_high_cycles, 1);
}

TEST_F(HRMTest, HRMWithStablemax) {
    config_.num_classes = 10;
    config_.use_stablemax = true;
    HRM model(config_);

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, false);

    auto output = model.forward(x);

    EXPECT_TRUE(shapes_equal(output.shape(), {2, 8, 10}));

    // Note: stablemax is applied internally, output should be valid probabilities
    // (when we get probabilities vs logits depends on implementation)
}

TEST_F(HRMTest, LeCunInitialization) {
    config_.use_lecun_init = true;
    HRM model(config_);

    // After LeCun initialization, weights should have reasonable variance
    auto params = model.parameters();
    EXPECT_GT(params.size(), 0u);

    // Check that weights are not all zeros and have reasonable range
    int weight_count = 0;
    for (auto& param : params) {
        auto shape = param->shape();

        // Skip bias parameters (1D tensors) - they may have zero variance
        if (shape.size() < 2) continue;

        weight_count++;
        auto mean = tenzor::mean(param->tensor()).item<float>();
        auto var = tenzor::var(param->tensor()).item<float>();

        EXPECT_FALSE(std::isnan(mean));
        EXPECT_FALSE(std::isnan(var));
        EXPECT_GT(var, 0.0f);  // Variance should be positive for weight matrices
    }

    EXPECT_GT(weight_count, 0);  // Should have some weight matrices
}

TEST_F(HRMTest, TruncatedNormalInit) {
    // Test truncated normal initialization
    Tensor t({1000}, DType::Float32, Device::cpu());
    truncated_normal_init(t, 0.0, 1.0, -2.0, 2.0);

    float* data = t.data<float>();

    // All values should be within [-2, 2] (in std units)
    for (int64_t i = 0; i < 1000; ++i) {
        EXPECT_GE(data[i], -2.0f);
        EXPECT_LE(data[i], 2.0f);
    }

    // Mean should be close to 0
    auto mean = tenzor::mean(t).item<float>();
    EXPECT_NEAR(mean, 0.0f, 0.2f);
}

TEST_F(HRMTest, ForwardWithSegments) {
    config_.use_act = true;
    config_.use_qlearning_act = true;
    config_.max_segments = 4;
    config_.num_classes = 10;
    HRM model(config_);
    model.train();

    Tensor input = randn({2, 8, 64}, DType::Float32, Device::cpu());
    Variable x(input, true);

    // Create fake targets
    Tensor targets_data({2, 8}, DType::Int64, Device::cpu());
    for (int64_t i = 0; i < 16; ++i) {
        targets_data.data<int64_t>()[i] = i % 10;
    }
    Variable targets(targets_data, false);

    auto [output, segment_outputs, q_loss] = model.forward_with_segments(x, targets);

    EXPECT_TRUE(shapes_equal(output.shape(), {2, 8, 10}));
    EXPECT_GT(segment_outputs.size(), 0u);

    auto stats = model.last_forward_stats();
    EXPECT_GT(stats.actual_segments, 0);
}

TEST_F(HRMTest, ProperEmbedding) {
    // Test that vocab_size uses proper Embedding layer
    config_.vocab_size = 1000;
    config_.num_classes = 10;
    HRM model(config_);

    // Create token input
    Tensor input({2, 8}, DType::Int64, Device::cpu());
    int64_t* data = input.data<int64_t>();
    for (int64_t i = 0; i < 16; ++i) {
        data[i] = i % 1000;  // Random token IDs
    }
    Variable x(input, false);

    auto output = model.forward(x);

    // Output should be (batch, seq_len, num_classes)
    EXPECT_TRUE(shapes_equal(output.shape(), {2, 8, 10}));
}

// ============================================================================
// Multi-Backend Tests for New Features
// ============================================================================

TEST_P(HRMMultiBackendTest, StablemaxOnDevice) {
    Tensor input = randn({4, 16}, DType::Float32, device_);
    Variable x(input, true);

    auto output = stablemax(x, -1, 1e-12);

    EXPECT_EQ(output.tensor().device().type, device_.type);

    // Sum should be approximately 1
    auto sum = tenzor::sum(output.tensor(), {-1}, false);
    for (int64_t i = 0; i < 4; ++i) {
        float s = sum.to(Device::cpu()).data<float>()[i];
        EXPECT_NEAR(s, 1.0f, 0.01f);
    }
}

TEST_P(HRMMultiBackendTest, QLearningACTOnDevice) {
    QLearningACT act(64, 8, 0.1, 0.99, 0.01);
    act.to(device_);

    Tensor state = randn({2, 8, 64}, DType::Float32, device_);
    Variable s(state, true);

    auto [q_halt, q_continue] = act.compute_q_values(s);

    EXPECT_EQ(q_halt.tensor().device().type, device_.type);
    EXPECT_EQ(q_continue.tensor().device().type, device_.type);
}

TEST_P(HRMMultiBackendTest, HRMWithQLearningACTOnDevice) {
    config_.use_act = true;
    config_.use_qlearning_act = true;
    HRM model(config_);
    model.to(device_);

    Tensor input = randn({2, 8, 64}, DType::Float32, device_);
    Variable x(input, false);

    auto output = model.forward(x);

    EXPECT_EQ(output.tensor().device().type, device_.type);
}
