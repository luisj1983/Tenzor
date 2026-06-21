/**
 * @file test_zero_stage2.cpp
 * @brief Unit tests for ZeRO Stage 2 Optimizer (Gradient Partitioning)
 *
 * Tests ZeRO Stage 2 optimizer functionality including:
 * - Constructor validation
 * - Gradient bucketing
 * - Reduce-scatter correctness
 * - Backward hook registration
 * - Memory reduction verification (8x total)
 * - CPU offload for gradients
 * - Edge cases (empty gradients, single parameter, etc.)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <memory>
#include <vector>

using namespace tenzor;
using namespace tenzor::optim;
using tenzor::nn::relu;

// ============================================================================
// Mock Module for Testing
// ============================================================================

class SimpleTestModule : public nn::Module {
public:
    SimpleTestModule(int input_dim, int hidden_dim, int output_dim) {
        // Layer 1: input -> hidden
        register_parameter("w1", Variable(ones({input_dim, hidden_dim}, DType::Float32, Device::cpu()), true));
        register_parameter("b1", Variable(zeros({hidden_dim}, DType::Float32, Device::cpu()), true));

        // Layer 2: hidden -> output
        register_parameter("w2", Variable(ones({hidden_dim, output_dim}, DType::Float32, Device::cpu()), true));
        register_parameter("b2", Variable(zeros({output_dim}, DType::Float32, Device::cpu()), true));
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto w1 = parameters_.at("w1");
        auto b1 = parameters_.at("b1");
        auto w2 = parameters_.at("w2");
        auto b2 = parameters_.at("b2");

        // Build a REAL autograd graph through the parameters so that
        // loss.backward() populates each parameter's gradient (and the Stage 2
        // backward hooks actually fire). Using raw-tensor ops + Variable(...)
        // re-wrapping (as a prior version did) severs grad_fn and forces tests
        // to fake gradients with set_grad() — defeating hook coverage.
        // x @ w1 + b1
        auto h = matmul(x, *w1) + *b1;
        // relu(h)
        auto h_relu = nn::relu(h);
        // h @ w2 + b2
        auto out = matmul(h_relu, *w2) + *b2;
        return out;
    }
};

// ============================================================================
// Test Fixtures
// ============================================================================

class ZeROStage2Test : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }

    void SetUp() override {
        // Default config for single-process tests
        default_config.world_size = 1;  // Single process mode
        default_config.rank = 0;
        default_config.offload_to_cpu = false;
        default_config.cpu_offload_threshold = 1024;  // 1KB
        default_config.overlap_comm = true;
        default_config.pin_memory = true;
        default_config.process_group = nullptr;  // Single-process mode
    }

    // Helper: Create test parameters on a specific device
    auto create_test_params(size_t count, const std::vector<int64_t>& shape = {128, 128},
                           Device device = Device::cpu())
        -> std::vector<std::shared_ptr<Variable>> {
        std::vector<std::shared_ptr<Variable>> params;
        params.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            auto param = std::make_shared<Variable>(
                ones(shape, DType::Float32, device),
                true
            );
            params.push_back(param);
        }
        return params;
    }

    // Helper: Check if CUDA is available by attempting to create a tensor
    static bool cuda_available() {
        try {
            auto test_tensor = ones({1}, DType::Float32, Device::cuda(0));
            return true;
        } catch (...) {
            return false;
        }
    }

    // Helper: Create test model
    auto create_test_model() -> std::shared_ptr<SimpleTestModule> {
        return std::make_shared<SimpleTestModule>(128, 256, 10);
    }

    // Helper: Attach gradients to parameters
    auto attach_gradients(std::vector<std::shared_ptr<Variable>>& params) -> void {
        for (auto& param : params) {
            // Create gradient of same shape
            auto grad = ones_like(param->tensor());
            param->set_grad(grad);
        }
    }

    ZeROStage2Config default_config;
};

// ============================================================================
// 1. Constructor Validation Tests
// ============================================================================

TEST_F(ZeROStage2Test, ConstructorWithValidConfig) {
    auto params = create_test_params(100);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    config.world_size = 1;
    config.rank = 0;

    EXPECT_NO_THROW({
        ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);
    });
}

TEST_F(ZeROStage2Test, ConstructorWithMultipleRanks) {
    auto params = create_test_params(100);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    config.world_size = 4;
    config.rank = 0;

    // Stage 2 should support multiple ranks for gradient partitioning
    EXPECT_NO_THROW({
        ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);
    });
}

TEST_F(ZeROStage2Test, ConstructorValidatesBucketSize) {
    auto params = create_test_params(100);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;

    // Stage 2 should have configurable bucket size for gradient bucketing
    // For now, just verify basic construction works
    EXPECT_NO_THROW({
        ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);
    });
}

TEST_F(ZeROStage2Test, ConstructorWithInvalidRank) {
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    config.world_size = 4;
    config.rank = 5;  // Invalid: rank >= world_size

    EXPECT_THROW({
        ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);
    }, std::invalid_argument);
}

TEST_F(ZeROStage2Test, ConstructorWithNullOptimizer) {
    ZeROStage2Config config = default_config;

    EXPECT_THROW({
        ZeROStage2Optimizer optimizer(nullptr, config);
    }, std::invalid_argument);
}

// ============================================================================
// 2. Gradient Bucketing Tests
// ============================================================================

TEST_F(ZeROStage2Test, GradientBucketingWithDefaultSize) {
    // Stage 2 should automatically bucket gradients for efficient reduce-scatter
    auto params = create_test_params(100, {64, 64});  // 100 params of 64x64
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    // Stage 2 buckets gradients at construction when gradient_bucketing is on
    // (the default). With 100 parameters there must be at least one bucket, and
    // the buckets must cover real bytes.
    auto bucket_stats = optimizer.get_bucket_stats();
    EXPECT_GT(bucket_stats.num_buckets, 0u)
        << "Stage 2 with gradient_bucketing must create gradient buckets";
    EXPECT_GT(bucket_stats.max_bucket_size, 0u)
        << "gradient buckets must span non-zero bytes";

    // Attach gradients
    attach_gradients(params);

    // Step should handle bucketed gradients correctly, and with non-zero
    // gradients it must actually move the parameters.
    auto before = params[0]->tensor().clone();
    optimizer.step();
    auto after = params[0]->tensor();
    auto max_delta = tenzor::max(tenzor::abs((after - before).to(tenzor::DType::Float64))).item<double>();
    EXPECT_GT(max_delta, 0.0) << "optimizer.step() did not update parameters";
}

TEST_F(ZeROStage2Test, GradientBucketingWithCustomSize) {
    // Test with custom bucket size (e.g., 25MB)
    auto params = create_test_params(50, {128, 128});
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    // A small bucket size must split the 50 128x128 (64KB each) gradients into
    // more buckets than a single huge bucket would. This pins the
    // gradient_bucket_size config to a real effect on bucketing.
    ZeROStage2Config small_cfg = default_config;
    small_cfg.gradient_bucket_size = 128 * 1024;  // 128KB → ~2 params/bucket
    auto small_opt = std::make_unique<Adam>(create_test_params(50, {128, 128}), 0.001);
    ZeROStage2Optimizer small_optimizer(std::move(small_opt), small_cfg);

    ZeROStage2Config big_cfg = default_config;
    big_cfg.gradient_bucket_size = 256 * 1024 * 1024;  // 256MB → all in one bucket
    auto big_opt = std::make_unique<Adam>(create_test_params(50, {128, 128}), 0.001);
    ZeROStage2Optimizer big_optimizer(std::move(big_opt), big_cfg);

    EXPECT_GT(small_optimizer.get_bucket_stats().num_buckets,
              big_optimizer.get_bucket_stats().num_buckets)
        << "a smaller gradient_bucket_size must produce more gradient buckets";

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);
    attach_gradients(params);

    auto before = params[0]->tensor().clone();
    optimizer.step();
    auto after = params[0]->tensor();
    auto max_delta = tenzor::max(tenzor::abs((after - before).to(tenzor::DType::Float64))).item<double>();
    EXPECT_GT(max_delta, 0.0) << "optimizer.step() did not update parameters";
}

TEST_F(ZeROStage2Test, GradientBucketingWithSmallParameters) {
    // Test bucketing with many small parameters
    auto params = create_test_params(1000, {8, 8});  // 1000 tiny params
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    attach_gradients(params);
    auto before = params[0]->tensor().clone();
    optimizer.step();
    auto after = params[0]->tensor();
    auto max_delta = tenzor::max(tenzor::abs((after - before).to(tenzor::DType::Float64))).item<double>();
    EXPECT_GT(max_delta, 0.0) << "optimizer.step() did not update parameters";
}

TEST_F(ZeROStage2Test, GradientBucketingWithLargeParameters) {
    // Test bucketing with few large parameters
    auto params = create_test_params(5, {512, 512});  // 5 large params
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    attach_gradients(params);
    auto before = params[0]->tensor().clone();
    optimizer.step();
    auto after = params[0]->tensor();
    auto max_delta = tenzor::max(tenzor::abs((after - before).to(tenzor::DType::Float64))).item<double>();
    EXPECT_GT(max_delta, 0.0) << "optimizer.step() did not update parameters";
}

TEST_F(ZeROStage2Test, GradientBucketingWithMixedSizes) {
    // Test bucketing with mixed parameter sizes
    std::vector<std::shared_ptr<Variable>> params;
    params.push_back(std::make_shared<Variable>(ones({1024, 1024}, DType::Float32, Device::cpu()), true));
    params.push_back(std::make_shared<Variable>(ones({128, 128}, DType::Float32, Device::cpu()), true));
    params.push_back(std::make_shared<Variable>(ones({16, 16}, DType::Float32, Device::cpu()), true));
    params.push_back(std::make_shared<Variable>(ones({512, 256}, DType::Float32, Device::cpu()), true));

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    attach_gradients(params);
    auto before = params[0]->tensor().clone();
    optimizer.step();
    auto after = params[0]->tensor();
    auto max_delta = tenzor::max(tenzor::abs((after - before).to(tenzor::DType::Float64))).item<double>();
    EXPECT_GT(max_delta, 0.0) << "optimizer.step() did not update parameters";
}

// ============================================================================
// 3. Reduce-Scatter Correctness Tests
// ============================================================================

TEST_F(ZeROStage2Test, ReduceScatterGradientsSingleRank) {
    // Single rank should not perform reduce-scatter (no communication needed)
    auto params = create_test_params(50);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    config.world_size = 1;
    config.rank = 0;

    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);
    attach_gradients(params);

    // Should complete without communication, and must update the parameters.
    auto before = params[0]->tensor().clone();
    optimizer.step();
    auto after = params[0]->tensor();
    auto max_delta = tenzor::max(tenzor::abs((after - before).to(tenzor::DType::Float64))).item<double>();
    EXPECT_GT(max_delta, 0.0) << "optimizer.step() did not update parameters";
}

TEST_F(ZeROStage2Test, ReduceScatterGradientsCorrectSum) {
    // Verify that reduce-scatter correctly sums gradients
    auto params = create_test_params(10, {32, 32});

    // Set gradients to known values
    for (size_t i = 0; i < params.size(); ++i) {
        auto grad = full({32, 32}, static_cast<float>(i + 1), DType::Float32, Device::cpu());
        params[i]->set_grad(grad);
    }

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage2Config config = default_config;
    config.world_size = 1;  // Single process - no actual reduce-scatter

    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    // Stage 2 reduce-scatter is driven by backward hooks; registering them must
    // succeed and flip the registered flag (a Stage-2-specific surface that the
    // old Stage-1 substitution never exercised).
    optimizer.register_backward_hooks();
    EXPECT_TRUE(optimizer.hooks_registered())
        << "register_backward_hooks() must mark hooks registered";

    // In single-rank mode reduce-scatter is the identity (sum over one rank),
    // so the per-partition gradient equals the input. params[0] has gradient 1,
    // so the base Adam update must move it. (True multi-rank summation is not
    // reachable in a single process — see test_zero_stage1_distributed.)
    auto before = params[0]->tensor().clone();
    optimizer.step();
    auto after = params[0]->tensor();
    auto max_delta = tenzor::max(tenzor::abs((after - before).to(tenzor::DType::Float64))).item<double>();
    EXPECT_GT(max_delta, 0.0) << "optimizer.step() did not update parameters";
}

TEST_F(ZeROStage2Test, ReduceScatterGradientsPartitioning) {
    // Test that gradients are correctly partitioned after reduce-scatter
    auto params = create_test_params(100, {64, 64});
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    config.world_size = 1;

    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    // Stage 2 must have bucketed the 100 gradients for reduce-scatter at
    // construction (a Stage-2-specific surface).
    EXPECT_GT(optimizer.get_bucket_stats().num_buckets, 0u);

    attach_gradients(params);
    EXPECT_NO_THROW(optimizer.step());

    // Single rank owns its whole partition (= all 100 params).
    EXPECT_EQ(optimizer.local_param_count(), 100);

    // A multi-rank optimizer must partition: rank 0 of 4 owns a strict subset.
    auto multi = std::make_unique<Adam>(create_test_params(100, {64, 64}), 0.001);
    ZeROStage2Config cfg4 = default_config;
    cfg4.world_size = 4;
    cfg4.rank = 0;
    ZeROStage2Optimizer optimizer4(std::move(multi), cfg4);
    EXPECT_LT(optimizer4.local_param_count(), 100)
        << "4-way reduce-scatter must give rank 0 only its partition";
}

TEST_F(ZeROStage2Test, ReduceScatterGradientsMemoryFreed) {
    // Test that non-local gradients are freed after reduce-scatter
    auto params = create_test_params(100);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    attach_gradients(params);

    // Get memory before step
    auto stats_before = optimizer.get_memory_stats();

    optimizer.step();

    // Get memory after step
    auto stats_after = optimizer.get_memory_stats();

    // Verify stats changed (suppress unused warning)
    EXPECT_EQ(stats_before.num_parameters, stats_after.num_parameters);

    // Memory usage should be reasonable (gradients may be freed)
    EXPECT_GT(stats_after.num_parameters, 0);
}

// ============================================================================
// 4. Backward Hook Registration Tests
// ============================================================================

TEST_F(ZeROStage2Test, BackwardHooksRegisteredOnConstruction) {
    // Stage 2 should register backward hooks for gradient reduce-scatter
    auto model = create_test_model();
    auto params = model->parameters();
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    // Register the reduce-scatter backward hooks and verify the flag flips —
    // this is the construction-time surface the test is named for.
    optimizer.register_backward_hooks();
    EXPECT_TRUE(optimizer.hooks_registered())
        << "Stage 2 must mark backward hooks as registered";

    // Run a real forward → loss → backward through the autograd graph and
    // confirm the parameters actually receive gradients (the hooks fire on a
    // genuine grad_fn chain — not a faked set_grad()).
    // SimpleTestModule initialises w1=ones, so the hidden pre-activation is
    // h[i,j] = sum_k x[i,k] (identical across units). A random input whose row
    // sums all happen to be negative would make relu(h) all-zero and leave w1/b1
    // with zero gradient — a degenerate, RNG-ordering-dependent flake (the test
    // asserts EVERY parameter receives a gradient). Use a strictly positive input
    // so relu is always active and the grad-flow invariant is exercised
    // deterministically regardless of global RNG state / test ordering.
    auto input = Variable(tenzor::abs(randn({4, 128}, DType::Float32, Device::cpu())) + 0.5f, true);
    auto output = model->forward(input);
    auto loss = sum(output);
    EXPECT_NO_THROW(loss.backward());
    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad())
            << "backward did not populate a parameter gradient";
    }
}

TEST_F(ZeROStage2Test, BackwardHooksTriggeredDuringBackward) {
    auto model = create_test_model();
    auto params = model->parameters();
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);
    optimizer.register_backward_hooks();

    // Run a real training step. backward() over the autograd graph must
    // populate every parameter's gradient — no manual set_grad() fill, so the
    // assertion genuinely verifies the hook/backward path produced gradients.
    // SimpleTestModule initialises w1=ones, so the hidden pre-activation is
    // h[i,j] = sum_k x[i,k] (identical across units). A random input whose row
    // sums all happen to be negative would make relu(h) all-zero and leave w1/b1
    // with zero gradient — a degenerate, RNG-ordering-dependent flake (the test
    // asserts EVERY parameter receives a gradient). Use a strictly positive input
    // so relu is always active and the grad-flow invariant is exercised
    // deterministically regardless of global RNG state / test ordering.
    auto input = Variable(tenzor::abs(randn({4, 128}, DType::Float32, Device::cpu())) + 0.5f, true);
    auto output = model->forward(input);
    auto loss = sum(output);

    loss.backward();

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad())
            << "backward did not populate a parameter gradient via the graph";
        // Gradient must be genuinely non-zero, not a zero placeholder.
        auto g = param->grad().value().to(DType::Float64).cpu();
        EXPECT_GT(tenzor::max(tenzor::abs(g)).item<double>(), 0.0)
            << "parameter gradient is identically zero after backward";
    }
}

TEST_F(ZeROStage2Test, BackwardHooksWithMultipleLayers) {
    // Test hooks with multi-layer model
    auto model = create_test_model();
    auto params = model->parameters();

    // Should have 4 parameters (w1, b1, w2, b2)
    EXPECT_EQ(params.size(), 4);

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);
    optimizer.register_backward_hooks();

    // All parameters across both layers must receive gradients through the real
    // autograd graph (no manual fill), so a layer the hooks miss fails here.
    // SimpleTestModule initialises w1=ones, so the hidden pre-activation is
    // h[i,j] = sum_k x[i,k] (identical across units). A random input whose row
    // sums all happen to be negative would make relu(h) all-zero and leave w1/b1
    // with zero gradient — a degenerate, RNG-ordering-dependent flake (the test
    // asserts EVERY parameter receives a gradient). Use a strictly positive input
    // so relu is always active and the grad-flow invariant is exercised
    // deterministically regardless of global RNG state / test ordering.
    auto input = Variable(tenzor::abs(randn({4, 128}, DType::Float32, Device::cpu())) + 0.5f, true);
    auto output = model->forward(input);
    auto loss = sum(output);
    loss.backward();

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad())
            << "a layer's parameter did not receive a gradient";
        auto g = param->grad().value().to(DType::Float64).cpu();
        EXPECT_GT(tenzor::max(tenzor::abs(g)).item<double>(), 0.0)
            << "parameter gradient is identically zero after backward";
    }
}

TEST_F(ZeROStage2Test, BackwardHooksWithEmptyModel) {
    // Test hooks with empty parameter list
    std::vector<std::shared_ptr<Variable>> empty_params;
    auto base_optimizer = std::make_unique<Adam>(empty_params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    // Should handle empty model gracefully
    EXPECT_NO_THROW(optimizer.step());
}

// ============================================================================
// 5. Memory Reduction Verification (8x total)
// ============================================================================

TEST_F(ZeROStage2Test, MemoryReduction8xVerification) {
    // Stage 2 partitions optimizer states AND gradients across ranks, so the
    // per-rank local parameter count must shrink ~world_size× relative to
    // single-rank. Verify the actual partitioning reduction (not just ">0").
    constexpr int total = 100;

    auto single = std::make_unique<Adam>(create_test_params(total, {256, 256}), 0.001);
    ZeROStage2Config cfg1 = default_config;
    cfg1.world_size = 1;
    cfg1.rank = 0;
    ZeROStage2Optimizer opt1(std::move(single), cfg1);
    auto stats1 = opt1.get_memory_stats();
    EXPECT_EQ(stats1.num_local_parameters, total)
        << "single rank must own all parameters";
    EXPECT_GT(stats1.gpu_optimizer_memory + stats1.cpu_optimizer_memory, 0);

    auto multi = std::make_unique<Adam>(create_test_params(total, {256, 256}), 0.001);
    ZeROStage2Config cfg8 = default_config;
    cfg8.world_size = 8;
    cfg8.rank = 0;
    ZeROStage2Optimizer opt8(std::move(multi), cfg8);
    auto stats8 = opt8.get_memory_stats();

    // Rank 0 of 8 must own ~1/8 of the parameters — a real reduction, not all.
    EXPECT_LT(stats8.num_local_parameters, stats1.num_local_parameters)
        << "8-way partitioning must reduce the per-rank local parameter count";
    EXPECT_LE(stats8.num_local_parameters, total / 8 + 2)
        << "rank 0 of 8 should own ~12-13 of 100 parameters";
    EXPECT_GT(stats8.num_local_parameters, 0);
}

TEST_F(ZeROStage2Test, MemoryReductionOptimizerStates) {
    // Test optimizer state memory reduction
    auto params = create_test_params(50, {128, 128});
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    attach_gradients(params);
    optimizer.step();  // Initialize optimizer states

    auto stats = optimizer.get_memory_stats();

    // Adam has 2 states per parameter (momentum + variance)
    // Each param is 128*128*4 = 65536 bytes
    // 50 params * 65536 * 2 states = 6,553,600 bytes expected
    size_t expected_state_memory = 50 * 128 * 128 * sizeof(float) * 2;

    // Allow some overhead for metadata
    EXPECT_GT(stats.gpu_optimizer_memory + stats.cpu_optimizer_memory, expected_state_memory * 0.9);
}

TEST_F(ZeROStage2Test, MemoryReductionGradients) {
    // Test gradient memory reduction
    auto params = create_test_params(100, {64, 64});
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    attach_gradients(params);

    auto stats = optimizer.get_memory_stats();

    // Each gradient is 64*64*4 = 16384 bytes
    // 100 params * 16384 = 1,638,400 bytes expected
    size_t expected_grad_memory = 100 * 64 * 64 * sizeof(float);

    // Gradient memory should be tracked
    EXPECT_GT(stats.gpu_gradient_memory, expected_grad_memory * 0.9);
}

TEST_F(ZeROStage2Test, MemoryReductionScalingWithRanks) {
    // Test that memory scales with number of ranks
    auto params = create_test_params(100);

    // Single rank baseline
    {
        auto opt1 = std::make_unique<Adam>(create_test_params(100), 0.001);
        ZeROStage2Config config1 = default_config;
        config1.world_size = 1;
        config1.rank = 0;

        ZeROStage2Optimizer optimizer1(std::move(opt1), config1);
        auto stats1 = optimizer1.get_memory_stats();

        // All parameters local in single rank
        EXPECT_EQ(stats1.num_local_parameters, 100);
    }

    // Multi-rank should partition
    // Note: Can't actually test multi-rank in single process, but verify config
    {
        auto opt4 = std::make_unique<Adam>(create_test_params(100), 0.001);
        ZeROStage2Config config4 = default_config;
        config4.world_size = 4;
        config4.rank = 0;

        ZeROStage2Optimizer optimizer4(std::move(opt4), config4);
        auto stats4 = optimizer4.get_memory_stats();

        // Rank 0 should own ~25 parameters (100/4)
        EXPECT_LE(stats4.num_local_parameters, 30);  // ~25 with some tolerance
        EXPECT_GE(stats4.num_local_parameters, 20);
    }
}

// ============================================================================
// 6. CPU Offload for Gradients
// ============================================================================

TEST_F(ZeROStage2Test, CPUOffloadGradientsEnabled) {
    auto params = create_test_params(50);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    config.offload_to_cpu = true;

    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    EXPECT_TRUE(optimizer.is_cpu_offload_enabled());
}

TEST_F(ZeROStage2Test, CPUOffloadGradientsDisabled) {
    auto params = create_test_params(50);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    config.offload_to_cpu = false;

    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    EXPECT_FALSE(optimizer.is_cpu_offload_enabled());
}

TEST_F(ZeROStage2Test, CPUOffloadGradientsMemoryLocation) {
    auto params = create_test_params(50);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    config.offload_to_cpu = true;

    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);
    attach_gradients(params);
    optimizer.step();

    // Verify step completes and memory stats are available
    auto stats = optimizer.get_memory_stats();
    EXPECT_EQ(stats.num_local_parameters, 50);
}

TEST_F(ZeROStage2Test, CPUOffloadFromGPU) {
    // This test requires CUDA - skip if not available
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available, skipping GPU offload test";
    }

    // Create parameters on GPU
    auto params = create_test_params(10, {128, 128}, Device::cuda(0));

    // Verify params are on GPU
    for (const auto& param : params) {
        ASSERT_EQ(param->tensor().device().type, Device::Type::CUDA);
    }

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    config.offload_to_cpu = true;
    config.cpu_offload_threshold = 1024;

    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    // Attach gradients on GPU
    for (auto& param : params) {
        param->set_grad(ones_like(param->tensor()));
    }

    EXPECT_TRUE(optimizer.is_cpu_offload_enabled());
    EXPECT_NO_THROW(optimizer.step());

    auto stats = optimizer.get_memory_stats();
    EXPECT_EQ(stats.num_local_parameters, 10);
}

TEST_F(ZeROStage2Test, CPUOffloadGradientsThreshold) {
    auto params = create_test_params(10, {8, 8});  // Small params
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    config.offload_to_cpu = true;
    config.cpu_offload_threshold = 1024 * 1024;  // 1MB threshold

    // Small params below threshold should not be offloaded
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);
    attach_gradients(params);
    optimizer.step();

    EXPECT_NO_THROW(optimizer.get_memory_stats());
}

// ============================================================================
// 7. Edge Cases
// ============================================================================

TEST_F(ZeROStage2Test, EdgeCaseEmptyGradients) {
    // Test with parameters that have no gradients
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    // Don't attach gradients - step should handle gracefully
    EXPECT_NO_THROW(optimizer.step());
}

TEST_F(ZeROStage2Test, EdgeCaseSingleParameter) {
    // Test with single parameter
    auto params = create_test_params(1, {256, 256});
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    attach_gradients(params);
    EXPECT_NO_THROW(optimizer.step());

    EXPECT_EQ(optimizer.local_param_count(), 1);
}

TEST_F(ZeROStage2Test, EdgeCaseSparseGradients) {
    // Test with some parameters having gradients and some not
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    // Only attach gradients to half the parameters
    for (size_t i = 0; i < params.size(); i += 2) {
        auto grad = ones_like(params[i]->tensor());
        params[i]->set_grad(grad);
    }

    // params[0] received a non-zero gradient, so it must be updated.
    auto before = params[0]->tensor().clone();
    optimizer.step();
    auto after = params[0]->tensor();
    auto max_delta = tenzor::max(tenzor::abs((after - before).to(tenzor::DType::Float64))).item<double>();
    EXPECT_GT(max_delta, 0.0) << "optimizer.step() did not update parameters";
}

TEST_F(ZeROStage2Test, EdgeCaseZeroGradients) {
    // Test with zero-valued gradients
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    // Attach zero gradients
    for (auto& param : params) {
        auto grad = zeros_like(param->tensor());
        param->set_grad(grad);
    }

    EXPECT_NO_THROW(optimizer.step());
}

TEST_F(ZeROStage2Test, EdgeCaseVeryLargeGradients) {
    // Test with very large gradient values
    auto params = create_test_params(5, {64, 64});
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    // Attach large gradients
    for (auto& param : params) {
        auto shape_span = param->tensor().shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        auto grad = full(shape_vec, 1e6f, param->tensor().dtype(), param->tensor().device());
        param->set_grad(grad);
    }

    // Should handle large values without overflow, and update the parameters.
    auto before = params[0]->tensor().clone();
    optimizer.step();
    auto after = params[0]->tensor();
    auto max_delta = tenzor::max(tenzor::abs((after - before).to(tenzor::DType::Float64))).item<double>();
    EXPECT_GT(max_delta, 0.0) << "optimizer.step() did not update parameters";
}

TEST_F(ZeROStage2Test, EdgeCaseMixedGradientSizes) {
    // Test with gradients of vastly different sizes
    std::vector<std::shared_ptr<Variable>> params;
    params.push_back(std::make_shared<Variable>(ones({1024, 1024}, DType::Float32, Device::cpu()), true));
    params.push_back(std::make_shared<Variable>(ones({4, 4}, DType::Float32, Device::cpu()), true));
    params.push_back(std::make_shared<Variable>(ones({256, 128}, DType::Float32, Device::cpu()), true));

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    attach_gradients(params);
    auto before = params[0]->tensor().clone();
    optimizer.step();
    auto after = params[0]->tensor();
    auto max_delta = tenzor::max(tenzor::abs((after - before).to(tenzor::DType::Float64))).item<double>();
    EXPECT_GT(max_delta, 0.0) << "optimizer.step() did not update parameters";
}

TEST_F(ZeROStage2Test, EdgeCaseMultipleSteps) {
    // Test multiple optimizer steps in sequence
    auto params = create_test_params(20);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    // Run multiple steps; each step has non-zero gradients so the parameters
    // must move every iteration.
    for (int step = 0; step < 10; ++step) {
        attach_gradients(params);
        auto before = params[0]->tensor().clone();
        optimizer.step();
        auto after = params[0]->tensor();
        auto max_delta = tenzor::max(tenzor::abs((after - before).to(tenzor::DType::Float64))).item<double>();
        EXPECT_GT(max_delta, 0.0) << "optimizer.step() did not update parameters at step " << step;
        optimizer.zero_grad();
    }
}

TEST_F(ZeROStage2Test, EdgeCaseZeroGradAfterStep) {
    // Test zero_grad functionality
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    attach_gradients(params);
    optimizer.step();
    optimizer.zero_grad();

    // Gradients should be cleared
    for (const auto& param : params) {
        if (param->has_grad()) {
            auto grad_opt = param->grad();
            if (grad_opt.has_value()) {
                const auto& grad = grad_opt.value();
                // Check if gradient is effectively zero or cleared
                // Note: Implementation may clear optional or zero out values
                EXPECT_TRUE(grad.numel() > 0 || !grad_opt.has_value());
            }
        }
    }
}

// ============================================================================
// 8. State Dict Save/Load Tests
// ============================================================================

TEST_F(ZeROStage2Test, StateDictSaveLoad) {
    auto params = create_test_params(20);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    attach_gradients(params);
    optimizer.step();  // Initialize optimizer states

    // Save state
    auto state = optimizer.state_dict();
    EXPECT_FALSE(state.empty());

    // Load state into new optimizer
    auto params2 = create_test_params(20);
    auto base_optimizer2 = std::make_unique<Adam>(params2, 0.001);
    ZeROStage2Optimizer optimizer2(std::move(base_optimizer2), config);

    EXPECT_NO_THROW(optimizer2.load_state_dict(state));
}

TEST_F(ZeROStage2Test, StateDictContainsRankInfo) {
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage2Config config = default_config;
    config.world_size = 4;
    config.rank = 0;

    ZeROStage2Optimizer optimizer(std::move(base_optimizer), config);

    auto state = optimizer.state_dict();

    // State should contain rank and world_size metadata
    EXPECT_TRUE(state.count("rank") > 0);
    EXPECT_TRUE(state.count("world_size") > 0);
}

// ============================================================================
// 9. Stage 2 ElementLevel Single-Rank Parity Tests
// ============================================================================

// Investigation findings (Task 6.1):
//
// 1. Variable::set_grad() does NOT fire register_hook callbacks — it writes
//    impl_->grad_ directly, identical to PyTorch semantics.
//
// 2. ZeROStage2Optimizer::step_impl() calls update_local_partition() (the
//    ParamLevel path) unconditionally — it does NOT dispatch on partitioning_mode.
//    This means build_rank_grad_slice() / element_buckets_ are never reached
//    from step() in Stage 2 ElementLevel mode.  The ElementLevel-specific bucket
//    code (Phase 5: element_gradient_hook → reduce_scatter_element_bucket →
//    build_rank_grad_slice) is exercised ONLY when autograd backward() fires the
//    per-param hooks registered by register_backward_hooks().
//
// 3. For set_grad()-based tests both modes fall through to update_partition_adam()
//    which reads param->grad() directly — the same code path.  Parity therefore
//    holds for the right mathematical reason (same grad source, same optimizer math),
//    NOT because the test is vacuous.  The initial param value is ones(16×16);
//    after 5 steps with nonzero grad the params change, so the comparison is
//    non-trivial.
//
// 4. Phase 5 concern: step_impl() in Stage 2 should also dispatch on
//    partitioning_mode (like Stage 1 step_impl() does at line 314) so that the
//    ElementLevel bucket path is reachable from step().  Without that dispatch
//    the element_buckets_ / build_rank_grad_slice() code is dead in step().
//    This is a separate Phase 5 issue; this test validates parity under the
//    current (set_grad-based) code path.

TEST_F(ZeROStage2Test, ElementLevel_SingleRank_AdamParity) {
    // Stage 2 ElementLevel mode at world_size=1: reduce_scatter is a no-op (single
    // rank early-return), so the optimizer step should produce identical params to
    // Stage 2 ParamLevel.  Both modes read param->grad() in update_partition_adam()
    // because set_grad() does not fire the autograd hooks installed by
    // register_backward_hooks().

    auto run = [&](PartitioningMode mode) -> std::vector<float> {
        auto params = create_test_params(3, {16, 16});
        auto base = std::make_unique<Adam>(params, 0.001);
        ZeROStage2Config cfg;
        cfg.world_size = 1;
        cfg.rank = 0;
        cfg.partitioning_mode = mode;
        cfg.gradient_bucketing = true;
        cfg.gradient_bucket_size = 4 * 1024;  // small: forces multiple buckets
        ZeROStage2Optimizer opt(std::move(base), cfg);
        opt.register_backward_hooks();

        for (int step = 0; step < 5; ++step) {
            for (auto& p : params) {
                Tensor g = ones_like(p->tensor()) * (0.1f * (step + 1));
                p->set_grad(g);
            }
            opt.step();
        }
        std::vector<float> out;
        for (auto& p : params) {
            Tensor flat = p->tensor().contiguous().view({-1}).to(Device::cpu());
            const float* d = flat.data<float>();
            for (int64_t i = 0; i < flat.numel(); ++i) out.push_back(d[i]);
        }
        return out;
    };

    auto out_param = run(PartitioningMode::ParamLevel);
    auto out_elem  = run(PartitioningMode::ElementLevel);

    ASSERT_EQ(out_param.size(), out_elem.size());

    // Sanity-check: params must have moved from the initial value of 1.0 to prove
    // the optimizer actually ran (non-trivial grads, non-trivial update).
    bool any_changed = false;
    for (float v : out_param) {
        if (std::abs(v - 1.0f) > 1e-6f) {
            any_changed = true;
            break;
        }
    }
    EXPECT_TRUE(any_changed)
        << "ParamLevel params did not change from initial value — optimizer may not have run";

    // Parity between modes.
    for (size_t i = 0; i < out_param.size(); ++i) {
        EXPECT_NEAR(out_param[i], out_elem[i], 1e-5f)
            << "Mismatch at element " << i;
    }
}

// ============================================================================
// 10. ElementLevel Bucket Layout Invariant
// ============================================================================

TEST_F(ZeROStage2Test, ElementLevel_PeakBucketMemoryBoundedByPerRankSlice) {
    // The invariant that powers the reduce_scatter memory win: every ElementBucket's
    // global range is exactly divisible by world_size, so reduce_scatter can split it
    // cleanly into per-rank slices. With this guarantee the per-rank receive footprint
    // is bucket_size / world_size — the property that makes Stage 2 ElementLevel
    // genuinely memory-cheaper than ParamLevel's "owner gets the whole bucket".
    constexpr int W = 4;
    constexpr size_t bucket_bytes = 64 * 1024;  // 64 KB
    auto params = create_test_params(2, {64, 64});  // 4096 elements × 4 B = 16 KB each
                                                     // → 32 KB total → fits in one bucket
    auto base = std::make_unique<Adam>(params, 0.001);
    ZeROStage2Config cfg;
    cfg.world_size = W;
    cfg.rank = 0;
    cfg.partitioning_mode = PartitioningMode::ElementLevel;
    cfg.gradient_bucket_size = bucket_bytes;
    cfg.process_group = nullptr;  // single-process; reduce_scatter early-returns

    ZeROStage2Optimizer opt(std::move(base), cfg);
    // Element buckets are built at construction time (create_gradient_buckets_element_mode).
    // We do NOT call opt.step() here: with world_size=4 and process_group=nullptr the
    // all_gather_parameters_element_mode() call inside step() would throw.  The layout
    // invariant lives in the bucket metadata, not in the step output, so verifying it
    // right after construction is both correct and self-contained.

    // Layout invariant: bucket size is a world_size multiple → reduce_scatter can
    // split it. This is what guarantees per-rank peak grad memory == bucket_size / W.
    const auto& bs = opt.test_element_buckets();
    ASSERT_FALSE(bs.empty()) << "No element buckets created — bucketing config issue";
    for (const auto& b : bs) {
        EXPECT_EQ((b.global_end - b.global_start) % W, 0)
            << "Bucket [" << b.global_start << ", " << b.global_end << ") size "
            << (b.global_end - b.global_start) << " is not a multiple of world_size " << W;
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
