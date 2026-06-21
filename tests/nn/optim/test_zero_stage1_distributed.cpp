/**
 * @file test_zero_stage1_distributed.cpp
 * @brief Distributed multi-GPU tests for ZeRO Stage 1 Optimizer
 *
 * Tests ZeRO Stage 1 in true distributed settings with multiple processes:
 * - Multi-GPU gradient all-reduce
 * - Multi-GPU parameter all-gather
 * - NCCL and Gloo backend compatibility
 * - Distributed training convergence
 * - Communication correctness
 * - Memory reduction verification
 * - Checkpoint compatibility across ranks
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <memory>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <string>
#include <unistd.h>  // getpid — CC.17 per-test, per-process temp path

using namespace tenzor;
using namespace tenzor::optim;
using namespace tenzor::nn;
using namespace tenzor::distributed;

// ============================================================================
// Test Fixtures
// ============================================================================

class ZeRODistributedTestBase : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }

    void SetUp() override {
        // Fixed seed so std::rand()-driven target generation in the body of
        // each test is reproducible across runs and across ranks.
        std::srand(42);

        // Check if distributed environment is available
        rank_env_ = std::getenv("RANK");
        world_size_env_ = std::getenv("WORLD_SIZE");

        if (!rank_env_ || !world_size_env_) {
            GTEST_SKIP() << "Distributed environment not available (RANK, WORLD_SIZE not set)";
        }

        rank_ = std::atoi(rank_env_);
        world_size_ = std::atoi(world_size_env_);

        if (world_size_ < 2) {
            GTEST_SKIP() << "Need at least 2 processes for distributed ZeRO tests";
        }

        // Initialize default ZeRO config
        default_config.world_size = world_size_;
        default_config.rank = rank_;
        default_config.offload_to_cpu = false;
        default_config.overlap_comm = true;
        default_config.process_group = nullptr;  // Will be set by subclass
    }

    void TearDown() override {
        if (is_initialized()) {
            barrier();  // Ensure all ranks finish together
            destroy_process_group();
        }
    }

    // Helper: Create test parameters
    auto create_test_params(size_t count, const std::vector<int64_t>& shape = {128, 128})
        -> std::vector<std::shared_ptr<Variable>> {
        std::vector<std::shared_ptr<Variable>> params;
        params.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            auto param = std::make_shared<Variable>(
                ones(shape, DType::Float32, Device::cpu()),
                true
            );
            params.push_back(param);
        }
        return params;
    }

    // Helper: Create simple MLP
    auto create_simple_mlp() -> Sequential {
        auto seq = Sequential();
        seq.add_module(std::make_shared<Linear>(784, 256))
           .add_module(std::make_shared<ReLU>())
           .add_module(std::make_shared<Linear>(256, 128))
           .add_module(std::make_shared<ReLU>())
           .add_module(std::make_shared<Linear>(128, 10));
        return seq;
    }

    const char* rank_env_{nullptr};
    const char* world_size_env_{nullptr};
    int rank_{0};
    int world_size_{1};
    ZeROStage1Config default_config;
};

// ============================================================================
// Gloo Backend Tests (CPU)
// ============================================================================

class ZeROGlooTest : public ZeRODistributedTestBase {
protected:
    void SetUp() override {
        ZeRODistributedTestBase::SetUp();
        if (!::testing::Test::IsSkipped()) {
            init_process_group("gloo");
            default_config.process_group = DistributedContext::get_process_group();
        }
    }
};

TEST_F(ZeROGlooTest, TwoRankGradientAllReduce) {
    if (world_size_ != 2) {
        GTEST_SKIP() << "This test requires exactly 2 ranks";
    }

    auto params = create_test_params(100, {64, 64});

    // Set different gradients on each rank
    for (auto& param : params) {
        float grad_value = static_cast<float>(rank_ + 1);
        auto shape_span = param->tensor().shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
        param->set_grad(full(shape, grad_value, DType::Float32, Device::cpu()));
    }

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Snapshot a locally-owned parameter before the step.
    auto before = params[0]->tensor().clone();

    // Step performs all-reduce of gradients (sum across ranks = 1 + 2 = 3) then
    // an Adam update on the summed gradient.
    optimizer.step();

    // The summed gradient is uniformly POSITIVE (3 everywhere), so Adam must
    // move every element of the parameter DOWN. A broken/identity all-reduce
    // that dropped a rank's contribution would still move it down, but a no-op
    // all-reduce that left the gradient untouched-yet-consumed, or one that
    // produced a zero/garbage gradient, fails this directional check.
    auto after = params[0]->tensor();
    auto delta = (after - before).to(DType::Float64).cpu();
    double max_elem = tenzor::max(delta).item<double>();
    EXPECT_LT(max_elem, 0.0)
        << "all-reduced positive gradient must drive every param element down";

    auto stats = optimizer.get_memory_stats();
    EXPECT_EQ(stats.num_parameters, 100);
}

TEST_F(ZeROGlooTest, FourRankParameterPartitioning) {
    if (world_size_ != 4) {
        GTEST_SKIP() << "This test requires exactly 4 ranks";
    }

    auto params = create_test_params(100, {32, 32});

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Each rank should own 25 parameters (100 / 4)
    EXPECT_EQ(optimizer.local_param_count(), 25)
        << "Rank " << rank_ << " has incorrect partition size";

    // Verify all ranks agree
    barrier();
}

TEST_F(ZeROGlooTest, GradientSynchronizationCorrectness) {
    if (world_size_ != 2) {
        GTEST_SKIP() << "This test requires exactly 2 ranks";
    }

    auto params = create_test_params(10, {8, 8});

    // Create known gradient pattern
    for (size_t i = 0; i < params.size(); ++i) {
        float grad_value = static_cast<float>(rank_ * 10 + i);
        auto shape_span = params[i]->tensor().shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
        params[i]->set_grad(full(shape, grad_value, DType::Float32, Device::cpu()));
    }

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    auto before = params[0]->tensor().clone();

    // Perform step (includes all-reduce of gradients across both ranks).
    optimizer.step();

    EXPECT_EQ(optimizer.rank(), rank_);
    EXPECT_EQ(optimizer.world_size(), world_size_);

    // Rank 0 grad for param 0 is 0, rank 1 grad is 10 → all-reduced sum is 10
    // (uniformly positive), so Adam must move param 0 strictly downward on every
    // rank. If synchronization silently used only the local gradient, rank 0
    // (whose local grad is 0) would NOT move — this catches that.
    auto after = params[0]->tensor();
    double max_elem = tenzor::max((after - before).to(DType::Float64).cpu()).item<double>();
    EXPECT_LT(max_elem, 0.0)
        << "after gradient sync, the summed positive gradient must move param 0 down "
           "on every rank (including rank 0 whose local grad was 0)";

    barrier();
}

TEST_F(ZeROGlooTest, DistributedTrainingConvergence) {
    auto model = create_simple_mlp();
    auto params = model.parameters();

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    const int num_steps = 50;
    std::vector<float> losses;

    for (int step = 0; step < num_steps; ++step) {
        optimizer.zero_grad();

        // Each rank generates different synthetic data
        auto inputs = randn({32, 784}, DType::Float32, Device::cpu());

        // LEARNABLE labels = bucket(row-mean), matching the single-process
        // integration file. `std::rand() % 10` (pure noise) made convergence a
        // coin flip — the model cannot fit random labels, so "any decrease"
        // passes on noise. With an input-dependent rule the loss must genuinely
        // drop, so the convergence assertion is meaningful.
        auto targets = empty({32}, DType::Int64, Device::cpu());
        auto* target_data = targets.data<int64_t>();
        const float* in_data = inputs.data<float>();
        for (int i = 0; i < 32; ++i) {
            double sum = 0.0;
            for (int j = 0; j < 784; ++j) sum += in_data[i * 784 + j];
            double mean = sum / 784.0;
            double squashed = 0.5 * (std::tanh(mean * 10.0) + 1.0);  // [0, 1]
            int64_t cls = static_cast<int64_t>(squashed * 10);
            if (cls >= 10) cls = 9;
            target_data[i] = cls;
        }

        auto outputs = model.forward(Variable(inputs, false));
        auto loss = cross_entropy(outputs, targets);
        losses.push_back(loss.tensor().data<float>()[0]);

        loss.backward();
        optimizer.step();
    }

    // On a learnable task the loss must drop by a meaningful margin, not just
    // be < the first sample. Require at least a 5% reduction.
    EXPECT_LT(losses.back(), losses.front() * 0.95f)
        << "Rank " << rank_ << " failed to converge on a learnable task "
        << "(initial: " << losses.front() << ", final: " << losses.back() << ")";
    EXPECT_FALSE(std::isnan(losses.back()))
        << "Rank " << rank_ << " got NaN loss";

    barrier();
}

TEST_F(ZeROGlooTest, MemoryReductionVerification) {
    if (world_size_ != 4) {
        GTEST_SKIP() << "This test requires exactly 4 ranks";
    }

    auto params = create_test_params(100, {128, 128});

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    auto stats = optimizer.get_memory_stats();

    // Each rank stores states for 25 params (100 / 4)
    EXPECT_EQ(stats.num_local_parameters, 25);
    EXPECT_EQ(stats.num_parameters, 100);

    // Memory should be ~4x less per rank (for optimizer states)
    // Note: Exact verification depends on Adam state size
    EXPECT_GT(stats.cpu_optimizer_memory + stats.gpu_optimizer_memory, 0);

    barrier();
}

TEST_F(ZeROGlooTest, StateDictSaveLoadAcrossRanks) {
    auto params = create_test_params(20, {16, 16});

    // Initialize gradients
    for (auto& param : params) {
        auto shape_span = param->tensor().shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
        param->set_grad(ones(shape, DType::Float32, Device::cpu()));
    }

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Take a step to create state
    optimizer.step();

    // Save state dict
    auto state_dict = optimizer.state_dict();

    // State dict should contain local partition
    EXPECT_GT(state_dict.size(), 0);

    // Create new optimizer and load state
    auto params2 = create_test_params(20, {16, 16});
    auto opt2 = std::make_unique<Adam>(params2, 0.001);
    ZeROStage1Optimizer optimizer2(std::move(opt2), default_config);

    EXPECT_NO_THROW(optimizer2.load_state_dict(state_dict));

    barrier();
}

TEST_F(ZeROGlooTest, CheckpointCompatibilityAcrossRanks) {
    auto params = create_test_params(20, {16, 16});

    // Set gradients
    for (auto& param : params) {
        auto shape_span = param->tensor().shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
        param->set_grad(ones(shape, DType::Float32, Device::cpu()));
    }

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Train for a few steps
    for (int i = 0; i < 10; ++i) {
        optimizer.step();
        for (auto& param : params) {
            auto shape_span = param->tensor().shape();
            std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
            param->set_grad(ones(shape, DType::Float32, Device::cpu()));
        }
    }

    // Save checkpoint (each rank saves its partition).  CC.17: include PID
    // and test name so parallel ctest invocations don't race.
    std::string checkpoint_path = (std::filesystem::temp_directory_path() /
        ("tenzor_zero_test_rank_" + std::to_string(rank_) + "_" +
         std::to_string(::getpid()) + "_" +
         std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()))).string();
    EXPECT_NO_THROW(optimizer.save_checkpoint(checkpoint_path));

    barrier();

    // Load checkpoint
    auto params2 = create_test_params(20, {16, 16});
    auto opt2 = std::make_unique<Adam>(params2, 0.001);
    ZeROStage1Optimizer optimizer2(std::move(opt2), default_config);

    EXPECT_NO_THROW(optimizer2.load_checkpoint(checkpoint_path));

    barrier();
}

// ============================================================================
// NCCL Backend Tests (GPU)
// ============================================================================

#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)

class ZeRONCCLTest : public ZeRODistributedTestBase {
protected:
    void SetUp() override {
        ZeRODistributedTestBase::SetUp();
        if (!::testing::Test::IsSkipped()) {
            // Audit-5 (EDGE): this SetUp previously wrapped init_process_group(
            // "nccl") in `try { ... } catch (...) { GTEST_SKIP("CUDA/ROCm not
            // available"); }`, a catch-all that made every NCCL init failure
            // (missing kernel, comm-build error, topology mismatch) look like a
            // clean "no GPU" skip. The genuine preconditions — RANK/WORLD_SIZE
            // set and world_size >= 2 — are already enforced in the base
            // SetUp() above, and this subclass only compiles under
            // TENZOR_USE_CUDA/ROCM, so a real NCCL backend is expected when we
            // reach here. Let init_process_group failures propagate as
            // failures instead of being buried as availability skips.
            Device gpu_dev = Device::cuda(rank_);
            (void)gpu_dev;
            init_process_group("nccl");
            default_config.process_group = DistributedContext::get_process_group();
        }
    }

    // Helper: Create GPU parameters
    auto create_gpu_params(size_t count, const std::vector<int64_t>& shape = {128, 128})
        -> std::vector<std::shared_ptr<Variable>> {
        std::vector<std::shared_ptr<Variable>> params;
        params.reserve(count);
        Device gpu_dev = Device::cuda(rank_);
        for (size_t i = 0; i < count; ++i) {
            auto param = std::make_shared<Variable>(
                ones(shape, DType::Float32, gpu_dev),
                true
            );
            params.push_back(param);
        }
        return params;
    }
};

TEST_F(ZeRONCCLTest, TwoGPUGradientAllReduce) {
    if (world_size_ != 2) {
        GTEST_SKIP() << "This test requires exactly 2 GPUs";
    }

    auto params = create_gpu_params(50, {64, 64});

    // Set different gradients on each GPU
    for (auto& param : params) {
        float grad_value = static_cast<float>(rank_ + 1);
        auto shape_span = param->tensor().shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
        param->set_grad(full(shape, grad_value, DType::Float32, Device::cuda(rank_)));
    }

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Step performs NCCL all-reduce
    EXPECT_NO_THROW(optimizer.step());

    barrier();
}

TEST_F(ZeRONCCLTest, FourGPUParameterPartitioning) {
    if (world_size_ != 4) {
        GTEST_SKIP() << "This test requires exactly 4 GPUs";
    }

    auto params = create_gpu_params(100, {32, 32});

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Each GPU should own 25 parameters
    EXPECT_EQ(optimizer.local_param_count(), 25)
        << "GPU " << rank_ << " has incorrect partition size";

    // Verify memory reduction
    auto stats = optimizer.get_memory_stats();
    EXPECT_EQ(stats.num_local_parameters, 25);

    barrier();
}

TEST_F(ZeRONCCLTest, EightGPUScaling) {
    if (world_size_ != 8) {
        GTEST_SKIP() << "This test requires exactly 8 GPUs";
    }

    auto params = create_gpu_params(800, {16, 16});

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Each GPU should own 100 parameters
    EXPECT_EQ(optimizer.local_param_count(), 100)
        << "GPU " << rank_ << " has incorrect partition size";

    // Set gradients and perform step
    for (auto& param : params) {
        auto shape_span = param->tensor().shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
        param->set_grad(ones(shape, DType::Float32, Device::cuda(rank_)));
    }

    EXPECT_NO_THROW(optimizer.step());

    barrier();
}

TEST_F(ZeRONCCLTest, GPUMemoryReduction) {
    if (world_size_ != 4) {
        GTEST_SKIP() << "This test requires exactly 4 GPUs";
    }

    auto params = create_gpu_params(100, {128, 128});

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    auto stats = optimizer.get_memory_stats();

    // Verify 4x memory reduction for optimizer states
    EXPECT_EQ(stats.num_local_parameters, 25);  // 100 / 4
    EXPECT_EQ(stats.num_parameters, 100);

    // GPU memory should be used for states
    EXPECT_GT(stats.gpu_optimizer_memory, 0);

    barrier();
}

TEST_F(ZeRONCCLTest, DistributedGPUTraining) {
    // Create simple model on GPU
    auto model = Sequential();
    model.add_module(std::make_shared<Linear>(784, 256))
         .add_module(std::make_shared<ReLU>())
         .add_module(std::make_shared<Linear>(256, 10));

    // Move model to GPU
    Device gpu_dev = Device::cuda(rank_);
    auto params = model.parameters();
    for (auto& param : params) {
        param->tensor() = param->tensor().to(gpu_dev);
    }

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    const int num_steps = 30;
    std::vector<float> losses;

    for (int step = 0; step < num_steps; ++step) {
        optimizer.zero_grad();

        // Generate GPU data
        auto inputs = randn({32, 784}, DType::Float32, gpu_dev);

        // LEARNABLE labels = bucket(row-mean), computed on a host copy of the
        // inputs (GPU memory can't be indexed on the host), then moved to GPU.
        // `std::rand() % 10` was pure noise — unlearnable, so any convergence
        // assertion passed on chance.
        auto inputs_cpu = inputs.to(Device::cpu());
        const float* in_data = inputs_cpu.data<float>();
        auto targets_cpu = empty({32}, DType::Int64, Device::cpu());
        auto* target_data = targets_cpu.data<int64_t>();
        for (int i = 0; i < 32; ++i) {
            double sum = 0.0;
            for (int j = 0; j < 784; ++j) sum += in_data[i * 784 + j];
            double mean = sum / 784.0;
            double squashed = 0.5 * (std::tanh(mean * 10.0) + 1.0);  // [0, 1]
            int64_t cls = static_cast<int64_t>(squashed * 10);
            if (cls >= 10) cls = 9;
            target_data[i] = cls;
        }
        auto targets = targets_cpu.to(gpu_dev);

        auto outputs = model.forward(Variable(inputs, false));
        auto loss = cross_entropy(outputs, targets);
        auto loss_cpu = Variable(loss.tensor().to(Device::cpu()), false);
        losses.push_back(loss_cpu.tensor().data<float>()[0]);

        loss.backward();
        optimizer.step();
    }

    // On a learnable task the loss must drop by a meaningful margin.
    EXPECT_LT(losses.back(), losses.front() * 0.95f)
        << "GPU " << rank_ << " failed to converge on a learnable task "
        << "(initial: " << losses.front() << ", final: " << losses.back() << ")";
    EXPECT_FALSE(std::isnan(losses.back()));

    barrier();
}

TEST_F(ZeRONCCLTest, CPUOffloadWithGPUTraining) {
    auto params = create_gpu_params(50, {64, 64});

    // Enable CPU offload
    default_config.offload_to_cpu = true;
    default_config.cpu_offload_threshold = 1024;

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Set gradients
    for (auto& param : params) {
        auto shape_span = param->tensor().shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
        param->set_grad(ones(shape, DType::Float32, Device::cuda(rank_)));
    }

    // Step should work with offload enabled
    EXPECT_NO_THROW(optimizer.step());

    // Verify offload is active
    EXPECT_TRUE(optimizer.is_cpu_offload_enabled());

    barrier();
}

TEST_F(ZeRONCCLTest, LargeModelGPUMemorySavings) {
    if (world_size_ != 4) {
        GTEST_SKIP() << "This test requires exactly 4 GPUs";
    }

    // Create large model
    auto params = create_gpu_params(1000, {256, 256});

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    auto stats = optimizer.get_memory_stats();

    // Each GPU stores 1/4 of optimizer states
    EXPECT_EQ(stats.num_local_parameters, 250);  // 1000 / 4

    // Memory savings should be significant
    // (can't measure exact GPU memory without CUDA calls)
    EXPECT_GT(stats.gpu_optimizer_memory, 0);

    barrier();
}

#endif  // TENZOR_USE_CUDA || TENZOR_USE_ROCM

// ============================================================================
// Edge Cases for Distributed Training
// ============================================================================

TEST_F(ZeROGlooTest, UnevenParameterDistribution) {
    if (world_size_ != 4) {
        GTEST_SKIP() << "This test requires exactly 4 ranks";
    }

    // 103 params: ranks get [26, 26, 26, 25]
    auto params = create_test_params(103, {16, 16});

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    std::vector<size_t> expected = {26, 26, 26, 25};
    EXPECT_EQ(optimizer.local_param_count(), expected[rank_])
        << "Rank " << rank_ << " has wrong partition size";

    barrier();
}

TEST_F(ZeROGlooTest, FewerParametersThanRanks) {
    if (world_size_ != 4) {
        GTEST_SKIP() << "This test requires exactly 4 ranks";
    }

    // Only 2 parameters for 4 ranks
    auto params = create_test_params(2, {8, 8});

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // First 2 ranks get 1 param each, others get 0
    size_t expected_count = (rank_ < 2) ? 1 : 0;
    EXPECT_EQ(optimizer.local_param_count(), expected_count);

    barrier();
}

TEST_F(ZeROGlooTest, EmptyGradientsOnSomeRanks) {
    auto params = create_test_params(10, {8, 8});

    // Only rank 0 sets gradients
    if (rank_ == 0) {
        for (auto& param : params) {
            auto shape_span = param->tensor().shape();
            std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
            param->set_grad(ones(shape, DType::Float32, Device::cpu()));
        }
    }

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Should handle mixed gradient state gracefully
    EXPECT_NO_THROW(optimizer.step());

    barrier();
}

TEST_F(ZeROGlooTest, BarrierSynchronization) {
    auto params = create_test_params(10, {8, 8});

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // All ranks should reach this point together
    barrier();

    // Simulate work that takes different time on each rank
    for (int i = 0; i < rank_ * 10; ++i) {
        volatile int dummy = i * i;
        (void)dummy;
    }

    // Synchronize again — if any rank failed to reach the barrier the collective
    // would hang/throw; reaching here means all ranks synchronized. Assert the
    // optimizer's view of the group is the expected one (a real post-barrier
    // invariant rather than an unconditional SUCCEED).
    barrier();

    EXPECT_EQ(optimizer.rank(), rank_);
    EXPECT_EQ(optimizer.world_size(), world_size_);
}

TEST_F(ZeROGlooTest, ConcurrentOptimizationSteps) {
    auto params = create_test_params(20, {16, 16});

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    auto before = params[0]->tensor().clone();

    // All ranks perform identical steps concurrently with all-ones gradients.
    for (int step = 0; step < 10; ++step) {
        for (auto& param : params) {
            auto shape_span = param->tensor().shape();
            std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
            param->set_grad(ones(shape, DType::Float32, Device::cpu()));
        }

        optimizer.step();
        optimizer.zero_grad();
    }

    barrier();

    // The all-reduced gradient is uniformly positive every step, so 10 Adam
    // updates must move param 0 strictly down — verifying the concurrent steps
    // actually applied updates rather than silently no-op'ing.
    auto after = params[0]->tensor();
    double max_elem = tenzor::max((after - before).to(DType::Float64).cpu()).item<double>();
    EXPECT_LT(max_elem, 0.0)
        << "10 concurrent optimization steps must move param 0 downward";
}

// =====================================================================
// ElementLevel-mode multi-rank tests
// =====================================================================

TEST_F(ZeROGlooTest, ElementLevel_ConsistentParamsAcrossRanks) {
    // After ElementLevel steps with deterministic grads, every rank should hold the
    // same final param values — proves that all_gather_parameters_element_mode
    // correctly distributes each rank's slice to all peers.
    if (world_size_ != 2 && world_size_ != 4) {
        GTEST_SKIP() << "Test requires world_size 2 or 4";
    }

    // Three params with mixed shapes; total elements 16+8+30=54 (divisible by 2 and
    // padded for 4: 56).
    std::vector<std::shared_ptr<Variable>> params;
    params.push_back(std::make_shared<Variable>(
        ones({4, 4}, DType::Float32, Device::cpu()), true));
    params.push_back(std::make_shared<Variable>(
        ones({8}, DType::Float32, Device::cpu()), true));
    params.push_back(std::make_shared<Variable>(
        ones({2, 3, 5}, DType::Float32, Device::cpu()), true));

    auto base = std::make_unique<Adam>(params, 0.001);
    auto cfg = default_config;
    cfg.partitioning_mode = PartitioningMode::ElementLevel;
    ZeROStage1Optimizer optimizer(std::move(base), cfg);

    // Deterministic grads: every rank computes the SAME grad for every param
    // (same as a normal data-parallel setup before all_reduce). Steps are 5.
    for (int step = 0; step < 5; ++step) {
        for (auto& p : params) {
            auto shape_span = p->tensor().shape();
            std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
            Tensor g = full(shape, 0.1f * (step + 1), DType::Float32, Device::cpu());
            p->set_grad(g);
        }
        optimizer.step();
    }

    // After all_gather every rank has the same final param values. Sanity-check by
    // verifying values are finite and non-zero.
    for (auto& p : params) {
        Tensor flat = p->tensor().contiguous().view({-1});
        const float* d = flat.data<float>();
        for (int64_t i = 0; i < flat.numel(); ++i) {
            EXPECT_TRUE(std::isfinite(d[i])) << "param has non-finite value at " << i;
        }
    }

    // Cross-rank consistency check: broadcast rank 0's final param[0] to all ranks
    // and assert local match. The broadcast uses the same process group.
    //
    // Snapshot the local param BEFORE the broadcast modifies it. We need an
    // independent copy (clone()) because contiguous() on an already-contiguous
    // tensor returns a shared-storage view — broadcast(probe, 0) would otherwise
    // mutate the underlying param and make local_snapshot == probe trivially.
    Tensor local_snapshot = params[0]->tensor().clone();
    Tensor probe = params[0]->tensor().clone();
    default_config.process_group->broadcast(probe, 0);
    Tensor diff = local_snapshot - probe;
    Tensor abs_diff = abs(diff);
    Tensor max_diff = max(abs_diff);
    float max_val = max_diff.to(Device::cpu()).data<float>()[0];
    EXPECT_LT(max_val, 1e-5f) << "Rank " << rank_ << " params diverge from rank 0";

    barrier();
}

TEST_F(ZeROGlooTest, ElementLevel_StateDictRoundTrip) {
    // ElementLevel state_dict / load_state_dict should be a no-op round trip.
    if (world_size_ < 2) {
        GTEST_SKIP() << "Test requires world_size >= 2";
    }

    std::vector<std::shared_ptr<Variable>> params;
    params.push_back(std::make_shared<Variable>(
        ones({16, 16}, DType::Float32, Device::cpu()), true));

    auto base = std::make_unique<Adam>(params, 0.001);
    auto cfg = default_config;
    cfg.partitioning_mode = PartitioningMode::ElementLevel;
    ZeROStage1Optimizer optimizer(std::move(base), cfg);

    // Run one step to populate optimizer state (m, v).
    params[0]->set_grad(ones({16, 16}, DType::Float32, Device::cpu()) * 0.1f);
    optimizer.step();

    auto state_before = optimizer.state_dict();
    optimizer.load_state_dict(state_before);
    auto state_after = optimizer.state_dict();

    EXPECT_EQ(state_before.size(), state_after.size());
    for (const auto& [k, v] : state_before) {
        ASSERT_TRUE(state_after.count(k)) << "Missing key after round-trip: " << k;
    }

    barrier();
}
