/**
 * @file test_distributed.cpp
 * @brief Tests for DistributedDataParallel and process group functionality
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/parallel/distributed_data_parallel.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <memory>

using namespace tenzor;
using namespace tenzor::nn;

// Simple test model for distributed training
class SimpleModel : public Module {
public:
    SimpleModel(int64_t input_size, int64_t hidden_size, int64_t output_size) {
        fc1_ = std::make_shared<Linear>(input_size, hidden_size);
        fc2_ = std::make_shared<Linear>(hidden_size, output_size);

        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        auto x = fc1_->forward(input);
        x = nn::relu(x);
        x = fc2_->forward(x);
        return x;
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
};

// ============================================================================
// ProcessGroup Tests
// ============================================================================

TEST(ProcessGroupTest, Construction) {
    // Valid construction
    EXPECT_NO_THROW({
        ProcessGroup pg(0, 1, "nccl");
    });

    EXPECT_NO_THROW({
        ProcessGroup pg(0, 4, "gloo");
    });

    // Invalid rank
    EXPECT_THROW({
        ProcessGroup pg(-1, 4, "nccl");
    }, std::invalid_argument);

    EXPECT_THROW({
        ProcessGroup pg(4, 4, "nccl");
    }, std::invalid_argument);

    // Invalid world size
    EXPECT_THROW({
        ProcessGroup pg(0, 0, "nccl");
    }, std::invalid_argument);

    EXPECT_THROW({
        ProcessGroup pg(0, -1, "nccl");
    }, std::invalid_argument);

    // Invalid backend
    EXPECT_THROW({
        ProcessGroup pg(0, 4, "invalid");
    }, std::invalid_argument);
}

TEST(ProcessGroupTest, Properties) {
    ProcessGroup pg(2, 8, "nccl");

    EXPECT_EQ(pg.rank(), 2);
    EXPECT_EQ(pg.world_size(), 8);
    EXPECT_EQ(pg.backend(), "nccl");
}

#if defined(TENZOR_USE_ROCM) || defined(TENZOR_USE_CUDA)
TEST(ProcessGroupTest, CommunicatorInitialization) {
    auto pg = std::make_shared<ProcessGroup>(0, 1, "nccl");

    // Generate unique ID
    ncclUniqueId id;
    ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);

    // Initialize communicator
    EXPECT_NO_THROW({
        pg->init_communicator(0, id);
    });

    // Get communicator
    ncclComm_t comm;
    EXPECT_NO_THROW({
        comm = pg->get_communicator(0);
    });
    EXPECT_NE(comm, nullptr);

    // Uninitialized device should throw
    EXPECT_THROW({
        pg->get_communicator(1);
    }, std::runtime_error);
}

TEST(ProcessGroupTest, Broadcast) {
    auto pg = std::make_shared<ProcessGroup>(0, 1, "nccl");

    // Initialize communicator
    ncclUniqueId id;
    ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
    pg->init_communicator(0, id);

    // Create test tensor
    auto tensor = randn({10, 20}, DType::Float32, Device::cuda(0));
    auto original_data = tensor.clone();

    // Broadcast (single process, should not change data)
    EXPECT_NO_THROW({
        pg->broadcast(tensor, 0, 0);
    });

    // Verify data unchanged
    auto* orig_data = original_data.data<float>();
    auto* new_data = tensor.data<float>();
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        EXPECT_FLOAT_EQ(orig_data[i], new_data[i]);
    }
}

TEST(ProcessGroupTest, AllReduce) {
    auto pg = std::make_shared<ProcessGroup>(0, 1, "nccl");

    // Initialize communicator
    ncclUniqueId id;
    ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
    pg->init_communicator(0, id);

    // Create test tensor filled with 1.0
    auto tensor = ones({10, 20}, DType::Float32, Device::cuda(0));

    // All-reduce sum (single process, should not change data)
    EXPECT_NO_THROW({
        pg->all_reduce(tensor, ncclSum, 0);
    });

    // Verify data (should still be 1.0 for single process)
    auto* data = tensor.data<float>();
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST(ProcessGroupTest, Barrier) {
    auto pg = std::make_shared<ProcessGroup>(0, 1, "nccl");

    // Initialize communicator
    ncclUniqueId id;
    ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
    pg->init_communicator(0, id);

    // Barrier should complete without error
    EXPECT_NO_THROW({
        pg->barrier();
    });
}
#endif

// ============================================================================
// GradientBucket Tests
// ============================================================================

TEST(GradientBucketTest, Construction) {
    GradientBucket bucket(25);  // 25 MB

    EXPECT_TRUE(bucket.is_empty());
    EXPECT_FALSE(bucket.is_full());
    EXPECT_EQ(bucket.size_bytes(), 0);
}

TEST(GradientBucketTest, AddGradients) {
    GradientBucket bucket(1);  // 1 MB = 1048576 bytes

    // Create parameters with gradients
    auto param1 = Variable(randn({1000, 100}, DType::Float32, Device::cpu()), true);
    auto param2 = Variable(randn({500, 500}, DType::Float32, Device::cpu()), true);

    // Set gradients
    param1.grad() = randn({1000, 100}, DType::Float32, Device::cpu());
    param2.grad() = randn({500, 500}, DType::Float32, Device::cpu());

    // Add first parameter (1000*100*4 = 400KB)
    bool is_full = bucket.add_gradient(&param1);
    EXPECT_FALSE(is_full);
    EXPECT_FALSE(bucket.is_empty());
    EXPECT_EQ(bucket.parameters().size(), 1);

    // Add second parameter (500*500*4 = 1000KB, total = 1400KB > 1MB)
    is_full = bucket.add_gradient(&param2);
    EXPECT_TRUE(is_full);
    EXPECT_EQ(bucket.parameters().size(), 2);
}

TEST(GradientBucketTest, Reset) {
    GradientBucket bucket(25);

    auto param = Variable(randn({100, 100}, DType::Float32, Device::cpu()), true);
    param.grad() = randn({100, 100}, DType::Float32, Device::cpu());

    bucket.add_gradient(&param);
    EXPECT_FALSE(bucket.is_empty());
    EXPECT_GT(bucket.size_bytes(), 0);

    bucket.reset();
    EXPECT_TRUE(bucket.is_empty());
    EXPECT_EQ(bucket.size_bytes(), 0);
}

// ============================================================================
// DistributedDataParallel Tests
// ============================================================================

TEST(DistributedDataParallelTest, Construction) {
    auto model = std::make_shared<SimpleModel>(10, 20, 5);
    auto pg = std::make_shared<ProcessGroup>(0, 1, "nccl");

#if defined(TENZOR_USE_ROCM) || defined(TENZOR_USE_CUDA)
    // Valid construction
    EXPECT_NO_THROW({
        auto ddp_model = std::make_shared<DistributedDataParallel>(
            model,
            pg,
            std::vector<int>{0},
            0
        );
    });

    // Null module
    EXPECT_THROW({
        auto ddp_model = std::make_shared<DistributedDataParallel>(
            nullptr,
            pg,
            std::vector<int>{0}
        );
    }, std::invalid_argument);

    // Null process group
    EXPECT_THROW({
        auto ddp_model = std::make_shared<DistributedDataParallel>(
            model,
            nullptr,
            std::vector<int>{0}
        );
    }, std::invalid_argument);

    // Output device not in device_ids
    EXPECT_THROW({
        auto ddp_model = std::make_shared<DistributedDataParallel>(
            model,
            pg,
            std::vector<int>{0},
            1  // Not in device_ids
        );
    }, std::invalid_argument);
#else
    // CPU-only build should throw
    EXPECT_THROW({
        auto ddp_model = std::make_shared<DistributedDataParallel>(
            model,
            pg,
            std::vector<int>{0}
        );
    }, std::runtime_error);
#endif
}

#if defined(TENZOR_USE_ROCM) || defined(TENZOR_USE_CUDA)
TEST(DistributedDataParallelTest, Properties) {
    auto model = std::make_shared<SimpleModel>(10, 20, 5);
    auto pg = std::make_shared<ProcessGroup>(0, 1, "nccl");

    auto ddp_model = std::make_shared<DistributedDataParallel>(
        model,
        pg,
        std::vector<int>{0},
        0
    );

    EXPECT_EQ(ddp_model->module(), model);
    EXPECT_EQ(ddp_model->process_group(), pg);
    EXPECT_EQ(ddp_model->device_ids().size(), 1);
    EXPECT_EQ(ddp_model->device_ids()[0], 0);
    EXPECT_EQ(ddp_model->output_device(), 0);
}

TEST(DistributedDataParallelTest, Parameters) {
    auto model = std::make_shared<SimpleModel>(10, 20, 5);
    auto pg = std::make_shared<ProcessGroup>(0, 1, "nccl");

    auto ddp_model = std::make_shared<DistributedDataParallel>(
        model,
        pg,
        std::vector<int>{0}
    );

    // DDP should expose underlying module's parameters
    auto params = ddp_model->parameters();
    auto model_params = model->parameters();

    EXPECT_EQ(params.size(), model_params.size());

    for (size_t i = 0; i < params.size(); ++i) {
        EXPECT_EQ(params[i], model_params[i]);
    }
}

TEST(DistributedDataParallelTest, NamedParameters) {
    auto model = std::make_shared<SimpleModel>(10, 20, 5);
    auto pg = std::make_shared<ProcessGroup>(0, 1, "nccl");

    auto ddp_model = std::make_shared<DistributedDataParallel>(
        model,
        pg,
        std::vector<int>{0}
    );

    auto named_params = ddp_model->named_parameters();
    auto model_named_params = model->named_parameters();

    EXPECT_EQ(named_params.size(), model_named_params.size());
}

TEST(DistributedDataParallelTest, ForwardPass) {
    auto model = std::make_shared<SimpleModel>(10, 20, 5);
    model->to(Device::cuda(0));

    auto pg = std::make_shared<ProcessGroup>(0, 1, "nccl");

    auto ddp_model = std::make_shared<DistributedDataParallel>(
        model,
        pg,
        std::vector<int>{0}
    );

    // Create input
    auto input = Variable(randn({4, 10}, DType::Float32, Device::cuda(0)), false);

    // Forward pass
    Variable output;
    EXPECT_NO_THROW({
        output = ddp_model->forward(input);
    });

    // Verify output shape
    auto output_shape = output.tensor().shape();
    EXPECT_EQ(output_shape.size(), 2);
    EXPECT_EQ(output_shape[0], 4);
    EXPECT_EQ(output_shape[1], 5);
}

TEST(DistributedDataParallelTest, TrainEvalMode) {
    auto model = std::make_shared<SimpleModel>(10, 20, 5);
    auto pg = std::make_shared<ProcessGroup>(0, 1, "nccl");

    auto ddp_model = std::make_shared<DistributedDataParallel>(
        model,
        pg,
        std::vector<int>{0}
    );

    // Default is training mode
    EXPECT_TRUE(ddp_model->is_training());

    // Set to eval mode
    ddp_model->eval();
    EXPECT_FALSE(ddp_model->is_training());
    EXPECT_FALSE(model->is_training());

    // Set to train mode
    ddp_model->train();
    EXPECT_TRUE(ddp_model->is_training());
    EXPECT_TRUE(model->is_training());
}

TEST(DistributedDataParallelTest, Join) {
    auto model = std::make_shared<SimpleModel>(10, 20, 5);
    auto pg = std::make_shared<ProcessGroup>(0, 1, "nccl");

    auto ddp_model = std::make_shared<DistributedDataParallel>(
        model,
        pg,
        std::vector<int>{0}
    );

    // Join should complete without error
    EXPECT_NO_THROW({
        ddp_model->join();
    });
}

TEST(DistributedDataParallelTest, GradientSynchronization) {
    auto model = std::make_shared<SimpleModel>(10, 20, 5);
    model->to(Device::cuda(0));

    auto pg = std::make_shared<ProcessGroup>(0, 1, "nccl");

    auto ddp_model = std::make_shared<DistributedDataParallel>(
        model,
        pg,
        std::vector<int>{0}
    );

    // Create input and target
    auto input = Variable(randn({4, 10}, DType::Float32, Device::cuda(0)), true);
    auto target = Variable(randn({4, 5}, DType::Float32, Device::cuda(0)), false);

    // Forward pass
    auto output = ddp_model->forward(input);

    // Compute loss (simple MSE)
    auto diff = output.tensor() - target.tensor();
    auto loss_tensor = (diff * diff).reshape({-1});
    auto loss = Variable(loss_tensor, true);

    // Backward pass
    loss.backward();

    // For single process, gradients should exist
    auto params = ddp_model->parameters();
    for (auto* param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}
#endif

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST(DistributedHelperTest, MakeDistributedDataParallel) {
    auto model = std::make_shared<SimpleModel>(10, 20, 5);
    auto pg = std::make_shared<ProcessGroup>(0, 1, "nccl");

#if defined(TENZOR_USE_ROCM) || defined(TENZOR_USE_CUDA)
    auto ddp_model = make_distributed_data_parallel(
        model,
        pg,
        std::vector<int>{0}
    );

    EXPECT_NE(ddp_model, nullptr);
    EXPECT_EQ(ddp_model->module(), model);
    EXPECT_EQ(ddp_model->process_group(), pg);
#else
    EXPECT_THROW({
        auto ddp_model = make_distributed_data_parallel(
            model,
            pg,
            std::vector<int>{0}
        );
    }, std::runtime_error);
#endif
}

TEST(DistributedHelperTest, InitProcessGroup) {
    // Without environment variables, should throw
    EXPECT_THROW({
        auto pg = init_process_group("nccl");
    }, std::runtime_error);

    // Set environment variables
    setenv("RANK", "0", 1);
    setenv("WORLD_SIZE", "1", 1);

    auto pg = init_process_group("nccl");
    EXPECT_NE(pg, nullptr);
    EXPECT_EQ(pg->rank(), 0);
    EXPECT_EQ(pg->world_size(), 1);

    // Cleanup
    unsetenv("RANK");
    unsetenv("WORLD_SIZE");
}

TEST(DistributedHelperTest, DestroyProcessGroup) {
    setenv("RANK", "0", 1);
    setenv("WORLD_SIZE", "1", 1);

    auto pg = init_process_group("nccl");
    EXPECT_NE(pg, nullptr);

    // Destroy should not throw
    EXPECT_NO_THROW({
        destroy_process_group(pg);
    });

    // Cleanup
    unsetenv("RANK");
    unsetenv("WORLD_SIZE");
}

// ============================================================================
// Integration Tests
// ============================================================================

#if defined(TENZOR_USE_ROCM) || defined(TENZOR_USE_CUDA)
TEST(DistributedIntegrationTest, EndToEndTraining) {
    // Setup environment
    setenv("RANK", "0", 1);
    setenv("WORLD_SIZE", "1", 1);

    // Initialize process group
    auto pg = init_process_group("nccl");

    // Create model
    auto model = std::make_shared<SimpleModel>(784, 128, 10);
    model->to(Device::cuda(0));

    // Wrap with DDP
    auto ddp_model = make_distributed_data_parallel(
        model,
        pg,
        std::vector<int>{0}
    );

    // Create dummy data
    auto input = Variable(randn({32, 784}, DType::Float32, Device::cuda(0)), true);
    auto target = Variable(randn({32, 10}, DType::Float32, Device::cuda(0)), false);

    // Forward pass
    auto output = ddp_model->forward(input);
    EXPECT_EQ(output.tensor().shape()[0], 32);
    EXPECT_EQ(output.tensor().shape()[1], 10);

    // Compute loss
    auto diff = output.tensor() - target.tensor();
    auto loss_tensor = (diff * diff).reshape({-1});
    auto loss = Variable(loss_tensor, true);

    // Backward pass
    EXPECT_NO_THROW({
        loss.backward();
    });

    // Verify gradients exist
    auto params = ddp_model->parameters();
    for (auto* param : params) {
        EXPECT_TRUE(param->has_grad());
    }

    // Cleanup
    destroy_process_group(pg);
    unsetenv("RANK");
    unsetenv("WORLD_SIZE");
}

TEST(DistributedIntegrationTest, MultipleIterations) {
    // Setup environment
    setenv("RANK", "0", 1);
    setenv("WORLD_SIZE", "1", 1);

    auto pg = init_process_group("nccl");
    auto model = std::make_shared<SimpleModel>(10, 20, 5);
    model->to(Device::cuda(0));

    auto ddp_model = make_distributed_data_parallel(model, pg, std::vector<int>{0});

    // Run multiple training iterations
    for (int iter = 0; iter < 5; ++iter) {
        // Create data
        auto input = Variable(randn({4, 10}, DType::Float32, Device::cuda(0)), true);
        auto target = Variable(randn({4, 5}, DType::Float32, Device::cuda(0)), false);

        // Forward
        auto output = ddp_model->forward(input);

        // Loss
        auto diff = output.tensor() - target.tensor();
        auto loss_tensor = (diff * diff).reshape({-1});
        auto loss = Variable(loss_tensor, true);

        // Backward
        loss.backward();

        // Zero gradients for next iteration
        ddp_model->zero_grad();
    }

    destroy_process_group(pg);
    unsetenv("RANK");
    unsetenv("WORLD_SIZE");
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    tenzor::initialize();
    return RUN_ALL_TESTS();
}
