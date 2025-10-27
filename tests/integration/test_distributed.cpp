/**
 * @file test_distributed.cpp
 * @brief Integration tests for distributed training
 *
 * Tests the distributed training system including NCCL and Gloo backends.
 * Tests automatically skip if multi-node environment is not available.
 */

#include <gtest/gtest.h>
#include "tenzor/distributed/distributed.hpp"
#include "tenzor/distributed/nccl_backend.hpp"
#include "tenzor/distributed/gloo_backend.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cstdlib>
#include <cmath>

using namespace tenzor;
using namespace tenzor::distributed;

// ============================================================================
// Test Fixtures
// ============================================================================

class DistributedTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if distributed environment is available
        rank_env_ = std::getenv("RANK");
        world_size_env_ = std::getenv("WORLD_SIZE");

        if (!rank_env_ || !world_size_env_) {
            GTEST_SKIP() << "Distributed environment not available (RANK, WORLD_SIZE not set)";
        }

        rank_ = std::atoi(rank_env_);
        world_size_ = std::atoi(world_size_env_);

        if (world_size_ < 2) {
            GTEST_SKIP() << "Need at least 2 processes for distributed tests";
        }
    }

    void TearDown() override {
        if (is_initialized()) {
            destroy_process_group();
        }
    }

    const char* rank_env_{nullptr};
    const char* world_size_env_{nullptr};
    int rank_{0};
    int world_size_{1};
};

class GlooBackendTest : public DistributedTestBase {
protected:
    void SetUp() override {
        DistributedTestBase::SetUp();
        if (!::testing::Test::IsSkipped()) {
            init_process_group("gloo");
        }
    }
};

#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
class NCCLBackendTest : public DistributedTestBase {
protected:
    void SetUp() override {
        DistributedTestBase::SetUp();
        if (!::testing::Test::IsSkipped()) {
            // Check if CUDA/ROCm is available
            try {
                Device gpu_dev = Device::cuda(0);
                init_process_group("nccl");
            } catch (...) {
                GTEST_SKIP() << "CUDA/ROCm not available for NCCL tests";
            }
        }
    }
};
#endif

// ============================================================================
// ProcessGroup Tests
// ============================================================================

TEST(ProcessGroupTest, CreateFromExplicitParams) {
    // Test creating process group without environment variables
    auto pg = ProcessGroup::create_process_group(
        distributed::Backend::GLOO,
        0,  // rank
        1,  // world_size
        "localhost",
        29500
    );

    EXPECT_EQ(pg->rank(), 0);
    EXPECT_EQ(pg->world_size(), 1);
    EXPECT_EQ(pg->backend(), distributed::Backend::GLOO);
}

TEST(ProcessGroupTest, InvalidRank) {
    EXPECT_THROW(
        ProcessGroup::create_process_group(distributed::Backend::GLOO, -1, 2),
        std::invalid_argument
    );

    EXPECT_THROW(
        ProcessGroup::create_process_group(distributed::Backend::GLOO, 2, 2),
        std::invalid_argument
    );
}

TEST(ProcessGroupTest, InvalidWorldSize) {
    EXPECT_THROW(
        ProcessGroup::create_process_group(distributed::Backend::GLOO, 0, 0),
        std::invalid_argument
    );

    EXPECT_THROW(
        ProcessGroup::create_process_group(distributed::Backend::GLOO, 0, -1),
        std::invalid_argument
    );
}

TEST(BackendConversionTest, BackendToString) {
    EXPECT_EQ(backend_to_string(distributed::Backend::NCCL), "nccl");
    EXPECT_EQ(backend_to_string(distributed::Backend::GLOO), "gloo");
    EXPECT_EQ(backend_to_string(distributed::Backend::MPI), "mpi");
}

TEST(BackendConversionTest, StringToBackend) {
    EXPECT_EQ(string_to_backend("nccl"), distributed::Backend::NCCL);
    EXPECT_EQ(string_to_backend("gloo"), distributed::Backend::GLOO);
    EXPECT_EQ(string_to_backend("mpi"), distributed::Backend::MPI);

    EXPECT_THROW(string_to_backend("invalid"), std::invalid_argument);
}

// ============================================================================
// Gloo Backend Tests
// ============================================================================

TEST_F(GlooBackendTest, AllReduceSum) {
    Tensor data = ones({10, 10}, DType::Float32, Device::cpu()) * (rank_ + 1);

    all_reduce(data, ReduceOp::SUM);

    // Expected: sum of (1 + 2 + ... + world_size)
    float expected_sum = (world_size_ * (world_size_ + 1)) / 2.0f;

    auto data_ptr = static_cast<float*>(data.data_ptr());
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(data_ptr[i], expected_sum, 1e-5);
    }
}

TEST_F(GlooBackendTest, AllReduceAverage) {
    Tensor data = ones({10, 10}, DType::Float32, Device::cpu()) * (rank_ + 1);

    all_reduce(data, ReduceOp::AVG);

    // Expected: average of (1 + 2 + ... + world_size)
    float expected_avg = (world_size_ + 1) / 2.0f;

    auto data_ptr = static_cast<float*>(data.data_ptr());
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(data_ptr[i], expected_avg, 1e-5);
    }
}

TEST_F(GlooBackendTest, AllReduceMax) {
    Tensor data = ones({10, 10}, DType::Float32, Device::cpu()) * (rank_ + 1);

    all_reduce(data, ReduceOp::MAX);

    float expected_max = static_cast<float>(world_size_);

    auto data_ptr = static_cast<float*>(data.data_ptr());
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(data_ptr[i], expected_max, 1e-5);
    }
}

TEST_F(GlooBackendTest, AllReduceMin) {
    Tensor data = ones({10, 10}, DType::Float32, Device::cpu()) * (rank_ + 1);

    all_reduce(data, ReduceOp::MIN);

    float expected_min = 1.0f;

    auto data_ptr = static_cast<float*>(data.data_ptr());
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(data_ptr[i], expected_min, 1e-5);
    }
}

TEST_F(GlooBackendTest, Broadcast) {
    Tensor data;

    if (rank_ == 0) {
        data = ones({10, 10}, DType::Float32, Device::cpu()) * 42.0f;
    } else {
        data = zeros({10, 10}, DType::Float32, Device::cpu());
    }

    broadcast(data, 0);

    auto data_ptr = static_cast<float*>(data.data_ptr());
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(data_ptr[i], 42.0f, 1e-5);
    }
}

TEST_F(GlooBackendTest, Barrier) {
    // Test barrier synchronization
    EXPECT_NO_THROW(barrier());
}

TEST_F(GlooBackendTest, GetRankAndWorldSize) {
    EXPECT_EQ(get_rank(), rank_);
    EXPECT_EQ(get_world_size(), world_size_);
}

// ============================================================================
// NCCL Backend Tests (GPU only)
// ============================================================================

#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)

TEST_F(NCCLBackendTest, AllReduceSumGPU) {
    Device gpu = Device::cuda(0);
    Tensor data = ones({10, 10}, DType::Float32, gpu) * (rank_ + 1);

    all_reduce(data, ReduceOp::SUM);

    // Copy to CPU for verification
    Tensor cpu_data = data.to(Device::cpu());

    float expected_sum = (world_size_ * (world_size_ + 1)) / 2.0f;

    auto data_ptr = static_cast<float*>(cpu_data.data_ptr());
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(data_ptr[i], expected_sum, 1e-5);
    }
}

TEST_F(NCCLBackendTest, AllReduceAverageGPU) {
    Device gpu = Device::cuda(0);
    Tensor data = ones({10, 10}, DType::Float32, gpu) * (rank_ + 1);

    all_reduce(data, ReduceOp::AVG);

    Tensor cpu_data = data.to(Device::cpu());

    float expected_avg = (world_size_ + 1) / 2.0f;

    auto data_ptr = static_cast<float*>(cpu_data.data_ptr());
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(data_ptr[i], expected_avg, 1e-5);
    }
}

TEST_F(NCCLBackendTest, BroadcastGPU) {
    Device gpu = Device::cuda(0);
    Tensor data;

    if (rank_ == 0) {
        data = ones({10, 10}, DType::Float32, gpu) * 42.0f;
    } else {
        data = zeros({10, 10}, DType::Float32, gpu);
    }

    broadcast(data, 0);

    Tensor cpu_data = data.to(Device::cpu());

    auto data_ptr = static_cast<float*>(cpu_data.data_ptr());
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(data_ptr[i], 42.0f, 1e-5);
    }
}

TEST_F(NCCLBackendTest, LargeTensorAllReduce) {
    // Test with large tensor (100MB)
    Device gpu = Device::cuda(0);
    size_t num_elements = 25 * 1024 * 1024;  // 100MB for float32
    Tensor data = ones({static_cast<int64_t>(num_elements)}, DType::Float32, gpu) * (rank_ + 1);

    all_reduce(data, ReduceOp::SUM);

    Tensor cpu_data = data.to(Device::cpu());

    float expected_sum = (world_size_ * (world_size_ + 1)) / 2.0f;

    auto data_ptr = static_cast<float*>(cpu_data.data_ptr());

    // Verify first, middle, and last elements
    EXPECT_NEAR(data_ptr[0], expected_sum, 1e-5);
    EXPECT_NEAR(data_ptr[num_elements / 2], expected_sum, 1e-5);
    EXPECT_NEAR(data_ptr[num_elements - 1], expected_sum, 1e-5);
}

#endif // TENZOR_USE_CUDA || TENZOR_USE_ROCM

// ============================================================================
// Gradient Bucket Tests
// ============================================================================

TEST(GradientBucketTest, AddGradient) {
    GradientBucket bucket(1);  // 1 MB

    Tensor grad1 = randn({100, 100}, DType::Float32, Device::cpu());
    Tensor grad2 = randn({100, 100}, DType::Float32, Device::cpu());

    EXPECT_FALSE(bucket.is_full());
    EXPECT_TRUE(bucket.is_empty());

    bucket.add_gradient(grad1);
    EXPECT_FALSE(bucket.is_empty());

    bucket.add_gradient(grad2);

    EXPECT_EQ(bucket.gradients().size(), 2);
}

TEST(GradientBucketTest, BucketFull) {
    GradientBucket bucket(1);  // 1 MB = 1048576 bytes

    // Add gradients until bucket is full
    // Each gradient: 100 * 100 * 4 bytes = 40000 bytes
    // Need ~26 gradients to fill 1MB

    bool is_full = false;
    for (int i = 0; i < 30; ++i) {
        Tensor grad = randn({100, 100}, DType::Float32, Device::cpu());
        is_full = bucket.add_gradient(grad);
        if (is_full) {
            break;
        }
    }

    EXPECT_TRUE(is_full);
}

TEST(GradientBucketTest, Reset) {
    GradientBucket bucket(1);

    Tensor grad = randn({100, 100}, DType::Float32, Device::cpu());
    bucket.add_gradient(grad);

    EXPECT_FALSE(bucket.is_empty());

    bucket.reset();

    EXPECT_TRUE(bucket.is_empty());
    EXPECT_EQ(bucket.size_bytes(), 0);
}

// ============================================================================
// DistributedContext Tests
// ============================================================================

TEST(DistributedContextTest, InitializeAndFinalize) {
    if (!DistributedContext::is_initialized()) {
        DistributedContext::initialize(distributed::Backend::GLOO, 0, 1, "localhost", 29500);
        EXPECT_TRUE(DistributedContext::is_initialized());

        auto pg = DistributedContext::get_process_group();
        EXPECT_NE(pg, nullptr);
        EXPECT_EQ(pg->rank(), 0);
        EXPECT_EQ(pg->world_size(), 1);

        DistributedContext::finalize();
        EXPECT_FALSE(DistributedContext::is_initialized());
    }
}

TEST(DistributedContextTest, DoubleInitialize) {
    if (!DistributedContext::is_initialized()) {
        DistributedContext::initialize(distributed::Backend::GLOO, 0, 1, "localhost", 29500);

        EXPECT_THROW(
            DistributedContext::initialize(distributed::Backend::GLOO, 0, 1, "localhost", 29500),
            std::runtime_error
        );

        DistributedContext::finalize();
    }
}

TEST(DistributedContextTest, AccessWithoutInitialize) {
    if (DistributedContext::is_initialized()) {
        DistributedContext::finalize();
    }

    EXPECT_THROW(
        DistributedContext::get_process_group(),
        std::runtime_error
    );

    EXPECT_THROW(
        DistributedContext::get_rank(),
        std::runtime_error
    );

    EXPECT_THROW(
        DistributedContext::get_world_size(),
        std::runtime_error
    );
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n";
    std::cout << "=================================================================\n";
    std::cout << "Distributed Training Tests\n";
    std::cout << "=================================================================\n";

    const char* rank_env = std::getenv("RANK");
    const char* world_size_env = std::getenv("WORLD_SIZE");

    if (!rank_env || !world_size_env) {
        std::cout << "WARNING: RANK and WORLD_SIZE not set.\n";
        std::cout << "Multi-process distributed tests will be skipped.\n";
        std::cout << "\nTo run distributed tests:\n";
        std::cout << "  export RANK=0 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=29500\n";
        std::cout << "  ./test_distributed &\n";
        std::cout << "  export RANK=1 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=29500\n";
        std::cout << "  ./test_distributed\n";
    } else {
        std::cout << "Running in distributed mode:\n";
        std::cout << "  RANK=" << rank_env << "\n";
        std::cout << "  WORLD_SIZE=" << world_size_env << "\n";
    }

    std::cout << "=================================================================\n\n";

    return RUN_ALL_TESTS();
}
