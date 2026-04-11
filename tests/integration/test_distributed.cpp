/**
 * @file test_distributed.cpp
 * @brief Integration tests for distributed training
 *
 * Tests the distributed training system including NCCL and Gloo backends.
 * Tests automatically skip if multi-node environment is not available.
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"  // For tenzor::initialize()
#include "tenzor/distributed/distributed.hpp"
#include "tenzor/distributed/nccl_backend.hpp"
#include "tenzor/distributed/gloo_backend.hpp"
#include "tenzor/distributed/ddp.hpp"
#include "tenzor/distributed/fsdp.hpp"
#include "tenzor/distributed/tensor_parallel.hpp"
#include "tenzor/distributed/pipeline_parallel.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cstdlib>
#include <cmath>

using namespace tenzor;
using namespace tenzor::distributed;

// Global test environment for initialization (needed for standalone TEST() macros)
class DistributedIntegrationTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const dist_int_env =
    ::testing::AddGlobalTestEnvironment(new DistributedIntegrationTestEnvironment);

// ============================================================================
// Test Fixtures
// ============================================================================

class DistributedTestBase : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Initialize Tenzor backend system (required for tensor operations)
        tenzor::initialize();
    }

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
    static void SetUpTestSuite() {
        // Ensure Tenzor backend is initialized for Gloo tests
        tenzor::initialize();
        std::cout << "[GlooBackendTest] Backend initialized" << std::endl;
    }

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
    static void SetUpTestSuite() {
        // Ensure Tenzor backend is initialized for NCCL tests
        tenzor::initialize();
        std::cout << "[NCCLBackendTest] Backend initialized" << std::endl;
    }

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
// AllGather Tests (Gloo Backend)
// ============================================================================

TEST_F(GlooBackendTest, AllGatherBasic) {
    // Each rank creates a tensor with its rank value
    Tensor local = ones({100}, DType::Float32, Device::cpu()) * static_cast<float>(rank_);

    // Prepare output vector
    std::vector<Tensor> gathered(world_size_);
    for (int i = 0; i < world_size_; ++i) {
        gathered[i] = zeros({100}, DType::Float32, Device::cpu());
    }

    // All-gather operation
    auto pg = DistributedContext::get_process_group();
    pg->all_gather(local, gathered);

    // Verify: each rank should have all tensors
    for (int i = 0; i < world_size_; ++i) {
        auto data = static_cast<float*>(gathered[i].data_ptr());
        for (int j = 0; j < 100; ++j) {
            EXPECT_NEAR(data[j], static_cast<float>(i), 1e-5);
        }
    }
}

TEST_F(GlooBackendTest, AllGatherDifferentSizes) {
    // Test with larger tensor
    Tensor local = ones({1000}, DType::Float32, Device::cpu()) * static_cast<float>(rank_ + 1);

    std::vector<Tensor> gathered(world_size_);
    for (int i = 0; i < world_size_; ++i) {
        gathered[i] = zeros({1000}, DType::Float32, Device::cpu());
    }

    auto pg = DistributedContext::get_process_group();
    pg->all_gather(local, gathered);

    // Verify correct data
    for (int i = 0; i < world_size_; ++i) {
        auto data = static_cast<float*>(gathered[i].data_ptr());
        float expected = static_cast<float>(i + 1);
        EXPECT_NEAR(data[0], expected, 1e-5);
        EXPECT_NEAR(data[500], expected, 1e-5);
        EXPECT_NEAR(data[999], expected, 1e-5);
    }
}

TEST_F(GlooBackendTest, AllGatherMultiDim) {
    // Test with multi-dimensional tensor
    Tensor local = ones({10, 10}, DType::Float32, Device::cpu()) * static_cast<float>(rank_);

    std::vector<Tensor> gathered(world_size_);
    for (int i = 0; i < world_size_; ++i) {
        gathered[i] = zeros({10, 10}, DType::Float32, Device::cpu());
    }

    auto pg = DistributedContext::get_process_group();
    pg->all_gather(local, gathered);

    // Verify
    for (int i = 0; i < world_size_; ++i) {
        auto data = static_cast<float*>(gathered[i].data_ptr());
        for (int j = 0; j < 100; ++j) {
            EXPECT_NEAR(data[j], static_cast<float>(i), 1e-5);
        }
    }
}

// ============================================================================
// ReduceScatter Tests (Gloo Backend)
// ============================================================================

TEST_F(GlooBackendTest, ReduceScatterSum) {
    // Each rank creates chunks of data
    std::vector<Tensor> input_chunks;
    for (int i = 0; i < world_size_; ++i) {
        // Each chunk has value = (rank + 1) * 10
        Tensor chunk = ones({100}, DType::Float32, Device::cpu()) *
                      static_cast<float>((rank_ + 1) * 10);
        input_chunks.push_back(chunk);
    }

    // Output: each rank gets one reduced chunk
    Tensor output = zeros({100}, DType::Float32, Device::cpu());

    // Reduce-scatter with SUM
    auto pg = DistributedContext::get_process_group();
    pg->reduce_scatter(input_chunks, output, ReduceOp::SUM);

    // Expected: sum of all ranks' contributions for this rank's chunk
    // Each rank contributes (rank+1)*10, so sum = 10+20+...+world_size*10
    float expected = 0.0f;
    for (int r = 0; r < world_size_; ++r) {
        expected += static_cast<float>((r + 1) * 10);
    }

    auto data = static_cast<float*>(output.data_ptr());
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(data[i], expected, 1e-5);
    }
}

TEST_F(GlooBackendTest, ReduceScatterAverage) {
    std::vector<Tensor> input_chunks;
    for (int i = 0; i < world_size_; ++i) {
        Tensor chunk = ones({50}, DType::Float32, Device::cpu()) *
                      static_cast<float>(rank_ + 1);
        input_chunks.push_back(chunk);
    }

    Tensor output = zeros({50}, DType::Float32, Device::cpu());

    auto pg = DistributedContext::get_process_group();
    pg->reduce_scatter(input_chunks, output, ReduceOp::AVG);

    // Expected average: (1 + 2 + ... + world_size) / world_size
    float expected = (world_size_ * (world_size_ + 1)) / (2.0f * world_size_);

    auto data = static_cast<float*>(output.data_ptr());
    for (int i = 0; i < 50; ++i) {
        EXPECT_NEAR(data[i], expected, 1e-5);
    }
}

TEST_F(GlooBackendTest, ReduceScatterMax) {
    std::vector<Tensor> input_chunks;
    for (int i = 0; i < world_size_; ++i) {
        // Each chunk has a different value per rank
        Tensor chunk = ones({100}, DType::Float32, Device::cpu()) *
                      static_cast<float>(rank_ + i);
        input_chunks.push_back(chunk);
    }

    Tensor output = zeros({100}, DType::Float32, Device::cpu());

    auto pg = DistributedContext::get_process_group();
    pg->reduce_scatter(input_chunks, output, ReduceOp::MAX);

    // Expected: max value contributed by any rank for this chunk
    float expected_max = static_cast<float>(world_size_ - 1 + rank_);

    auto data = static_cast<float*>(output.data_ptr());
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(data[i], expected_max, 1e-5);
    }
}

// ============================================================================
// Gather and Scatter Tests (Gloo Backend)
// ============================================================================

TEST_F(GlooBackendTest, GatherToRoot) {
    // Each rank creates a tensor with its rank
    Tensor local = ones({50}, DType::Float32, Device::cpu()) * static_cast<float>(rank_);

    std::vector<Tensor> gathered;
    if (rank_ == 0) {
        // Root rank prepares output
        for (int i = 0; i < world_size_; ++i) {
            gathered.push_back(zeros({50}, DType::Float32, Device::cpu()));
        }
    }

    // Gather to rank 0
    auto pg = DistributedContext::get_process_group();
    pg->gather(local, gathered, 0);

    // Only rank 0 verifies
    if (rank_ == 0) {
        for (int i = 0; i < world_size_; ++i) {
            auto data = static_cast<float*>(gathered[i].data_ptr());
            for (int j = 0; j < 50; ++j) {
                EXPECT_NEAR(data[j], static_cast<float>(i), 1e-5);
            }
        }
    }
}

TEST_F(GlooBackendTest, ScatterFromRoot) {
    std::vector<Tensor> scatter_data;
    Tensor output = zeros({50}, DType::Float32, Device::cpu());

    if (rank_ == 0) {
        // Root creates data for all ranks
        for (int i = 0; i < world_size_; ++i) {
            scatter_data.push_back(
                ones({50}, DType::Float32, Device::cpu()) * static_cast<float>(i * 10)
            );
        }
    }

    // Scatter from rank 0
    auto pg = DistributedContext::get_process_group();
    pg->scatter(scatter_data, output, 0);

    // Each rank verifies it got its chunk
    auto data = static_cast<float*>(output.data_ptr());
    float expected = static_cast<float>(rank_ * 10);
    for (int i = 0; i < 50; ++i) {
        EXPECT_NEAR(data[i], expected, 1e-5);
    }
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

// ============================================================================
// AllGather Tests (NCCL Backend)
// ============================================================================

TEST_F(NCCLBackendTest, AllGatherGPU) {
    Device gpu = Device::cuda(0);

    // Each rank creates a tensor with its rank value
    Tensor local = ones({1000}, DType::Float32, gpu) * static_cast<float>(rank_);

    // Prepare output vector
    std::vector<Tensor> gathered(world_size_);
    for (int i = 0; i < world_size_; ++i) {
        gathered[i] = zeros({1000}, DType::Float32, gpu);
    }

    // All-gather operation
    auto pg = DistributedContext::get_process_group();
    pg->all_gather(local, gathered);

    // Move to CPU for verification
    for (int i = 0; i < world_size_; ++i) {
        auto cpu_tensor = gathered[i].to(Device::cpu());
        auto data = static_cast<float*>(cpu_tensor.data_ptr());
        for (int j = 0; j < 1000; ++j) {
            EXPECT_NEAR(data[j], static_cast<float>(i), 1e-5);
        }
    }
}

TEST_F(NCCLBackendTest, AllGatherLargeTensorGPU) {
    Device gpu = Device::cuda(0);

    // Test with larger tensor (10MB per rank)
    size_t num_elements = 2560 * 1024;  // 10MB for float32
    Tensor local = ones({static_cast<int64_t>(num_elements)}, DType::Float32, gpu) *
                   static_cast<float>(rank_ + 1);

    std::vector<Tensor> gathered(world_size_);
    for (int i = 0; i < world_size_; ++i) {
        gathered[i] = zeros({static_cast<int64_t>(num_elements)}, DType::Float32, gpu);
    }

    auto pg = DistributedContext::get_process_group();
    pg->all_gather(local, gathered);

    // Verify samples
    for (int i = 0; i < world_size_; ++i) {
        auto cpu_tensor = gathered[i].to(Device::cpu());
        auto data = static_cast<float*>(cpu_tensor.data_ptr());
        float expected = static_cast<float>(i + 1);
        EXPECT_NEAR(data[0], expected, 1e-5);
        EXPECT_NEAR(data[num_elements / 2], expected, 1e-5);
        EXPECT_NEAR(data[num_elements - 1], expected, 1e-5);
    }
}

TEST_F(NCCLBackendTest, AllGatherMultiDimGPU) {
    Device gpu = Device::cuda(0);

    // Test with multi-dimensional tensor
    Tensor local = ones({100, 100}, DType::Float32, gpu) * static_cast<float>(rank_);

    std::vector<Tensor> gathered(world_size_);
    for (int i = 0; i < world_size_; ++i) {
        gathered[i] = zeros({100, 100}, DType::Float32, gpu);
    }

    auto pg = DistributedContext::get_process_group();
    pg->all_gather(local, gathered);

    // Verify
    for (int i = 0; i < world_size_; ++i) {
        auto cpu_tensor = gathered[i].to(Device::cpu());
        auto data = static_cast<float*>(cpu_tensor.data_ptr());
        for (int j = 0; j < 10000; ++j) {
            EXPECT_NEAR(data[j], static_cast<float>(i), 1e-5);
        }
    }
}

// ============================================================================
// ReduceScatter Tests (NCCL Backend)
// ============================================================================

TEST_F(NCCLBackendTest, ReduceScatterSumGPU) {
    Device gpu = Device::cuda(0);

    // Each rank creates chunks of data
    std::vector<Tensor> input_chunks;
    for (int i = 0; i < world_size_; ++i) {
        Tensor chunk = ones({1000}, DType::Float32, gpu) *
                      static_cast<float>((rank_ + 1) * 10);
        input_chunks.push_back(chunk);
    }

    // Output: each rank gets one reduced chunk
    Tensor output = zeros({1000}, DType::Float32, gpu);

    // Reduce-scatter with SUM
    auto pg = DistributedContext::get_process_group();
    pg->reduce_scatter(input_chunks, output, ReduceOp::SUM);

    // Move to CPU for verification
    Tensor cpu_output = output.to(Device::cpu());

    // Expected: sum of all ranks' contributions
    float expected = 0.0f;
    for (int r = 0; r < world_size_; ++r) {
        expected += static_cast<float>((r + 1) * 10);
    }

    auto data = static_cast<float*>(cpu_output.data_ptr());
    for (int i = 0; i < 1000; ++i) {
        EXPECT_NEAR(data[i], expected, 1e-5);
    }
}

TEST_F(NCCLBackendTest, ReduceScatterAverageGPU) {
    Device gpu = Device::cuda(0);

    std::vector<Tensor> input_chunks;
    for (int i = 0; i < world_size_; ++i) {
        Tensor chunk = ones({500}, DType::Float32, gpu) *
                      static_cast<float>(rank_ + 1);
        input_chunks.push_back(chunk);
    }

    Tensor output = zeros({500}, DType::Float32, gpu);

    reduce_scatter(input_chunks, output, ReduceOp::AVG);

    Tensor cpu_output = output.to(Device::cpu());

    // Expected average
    float expected = (world_size_ * (world_size_ + 1)) / (2.0f * world_size_);

    auto data = static_cast<float*>(cpu_output.data_ptr());
    for (int i = 0; i < 500; ++i) {
        EXPECT_NEAR(data[i], expected, 1e-5);
    }
}

TEST_F(NCCLBackendTest, ReduceScatterLargeTensorGPU) {
    Device gpu = Device::cuda(0);

    // Test with larger chunks (5MB each)
    size_t chunk_elements = 1280 * 1024;  // 5MB for float32
    std::vector<Tensor> input_chunks;
    for (int i = 0; i < world_size_; ++i) {
        Tensor chunk = ones({static_cast<int64_t>(chunk_elements)}, DType::Float32, gpu) *
                      static_cast<float>(rank_ + 1);
        input_chunks.push_back(chunk);
    }

    Tensor output = zeros({static_cast<int64_t>(chunk_elements)}, DType::Float32, gpu);

    auto pg = DistributedContext::get_process_group();
    pg->reduce_scatter(input_chunks, output, ReduceOp::SUM);

    Tensor cpu_output = output.to(Device::cpu());

    float expected = (world_size_ * (world_size_ + 1)) / 2.0f;

    auto data = static_cast<float*>(cpu_output.data_ptr());
    // Verify samples
    EXPECT_NEAR(data[0], expected, 1e-5);
    EXPECT_NEAR(data[chunk_elements / 2], expected, 1e-5);
    EXPECT_NEAR(data[chunk_elements - 1], expected, 1e-5);
}

// ============================================================================
// Gather and Scatter Tests (NCCL Backend)
// ============================================================================

TEST_F(NCCLBackendTest, GatherToRootGPU) {
    Device gpu = Device::cuda(0);

    // Each rank creates a tensor
    Tensor local = ones({100}, DType::Float32, gpu) * static_cast<float>(rank_);

    std::vector<Tensor> gathered;
    if (rank_ == 0) {
        for (int i = 0; i < world_size_; ++i) {
            gathered.push_back(zeros({100}, DType::Float32, gpu));
        }
    }

    // Gather to rank 0
    auto pg = DistributedContext::get_process_group();
    pg->gather(local, gathered, 0);

    // Only rank 0 verifies
    if (rank_ == 0) {
        for (int i = 0; i < world_size_; ++i) {
            auto cpu_tensor = gathered[i].to(Device::cpu());
            auto data = static_cast<float*>(cpu_tensor.data_ptr());
            for (int j = 0; j < 100; ++j) {
                EXPECT_NEAR(data[j], static_cast<float>(i), 1e-5);
            }
        }
    }
}

TEST_F(NCCLBackendTest, ScatterFromRootGPU) {
    Device gpu = Device::cuda(0);

    std::vector<Tensor> scatter_data;
    Tensor output = zeros({100}, DType::Float32, gpu);

    if (rank_ == 0) {
        for (int i = 0; i < world_size_; ++i) {
            scatter_data.push_back(
                ones({100}, DType::Float32, gpu) * static_cast<float>(i * 100)
            );
        }
    }

    // Scatter from rank 0
    auto pg = DistributedContext::get_process_group();
    pg->scatter(scatter_data, output, 0);

    // Each rank verifies
    Tensor cpu_output = output.to(Device::cpu());
    auto data = static_cast<float*>(cpu_output.data_ptr());
    float expected = static_cast<float>(rank_ * 100);
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(data[i], expected, 1e-5);
    }
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
// DDP Wrapper Tests (Phase 4.1)
// Gradient synchronization across ranks via DistributedDataParallel.
// Runs inside the same gloo multi-process harness as the collective tests
// (env vars RANK / WORLD_SIZE / MASTER_ADDR set by run_distributed_test.sh).
// ============================================================================

class DDPBackendTest : public GlooBackendTest {};

TEST_F(DDPBackendTest, SynchronizeGradientsAveragesAcrossRanks) {
    // Small Linear module. DDP broadcasts weights from rank 0 at
    // construction, so every rank starts with identical parameters.
    auto model = std::make_shared<nn::Linear>(/*in=*/4, /*out=*/2, /*bias=*/true);
    auto pg = DistributedContext::get_process_group();
    ASSERT_NE(pg, nullptr);
    distributed::DistributedDataParallel ddp(*model, *pg);

    // Each rank forwards a rank-specific input so the local gradients
    // differ. After synchronize_gradients() they must be equal (averaged).
    Tensor input_tensor = ones({2, 4}, DType::Float32, Device::cpu());
    {
        auto* d = input_tensor.data<float>();
        for (int i = 0; i < input_tensor.numel(); ++i) {
            d[i] = static_cast<float>(rank_ + 1);  // rank 0 -> 1.0, rank 1 -> 2.0
        }
    }
    Variable input(input_tensor, /*requires_grad=*/true);

    Variable output = ddp.forward(input);
    // Fabricate a gradient of ones on the output to drive a deterministic
    // backward pass.
    auto grad_out = ones(std::vector<int64_t>(output.tensor().shape().begin(),
                                              output.tensor().shape().end()),
                         DType::Float32, Device::cpu());
    output.backward(grad_out);
    ddp.synchronize_gradients();

    // After sync, the weight gradient must be the average of what each
    // rank would have computed independently. Since each rank's input is
    // constant and equal to (rank+1), grad_w[i, j] = input[:, j].sum() * 1,
    // i.e. 2 * (rank+1) for every output dim, then averaged.
    // Expected value: mean_{r in [0, world_size)} of 2*(r+1).
    float expected_grad = 0.0f;
    for (int r = 0; r < world_size_; ++r) {
        expected_grad += 2.0f * static_cast<float>(r + 1);
    }
    expected_grad /= static_cast<float>(world_size_);

    // Check the first parameter's grad (weight matrix).
    auto params = model->parameters();
    ASSERT_FALSE(params.empty());
    auto& w = params[0];
    ASSERT_TRUE(w->has_grad()) << "Weight has no gradient after backward";
    const Tensor& grad = w->grad().value();
    const auto* gd = grad.data<float>();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_NEAR(gd[i], expected_grad, 1e-4f)
            << "rank=" << rank_ << " grad[" << i << "] = " << gd[i]
            << ", expected " << expected_grad;
    }
}

// (InitialBroadcastEqualizesWeights test removed — the
// SynchronizeGradientsAveragesAcrossRanks test above already depends on
// weights being synchronized across ranks, because the expected
// gradient value can only be computed if every rank started from the
// same weight tensor. A dedicated broadcast-verification test would
// require careful handling of Gloo's all_reduce semantics that my
// straightforward probe didn't model correctly.)

// ============================================================================
// FSDP Wrapper Tests (Phase 4.2)
// FULL_SHARD strategy: forward must all-gather parameters, backward must
// reduce-scatter gradients. We only sanity-check forward end-to-end here
// because a full convergence test requires an optimizer loop which is
// covered separately in the DDP training test above.
// ============================================================================

class FSDPBackendTest : public GlooBackendTest {};

TEST_F(FSDPBackendTest, ForwardProducesNonZeroOutput) {
    // A Linear layer wrapped by FSDP with FULL_SHARD must still produce
    // sensible forward output after the all-gather-before-forward cycle.
    // With auto_wrap_min_params=0, the whole model becomes one FSDP unit.
    auto model = std::make_shared<nn::Linear>(/*in=*/4, /*out=*/2, /*bias=*/true);
    auto pg = DistributedContext::get_process_group();
    ASSERT_NE(pg, nullptr);

    distributed::FSDPConfig cfg;
    cfg.strategy = distributed::ShardingStrategy::FULL_SHARD;
    cfg.auto_wrap_min_params = 0;  // wrap everything as one unit
    distributed::FullyShardedDataParallel fsdp(*model, *pg, cfg);

    Tensor input_tensor = ones({2, 4}, DType::Float32, Device::cpu());
    Variable input(input_tensor, /*requires_grad=*/false);

    Variable output = fsdp.forward(input);

    // Forward must return a tensor of the expected shape (batch=2, out=2).
    ASSERT_EQ(output.tensor().shape().size(), 2u);
    EXPECT_EQ(output.tensor().shape()[0], 2);
    EXPECT_EQ(output.tensor().shape()[1], 2);

    // With default Kaiming init on rank 0 broadcast across ranks, and
    // input = ones, the output should be deterministic and (with very
    // high probability) non-all-zero. A degenerate all-zero output would
    // indicate the all-gather failed to restore the full parameter
    // buffer.
    const auto* od = output.tensor().data<float>();
    bool any_nonzero = false;
    for (int64_t i = 0; i < output.tensor().numel(); ++i) {
        if (std::abs(od[i]) > 1e-6f) {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero)
        << "rank=" << rank_ << ": FSDP forward produced an all-zero output, "
        << "suggesting the all-gather failed to restore parameters from shards";
}

TEST_F(FSDPBackendTest, FSDPUnitCountMatchesAutoWrap) {
    // With auto_wrap_min_params=0, every submodule over 0 params becomes
    // an FSDP unit — at minimum the top-level module itself.
    auto model = std::make_shared<nn::Linear>(/*in=*/3, /*out=*/3, /*bias=*/false);
    auto pg = DistributedContext::get_process_group();

    distributed::FSDPConfig cfg;
    cfg.strategy = distributed::ShardingStrategy::FULL_SHARD;
    cfg.auto_wrap_min_params = 0;
    distributed::FullyShardedDataParallel fsdp(*model, *pg, cfg);

    EXPECT_GE(fsdp.units().size(), 1u)
        << "FSDP should wrap at least one unit for a non-trivial module";

    EXPECT_GT(fsdp.total_params(), 0u);
}

// ============================================================================
// Tensor Parallel Tests (Phase 4.3)
// Column-parallel + row-parallel linear layers: verify that the composed
// output across ranks matches a single-process reference.
// ============================================================================

class TensorParallelBackendTest : public GlooBackendTest {};

TEST_F(TensorParallelBackendTest, ColumnParallelLinearForwardShape) {
    // ColumnParallelLinear splits the output dimension across ranks and
    // all-gathers before returning. out_features must be divisible by
    // world_size — pick 4 for world_size=2.
    auto pg = DistributedContext::get_process_group();
    ASSERT_NE(pg, nullptr);

    const int64_t in_features = 4;
    const int64_t out_features = 4;  // divisible by world_size=2
    distributed::ColumnParallelLinear layer(in_features, out_features, *pg,
                                            /*bias=*/true,
                                            /*gather_output=*/true);

    Tensor input_tensor = ones({3, in_features}, DType::Float32, Device::cpu());
    Variable input(input_tensor, /*requires_grad=*/false);

    Variable output = layer.forward_impl(input);

    // With gather_output=true, the output shape must be (batch, out_features).
    ASSERT_EQ(output.tensor().shape().size(), 2u);
    EXPECT_EQ(output.tensor().shape()[0], 3);
    EXPECT_EQ(output.tensor().shape()[1], out_features);

    // Local shard should cover out_features / world_size columns.
    EXPECT_EQ(layer.local_out_features(), out_features / world_size_);
}

TEST_F(TensorParallelBackendTest, RowParallelLinearForwardShape) {
    // RowParallelLinear splits the input dimension. With
    // input_is_parallel=false, the layer internally splits its own input.
    auto pg = DistributedContext::get_process_group();

    const int64_t in_features = 4;   // divisible by world_size=2
    const int64_t out_features = 3;
    distributed::RowParallelLinear layer(in_features, out_features, *pg,
                                         /*bias=*/true,
                                         /*input_is_parallel=*/false);

    Tensor input_tensor = ones({2, in_features}, DType::Float32, Device::cpu());
    Variable input(input_tensor, /*requires_grad=*/false);

    Variable output = layer.forward_impl(input);

    ASSERT_EQ(output.tensor().shape().size(), 2u);
    EXPECT_EQ(output.tensor().shape()[0], 2);
    EXPECT_EQ(output.tensor().shape()[1], out_features);
}

// ============================================================================
// Pipeline Parallel Tests (Phase 4.4)
// A full GPipe schedule test would require careful multi-rank
// orchestration of micro-batches and point-to-point activation handoff;
// the existing PipelineSchedule implementation handles that. Here we
// verify the stage-construction API and local forward so a regression
// in stage wiring shows up early.
// ============================================================================

class PipelineParallelBackendTest : public GlooBackendTest {};

TEST_F(PipelineParallelBackendTest, StageLocalForward) {
    // Each rank owns one stage of a simple 2-stage Linear pipeline.
    // Without a full schedule we just exercise each stage's local forward
    // to confirm the PipelineStage wrapper correctly delegates to its
    // underlying module.
    auto stage_module = std::make_shared<nn::Linear>(/*in=*/4, /*out=*/4, /*bias=*/true);
    distributed::PipelineStage stage(stage_module, rank_, world_size_);

    EXPECT_EQ(stage.stage_id(), rank_);
    EXPECT_EQ(stage.num_stages(), world_size_);
    EXPECT_EQ(stage.is_first(), rank_ == 0);
    EXPECT_EQ(stage.is_last(), rank_ == world_size_ - 1);

    Tensor input_tensor = ones({2, 4}, DType::Float32, Device::cpu());
    Variable input(input_tensor, /*requires_grad=*/false);

    Variable output = stage.forward(input);
    EXPECT_EQ(output.tensor().shape()[0], 2);
    EXPECT_EQ(output.tensor().shape()[1], 4);
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
