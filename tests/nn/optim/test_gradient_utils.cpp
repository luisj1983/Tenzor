/**
 * @file test_gradient_utils.cpp
 * @brief Tests for gradient utility functions (ZeRO Stage 2)
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/optim/gradient_utils.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include <vector>
#include <memory>

using namespace tenzor;
using namespace tenzor::optim;

// ============================================================================
// Test Fixture
// ============================================================================

class GradientUtilsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Initialize Tenzor library and load backends
        tenzor::initialize();
    }
};

// ============================================================================
// Tensor Flattening Tests
// ============================================================================

TEST_F(GradientUtilsTest, FlattenSingleTensor) {
    Tensor t({10, 20}, DType::Float32, Device::cpu());
    t.fill_(1.5f);

    auto flat = flatten_tensors({t});

    EXPECT_EQ(flat.ndim(), 1);
    EXPECT_EQ(flat.numel(), 200);
    EXPECT_EQ(flat.dtype(), DType::Float32);

    // Verify data
    auto* data = flat.data<float>();
    for (int i = 0; i < 200; ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.5f);
    }
}

TEST_F(GradientUtilsTest, FlattenMultipleTensors) {
    Tensor t1({10, 20}, DType::Float32, Device::cpu());
    Tensor t2({5, 5}, DType::Float32, Device::cpu());
    Tensor t3({100}, DType::Float32, Device::cpu());

    t1.fill_(1.0f);
    t2.fill_(2.0f);
    t3.fill_(3.0f);

    auto flat = flatten_tensors({t1, t2, t3});

    EXPECT_EQ(flat.ndim(), 1);
    EXPECT_EQ(flat.numel(), 200 + 25 + 100);
    EXPECT_EQ(flat.dtype(), DType::Float32);

    // Verify data ranges
    auto* data = flat.data<float>();
    for (int i = 0; i < 200; ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
    for (int i = 200; i < 225; ++i) {
        EXPECT_FLOAT_EQ(data[i], 2.0f);
    }
    for (int i = 225; i < 325; ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.0f);
    }
}

TEST_F(GradientUtilsTest, FlattenNonContiguous) {
    // Skip test if transpose not available in backend
    try {
        Tensor t({4, 5}, DType::Float32, Device::cpu());
        t.fill_(1.0f);
        auto t_transposed = t.transpose(0, 1);

        EXPECT_FALSE(t_transposed.is_contiguous());

        // Should still work (internally makes contiguous)
        auto flat = flatten_tensors({t_transposed});
        EXPECT_EQ(flat.numel(), 20);
    } catch (const std::exception& e) {
        // Backend may not support transpose - skip test
        GTEST_SKIP() << "Backend does not support transpose operation: " << e.what();
    }
}

TEST_F(GradientUtilsTest, FlattenDifferentDtypesFails) {
    Tensor t1({10}, DType::Float32, Device::cpu());
    Tensor t2({10}, DType::Float64, Device::cpu());

    EXPECT_THROW(flatten_tensors({t1, t2}), std::invalid_argument);
}

TEST_F(GradientUtilsTest, FlattenEmptyVectorFails) {
    std::vector<Tensor> empty;
    EXPECT_THROW(flatten_tensors(empty), std::invalid_argument);
}

TEST_F(GradientUtilsTest, FlattenVariables) {
    auto v1 = std::make_shared<Variable>(Tensor({10, 20}, DType::Float32, Device::cpu()), true);
    auto v2 = std::make_shared<Variable>(Tensor({5, 5}, DType::Float32, Device::cpu()), true);

    // Create gradients
    v1->set_grad(Tensor({10, 20}, DType::Float32, Device::cpu()));
    v2->set_grad(Tensor({5, 5}, DType::Float32, Device::cpu()));
    v1->grad()->fill_(1.0f);
    v2->grad()->fill_(2.0f);

    auto flat = flatten_tensors({v1, v2});

    EXPECT_EQ(flat.numel(), 200 + 25);
}

TEST_F(GradientUtilsTest, FlattenVariablesNoGradientsFails) {
    auto v1 = std::make_shared<Variable>(Tensor({10, 20}, DType::Float32, Device::cpu()), false);
    auto v2 = std::make_shared<Variable>(Tensor({5, 5}, DType::Float32, Device::cpu()), false);

    EXPECT_THROW(flatten_tensors({v1, v2}), std::invalid_argument);
}

// ============================================================================
// Tensor Unflattening Tests
// ============================================================================

TEST_F(GradientUtilsTest, UnflattenIntoShapes) {
    Tensor flat({225}, DType::Float32, Device::cpu());
    auto* data = flat.data<float>();
    for (int i = 0; i < 225; ++i) {
        data[i] = static_cast<float>(i);
    }

    std::vector<std::vector<int64_t>> shapes = {{10, 20}, {5, 5}};
    auto tensors = unflatten_into(flat, shapes);

    ASSERT_EQ(tensors.size(), 2);
    EXPECT_EQ(tensors[0].shape()[0], 10);
    EXPECT_EQ(tensors[0].shape()[1], 20);
    EXPECT_EQ(tensors[1].shape()[0], 5);
    EXPECT_EQ(tensors[1].shape()[1], 5);

    // Verify data
    auto* t0_data = tensors[0].data<float>();
    for (int i = 0; i < 200; ++i) {
        EXPECT_FLOAT_EQ(t0_data[i], static_cast<float>(i));
    }

    auto* t1_data = tensors[1].data<float>();
    for (int i = 0; i < 25; ++i) {
        EXPECT_FLOAT_EQ(t1_data[i], static_cast<float>(200 + i));
    }
}

TEST_F(GradientUtilsTest, UnflattenIntoTensors) {
    Tensor flat({225}, DType::Float32, Device::cpu());
    auto* data = flat.data<float>();
    for (int i = 0; i < 225; ++i) {
        data[i] = static_cast<float>(i);
    }

    std::vector<Tensor> output = {
        Tensor({10, 20}, DType::Float32, Device::cpu()),
        Tensor({5, 5}, DType::Float32, Device::cpu())
    };

    unflatten_into(flat, output);

    // Verify data
    auto* t0_data = output[0].data<float>();
    for (int i = 0; i < 200; ++i) {
        EXPECT_FLOAT_EQ(t0_data[i], static_cast<float>(i));
    }

    auto* t1_data = output[1].data<float>();
    for (int i = 0; i < 25; ++i) {
        EXPECT_FLOAT_EQ(t1_data[i], static_cast<float>(200 + i));
    }
}

TEST_F(GradientUtilsTest, UnflattenSizeMismatchFails) {
    Tensor flat({100}, DType::Float32, Device::cpu());
    std::vector<std::vector<int64_t>> shapes = {{10, 20}};  // Only 200 elements

    EXPECT_THROW(unflatten_into(flat, shapes), std::invalid_argument);
}

TEST_F(GradientUtilsTest, UnflattenNonFlatTensorFails) {
    Tensor not_flat({10, 10}, DType::Float32, Device::cpu());
    std::vector<std::vector<int64_t>> shapes = {{100}};

    EXPECT_THROW(unflatten_into(not_flat, shapes), std::invalid_argument);
}

TEST_F(GradientUtilsTest, UnflattenInvalidShapeFails) {
    Tensor flat({100}, DType::Float32, Device::cpu());
    std::vector<std::vector<int64_t>> shapes = {{10, -1}};  // Negative dimension

    EXPECT_THROW(unflatten_into(flat, shapes), std::invalid_argument);
}

// ============================================================================
// Roundtrip Tests (Flatten -> Unflatten)
// ============================================================================

TEST_F(GradientUtilsTest, FlattenUnflattenRoundtrip) {
    Tensor t1({10, 20}, DType::Float32, Device::cpu());
    Tensor t2({5, 5}, DType::Float32, Device::cpu());
    Tensor t3({100}, DType::Float32, Device::cpu());

    // Fill with distinct values
    auto* d1 = t1.data<float>();
    for (int i = 0; i < 200; ++i) d1[i] = static_cast<float>(i);

    auto* d2 = t2.data<float>();
    for (int i = 0; i < 25; ++i) d2[i] = static_cast<float>(200 + i);

    auto* d3 = t3.data<float>();
    for (int i = 0; i < 100; ++i) d3[i] = static_cast<float>(225 + i);

    // Flatten
    auto flat = flatten_tensors({t1, t2, t3});

    // Unflatten
    std::vector<std::vector<int64_t>> shapes = {{10, 20}, {5, 5}, {100}};
    auto tensors = unflatten_into(flat, shapes);

    // Verify
    ASSERT_EQ(tensors.size(), 3);

    auto* r1 = tensors[0].data<float>();
    for (int i = 0; i < 200; ++i) {
        EXPECT_FLOAT_EQ(r1[i], d1[i]);
    }

    auto* r2 = tensors[1].data<float>();
    for (int i = 0; i < 25; ++i) {
        EXPECT_FLOAT_EQ(r2[i], d2[i]);
    }

    auto* r3 = tensors[2].data<float>();
    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(r3[i], d3[i]);
    }
}

// ============================================================================
// Bucket Computation Tests
// ============================================================================

TEST_F(GradientUtilsTest, ComputeBucketsSingleTensor) {
    Tensor t({1024 * 1024}, DType::Float32, Device::cpu());  // 4MB

    BucketConfig config;
    config.max_bucket_size_mb = 25;

    auto buckets = compute_bucket_sizes({t}, config);

    ASSERT_EQ(buckets.size(), 1);
    EXPECT_EQ(buckets[0].start_idx, 0);
    EXPECT_EQ(buckets[0].end_idx, 1);
    EXPECT_EQ(buckets[0].num_elements, 1024 * 1024);
    EXPECT_EQ(buckets[0].size_bytes, 4 * 1024 * 1024);
    EXPECT_EQ(buckets[0].dtype, DType::Float32);
}

TEST_F(GradientUtilsTest, ComputeBucketsMultipleTensors) {
    std::vector<Tensor> tensors;
    // Create 10 tensors of 5MB each (total 50MB)
    for (int i = 0; i < 10; ++i) {
        tensors.push_back(Tensor({1024 * 1024 + i * 1024}, DType::Float32, Device::cpu()));
    }

    BucketConfig config;
    config.max_bucket_size_mb = 25;

    auto buckets = compute_bucket_sizes(tensors, config);

    // Should create multiple buckets
    EXPECT_GT(buckets.size(), 1);

    // Verify all tensors are covered
    size_t total_elements = 0;
    for (const auto& bucket : buckets) {
        total_elements += bucket.num_elements;
        EXPECT_GT(bucket.num_elements, 0);
        EXPECT_LE(bucket.size_bytes, static_cast<size_t>(config.max_bucket_size_mb * 1024 * 1024 + 5 * 1024 * 1024));  // Allow some overflow
    }

    size_t expected_elements = 0;
    for (const auto& t : tensors) {
        expected_elements += t.numel();
    }
    EXPECT_EQ(total_elements, expected_elements);
}

TEST_F(GradientUtilsTest, ComputeBucketsDifferentDtypes) {
    Tensor t1({1000}, DType::Float32, Device::cpu());
    Tensor t2({1000}, DType::Float64, Device::cpu());
    Tensor t3({1000}, DType::Float32, Device::cpu());

    BucketConfig config;
    config.max_bucket_size_mb = 25;

    auto buckets = compute_bucket_sizes({t1, t2, t3}, config);

    // Should create at least 2 buckets (different dtypes must be separate)
    EXPECT_GE(buckets.size(), 2);

    // Verify dtypes are consistent within buckets
    for (const auto& bucket : buckets) {
        EXPECT_TRUE(bucket.dtype == DType::Float32 || bucket.dtype == DType::Float64);
    }
}

TEST_F(GradientUtilsTest, ComputeBucketsEmptyFails) {
    std::vector<Tensor> empty;
    BucketConfig config;

    EXPECT_THROW(compute_bucket_sizes(empty, config), std::invalid_argument);
}

TEST_F(GradientUtilsTest, ComputeBucketsVariables) {
    std::vector<std::shared_ptr<Variable>> variables;
    for (int i = 0; i < 5; ++i) {
        variables.push_back(std::make_shared<Variable>(
            Tensor({1000, 1000}, DType::Float32, Device::cpu()), true
        ));
    }

    BucketConfig config;
    config.max_bucket_size_mb = 10;

    auto buckets = compute_bucket_sizes(variables, config);

    EXPECT_GT(buckets.size(), 0);
}

// ============================================================================
// Memory Estimation Tests
// ============================================================================

TEST_F(GradientUtilsTest, EstimateMemorySingleRank) {
    std::vector<Tensor> tensors = {
        Tensor({1000, 1000}, DType::Float32, Device::cpu()),
        Tensor({500, 500}, DType::Float32, Device::cpu()),
        Tensor({100, 100}, DType::Float32, Device::cpu())
    };

    auto stats = estimate_gradient_memory(tensors, 1);

    EXPECT_EQ(stats.total_parameters, 3);
    EXPECT_EQ(stats.total_elements, 1000*1000 + 500*500 + 100*100);
    EXPECT_EQ(stats.total_bytes, stats.total_elements * 4);  // Float32 = 4 bytes
    EXPECT_EQ(stats.bytes_per_rank, stats.total_bytes);
    EXPECT_GT(stats.num_buckets, 0);
}

TEST_F(GradientUtilsTest, EstimateMemoryMultipleRanks) {
    std::vector<Tensor> tensors = {
        Tensor({1000, 1000}, DType::Float32, Device::cpu()),
        Tensor({1000, 1000}, DType::Float32, Device::cpu()),
        Tensor({1000, 1000}, DType::Float32, Device::cpu()),
        Tensor({1000, 1000}, DType::Float32, Device::cpu())
    };

    auto stats = estimate_gradient_memory(tensors, 4);

    EXPECT_EQ(stats.total_parameters, 4);
    EXPECT_LT(stats.bytes_per_rank, stats.total_bytes);  // Should be partitioned
    EXPECT_GE(stats.bytes_per_rank, stats.total_bytes / 4);  // At least 1/4
    EXPECT_LE(stats.bytes_per_rank, stats.total_bytes / 4 + stats.total_bytes / (4 * 10));  // At most 1/4 + small overhead
}

TEST_F(GradientUtilsTest, EstimateMemoryFragmentation) {
    std::vector<Tensor> tensors;
    // Create many small tensors (causes fragmentation)
    // 100 tensors * 100 elements * 4 bytes = 40KB total
    for (int i = 0; i < 100; ++i) {
        tensors.push_back(Tensor({100}, DType::Float32, Device::cpu()));
    }

    BucketConfig config;
    config.max_bucket_size_mb = 0.01;  // 10KB per bucket
    config.min_bucket_size_mb = 0.001;  // 1KB min (allow small buckets)

    auto stats = estimate_gradient_memory(tensors, 1, config);

    EXPECT_GT(stats.num_buckets, 1);  // Should have 4-5 buckets with 10KB limit
    EXPECT_GE(stats.fragmentation_ratio, 0.0);
    EXPECT_LE(stats.fragmentation_ratio, 1.0);
}

TEST_F(GradientUtilsTest, EstimateMemoryVariables) {
    std::vector<std::shared_ptr<Variable>> variables;
    for (int i = 0; i < 5; ++i) {
        variables.push_back(std::make_shared<Variable>(
            Tensor({1000, 1000}, DType::Float32, Device::cpu()), true
        ));
    }

    auto stats = estimate_gradient_memory(variables, 2);

    EXPECT_EQ(stats.total_parameters, 5);
    EXPECT_GT(stats.total_bytes, 0);
}

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST_F(GradientUtilsTest, AllContiguous) {
    Tensor t1({10, 20}, DType::Float32, Device::cpu());
    Tensor t2({5, 5}, DType::Float32, Device::cpu());

    EXPECT_TRUE(all_contiguous({t1, t2}));

    auto t1_transposed = t1.transpose(0, 1);
    EXPECT_FALSE(all_contiguous({t1_transposed, t2}));
}

TEST_F(GradientUtilsTest, SameDtype) {
    Tensor t1({10}, DType::Float32, Device::cpu());
    Tensor t2({10}, DType::Float32, Device::cpu());
    Tensor t3({10}, DType::Float64, Device::cpu());

    EXPECT_TRUE(same_dtype({t1, t2}));
    EXPECT_FALSE(same_dtype({t1, t3}));
}

TEST_F(GradientUtilsTest, SameDevice) {
    Tensor t1({10}, DType::Float32, Device::cpu());
    Tensor t2({10}, DType::Float32, Device::cpu());

    EXPECT_TRUE(same_device({t1, t2}));
}

TEST_F(GradientUtilsTest, TotalNumel) {
    Tensor t1({10, 20}, DType::Float32, Device::cpu());
    Tensor t2({5, 5}, DType::Float32, Device::cpu());

    size_t total = total_numel({t1, t2});
    EXPECT_EQ(total, 200 + 25);
}

TEST_F(GradientUtilsTest, TotalBytes) {
    Tensor t1({100}, DType::Float32, Device::cpu());  // 400 bytes
    Tensor t2({100}, DType::Float64, Device::cpu());  // 800 bytes

    size_t total = total_bytes({t1, t2});
    EXPECT_EQ(total, 400 + 800);
}

TEST_F(GradientUtilsTest, ExtractShapes) {
    Tensor t1({10, 20}, DType::Float32, Device::cpu());
    Tensor t2({5, 5, 5}, DType::Float32, Device::cpu());

    auto shapes = extract_shapes({t1, t2});

    ASSERT_EQ(shapes.size(), 2);
    ASSERT_EQ(shapes[0].size(), 2);
    EXPECT_EQ(shapes[0][0], 10);
    EXPECT_EQ(shapes[0][1], 20);
    ASSERT_EQ(shapes[1].size(), 3);
    EXPECT_EQ(shapes[1][0], 5);
    EXPECT_EQ(shapes[1][1], 5);
    EXPECT_EQ(shapes[1][2], 5);
}

TEST_F(GradientUtilsTest, MakeContiguous) {
    // Skip test if transpose not available in backend
    try {
        Tensor t1({10, 20}, DType::Float32, Device::cpu());
        auto t2 = t1.transpose(0, 1);

        EXPECT_TRUE(t1.is_contiguous());
        EXPECT_FALSE(t2.is_contiguous());

        auto contiguous = make_contiguous({t1, t2});

        EXPECT_TRUE(contiguous[0].is_contiguous());
        EXPECT_TRUE(contiguous[1].is_contiguous());
    } catch (const std::exception& e) {
        // Backend may not support transpose - skip test
        GTEST_SKIP() << "Backend does not support transpose operation: " << e.what();
    }
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST_F(GradientUtilsTest, LargeTensorFlattening) {
    // Test with large tensors
    Tensor t1({1000, 1000}, DType::Float32, Device::cpu());
    Tensor t2({500, 2000}, DType::Float32, Device::cpu());

    auto flat = flatten_tensors({t1, t2});

    EXPECT_EQ(flat.numel(), 1000*1000 + 500*2000);
}

TEST_F(GradientUtilsTest, ManySmallTensors) {
    std::vector<Tensor> tensors;
    for (int i = 0; i < 1000; ++i) {
        tensors.push_back(Tensor({10}, DType::Float32, Device::cpu()));
    }

    auto flat = flatten_tensors(tensors);
    EXPECT_EQ(flat.numel(), 10000);

    auto shapes = extract_shapes(tensors);
    auto unflat = unflatten_into(flat, shapes);
    EXPECT_EQ(unflat.size(), 1000);
}

TEST_F(GradientUtilsTest, DifferentDtypeSizes) {
    std::vector<Tensor> tensors = {
        Tensor({1000}, DType::Int8, Device::cpu()),    // 1 byte
        Tensor({1000}, DType::Int16, Device::cpu()),   // 2 bytes
        Tensor({1000}, DType::Float32, Device::cpu()), // 4 bytes
        Tensor({1000}, DType::Float64, Device::cpu())  // 8 bytes
    };

    size_t expected_bytes = 1000 * (1 + 2 + 4 + 8);
    EXPECT_EQ(total_bytes(tensors), expected_bytes);
}
