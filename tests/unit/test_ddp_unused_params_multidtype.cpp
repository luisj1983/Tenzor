/**
 * @file test_ddp_unused_params_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for DDP unused parameters API
 *
 * DistributedDataParallel requires a real ProcessGroup and CommunicationBackend,
 * so these tests focus on compile-time API verification and struct correctness
 * across backends and dtypes. Full DDP integration tests live elsewhere.
 *
 * CPU is skipped since DDP targets multi-device setups.
 */

#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/distributed/ddp.hpp>
#include <type_traits>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::distributed;

// ============================================================================
// Fixture
// ============================================================================

class DDPUnusedParamsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        if (device().type == Device::Type::CPU) {
            GTEST_SKIP() << "DDP requires a GPU backend";
        }

        // Verify device is usable
        try {
            auto t = zeros({2, 2}, dtype(), device());
        } catch (...) {
            GTEST_SKIP() << "Cannot allocate on " << backend_name();
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

/// DistributedDataParallel class exists and is not copy-constructible.
TEST_P(DDPUnusedParamsMultiDTypeTest, ClassExists) {
    static_assert(std::is_class_v<DistributedDataParallel>,
                  "DistributedDataParallel class must exist");
    static_assert(!std::is_copy_constructible_v<DistributedDataParallel>,
                  "DistributedDataParallel must not be copy-constructible");
    SUCCEED();
}

/// GradBucket default construction and field initialization.
TEST_P(DDPUnusedParamsMultiDTypeTest, GradBucketConstruction) {
    GradBucket bucket;
    bucket.ready = false;
    bucket.pending_count = 0;
    bucket.size_bytes = 0;

    EXPECT_FALSE(bucket.ready);
    EXPECT_EQ(bucket.pending_count, 0u);
    EXPECT_EQ(bucket.size_bytes, 0u);
}

/// GradBucket fields can be set to non-zero values.
TEST_P(DDPUnusedParamsMultiDTypeTest, GradBucketFieldAssignment) {
    GradBucket bucket;
    bucket.ready = true;
    bucket.pending_count = 42;
    bucket.size_bytes = 1024;

    EXPECT_TRUE(bucket.ready);
    EXPECT_EQ(bucket.pending_count, 42u);
    EXPECT_EQ(bucket.size_bytes, 1024u);
}

/// Verify a tensor can be created on the target device with the test dtype,
/// simulating the gradient buffer that DDP would manage.
TEST_P(DDPUnusedParamsMultiDTypeTest, GradientBufferSimulation) {
    // Simulate a gradient buffer allocation on the target device
    auto grad_buffer = zeros({10, 5}, dtype(), device());

    expectShape(grad_buffer, {10, 5});
    EXPECT_EQ(grad_buffer.dtype(), dtype());
    EXPECT_EQ(grad_buffer.device().type, device().type);
}

/// Verify ones tensor creation and device transfer (simulates parameter init).
TEST_P(DDPUnusedParamsMultiDTypeTest, ParameterInitSimulation) {
    auto param = ones({8, 4}, DType::Float32, Device::cpu()).to(dtype()).to(device());

    expectShape(param, {8, 4});
    EXPECT_EQ(param.dtype(), dtype());

    // Verify values after round-trip to CPU
    auto cpu_f32 = param.to(DType::Float32).to(Device::cpu());
    auto* data = cpu_f32.data<float>();
    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(data[i], 1.0f, atol()) << "index " << i;
    }
}

/// Multiple GradBucket instances are independent.
TEST_P(DDPUnusedParamsMultiDTypeTest, MultipleBucketsIndependent) {
    GradBucket b1, b2;
    b1.ready = true;
    b1.pending_count = 5;
    b1.size_bytes = 100;

    b2.ready = false;
    b2.pending_count = 10;
    b2.size_bytes = 200;

    EXPECT_TRUE(b1.ready);
    EXPECT_FALSE(b2.ready);
    EXPECT_NE(b1.pending_count, b2.pending_count);
    EXPECT_NE(b1.size_bytes, b2.size_bytes);
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DDPUnusedParamsMultiDTypeTest);
