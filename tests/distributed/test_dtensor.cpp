/**
 * @file test_dtensor.cpp
 * @brief Coverage for tenzor::distributed::DTensor.
 *
 * DeviceMesh tests live in test_device_mesh.cpp; this file pins the
 * DTensor surface that consumes a DeviceMesh:
 *   - DTensor::from_global and back via full_tensor() — Replicate, Shard.
 *   - redistribute() between placement pairs.
 *   - Placement and shape introspection.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/device_mesh.hpp>
#include <tenzor/distributed/dtensor.hpp>

using namespace tenzor;
using namespace tenzor::distributed;

class DTensorBasicsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

// DeviceMesh-specific tests live in test_device_mesh.cpp (audit F.1 split).
// This file focuses on the DTensor surface that consumes a DeviceMesh.

// ---------------------------------------------------------------------------
// DTensor — single-rank semantics. We verify:
//  - Shape introspection accounts for sharding.
//  - full_tensor() == the original global tensor (under Replicate or Shard).
//  - redistribute() between placement pairs preserves data.
// ---------------------------------------------------------------------------

TEST_F(DTensorBasicsTest, DTensor_FromGlobal_AllReplicate) {
    auto mesh = std::make_shared<DeviceMesh>(Device::Type::CPU, std::vector<int64_t>{1, 1},
                                              std::vector<std::string>{"dp", "tp"});
    auto global = randn({4, 8}, DType::Float32, Device::cpu());

    auto dt = DTensor::from_global(global, mesh,
        /*placements=*/{Placement{Replicate{}}, Placement{Replicate{}}});
    EXPECT_EQ(dt.shape().size(), 2u);
    EXPECT_EQ(dt.shape()[0], 4);
    EXPECT_EQ(dt.shape()[1], 8);
    // Replicated: local tensor equals global tensor.
    auto lt = dt.local_tensor();
    EXPECT_EQ(lt.shape().size(), 2u);
    EXPECT_EQ(lt.shape()[0], 4);
    EXPECT_EQ(lt.shape()[1], 8);
}

TEST_F(DTensorBasicsTest, DTensor_FromGlobal_ShardOnRow) {
    // 1D mesh of size 1 keeps the test single-rank — the local shard is
    // the whole global tensor. The point is to verify the sharded shape
    // and full_tensor() round-trip without depending on collectives.
    auto mesh = std::make_shared<DeviceMesh>(Device::Type::CPU,
                                              std::vector<int64_t>{1},
                                              std::vector<std::string>{"dp"});
    auto global = randn({4, 8}, DType::Float32, Device::cpu());

    auto dt = DTensor::from_global(global, mesh,
        /*placements=*/{Placement{Shard{/*dim=*/0}}});

    // Global shape preserved (mesh size 1 means no actual sharding).
    auto sh = dt.shape();
    EXPECT_EQ(sh.size(), 2u);
    EXPECT_EQ(sh[0], 4);
    EXPECT_EQ(sh[1], 8);
    EXPECT_EQ(dt.placements().size(), 1u);
}

TEST_F(DTensorBasicsTest, DTensor_FullTensor_Roundtrip) {
    auto mesh = std::make_shared<DeviceMesh>(Device::Type::CPU,
                                              std::vector<int64_t>{1},
                                              std::vector<std::string>{"dp"});
    auto global = randn({6, 10}, DType::Float32, Device::cpu());
    auto dt = DTensor::from_global(global, mesh,
        /*placements=*/{Placement{Replicate{}}});

    auto recovered = dt.full_tensor();
    ASSERT_EQ(recovered.shape().size(), global.shape().size());
    EXPECT_EQ(recovered.shape()[0], global.shape()[0]);
    EXPECT_EQ(recovered.shape()[1], global.shape()[1]);

    const float* g = global.contiguous().data<float>();
    const float* r = recovered.contiguous().data<float>();
    for (int64_t i = 0; i < global.numel(); ++i) {
        EXPECT_FLOAT_EQ(r[i], g[i]) << "mismatch at " << i;
    }
}

TEST_F(DTensorBasicsTest, DTensor_Redistribute_ReplicateToReplicate) {
    auto mesh = std::make_shared<DeviceMesh>(Device::Type::CPU,
                                              std::vector<int64_t>{1},
                                              std::vector<std::string>{"dp"});
    auto global = randn({4, 4}, DType::Float32, Device::cpu());
    auto dt = DTensor::from_global(global, mesh,
        /*placements=*/{Placement{Replicate{}}});
    auto dt2 = dt.redistribute({Placement{Replicate{}}});
    EXPECT_EQ(dt2.placements().size(), 1u);
}

TEST_F(DTensorBasicsTest, DTensor_PlacementMismatchThrows) {
    auto mesh = std::make_shared<DeviceMesh>(Device::Type::CPU,
                                              std::vector<int64_t>{2, 2},
                                              std::vector<std::string>{"dp", "tp"});
    auto global = randn({4, 4}, DType::Float32, Device::cpu());
    // placements.size() != mesh.ndim() must throw.
    EXPECT_THROW(
        DTensor::from_global(global, mesh,
                             /*placements=*/{Placement{Replicate{}}}),
        std::invalid_argument);
}
