/**
 * @file test_device_mesh.cpp
 * @brief Coverage for tenzor::distributed::DeviceMesh.
 *
 * Split out from test_dtensor.cpp (audit 2026-05-02 F.1) so the DeviceMesh
 * surface — shape, coordinate round-trip, named-dim lookup, error paths —
 * lives in its own file. DTensor tests in test_dtensor.cpp consume a
 * DeviceMesh but don't re-test its internals.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/device_mesh.hpp>

using namespace tenzor;
using namespace tenzor::distributed;

class DeviceMeshTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

TEST_F(DeviceMeshTest, ShapeAndDims) {
    DeviceMesh mesh(Device::Type::CPU, /*shape=*/{2, 4}, /*dim_names=*/{"dp", "tp"});
    EXPECT_EQ(mesh.ndim(), 2);
    EXPECT_EQ(mesh.size(), 8);
    EXPECT_EQ(mesh.shape().size(), 2u);
    EXPECT_EQ(mesh.shape()[0], 2);
    EXPECT_EQ(mesh.shape()[1], 4);
    EXPECT_EQ(mesh.dim_names()[0], "dp");
    EXPECT_EQ(mesh.dim_names()[1], "tp");
    EXPECT_EQ(mesh.device_type(), Device::Type::CPU);
}

TEST_F(DeviceMeshTest, CoordinateRoundTrip) {
    DeviceMesh mesh(Device::Type::CPU, {2, 4}, {"dp", "tp"});
    for (int64_t flat = 0; flat < mesh.size(); ++flat) {
        auto coord = mesh.get_coordinate(flat);
        ASSERT_EQ(coord.size(), 2u);
        EXPECT_EQ(mesh.get_device_id(coord), flat)
            << "Round-trip failed at flat=" << flat;
    }
}

TEST_F(DeviceMeshTest, RowMajorOrdering) {
    // Row-major: (0,0)→0, (0,1)→1, (1,0)→4 with shape {2,4}.
    DeviceMesh mesh(Device::Type::CPU, {2, 4}, {"dp", "tp"});
    EXPECT_EQ(mesh.get_device_id({0, 0}), 0);
    EXPECT_EQ(mesh.get_device_id({0, 1}), 1);
    EXPECT_EQ(mesh.get_device_id({1, 0}), 4);
    EXPECT_EQ(mesh.get_device_id({1, 3}), 7);
}

TEST_F(DeviceMeshTest, BadConstruction_ShapeNamesMismatch) {
    EXPECT_THROW(DeviceMesh(Device::Type::CPU, {2, 4}, {"only_one"}),
                 std::invalid_argument);
}

TEST_F(DeviceMeshTest, BadConstruction_DuplicateDimName) {
    EXPECT_THROW(DeviceMesh(Device::Type::CPU, {2, 4}, {"dup", "dup"}),
                 std::invalid_argument);
}

TEST_F(DeviceMeshTest, OutOfRangeCoordinate) {
    DeviceMesh mesh(Device::Type::CPU, {2, 2}, {"a", "b"});
    EXPECT_THROW(mesh.get_device_id({2, 0}), std::out_of_range);
}

TEST_F(DeviceMeshTest, OutOfRangeFlatId) {
    DeviceMesh mesh(Device::Type::CPU, {2, 2}, {"a", "b"});
    EXPECT_THROW(mesh.get_coordinate(4), std::out_of_range);
}

TEST_F(DeviceMeshTest, OneDimensionalMesh) {
    // Smallest non-trivial mesh: 1-D.
    DeviceMesh mesh(Device::Type::CPU, {4}, {"dp"});
    EXPECT_EQ(mesh.ndim(), 1);
    EXPECT_EQ(mesh.size(), 4);
    EXPECT_EQ(mesh.get_device_id({2}), 2);
    EXPECT_EQ(mesh.get_coordinate(3)[0], 3);
}
