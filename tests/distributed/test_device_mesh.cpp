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
#include <tenzor/distributed/process_group.hpp>
#include <tenzor/distributed/distributed.hpp>  // for ReduceOp definition
#include <tenzor/ops/creation.hpp>  // for zeros()
#include <cstdlib>

using namespace tenzor;
using namespace tenzor::distributed;

class DeviceMeshTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};


// --------------------------------------------------------------------------
// Audit remediation A3: DeviceMesh process-group accessors.
// --------------------------------------------------------------------------

// Single-rank fake PG: mimics WS=1, returns the input unchanged. Used to
// verify that `set_process_group` / `process_group` / `process_group_for_dim`
// route correctly. Multi-rank correctness of the collectives themselves --
// and in particular whether GlooProcessGroup::split() actually partitions
// real ranks into independent sub-groups rather than SplittablePG's fake
// stub -- is covered by MultiRankDeviceMeshTest below: a real 4-process
// GlooProcessGroup, launched via tests/distributed/run_multirank_test.sh
// (FINDING 21 "plan K3" -- the multi-process C++ test job this file used to
// defer to but which never existed).
class SingleRankFakePG : public tenzor::distributed::ProcessGroupBase {
public:
    int seen_alltoall = 0;
    auto rank() const -> int override { return 0; }
    auto world_size() const -> int override { return 1; }
    auto all_reduce(tenzor::Tensor& /*tensor*/, tenzor::distributed::ReduceOp) -> void override {}
    auto broadcast(tenzor::Tensor& /*tensor*/, int /*src_rank*/) -> void override {}
    auto all_gather(std::vector<tenzor::Tensor>& output, const tenzor::Tensor& input) -> void override {
        if (output.empty()) output.resize(1);
        output[0] = input;
    }
    auto reduce_scatter(tenzor::Tensor& output, std::span<const tenzor::Tensor> input) -> void override {
        if (!input.empty()) output = input[0];
    }
    auto barrier() -> void override {}
    auto all_to_all_single(tenzor::Tensor& output, const tenzor::Tensor& input) -> void override {
        ++seen_alltoall;
        // With WS=1, all_to_all is identity.
        output = input;
    }
};


// --------------------------------------------------------------------------
// Audit remediation A3-extended / B3: fake PG that supports split().
// --------------------------------------------------------------------------

// Fake PG with configurable world_size. `split` returns a fresh sub-PG
// whose world_size is determined by the test (the caller verifies the
// `split` call was issued with the expected color/key).
class SplittablePG : public tenzor::distributed::ProcessGroupBase {
public:
    int my_rank;
    int reported_world_size;
    int sub_world_size;       // world_size to report for sub-PGs from split()
    int last_color = -999;
    int last_key = -999;
    int split_calls = 0;

    SplittablePG(int rank, int ws, int sub_ws)
        : my_rank(rank), reported_world_size(ws), sub_world_size(sub_ws) {}

    auto rank() const -> int override { return my_rank; }
    auto world_size() const -> int override { return reported_world_size; }
    auto all_reduce(tenzor::Tensor&, tenzor::distributed::ReduceOp) -> void override {}
    auto broadcast(tenzor::Tensor&, int) -> void override {}
    auto all_gather(std::vector<tenzor::Tensor>& output, const tenzor::Tensor& input) -> void override {
        output.assign(static_cast<size_t>(reported_world_size), input);
    }
    auto reduce_scatter(tenzor::Tensor&, std::span<const tenzor::Tensor>) -> void override {}
    auto barrier() -> void override {}
    auto split(int color, int key)
        -> std::shared_ptr<tenzor::distributed::ProcessGroupBase> override {
        ++split_calls;
        last_color = color;
        last_key = key;
        // Return a smaller fake with sub_world_size; the caller's "rank" in
        // the new sub-PG is `key`.
        return std::make_shared<SplittablePG>(key, sub_world_size,
                                              sub_world_size);
    }
};

TEST_F(DeviceMeshTest, ProcessGroupSetGet) {
    DeviceMesh mesh(Device::Type::CPU, {2, 4}, {"dp", "tp"}, /*mesh_rank=*/0);
    EXPECT_EQ(mesh.process_group(), nullptr);

    auto pg = std::make_shared<SingleRankFakePG>();
    mesh.set_process_group(pg);
    EXPECT_EQ(mesh.process_group().get(), pg.get());

    mesh.set_process_group(nullptr);
    EXPECT_EQ(mesh.process_group(), nullptr);
}

TEST_F(DeviceMeshTest, ProcessGroupForDim) {
    DeviceMesh mesh(Device::Type::CPU, {2, 4}, {"dp", "tp"}, /*mesh_rank=*/0);
    auto pg = std::make_shared<SingleRankFakePG>();
    mesh.set_process_group(pg);

    // For now both named dims return the same global PG (A3-extended will
    // produce true per-axis sub-groups).
    EXPECT_EQ(mesh.process_group_for_dim("dp").get(), pg.get());
    EXPECT_EQ(mesh.process_group_for_dim("tp").get(), pg.get());

    // Unknown dim name throws (matching get_dim semantics).
    EXPECT_THROW(mesh.process_group_for_dim("nope"), std::invalid_argument);
}

// --------------------------------------------------------------------------
// Audit remediation A4: ProcessGroupBase::all_to_all_single API surface.
// --------------------------------------------------------------------------
// Multi-rank correctness lives in the distributed test job; here we exercise
// the API surface with WS=1 (identity) and confirm the shape-validation path
// throws for misshapen inputs.

TEST_F(DeviceMeshTest, AllToAllSingleIdentityOnSingleRank) {
    SingleRankFakePG pg;
    auto input = zeros({4, 3}, DType::Float32, Device::cpu());
    auto output = zeros({4, 3}, DType::Float32, Device::cpu());
    pg.all_to_all_single(output, input);
    EXPECT_EQ(pg.seen_alltoall, 1);
}

// Verify the ProcessGroupBase default impl's shape-validation paths. We need
// a PG whose `all_gather` actually populates outputs so the default path runs.
// Using SingleRankFakePG's all_gather (which fills one entry).
TEST_F(DeviceMeshTest, AllToAllSingleDefaultViaAllGather) {
    class WS1DefaultPG : public tenzor::distributed::ProcessGroupBase {
    public:
        auto rank() const -> int override { return 0; }
        auto world_size() const -> int override { return 1; }
        auto all_reduce(tenzor::Tensor&, tenzor::distributed::ReduceOp) -> void override {}
        auto broadcast(tenzor::Tensor&, int) -> void override {}
        auto all_gather(std::vector<tenzor::Tensor>& output, const tenzor::Tensor& input) -> void override {
            output.assign(1, input);
        }
        auto reduce_scatter(tenzor::Tensor&, std::span<const tenzor::Tensor>) -> void override {}
        auto barrier() -> void override {}
        // intentionally NOT overriding all_to_all_single — use the base default.
    };

    WS1DefaultPG pg;
    auto input = zeros({4, 3}, DType::Float32, Device::cpu());
    auto output = zeros({4, 3}, DType::Float32, Device::cpu());

    // WS=1 default path: gather one tensor (= input), slice [0:4] (whole),
    // cat -> output rebound to input. No throw expected.
    EXPECT_NO_THROW(pg.all_to_all_single(output, input));
    EXPECT_EQ(output.shape().size(), 2u);
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 3);
}

TEST_F(DeviceMeshTest, AllToAllSingleRejectsZeroDim) {
    SingleRankFakePG pg;
    auto input  = zeros({}, DType::Float32, Device::cpu());  // 0-D scalar
    auto output = zeros({}, DType::Float32, Device::cpu());
    // Custom override on SingleRankFakePG doesn't validate, so reach for the
    // base default by routing through a separate path.
    class StrictPG : public tenzor::distributed::ProcessGroupBase {
    public:
        auto rank() const -> int override { return 0; }
        auto world_size() const -> int override { return 1; }
        auto all_reduce(tenzor::Tensor&, tenzor::distributed::ReduceOp) -> void override {}
        auto broadcast(tenzor::Tensor&, int) -> void override {}
        auto all_gather(std::vector<tenzor::Tensor>& output, const tenzor::Tensor& input) -> void override {
            output.assign(1, input);
        }
        auto reduce_scatter(tenzor::Tensor&, std::span<const tenzor::Tensor>) -> void override {}
        auto barrier() -> void override {}
    };
    StrictPG strict;
    EXPECT_THROW(strict.all_to_all_single(output, input), std::invalid_argument);
}


// --------------------------------------------------------------------------
// Audit remediation A2: explicit mesh_rank constructor & env-var resolution.
// --------------------------------------------------------------------------

TEST_F(DeviceMeshTest, ExplicitMeshRank) {
    DeviceMesh mesh(Device::Type::CPU, {2, 4}, {"dp", "tp"}, /*mesh_rank=*/5);
    EXPECT_EQ(mesh.get_local_rank(), 5);
    EXPECT_EQ(mesh.get_mesh_rank(), 5);
    auto coord = mesh.get_coordinate(mesh.get_local_rank());
    EXPECT_EQ(coord[0], 1);
    EXPECT_EQ(coord[1], 1);
}

TEST_F(DeviceMeshTest, ExplicitMeshRankOutOfRangeThrows) {
    EXPECT_THROW(
        (DeviceMesh{Device::Type::CPU, {2, 4}, {"dp", "tp"}, /*mesh_rank=*/8}),
        std::out_of_range);
    EXPECT_THROW(
        (DeviceMesh{Device::Type::CPU, {2, 4}, {"dp", "tp"}, /*mesh_rank=*/-1}),
        std::out_of_range);
}

TEST_F(DeviceMeshTest, ResolveRankFromRankEnv) {
    // Save & clear any pre-set env so we control the inputs.
    const char* saved_rank = std::getenv("RANK");
    const char* saved_lr = std::getenv("LOCAL_RANK");
    std::string saved_rank_s = saved_rank ? saved_rank : "";
    std::string saved_lr_s = saved_lr ? saved_lr : "";

    setenv("RANK", "3", 1);
    unsetenv("LOCAL_RANK");
    {
        DeviceMesh mesh(Device::Type::CPU, {2, 4}, {"dp", "tp"});
        EXPECT_EQ(mesh.get_mesh_rank(), 3);
    }

    // RANK out of range -> falls back to 0 (single-process safety net).
    setenv("RANK", "99", 1);
    {
        DeviceMesh mesh(Device::Type::CPU, {2, 4}, {"dp", "tp"});
        EXPECT_EQ(mesh.get_mesh_rank(), 0);
    }

    // No RANK, but LOCAL_RANK set: use LOCAL_RANK as the mesh rank.
    unsetenv("RANK");
    setenv("LOCAL_RANK", "2", 1);
    {
        DeviceMesh mesh(Device::Type::CPU, {2, 4}, {"dp", "tp"});
        EXPECT_EQ(mesh.get_mesh_rank(), 2);
    }

    // Restore.
    if (!saved_rank_s.empty()) setenv("RANK", saved_rank_s.c_str(), 1);
    else unsetenv("RANK");
    if (!saved_lr_s.empty()) setenv("LOCAL_RANK", saved_lr_s.c_str(), 1);
    else unsetenv("LOCAL_RANK");
}

TEST_F(DeviceMeshTest, HostLocalRankReadsLocalRankEnv) {
    const char* saved_lr = std::getenv("LOCAL_RANK");
    std::string saved_lr_s = saved_lr ? saved_lr : "";

    setenv("LOCAL_RANK", "7", 1);
    EXPECT_EQ(DeviceMesh::get_host_local_rank(), 7);

    unsetenv("LOCAL_RANK");
    EXPECT_EQ(DeviceMesh::get_host_local_rank(), 0);

    if (!saved_lr_s.empty()) setenv("LOCAL_RANK", saved_lr_s.c_str(), 1);
}

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

// --------------------------------------------------------------------------
// Audit B3 / A3-extended: per-axis sub-PG via ProcessGroupBase::split.
// --------------------------------------------------------------------------

TEST_F(DeviceMeshTest, ProcessGroupForDim_OneDMesh_ReturnsParentPG) {
    DeviceMesh mesh(Device::Type::CPU, {4}, {"dp"}, /*mesh_rank=*/0);
    auto pg = std::make_shared<SplittablePG>(0, 4, 4);
    mesh.set_process_group(pg);
    // 1-D mesh: single axis IS the entire mesh — no split needed.
    auto sub = mesh.process_group_for_dim("dp");
    EXPECT_EQ(sub.get(), pg.get());
    EXPECT_EQ(pg->split_calls, 0);
}

TEST_F(DeviceMeshTest, ProcessGroupForDim_MultiDMesh_CallsSplitOnce) {
    // 2D mesh [2 x 2]; world_size=4. mesh_rank=0 -> coord=(0,0).
    DeviceMesh mesh(Device::Type::CPU, {2, 2}, {"dp", "tp"}, /*mesh_rank=*/0);
    auto pg = std::make_shared<SplittablePG>(/*rank=*/0, /*ws=*/4,
                                              /*sub_ws=*/2);
    mesh.set_process_group(pg);

    auto tp_sub = mesh.process_group_for_dim("tp");
    ASSERT_NE(tp_sub, nullptr);
    EXPECT_EQ(pg->split_calls, 1);
    // color = my_coord[0] (other axes only) = 0
    // key   = my_coord[1] = 0
    EXPECT_EQ(pg->last_color, 0);
    EXPECT_EQ(pg->last_key, 0);
    EXPECT_EQ(tp_sub->world_size(), 2);

    // Second call is cached — no second split.
    auto tp_sub2 = mesh.process_group_for_dim("tp");
    EXPECT_EQ(tp_sub2.get(), tp_sub.get());
    EXPECT_EQ(pg->split_calls, 1);
}

TEST_F(DeviceMeshTest, ProcessGroupForDim_MultiDMesh_DifferentDimsDistinctSubPGs) {
    DeviceMesh mesh(Device::Type::CPU, {2, 2}, {"dp", "tp"}, /*mesh_rank=*/0);
    auto pg = std::make_shared<SplittablePG>(0, 4, 2);
    mesh.set_process_group(pg);

    auto dp_sub = mesh.process_group_for_dim("dp");
    auto tp_sub = mesh.process_group_for_dim("tp");

    EXPECT_NE(dp_sub.get(), tp_sub.get())
        << "Different mesh dims must produce distinct sub-PGs";
    EXPECT_EQ(pg->split_calls, 2);
}

TEST_F(DeviceMeshTest, ProcessGroupForDim_AxisSpansWholeWorld_ReturnsParent) {
    // Multi-D mesh whose `tp` axis equals the whole world (other axes
    // size 1). No split should happen — the tp sub-PG IS the parent.
    DeviceMesh mesh(Device::Type::CPU, {1, 4}, {"dp", "tp"}, /*mesh_rank=*/0);
    auto pg = std::make_shared<SplittablePG>(0, 4, 4);
    mesh.set_process_group(pg);
    auto sub = mesh.process_group_for_dim("tp");
    EXPECT_EQ(sub.get(), pg.get());
    EXPECT_EQ(pg->split_calls, 0);
}

TEST_F(DeviceMeshTest, ProcessGroupForDim_KeyMatchesAxisCoord) {
    // mesh_rank=3 in a 2x2 mesh -> coord=(1,1).
    // For axis "tp" (dim_idx=1): color = my_coord[0] = 1; key = my_coord[1] = 1.
    DeviceMesh mesh(Device::Type::CPU, {2, 2}, {"dp", "tp"}, /*mesh_rank=*/3);
    auto pg = std::make_shared<SplittablePG>(/*rank=*/3, /*ws=*/4, /*sub_ws=*/2);
    mesh.set_process_group(pg);
    (void)mesh.process_group_for_dim("tp");
    EXPECT_EQ(pg->last_color, 1);
    EXPECT_EQ(pg->last_key, 1);
}

// ============================================================================
// Multi-rank tests (real GlooProcessGroup, real OS processes) — FINDING 21
// ============================================================================
// SplittablePG's fake split() just fabricates a stub sub-PG with a given
// world_size -- it never actually negotiates group membership between real
// peers. GlooProcessGroup::split() does real work here (see its own header
// comment): it all_gathers (rank, color, key) triples over the PARENT group,
// computes new rank/size locally per color, and spins a fresh rendezvous on
// a per-color derived port. That negotiation can only be wrong in a way that
// involves multiple real processes -- e.g. two different colors' ranks
// cross-talking, or a sub-group's rank/world_size disagreeing with what its
// peers computed.
namespace {

class MultiRankDeviceMeshTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }

    void SetUp() override {
        const char* rank_env = std::getenv("RANK");
        const char* world_size_env = std::getenv("WORLD_SIZE");
        if (!rank_env || !world_size_env) {
            GTEST_SKIP() << "Distributed environment not available (RANK, WORLD_SIZE not set)";
        }
        rank_ = std::atoi(rank_env);
        world_size_ = std::atoi(world_size_env);
        if (world_size_ != 4) {
            GTEST_SKIP() << "MultiRankDeviceMeshTest is written for exactly 4 ranks";
        }

        const char* addr_env = std::getenv("MASTER_ADDR");
        const char* port_env = std::getenv("MASTER_PORT");
        std::string addr = addr_env ? addr_env : "127.0.0.1";
        int port = port_env ? std::atoi(port_env) : 29500;
        pg_ = std::make_shared<GlooProcessGroup>(rank_, world_size_, addr, port);
    }

    void TearDown() override {
        if (pg_) {
            pg_->barrier();
            pg_.reset();
        }
    }

    int rank_{-1};
    int world_size_{-1};
    std::shared_ptr<GlooProcessGroup> pg_;
};

} // namespace

// 2x2 mesh over 4 real ranks: "dp" is the row axis, "tp" is the column axis.
// mesh_rank = coord (dp_coord, tp_coord) = (rank/2, rank%2), so ranks {0,1}
// share dp_coord=0 (tp sub-group A) and ranks {2,3} share dp_coord=1 (tp
// sub-group B). Splitting on "tp" must produce two INDEPENDENT 2-rank
// sub-groups -- verified with a real all_reduce inside the sub-group: each
// sub-group rank contributes (sub_rank+1), so a correctly isolated 2-rank
// sub-group always sums to 1+2=3, regardless of which color it is. If
// split() ever let the two colors' ranks talk to each other (or miscounted
// world_size), this would either hang (waiting on a peer that isn't really
// in this sub-group) or observe a wrong sum.
TEST_F(MultiRankDeviceMeshTest, SplitProducesIndependentSubGroupsAcrossRealRanks) {
    DeviceMesh mesh(Device::Type::CPU, {2, 2}, {"dp", "tp"}, /*mesh_rank=*/rank_);
    mesh.set_process_group(pg_);

    auto tp_sub = mesh.process_group_for_dim("tp");
    ASSERT_NE(tp_sub, nullptr);
    EXPECT_EQ(tp_sub->world_size(), 2);

    int expected_color = rank_ / 2;
    int expected_key = rank_ % 2;
    EXPECT_EQ(tp_sub->rank(), expected_key)
        << "rank " << rank_ << " (color " << expected_color << ")";

    auto local = full({4}, static_cast<double>(tp_sub->rank() + 1),
                      DType::Float32, Device::cpu());
    tp_sub->all_reduce(local, ReduceOp::SUM);

    auto cpu_local = local.to(Device::cpu()).contiguous();
    const float* p = cpu_local.data<float>();
    for (int64_t i = 0; i < cpu_local.numel(); ++i) {
        EXPECT_FLOAT_EQ(p[i], 3.0f)
            << "rank " << rank_ << " (color " << expected_color
            << "): sub-group all_reduce should sum exactly its own 2 members (1+2=3)";
    }
}

// process_group_for_dim() caches the sub-PG: calling it twice for the same
// dim on the same rank must not re-split (matching the single-process
// ProcessGroupForDim_MultiDMesh_CallsSplitOnce contract), verified here
// against the REAL GlooProcessGroup::split() implementation.
TEST_F(MultiRankDeviceMeshTest, ProcessGroupForDimIsCachedWithRealSplit) {
    DeviceMesh mesh(Device::Type::CPU, {2, 2}, {"dp", "tp"}, /*mesh_rank=*/rank_);
    mesh.set_process_group(pg_);

    auto tp_sub_1 = mesh.process_group_for_dim("tp");
    auto tp_sub_2 = mesh.process_group_for_dim("tp");
    EXPECT_EQ(tp_sub_1.get(), tp_sub_2.get())
        << "rank " << rank_ << ": second process_group_for_dim call should be cached";
}
