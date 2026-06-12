// test_process_group_inf_f.cpp
//
// Wave Inf-F: native split() + all_to_all_single() on Gloo (paired
// send/recv) and MPI (MPI_Comm_split + MPI_Alltoall). True multi-rank
// coverage requires mpirun -n 4 / a Gloo cluster; this file pins the
// single-process degenerate cases + contract validation we can run
// inside the test harness.

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/process_group.hpp>
#include <tenzor/distributed/distributed.hpp>

using namespace tenzor;
using namespace tenzor::distributed;

namespace {

constexpr int kSmokePort = 29611;  // distinct from existing smoke ports

class ProcessGroupInfFTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

// Try to construct a single-rank Gloo PG. May fail if TCP setup
// errors on the host — caller falls back to GTEST_SKIP.
std::unique_ptr<GlooProcessGroup> try_make_gloo_pg() {
    try {
        return std::make_unique<GlooProcessGroup>(
            0, /*world_size=*/1, "localhost", kSmokePort);
    } catch (...) {
        return nullptr;
    }
}

}  // namespace

// ----------------------------------------------------------------------------
// Inf-F5: native GlooProcessGroup::all_to_all_single — single-rank case.
// ----------------------------------------------------------------------------
//
// In world_size=1 every chunk stays local. The native paired-send/recv
// loop body never executes (no remote peers); the function reduces to a
// single self-slice that becomes the output.
TEST_F(ProcessGroupInfFTest, GlooAllToAllSingle_WorldSize1_Identity) {
    auto pg = try_make_gloo_pg();
    if (!pg) {
        SKIP_WITH_REASON(tenzor::testing::SkipReason::BackendUnavailable,
            "Gloo TCP rendezvous unavailable on this host");
        return;
    }

    auto in = zeros({4}, DType::Float32, Device::cpu());
    auto* p = in.data<float>();
    p[0] = 1.0f; p[1] = 2.0f; p[2] = 3.0f; p[3] = 4.0f;
    auto out = zeros({4}, DType::Float32, Device::cpu());
    pg->all_to_all_single(out, in);

    auto* op = out.data<float>();
    EXPECT_FLOAT_EQ(op[0], 1.0f);
    EXPECT_FLOAT_EQ(op[1], 2.0f);
    EXPECT_FLOAT_EQ(op[2], 3.0f);
    EXPECT_FLOAT_EQ(op[3], 4.0f);
}

// ----------------------------------------------------------------------------
// Inf-F5: contract — input.shape[0] must be divisible by world_size.
// ----------------------------------------------------------------------------
TEST_F(ProcessGroupInfFTest, GlooAllToAllSingle_NonDivisibleShapeThrows) {
    auto pg = try_make_gloo_pg();
    if (!pg) GTEST_SKIP() << "Gloo TCP rendezvous unavailable on this host";
    // world_size = 1 divides everything; this branch only catches the
    // empty-shape validation. Empty input → throws "at least 1 dim".
    auto empty_in  = zeros({}, DType::Float32, Device::cpu());
    auto empty_out = zeros({}, DType::Float32, Device::cpu());
    EXPECT_THROW(pg->all_to_all_single(empty_out, empty_in),
                 std::invalid_argument);
}

// ----------------------------------------------------------------------------
// Inf-F4: Gloo split — single-rank degenerate. color < 0 returns nullptr;
// color ≥ 0 produces a 1-rank subgroup.
// ----------------------------------------------------------------------------
TEST_F(ProcessGroupInfFTest, GlooSplit_NoColor_ReturnsNull) {
    auto pg = try_make_gloo_pg();
    if (!pg) GTEST_SKIP() << "Gloo TCP rendezvous unavailable on this host";
    EXPECT_EQ(pg->split(-1, 0), nullptr);
}

TEST_F(ProcessGroupInfFTest, GlooSplit_SingleRank_ReturnsSubGroupOfOne) {
    auto pg = try_make_gloo_pg();
    if (!pg) GTEST_SKIP() << "Gloo TCP rendezvous unavailable on this host";
    // Audit-5: removed a `try { sub = pg->split(...); } catch (...) {
    // GTEST_SKIP("child rendezvous unavailable"); }` wrapper. split() is the
    // operation under test here — wrapping it turned a real split() bug (bad
    // subgroup construction, wrong rank/world_size, throw on the single-rank
    // path) into a silent skip. The parent PG already constructed successfully
    // via the try_make_gloo_pg() precondition above, so a throw from split()
    // should surface as a failure. The documented null-return rendezvous-
    // failure path is still tolerated below.
    std::shared_ptr<ProcessGroupBase> sub = pg->split(/*color=*/0, /*key=*/0);
    if (!sub) GTEST_SKIP() << "Gloo split returned null (rendezvous failure)";
    EXPECT_EQ(sub->rank(), 0);
    EXPECT_EQ(sub->world_size(), 1);
}

// ----------------------------------------------------------------------------
// Inf-F5: contract — output must match input shape and dtype.
// ----------------------------------------------------------------------------
TEST_F(ProcessGroupInfFTest, GlooAllToAllSingle_ShapeMismatchThrows) {
    auto pg = try_make_gloo_pg();
    if (!pg) {
        SKIP_WITH_REASON(tenzor::testing::SkipReason::BackendUnavailable,
            "Gloo TCP rendezvous unavailable on this host");
        return;
    }

    auto in  = zeros({4}, DType::Float32, Device::cpu());
    auto out_wrong_shape = zeros({2}, DType::Float32, Device::cpu());
    EXPECT_THROW(pg->all_to_all_single(out_wrong_shape, in),
                 std::invalid_argument);

    auto out_wrong_dtype = zeros({4}, DType::Float64, Device::cpu());
    EXPECT_THROW(pg->all_to_all_single(out_wrong_dtype, in),
                 std::invalid_argument);
}

// ----------------------------------------------------------------------------
// Inf-F2/F3: MPIProcessGroup tests — only compile when MPI is enabled.
// ----------------------------------------------------------------------------
#ifdef TENZOR_HAS_MPI

namespace {
// Try to construct a single-rank MPI PG; MPI may not be runnable in this
// test process (no mpirun harness), so all of these gracefully skip.
std::unique_ptr<MPIProcessGroup> try_make_mpi_pg() {
    try {
        // MPI_Init from inside the test process; size=1, rank=0.
        return std::make_unique<MPIProcessGroup>(0, 1);
    } catch (...) {
        return nullptr;
    }
}
}  // namespace

TEST_F(ProcessGroupInfFTest, MPIProcessGroup_ConstructsSingleProcess) {
    auto pg = try_make_mpi_pg();
    if (!pg) GTEST_SKIP() << "MPI not runnable as a single in-process rank";
    EXPECT_EQ(pg->rank(), 0);
    EXPECT_EQ(pg->world_size(), 1);
}

TEST_F(ProcessGroupInfFTest, MPIAllToAllSingle_WorldSize1_Identity) {
    auto pg = try_make_mpi_pg();
    if (!pg) GTEST_SKIP() << "MPI not runnable as a single in-process rank";

    auto in = zeros({4}, DType::Float32, Device::cpu());
    auto* p = in.data<float>();
    p[0] = 1.0f; p[1] = 2.0f; p[2] = 3.0f; p[3] = 4.0f;
    auto out = zeros({4}, DType::Float32, Device::cpu());
    pg->all_to_all_single(out, in);
    auto* op = out.data<float>();
    for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(op[i], p[i]);
}

TEST_F(ProcessGroupInfFTest, MPISplit_NoColor_ReturnsNull) {
    auto pg = try_make_mpi_pg();
    if (!pg) GTEST_SKIP() << "MPI not runnable as a single in-process rank";
    // color == -1 → MPI_UNDEFINED → MPI_COMM_NULL → nullptr.
    auto sub = pg->split(-1, /*key=*/0);
    EXPECT_EQ(sub, nullptr);
}

TEST_F(ProcessGroupInfFTest, MPISplit_SameColor_NewGroupOfOne) {
    auto pg = try_make_mpi_pg();
    if (!pg) GTEST_SKIP() << "MPI not runnable as a single in-process rank";
    auto sub = pg->split(/*color=*/0, /*key=*/0);
    ASSERT_NE(sub, nullptr);
    EXPECT_EQ(sub->rank(), 0);
    EXPECT_EQ(sub->world_size(), 1);
}

#endif  // TENZOR_HAS_MPI
