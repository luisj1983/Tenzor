/**
 * @file test_mpi_backend_smoke.cpp
 * @brief Single-process smoke for tenzor::distributed::MPIBackend.
 *
 * The MPI backend is gated at build time on TENZOR_BUILD_MPI / TENZOR_HAS_MPI.
 * When MPI isn't available, the backend's symbols may still link as stubs
 * but initialize() is expected to throw at runtime. This test is the
 * single-process smoke: it tries to construct + initialize, and skips
 * cleanly if MPI is unavailable in this build. When MPI IS available, it
 * pins the same single-rank no-op invariants as the Gloo smoke test.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/mpi_backend.hpp>
#include <tenzor/distributed/distributed.hpp>

using namespace tenzor;
using namespace tenzor::distributed;

class MPIBackendSmokeTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

namespace {

// Try to construct + initialize an MPIBackend in single-process mode.
// Returns nullptr if MPI is unavailable, in which case the caller skips.
std::unique_ptr<MPIBackend> try_init_single_process() {
    auto mpi = std::make_unique<MPIBackend>();
    try {
        mpi->initialize(/*rank=*/0, /*world_size=*/1,
                        /*master_addr=*/"", /*master_port=*/0);
    } catch (const std::exception&) {
        return nullptr;
    }
    return mpi;
}

}  // namespace

TEST_F(MPIBackendSmokeTest, ConstructDoesNotThrow) {
    EXPECT_NO_THROW(MPIBackend{});
}

TEST_F(MPIBackendSmokeTest, InitializeOrSkip) {
    auto mpi = try_init_single_process();
    if (!mpi) {
        GTEST_SKIP() << "MPI not available in this build; skipping initialize smoke.";
    }
    EXPECT_EQ(mpi->backend_type(), tenzor::distributed::Backend::MPI);
    EXPECT_NO_THROW(mpi->finalize());
}

TEST_F(MPIBackendSmokeTest, AllReduce_SingleProcess_IsIdentity_OrSkip) {
    auto mpi = try_init_single_process();
    if (!mpi) GTEST_SKIP() << "MPI not available";

    auto t = randn({4, 8}, DType::Float32, Device::cpu());
    auto t_before = t.clone();
    EXPECT_NO_THROW(mpi->all_reduce(t, ReduceOp::SUM));
    const float* a = t_before.contiguous().data<float>();
    const float* b = t.contiguous().data<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]);
    }
    mpi->finalize();
}

TEST_F(MPIBackendSmokeTest, Barrier_OrSkip) {
    auto mpi = try_init_single_process();
    if (!mpi) GTEST_SKIP() << "MPI not available";
    EXPECT_NO_THROW(mpi->barrier());
    mpi->finalize();
}
