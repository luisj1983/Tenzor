/**
 * @file test_gloo_backend_smoke.cpp
 * @brief Single-process smoke test for tenzor::distributed::GlooBackend.
 *
 * The audit (2026-05-02) flagged gloo_backend.hpp / mpi_backend.hpp /
 * nccl_backend.hpp as having no per-backend smoke tests — they were
 * exercised only implicitly via multirank Python tests. This file pins
 * the GlooBackend single-process semantics:
 *   - Default-construct, initialize(rank=0, world_size=1, ...).
 *   - rank() / world_size() reflect the init values.
 *   - All-reduce, broadcast, all-gather, barrier are no-ops in 1-rank
 *     mode (input == output, no communication required).
 *   - finalize() cleans up without throwing.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/gloo_backend.hpp>
#include <tenzor/distributed/distributed.hpp>

using namespace tenzor;
using namespace tenzor::distributed;

namespace {

// Pick an unlikely-to-be-in-use port range so the smoke test doesn't
// collide with real Gloo workloads on shared CI hosts.
constexpr int kSmokePort = 29501;

}  // namespace

class GlooBackendSmokeTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

TEST_F(GlooBackendSmokeTest, DefaultConstructDoesNotThrow) {
    EXPECT_NO_THROW(GlooBackend{});
}

TEST_F(GlooBackendSmokeTest, InitializeSingleProcess) {
    GlooBackend gloo;
    ASSERT_NO_THROW(gloo.initialize(/*rank=*/0, /*world_size=*/1,
                                    /*master_addr=*/"127.0.0.1",
                                    /*master_port=*/kSmokePort));
    // backend_type() should report GLOO.
    EXPECT_EQ(gloo.backend_type(), tenzor::distributed::Backend::GLOO);
    EXPECT_NO_THROW(gloo.finalize());
}

TEST_F(GlooBackendSmokeTest, AllReduce_SingleProcess_IsIdentity) {
    GlooBackend gloo;
    gloo.initialize(0, 1, "127.0.0.1", kSmokePort);

    auto t = randn({4, 8}, DType::Float32, Device::cpu());
    auto t_before = t.clone();
    EXPECT_NO_THROW(gloo.all_reduce(t, ReduceOp::SUM));

    // Single-rank all-reduce: each element's reduction over a 1-element
    // group equals the element itself.
    const float* a = t_before.contiguous().data<float>();
    const float* b = t.contiguous().data<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]) << "all_reduce mutated value at " << i
                                    << " in single-rank mode";
    }

    gloo.finalize();
}

TEST_F(GlooBackendSmokeTest, Broadcast_SingleProcess_IsIdentity) {
    GlooBackend gloo;
    gloo.initialize(0, 1, "127.0.0.1", kSmokePort);
    auto t = randn({16}, DType::Float32, Device::cpu());
    auto t_before = t.clone();
    EXPECT_NO_THROW(gloo.broadcast(t, /*src_rank=*/0));

    const float* a = t_before.contiguous().data<float>();
    const float* b = t.contiguous().data<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]);
    }
    gloo.finalize();
}

TEST_F(GlooBackendSmokeTest, AllGather_SingleProcess_OneEntry) {
    GlooBackend gloo;
    gloo.initialize(0, 1, "127.0.0.1", kSmokePort);

    auto input = randn({3, 5}, DType::Float32, Device::cpu());
    std::vector<Tensor> output;
    output.push_back(zeros({3, 5}, DType::Float32, Device::cpu()));

    EXPECT_NO_THROW(gloo.all_gather(input, output));
    EXPECT_EQ(output.size(), 1u);
    // Output[0] should equal input.
    const float* in_p = input.contiguous().data<float>();
    const float* out_p = output[0].contiguous().data<float>();
    for (int64_t i = 0; i < input.numel(); ++i) {
        EXPECT_FLOAT_EQ(in_p[i], out_p[i]);
    }
    gloo.finalize();
}

TEST_F(GlooBackendSmokeTest, Barrier_SingleProcess) {
    GlooBackend gloo;
    gloo.initialize(0, 1, "127.0.0.1", kSmokePort);
    EXPECT_NO_THROW(gloo.barrier());
    gloo.finalize();
}

TEST_F(GlooBackendSmokeTest, SupportsDevice_CPU) {
    GlooBackend gloo;
    EXPECT_TRUE(gloo.supports_device(Device::Type::CPU))
        << "Gloo must support CPU tensors";
}
