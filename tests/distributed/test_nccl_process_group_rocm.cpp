/**
 * @file test_nccl_process_group_rocm.cpp
 * @brief Single-process smoke + contract tests for NCCLProcessGroup, ROCm/RCCL flavor.
 *
 * FINDING 21 + FINDING 4: mirrors test_nccl_process_group.cpp but linked
 * against tenzor_backend_rocm only, for the same reason
 * test_nccl_backend_smoke_rocm.cpp is a separate executable from
 * test_nccl_backend_smoke.cpp -- NCCLProcessGroup is a distinct compiled
 * type per backend DSO (TENZOR_USE_CUDA vs TENZOR_USE_ROCM), and the
 * dynamic loader would silently resolve tenzor::distributed::
 * NCCLProcessGroup to whichever .so loads first if both were linked into
 * one binary.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/process_group.hpp>
#include <tenzor/distributed/distributed.hpp>

using namespace tenzor;
using namespace tenzor::distributed;

namespace {

constexpr int kSmokePort = 29507;

bool rocm_available() {
    try {
        Device::rocm(0);
        auto t = randn({2}, DType::Float32, Device::cpu());
        (void)t.to(Device::rocm(0));
        return true;
    } catch (...) {
        return false;
    }
}

// TENZOR_REQUIRE_MULTI_BACKEND=1 escalates "ROCm/RCCL not available" from a silent
// GTEST_SKIP() to a hard FAIL() — matches the project-wide convention (see
// tests/backend_parity/parity_test_utils.hpp's REQUIRE_MULTI_BACKEND_OR_SKIP) so a CI
// job that expects a GPU backend to be present hard-fails when it silently failed to
// initialize, instead of the whole suite reading as "all skipped, green".
bool require_multi_backend() {
    const char* v = std::getenv("TENZOR_REQUIRE_MULTI_BACKEND");
    return v != nullptr && std::string(v) == "1";
}

std::unique_ptr<NCCLProcessGroup> try_make_nccl_pg() {
    try {
        return std::make_unique<NCCLProcessGroup>(
            0, /*world_size=*/1, "127.0.0.1", kSmokePort);
    } catch (...) {
        return nullptr;
    }
}

}  // namespace

class NCCLProcessGroupRocmTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

TEST_F(NCCLProcessGroupRocmTest, ConstructSingleRank_OrSkip) {
    if (!rocm_available()) {
        if (require_multi_backend()) FAIL() << "ROCm not available (TENZOR_REQUIRE_MULTI_BACKEND=1)";
        GTEST_SKIP() << "ROCm not available";
    }
    auto pg = try_make_nccl_pg();
    if (!pg) {
        if (require_multi_backend()) FAIL() << "RCCL not available in this build (TENZOR_REQUIRE_MULTI_BACKEND=1)";
        GTEST_SKIP() << "RCCL not available in this build";
    }
    EXPECT_EQ(pg->rank(), 0);
    EXPECT_EQ(pg->world_size(), 1);
    EXPECT_TRUE(pg->supports_async_stream());
}

TEST_F(NCCLProcessGroupRocmTest, AllReduce_SingleRank_IsIdentity_OrSkip) {
    if (!rocm_available()) {
        if (require_multi_backend()) FAIL() << "ROCm not available (TENZOR_REQUIRE_MULTI_BACKEND=1)";
        GTEST_SKIP() << "ROCm not available";
    }
    auto pg = try_make_nccl_pg();
    if (!pg) {
        if (require_multi_backend()) FAIL() << "RCCL not available (TENZOR_REQUIRE_MULTI_BACKEND=1)";
        GTEST_SKIP() << "RCCL not available";
    }

    auto t = randn({4, 4}, DType::Float32, Device::cpu()).to(Device::rocm(0));
    auto t_before = t.clone();
    EXPECT_NO_THROW(pg->all_reduce(t, ReduceOp::SUM));
    Device::rocm(0).synchronize();
    auto a_cpu = t_before.to(Device::cpu()).contiguous();
    auto b_cpu = t.to(Device::cpu()).contiguous();
    const float* a = a_cpu.data<float>();
    const float* b = b_cpu.data<float>();
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]) << "all_reduce mutated tensor in single-rank mode";
    }
}

TEST_F(NCCLProcessGroupRocmTest, Broadcast_SingleRank_IsIdentity_OrSkip) {
    if (!rocm_available()) {
        if (require_multi_backend()) FAIL() << "ROCm not available (TENZOR_REQUIRE_MULTI_BACKEND=1)";
        GTEST_SKIP() << "ROCm not available";
    }
    auto pg = try_make_nccl_pg();
    if (!pg) {
        if (require_multi_backend()) FAIL() << "RCCL not available (TENZOR_REQUIRE_MULTI_BACKEND=1)";
        GTEST_SKIP() << "RCCL not available";
    }

    auto t = randn({16}, DType::Float32, Device::cpu()).to(Device::rocm(0));
    auto t_before = t.clone();
    EXPECT_NO_THROW(pg->broadcast(t, /*src_rank=*/0));
    Device::rocm(0).synchronize();
    auto a_cpu = t_before.to(Device::cpu()).contiguous();
    auto b_cpu = t.to(Device::cpu()).contiguous();
    const float* a = a_cpu.data<float>();
    const float* b = b_cpu.data<float>();
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]);
    }
}

TEST_F(NCCLProcessGroupRocmTest, AllGather_SingleRank_OneEntry_OrSkip) {
    if (!rocm_available()) {
        if (require_multi_backend()) FAIL() << "ROCm not available (TENZOR_REQUIRE_MULTI_BACKEND=1)";
        GTEST_SKIP() << "ROCm not available";
    }
    auto pg = try_make_nccl_pg();
    if (!pg) {
        if (require_multi_backend()) FAIL() << "RCCL not available (TENZOR_REQUIRE_MULTI_BACKEND=1)";
        GTEST_SKIP() << "RCCL not available";
    }

    auto input = randn({3, 5}, DType::Float32, Device::cpu()).to(Device::rocm(0));
    std::vector<Tensor> output;
    output.push_back(zeros({3, 5}, DType::Float32, Device::rocm(0)));

    EXPECT_NO_THROW(pg->all_gather(output, input));
    ASSERT_EQ(output.size(), 1u);
    Device::rocm(0).synchronize();
    auto in_cpu = input.to(Device::cpu()).contiguous();
    auto out_cpu = output[0].to(Device::cpu()).contiguous();
    const float* in_p = in_cpu.data<float>();
    const float* out_p = out_cpu.data<float>();
    for (int64_t i = 0; i < in_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(in_p[i], out_p[i]);
    }
}

TEST_F(NCCLProcessGroupRocmTest, ReduceScatter_SingleRank_MatchesInput_OrSkip) {
    if (!rocm_available()) {
        if (require_multi_backend()) FAIL() << "ROCm not available (TENZOR_REQUIRE_MULTI_BACKEND=1)";
        GTEST_SKIP() << "ROCm not available";
    }
    auto pg = try_make_nccl_pg();
    if (!pg) {
        if (require_multi_backend()) FAIL() << "RCCL not available (TENZOR_REQUIRE_MULTI_BACKEND=1)";
        GTEST_SKIP() << "RCCL not available";
    }

    auto input = full({3}, 7.0, DType::Float32, Device::rocm(0));
    auto output = zeros({3}, DType::Float32, Device::rocm(0));
    std::vector<Tensor> inputs{input};

    EXPECT_NO_THROW(pg->reduce_scatter(output, std::span<const Tensor>(inputs)));
    Device::rocm(0).synchronize();
    auto out_cpu = output.to(Device::cpu()).contiguous();
    EXPECT_FLOAT_EQ(out_cpu.data<float>()[0], 7.0f);
}

TEST_F(NCCLProcessGroupRocmTest, AllToAllSingle_SingleRank_IsIdentity_OrSkip) {
    if (!rocm_available()) {
        if (require_multi_backend()) FAIL() << "ROCm not available (TENZOR_REQUIRE_MULTI_BACKEND=1)";
        GTEST_SKIP() << "ROCm not available";
    }
    auto pg = try_make_nccl_pg();
    if (!pg) {
        if (require_multi_backend()) FAIL() << "RCCL not available (TENZOR_REQUIRE_MULTI_BACKEND=1)";
        GTEST_SKIP() << "RCCL not available";
    }

    auto in_cpu_src = zeros({4}, DType::Float32, Device::cpu());
    float* p = in_cpu_src.data<float>();
    p[0] = 1.0f; p[1] = 2.0f; p[2] = 3.0f; p[3] = 4.0f;
    auto in = in_cpu_src.to(Device::rocm(0));
    auto out = zeros({4}, DType::Float32, Device::rocm(0));

    EXPECT_NO_THROW(pg->all_to_all_single(out, in));
    Device::rocm(0).synchronize();
    auto in_c = in.to(Device::cpu()).contiguous();
    auto out_c = out.to(Device::cpu()).contiguous();
    for (int64_t i = 0; i < in_c.numel(); ++i) {
        EXPECT_FLOAT_EQ(in_c.data<float>()[i], out_c.data<float>()[i]);
    }
}

TEST_F(NCCLProcessGroupRocmTest, Split_SingleRank_ReturnsSubGroupOfOne_OrSkip) {
    if (!rocm_available()) {
        if (require_multi_backend()) FAIL() << "ROCm not available (TENZOR_REQUIRE_MULTI_BACKEND=1)";
        GTEST_SKIP() << "ROCm not available";
    }
    auto pg = try_make_nccl_pg();
    if (!pg) {
        if (require_multi_backend()) FAIL() << "RCCL not available (TENZOR_REQUIRE_MULTI_BACKEND=1)";
        GTEST_SKIP() << "RCCL not available";
    }

    std::shared_ptr<ProcessGroupBase> sub;
    EXPECT_NO_THROW(sub = pg->split(/*color=*/0, /*key=*/0));
    ASSERT_NE(sub, nullptr);
    EXPECT_EQ(sub->rank(), 0);
    EXPECT_EQ(sub->world_size(), 1);
}

TEST_F(NCCLProcessGroupRocmTest, Barrier_SingleRank_OrSkip) {
    if (!rocm_available()) {
        if (require_multi_backend()) FAIL() << "ROCm not available (TENZOR_REQUIRE_MULTI_BACKEND=1)";
        GTEST_SKIP() << "ROCm not available";
    }
    auto pg = try_make_nccl_pg();
    if (!pg) {
        if (require_multi_backend()) FAIL() << "RCCL not available (TENZOR_REQUIRE_MULTI_BACKEND=1)";
        GTEST_SKIP() << "RCCL not available";
    }
    EXPECT_NO_THROW(pg->barrier());
}
