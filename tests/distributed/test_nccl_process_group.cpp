/**
 * @file test_nccl_process_group.cpp
 * @brief Single-process smoke + contract tests for tenzor::distributed::NCCLProcessGroup.
 *
 * FINDING 21: NCCLProcessGroup (include/tenzor/distributed/process_group.hpp:
 * 163-236, src/distributed/nccl_process_group.cpp, 811 lines) is the class
 * DistributedContext actually constructs for Backend::NCCL (see
 * src/distributed/distributed.cpp's DistributedContext::initialize()) — it
 * is distinct from NCCLBackend (tests/distributed/test_nccl_backend_smoke.cpp),
 * which implements the older CommunicationBackend interface used by
 * ProcessGroup::create_process_group(). Before this file, NCCLProcessGroup
 * was never constructed by any test anywhere in the repo (its siblings
 * GlooProcessGroup and MPIProcessGroup both get real construction + collective
 * tests in test_process_group_inf_f.cpp, which links only tenzor_core and so
 * structurally cannot reference NCCLProcessGroup — it lives in the GPU
 * backend DSOs, same reasoning as NCCLBackend). This file mirrors
 * test_process_group_inf_f.cpp's GlooProcessGroup coverage for the single-rank
 * degenerate case, directly against NCCLProcessGroup rather than through the
 * ProcessGroupBase interface.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/process_group.hpp>
#include <tenzor/distributed/distributed.hpp>

using namespace tenzor;
using namespace tenzor::distributed;

namespace {

constexpr int kSmokePort = 29506;

bool cuda_available() {
    try {
        Device::cuda(0);
        auto t = randn({2}, DType::Float32, Device::cpu());
        (void)t.to(Device::cuda(0));
        return true;
    } catch (...) {
        return false;
    }
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

class NCCLProcessGroupTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

TEST_F(NCCLProcessGroupTest, ConstructSingleRank_OrSkip) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    auto pg = try_make_nccl_pg();
    if (!pg) GTEST_SKIP() << "NCCL not available in this build";
    EXPECT_EQ(pg->rank(), 0);
    EXPECT_EQ(pg->world_size(), 1);
    EXPECT_TRUE(pg->supports_async_stream());
}

TEST_F(NCCLProcessGroupTest, AllReduce_SingleRank_IsIdentity_OrSkip) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    auto pg = try_make_nccl_pg();
    if (!pg) GTEST_SKIP() << "NCCL not available";

    auto t = randn({4, 4}, DType::Float32, Device::cpu()).to(Device::cuda(0));
    auto t_before = t.clone();
    EXPECT_NO_THROW(pg->all_reduce(t, ReduceOp::SUM));
    Device::cuda(0).synchronize();
    auto a_cpu = t_before.to(Device::cpu()).contiguous();
    auto b_cpu = t.to(Device::cpu()).contiguous();
    const float* a = a_cpu.data<float>();
    const float* b = b_cpu.data<float>();
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]) << "all_reduce mutated tensor in single-rank mode";
    }
}

TEST_F(NCCLProcessGroupTest, Broadcast_SingleRank_IsIdentity_OrSkip) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    auto pg = try_make_nccl_pg();
    if (!pg) GTEST_SKIP() << "NCCL not available";

    auto t = randn({16}, DType::Float32, Device::cpu()).to(Device::cuda(0));
    auto t_before = t.clone();
    EXPECT_NO_THROW(pg->broadcast(t, /*src_rank=*/0));
    Device::cuda(0).synchronize();
    auto a_cpu = t_before.to(Device::cpu()).contiguous();
    auto b_cpu = t.to(Device::cpu()).contiguous();
    const float* a = a_cpu.data<float>();
    const float* b = b_cpu.data<float>();
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]);
    }
}

TEST_F(NCCLProcessGroupTest, AllGather_SingleRank_OneEntry_OrSkip) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    auto pg = try_make_nccl_pg();
    if (!pg) GTEST_SKIP() << "NCCL not available";

    // NOTE: NCCLProcessGroup::all_gather(output, input) takes output FIRST,
    // input SECOND -- the reverse parameter order of NCCLBackend::all_gather
    // (input, output). Both match their respective base-class contracts
    // (ProcessGroupBase vs CommunicationBackend); this test uses
    // ProcessGroupBase's order since NCCLProcessGroup implements that one.
    auto input = randn({3, 5}, DType::Float32, Device::cpu()).to(Device::cuda(0));
    std::vector<Tensor> output;
    output.push_back(zeros({3, 5}, DType::Float32, Device::cuda(0)));

    EXPECT_NO_THROW(pg->all_gather(output, input));
    ASSERT_EQ(output.size(), 1u);
    Device::cuda(0).synchronize();
    auto in_cpu = input.to(Device::cpu()).contiguous();
    auto out_cpu = output[0].to(Device::cpu()).contiguous();
    const float* in_p = in_cpu.data<float>();
    const float* out_p = out_cpu.data<float>();
    for (int64_t i = 0; i < in_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(in_p[i], out_p[i]);
    }
}

TEST_F(NCCLProcessGroupTest, ReduceScatter_SingleRank_MatchesInput_OrSkip) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    auto pg = try_make_nccl_pg();
    if (!pg) GTEST_SKIP() << "NCCL not available";

    auto input = full({3}, 7.0, DType::Float32, Device::cuda(0));
    auto output = zeros({3}, DType::Float32, Device::cuda(0));
    std::vector<Tensor> inputs{input};

    EXPECT_NO_THROW(pg->reduce_scatter(output, std::span<const Tensor>(inputs)));
    Device::cuda(0).synchronize();
    auto out_cpu = output.to(Device::cpu()).contiguous();
    EXPECT_FLOAT_EQ(out_cpu.data<float>()[0], 7.0f);
}

TEST_F(NCCLProcessGroupTest, AllToAllSingle_SingleRank_IsIdentity_OrSkip) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    auto pg = try_make_nccl_pg();
    if (!pg) GTEST_SKIP() << "NCCL not available";

    auto in = full({4}, 0.0, DType::Float32, Device::cuda(0));
    auto in_cpu_src = zeros({4}, DType::Float32, Device::cpu());
    float* p = in_cpu_src.data<float>();
    p[0] = 1.0f; p[1] = 2.0f; p[2] = 3.0f; p[3] = 4.0f;
    in = in_cpu_src.to(Device::cuda(0));
    auto out = zeros({4}, DType::Float32, Device::cuda(0));

    EXPECT_NO_THROW(pg->all_to_all_single(out, in));
    Device::cuda(0).synchronize();
    auto in_c = in.to(Device::cpu()).contiguous();
    auto out_c = out.to(Device::cpu()).contiguous();
    for (int64_t i = 0; i < in_c.numel(); ++i) {
        EXPECT_FLOAT_EQ(in_c.data<float>()[i], out_c.data<float>()[i]);
    }
}

TEST_F(NCCLProcessGroupTest, Split_SingleRank_ReturnsSubGroupOfOne_OrSkip) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    auto pg = try_make_nccl_pg();
    if (!pg) GTEST_SKIP() << "NCCL not available";

    std::shared_ptr<ProcessGroupBase> sub;
    EXPECT_NO_THROW(sub = pg->split(/*color=*/0, /*key=*/0));
    ASSERT_NE(sub, nullptr);
    EXPECT_EQ(sub->rank(), 0);
    EXPECT_EQ(sub->world_size(), 1);
}

TEST_F(NCCLProcessGroupTest, Barrier_SingleRank_OrSkip) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    auto pg = try_make_nccl_pg();
    if (!pg) GTEST_SKIP() << "NCCL not available";
    EXPECT_NO_THROW(pg->barrier());
}
