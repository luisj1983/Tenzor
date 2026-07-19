/**
 * @file test_nccl_backend_smoke.cpp
 * @brief Single-process smoke for tenzor::distributed::NCCLBackend.
 *
 * NCCL is gated on TENZOR_HAS_NCCL + a CUDA-capable GPU. The single-rank
 * mode is the natural smoke-test surface here: NCCL 1-rank all-reduce is
 * a no-op, broadcast is identity, etc. If NCCL or CUDA isn't available
 * the test skips cleanly.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/nccl_backend.hpp>
#include <tenzor/distributed/distributed.hpp>

using namespace tenzor;
using namespace tenzor::distributed;

class NCCLBackendSmokeTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

namespace {

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

std::unique_ptr<NCCLBackend> try_init_single_process() {
    auto nccl = std::make_unique<NCCLBackend>();
    try {
        nccl->initialize(/*rank=*/0, /*world_size=*/1,
                         /*master_addr=*/"127.0.0.1", /*master_port=*/29502);
    } catch (const std::exception&) {
        return nullptr;
    }
    return nccl;
}

}  // namespace

TEST_F(NCCLBackendSmokeTest, ConstructDoesNotThrow) {
    EXPECT_NO_THROW(NCCLBackend{});
}

TEST_F(NCCLBackendSmokeTest, InitializeOrSkip) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    auto nccl = try_init_single_process();
    if (!nccl) GTEST_SKIP() << "NCCL not available in this build";
    EXPECT_EQ(nccl->backend_type(), tenzor::distributed::Backend::NCCL);
    EXPECT_TRUE(nccl->supports_async_stream());
    EXPECT_NO_THROW(nccl->finalize());
}

TEST_F(NCCLBackendSmokeTest, AllReduce_SingleProcess_OnGPU_OrSkip) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    auto nccl = try_init_single_process();
    if (!nccl) GTEST_SKIP() << "NCCL not available";

    auto t = randn({4, 4}, DType::Float32, Device::cpu()).to(Device::cuda(0));
    auto t_before = t.clone();
    EXPECT_NO_THROW(nccl->all_reduce(t, ReduceOp::SUM));
    Device::cuda(0).synchronize();
    auto a_cpu = t_before.to(Device::cpu()).contiguous();
    auto b_cpu = t.to(Device::cpu()).contiguous();
    const float* a = a_cpu.data<float>();
    const float* b = b_cpu.data<float>();
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]) << "all_reduce mutated tensor in single-rank mode";
    }
    nccl->finalize();
}

// FINDING 4: exercises the actual backend-registry factory hook
// (Backend::create_comm_backend()) via ProcessGroup::create_process_group(),
// not just direct NCCLBackend construction — closes the "backend-registry
// factory" gap the finding called out separately from build wiring.
TEST_F(NCCLBackendSmokeTest, CreateProcessGroup_ViaFactory_OrSkip) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    std::shared_ptr<ProcessGroup> pg;
    try {
        pg = ProcessGroup::create_process_group(
            tenzor::distributed::Backend::NCCL, /*rank=*/0, /*world_size=*/1,
            "127.0.0.1", /*master_port=*/29504);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "NCCL not available via factory: " << e.what();
    }
    ASSERT_NE(pg, nullptr);
    auto t = randn({4, 4}, DType::Float32, Device::cpu()).to(Device::cuda(0));
    auto t_before = t.clone();
    EXPECT_NO_THROW(pg->all_reduce(t, ReduceOp::SUM));
    Device::cuda(0).synchronize();
    auto a_cpu = t_before.to(Device::cpu()).contiguous();
    auto b_cpu = t.to(Device::cpu()).contiguous();
    const float* a = a_cpu.data<float>();
    const float* b = b_cpu.data<float>();
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i])
            << "factory-created NCCL process group's all_reduce mutated "
               "tensor in single-rank mode";
    }
}

TEST_F(NCCLBackendSmokeTest, Barrier_OrSkip) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    auto nccl = try_init_single_process();
    if (!nccl) GTEST_SKIP() << "NCCL not available";
    EXPECT_NO_THROW(nccl->barrier());
    nccl->finalize();
}

// FINDING 4: test_gloo_backend_smoke.cpp covers Broadcast and AllGather (both
// implemented in NCCLProcessGroup / exercised here at the NCCLBackend layer);
// this file previously stopped at all_reduce/barrier, leaving those two
// collectives with zero smoke coverage on either GPU backend.
TEST_F(NCCLBackendSmokeTest, Broadcast_SingleProcess_IsIdentity_OrSkip) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    auto nccl = try_init_single_process();
    if (!nccl) GTEST_SKIP() << "NCCL not available";

    auto t = randn({16}, DType::Float32, Device::cpu()).to(Device::cuda(0));
    auto t_before = t.clone();
    EXPECT_NO_THROW(nccl->broadcast(t, /*src_rank=*/0));
    Device::cuda(0).synchronize();
    auto a_cpu = t_before.to(Device::cpu()).contiguous();
    auto b_cpu = t.to(Device::cpu()).contiguous();
    const float* a = a_cpu.data<float>();
    const float* b = b_cpu.data<float>();
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]) << "broadcast mutated tensor in single-rank mode";
    }
    nccl->finalize();
}

TEST_F(NCCLBackendSmokeTest, AllGather_SingleProcess_OneEntry_OrSkip) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA not available";
    auto nccl = try_init_single_process();
    if (!nccl) GTEST_SKIP() << "NCCL not available";

    auto input = randn({3, 5}, DType::Float32, Device::cpu()).to(Device::cuda(0));
    std::vector<Tensor> output;
    output.push_back(zeros({3, 5}, DType::Float32, Device::cuda(0)));

    EXPECT_NO_THROW(nccl->all_gather(input, output));
    ASSERT_EQ(output.size(), 1u);
    Device::cuda(0).synchronize();
    auto in_cpu = input.to(Device::cpu()).contiguous();
    auto out_cpu = output[0].to(Device::cpu()).contiguous();
    const float* in_p = in_cpu.data<float>();
    const float* out_p = out_cpu.data<float>();
    for (int64_t i = 0; i < in_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(in_p[i], out_p[i]);
    }
    nccl->finalize();
}
