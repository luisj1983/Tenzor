/**
 * @file test_nccl_backend_smoke_rocm.cpp
 * @brief Single-process smoke for tenzor::distributed::NCCLBackend, ROCm/RCCL flavor.
 *
 * FINDING 4: nccl_backend.cpp/nccl_process_group.cpp are one shared,
 * source-compatible implementation for NVIDIA NCCL and AMD RCCL (see
 * nccl_backend.cpp's own "NCCL and RCCL are source-compatible" comment), but
 * NCCLBackend is a distinct compiled type per backend DSO: the symbols in
 * tenzor_backend_cuda.so are built with TENZOR_USE_CUDA (real cudaStream_t/
 * nccl.h), the symbols in tenzor_backend_rocm.so are built with
 * TENZOR_USE_ROCM (real hipStream_t/rccl.h). Linking BOTH DSOs into one test
 * binary would let the dynamic loader silently resolve tenzor::distributed::
 * NCCLBackend to whichever .so loads first, exercising only one backend's
 * codepath under both backends' names. This file is therefore a SEPARATE
 * executable from test_nccl_backend_smoke.cpp, linking only
 * tenzor_backend_rocm, so ROCm's RCCL-backed NCCLBackend gets equal,
 * unambiguous single-process smoke coverage — the gap this finding reported
 * (RCCL built and exercised nowhere).
 *
 * Uses a distinct rendezvous port from the CUDA variant (29502) so the two
 * executables never collide if a ctest run schedules them concurrently.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/nccl_backend.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <tenzor/backend/loader_fwd.hpp>
#include <tenzor/backend/backend.hpp>

using namespace tenzor;
using namespace tenzor::distributed;

class NCCLBackendSmokeRocmTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

namespace {

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

std::unique_ptr<NCCLBackend> try_init_single_process() {
    auto nccl = std::make_unique<NCCLBackend>();
    try {
        nccl->initialize(/*rank=*/0, /*world_size=*/1,
                         /*master_addr=*/"127.0.0.1", /*master_port=*/29503);
    } catch (const std::exception&) {
        return nullptr;
    }
    return nccl;
}

}  // namespace

TEST_F(NCCLBackendSmokeRocmTest, ConstructDoesNotThrow) {
    EXPECT_NO_THROW(NCCLBackend{});
}

TEST_F(NCCLBackendSmokeRocmTest, InitializeOrSkip) {
    if (!rocm_available()) GTEST_SKIP() << "ROCm not available";
    auto nccl = try_init_single_process();
    if (!nccl) GTEST_SKIP() << "RCCL not available in this build";
    EXPECT_EQ(nccl->backend_type(), tenzor::distributed::Backend::NCCL);
    EXPECT_TRUE(nccl->supports_async_stream());
    EXPECT_NO_THROW(nccl->finalize());
}

TEST_F(NCCLBackendSmokeRocmTest, AllReduce_SingleProcess_OnGPU_OrSkip) {
    if (!rocm_available()) GTEST_SKIP() << "ROCm not available";
    auto nccl = try_init_single_process();
    if (!nccl) GTEST_SKIP() << "RCCL not available";

    auto t = randn({4, 4}, DType::Float32, Device::cpu()).to(Device::rocm(0));
    auto t_before = t.clone();
    EXPECT_NO_THROW(nccl->all_reduce(t, ReduceOp::SUM));
    Device::rocm(0).synchronize();
    auto a_cpu = t_before.to(Device::cpu()).contiguous();
    auto b_cpu = t.to(Device::cpu()).contiguous();
    const float* a = a_cpu.data<float>();
    const float* b = b_cpu.data<float>();
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]) << "all_reduce mutated tensor in single-rank mode";
    }
    nccl->finalize();
}

TEST_F(NCCLBackendSmokeRocmTest, Broadcast_SingleProcess_IsIdentity_OrSkip) {
    if (!rocm_available()) GTEST_SKIP() << "ROCm not available";
    auto nccl = try_init_single_process();
    if (!nccl) GTEST_SKIP() << "RCCL not available";

    auto t = randn({16}, DType::Float32, Device::cpu()).to(Device::rocm(0));
    auto t_before = t.clone();
    EXPECT_NO_THROW(nccl->broadcast(t, /*src_rank=*/0));
    Device::rocm(0).synchronize();
    auto a_cpu = t_before.to(Device::cpu()).contiguous();
    auto b_cpu = t.to(Device::cpu()).contiguous();
    const float* a = a_cpu.data<float>();
    const float* b = b_cpu.data<float>();
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]) << "broadcast mutated tensor in single-rank mode";
    }
    nccl->finalize();
}

TEST_F(NCCLBackendSmokeRocmTest, AllGather_SingleProcess_OneEntry_OrSkip) {
    if (!rocm_available()) GTEST_SKIP() << "ROCm not available";
    auto nccl = try_init_single_process();
    if (!nccl) GTEST_SKIP() << "RCCL not available";

    auto input = randn({3, 5}, DType::Float32, Device::cpu()).to(Device::rocm(0));
    std::vector<Tensor> output;
    output.push_back(zeros({3, 5}, DType::Float32, Device::rocm(0)));

    EXPECT_NO_THROW(nccl->all_gather(input, output));
    ASSERT_EQ(output.size(), 1u);
    Device::rocm(0).synchronize();
    auto in_cpu = input.to(Device::cpu()).contiguous();
    auto out_cpu = output[0].to(Device::cpu()).contiguous();
    const float* in_p = in_cpu.data<float>();
    const float* out_p = out_cpu.data<float>();
    for (int64_t i = 0; i < in_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(in_p[i], out_p[i]);
    }
    nccl->finalize();
}

// FINDING 4: exercises the actual backend-registry factory hook
// (ROCmBackend::create_comm_backend()) directly, rather than through the
// generic ProcessGroup::create_process_group(Backend::NCCL, ...) factory —
// on a host with BOTH CUDA and ROCm loaded (like this dev machine),
// create_process_group()'s Backend::NCCL case tries
// backend_registry().get_backend(Device::Type::CUDA) FIRST and only falls
// back to ROCm when no CUDA backend is registered (see distributed.cpp's own
// comment), so going through that generic entry point here would silently
// re-test CUDA's create_comm_backend() instead of ROCm's. Calling the ROCm
// backend's override directly is the only way to isolate and verify the
// specific fix this file is about: create_comm_backend() previously always
// returned nullptr for ROCm (backend.hpp's unconditional default), even on a
// build with RCCL linked and no CUDA present at all.
TEST_F(NCCLBackendSmokeRocmTest, CreateCommBackend_ViaRegistry_OrSkip) {
    if (!rocm_available()) GTEST_SKIP() << "ROCm not available";
    tenzor::Backend* rocm_backend =
        tenzor::backend_registry().get_backend(tenzor::Device::Type::ROCm);
    ASSERT_NE(rocm_backend, nullptr) << "ROCm backend not registered";

    void* raw = rocm_backend->create_comm_backend();
    if (raw == nullptr) {
        GTEST_SKIP() << "ROCm backend built without RCCL "
                         "(create_comm_backend() returned nullptr)";
    }
    std::unique_ptr<CommunicationBackend> comm_backend(
        static_cast<CommunicationBackend*>(raw));
    ASSERT_NO_THROW(comm_backend->initialize(
        /*rank=*/0, /*world_size=*/1, "127.0.0.1", /*master_port=*/29505));
    auto pg = std::make_shared<ProcessGroup>(std::move(comm_backend), 0, 1);

    auto t = randn({4, 4}, DType::Float32, Device::cpu()).to(Device::rocm(0));
    auto t_before = t.clone();
    EXPECT_NO_THROW(pg->all_reduce(t, ReduceOp::SUM));
    Device::rocm(0).synchronize();
    auto a_cpu = t_before.to(Device::cpu()).contiguous();
    auto b_cpu = t.to(Device::cpu()).contiguous();
    const float* a = a_cpu.data<float>();
    const float* b = b_cpu.data<float>();
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i])
            << "registry-created RCCL process group's all_reduce mutated "
               "tensor in single-rank mode";
    }
}

TEST_F(NCCLBackendSmokeRocmTest, Barrier_OrSkip) {
    if (!rocm_available()) GTEST_SKIP() << "ROCm not available";
    auto nccl = try_init_single_process();
    if (!nccl) GTEST_SKIP() << "RCCL not available";
    EXPECT_NO_THROW(nccl->barrier());
    nccl->finalize();
}
