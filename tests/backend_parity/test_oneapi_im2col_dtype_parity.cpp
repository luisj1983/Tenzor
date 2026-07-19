// test_oneapi_im2col_dtype_parity.cpp
//
// Wave F2: OneAPI im2col now natively supports F16/BF16 (no tensor-wide
// widen-narrow). The kernel is a pure data shuffle so SYCL templates
// directly on `sycl::half` (F16) and `uint16_t` (BF16 storage). col2im
// (which needs atomic_add) remains a documented follow-up.

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>
#include "golden_util.hpp"

using namespace tenzor;
using namespace tenzor::testing;

namespace {

class OneAPIIm2ColDtypeParity : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    static bool has_oneapi() {
        try {
            auto t = zeros({1}, DType::Float32, Device::oneapi(0));
            (void)t; return true;
        } catch (...) { return false; }
    }
};

static auto seq_f32(std::vector<int64_t> shape) -> Tensor {
    auto t = zeros(shape, DType::Float32, Device::cpu());
    auto* p = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        p[i] = static_cast<float>((i & 0x3F) - 32) / 32.0f;  // in [-1, 1]
    }
    return t;
}

static void check_close(const Tensor& a, const Tensor& b, float atol) {
    ASSERT_EQ(a.shape().size(), b.shape().size());
    for (size_t i = 0; i < a.shape().size(); ++i) {
        ASSERT_EQ(a.shape()[i], b.shape()[i]) << " dim " << i;
    }
    auto* ap = a.data<float>();
    auto* bp = b.data<float>();
    for (int64_t i = 0; i < a.numel(); ++i) {
        EXPECT_NEAR(ap[i], bp[i], atol) << " elem " << i;
    }
}

}  // namespace

// FINDING 17: previously an unconditional GTEST_SKIP() on a host without
// OneAPI, with zero golden::* integration. This test's shape is unusual
// (F32 CPU reference cast down for comparison against a *native* low-
// precision OneAPI result, not a same-dtype-both-sides comparison), so it
// can't drop directly into test_operation_parity_single's contract — wire
// golden::maybe_record/maybe_load directly instead, storing the OneAPI
// native-dtype result (cast to F32, matching every other committed golden's
// CPU-F32 convention) as the reference for a future CPU-only replay.
TEST_F(OneAPIIm2ColDtypeParity, Im2Col_F16_NativeMatchesCPU) {
    auto in_f32 = seq_f32({1, 2, 6, 6});  // (N, C, H, W)
    OpAttributes attrs;
    attrs.set(AttrKey::KernelSize, static_cast<int64_t>(3));
    attrs.set(AttrKey::Stride, static_cast<int64_t>(1));
    attrs.set(AttrKey::Padding, static_cast<int64_t>(1));
    attrs.set(AttrKey::Dilation, static_cast<int64_t>(1));

    // F32 reference on CPU.
    std::vector<Tensor> cpu_inputs = {in_f32};
    Tensor cpu_out = dispatch(OpId::Unfold, cpu_inputs, attrs)[0];
    Tensor cpu_out_f16_back = cpu_out.to(DType::Float16).to(DType::Float32);

    const std::string test_name = "Im2Col_F16_Native";

    if (!has_oneapi()) {
        if (auto golden_result = golden::maybe_load(test_name, cpu_inputs)) {
            golden::note_comparison();
            check_close(cpu_out_f16_back, *golden_result, 5e-3f);
            return;
        }
        if (golden::require_multi_backend()) {
            FAIL() << test_name << ": no recorded golden and OneAPI is unavailable.\n"
                   << "  Record one from a multi-backend host with TENZOR_RECORD_GOLDENS=1.";
        }
        GTEST_SKIP() << test_name << ": OneAPI not available and no recorded golden — "
                        "nothing to compare against.";
        return;
    }

    // Native F16 on OneAPI.
    auto in_f16_oneapi = in_f32.to(DType::Float16).to(Device::oneapi(0));
    std::vector<Tensor> oneapi_inputs = {in_f16_oneapi};
    Tensor oneapi_out_f16 = dispatch(OpId::Unfold, oneapi_inputs, attrs)[0];

    // Verify the result was returned in Float16 (no tensor-wide widen).
    ASSERT_EQ(oneapi_out_f16.dtype(), DType::Float16);

    // Compare values (F32 reference cast down to F16 vs native F16 output).
    Tensor oneapi_out_f32 = oneapi_out_f16.to(Device::cpu()).to(DType::Float32);

    if (golden::recording_enabled()) {
        golden::maybe_record(test_name, cpu_inputs, oneapi_out_f32);
    }

    check_close(cpu_out_f16_back, oneapi_out_f32, 5e-3f);
}

TEST_F(OneAPIIm2ColDtypeParity, Im2Col_BF16_NativeMatchesCPU) {
    auto in_f32 = seq_f32({1, 2, 6, 6});
    OpAttributes attrs;
    attrs.set(AttrKey::KernelSize, static_cast<int64_t>(3));
    attrs.set(AttrKey::Stride, static_cast<int64_t>(1));
    attrs.set(AttrKey::Padding, static_cast<int64_t>(1));
    attrs.set(AttrKey::Dilation, static_cast<int64_t>(1));

    std::vector<Tensor> cpu_inputs = {in_f32};
    Tensor cpu_out = dispatch(OpId::Unfold, cpu_inputs, attrs)[0];
    Tensor cpu_out_bf16_back = cpu_out.to(DType::BFloat16).to(DType::Float32);

    const std::string test_name = "Im2Col_BF16_Native";

    if (!has_oneapi()) {
        if (auto golden_result = golden::maybe_load(test_name, cpu_inputs)) {
            golden::note_comparison();
            check_close(cpu_out_bf16_back, *golden_result, 1e-2f);
            return;
        }
        if (golden::require_multi_backend()) {
            FAIL() << test_name << ": no recorded golden and OneAPI is unavailable.\n"
                   << "  Record one from a multi-backend host with TENZOR_RECORD_GOLDENS=1.";
        }
        GTEST_SKIP() << test_name << ": OneAPI not available and no recorded golden — "
                        "nothing to compare against.";
        return;
    }

    auto in_bf16_oneapi = in_f32.to(DType::BFloat16).to(Device::oneapi(0));
    std::vector<Tensor> oneapi_inputs = {in_bf16_oneapi};
    Tensor oneapi_out_bf16 = dispatch(OpId::Unfold, oneapi_inputs, attrs)[0];

    ASSERT_EQ(oneapi_out_bf16.dtype(), DType::BFloat16);

    Tensor oneapi_out_f32 = oneapi_out_bf16.to(Device::cpu()).to(DType::Float32);

    if (golden::recording_enabled()) {
        golden::maybe_record(test_name, cpu_inputs, oneapi_out_f32);
    }

    check_close(cpu_out_bf16_back, oneapi_out_f32, 1e-2f);
}
