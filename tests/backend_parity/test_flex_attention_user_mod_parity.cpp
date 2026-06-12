// test_flex_attention_user_mod_parity.cpp
//
// Wave C: FlexAttention with user-registered score_mod (ScoreModId >= 3) on
// Vulkan and ROCm. CUDA and OneAPI already supported this path; Vulkan and
// ROCm previously threw. The fix mirrors the CUDA pattern: registry lookup
// via tenzor::nn::find_registered_score_mod(id) + composed Q@K^T → user fn →
// softmax → @V (forward) and corresponding composed backward.
//
// Each test registers a simple custom score_mod (subtract 1.0 from every
// score) and verifies the GPU result matches CPU values.

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>
#include <tenzor/nn/layers/flex_attention.hpp>

using namespace tenzor;

namespace {

constexpr int64_t kUserModId = 42;

// Register a simple user score_mod (idempotent — guard against double
// registration across multiple TEST_F runs).
struct ScoreModRegistration {
    ScoreModRegistration() {
        static bool registered = false;
        if (!registered) {
            tenzor::nn::register_score_mod(kUserModId,
                [](const Tensor& score, int64_t /*b*/, int64_t /*h*/,
                   int64_t /*q_start*/, int64_t /*kv_start*/) -> Tensor {
                    auto shape = std::vector<int64_t>(score.shape().begin(), score.shape().end());
                    Tensor one = tenzor::full(shape, 1.0,
                                              score.dtype(), score.device());
                    return tenzor::sub(score, one);
                });
            registered = true;
        }
    }
};

class FlexAttentionUserModParity : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
        static ScoreModRegistration s_register;
        (void)s_register;
    }

    static bool has_device(Device::Type t, int index = 0) {
        try {
            Device d{t, index};
            auto tensor = zeros({1}, DType::Float32, d);
            (void)tensor;
            return true;
        } catch (...) {
            return false;
        }
    }
};

static auto random_qkv(std::vector<int64_t> shape, Device dev) -> Tensor {
    auto cpu = zeros(shape, DType::Float32, Device::cpu());
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        // Deterministic [−0.5, 0.5).
        uint32_t bits = static_cast<uint32_t>(i * 2654435761u);
        p[i] = (static_cast<float>(bits & 0xFFFFu) / 65536.0f) - 0.5f;
    }
    return (dev.type == Device::Type::CPU) ? cpu : cpu.to(dev);
}

// Reference forward: bmm(Q, K^T) * scale → user_fn → softmax → @V.
static auto ref_flex_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                              float scale, int64_t score_mod_id) -> Tensor {
    auto fn = tenzor::nn::find_registered_score_mod(score_mod_id);
    Tensor Kt = tenzor::transpose(K, -1, -2);
    Tensor scores = tenzor::bmm(Q, Kt);
    auto scores_shape = std::vector<int64_t>(scores.shape().begin(), scores.shape().end());
    Tensor scale_t = tenzor::full(scores_shape, static_cast<double>(scale),
                                   scores.dtype(), scores.device());
    scores = scores * scale_t;
    if (fn) scores = fn(scores, 0, 0, 0, 0);
    NewOpAttributes sm_attrs;
    sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
    std::vector<Tensor> sm_in = {scores};
    Tensor probs = tenzor::dispatch(OpId::Softmax, sm_in, sm_attrs)[0];
    return tenzor::bmm(probs, V);
}

static void check_close(const Tensor& a, const Tensor& b, float atol) {
    ASSERT_EQ(a.shape().size(), b.shape().size());
    for (size_t i = 0; i < a.shape().size(); ++i) {
        ASSERT_EQ(a.shape()[i], b.shape()[i]) << " dim " << i;
    }
    auto* ap = a.data<float>();
    auto* bp = b.data<float>();
    int64_t n = a.numel();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(ap[i], bp[i], atol) << " elem " << i;
    }
}

}  // namespace

TEST_F(FlexAttentionUserModParity, Vulkan_Forward_UserModMatchesCPU) {
    if (!has_device(Device::Type::Vulkan)) GTEST_SKIP() << "Vulkan not available";

    auto Q_cpu = random_qkv({2, 8, 4}, Device::cpu());  // (B*H, S_q, D)
    auto K_cpu = random_qkv({2, 8, 4}, Device::cpu());
    auto V_cpu = random_qkv({2, 8, 4}, Device::cpu());

    constexpr float scale = 0.5f;
    Tensor ref = ref_flex_forward(Q_cpu, K_cpu, V_cpu, scale, kUserModId);

    auto Q_gpu = Q_cpu.to(Device::vulkan(0));
    auto K_gpu = K_cpu.to(Device::vulkan(0));
    auto V_gpu = V_cpu.to(Device::vulkan(0));

    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(scale));
    attrs.set(AttrKey::ScoreModId, kUserModId);

    std::vector<Tensor> gpu_inputs = {Q_gpu, K_gpu, V_gpu};
    auto gpu_outs = dispatch(OpId::FlexAttention, gpu_inputs, attrs);
    ASSERT_GE(gpu_outs.size(), 1u);
    Tensor out_cpu = gpu_outs[0].to(Device::cpu());

    check_close(ref, out_cpu, 1e-4f);
}

TEST_F(FlexAttentionUserModParity, ROCm_Forward_UserModMatchesCPU) {
    if (!has_device(Device::Type::ROCm)) GTEST_SKIP() << "ROCm not available";

    auto Q_cpu = random_qkv({2, 8, 4}, Device::cpu());
    auto K_cpu = random_qkv({2, 8, 4}, Device::cpu());
    auto V_cpu = random_qkv({2, 8, 4}, Device::cpu());

    constexpr float scale = 0.5f;
    Tensor ref = ref_flex_forward(Q_cpu, K_cpu, V_cpu, scale, kUserModId);

    auto Q_gpu = Q_cpu.to(Device::rocm(0));
    auto K_gpu = K_cpu.to(Device::rocm(0));
    auto V_gpu = V_cpu.to(Device::rocm(0));

    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(scale));
    attrs.set(AttrKey::ScoreModId, kUserModId);

    std::vector<Tensor> gpu_inputs = {Q_gpu, K_gpu, V_gpu};
    auto gpu_outs = dispatch(OpId::FlexAttention, gpu_inputs, attrs);
    ASSERT_GE(gpu_outs.size(), 1u);
    Tensor out_cpu = gpu_outs[0].to(Device::cpu());

    check_close(ref, out_cpu, 1e-4f);
}

// Sliding-window backward: previously Vulkan/ROCm threw for score_mod_id=2.
// CUDA's composed backward is the reference (mathematically: replay forward
// to recover masked scores, then apply softmax-attention chain rule).
TEST_F(FlexAttentionUserModParity, Vulkan_Backward_SlidingWindowMatchesCPU) {
    if (!has_device(Device::Type::Vulkan)) GTEST_SKIP() << "Vulkan not available";

    auto Q_cpu = random_qkv({2, 8, 4}, Device::cpu());
    auto K_cpu = random_qkv({2, 8, 4}, Device::cpu());
    auto V_cpu = random_qkv({2, 8, 4}, Device::cpu());
    auto dO_cpu = random_qkv({2, 8, 4}, Device::cpu());
    auto O_cpu = random_qkv({2, 8, 4}, Device::cpu());  // placeholder; backward only uses scores/V

    constexpr float scale = 0.5f;
    constexpr int64_t window = 4;

    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(scale));
    attrs.set(AttrKey::ScoreModId, static_cast<int64_t>(2));
    attrs.set(AttrKey::WindowSize, window);

    std::vector<Tensor> cpu_inputs = {dO_cpu, Q_cpu, K_cpu, V_cpu, O_cpu};
    // Audit: previously wrapped in try{...}catch(...){GTEST_SKIP("CPU
    // FlexAttentionBackward unsupported for sliding window")}. CUDA's composed
    // backward is the documented reference and the sliding-window CPU/GPU
    // kernels were landed (file comment: "previously Vulkan/ROCm threw"). A
    // reference failure here is a real dispatch break, not a clean skip.
    std::vector<Tensor> cpu_grads =
        dispatch(OpId::FlexAttentionBackward, cpu_inputs, attrs);
    ASSERT_GE(cpu_grads.size(), 3u);  // dQ, dK, dV

    auto dO_gpu = dO_cpu.to(Device::vulkan(0));
    auto Q_gpu = Q_cpu.to(Device::vulkan(0));
    auto K_gpu = K_cpu.to(Device::vulkan(0));
    auto V_gpu = V_cpu.to(Device::vulkan(0));
    auto O_gpu = O_cpu.to(Device::vulkan(0));
    std::vector<Tensor> gpu_inputs = {dO_gpu, Q_gpu, K_gpu, V_gpu, O_gpu};
    auto gpu_grads = dispatch(OpId::FlexAttentionBackward, gpu_inputs, attrs);
    ASSERT_GE(gpu_grads.size(), 3u);

    for (int i = 0; i < 3; ++i) {
        Tensor cpu_t = cpu_grads[i];
        Tensor gpu_t = gpu_grads[i].to(Device::cpu());
        check_close(cpu_t, gpu_t, 5e-4f);
    }
}
