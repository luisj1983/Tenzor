/**
 * @file test_s5_surgical_fixes.cpp
 * @brief Stream-5 P0 regression tests for three audit-confirmed surgical
 *        fixes:
 *
 *   1. DataParallel::gather no longer severs the autograd graph
 *      (was: wrapping cat-of-tensors with the leaf Variable ctor → all
 *       replica forwards invisible to backward; loss.backward() silently
 *       returned zero gradients on every input).
 *
 *   2. PANet::forward_impl rejects single-input invocation with a clear
 *      diagnostic instead of an out-of-bounds vector dereference inside
 *      forward_multi (which indexes features[0..2] for the P3/P4/P5
 *      pyramid levels — a 1-element vector triggered UB).
 *
 *   3. DeepLabV3PlusDecoder::forward_impl performs real work (upsample +
 *      classifier + upsample) instead of returning `input` unchanged. A
 *      user wiring the decoder into a Sequential previously got an
 *      identity stub instead of segmentation logits.
 *
 * Each test is backend-agnostic (CPU-only) wherever possible; the
 * DataParallel test skips cleanly when CUDA is unavailable because the
 * DataParallel constructor itself requires CUDA (validate_devices()
 * throws on non-CUDA builds).
 */

#include <gtest/gtest.h>

#include <memory>
#include <vector>
#include <stdexcept>

#include "tenzor/tenzor.hpp"  // tenzor::initialize()
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"

#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/parallel/data_parallel.hpp"
#include "tenzor/backend/loader.hpp"  // runtime backend probe
#include "tenzor/models/yolo.hpp"
#include "tenzor/models/deeplabv3plus.hpp"

#include "../grad_flow_helpers.hpp"

#ifdef TENZOR_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace {

// ---------------------------------------------------------------------------
// Global init — mirrors the singleton-style initialize() block used by
// other tests/nn/* fixtures.
// ---------------------------------------------------------------------------
class S5SurgicalFixesEnv : public ::testing::Environment {
public:
    void SetUp() override {
        ::tenzor::initialize();
    }
};

[[maybe_unused]] auto* s5_env =
    ::testing::AddGlobalTestEnvironment(new S5SurgicalFixesEnv);

inline bool cuda_available() {
    // Runtime probe via the loaded backend registry. A compile-time
    // `#ifdef TENZOR_USE_CUDA` here is wrong: this test TU is compiled
    // without the CUDA toolkit macro, so the guard would be false even on a
    // machine with a working GPU and a CUDA backend .so loaded by
    // tenzor::initialize(). The core library (which owns DataParallel and
    // links libcudart) is what actually decides whether DataParallel works,
    // so we ask the runtime, not the test compiler.
    try {
        auto* cuda = tenzor::backend_registry().get_backend("cuda");
        return cuda != nullptr && cuda->is_available() && cuda->device_count() > 0;
    } catch (...) {
        return false;
    }
}

}  // namespace

// ============================================================================
// Fix 1: DataParallel::gather autograd chain preservation
// ============================================================================
//
// Construction of `DataParallel` itself is CUDA-only (validate_devices()
// throws when TENZOR_USE_CUDA is off). On CPU-only builds we skip and rely
// on the orchestrator's CUDA-capable lane to exercise the gather path.
//
// The test builds a single-replica DataParallel (device {0} on CUDA),
// scatters → parallel_applies → gathers, and asserts that
//   (a) the gathered Variable's grad_fn is non-null when any input
//       carried requires_grad, OR (single-replica fast-path returns the
//       sole replica output verbatim), AND
//   (b) gradient actually reaches the input after backward().

TEST(S5DataParallel, GatherBackwardChainsThroughReplicas) {
    if (!cuda_available()) {
        GTEST_SKIP() << "DataParallel requires CUDA; CPU-only build cannot "
                        "instantiate it (validate_devices throws).";
    }

    // Small linear regression replica so the forward graph is non-trivial.
    auto module = std::make_shared<tenzor::nn::Linear>(4, 2, /*bias=*/true);

    tenzor::nn::DataParallel dp(module, /*device_ids=*/{0}, /*output_device=*/0,
                                /*dim=*/0);

    // Build a Variable input that requires grad on the output device (cuda:0).
    auto x_tensor = tenzor::ops::randn(
        {4, 4}, tenzor::DType::Float32, tenzor::Device::cuda(0));
    tenzor::Variable x(x_tensor, /*requires_grad=*/true);

    auto out = dp.forward(x);

    ASSERT_TRUE(out.requires_grad())
        << "DataParallel forward output should require grad";

    // grad_fn should be set: either via the gather's CatBackward (multi-replica)
    // or via the replica's own forward grad_fn (single-replica fast path).
    EXPECT_NE(out.grad_fn(), nullptr)
        << "Gather severed grad_fn — backward will dead-end at the gather "
           "node and upstream gradients silently zero. This is the exact "
           "bug Stream-5 Fix 1 closes.";

    auto loss = tenzor::sum(out);
    loss.backward();

    // Input gradient must actually populate (the symptom of severed grad_fn
    // is a zero-filled but technically present gradient tensor).
    EXPECT_GRAD_FLOWS(x);
}

// ============================================================================
// Fix 2: PANet::forward_impl rejects single-input
// ============================================================================
//
// Before: forward_impl built `std::vector<Variable> features = {input};`
// then handed it to forward_multi which dereferences features[1] and
// features[2] → OOB read on a 1-element vector. Undefined behaviour.
//
// After: TENZOR_CHECK(false, ...) throws tenzor::TenzorException (which
// derives from std::runtime_error) with a clear diagnostic naming
// forward_multi as the correct entry point.

TEST(S5PANet, ForwardImplSingleInputThrowsWithDiagnostic) {
    // PANet({64, 128, 256}) — channel counts for P3/P4/P5 typical of a
    // small backbone. We never actually invoke the multiscale path so
    // exact channel choice doesn't matter; we only need a valid module.
    tenzor::models::PANet panet({64, 128, 256});

    // Build a dummy single input — its shape doesn't reach forward_multi
    // because forward_impl now rejects single-input invocation up front.
    auto x_tensor = tenzor::ops::randn(
        {1, 256, 8, 8}, tenzor::DType::Float32, tenzor::Device::cpu());
    tenzor::Variable input(x_tensor, /*requires_grad=*/false);

    // Should throw — catches both TenzorException and the std::runtime_error
    // base it derives from. Anything that escapes (or that succeeds silently)
    // is a regression.
    EXPECT_THROW({ panet.forward_impl(input); }, std::runtime_error);
}

// ============================================================================
// Fix 3: DeepLabV3PlusDecoder::forward_impl does real work, not identity
// ============================================================================
//
// Before: `return input;` — silently identity. A user composing the
// decoder into a Sequential got unmodified features instead of logits.
//
// After: 4× upsample → classifier (1×1 conv to num_classes) → 4× upsample.
// The output shape (channels = num_classes, spatial = input.spatial × 16)
// differs from the input shape, so a simple shape comparison detects the
// fix without depending on numerical specifics.

TEST(S5DeepLabDecoder, ForwardImplIsNotIdentityAndChangesShape) {
    constexpr int64_t kNumClasses = 21;        // PASCAL VOC convention.
    constexpr int64_t kLowLevelChannels = 256; // matches encoder default.
    constexpr int64_t kAsppChannels = 256;     // standard ASPP output.

    tenzor::models::DeepLabV3PlusDecoder decoder(
        kNumClasses, kLowLevelChannels, kAsppChannels);

    // Typical post-encoder feature map: (N, aspp_channels, H/16, W/16).
    // With H=W=64 (small to keep the test fast) we get a 4×4 spatial map
    // — final output should be 64×64 after the two 4× upsamples.
    constexpr int64_t kBatch = 1;
    constexpr int64_t kInputH = 4;
    constexpr int64_t kInputW = 4;

    auto x_tensor = tenzor::ops::randn(
        {kBatch, kAsppChannels, kInputH, kInputW},
        tenzor::DType::Float32, tenzor::Device::cpu());
    tenzor::Variable input(x_tensor, /*requires_grad=*/false);

    auto output = decoder.forward_impl(input);

    // Shape must differ — channels go from aspp_channels (256) to
    // num_classes (21); spatial dims go up by 16×. If forward_impl is
    // still returning `input`, shape will be identical and this fails.
    const auto& in_shape = input.tensor().shape();
    const auto& out_shape = output.tensor().shape();

    ASSERT_EQ(out_shape.size(), 4u)
        << "Decoder output must be 4D (N, C, H, W)";

    EXPECT_EQ(out_shape[0], in_shape[0])
        << "Batch dim should be preserved";

    EXPECT_EQ(out_shape[1], kNumClasses)
        << "Channel dim should be num_classes after classifier; identity "
           "stub would leave channels = aspp_channels (" << kAsppChannels
        << ").";

    EXPECT_EQ(out_shape[2], kInputH * 16)
        << "Height must be upsampled 16× (4× then 4×); identity stub "
           "would leave it equal to input height (" << kInputH << ").";

    EXPECT_EQ(out_shape[3], kInputW * 16)
        << "Width must be upsampled 16×; identity stub would leave it "
           "equal to input width (" << kInputW << ").";
}
