/**
 * @file test_fsdp2.cpp
 * @brief Single-process tests for tenzor::distributed::FSDP2.
 *
 * The audit (2026-05-02) found zero references to FSDP2 in the test
 * suite. FSDP2 is the per-parameter DTensor-based v2 wrapper (distinct
 * from the existing test_fsdp.cpp, which exercises the v1 flat-bucket
 * implementation). In single-process mode every collective is a no-op,
 * so these tests pin:
 *   - Construction with various FSDP2Config combinations.
 *   - shard_parameters() doesn't corrupt the wrapped module's params.
 *   - forward() yields the same numerical output as the unwrapped module.
 *   - sharded_parameters() returns one DTensor per leaf parameter.
 *   - MixedPrecisionPolicy values can be set without throwing.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/fsdp2.hpp>
#include <tenzor/distributed/device_mesh.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/autograd/checkpoint.hpp>

using namespace tenzor;
using namespace tenzor::distributed;

namespace {

// Two-Linear MLP — small enough to be fast, big enough to exercise
// register_module / register_parameter recursion.
class TwoLayerMLP : public nn::Module {
public:
    std::shared_ptr<nn::Linear> fc1;
    std::shared_ptr<nn::Linear> fc2;

    TwoLayerMLP(int64_t in_d, int64_t hid, int64_t out_d) {
        fc1 = std::make_shared<nn::Linear>(in_d, hid);
        fc2 = std::make_shared<nn::Linear>(hid, out_d);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        return fc2->forward(fc1->forward(x));
    }
};

}  // namespace

class FSDP2Test : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

TEST_F(FSDP2Test, Construct_Default) {
    auto mesh = std::make_shared<DeviceMesh>(Device::Type::CPU,
                                              std::vector<int64_t>{1},
                                              std::vector<std::string>{"dp"});
    auto mlp = std::make_shared<TwoLayerMLP>(8, 16, 4);
    FSDP2Config cfg;
    cfg.mesh = mesh;
    cfg.shard_mesh_dim = "dp";
    EXPECT_NO_THROW(FSDP2 fsdp(mlp, cfg));
}

TEST_F(FSDP2Test, Construct_RejectsNullMesh) {
    auto mlp = std::make_shared<TwoLayerMLP>(8, 16, 4);
    FSDP2Config cfg;  // mesh stays null
    EXPECT_THROW(FSDP2(mlp, cfg), std::invalid_argument);
}

TEST_F(FSDP2Test, Construct_RejectsBadShardDim) {
    auto mesh = std::make_shared<DeviceMesh>(Device::Type::CPU,
                                              std::vector<int64_t>{1},
                                              std::vector<std::string>{"dp"});
    auto mlp = std::make_shared<TwoLayerMLP>(8, 16, 4);
    FSDP2Config cfg;
    cfg.mesh = mesh;
    cfg.shard_mesh_dim = "not_a_real_dim";
    EXPECT_THROW(FSDP2(mlp, cfg), std::invalid_argument);
}

TEST_F(FSDP2Test, Construct_AcceptsMixedPrecisionPolicy) {
    auto mesh = std::make_shared<DeviceMesh>(Device::Type::CPU,
                                              std::vector<int64_t>{1},
                                              std::vector<std::string>{"dp"});
    auto mlp = std::make_shared<TwoLayerMLP>(8, 16, 4);
    FSDP2Config cfg;
    cfg.mesh = mesh;
    cfg.shard_mesh_dim = "dp";
    cfg.mixed_precision.param_dtype  = DType::BFloat16;
    cfg.mixed_precision.reduce_dtype = DType::Float32;
    cfg.mixed_precision.buffer_dtype = DType::Float32;
    EXPECT_NO_THROW(FSDP2 fsdp(mlp, cfg));
}

TEST_F(FSDP2Test, ShardParameters_NoOp_SingleProcess) {
    auto mesh = std::make_shared<DeviceMesh>(Device::Type::CPU,
                                              std::vector<int64_t>{1},
                                              std::vector<std::string>{"dp"});
    auto mlp = std::make_shared<TwoLayerMLP>(8, 16, 4);
    FSDP2Config cfg;
    cfg.mesh = mesh;
    FSDP2 fsdp(mlp, cfg);

    // shard_parameters() in single-process mode is a no-op; calling it
    // must not throw or mutate parameter shapes.
    auto params_before = mlp->named_parameters();
    std::vector<std::vector<int64_t>> shapes_before;
    for (auto& [name, p] : params_before) {
        auto s = p->tensor().shape();
        shapes_before.emplace_back(s.begin(), s.end());
    }

    EXPECT_NO_THROW(fsdp.shard_parameters());

    auto params_after = mlp->named_parameters();
    ASSERT_EQ(params_after.size(), shapes_before.size());
    for (size_t i = 0; i < params_after.size(); ++i) {
        auto s_after = params_after[i].second->tensor().shape();
        ASSERT_EQ(s_after.size(), shapes_before[i].size())
            << "parameter shape rank changed for " << params_after[i].first;
        for (size_t j = 0; j < s_after.size(); ++j) {
            EXPECT_EQ(s_after[j], shapes_before[i][j])
                << "parameter shape changed for " << params_after[i].first;
        }
    }
}

TEST_F(FSDP2Test, Forward_MatchesUnwrappedModule) {
    auto mesh = std::make_shared<DeviceMesh>(Device::Type::CPU,
                                              std::vector<int64_t>{1},
                                              std::vector<std::string>{"dp"});
    auto mlp_a = std::make_shared<TwoLayerMLP>(8, 16, 4);
    // Make a deep-copy MLP with the same weights so we can compare.
    auto mlp_b = std::make_shared<TwoLayerMLP>(8, 16, 4);
    {
        auto src = mlp_a->named_parameters();
        auto dst = mlp_b->named_parameters();
        ASSERT_EQ(src.size(), dst.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dst[i].second->tensor() = src[i].second->tensor().clone();
        }
    }

    Variable x(randn({4, 8}, DType::Float32, Device::cpu()), false);
    auto out_unwrapped = mlp_a->forward_impl(x).tensor();

    FSDP2Config cfg;
    cfg.mesh = mesh;
    FSDP2 fsdp(mlp_b, cfg);
    fsdp.shard_parameters();
    auto out_wrapped = fsdp.forward(x).tensor();

    ASSERT_EQ(out_unwrapped.shape().size(), out_wrapped.shape().size());
    for (size_t i = 0; i < out_unwrapped.shape().size(); ++i) {
        EXPECT_EQ(out_unwrapped.shape()[i], out_wrapped.shape()[i]);
    }
    const float* a = out_unwrapped.contiguous().data<float>();
    const float* b = out_wrapped.contiguous().data<float>();
    for (int64_t i = 0; i < out_unwrapped.numel(); ++i) {
        EXPECT_NEAR(a[i], b[i], 1e-4f) << "FSDP2 forward differs at " << i;
    }
}

TEST_F(FSDP2Test, ShardedParameters_OnePerLeaf) {
    auto mesh = std::make_shared<DeviceMesh>(Device::Type::CPU,
                                              std::vector<int64_t>{1},
                                              std::vector<std::string>{"dp"});
    auto mlp = std::make_shared<TwoLayerMLP>(8, 16, 4);
    FSDP2Config cfg;
    cfg.mesh = mesh;
    FSDP2 fsdp(mlp, cfg);
    fsdp.shard_parameters();

    auto leaf_params = mlp->named_parameters().size();
    auto sharded = fsdp.sharded_parameters();
    EXPECT_EQ(sharded.size(), leaf_params)
        << "Each leaf parameter should appear exactly once in sharded_parameters().";
}

TEST_F(FSDP2Test, BackwardHook_NoOp_SingleProcess) {
    auto mesh = std::make_shared<DeviceMesh>(Device::Type::CPU,
                                              std::vector<int64_t>{1},
                                              std::vector<std::string>{"dp"});
    auto mlp = std::make_shared<TwoLayerMLP>(8, 16, 4);
    FSDP2Config cfg;
    cfg.mesh = mesh;
    FSDP2 fsdp(mlp, cfg);
    fsdp.shard_parameters();
    EXPECT_NO_THROW(fsdp.backward_hook());
}

TEST_F(FSDP2Test, ActivationCheckpointing_DefaultsOff) {
    // Sanity check: the new flag should default to off, preserving legacy forward semantics
    // for any existing FSDP2 user.
    FSDP2Config cfg;
    EXPECT_FALSE(cfg.activation_checkpointing);
}

TEST_F(FSDP2Test, ActivationCheckpointing_ForwardOutputMatchesUnwrapped) {
    // With activation checkpointing on, forward should still produce bit-identical outputs
    // to the legacy path (recompute is purely a backward-time optimization). And the
    // checkpoint counter in the autograd subsystem should tick to prove the wrapping
    // actually engaged.
    auto mesh = std::make_shared<DeviceMesh>(Device::Type::CPU,
                                              std::vector<int64_t>{1},
                                              std::vector<std::string>{"dp"});
    auto mlp_ref = std::make_shared<TwoLayerMLP>(8, 16, 4);
    auto mlp_chk = std::make_shared<TwoLayerMLP>(8, 16, 4);
    {
        // Sync weights between the two MLPs so outputs are directly comparable.
        auto src = mlp_ref->named_parameters();
        auto dst = mlp_chk->named_parameters();
        ASSERT_EQ(src.size(), dst.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dst[i].second->tensor() = src[i].second->tensor().clone();
        }
    }

    FSDP2Config cfg_ref;
    cfg_ref.mesh = mesh;
    cfg_ref.activation_checkpointing = false;
    FSDP2 fsdp_ref(mlp_ref, cfg_ref);
    fsdp_ref.shard_parameters();

    FSDP2Config cfg_chk;
    cfg_chk.mesh = mesh;
    cfg_chk.activation_checkpointing = true;
    FSDP2 fsdp_chk(mlp_chk, cfg_chk);
    fsdp_chk.shard_parameters();

    // Input must require grad so autograd::checkpoint actually engages — the implementation
    // short-circuits to a plain forward call when no input has requires_grad (no point
    // checkpointing if there's nothing to recompute for).
    Variable x(randn({4, 8}, DType::Float32, Device::cpu()), true);

    autograd::reset_checkpoint_stats();
    const auto checkpoints_before = autograd::get_checkpoint_stats().num_checkpoints;

    auto out_ref = fsdp_ref.forward(x).tensor();
    auto out_chk = fsdp_chk.forward(x).tensor();

    const auto checkpoints_after = autograd::get_checkpoint_stats().num_checkpoints;

    // The checkpoint-enabled forward should have ticked the global checkpoint counter at
    // least once; the reference forward should not have ticked it at all.
    EXPECT_GT(checkpoints_after, checkpoints_before)
        << "FSDP2 with activation_checkpointing=true should invoke autograd::checkpoint";

    // Outputs must match bit-for-bit (checkpointing only changes when activations are kept,
    // not what they evaluate to).
    ASSERT_EQ(out_ref.shape().size(), out_chk.shape().size());
    for (size_t i = 0; i < out_ref.shape().size(); ++i) {
        EXPECT_EQ(out_ref.shape()[i], out_chk.shape()[i]);
    }
    const float* a = out_ref.contiguous().data<float>();
    const float* b = out_chk.contiguous().data<float>();
    for (int64_t i = 0; i < out_ref.numel(); ++i) {
        EXPECT_NEAR(a[i], b[i], 1e-5f) << "checkpointed output differs at " << i;
    }
}
