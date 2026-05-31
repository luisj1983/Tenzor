/**
 * @file test_optimizer_holes.cpp
 * @brief Regression coverage for S27 fixes:
 *
 *   1. LBFGS::load_state_dict now throws on missing required keys.
 *   2. LazyLinear::materialize honours a pre-materialisation `.to(device)`.
 *   3. HRM::compute_participation_ratio reflects activation rank structure
 *      (was a constant 1/numel before the fix).
 *   4. Adam (and Adam/Adagrad/AdamW) detect a producer-supplied sparse
 *      gradient and densify via the standard dense step instead of silently
 *      ignoring it.
 *
 * Each test is independent and pins one of the four audit items.
 */

#include <gtest/gtest.h>

#include "../backend_test_fixture.hpp"

#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/core/device.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/nn/layers/hrm.hpp>
#include <tenzor/nn/layers/lazy_linear.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/nn/optim/lbfgs.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/sparse/sparse_tensor.hpp>

#include <memory>
#include <stdexcept>
#include <unordered_map>

using namespace tenzor;

class OptimizerHolesTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ----------------------------------------------------------------------------
// Fix 1: LBFGS::load_state_dict throws on missing keys
// ----------------------------------------------------------------------------
TEST_P(OptimizerHolesTest, LBFGSLoadStateDictMissingKeyRaises) {
    auto param = std::make_shared<Variable>(
        zeros({4}, DType::Float32, device), /*requires_grad=*/true);
    std::vector<std::shared_ptr<Variable>> params = { param };
    optim::LBFGS lbfgs(params);

    // A "saved" state dict that's missing the required "tolerance_grad" key.
    // Pre-S27 this would silently fall back to the constructor default;
    // post-fix it must throw. State-dict scalars are bookkeeping values read
    // on the host; build them on CPU and move onto the active device.
    std::unordered_map<std::string, Tensor> bad_state;
    Device dev = device;
    auto scalar_i64 = [dev](int64_t v) {
        Tensor t({1}, DType::Int64, Device::cpu());
        t.data<int64_t>()[0] = v;
        return t.to(dev);
    };
    auto scalar_f64 = [dev](double v) {
        Tensor t({1}, DType::Float64, Device::cpu());
        t.data<double>()[0] = v;
        return t.to(dev);
    };
    bad_state["n_iter"]            = scalar_i64(0);
    bad_state["lr"]                = scalar_f64(1.0);
    bad_state["max_iter"]          = scalar_i64(20);
    bad_state["max_eval"]          = scalar_i64(25);
    // tolerance_grad deliberately omitted
    bad_state["tolerance_change"]  = scalar_f64(1e-9);
    bad_state["history_size"]      = scalar_i64(100);
    bad_state["has_prev_state"]    = scalar_i64(0);
    bad_state["prev_loss"]         = scalar_f64(0.0);

    EXPECT_THROW(lbfgs.load_state_dict(bad_state), std::invalid_argument);

    // Positive control: with the key present, load_state_dict succeeds.
    bad_state["tolerance_grad"] = scalar_f64(1e-7);
    EXPECT_NO_THROW(lbfgs.load_state_dict(bad_state));
}

// ----------------------------------------------------------------------------
// Fix 2: LazyLinear honours .to(Device) before forward
// ----------------------------------------------------------------------------
TEST_P(OptimizerHolesTest, LazyLinearHonoursToBeforeForward) {
    // The assertion is "materialize() honours requested_device_ over the
    // input's device". We request the active backend device explicitly, feed
    // a CPU input, and assert both the output and the materialised parameters
    // land on the requested (active) device.
    nn::LazyLinear layer(/*out_features=*/4, /*has_bias=*/true);

    // Explicit pre-materialisation .to() request onto the active device.
    layer.to(device);

    // First forward materialises. Use a CPU input so the explicit request and
    // the input's device may differ — the requested device must win.
    auto x = zeros({2, 8}, DType::Float32, Device::cpu());
    Variable input(x, /*requires_grad=*/false);
    auto out = layer.forward(input);

    EXPECT_EQ(out.tensor().device().type, device.type);

    auto params = layer.parameters();
    ASSERT_FALSE(params.empty());
    for (auto& p : params) {
        EXPECT_EQ(p->tensor().device().type, device.type)
            << "LazyLinear parameter not on requested device";
    }
}

// ----------------------------------------------------------------------------
// Fix 3: HRM participation ratio varies with activation rank structure
// ----------------------------------------------------------------------------
// HRM::compute_participation_ratio uses the diagonal-variance proxy
//   PR_diag = (Σ σ_d²)² / Σ σ_d⁴
// over per-feature variances σ_d². This measures how UNIFORMLY variance is
// spread across the D features (the effective dimensionality), NOT matrix
// rank: PR_diag == D when every feature has equal variance, and → 1 when one
// feature's variance dominates. The test therefore contrasts a uniform-
// variance activation (high PR, ≈ D) against a concentrated-variance one
// (low PR, ≈ 1).
TEST_P(OptimizerHolesTest, HRMParticipationRatioVaries) {
    nn::HRMConfig cfg;
    cfg.d_model        = 8;
    cfg.n_heads        = 2;
    cfg.d_feedforward  = 16;
    cfg.n_high_cycles  = 1;
    cfg.t_low_steps    = 1;
    cfg.dropout        = 0.0;
    cfg.max_seq_len    = 4;
    cfg.vocab_size     = 0;
    cfg.num_classes    = 0;
    cfg.use_act        = false;
    cfg.deep_supervision = false;
    nn::HRM hrm(cfg);
    hrm.to(device);

    const int64_t N = 16, D = 8;

    // Activation A: uniform per-feature variance — each of the D features
    // carries the same spread across the N rows. PR_diag must be ≈ D.
    // Build the pattern on CPU then move onto the active device.
    Tensor uniform = zeros({N, D}, DType::Float32, Device::cpu());
    {
        auto* d = uniform.data<float>();
        for (int64_t r = 0; r < N; ++r) {
            for (int64_t c = 0; c < D; ++c) {
                // Same alternating pattern per column => identical variance.
                d[r * D + c] = (r % 2 == 0) ? 1.0f : -1.0f;
            }
        }
    }
    Variable uniform_var(uniform.to(device), /*requires_grad=*/false);

    // Activation B: variance concentrated in a single feature — column 0 has
    // a large spread, the rest are (near) constant. PR_diag must be ≈ 1.
    Tensor concentrated = zeros({N, D}, DType::Float32, Device::cpu());
    {
        auto* d = concentrated.data<float>();
        for (int64_t r = 0; r < N; ++r) {
            d[r * D + 0] = static_cast<float>(r) * 100.0f;  // big spread
            for (int64_t c = 1; c < D; ++c) {
                d[r * D + c] = 0.001f * static_cast<float>(r % 2);  // ~constant
            }
        }
    }
    Variable concentrated_var(concentrated.to(device), /*requires_grad=*/false);

    double pr_uniform      = hrm.participation_ratio_for(uniform_var);
    double pr_concentrated = hrm.participation_ratio_for(concentrated_var);

    EXPECT_GE(pr_uniform,      0.0);
    EXPECT_GE(pr_concentrated, 0.0);

    // Uniform variance => effective dimensionality near the full D.
    EXPECT_NEAR(pr_uniform, static_cast<double>(D), 0.5)
        << "Uniform per-feature variance should give PR ≈ D=" << D
        << ", got " << pr_uniform;

    // Concentrated variance => effective dimensionality near 1.
    EXPECT_LT(pr_concentrated, 1.5)
        << "Variance concentrated in one feature should give PR ≈ 1, got "
        << pr_concentrated;

    // And the metric must actually distinguish the two (regression guard for
    // the pre-S27 implementation that was constant in the activations).
    EXPECT_GT(pr_uniform, pr_concentrated + 1.0)
        << "PR failed to separate uniform vs concentrated variance: "
        << "pr_uniform=" << pr_uniform << " pr_concentrated=" << pr_concentrated;
}

// ----------------------------------------------------------------------------
// Fix 4: Adam routes a sparse-grad-only parameter through the dense fallback
// ----------------------------------------------------------------------------
// Adam routes a sparse-grad-only parameter through the dense fallback.
TEST_P(OptimizerHolesTest, AdamSparseGradPathDispatches) {
    // Build a parameter with NO dense grad but a populated sparse_grad slot.
    // Pre-S27 the Adam step skipped this parameter entirely (since has_grad()
    // was false); post-fix it densifies the sparse grad, warns once, and
    // performs a standard Adam update.
    Tensor weight = zeros({4, 3}, DType::Float32, Device::cpu());
    // Seed weight with a known value so we can detect any update.
    {
        auto* d = weight.data<float>();
        for (int64_t i = 0; i < weight.numel(); ++i) d[i] = 1.0f;
    }
    auto param = std::make_shared<Variable>(weight.to(device), /*requires_grad=*/true);

    // Construct a 2-D COO grad: row 1 is active across all 3 columns
    // (embedding-style row-sparse update). COO indices have shape
    // (sparse_dim, nnz) = (2, 3): row coords {1,1,1}, col coords {0,1,2}.
    // Build the COO components on CPU then move them onto the active device.
    Tensor indices({2, 3}, DType::Int64, Device::cpu());
    {
        auto* ix = indices.data<int64_t>();
        ix[0] = 1; ix[1] = 1; ix[2] = 1;  // dim-0 (row) coordinates
        ix[3] = 0; ix[4] = 1; ix[5] = 2;  // dim-1 (col) coordinates
    }
    Tensor values = zeros({3}, DType::Float32, Device::cpu());
    {
        auto* d = values.data<float>();
        d[0] = 1.0f; d[1] = 1.0f; d[2] = 1.0f;
    }
    auto sparse_grad =
        SparseTensor::sparse_coo(indices.to(device), values.to(device), /*shape=*/{4, 3});
    param->set_sparse_grad(sparse_grad);
    // Ensure the dense grad slot is genuinely empty.
    ASSERT_FALSE(param->has_grad());
    ASSERT_TRUE(param->has_sparse_grad());

    std::vector<std::shared_ptr<Variable>> params = { param };
    optim::Adam adam(params, /*lr=*/0.1);

    // Capture the param value before the step.
    Tensor before = param->tensor().clone();

    EXPECT_NO_THROW(adam.step());

    // After the step, *some* entry should have changed (the densified
    // sparse-grad row triggers the standard Adam update). Read both on the
    // host via .cpu() before touching the raw buffer.
    Tensor before_cpu = before.cpu();
    Tensor after_cpu = param->tensor().cpu();
    const auto* a = before_cpu.data<float>();
    const auto* b = after_cpu.data<float>();
    double max_abs_delta = 0.0;
    for (int64_t i = 0; i < before_cpu.numel(); ++i) {
        max_abs_delta = std::max(max_abs_delta,
            static_cast<double>(std::abs(a[i] - b[i])));
    }
    EXPECT_GT(max_abs_delta, 0.0)
        << "Adam.step() did not update the parameter even though a sparse "
        << "grad was provided — the sparse fallback is missing.";

    // The sparse slot must have been cleared so the next step doesn't
    // re-densify the same accumulation.
    EXPECT_FALSE(param->has_sparse_grad())
        << "Sparse grad slot was not cleared after the dense fallback ran.";
}

INSTANTIATE_BACKEND_TESTS(OptimizerHolesTest);
