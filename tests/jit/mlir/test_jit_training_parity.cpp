// Training-through-JIT for the MLIR-configured backend.
//
// This is the regression test for the parity fix: `CompileConfig{.backend =
// "mlir"}` must TRAIN through the compiled graph exactly like the nvrtc
// backend, instead of silently short-circuiting a requires-grad call to plain
// eager `fn_`. It proves, on every available backend (CPU / CUDA / ROCm):
//
//   INFERENCE (no-grad): a compiled call with backend="mlir" returns the
//     correct result and — where the IREE target is actually runnable on this
//     host — went through the IREE runtime (cache_stats() misses increment).
//
//   TRAINING (grad): a compiled function that CLOSURE-CAPTURES parameters
//     (declared via with_parameters) and also takes a requires_grad input runs
//     forward + backward, and:
//       (a) captured-parameter AND input grads are non-null and match eager
//           autograd within tight tolerance,
//       (b) num_grad_forwards() > 0 — the load-bearing assertion that the
//           COMPILED differentiable graph produced the grads, not eager fn_,
//       (c) the loss tracks an eager reference and decreases over a few SGD
//           steps (with per-step parameter parity, i.e. no frozen-constant
//           staleness).
//
//   GRADCHECK (CPU): central finite differences validate the compiled
//     backward for a closure-captured parameter numerically.
//
// The design rationale for why this works backend-agnostically: a requires-grad
// call routes to CompiledFunction::grad_invoke, which replays the un-fused
// captured graph through the autograd-aware INTERPRETER
// (CompiledModule::forward_grad -> Graph::forward(grad_mode=true)) on the
// tensors' OWN device and never touches the IREE runtime. Only the no-grad
// INFERENCE forward of an mlir-configured function uses IREE (mlir_invoke).

#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/nn/functional.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "mlir_target_util.hpp"

namespace {

using namespace tenzor;
namespace mt = ::tenzor::testing::mlir;
namespace mj = ::tenzor::jit::mlir_jit;

// The IREE target a device maps to under target="auto".
auto iree_target_for(const Device& dev) -> std::string {
    switch (dev.type) {
        case Device::Type::CUDA: return "cuda";
        case Device::Type::ROCm: return "rocm";
        default:                 return "llvm-cpu";
    }
}

// Is the mlir/IREE path actually runnable for this device on this host? When
// false, inference degrades to eager (still correct) so the IREE-usage proof is
// skipped for that case.
auto iree_runnable_for(const Device& dev) -> bool {
    const std::string t = iree_target_for(dev);
    return mt::target_hw_present(t) && mt::iree_target_supported(t);
}

auto max_abs_diff(const Tensor& a, const Tensor& b) -> float {
    auto ac = a.to(DType::Float32).to(Device::cpu());
    auto bc = b.to(DType::Float32).to(Device::cpu());
    return ::tenzor::max(::tenzor::abs(ac - bc)).item<float>();
}

// Assert |a-b| <= atol + rtol*max|a| elementwise (checked on the max).
void expect_close(const Tensor& a, const Tensor& b, float rtol, float atol,
                  const std::string& what) {
    const float scale =
        ::tenzor::max(::tenzor::abs(a.to(DType::Float32).to(Device::cpu())))
            .item<float>();
    const float diff = max_abs_diff(a, b);
    EXPECT_LT(diff, atol + rtol * scale)
        << what << ": max abs diff " << diff << " exceeds "
        << (atol + rtol * scale);
}

// Loss = scaled sum-of-squares error of a 2-layer ReLU MLP against a target.
// Written from autograd ops so the SAME closure runs compiled AND eagerly.
// `p` = {W1[D0,H], b1[H], W2[H,D2], b2[D2]} (closure-captured parameters).
auto closure_mlp_loss(const Variable& x,
                      const std::vector<std::shared_ptr<Variable>>& p,
                      const Variable& target, double scale) -> Variable {
    Variable h = ::tenzor::matmul(x, *p[0]) + *p[1];
    h = nn::relu(h);
    Variable y = ::tenzor::matmul(h, *p[2]) + *p[3];
    Variable diff = y - target;
    return ::tenzor::sum(diff * diff) * scale;
}

auto make_param_leaves(const std::vector<Tensor>& seeds, const Device& dev)
    -> std::vector<std::shared_ptr<Variable>> {
    std::vector<std::shared_ptr<Variable>> ps;
    ps.reserve(seeds.size());
    for (const auto& s : seeds) {
        ps.push_back(std::make_shared<Variable>(s.clone().to(dev),
                                                /*requires_grad=*/true));
    }
    return ps;
}

// SGD step that REPLACES each parameter's storage (data pointer changes) — the
// stronger staleness test: the next compiled forward must read the new weights.
void sgd_step(std::vector<std::shared_ptr<Variable>>& ps, double lr) {
    for (auto& p : ps) {
        ASSERT_TRUE(p->grad().has_value());
        Tensor updated = p->tensor() - p->grad().value() * lr;
        p->tensor() = updated;
    }
}

void zero_grads(std::vector<std::shared_ptr<Variable>>& ps, Variable& x) {
    for (auto& p : ps) p->zero_grad();
    x.zero_grad();
}

// ── INFERENCE: no-grad compiled call matches eager; uses IREE where runnable ──
void run_inference(const Device& dev, const std::string& label) {
    mt::ensure_core_init();
    SCOPED_TRACE("inference on " + label);

    // Paramless, no-grad function of one input: relu(x + x). Add + Relu both
    // lower to StableHLO primitives.
    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [](const Variable& x) -> Variable { return nn::relu(x + x); });

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "auto";  // derive the IREE target from the input's device
    ::tenzor::jit::CompiledFunction compiled(fn, cfg);

    Tensor x_t = ::tenzor::randn({4, 8}, DType::Float32, Device::cpu()).to(dev);
    Variable x(x_t, /*requires_grad=*/false);

    const Variable eager = nn::relu(x + x);

    const bool iree_ok = iree_runnable_for(dev);
    if (iree_ok) mj::reset_cache_stats();

    Variable out;
    ASSERT_NO_THROW({ out = compiled(x); })
        << "no-grad mlir inference must not throw on " << label;
    dev.synchronize();

    EXPECT_LT(max_abs_diff(eager.tensor(), out.tensor()), 1e-5F)
        << "mlir inference result mismatch on " << label;

    if (iree_ok) {
        // The compiled inference forward must have gone through the IREE
        // runtime — a miss records a real trace+compile+run, proving it did
        // NOT silently fall back to eager on a runnable target.
        const auto stats = mj::cache_stats();
        EXPECT_GE(stats.misses, 1u)
            << "inference on a runnable IREE target (" << label
            << ") did not exercise the IREE compile path (misses="
            << stats.misses << ")";
    }
}

// ── TRAINING: closure-captured params + requires_grad input, through JIT ──────
void run_training(const Device& dev, const std::string& label) {
    mt::ensure_core_init();
    SCOPED_TRACE("training on " + label);

    const int64_t N = 4, D0 = 6, H = 5, D2 = 4;
    std::vector<Tensor> pseeds = {
        ::tenzor::randn({D0, H}, DType::Float32, Device::cpu()),  // W1
        ::tenzor::randn({H},     DType::Float32, Device::cpu()),  // b1
        ::tenzor::randn({H, D2}, DType::Float32, Device::cpu()),  // W2
        ::tenzor::randn({D2},    DType::Float32, Device::cpu()),  // b2
    };
    const Tensor x_seed = ::tenzor::randn({N, D0}, DType::Float32, Device::cpu());
    const Tensor t_seed = ::tenzor::randn({N, D2}, DType::Float32, Device::cpu());
    const double scale = 1.0 / static_cast<double>(N * D2);
    const double lr = 5e-2;
    const int steps = 5;

    // Two independent parameter sets from identical seeds: one trained eagerly,
    // one trained through the mlir-configured compiled function.
    auto eager_p = make_param_leaves(pseeds, dev);
    auto comp_p  = make_param_leaves(pseeds, dev);

    // The batch input ALSO requires grad, so we exercise input-grad parity too.
    Variable x_eager(x_seed.clone().to(dev), /*requires_grad=*/true);
    Variable x_comp(x_seed.clone().to(dev),  /*requires_grad=*/true);
    const Variable target(t_seed.clone().to(dev), /*requires_grad=*/false);

    // Closure captures comp_p; only x is a graph input. Declare comp_p as the
    // trainable parameters so they trace as live parameter leaves.
    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "auto";
    auto compiled = ::tenzor::jit::compile(
        ::tenzor::jit::CompiledFunction::FnTypeN(
            [&](std::span<const Variable> ins) {
                return closure_mlp_loss(ins[0], comp_p, target, scale);
            }),
        comp_p, cfg);

    float first_loss = 0.0F, last_loss = 0.0F;
    for (int step = 0; step < steps; ++step) {
        SCOPED_TRACE("step " + std::to_string(step));
        zero_grads(eager_p, x_eager);
        zero_grads(comp_p, x_comp);

        // Eager reference step.
        Variable le = closure_mlp_loss(x_eager, eager_p, target, scale);
        le.backward();

        // Compiled step (forward through the captured graph, then backward).
        Variable lc = compiled(x_comp);
        lc.backward();
        dev.synchronize();

        const float le_v =
            le.tensor().to(DType::Float32).to(Device::cpu()).item<float>();
        const float lc_v =
            lc.tensor().to(DType::Float32).to(Device::cpu()).item<float>();
        EXPECT_NEAR(le_v, lc_v, 1e-3F + 1e-3F * std::fabs(le_v));
        if (step == 0) first_loss = lc_v;
        last_loss = lc_v;

        // Captured-parameter grads must be non-null and match eager.
        for (size_t i = 0; i < comp_p.size(); ++i) {
            SCOPED_TRACE("param #" + std::to_string(i));
            ASSERT_TRUE(comp_p[i]->grad().has_value())
                << "captured parameter " << i << " got no gradient through JIT";
            expect_close(eager_p[i]->grad().value(),
                         comp_p[i]->grad().value(), 1e-4F, 1e-4F,
                         "param grad #" + std::to_string(i));
        }
        // Input grad must be non-null and match eager.
        ASSERT_TRUE(x_comp.grad().has_value())
            << "requires_grad input got no gradient through JIT";
        expect_close(x_eager.grad().value(), x_comp.grad().value(), 1e-4F, 1e-4F,
                     "input grad");

        sgd_step(eager_p, lr);
        sgd_step(comp_p, lr);

        // No frozen-constant staleness: post-step parameter values stay
        // identical, so the next compiled forward reads the UPDATED weights.
        for (size_t i = 0; i < comp_p.size(); ++i) {
            SCOPED_TRACE("post-step param #" + std::to_string(i));
            expect_close(eager_p[i]->tensor(), comp_p[i]->tensor(), 1e-4F, 1e-4F,
                         "post-step param #" + std::to_string(i));
        }
    }

    // Load-bearing: the COMPILED differentiable graph (not eager fn_) trained
    // the model. Without the parity fix this stays 0 (mlir short-circuited to
    // eager) and the test fails here.
    EXPECT_GT(compiled.num_grad_forwards(), 0u)
        << "compiled differentiable graph was not used on " << label
        << " (mlir short-circuited to eager)";
    EXPECT_LT(last_loss, first_loss)
        << "training-through-JIT did not reduce the loss on " << label;
}

}  // namespace

// ── INFERENCE parity per backend ─────────────────────────────────────────────
TEST(MlirTrainingParity, InferenceMatchesEager_Cpu) {
    run_inference(Device::cpu(), "cpu");
}
TEST(MlirTrainingParity, InferenceMatchesEager_Cuda) {
    if (!mt::backend_present("cuda")) {
        GTEST_SKIP() << "no CUDA backend";
    }
    run_inference(Device::cuda(0), "cuda");
}
TEST(MlirTrainingParity, InferenceMatchesEager_Rocm) {
    if (!mt::backend_present("rocm")) {
        GTEST_SKIP() << "no ROCm backend";
    }
    run_inference(Device::rocm(0), "rocm");
}
// JIT-F036: Vulkan was omitted from the MLIR training-parity coverage. Vulkan has
// an IREE target (vulkan-spirv), so both inference and backward-through-JIT are
// exercised here. (OneAPI has no IREE HAL — see JIT-F030/F038 — so its JIT path
// runs eagerly and is covered by the inference-vs-eager equivalence there.)
TEST(MlirTrainingParity, InferenceMatchesEager_Vulkan) {
    if (!mt::backend_present("vulkan")) {
        GTEST_SKIP() << "no Vulkan backend";
    }
    run_inference(Device::vulkan(0), "vulkan");
}

// ── TRAINING parity per backend ──────────────────────────────────────────────
TEST(MlirTrainingParity, TrainingThroughJIT_Cpu) {
    run_training(Device::cpu(), "cpu");
}
TEST(MlirTrainingParity, TrainingThroughJIT_Cuda) {
    if (!mt::backend_present("cuda")) {
        GTEST_SKIP() << "no CUDA backend";
    }
    run_training(Device::cuda(0), "cuda");
}
TEST(MlirTrainingParity, TrainingThroughJIT_Rocm) {
    if (!mt::backend_present("rocm")) {
        GTEST_SKIP() << "no ROCm backend";
    }
    run_training(Device::rocm(0), "rocm");
}
TEST(MlirTrainingParity, TrainingThroughJIT_Vulkan) {
    if (!mt::backend_present("vulkan")) {
        GTEST_SKIP() << "no Vulkan backend";
    }
    run_training(Device::vulkan(0), "vulkan");  // JIT-F036
}

// ── CPU finite-difference gradcheck on a CLOSURE-CAPTURED parameter ───────────
TEST(MlirTrainingParity, CapturedParamGradcheck_Cpu) {
    mt::ensure_core_init();
    const Device cpu = Device::cpu();
    const int64_t N = 3, D0 = 4, H = 4, D2 = 3;
    std::vector<Tensor> pseeds = {
        ::tenzor::randn({D0, H}, DType::Float32, cpu),  // W1 (checked param)
        ::tenzor::randn({H},     DType::Float32, cpu),  // b1
        ::tenzor::randn({H, D2}, DType::Float32, cpu),  // W2
        ::tenzor::randn({D2},    DType::Float32, cpu),  // b2
    };
    const Tensor x_seed = ::tenzor::randn({N, D0}, DType::Float32, cpu);
    const Tensor t_seed = ::tenzor::randn({N, D2}, DType::Float32, cpu);
    const double scale = 1.0 / static_cast<double>(N * D2);

    auto comp_p = make_param_leaves(pseeds, cpu);
    const Variable x(x_seed.clone(), /*requires_grad=*/false);
    const Variable target(t_seed.clone(), /*requires_grad=*/false);

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "auto";
    auto compiled = ::tenzor::jit::compile(
        ::tenzor::jit::CompiledFunction::FnTypeN(
            [&](std::span<const Variable> ins) {
                return closure_mlp_loss(ins[0], comp_p, target, scale);
            }),
        comp_p, cfg);

    Variable loss = compiled(x);
    loss.backward();
    ASSERT_GT(compiled.num_grad_forwards(), 0u)
        << "compiled graph not used for gradcheck";
    ASSERT_TRUE(comp_p[0]->grad().has_value());
    Tensor analytic_dW1 = comp_p[0]->grad().value().contiguous();

    // Central finite differences of the forward w.r.t. each element of W1,
    // evaluated eagerly with all params as non-grad leaves.
    auto eval_loss = [&](const Tensor& W1) -> float {
        std::vector<std::shared_ptr<Variable>> ps = {
            std::make_shared<Variable>(W1, false),
            std::make_shared<Variable>(pseeds[1], false),
            std::make_shared<Variable>(pseeds[2], false),
            std::make_shared<Variable>(pseeds[3], false),
        };
        Variable l = closure_mlp_loss(x, ps, target, scale);
        return l.tensor().item<float>();
    };

    const float eps = 2e-3F;
    Tensor W1 = pseeds[0].contiguous();
    const int64_t numel = W1.numel();
    const float* a = analytic_dW1.data<float>();
    float max_abs_err = 0.0F;
    for (int64_t idx = 0; idx < numel; ++idx) {
        Tensor wp = W1.clone();
        Tensor wm = W1.clone();
        wp.data<float>()[idx] += eps;
        wm.data<float>()[idx] -= eps;
        float fd = (eval_loss(wp) - eval_loss(wm)) / (2.0F * eps);
        max_abs_err = std::max(max_abs_err, std::fabs(a[idx] - fd));
    }
    EXPECT_LT(max_abs_err, 2e-2F)
        << "compiled backward for captured parameter W1 disagrees with finite "
           "differences (max abs err " << max_abs_err << ")";
}

int main(int argc, char** argv) {
    // Full IEEE-754 FP32 on GPU matmul (disable TF32) so compiled-vs-eager
    // backward parity holds at the tight tolerance used here.
    setenv("TENZOR_DISABLE_TF32", "1", /*overwrite=*/1);
    ::tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    ::tenzor::finalize();
    return result;
}
