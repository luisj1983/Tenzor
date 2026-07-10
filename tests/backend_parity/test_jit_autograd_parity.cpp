/**
 * @file test_jit_autograd_parity.cpp
 * @brief Training-through-JIT: differentiable execution of the COMPILED graph.
 *
 * This test exercises the real feature — backward through a jit::compile'd
 * graph. A small model is wrapped in jit::compile(), invoked on requires_grad
 * inputs, and .backward() is called on the compiled output. We assert:
 *   1. the input/parameter gradients from the COMPILED backward match eager
 *      autograd within tight tolerance, on every available backend
 *      (CPU / CUDA / ROCm),
 *   2. the compiled graph (not the eager fn_) actually produced the grads
 *      (CompiledFunction::num_grad_forwards() > 0), and
 *   3. a finite-difference gradcheck on CPU validates the compiled backward
 *      against numerical derivatives of the forward.
 *
 * The parameters are passed as explicit graph INPUTS (functional style) so the
 * differentiable replay produces gradients for them exactly as eager does — no
 * frozen-constant severing. The grad variant is compiled WITHOUT fusion, so the
 * replay never hits a backward-less fused GPU kernel (Graph::execute_node throws
 * on a fusion node in grad mode; num_grad_forwards proves the un-fused
 * differentiable graph ran instead).
 */

#include <gtest/gtest.h>
#include <cmath>
#include <span>
#include <vector>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/compile.hpp>
#include <tenzor/jit/control_flow.hpp>
#include <tenzor/nn/functional.hpp>
#include "../backend_test_fixture.hpp"
#include "../grad_flow_helpers.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

namespace {

// ---------------------------------------------------------------------------
// Models. Each takes a span of Variables (x + parameters) and returns a scalar
// loss (sum of the output). Written entirely from autograd ops so the SAME
// lambda can be jit::compile'd AND evaluated eagerly for finite differences.
// ---------------------------------------------------------------------------

// inputs: {x[N,D0], W1[D0,H], b1[H], W2[H,D2], b2[D2]}
auto linear_chain_model(std::span<const Variable> in) -> Variable {
    Variable h = tenzor::matmul(in[0], in[1]) + in[2];
    h = nn::relu(h);
    Variable y = tenzor::matmul(h, in[3]) + in[4];
    return tenzor::sum(y);
}

// inputs: {x[N,D], W1[D,D], b1[D], gamma[D], beta[D], W2[D,D2], b2[D2]}
auto layernorm_mlp_model(std::span<const Variable> in) -> Variable {
    Variable h = tenzor::matmul(in[0], in[1]) + in[2];
    // Exercises the OpType::LayerNorm differentiable-replay branch. gamma/beta
    // are graph inputs, so their grads flow through the compiled replay.
    h = nn::functional::layer_norm(h, {static_cast<int64_t>(in[0].tensor().shape().back())},
                                   in[3], in[4]);
    h = nn::relu(h);
    Variable y = tenzor::matmul(h, in[5]) + in[6];
    return tenzor::sum(y);
}

// Build a fresh, independent set of leaf Variables (requires_grad=true) from a
// list of CPU seed tensors placed on `dev`. Independent = no shared grad state
// between the eager and compiled runs.
auto make_leaves(const std::vector<Tensor>& seeds, const Device& dev,
                 bool requires_grad) -> std::vector<Variable> {
    std::vector<Variable> vars;
    vars.reserve(seeds.size());
    for (const auto& s : seeds) {
        vars.emplace_back(s.clone().to(dev), requires_grad);
    }
    return vars;
}

// Evaluate a model eagerly (no autograd graph) and return the scalar loss as a
// host float — used for finite differences.
template <class Model>
auto eval_scalar(Model&& model, const std::vector<Tensor>& tensors) -> float {
    std::vector<Variable> vars;
    vars.reserve(tensors.size());
    for (const auto& t : tensors) vars.emplace_back(t, /*requires_grad=*/false);
    Variable loss = model(std::span<const Variable>(vars.data(), vars.size()));
    return loss.tensor().to(DType::Float32).to(Device::cpu()).item<float>();
}

// Run a model through jit::compile on `dev`, backprop, and return the per-input
// gradients (moved to CPU) plus the compiled handle (for num_grad_forwards).
template <class Model>
struct CompiledRun {
    std::vector<Tensor> grads;
    size_t grad_forwards{0};
    float loss{0.0f};
};

template <class Model>
auto run_compiled(Model&& model, const std::vector<Tensor>& seeds,
                  const Device& dev) -> CompiledRun<Model> {
    auto vars = make_leaves(seeds, dev, /*requires_grad=*/true);
    jit::CompileConfig cfg;  // nvrtc backend, fusion on (disabled internally for grad)
    auto compiled = jit::compile(
        jit::CompiledFunction::FnTypeN(
            [&model](std::span<const Variable> ins) { return model(ins); }),
        cfg);

    Variable loss = compiled(std::span<const Variable>(vars.data(), vars.size()));
    loss.backward();
    dev.synchronize();

    CompiledRun<Model> out;
    out.loss = loss.tensor().to(DType::Float32).to(Device::cpu()).item<float>();
    out.grad_forwards = compiled.num_grad_forwards();
    for (auto& v : vars) {
        out.grads.push_back(v.grad().value().to(Device::cpu()));
    }
    return out;
}

// Eager reference gradients on the same device.
template <class Model>
auto run_eager(Model&& model, const std::vector<Tensor>& seeds,
               const Device& dev) -> std::vector<Tensor> {
    auto vars = make_leaves(seeds, dev, /*requires_grad=*/true);
    Variable loss = model(std::span<const Variable>(vars.data(), vars.size()));
    loss.backward();
    dev.synchronize();
    std::vector<Tensor> grads;
    for (auto& v : vars) grads.push_back(v.grad().value().to(Device::cpu()));
    return grads;
}

// ---------------------------------------------------------------------------
// Closure-capture (real training pattern): the compiled function CLOSES OVER a
// set of module-style parameter Variables and calls a forward that reads them,
// rather than receiving them as explicit graph inputs. The parameters are
// declared to the compiler via the new parameter-aware API so they are traced
// as live parameter leaves (correct grads + no staleness), NOT frozen
// constants.
// ---------------------------------------------------------------------------

// Loss = mean-squared error of a 2-layer ReLU MLP against a fixed target.
// Written from autograd ops so the SAME closure runs compiled AND eagerly.
// `p` = {W1[D0,H], b1[H], W2[H,D2], b2[D2]} (captured parameters).
auto closure_mlp_loss(const Variable& x,
                      const std::vector<std::shared_ptr<Variable>>& p,
                      const Variable& target, double scale) -> Variable {
    Variable h = tenzor::matmul(x, *p[0]) + *p[1];
    h = nn::relu(h);
    Variable y = tenzor::matmul(h, *p[2]) + *p[3];
    Variable diff = y - target;
    return tenzor::sum(diff * diff) * scale;
}

// Build a fresh set of INDEPENDENT parameter leaves (requires_grad=true) as
// shared_ptr<Variable> — the shape nn::Module::parameters() returns.
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

// One SGD step that REPLACES each parameter's tensor (new storage, not in-place)
// — this is the stronger staleness test: a compiled forward must read the new
// weights on the next call even though the parameter's data pointer changed.
auto sgd_step(std::vector<std::shared_ptr<Variable>>& ps, double lr) -> void {
    for (auto& p : ps) {
        ASSERT_TRUE(p->grad().has_value());
        Tensor updated = p->tensor() - p->grad().value() * lr;
        p->tensor() = updated;  // replace storage (data_ptr changes)
    }
}

auto zero_grads(std::vector<std::shared_ptr<Variable>>& ps) -> void {
    for (auto& p : ps) p->zero_grad();
}

// ---------------------------------------------------------------------------
// Control-flow closure captures: a compiled function closes over a parameter
// and USES it INSIDE a jit::cond branch / jit::while_loop body. The captured
// parameter must become a live parameter leaf in the SUBGRAPH (not a frozen
// constant), so backward through the compiled control-flow graph reaches
// param->grad() and matches eager autograd.
// ---------------------------------------------------------------------------

// If-branch model. `pred` is a deterministic (constant) condition so a fixed
// branch runs. then-branch USES the captured params: relu(matmul(x,*W) + *b).
// else-branch is identity. *W is square [D,D] and *b is [D] so both branches
// yield [N,D] (shape-consistent). loss = sum(y).
// `p` = {W[D,D], b[D]} (captured parameters used inside the then-branch).
auto if_branch_loss(const Variable& x,
                    const std::vector<std::shared_ptr<Variable>>& p,
                    const Tensor& pred) -> Variable {
    Variable y = jit::cond(
        pred,
        [&p](const Variable& in) -> Variable {
            return nn::relu(tenzor::matmul(in, *p[0]) + *p[1]);
        },
        [](const Variable& in) -> Variable { return in; },
        x);
    return tenzor::sum(y);
}

// While-loop model. Data-dependent counter loop (mirrors the proven H5 replay
// pattern): carried = {x, counter, limit}, cond = (limit - counter) > 0, body
// transforms x with the CAPTURED weight — x' = relu(matmul(x, *W)) — and
// increments the counter. Runs exactly `limit` iterations, so *W is used
// `limit` times and its gradient is the sum of per-iteration contributions.
// loss = sum(x_final).  `p` = {W[D,D]} (captured parameter used in the body).
auto loop_matmul_loss(const Variable& x,
                      const std::vector<std::shared_ptr<Variable>>& p,
                      const Variable& counter0, const Variable& limit,
                      const Device& dev) -> Variable {
    auto cond_fn = [](const std::vector<Variable>& s) -> Tensor {
        // Real op (Sub) so the condition is a PRODUCED value surfaced as the
        // body subgraph's first output — a bare constant would not be.
        return s[2].tensor() - s[1].tensor();  // limit - counter, positive while running
    };
    auto body_fn = [&p, dev](const std::vector<Variable>& s) -> std::vector<Variable> {
        auto one = Variable(tenzor::full({1}, 1.0f, DType::Float32, dev), false);
        Variable x_new = nn::relu(tenzor::matmul(s[0], *p[0]));
        return {x_new, s[1] + one, s[2]};
    };
    auto out = jit::while_loop(1000, cond_fn, body_fn, {x, counter0, limit});
    return tenzor::sum(out[0]);
}

}  // namespace

class JITAutogradParity : public BackendTest {};

// ---------------------------------------------------------------------------
// Linear chain: matmul -> +bias -> relu -> matmul -> +bias -> sum
// ---------------------------------------------------------------------------
TEST_P(JITAutogradParity, LinearChain_TrainingThroughJIT) {
    const Device dev = device;  // parametrized backend (fixture resolves/skips)
    SCOPED_TRACE(std::string("LinearChain on ") + backend_name(dev));

    const int64_t N = 4, D0 = 8, H = 6, D2 = 5;
    std::vector<Tensor> seeds = {
        randn({N, D0}, DType::Float32, Device::cpu()),
        randn({D0, H}, DType::Float32, Device::cpu()),
        randn({H},     DType::Float32, Device::cpu()),
        randn({H, D2}, DType::Float32, Device::cpu()),
        randn({D2},    DType::Float32, Device::cpu()),
    };

    auto eager_grads = run_eager(linear_chain_model, seeds, dev);
    auto run = run_compiled(linear_chain_model, seeds, dev);

    // The compiled graph — not fn_ — must have produced the gradients.
    EXPECT_GT(run.grad_forwards, 0u)
        << "compiled differentiable graph was not used on " << backend_name(dev);

    ASSERT_EQ(run.grads.size(), eager_grads.size());
    for (size_t i = 0; i < eager_grads.size(); ++i) {
        SCOPED_TRACE(std::string("input #") + std::to_string(i));
        // Same backend, same kernels, same op order: compiled backward must
        // match eager backward to near machine precision.
        EXPECT_TENSORS_CLOSE(eager_grads[i], run.grads[i], 1e-4f, 1e-5f);
    }
}

// ---------------------------------------------------------------------------
// LayerNorm MLP: matmul -> +bias -> layernorm(gamma,beta) -> relu -> matmul ->
//                +bias -> sum   (exercises the OpType::LayerNorm grad branch)
// ---------------------------------------------------------------------------
TEST_P(JITAutogradParity, LayerNormMLP_TrainingThroughJIT) {
    const Device dev = device;  // parametrized backend (fixture resolves/skips)
    SCOPED_TRACE(std::string("LayerNormMLP on ") + backend_name(dev));

    const int64_t N = 4, D = 8, D2 = 5;
    std::vector<Tensor> seeds = {
        randn({N, D},  DType::Float32, Device::cpu()),  // x
        randn({D, D},  DType::Float32, Device::cpu()),  // W1
        randn({D},     DType::Float32, Device::cpu()),  // b1
        randn({D},     DType::Float32, Device::cpu()),  // gamma
        randn({D},     DType::Float32, Device::cpu()),  // beta
        randn({D, D2}, DType::Float32, Device::cpu()),  // W2
        randn({D2},    DType::Float32, Device::cpu()),  // b2
    };

    auto eager_grads = run_eager(layernorm_mlp_model, seeds, dev);
    auto run = run_compiled(layernorm_mlp_model, seeds, dev);

    EXPECT_GT(run.grad_forwards, 0u)
        << "compiled differentiable graph was not used on " << backend_name(dev);

    ASSERT_EQ(run.grads.size(), eager_grads.size());
    for (size_t i = 0; i < eager_grads.size(); ++i) {
        SCOPED_TRACE(std::string("input #") + std::to_string(i));
        // LayerNorm backward accumulates a few extra rounding layers; allow a
        // slightly looser atol than the pure-linear chain.
        EXPECT_TENSORS_CLOSE(eager_grads[i], run.grads[i], 1e-3f, 5e-4f);
    }
}

// ---------------------------------------------------------------------------
// Finite-difference gradcheck (CPU): validate the COMPILED backward against
// central differences of the forward. Proves the compiled backward is correct,
// not merely self-consistent with eager.
// ---------------------------------------------------------------------------
TEST(JITAutogradGradcheck, CompiledBackward_FiniteDifference_CPU) {
    const Device cpu = Device::cpu();
    const int64_t N = 3, D0 = 5, H = 4, D2 = 3;
    std::vector<Tensor> seeds = {
        randn({N, D0}, DType::Float32, cpu),
        randn({D0, H}, DType::Float32, cpu),
        randn({H},     DType::Float32, cpu),
        randn({H, D2}, DType::Float32, cpu),
        randn({D2},    DType::Float32, cpu),
    };

    // Analytic gradient of the loss w.r.t. x, via the COMPILED backward.
    auto run = run_compiled(linear_chain_model, seeds, cpu);
    ASSERT_GT(run.grad_forwards, 0u) << "compiled graph not used";
    Tensor analytic_dx = run.grads[0].contiguous();  // grad w.r.t. seeds[0] == x

    // Central finite differences of the forward w.r.t. each element of x.
    const float eps = 2e-3f;
    Tensor x = seeds[0].contiguous();
    const int64_t numel = x.numel();
    std::vector<float> fd(static_cast<size_t>(numel));
    for (int64_t idx = 0; idx < numel; ++idx) {
        Tensor xp = x.clone();
        Tensor xm = x.clone();
        xp.data<float>()[idx] += eps;
        xm.data<float>()[idx] -= eps;

        std::vector<Tensor> tp = seeds; tp[0] = xp;
        std::vector<Tensor> tm = seeds; tm[0] = xm;
        float lp = eval_scalar(linear_chain_model, tp);
        float lm = eval_scalar(linear_chain_model, tm);
        fd[static_cast<size_t>(idx)] = (lp - lm) / (2.0f * eps);
    }

    const float* a = analytic_dx.data<float>();
    float max_abs_err = 0.0f;
    for (int64_t idx = 0; idx < numel; ++idx) {
        float err = std::fabs(a[idx] - fd[static_cast<size_t>(idx)]);
        max_abs_err = std::max(max_abs_err, err);
    }
    // ReLU-piecewise-linear model: central differences are exact away from the
    // kinks, so a tight bound holds.
    EXPECT_LT(max_abs_err, 2e-2f)
        << "compiled backward disagrees with finite differences (max abs err "
        << max_abs_err << ")";
}

// ---------------------------------------------------------------------------
// Closure-capture training loop: forward THROUGH the compiled function, backward,
// SGD step — over several steps — and assert step-for-step parity with an eager
// training loop starting from the same seeds. Proves:
//   (a) every captured parameter gets a non-null grad matching eager,
//   (b) the loss decreases and tracks eager exactly,
//   (c) after an optimizer step the next compiled forward uses the UPDATED
//       weights (parameter values stay identical to eager -> no staleness),
//   (d) the compiled differentiable graph actually ran (num_grad_forwards > 0).
// ---------------------------------------------------------------------------
TEST_P(JITAutogradParity, ClosureCapturedParams_TrainingLoop) {
    const Device dev = device;  // parametrized backend (fixture resolves/skips)
    SCOPED_TRACE(std::string("ClosureTraining on ") + backend_name(dev));

    const int64_t N = 4, D0 = 6, H = 5, D2 = 4;
    std::vector<Tensor> pseeds = {
        randn({D0, H}, DType::Float32, Device::cpu()),  // W1
        randn({H},     DType::Float32, Device::cpu()),  // b1
        randn({H, D2}, DType::Float32, Device::cpu()),  // W2
        randn({D2},    DType::Float32, Device::cpu()),  // b2
    };
    const Tensor x_seed = randn({N, D0}, DType::Float32, Device::cpu());
    const Tensor t_seed = randn({N, D2}, DType::Float32, Device::cpu());
    const double scale = 1.0 / static_cast<double>(N * D2);
    const double lr = 5e-2;
    const int steps = 5;

    // Two independent parameter sets from identical seeds: one trained eagerly,
    // one trained through the compiled function.
    auto eager_p = make_param_leaves(pseeds, dev);
    auto comp_p  = make_param_leaves(pseeds, dev);

    const Variable x(x_seed.clone().to(dev), /*requires_grad=*/false);
    const Variable target(t_seed.clone().to(dev), /*requires_grad=*/false);

    // The compiled function CLOSES OVER comp_p (parameters NOT passed as inputs)
    // and receives only x. Declare comp_p as the trainable parameters.
    jit::CompileConfig cfg;
    auto compiled = jit::compile(
        jit::CompiledFunction::FnTypeN([&](std::span<const Variable> ins) {
            return closure_mlp_loss(ins[0], comp_p, target, scale);
        }),
        comp_p, cfg);

    float first_loss = 0.0f, last_loss = 0.0f;
    for (int step = 0; step < steps; ++step) {
        SCOPED_TRACE(std::string("step ") + std::to_string(step));
        zero_grads(eager_p);
        zero_grads(comp_p);

        // Eager reference step.
        Variable le = closure_mlp_loss(x, eager_p, target, scale);
        le.backward();

        // Compiled step (forward through the captured graph).
        Variable lc = compiled(x);
        lc.backward();
        dev.synchronize();

        const float le_v = le.tensor().to(DType::Float32).to(Device::cpu()).item<float>();
        const float lc_v = lc.tensor().to(DType::Float32).to(Device::cpu()).item<float>();
        // Loss tracks eager step-for-step.
        EXPECT_NEAR(le_v, lc_v, 1e-3f + 1e-3f * std::fabs(le_v));
        if (step == 0) first_loss = lc_v;
        last_loss = lc_v;

        // Every captured parameter must receive a non-null gradient matching
        // eager to near machine precision (same backend, same kernels).
        for (size_t i = 0; i < comp_p.size(); ++i) {
            SCOPED_TRACE(std::string("param #") + std::to_string(i));
            ASSERT_TRUE(comp_p[i]->grad().has_value())
                << "captured parameter " << i << " got no gradient through JIT";
            Tensor ge = eager_p[i]->grad().value().to(Device::cpu());
            Tensor gc = comp_p[i]->grad().value().to(Device::cpu());
            EXPECT_TENSORS_CLOSE(ge, gc, 1e-4f, 1e-5f);
        }

        // Optimizer step on both (REPLACES parameter storage).
        sgd_step(eager_p, lr);
        sgd_step(comp_p, lr);

        // After the step the parameter values must stay identical: the next
        // compiled forward reads the UPDATED weights (no frozen-constant
        // staleness).
        for (size_t i = 0; i < comp_p.size(); ++i) {
            SCOPED_TRACE(std::string("post-step param #") + std::to_string(i));
            Tensor pe = eager_p[i]->tensor().to(Device::cpu());
            Tensor pc = comp_p[i]->tensor().to(Device::cpu());
            EXPECT_TENSORS_CLOSE(pe, pc, 1e-4f, 1e-5f);
        }
    }

    EXPECT_GT(compiled.num_grad_forwards(), 0u)
        << "compiled differentiable graph was not used on " << backend_name(dev);
    // Gradient descent on this convex-in-outputs MSE strictly reduces the loss.
    EXPECT_LT(last_loss, first_loss)
        << "training-through-JIT did not reduce the loss";
}

// ---------------------------------------------------------------------------
// Finite-difference gradcheck on a CAPTURED parameter (W1), CPU. Validates the
// compiled backward's gradient for a closure-captured parameter against central
// differences of the forward — not merely against eager autograd.
// ---------------------------------------------------------------------------
TEST(JITAutogradGradcheck, ClosureCapturedParam_FiniteDifference_CPU) {
    const Device cpu = Device::cpu();
    const int64_t N = 3, D0 = 4, H = 4, D2 = 3;

    // Central finite-difference gradchecks are only valid away from a
    // ReLU's non-differentiable kink at 0: closure_mlp_loss applies
    // nn::relu(matmul(x, W1) + b1), and if any pre-activation element lands
    // within the +-eps perturbation radius used below, the probe can cross
    // the kink -- the analytic (single-sided) backprop gradient then
    // genuinely disagrees with the finite-difference estimate for that
    // element. This is a textbook FD/gradcheck pitfall, not a JIT bug
    // (confirmed by bisecting against the pre-fusion-fix baseline, which
    // reproduces the identical ~1%-of-random-draws / 10-20x-over-tolerance
    // failure). Reject-sample until every pre-activation clears a safe
    // margin from the kink.
    std::vector<Tensor> pseeds;
    Tensor x_seed, t_seed;
    constexpr float kKinkMargin = 0.5f;
    constexpr int kMaxAttempts = 64;
    int attempt = 0;
    for (; attempt < kMaxAttempts; ++attempt) {
        pseeds = {
            randn({D0, H}, DType::Float32, cpu),  // W1 (the checked parameter)
            randn({H},     DType::Float32, cpu),  // b1
            randn({H, D2}, DType::Float32, cpu),  // W2
            randn({D2},    DType::Float32, cpu),  // b2
        };
        x_seed = randn({N, D0}, DType::Float32, cpu);
        t_seed = randn({N, D2}, DType::Float32, cpu);

        Tensor h = (tenzor::matmul(x_seed, pseeds[0]) + pseeds[1]).contiguous();
        const float* hp = h.data<float>();
        bool safe = true;
        for (int64_t i = 0; i < h.numel() && safe; ++i) {
            if (std::fabs(hp[i]) < kKinkMargin) safe = false;
        }
        if (safe) break;
    }
    ASSERT_LT(attempt, kMaxAttempts)
        << "could not find a ReLU-kink-safe random draw for the gradcheck";

    const double scale = 1.0 / static_cast<double>(N * D2);

    auto comp_p = make_param_leaves(pseeds, cpu);
    const Variable x(x_seed.clone(), /*requires_grad=*/false);
    const Variable target(t_seed.clone(), /*requires_grad=*/false);

    jit::CompileConfig cfg;
    auto compiled = jit::compile(
        jit::CompiledFunction::FnTypeN([&](std::span<const Variable> ins) {
            return closure_mlp_loss(ins[0], comp_p, target, scale);
        }),
        comp_p, cfg);

    Variable loss = compiled(x);
    loss.backward();
    ASSERT_GT(compiled.num_grad_forwards(), 0u) << "compiled graph not used";
    ASSERT_TRUE(comp_p[0]->grad().has_value());
    Tensor analytic_dW1 = comp_p[0]->grad().value().contiguous();

    // Central finite differences of the forward w.r.t. each element of W1.
    // Evaluate the loss eagerly with W1 perturbed (all params as non-grad).
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

    const float eps = 2e-3f;
    Tensor W1 = pseeds[0].contiguous();
    const int64_t numel = W1.numel();
    const float* a = analytic_dW1.data<float>();
    float max_abs_err = 0.0f;
    for (int64_t idx = 0; idx < numel; ++idx) {
        Tensor wp = W1.clone();
        Tensor wm = W1.clone();
        wp.data<float>()[idx] += eps;
        wm.data<float>()[idx] -= eps;
        float fd = (eval_loss(wp) - eval_loss(wm)) / (2.0f * eps);
        max_abs_err = std::max(max_abs_err, std::fabs(a[idx] - fd));
    }
    EXPECT_LT(max_abs_err, 2e-2f)
        << "compiled backward for captured parameter W1 disagrees with finite "
           "differences (max abs err " << max_abs_err << ")";
}

// ---------------------------------------------------------------------------
// Closure-captured parameter used INSIDE a jit::cond branch. The compiled graph
// records the then-branch as a SUBGRAPH; the captured W/b must be live parameter
// leaves in that subgraph so backward through the compiled If reaches
// param->grad(). Deterministic (true) condition -> the then-branch (which uses
// the params) always runs, matching the eager reference. Asserts:
//   (a) each captured param gets a non-null grad matching eager autograd,
//   (b) the compiled differentiable graph actually ran (num_grad_forwards > 0),
//       i.e. it did NOT silently fall back to eager.
// ---------------------------------------------------------------------------
TEST_P(JITAutogradParity, ClosureCapturedParams_IfBranch) {
    const Device dev = device;  // parametrized backend (fixture resolves/skips)
    SCOPED_TRACE(std::string("IfBranch on ") + backend_name(dev));

    const int64_t N = 4, D = 6;
    std::vector<Tensor> pseeds = {
        randn({D, D}, DType::Float32, Device::cpu()),  // W (used inside then-branch)
        randn({D},    DType::Float32, Device::cpu()),  // b (used inside then-branch)
    };
    const Tensor x_seed = randn({N, D}, DType::Float32, Device::cpu());
    // Deterministic TRUE condition -> the then-branch (which uses W,b) runs.
    const Tensor pred = tenzor::full({1}, 1.0f, DType::Float32, Device::cpu());

    auto eager_p = make_param_leaves(pseeds, dev);
    auto comp_p  = make_param_leaves(pseeds, dev);
    const Variable x(x_seed.clone().to(dev), /*requires_grad=*/false);

    // Eager reference: same closure, jit::cond runs the branch eagerly.
    Variable le = if_branch_loss(x, eager_p, pred);
    le.backward();
    dev.synchronize();

    // Compiled: closes over comp_p, uses them inside the then-branch subgraph.
    jit::CompileConfig cfg;
    auto compiled = jit::compile(
        jit::CompiledFunction::FnTypeN([&](std::span<const Variable> ins) {
            return if_branch_loss(ins[0], comp_p, pred);
        }),
        comp_p, cfg);

    Variable lc = compiled(x);
    lc.backward();
    dev.synchronize();

    // The compiled differentiable graph — not the eager fn_ fallback — must have
    // produced these gradients. This is the load-bearing check: an eager
    // fallback would still yield correct grads, so parity alone is insufficient.
    ASSERT_GT(compiled.num_grad_forwards(), 0u)
        << "compiled If-branch graph was not used (fell back to eager) on "
        << backend_name(dev);

    const float le_v = le.tensor().to(DType::Float32).to(Device::cpu()).item<float>();
    const float lc_v = lc.tensor().to(DType::Float32).to(Device::cpu()).item<float>();
    EXPECT_NEAR(le_v, lc_v, 1e-3f + 1e-3f * std::fabs(le_v));

    ASSERT_EQ(comp_p.size(), eager_p.size());
    for (size_t i = 0; i < comp_p.size(); ++i) {
        SCOPED_TRACE(std::string("captured param #") + std::to_string(i));
        ASSERT_TRUE(comp_p[i]->grad().has_value())
            << "captured parameter " << i
            << " used inside the If branch got no gradient through JIT";
        Tensor ge = eager_p[i]->grad().value().to(Device::cpu());
        Tensor gc = comp_p[i]->grad().value().to(Device::cpu());
        EXPECT_TENSORS_CLOSE(ge, gc, 1e-4f, 1e-5f);
    }
}

// ---------------------------------------------------------------------------
// Closure-captured parameter used INSIDE a jit::while_loop body. The compiled
// graph records the body as a SUBGRAPH executed once per iteration; the captured
// W must be a live parameter leaf there so backward accumulates the per-iteration
// contributions into param->grad() (W is used `limit` times). Asserts non-null
// grad matching eager, and num_grad_forwards > 0 (compiled loop actually ran).
// ---------------------------------------------------------------------------
TEST_P(JITAutogradParity, ClosureCapturedParams_WhileLoop) {
    const Device dev = device;  // parametrized backend (fixture resolves/skips)
    SCOPED_TRACE(std::string("WhileLoop on ") + backend_name(dev));

    const int64_t N = 4, D = 5;
    const float K = 3.0f;  // fixed trip count
    std::vector<Tensor> pseeds = {
        randn({D, D}, DType::Float32, Device::cpu()),  // W (used inside the loop body)
    };
    const Tensor x_seed = randn({N, D}, DType::Float32, Device::cpu());

    auto eager_p = make_param_leaves(pseeds, dev);
    auto comp_p  = make_param_leaves(pseeds, dev);
    const Variable x(x_seed.clone().to(dev), /*requires_grad=*/false);

    auto make_counter = [&] {
        return Variable(tenzor::full({1}, 0.0f, DType::Float32, dev), false);
    };
    auto make_limit = [&] {
        return Variable(tenzor::full({1}, K, DType::Float32, dev), false);
    };

    // Eager reference.
    Variable le = loop_matmul_loss(x, eager_p, make_counter(), make_limit(), dev);
    le.backward();
    dev.synchronize();

    // Compiled: closes over comp_p, uses W inside the loop body subgraph.
    jit::CompileConfig cfg;
    auto compiled = jit::compile(
        jit::CompiledFunction::FnTypeN([&](std::span<const Variable> ins) {
            return loop_matmul_loss(ins[0], comp_p, make_counter(), make_limit(), dev);
        }),
        comp_p, cfg);

    Variable lc = compiled(x);
    lc.backward();
    dev.synchronize();

    ASSERT_GT(compiled.num_grad_forwards(), 0u)
        << "compiled while-loop graph was not used (fell back to eager) on "
        << backend_name(dev);

    const float le_v = le.tensor().to(DType::Float32).to(Device::cpu()).item<float>();
    const float lc_v = lc.tensor().to(DType::Float32).to(Device::cpu()).item<float>();
    EXPECT_NEAR(le_v, lc_v, 1e-3f + 1e-3f * std::fabs(le_v));

    ASSERT_TRUE(comp_p[0]->grad().has_value())
        << "captured parameter used inside the loop body got no gradient through JIT";
    Tensor ge = eager_p[0]->grad().value().to(Device::cpu());
    Tensor gc = comp_p[0]->grad().value().to(Device::cpu());
    // K matmuls accumulate a few extra rounding layers -> slightly looser atol.
    EXPECT_TENSORS_CLOSE(ge, gc, 1e-3f, 5e-4f);
}

// ---------------------------------------------------------------------------
// Finite-difference gradcheck (CPU) on a parameter captured INSIDE an If branch.
// Validates the compiled control-flow backward against central differences of
// the forward — not merely against eager autograd.
// ---------------------------------------------------------------------------
TEST(JITAutogradGradcheck, ClosureCapturedParam_IfBranch_FiniteDifference_CPU) {
    const Device cpu = Device::cpu();
    const int64_t N = 3, D = 4;
    std::vector<Tensor> pseeds = {
        randn({D, D}, DType::Float32, cpu),  // W (the checked parameter)
        randn({D},    DType::Float32, cpu),  // b
    };
    const Tensor x_seed = randn({N, D}, DType::Float32, cpu);
    const Tensor pred = tenzor::full({1}, 1.0f, DType::Float32, cpu);  // true

    auto comp_p = make_param_leaves(pseeds, cpu);
    const Variable x(x_seed.clone(), /*requires_grad=*/false);

    jit::CompileConfig cfg;
    auto compiled = jit::compile(
        jit::CompiledFunction::FnTypeN([&](std::span<const Variable> ins) {
            return if_branch_loss(ins[0], comp_p, pred);
        }),
        comp_p, cfg);

    Variable loss = compiled(x);
    loss.backward();
    ASSERT_GT(compiled.num_grad_forwards(), 0u) << "compiled If graph not used";
    ASSERT_TRUE(comp_p[0]->grad().has_value());
    Tensor analytic_dW = comp_p[0]->grad().value().contiguous();

    // Central finite differences of the forward w.r.t. each element of W. The
    // forward runs eagerly (jit::cond executes the branch directly).
    auto eval_loss = [&](const Tensor& W) -> float {
        std::vector<std::shared_ptr<Variable>> ps = {
            std::make_shared<Variable>(W, false),
            std::make_shared<Variable>(pseeds[1], false),
        };
        Variable l = if_branch_loss(x, ps, pred);
        return l.tensor().item<float>();
    };

    const float eps = 2e-3f;
    Tensor W = pseeds[0].contiguous();
    const int64_t numel = W.numel();
    const float* a = analytic_dW.data<float>();
    float max_abs_err = 0.0f;
    for (int64_t idx = 0; idx < numel; ++idx) {
        Tensor wp = W.clone();
        Tensor wm = W.clone();
        wp.data<float>()[idx] += eps;
        wm.data<float>()[idx] -= eps;
        float fd = (eval_loss(wp) - eval_loss(wm)) / (2.0f * eps);
        max_abs_err = std::max(max_abs_err, std::fabs(a[idx] - fd));
    }
    EXPECT_LT(max_abs_err, 2e-2f)
        << "compiled If-branch backward for captured parameter W disagrees with "
           "finite differences (max abs err " << max_abs_err << ")";
}

INSTANTIATE_BACKEND_TESTS(JITAutogradParity);

int main(int argc, char** argv) {
    // Force full IEEE 754 FP32 on CUDA matmul (disable TF32 tensor cores) so the
    // compiled-vs-eager backward parity holds at the tight tolerance used here.
    setenv("TENZOR_DISABLE_TF32", "1", /*overwrite=*/1);
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    tenzor::finalize();
    return result;
}
