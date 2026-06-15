#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>
#include <iostream>

namespace tenzor::nn {

// Backward function for ReLU
class ReLUBackward : public Function {
public:
    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("ReLUBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output = grad_outputs[0];
        const auto& input = saved_tensors()[0];

        // Use the backend's native relu_backward kernel for efficiency
        // This avoids CPU transfers that happen when using comparison + dtype conversion
        std::vector<Tensor> backward_inputs = {grad_output, input};
        auto grad_input = dispatch(OpId::ReLUBackward, backward_inputs)[0];

        return {grad_input};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // ReLU backward: grad_input = grad_output * (input > 0)
        // The mask is non-differentiable, compute at Tensor level
        const auto& input = saved_tensors()[0];
        std::vector<Tensor> mask_inputs = {ones(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                                                 input.dtype(), input.device()), input};
        auto mask = dispatch(OpId::ReLUBackward, mask_inputs)[0];
        Variable mask_var(mask, false);
        return {grad_outputs[0] * mask_var};
    }

    // Audit A.2: opt in to the OpId-keyed graph-optimizer matchers so a
    // future switch from RTTI substring to OpId never silently loses
    // this Function from the fusion candidate set.
    auto op_id() const -> OpId override { return OpId::ReLU; }
    auto name() const -> std::string override { return "ReLUBackward"; }
};

// NOTE: The former local backward classes (TanhBackward, LeakyReLUBackward,
// GeLUBackward, SwishBackward, ELUBackward, SELUBackward, MishBackward,
// SigmoidBackward) have been removed. Production autograd for these ops goes
// through the _AG variants in src/autograd/function_activations.cpp — the
// nn-layer functions just delegate to ::tenzor:: AG-aware ops. The local
// classes never had backward_with_variables() support and their dispatch
// would produce zero 2nd derivatives silently. See GRADCHECK_COVERAGE_ANALYSIS.md.

// Module implementations
auto ReLU::forward_impl(const Variable& input) -> Variable {
    return relu(input);
}

auto ReLU6::forward_impl(const Variable& input) -> Variable {
    // ReLU6(x) = min(max(0, x), 6)
    return clamp(relu(input), 0.0f, 6.0f);
}

LeakyReLU::LeakyReLU(double negative_slope) : negative_slope_(negative_slope) {}

auto LeakyReLU::forward_impl(const Variable& input) -> Variable {
    return nn::leaky_relu(input, negative_slope_);
}

auto Sigmoid::forward_impl(const Variable& input) -> Variable {
    return nn::sigmoid(input);
}

auto Tanh::forward_impl(const Variable& input) -> Variable {
    return nn::tanh(input);
}

Softmax::Softmax(int64_t dim) : dim_(dim) {}

auto Softmax::forward_impl(const Variable& input) -> Variable {
    return tenzor::softmax(input, dim_);
}

// Functional implementations with autograd support
auto relu(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = dispatch(OpId::ReLU, inputs)[0];
        return Variable(result, false);
    }

    // Compute forward
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = dispatch(OpId::ReLU, inputs_vec)[0];

    // Set up autograd
    auto grad_fn = std::make_shared<ReLUBackward>();
    // Save only input - relu_backward kernel only needs input
    grad_fn->save_for_backward({input.tensor()});

    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) {
        next_funcs.push_back(input.grad_fn());
    }
    grad_fn->set_next_functions(next_funcs);

    // Track input variable for gradient accumulation
    std::vector<Variable> input_vars;
    input_vars.push_back(input);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

auto sigmoid(const Variable& input) -> Variable {
    // Delegate to the tenzor:: level op which uses SigmoidBackward_AG
    // (a proper backward_with_variables that threads through Variable
    // ops) so higher-order gradients work — previously this path
    // instantiated the local SigmoidBackward stub and silently produced
    // zero second derivatives. P4.2: higher-order coverage completion.
    return ::tenzor::sigmoid(input);
}

auto tanh(const Variable& input) -> Variable {
    // Same rationale as sigmoid — delegate to the _AG path for proper
    // higher-order support via TanhBackward_AG.
    return ::tenzor::tanh(input);
}

auto leaky_relu(const Variable& input, double negative_slope) -> Variable {
    // P4.2: delegate to the tenzor-level op which uses LeakyReluBackward
    // (a real backward_with_variables that threads through Variable-level
    // ops) so higher-order gradients work. Previously this path used the
    // local LeakyReLUBackward stub.
    return ::tenzor::leaky_relu(input, negative_slope);
}

auto gelu(const Variable& input, const std::string& approximate) -> Variable {
    // P4.2: delegate to the tenzor-level op for higher-order support via
    // GeluBackward in function_activations.cpp. The tanh approximation
    // path (when approximate == "tanh") composes existing Variable ops
    // which already support higher-order naturally.
    if (approximate == "tanh") {
        auto x = input;
        auto x_cubed = x * x * x;
        Tensor coef_t = full({1}, 0.044715, input.tensor().dtype(), input.tensor().device());
        Tensor sqrt_2_pi_t = full({1}, std::sqrt(2.0 / M_PI),
                                  input.tensor().dtype(), input.tensor().device());
        Tensor half_t = full({1}, 0.5, input.tensor().dtype(), input.tensor().device());
        Tensor one_t = full({1}, 1.0, input.tensor().dtype(), input.tensor().device());
        Variable coef(coef_t, false);
        Variable sqrt_2_pi(sqrt_2_pi_t, false);
        Variable half(half_t, false);
        Variable one(one_t, false);
        auto inner = sqrt_2_pi * (x + coef * x_cubed);
        auto tanh_val = nn::tanh(inner);  // nn::tanh already routes to the _AG path
        return half * x * (one + tanh_val);
    }
    return ::tenzor::gelu(input);
}

auto softmax(const Variable& input, int64_t dim) -> Variable {
    // Use autograd-aware version from tenzor namespace
    return tenzor::softmax(input, dim);
}

auto log_softmax(const Variable& input, int64_t dim) -> Variable {
    // Use autograd-aware version from tenzor namespace
    return tenzor::log_softmax(input, dim);
}

LogSoftmax::LogSoftmax(int64_t dim) : dim_(dim) {}

auto LogSoftmax::forward_impl(const Variable& input) -> Variable {
    return tenzor::log_softmax(input, dim_);
}

GELU::GELU(const std::string& approximate) : approximate_(approximate) {
    if (approximate != "none" && approximate != "tanh") {
        throw std::invalid_argument("GELU approximate must be 'none' or 'tanh', got '" + approximate + "'");
    }
}

auto GELU::forward_impl(const Variable& input) -> Variable {
    return gelu(input, approximate_);
}

auto ELU::forward_impl(const Variable& input) -> Variable {
    return elu(input, alpha_);
}

ELU::ELU(double alpha) : alpha_(alpha) {}

auto elu(const Variable& input, double alpha) -> Variable {
    // P4.2: delegate to the tenzor-level op which uses EluBackward (real
    // backward_with_variables).
    return ::tenzor::elu(input, static_cast<float>(alpha));
}

// CELU: α · ELU(x/α, 1). Pure Variable-level composition — autograd threads
// naturally through the existing ELU backward.
CELU::CELU(double alpha) : alpha_(alpha) {
    if (alpha == 0.0) {
        throw std::invalid_argument("CELU alpha must be non-zero");
    }
}

auto CELU::forward_impl(const Variable& input) -> Variable {
    return nn::celu(input, alpha_);
}

auto celu(const Variable& input, double alpha) -> Variable {
    if (alpha == 0.0) {
        throw std::invalid_argument("celu: alpha must be non-zero");
    }
    auto scaled = input * (1.0 / alpha);
    auto elu_out = nn::elu(scaled, 1.0);
    return elu_out * alpha;
}

auto SELU::forward_impl(const Variable& input) -> Variable {
    return nn::selu(input);
}

auto selu(const Variable& input) -> Variable {
    // P4.2: delegate for higher-order support via SeluBackward.
    return ::tenzor::selu(input);
}

auto Swish::forward_impl(const Variable& input) -> Variable {
    return swish(input);
}

auto swish(const Variable& input) -> Variable {
    // P4.2c: express swish as a Variable-level composition —
    // swish(x) = x * sigmoid(x). Both operand factors are autograd-
    // aware (nn::sigmoid already delegates to the tenzor:: _AG path),
    // so higher-order gradients flow naturally through the chain rule
    // without needing a custom SwishBackward_AG.
    //
    // For inference (no grad), fall back to the fused kernel dispatch
    // which is faster than two separate element-wise ops.
    if (!input.requires_grad() || !is_grad_enabled()) {
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = dispatch(OpId::Swish, inputs)[0];
        return Variable(result, false);
    }
    return input * nn::sigmoid(input);
}

auto Mish::forward_impl(const Variable& input) -> Variable {
    return nn::mish(input);
}

auto mish(const Variable& input) -> Variable {
    // P4.2: delegate for higher-order support via MishBackward.
    return ::tenzor::mish(input);
}

// In-place activation functions
auto relu_(Tensor& input) -> Tensor& {
    if (!input.is_contiguous()) {
        throw std::runtime_error("In-place relu requires contiguous tensor");
    }

    dispatch_inplace(OpId::ReLUInplace, input, std::span<const Tensor>{});

    return input;
}

auto sigmoid_(Tensor& input) -> Tensor& {
    if (!input.is_contiguous()) {
        throw std::runtime_error("In-place sigmoid requires contiguous tensor");
    }

    dispatch_inplace(OpId::SigmoidInplace, input, std::span<const Tensor>{});

    return input;
}

auto tanh_(Tensor& input) -> Tensor& {
    if (!input.is_contiguous()) {
        throw std::runtime_error("In-place tanh requires contiguous tensor");
    }

    dispatch_inplace(OpId::TanhInplace, input, std::span<const Tensor>{});

    return input;
}

auto leaky_relu_(Tensor& input, double negative_slope) -> Tensor& {
    if (!input.is_contiguous()) {
        throw std::runtime_error("In-place leaky_relu requires contiguous tensor");
    }

    NewOpAttributes attrs;
    // Every backend kernel reads AttrKey::Alpha; MPS and the JIT tracer read
    // AttrKey::Negative_slope. Set both so the value isn't silently dropped
    // on either side. (See feedback_attr_key_bug.md in MEMORY.md.)
    attrs.set(AttrKey::Alpha, negative_slope);
    attrs.set(AttrKey::Negative_slope, negative_slope);
    dispatch_inplace(OpId::LeakyReLUInplace, input, std::span<const Tensor>{}, attrs);

    return input;
}

auto gelu_(Tensor& input) -> Tensor& {
    if (!input.is_contiguous()) {
        throw std::runtime_error("In-place gelu requires contiguous tensor");
    }

    dispatch_inplace(OpId::GeluInplace, input, std::span<const Tensor>{});

    return input;
}

// PReLU implementation
PReLU::PReLU(int64_t num_parameters, double init)
    : num_parameters_(num_parameters) {
    auto weight_tensor = tenzor::full({num_parameters}, init, DType::Float32, Device::cpu());
    Variable weight_var(weight_tensor, true);
    register_parameter("weight", std::move(weight_var));
}

auto PReLU::forward_impl(const Variable& input) -> Variable {
    // PReLU(x) = max(0, x) + weight * min(0, x)
    // = relu(x) + weight * (x - relu(x))  [avoiding min for autograd]
    auto& weight = *parameters_["weight"];
    auto relu_x = relu(input);
    auto neg_part = input - relu_x;  // min(0, x)

    // The learnable weight has shape [num_parameters]. With right-aligned
    // broadcasting a per-channel weight [C] would align with the LAST input
    // axis (W for [N,C,H,W]) instead of the channel dim. Reshape it to
    // [1, C, 1, ..., 1] (rank == input.ndim()) so it broadcasts along dim 1,
    // matching PyTorch's per-channel PReLU. Single-parameter weights ([1])
    // broadcast against every element regardless, so leave them untouched.
    if (num_parameters_ > 1 && input.tensor().ndim() > 2) {
        std::vector<int64_t> wshape(input.tensor().ndim(), 1);
        wshape[1] = num_parameters_;
        auto weight_bcast = ::tenzor::reshape(weight, wshape);
        return relu_x + weight_bcast * neg_part;
    }

    // [N, C] (or num_parameters_ == 1): right-aligned broadcasting is correct.
    return relu_x + weight * neg_part;
}

auto Hardswish::forward_impl(const Variable& input) -> Variable {
    return hardswish(input);
}

auto Hardsigmoid::forward_impl(const Variable& input) -> Variable {
    return hardsigmoid(input);
}

// Functional Hardswish
//
// P4.2c: Hardswish(x) = x * clamp(x + 3, 0, 6) / 6 is piecewise linear
// on the input outside the [-3, 3] transition region, and piecewise-
// quadratic inside. The second derivative is zero a.e. for most ops
// but has a delta at the clamp boundaries — PyTorch treats it as zero
// for higher-order purposes. We express the forward as a pure
// Variable-level composition so create_graph=true threads through
// existing autograd-aware clamp / mul / add chain; no custom
// HardswishBackward needed.
auto hardswish(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        auto x_plus_3_t = input.tensor() + 3.0f;
        auto clamped_t = tenzor::clamp(x_plus_3_t, 0.0f, 6.0f);
        return Variable(input.tensor() * clamped_t / 6.0f, false);
    }
    // Variable-level composition: autograd threads naturally.
    auto shifted = input + 3.0;
    auto clamped = tenzor::clamp(shifted, 0.0, 6.0);
    return input * clamped * (1.0 / 6.0);
}

// Functional Hardsigmoid
//
// Same story: piecewise linear, Variable-level composition carries
// create_graph=true through without a custom backward.
auto hardsigmoid(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        auto x_plus_3_t = input.tensor() + 3.0f;
        auto clamped_t = tenzor::clamp(x_plus_3_t, 0.0f, 6.0f);
        return Variable(clamped_t / 6.0f, false);
    }
    auto shifted = input + 3.0;
    auto clamped = tenzor::clamp(shifted, 0.0, 6.0);
    return clamped * (1.0 / 6.0);
}

// GLU module
auto GLU::forward_impl(const Variable& input) -> Variable {
    return glu(input, dim_);
}

// Functional GLU
auto glu(const Variable& input, int64_t dim) -> Variable {
    auto shape = input.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("GLU: dim " + std::to_string(dim) + " out of range for " + std::to_string(ndim) + "D input");
    }
    if (shape[dim] % 2 != 0) {
        throw std::invalid_argument("GLU: input size along dim " + std::to_string(dim) + " must be even, got " + std::to_string(shape[dim]));
    }
    int64_t half = shape[dim] / 2;
    // Split input into two halves along dim
    auto a = tenzor::slice(input, dim, 0, half);
    auto b = tenzor::slice(input, dim, half, shape[dim]);
    // GLU(a, b) = a * sigmoid(b)
    return a * nn::sigmoid(b);
}

// ----------------------------------------------------------------------------
// Additional activations: compositions over existing autograd-aware primitives.
// These use the existing Variable operator overloads so they thread the
// gradient graph without custom Function classes.
//
// Softmin / Tanhshrink / Softshrink / Softsign compose cleanly from existing
// primitives. Hardshrink and Threshold have discontinuous-derivative points
// that require either a `where` op or a custom Function; they are provided
// as modules that throw at runtime with a pointer to the follow-up work.
// ----------------------------------------------------------------------------

auto Softmin::forward_impl(const Variable& input) -> Variable {
    return softmin(input, dim_);
}

auto softmin(const Variable& input, int64_t dim) -> Variable {
    // softmin(x, dim) = softmax(-x, dim)
    auto neg_input = input * -1.0;
    return nn::softmax(neg_input, dim);
}

auto Tanhshrink::forward_impl(const Variable& input) -> Variable {
    return tanhshrink(input);
}

auto tanhshrink(const Variable& input) -> Variable {
    // x - tanh(x)
    return input - nn::tanh(input);
}

auto Softshrink::forward_impl(const Variable& input) -> Variable {
    return softshrink(input, lambda_);
}

auto softshrink(const Variable& input, double lambda) -> Variable {
    if (lambda < 0.0) {
        throw std::invalid_argument("softshrink: lambda must be non-negative");
    }
    // sign(x) * max(|x| - lambda, 0)
    //   = relu(x - lambda) - relu(-x - lambda)
    auto pos = nn::relu(input - lambda);
    auto neg = nn::relu((input * -1.0) - lambda);
    return pos - neg;
}

auto Softsign::forward_impl(const Variable& input) -> Variable {
    return softsign(input);
}

auto softsign(const Variable& input) -> Variable {
    // x / (1 + |x|). abs() is non-differentiable at 0 but subgradient is 0.
    // audit-2026-05-03 — must compute |x| on the Variable (not the raw
    // tensor) so the backward through abs propagates. Previously the
    // denominator was wrapped as `Variable(t, requires_grad=false)`, which
    // treats it as a constant; backward then returns 1/(1+|x|) instead of
    // the correct 1/(1+|x|)^2.
    auto abs_input = tenzor::abs(input);
    auto denom = abs_input + Variable(::tenzor::ones(
        std::vector<int64_t>(input.shape().begin(), input.shape().end()),
        input.dtype(), input.device()), false);
    return input / denom;
}

auto Hardshrink::forward_impl(const Variable& input) -> Variable {
    return hardshrink(input, lambda_);
}

auto hardshrink(const Variable& input, double lambda) -> Variable {
    // P2.6b: hardshrink(x) = x if |x| > lambda else 0
    //
    // Expressed as a composition over Variable-level `where` so
    // create_graph=true threads through naturally. The step at
    // |x| == lambda is a measure-zero non-differentiability; autograd
    // conventions (matching PyTorch) treat it as zero subgradient
    // there, which falls out of the where-based construction.
    if (lambda < 0.0) {
        throw std::invalid_argument("hardshrink: lambda must be non-negative");
    }
    const auto dtype = input.tensor().dtype();
    const auto device = input.tensor().device();
    const auto shape = std::vector<int64_t>(
        input.tensor().shape().begin(), input.tensor().shape().end());

    // mask = |x| > lambda, computed at the Tensor level (non-differentiable).
    auto abs_t = tenzor::abs(input.tensor());
    auto lambda_t = tenzor::full(shape, lambda, dtype, device);
    auto mask_t = tenzor::gt(abs_t, lambda_t);

    // Variable-level where threads gradients through `x` on the pass-
    // through branch; the zero branch contributes no gradient.
    auto zero = Variable(tenzor::zeros(shape, dtype, device), false);
    auto mask_var = Variable(mask_t, false);
    return tenzor::where(mask_var, input, zero);
}

auto Threshold::forward_impl(const Variable& input) -> Variable {
    return threshold(input, threshold_, value_);
}

auto threshold(const Variable& input, double t, double value) -> Variable {
    // P2.6b: threshold(x, t, value) = x if x > t else value
    // Variable-level where preserves the gradient through the pass-
    // through branch.
    const auto dtype = input.tensor().dtype();
    const auto device = input.tensor().device();
    const auto shape = std::vector<int64_t>(
        input.tensor().shape().begin(), input.tensor().shape().end());

    auto t_t = tenzor::full(shape, t, dtype, device);
    auto mask_t = tenzor::gt(input.tensor(), t_t);

    auto value_var = Variable(tenzor::full(shape, value, dtype, device), false);
    auto mask_var = Variable(mask_t, false);
    return tenzor::where(mask_var, input, value_var);
}

auto Hardtanh::forward_impl(const Variable& input) -> Variable {
    return hardtanh(input, min_val_, max_val_);
}

auto hardtanh(const Variable& input, double min_val, double max_val) -> Variable {
    // hardtanh(x) = clamp(x, min_val, max_val)
    //
    // Expressed via Variable-level `where` so create_graph=true threads
    // through naturally for higher-order gradients. The gradient is 1 where
    // min_val <= x <= max_val and 0 otherwise.
    if (min_val > max_val) {
        throw std::invalid_argument("hardtanh: min_val must be <= max_val");
    }
    const auto dtype = input.tensor().dtype();
    const auto device = input.tensor().device();
    const auto shape = std::vector<int64_t>(
        input.tensor().shape().begin(), input.tensor().shape().end());

    auto min_t = tenzor::full(shape, min_val, dtype, device);
    auto max_t = tenzor::full(shape, max_val, dtype, device);

    // mask_below = x < min_val  →  clamp to min_val
    auto mask_below = Variable(tenzor::lt(input.tensor(), min_t), false);
    auto min_var = Variable(min_t, false);
    auto clamped_low = tenzor::where(mask_below, min_var, input);

    // mask_above = x > max_val  →  clamp to max_val
    auto max_t2 = tenzor::full(shape, max_val, dtype, device);
    auto mask_above = Variable(tenzor::gt(input.tensor(), max_t2), false);
    auto max_var = Variable(max_t, false);
    return tenzor::where(mask_above, max_var, clamped_low);
}

// ============================================================================
// Functional RReLU (Randomized Leaky ReLU)
// ============================================================================

class RReLUBackward : public Function {
public:
    RReLUBackward(double lower, double upper, bool training)
        : lower_(lower), upper_(upper), training_(training) {}

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("RReLUBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output = grad_outputs[0];
        const auto& input = saved_tensors()[0];

        if (training_) {
            // Training draws an independent random slope per negative element in
            // forward. The kernel-based midpoint backward (below) would apply the
            // fixed slope (lower+upper)/2 to every negative element, which is NOT
            // the slope actually applied, making the gradient wrong everywhere
            // x<=0. The kernel discards the noise mask, so we instead reconstruct
            // the exact per-element slope from the saved input and output:
            //   y = x for x>0, y = a*x for x<=0  =>  a = y/x  (x != 0)
            // slope = 1 for x>0, y/x for x<0, and 0 at the measure-zero kink x==0.
            const Tensor& output = saved_tensors()[1];
            Tensor zeros = ::tenzor::zeros_like(input);
            Tensor ones = ::tenzor::ones_like(input);
            // Safe denominator avoids div-by-zero at x==0 (masked out below).
            Tensor safe_input = ::tenzor::where(::tenzor::eq(input, zeros), ones, input);
            Tensor slope_neg = ::tenzor::div(output, safe_input);
            Tensor slope = ::tenzor::where(::tenzor::gt(input, zeros), ones, slope_neg);
            return {::tenzor::mul(grad_output, slope)};
        }

        // Eval mode: forward uses the deterministic midpoint slope, so the
        // kernel's midpoint backward is exactly correct.
        OpAttributes attrs;
        attrs.set(AttrKey::Lower, lower_);
        attrs.set(AttrKey::High, upper_);
        std::vector<Tensor> backward_inputs = {grad_output, input};
        auto grad_input = dispatch(OpId::RReLUBackward, backward_inputs, attrs)[0];
        return {grad_input};
    }

    // RReLU is piecewise-linear (y=x for x>0, y=a*x for x<=0 with random
    // a in [lower, upper] sampled at forward time). Its second derivative
    // is structurally zero everywhere except the kink at x=0, which has
    // measure zero. Use the structural-zero stub rather than throwing on
    // create_graph=true.
    TENZOR_HIGHER_ORDER_STRUCTURAL_ZERO_STUB()

    auto op_id() const -> OpId override { return OpId::RReLU; }
    auto name() const -> std::string override { return "RReLUBackward"; }

private:
    double lower_;
    double upper_;
    bool training_;
};

auto rrelu(const Variable& input, double lower, double upper, bool training) -> Variable {
    OpAttributes attrs;
    attrs.set(AttrKey::Lower, lower);
    attrs.set(AttrKey::High, upper);
    attrs.set(AttrKey::Training, training);

    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = dispatch(OpId::RReLU, inputs_vec, attrs)[0];

    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result_tensor, false);
    }

    auto grad_fn = std::make_shared<RReLUBackward>(lower, upper, training);
    // In training mode the per-element slope must be reconstructed from the
    // output (y = a*x for x<=0), so save both input and output. Eval mode only
    // needs the input for the deterministic midpoint backward.
    if (training) {
        grad_fn->save_for_backward({input.tensor(), result_tensor});
    } else {
        grad_fn->save_for_backward({input.tensor()});
    }

    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) {
        next_funcs.push_back(input.grad_fn());
    }
    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    input_vars.push_back(input);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

// ============================================================================
// Functional LogSigmoid: log(sigmoid(x)) = -softplus(-x)
// ============================================================================

class LogSigmoidBackward : public Function {
public:
    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("LogSigmoidBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output = grad_outputs[0];
        const auto& input = saved_tensors()[0];

        std::vector<Tensor> backward_inputs = {grad_output, input};
        auto grad_input = dispatch(OpId::LogSigmoidBackward, backward_inputs)[0];
        return {grad_input};
    }

    // Higher-order support: d(log_sigmoid(x))/dx = 1 - sigmoid(x). Building
    // the derivative with Variable-level ops lets the engine re-differentiate
    // it when create_graph=true (gradgradcheck path). Without this override
    // the default backward_with_variables returns zero 2nd derivatives.
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        const auto& input_tensor = saved_tensors()[0];
        Variable input_var(input_tensor, false);
        auto sig = ::tenzor::sigmoid(input_var);    // SigmoidBackward_AG → HO-capable
        auto shape_vec = std::vector<int64_t>(input_tensor.shape().begin(),
                                              input_tensor.shape().end());
        Variable one_var(ones(shape_vec, input_tensor.dtype(), input_tensor.device()),
                         false);
        return {grad_outputs[0] * (one_var - sig)};
    }

    auto op_id() const -> OpId override { return OpId::LogSigmoid; }
    auto name() const -> std::string override { return "LogSigmoidBackward"; }
};

auto log_sigmoid(const Variable& input) -> Variable {
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = dispatch(OpId::LogSigmoid, inputs_vec)[0];

    if (!input.requires_grad() || !is_grad_enabled()) {
        return Variable(result_tensor, false);
    }

    auto grad_fn = std::make_shared<LogSigmoidBackward>();
    grad_fn->save_for_backward({input.tensor()});

    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) {
        next_funcs.push_back(input.grad_fn());
    }
    grad_fn->set_next_functions(next_funcs);

    std::vector<Variable> input_vars;
    input_vars.push_back(input);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);
    return output;
}

} // namespace tenzor::nn
