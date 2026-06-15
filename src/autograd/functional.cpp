#include "tenzor/autograd/functional.hpp"
#include "tenzor/autograd/dual.hpp"
#include "tenzor/autograd/jvp_rules.hpp"
#include "tenzor/autograd/jvp_dispatch.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/utils/variable_cast.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tenzor {

namespace {

// A.4 multi-op JVP traversal: walk the autograd graph rooted at `out_grad_fn`
// in reverse-topological (i.e. forward-execution) order and chain
// `dispatch_jvp` calls.
//
// Returns `std::nullopt` if any node in the graph has no registered
// forward-mode rule (or its rule rejects the call); the caller then falls
// back to finite differences for the whole chain. This is the
// "no-workaround" contract: we either run the exact JVP for every op or
// admit defeat and switch to FD.
//
// `user_input_ptr` is the data pointer of the user-supplied input tensor.
// Any input_variable seen during the walk whose data pointer matches is
// treated as the user input (carries `seed_tangent`); any other leaf input
// (next_functions entry is nullptr and data_ptr differs) is treated as a
// constant and carries a zero tangent.
auto try_traverse_jvp(const std::shared_ptr<Function>& out_grad_fn,
                      const void* user_input_ptr,
                      const Tensor& seed_tangent,
                      const void* output_data_ptr) -> std::optional<Tensor> {
    if (!out_grad_fn) return std::nullopt;

    // 1. Collect every Function reachable from the output in DFS post-order.
    std::vector<std::shared_ptr<Function>> topo;
    std::unordered_set<Function*> visited;

    std::function<bool(const std::shared_ptr<Function>&)> dfs =
        [&](const std::shared_ptr<Function>& node) -> bool {
            if (!node) return true;                       // leaf edge
            if (!visited.insert(node.get()).second) return true;
            for (const auto& next : node->next_functions()) {
                if (!dfs(next)) return false;
            }
            // Early-reject any node we can't dispatch through: stops the
            // walk before we spend effort on an unreachable fast path.
            // A.4 multi-output extension: accept the node if EITHER a
            // single-output rule OR a multi-output rule is registered.
            OpId op = node->op_id();
            if (op == OpId::Unknown ||
                (!has_jvp_rule(op) && !has_jvp_rule_multi(op))) {
                return false;
            }
            topo.push_back(node);
            return true;
        };

    if (!dfs(out_grad_fn)) return std::nullopt;
    if (topo.empty()) return std::nullopt;

    // 2. Tangent tables.
    //    - `node_tangents` holds the primary (output-0) tangent for each
    //      Function, indexed by raw Function*. Single-output ops use it
    //      directly; multi-output ops also publish output-0 here as a
    //      fast path.
    //    - `node_tangents_multi` holds ALL output tangents for multi-output
    //      Functions. Keyed by `(Function*, output_idx)`. Downstream
    //      consumers that read a non-zero output slot of a multi-output
    //      producer find their tangent here via the data_ptr-resolved
    //      output_idx (see resolve_input_tangent below).
    std::unordered_map<Function*, Tensor> node_tangents;

    struct PairHash {
        std::size_t operator()(const std::pair<Function*, size_t>& p) const noexcept {
            std::size_t h1 = std::hash<Function*>{}(p.first);
            std::size_t h2 = std::hash<size_t>{}(p.second);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };
    std::unordered_map<std::pair<Function*, size_t>, Tensor, PairHash>
        node_tangents_multi;

    // Resolve the output slot a consumer's input reads from its multi-output
    // producer. The grad_fn graph carries no native output_nr; we recover it
    // by matching the consumer's `input_variables[i].tensor()` data_ptr
    // against the producer's saved_tensors data_ptrs. Multi-output forward
    // ops typically save their outputs for backward (e.g. EighBackward saves
    // {W, V}; LayerNormBackward saves {x, mean, rstd, gamma} where mean is
    // output 1 and rstd is output 2; QrBackward saves {Q, R}; etc.). When
    // the consumer's input matches a saved tensor that corresponds to a
    // known output slot, we return that slot; otherwise we default to 0.
    // Producer-specific saved_tensors→output_idx maps live in the Function
    // subclasses via the override returning a non-zero output_idx.
    auto find_output_idx = [](const std::shared_ptr<Function>& prod,
                              const Tensor& consumer_input) -> size_t {
        if (!prod) return 0;
        const void* cp = consumer_input.data_ptr();
        if (!cp) return 0;
        const auto& saved = const_cast<Function*>(prod.get())->saved_tensors();
        for (size_t i = 0; i < saved.size(); ++i) {
            if (saved[i].data_ptr() == cp) {
                return prod->jvp_saved_tensor_to_output_idx(i);
            }
        }
        return 0;
    };

    auto resolve_input_tangent = [&](const std::shared_ptr<Function>& prod,
                                     const Tensor& primal) -> Tensor {
        if (prod) {
            // Try the multi-output path first: if the producer has any
            // entries in node_tangents_multi, we may need to pick a
            // non-zero output slot.
            size_t out_idx = find_output_idx(prod, primal);
            if (out_idx != 0) {
                auto it_m = node_tangents_multi.find({prod.get(), out_idx});
                if (it_m != node_tangents_multi.end()) return it_m->second;
                // Fall through to single-output table — shouldn't happen
                // if the producer published all outputs, but safe.
            }
            auto it = node_tangents.find(prod.get());
            if (it != node_tangents.end()) return it->second;
            return Tensor();
        }
        // Leaf: either the user input or a constant.
        if (primal.data_ptr() != nullptr && primal.data_ptr() == user_input_ptr) {
            return seed_tangent;
        }
        // Constant input: zero tangent with matching dtype/device/shape.
        auto shape_vec = std::vector<int64_t>(primal.shape().begin(),
                                              primal.shape().end());
        return zeros(shape_vec, primal.dtype(), primal.device());
    };

    // 3. Forward-walk the topological order, dispatching each op's JVP.
    for (const auto& node : topo) {
        const auto& ivars = node->input_variables();
        const auto& nexts = node->next_functions();

        std::vector<Tensor> primals;
        std::vector<Tensor> tangents;
        primals.reserve(ivars.size());
        tangents.reserve(ivars.size());

        for (size_t i = 0; i < ivars.size(); ++i) {
            const auto& primal = ivars[i].tensor();
            std::shared_ptr<Function> prod;
            if (i < nexts.size()) prod = nexts[i];
            auto tang = resolve_input_tangent(prod, primal);
            if (!tang.is_valid()) {
                // Producer's tangent missing — bail.
                return std::nullopt;
            }
            primals.push_back(primal);
            tangents.push_back(std::move(tang));
        }

        OpId op = node->op_id();
        OpAttributes attrs = node->saved_attributes();

        // A.4 multi-output walker integration. If the Function provides a
        // `jvp_pack_inputs_for_walker` override, repack the (primals,
        // tangents) into the JVP rule's expected contract. This lets ops
        // like LayerNormBackward (which only carries `input` in
        // input_variables but whose JVP rule needs (x, gamma, beta))
        // surface their saved gamma/beta tensors without us inventing a
        // separate per-op input_variables convention.
        if (auto packed = node->jvp_pack_inputs_for_walker(primals, tangents)) {
            primals  = std::move(packed->first);
            tangents = std::move(packed->second);
        }

        try {
            // Prefer the multi-output rule if registered.
            if (has_jvp_rule_multi(op)) {
                auto result = dispatch_jvp_multi(op,
                                                 std::span<const Tensor>(primals),
                                                 std::span<const Tensor>(tangents),
                                                 attrs);
                if (result.tangents.empty()) return std::nullopt;
                for (size_t k = 0; k < result.tangents.size(); ++k) {
                    node_tangents_multi.emplace(std::make_pair(node.get(), k),
                                                 result.tangents[k]);
                }
                // Primary (output-0) tangent shared with the single-output
                // table so the canonical resolution path still works.
                node_tangents.emplace(node.get(), std::move(result.tangents[0]));
            } else {
                auto result = dispatch_jvp(op,
                                           std::span<const Tensor>(primals),
                                           std::span<const Tensor>(tangents),
                                           attrs);
                node_tangents.emplace(node.get(), std::move(result.tangent));
            }
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }

    // 4. The output tangent is the tangent computed for the root Function.
    //    For multi-output roots the root grad_fn is shared across all output
    //    Variables (e.g. EighBackward is the grad_fn of both W and V), so we
    //    must recover WHICH output slot `func` actually returned. We do this
    //    exactly like consumer-input resolution: match the returned output's
    //    data_ptr against the root Function's saved tensors and map the
    //    matching saved-slot to its output index via
    //    `jvp_saved_tensor_to_output_idx`. When the matched slot is non-zero
    //    we return that slot's tangent from `node_tangents_multi`; otherwise
    //    output-0 (the canonical single-output table) is correct.
    if (output_data_ptr != nullptr) {
        const auto& root_saved =
            const_cast<Function*>(out_grad_fn.get())->saved_tensors();
        for (size_t i = 0; i < root_saved.size(); ++i) {
            if (root_saved[i].data_ptr() == output_data_ptr) {
                size_t out_idx =
                    out_grad_fn->jvp_saved_tensor_to_output_idx(i);
                if (out_idx != 0) {
                    auto it_m =
                        node_tangents_multi.find({out_grad_fn.get(), out_idx});
                    if (it_m != node_tangents_multi.end()) return it_m->second;
                    // Root is a multi-output op and `func` returned a non-zero
                    // slot, but that slot's tangent was not published. Returning
                    // the output-0 tangent here would be a silent wrong-output
                    // result, so bail to the finite-difference fallback instead.
                    return std::nullopt;
                }
                break;  // matched output-0; canonical table below is correct.
            }
        }
    }
    auto it = node_tangents.find(out_grad_fn.get());
    if (it == node_tangents.end()) return std::nullopt;
    return it->second;
}

}  // anonymous namespace

auto jvp(std::function<Variable(const Variable&)> func,
         const Variable& input,
         const Tensor& tangent,
         JvpMode mode) -> std::pair<Variable, Tensor> {
    // Forward-mode AD via the registered dispatch_jvp rules when possible,
    // with a finite-difference fallback for ops/compositions not yet covered.
    //
    // Strategy (A.4 audit item, multi-op graph traversal):
    //
    //   1. Evaluate primal output: `output = func(input)`. This builds the
    //      backward graph as a side effect, giving us `output.grad_fn()`.
    //
    //   2. Fast path — walk the grad_fn graph in reverse-topological order
    //      (producers before consumers). For each node, look up its OpId,
    //      reconstruct its OpAttributes via `Function::saved_attributes()`,
    //      resolve input tangents from previously-computed node tangents
    //      (or the seed tangent / a zero tangent for leaf inputs), and
    //      call `dispatch_jvp`. The output tangent of the chain's root
    //      Function is the tangent of `func`'s output.
    //
    //   3. Fallback — central finite differences. If *any* node in the
    //      chain has no registered JVP rule or its rule rejects the call,
    //      the entire walk aborts and we fall back to FD for the whole
    //      function. No partial / "stitched" walks (per project no-
    //      workaround policy).
    //
    // JvpMode::Dual additionally raises the `is_dual_mode()` TLS flag for
    // the duration of `func(input)`. Per-op Variable interceptors (a
    // follow-up; see jvp_dispatch.hpp) can read that flag to route directly
    // through `dual_apply<>`. With no interceptors wired yet the behaviour
    // is identical to JvpMode::Walker because the walker is still the
    // source of truth once `func(input)` returns.

    std::optional<DualModeGuard> dual_guard;
    if (mode == JvpMode::Dual) {
        dual_guard.emplace();  // sets is_dual_mode()=true for this scope
    }

    auto output = func(input);

    // ---- Fast path: walk the autograd graph -------------------------------
    if (auto grad_fn = output.grad_fn()) {
        const void* user_ptr = input.tensor().data_ptr();
        const void* output_ptr = output.tensor().data_ptr();
        if (auto tangent_opt = try_traverse_jvp(grad_fn, user_ptr, tangent, output_ptr)) {
            return {output, std::move(*tangent_opt)};
        }
    }

    // ---- Fallback: central finite differences ----------------------------
    static std::once_flag warned_flag;
    std::call_once(warned_flag, [] {
        std::fprintf(stderr,
            "[autograd::jvp] WARNING: falling back to finite-difference JVP. "
            "The op chain in `func` has no registered forward-mode rule "
            "(see jvp_dispatch.hpp). Results are numerically approximate.\n");
    });

    const double eps = 1e-4;

    // Half-precision eps=1e-4 is below representable resolution — the
    // perturbation quantises to zero and the tangent collapses to ~0/garbage.
    // Widen the FD step to Float32 and cast the tangent back at the end,
    // mirroring hvp/vhp. The primal `output` was already computed above at the
    // caller's native dtype, so it is returned unchanged.
    const DType orig_dtype = input.tensor().dtype();
    const bool widen = (orig_dtype == DType::Float16 ||
                        orig_dtype == DType::BFloat16);

    Tensor probe_input = widen ? input.tensor().to(DType::Float32) : input.tensor();
    Tensor probe_tangent = widen ? tangent.to(DType::Float32) : tangent;

    auto perturbed_data = tenzor::add(probe_input, tenzor::mul(probe_tangent, eps));
    Variable perturbed_input(perturbed_data, false);
    auto perturbed_output_fwd = func(perturbed_input);

    auto perturbed_data_bwd = tenzor::sub(probe_input, tenzor::mul(probe_tangent, eps));
    Variable perturbed_input_bwd(perturbed_data_bwd, false);
    auto perturbed_output_bwd = func(perturbed_input_bwd);

    auto tangent_output = tenzor::mul(
        tenzor::sub(perturbed_output_fwd.tensor(), perturbed_output_bwd.tensor()),
        1.0 / (2.0 * eps)
    );

    if (widen) tangent_output = tangent_output.to(orig_dtype);

    return {output, tangent_output};
}

auto jacobian(std::function<Variable(const Variable&)> func,
              const Variable& input) -> Tensor {
    auto input_data = input.tensor();
    int64_t n = input_data.numel();  // input size

    // Evaluate once to get output size
    auto output = func(input);
    int64_t m = output.tensor().numel();  // output size

    if (n <= m) {
        // Forward-mode: iterate over input basis vectors
        // Each JVP gives one column of the Jacobian
        std::vector<Tensor> columns;
        columns.reserve(n);

        auto flat_input = tenzor::reshape(input_data, {n});

        for (int64_t i = 0; i < n; ++i) {
            // Create standard basis vector e_i. Build in Float32 on CPU (so we
            // can always use data<float>()), then cast to the input dtype and
            // device. This avoids hard-coding data<float>() on a tensor of an
            // arbitrary dtype (Float64/Float16/BFloat16 would dtype-mismatch).
            auto e_i_cpu = tenzor::zeros({n}, DType::Float32, Device::cpu());
            e_i_cpu.data<float>()[i] = 1.0f;
            auto e_i = e_i_cpu.to(input_data.dtype()).to(input_data.device());
            e_i = tenzor::reshape(e_i, std::vector<int64_t>(input_data.shape().begin(), input_data.shape().end()));

            auto [_, jvp_col] = jvp(func, input, e_i);
            columns.push_back(tenzor::reshape(jvp_col, {m}));
        }

        // Stack columns -> (n, m), then transpose to (m, n)
        auto J = tenzor::stack(std::span<const Tensor>(columns), 0);  // (n, m)
        return tenzor::transpose(J, 0, 1);  // (m, n)
    } else {
        // Reverse-mode: iterate over output basis vectors
        // Each backward pass gives one row of the Jacobian
        std::vector<Tensor> rows;
        rows.reserve(m);

        for (int64_t j = 0; j < m; ++j) {
            // Re-evaluate to get fresh computation graph
            Variable inp(input_data, true);
            auto out = func(inp);

            // Create gradient vector e_j. Use Float32 for the construction
            // (so data<float>() is valid) and cast to the input dtype after.
            auto e_j_cpu = tenzor::zeros({m}, DType::Float32, Device::cpu());
            e_j_cpu.data<float>()[j] = 1.0f;
            auto e_j = e_j_cpu.to(input_data.dtype()).to(input_data.device());
            e_j = tenzor::reshape(e_j, std::vector<int64_t>(out.tensor().shape().begin(), out.tensor().shape().end()));

            // Backward with e_j (shaped like out) to get row j of Jacobian.
            out.backward(e_j, /*retain_graph=*/false, /*create_graph=*/false);

            if (inp.grad().has_value()) {
                rows.push_back(tenzor::reshape(inp.grad().value(), {n}));
            } else {
                rows.push_back(tenzor::zeros({n}, input_data.dtype(), input_data.device()));
            }
        }

        // Stack rows -> (m, n)
        return tenzor::stack(std::span<const Tensor>(rows), 0);
    }
}

auto hessian(std::function<Variable(const Variable&)> func,
             const Variable& input) -> Tensor {
    auto input_data = input.tensor();
    int64_t n = input_data.numel();

    // Forward-over-reverse: compute Jacobian of the gradient function.
    // The gradient function: x -> grad(f(x), x)
    auto grad_func = [&func, n](const Variable& x) -> Variable {
        Variable x_grad(x.tensor(), true);
        auto out = func(x_grad);
        out.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/false);

        Tensor grad_val;
        if (x_grad.grad().has_value()) {
            grad_val = x_grad.grad().value();
        } else {
            grad_val = tenzor::zeros_like(x.tensor());
        }

        return Variable(grad_val, false);
    };

    // Compute the Jacobian of the gradient function -> Hessian
    Variable inp(input_data, false);
    return jacobian(grad_func, inp);
}

auto hvp(std::function<Variable(const Variable&)> func,
         const Variable& input,
         const Tensor& v) -> std::pair<Variable, Tensor> {
    // Forward-over-reverse: compute JVP of the gradient function with tangent v.
    // This avoids materializing the full Hessian matrix.

    // 1. Compute the primal output
    auto output = func(input);

    // 2. Define the gradient function: x -> grad(f(x), x)
    auto input_data = input.tensor();

    // Float16 / BFloat16 finite-difference step (jvp uses eps=1e-4) is
    // below the dtype's representable resolution, so perturbations get
    // quantised to zero and the Hessian-vector product collapses to
    // noise. Widen the probe to Float32 and narrow the result back.
    const DType orig_dtype = input_data.dtype();
    const bool widen = (orig_dtype == DType::Float16 ||
                        orig_dtype == DType::BFloat16);

    // When widening, evaluate `func` at Float32 precision throughout —
    // the finite-difference step probes how f varies and needs real
    // sensitivity. If we cast back to Float16 inside func, the perturbed
    // inputs quantise to the same value as the base point and the
    // derivative collapses to zero. Callers who genuinely need the
    // reduced-precision forward semantics can invoke hvp with a Float32
    // probe directly.
    auto grad_func = [&func](const Variable& x) -> Variable {
        Variable x_grad(x.tensor(), true);
        auto out = func(x_grad);
        out.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/false);

        Tensor grad_val;
        if (x_grad.grad().has_value()) {
            grad_val = x_grad.grad().value();
        } else {
            grad_val = tenzor::zeros_like(x.tensor());
        }

        return Variable(grad_val, false);
    };

    Tensor probe_input = widen ? input_data.to(DType::Float32) : input_data;
    Tensor probe_v = widen ? v.to(DType::Float32) : v;

    // 3. Compute JVP of the gradient function with tangent v -> H @ v
    Variable inp(probe_input, false);
    auto [_, hvp_result] = jvp(grad_func, inp, probe_v);

    if (widen) hvp_result = hvp_result.to(orig_dtype);
    return {output, hvp_result};
}

auto vhp(std::function<Variable(const Variable&)> func,
         const Variable& input,
         const Tensor& v) -> std::pair<Variable, Tensor> {
    // Reverse-over-reverse: compute how the gradient changes in direction v.
    // v^T @ H = d/dt [grad(f, x + t*v)] at t=0
    //
    // We use central differences on the gradient function (consistent with
    // the JVP implementation): grad(f, x+eps*v) - grad(f, x-eps*v) / (2*eps).
    // For symmetric Hessians this equals H @ v, but the approach is formally
    // the VHP (reverse-mode differentiation of the gradient).

    auto input_data = input.tensor();

    // Compute the primal output
    auto output = func(input);

    // Half-precision eps=1e-4 is below representable resolution — the
    // perturbation quantises to zero and vhp_result ≈ 0/0. Widen to
    // Float32 for the finite-difference step and cast back at the end.
    const DType orig_dtype = input_data.dtype();
    const bool widen = (orig_dtype == DType::Float16 ||
                        orig_dtype == DType::BFloat16);

    Tensor probe_input = widen ? input_data.to(DType::Float32) : input_data;
    Tensor probe_v = widen ? v.to(DType::Float32) : v;

    const double eps = 1e-4;

    // Evaluate gradient at x + eps*v (at probe precision — see hvp above
    // for why we don't cast back to the caller's dtype inside `func`).
    auto perturbed_fwd = tenzor::add(probe_input, tenzor::mul(probe_v, eps));
    Variable x_fwd(perturbed_fwd, true);
    auto out_fwd = func(x_fwd);
    out_fwd.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/false);
    Tensor grad_fwd = x_fwd.grad().has_value()
        ? x_fwd.grad().value()
        : tenzor::zeros_like(probe_input);

    // Evaluate gradient at x - eps*v
    auto perturbed_bwd = tenzor::sub(probe_input, tenzor::mul(probe_v, eps));
    Variable x_bwd(perturbed_bwd, true);
    auto out_bwd = func(x_bwd);
    out_bwd.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/false);
    Tensor grad_bwd = x_bwd.grad().has_value()
        ? x_bwd.grad().value()
        : tenzor::zeros_like(probe_input);

    // Central difference
    auto vhp_result = tenzor::mul(
        tenzor::sub(grad_fwd, grad_bwd),
        1.0 / (2.0 * eps)
    );

    if (widen) vhp_result = vhp_result.to(orig_dtype);
    return {output, vhp_result};
}

auto vjp(std::function<Variable(const Variable&)> func,
         const Variable& input,
         const Tensor& cotangent) -> std::pair<Variable, Tensor> {
    // Create a new variable that requires grad, detached from any existing graph
    Variable x(input.tensor().clone(), /*requires_grad=*/true);

    // Forward pass
    auto output = func(x);

    // Backward pass with cotangent as upstream gradient
    output.backward(cotangent, /*retain_graph=*/false, /*create_graph=*/false);

    Tensor vjp_result = x.grad().has_value()
        ? x.grad().value()
        : tenzor::zeros_like(input.tensor());

    return {output, vjp_result};
}

} // namespace tenzor
