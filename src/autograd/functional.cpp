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
// M25: thread-local signal so callers can programmatically detect that the
// most recent jvp() call degraded to a finite-difference approximation
// (previously only an stderr print — invisible to code, not just humans).
thread_local bool g_jvp_used_fd_fallback{false};
}  // namespace

auto jvp_used_fd_fallback() -> bool {
    return g_jvp_used_fd_fallback;
}

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

    // Iterative post-order DFS with an explicit stack (not recursion) so a deep
    // graph — e.g. a many-layer Hessian/JVP probe — cannot overflow the C++
    // stack, matching BackwardEngine::topological_sort. Each frame tracks how
    // many of its children have been expanded; the node is post-order visited
    // once all children are done. Any node we can't dispatch through
    // (Unknown op, or neither a single- nor multi-output JVP rule) rejects the
    // whole traversal (no-workaround contract), exactly as the recursion did.
    {
        struct Frame { std::shared_ptr<Function> node; size_t child; };
        std::vector<Frame> stack;
        visited.insert(out_grad_fn.get());
        stack.push_back({out_grad_fn, 0});
        while (!stack.empty()) {
            Frame& f = stack.back();
            const auto& nexts = f.node->next_functions();
            if (f.child < nexts.size()) {
                const auto& next = nexts[f.child++];
                if (next && visited.insert(next.get()).second) {
                    stack.push_back({next, 0});
                }
                continue;
            }
            OpId op = f.node->op_id();
            if (op == OpId::Unknown ||
                (!has_jvp_rule(op) && !has_jvp_rule_multi(op))) {
                return std::nullopt;
            }
            topo.push_back(f.node);
            stack.pop_back();
        }
    }
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
            g_jvp_used_fd_fallback = false;
            return {output, std::move(*tangent_opt)};
        }
    }

    // M25: mark this call as FD-degraded before doing any of the fallback
    // work below, so jvp_used_fd_fallback() is correct even if a caller
    // inspects it from within a nested/recursive jvp() invocation.
    g_jvp_used_fd_fallback = true;

    // ---- Fallback: central finite differences ----------------------------
    // Warn on EVERY fallback (not once-per-process): the previous std::once
    // flag meant a single early warning could scroll out of the logs and every
    // later silently-approximate JVP — including gradcheck-style verification
    // via jacobian() — looked exact. A per-call warning makes the degradation
    // visible at the call site that triggers it. The fallback is meant to be
    // rare (some op in `func` lacks a registered forward-mode rule), so this
    // does not flood logs in the analytic-rule-complete common case.
    std::fprintf(stderr,
        "[autograd::jvp] WARNING: falling back to finite-difference JVP for "
        "this call. An op in `func` has no registered forward-mode rule "
        "(see jvp_dispatch.hpp); the result is a numerical approximation, not "
        "an analytic JVP.\n");

    const DType orig_dtype = input.tensor().dtype();
    const bool widen = (orig_dtype == DType::Float16 ||
                        orig_dtype == DType::BFloat16);

    // Half-precision eps=1e-4 is below representable resolution near O(1), so a
    // probe cast back to half would quantise the perturbation to zero. Stage the
    // perturbation in Float32 with a half-representable step (1e-2, matching
    // gradcheck's half eps floor), but hand `func` a probe at the input's NATIVE
    // dtype — `func` may capture other operands at that dtype and reject a
    // widened Float32 probe, mirroring numerical_gradient's make_probe. The
    // output difference is widened to Float32 to limit cancellation and the
    // final tangent is narrowed back to the native dtype.
    const double eps = widen ? 1e-2 : 1e-4;

    Tensor probe_input = widen ? input.tensor().to(DType::Float32) : input.tensor();
    Tensor probe_tangent = widen ? tangent.to(DType::Float32) : tangent;

    auto perturbed_data = tenzor::add(probe_input, tenzor::mul(probe_tangent, eps));
    Variable perturbed_input(widen ? perturbed_data.to(orig_dtype) : perturbed_data, false);
    auto perturbed_output_fwd = func(perturbed_input);

    auto perturbed_data_bwd = tenzor::sub(probe_input, tenzor::mul(probe_tangent, eps));
    Variable perturbed_input_bwd(widen ? perturbed_data_bwd.to(orig_dtype) : perturbed_data_bwd, false);
    auto perturbed_output_bwd = func(perturbed_input_bwd);

    Tensor out_fwd = perturbed_output_fwd.tensor();
    Tensor out_bwd = perturbed_output_bwd.tensor();
    if (widen) {
        out_fwd = out_fwd.to(DType::Float32);
        out_bwd = out_bwd.to(DType::Float32);
    }
    auto tangent_output = tenzor::mul(
        tenzor::sub(out_fwd, out_bwd),
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
    // Hessian by columns: column i is the Hessian-vector product H·e_i, computed
    // by `hvp` as an EXACT analytic forward-over-reverse double-backward (no
    // finite differences). H is symmetric, so stacking the n column products
    // yields the Hessian.
    auto input_data = input.tensor();
    int64_t n = input_data.numel();
    const DType dtype = input_data.dtype();
    const Device device = input_data.device();

    std::vector<Tensor> columns;
    columns.reserve(static_cast<size_t>(n));

    for (int64_t i = 0; i < n; ++i) {
        // Basis vector e_i, shaped like the input. Build in Float32 on CPU so
        // data<float>() is always valid, then cast to the input dtype/device.
        auto e_i_cpu = tenzor::zeros({n}, DType::Float32, Device::cpu());
        e_i_cpu.data<float>()[i] = 1.0f;
        auto e_i = e_i_cpu.to(dtype).to(device);
        e_i = tenzor::reshape(
            e_i, std::vector<int64_t>(input_data.shape().begin(),
                                      input_data.shape().end()));

        auto [_, col] = hvp(func, input, e_i);
        columns.push_back(tenzor::reshape(col, {n}));
    }

    // Stack columns -> (n, n). Symmetric, so this is the Hessian directly.
    return tenzor::stack(std::span<const Tensor>(columns), 0);
}

auto hvp(std::function<Variable(const Variable&)> func,
         const Variable& input,
         const Tensor& v) -> std::pair<Variable, Tensor> {
    // Exact Hessian-vector product H·v = d/dt[ grad(f, x + t·v) ]|_{t=0} via
    // forward-over-reverse: differentiate (forward-mode / JVP) the reverse-mode
    // gradient function.
    //
    //   grad_func(x) := grad_x( sum f(x) )   built with create_graph=true so the
    //                    returned gradient Variable carries a grad_fn (the full
    //                    second-order graph). The engine now keeps that graph
    //                    connected across every Function hand-off (parallel
    //                    Variable accumulator), so the gradient of x*x*x is the
    //                    true 3x² graph in which all occurrences of x are linked.
    //
    //   H·v = jvp(grad_func, x, v).second
    //
    // The jvp walker accumulates the seed tangent across every occurrence of x
    // in that second-order graph (forward-mode chain rule sums at each Add /
    // each multi-use leaf), so this is the exact analytic H·v — e.g. for x³ the
    // second derivative is the exact 6x, not the 4x the previously-severed graph
    // produced. jvp() internally falls back to a (high-precision Float64) finite
    // difference ONLY if some op in grad_func has no registered forward-mode
    // rule; for the analytic-rule-complete common case no finite differencing
    // happens at all.
    auto output = func(input);

    const DType orig_dtype = input.tensor().dtype();
    const Device orig_device = input.tensor().device();

    // Half precision: widen the whole forward-over-reverse computation to
    // Float32 (Float16/BFloat16 cannot represent the perturbations / accumulate
    // the second-order graph accurately), cast the result back at the end.
    const bool widen = (orig_dtype == DType::Float16 ||
                        orig_dtype == DType::BFloat16);
    const DType work_dtype = widen ? DType::Float32 : orig_dtype;

    Tensor x_work = input.tensor();
    if (x_work.dtype() != work_dtype) x_work = x_work.to(work_dtype);
    Tensor v_work = (v.dtype() != work_dtype) ? v.to(work_dtype) : v;

    Variable x_var(x_work, /*requires_grad=*/true);

    // grad_func(p) := grad_p( sum func(p) ), graph-carrying (create_graph=true).
    auto grad_func = [&func](const Variable& p) -> Variable {
        // Retain input Variables in every forward op of `func` so the
        // subsequent create_graph backward can build the second-order graph
        // THROUGH saved intermediates (e.g. the x² inside d/dx(x³)). Without
        // this the saved operands are detached and forward-over-reverse drops
        // the dependence-through-saved-tensor contributions.
        HigherOrderGraphRetentionGuard retain_guard;
        // Fresh leaf for each evaluation so its grad_fn / accumulators are clean.
        Variable pv(p.tensor(), /*requires_grad=*/true);
        Variable out = func(pv);
        Variable scalar = (out.tensor().numel() == 1) ? out : tenzor::sum(out);
        // create_graph=true: the engine builds the connected second-order graph.
        scalar.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
        const auto& gv = pv.grad_variable();
        if (gv) {
            // H2: pv is a fresh, single-use leaf discarded at the end of this
            // call. Its grad_with_graph_impl_ strongly references gv's graph,
            // which (via save_variables_for_backward retaining live Variables
            // under HigherOrderGraphRetentionGuard) strongly references pv
            // right back — a shared_ptr cycle that would otherwise leak pv's
            // VariableImpl, and its device tensor storage, forever on every
            // hvp()/vhp()/hessian() call. Take our own independent copy of
            // the graph-connected gradient first, then zero_grad() to clear
            // pv's own back-reference to it — result already holds what we
            // need, so nothing is lost.
            Variable result = *gv;
            pv.zero_grad();
            return result;
        }
        // Output does not depend on p — gradient is zero everywhere.
        return Variable(tenzor::zeros_like(pv.tensor()), false);
    };

    // Forward-mode differentiate the gradient at x with seed v -> exact H·v.
    auto [_, hv] = jvp(grad_func, x_var, v_work);

    Tensor hvp_result = (hv.dtype() != orig_dtype) ? hv.to(orig_dtype) : hv;
    if (hvp_result.device() != orig_device) hvp_result = hvp_result.to(orig_device);
    return {output, hvp_result};
}

auto vhp(std::function<Variable(const Variable&)> func,
         const Variable& input,
         const Tensor& v) -> std::pair<Variable, Tensor> {
    // v^T·H for a scalar f. The Hessian of a scalar function is symmetric, so
    // v^T·H == H·v, and we compute it with the SAME exact forward-over-reverse
    // double-backward as hvp (analytic — no finite differences). For the
    // intended scalar-valued `func` this is exact; for a (non-standard)
    // vector-valued `func` the Hessian-of-grad is generally non-symmetric and
    // v^T·H != H·v — vhp on such inputs is not well-defined and is not
    // supported (use jacobian/vjp of the gradient function explicitly).
    return hvp(std::move(func), input, v);
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
