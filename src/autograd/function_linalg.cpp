#include "tenzor/autograd/function.hpp"
#include <cassert>
#include "tenzor/autograd/ops.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/fft.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/utils/safe_math.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <typeinfo>
#include <unordered_set>
#ifdef __GNUC__
#include <cxxabi.h>
#include <cstdlib>
#endif

namespace tenzor {

namespace {

// The backward engine collapses all grads accumulated on a function into a
// single entry in `grad_outputs`, so a tuple-returning linalg op whose caller
// only differentiates one of its outputs (e.g. `sum(S)` from `svd`) will
// arrive here with grad_outputs.size() == 1 while the math expects one slot
// per tuple component. This helper fills the missing slots with zeros shaped
// like the corresponding saved forward output and routes the single incoming
// grad to the slot whose shape matches it. If no shape matches, the grad is
// placed in `default_slot` (used by slogdet, whose two outputs have the same
// shape but whose sign gradient is mathematically zero anyway).
//
// When the engine grows real per-output-slot accumulation this helper can
// collapse to a plain index operation. For now it keeps gradcheck on tuple-
// output linalg ops honest.
inline auto pad_tuple_grad_outputs(
        std::vector<Tensor> grad_outputs,
        const std::vector<Tensor>& slot_prototypes,
        size_t default_slot = 0) -> std::vector<Tensor> {
    const size_t expected = slot_prototypes.size();
    if (grad_outputs.size() == expected) return grad_outputs;

    std::vector<Tensor> out;
    out.reserve(expected);
    for (size_t i = 0; i < expected; ++i) {
        const auto& proto = slot_prototypes[i];
        out.push_back(zeros(std::vector<int64_t>(proto.shape().begin(), proto.shape().end()),
                             proto.dtype(), proto.device()));
    }
    if (grad_outputs.size() == 1) {
        const auto& g = grad_outputs.front();
        auto matches = [&](const Tensor& proto) {
            if (proto.ndim() != g.ndim()) return false;
            for (int64_t d = 0; d < g.ndim(); ++d) {
                if (proto.shape()[d] != g.shape()[d]) return false;
            }
            return true;
        };
        // Prefer the default slot when its prototype matches the grad's
        // shape — important for ops like Slogdet where multiple output
        // slots have the same shape (sign and logabsdet are both scalars
        // of identical layout). The previous "first matching slot" search
        // misrouted the grad to slot 0 (sign, mathematically zero-grad)
        // and wiped the real gradient on slot 1 (logabsdet).
        size_t slot = default_slot;
        bool found = false;
        if (default_slot < expected && matches(slot_prototypes[default_slot])) {
            found = true;
        } else {
            for (size_t i = 0; i < expected; ++i) {
                if (matches(slot_prototypes[i])) { slot = i; found = true; break; }
            }
        }
        if (!found && g.numel() != slot_prototypes[slot].numel()) {
            // Mismatched shape and no prototype matches — fall through with
            // zeros; the backward will produce a zero input-gradient which
            // is the correct answer in the "no gradient flowed" sense.
            return out;
        }
        out[slot] = g;
    }
    return out;
}

} // anonymous namespace

// =========================================================================
// Linear Algebra Backward Functions
// =========================================================================

// DetBackward implementation
// Forward: y = det(A)
// Backward: dL/dA = dL/dy * det(A) * A^{-T}
auto DetBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("DetBackward::forward should not be called directly");
}

auto DetBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];          // dL/dy, scalar or (batch,)
    const auto& det_val = saved_tensors_[0];      // det(A), same shape as grad
    const auto& inv_A = saved_tensors_[1];        // A^{-1}, (..., N, N)

    auto ndim = inv_A.ndim();
    auto inv_At = transpose(inv_A, ndim - 2, ndim - 1);
    auto grad_det = mul(grad, det_val);

    // Build a broadcast-compatible shape with two trailing singleton dims,
    // but only if the result carries batch dimensions; for a plain 2-D
    // matrix the det is a 1-element tensor that must collapse back to ()
    // so grad_A keeps the input's (N, N) rank.
    std::vector<int64_t> gd_shape;
    int64_t expected_batch_rank = ndim - 2;  // 0 for a plain matrix
    if (expected_batch_rank == 0) {
        gd_shape = {1, 1};  // broadcasts to (N, N) against inv_At
    } else {
        gd_shape.assign(grad_det.shape().begin(), grad_det.shape().end());
        gd_shape.push_back(1);
        gd_shape.push_back(1);
    }
    auto grad_det_expanded = reshape(grad_det, gd_shape);

    auto grad_A = mul(grad_det_expanded, inv_At);
    return {grad_A};
}

// InvBackward implementation
// Forward: Y = A^{-1}
// Backward: dL/dA = -Y^T @ dL/dY @ Y^T
auto InvBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("InvBackward::forward should not be called directly");
}

auto InvBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];       // dL/dY, (..., N, N)
    const auto& inv_A = saved_tensors_[0];    // Y = A^{-1}, (..., N, N)

    auto ndim = inv_A.ndim();
    auto inv_At = transpose(inv_A, ndim - 2, ndim - 1);

    // -Y^T @ dL/dY @ Y^T
    auto temp = matmul(inv_At, grad);
    auto result = matmul(temp, inv_At);
    auto grad_A = neg(result);

    return {grad_A};
}

// SolveBackward implementation
// Forward: X = solve(A, B) where AX = B
// Backward:
//   dL/dB = solve(A^T, dL/dX)
//   dL/dA = -dL/dB @ X^T
auto SolveBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SolveBackward::forward should not be called directly");
}

auto SolveBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];    // dL/dX, (..., N, K)
    const auto& A = saved_tensors_[0];     // A, (..., N, N)
    const auto& X = saved_tensors_[1];     // solution X, (..., N, K)

    auto ndim = A.ndim();
    auto At = transpose(A, ndim - 2, ndim - 1);

    // dL/dB = solve(A^T, dL/dX)
    auto grad_B = tenzor::linalg::solve(At, grad);

    // dL/dA = -grad_B @ X^T
    auto x_ndim = X.ndim();
    auto Xt = transpose(X, x_ndim - 2, x_ndim - 1);
    auto grad_A = neg(matmul(grad_B, Xt));

    return {grad_A, grad_B};
}

// CholeskyBackward implementation
// Forward: L = cholesky(A)  where A = L @ L^T
// Backward: Uses the formula from Murray 2016 / PyTorch:
//   S = L^T @ dL/dL
//   S = tril(S) with diagonal halved: phi(S)
//   dL/dA = L^{-T} @ phi(S + S^T) @ L^{-1}
//   Simplified: dL/dA = solve(L^T, phi(L^T @ grad_L)) then symmetrize
auto CholeskyBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CholeskyBackward::forward should not be called directly");
}

auto CholeskyBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    auto grad_L = grad_outputs[0];          // dL/dL, (..., N, N)
    const auto& L = saved_tensors_[0];      // Cholesky factor, (..., N, N)

    if (upper_) {
        // If upper triangular was returned, transpose to work with lower
        auto ndim = L.ndim();
        grad_L = transpose(grad_L, ndim - 2, ndim - 1);
        // L is actually U, transpose it
        auto L_lower = transpose(L, ndim - 2, ndim - 1);

        auto Lt = transpose(L_lower, ndim - 2, ndim - 1);
        auto S = matmul(Lt, grad_L);

        // phi(S): take lower triangle and halve diagonal
        auto S_tril = tril(S);
        auto S_strict_lower = tril(S, -1);
        // phi(S): tril(S) with diagonal halved
        auto phi_S = add(S_strict_lower, mul(sub(S_tril, S_strict_lower), 0.5));

        // dL/dA = L^{-T} @ phi_S @ L^{-1}
        // = solve(L^T, phi_S) then solve(L, result^T)^T
        // Simpler: use solve twice
        auto temp = tenzor::linalg::solve(Lt, phi_S);
        auto grad_A = tenzor::linalg::solve(Lt, transpose(temp, ndim - 2, ndim - 1));
        grad_A = transpose(grad_A, ndim - 2, ndim - 1);

        // Symmetrize: dL/dA = 0.5 * (dL/dA + dL/dA^T)
        auto grad_At = transpose(grad_A, ndim - 2, ndim - 1);
        grad_A = mul(add(grad_A, grad_At), 0.5);

        return {grad_A};
    }

    auto ndim = L.ndim();
    auto Lt = transpose(L, ndim - 2, ndim - 1);

    // S = L^T @ grad_L
    auto S = matmul(Lt, grad_L);

    // phi(S): tril(S) with diagonal halved
    auto S_tril = tril(S);
    auto S_strict_lower = tril(S, -1);
    auto phi_S = add(S_strict_lower, mul(sub(S_tril, S_strict_lower), 0.5));

    // dL/dA = L^{-T} @ phi_S @ L^{-1}
    auto temp = tenzor::linalg::solve(Lt, phi_S);
    auto grad_A = tenzor::linalg::solve(Lt, transpose(temp, ndim - 2, ndim - 1));
    grad_A = transpose(grad_A, ndim - 2, ndim - 1);

    // Symmetrize: dL/dA = 0.5 * (dL/dA + dL/dA^T)
    auto grad_At = transpose(grad_A, ndim - 2, ndim - 1);
    grad_A = mul(add(grad_A, grad_At), 0.5);

    return {grad_A};
}

// NormBackward_Linalg implementation
// Forward: y = norm(A, ord)
// Backward (Frobenius): dL/dA = dL/dy * A / norm(A)
auto NormBackward_Linalg::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("NormBackward_Linalg::forward should not be called directly");
}

auto NormBackward_Linalg::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];       // dL/dy
    const auto& input = saved_tensors_[0];    // A
    const auto& norm_val = saved_tensors_[1]; // norm(A)

    if (ord_ == "fro") {
        // dL/dA = dL/dy * A / norm(A)
        auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
        auto scale = div(grad, norm_val);
        auto scale_shape = std::vector<int64_t>(scale.shape().begin(), scale.shape().end());
        while (scale_shape.size() < input_shape.size()) {
            scale_shape.push_back(1);
        }
        auto scale_expanded = reshape(scale, scale_shape);
        return {mul(scale_expanded, input)};
    }

    if (ord_ == "nuc") {
        // Nuclear norm: y = sum_i σ_i(A). Gradient is U @ V^H (∂/∂A Σ σ_i).
        auto [U, S, Vh] = tenzor::linalg::svd(input, /*full_matrices=*/false);
        auto outer = matmul(U, Vh);  // (..., M, N)
        auto grad_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
        while (grad_shape.size() < static_cast<size_t>(outer.ndim())) {
            grad_shape.push_back(1);
        }
        auto grad_reshaped = reshape(grad, grad_shape);
        return {mul(grad_reshaped, outer)};
    }

    // Induced / spectral norms — same math as LinalgMatrixNormBackward, kept
    // inline here so this node owns its full backward and doesn't depend on
    // a separate function's saved-tensors layout.
    auto induced_grad = [&](double ord_d) -> Tensor {
        if (std::abs(ord_d - 2.0) < 1e-10 || std::abs(ord_d + 2.0) < 1e-10) {
            // Spectral norm (ord = ±2): grad = u_k v_k^H.
            auto [U, S, Vh] = tenzor::linalg::svd(input, /*full_matrices=*/false);
            const int64_t U_ndim  = U.ndim();
            const int64_t Vh_ndim = Vh.ndim();
            const int64_t K = U.size(U_ndim - 1);
            const bool use_smallest = (ord_d < 0.0);
            const int64_t sv_idx = use_smallest ? K - 1 : 0;

            auto uk  = tenzor::slice(U,  /*dim=*/U_ndim - 1,  /*start=*/sv_idx, /*end=*/sv_idx + 1);
            auto vkh = tenzor::slice(Vh, /*dim=*/Vh_ndim - 2, /*start=*/sv_idx, /*end=*/sv_idx + 1);
            auto outer = matmul(uk, vkh);

            auto gs = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
            while (gs.size() < static_cast<size_t>(outer.ndim())) gs.push_back(1);
            auto grad_reshaped = reshape(grad, gs);
            return mul(grad_reshaped, outer);
        }

        // Induced 1 / inf norms (and their negatives): argmax/argmin of
        // column / row absolute sums, then sign mask.
        const int64_t ndim    = input.ndim();
        const int64_t M       = input.size(ndim - 2);
        const int64_t N       = input.size(ndim - 1);
        const bool col_norm   = (std::abs(ord_d) == 1.0);
        const int64_t reduce_dim  = col_norm ? ndim - 2 : ndim - 1;
        const int64_t num_classes = col_norm ? N : M;
        const bool use_min    = (ord_d < 0.0);
        (void)M;

        auto abs_A = tenzor::abs(input);
        auto sums  = tenzor::sum(abs_A, /*dim=*/reduce_dim, /*keepdim=*/false);
        auto idx = use_min ? tenzor::argmin(sums, /*dim=*/-1, /*keepdim=*/false)
                           : tenzor::argmax(sums, /*dim=*/-1, /*keepdim=*/false);
        auto mask = tenzor::one_hot(idx, num_classes).to(input.dtype());
        // Reshape the one-hot mask to its explicit broadcast target (the matrix
        // dim that was reduced becomes 1). This is rank-robust: argmax/one_hot
        // keepdim conventions differ across backends (CUDA argmax of a 1-D tensor
        // yields shape (1,) vs CPU's scalar), which previously left a stray leading
        // dim -> [1,M,N] gradient / [1,1,N] broadcast failures.
        std::vector<int64_t> mask_shape(input.shape().begin(), input.shape().end());
        mask_shape[col_norm ? (ndim - 2) : (ndim - 1)] = 1;
        mask = reshape(mask, mask_shape);

        auto gs = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
        while (gs.size() < static_cast<size_t>(ndim)) gs.push_back(1);
        auto grad_reshaped = reshape(grad, gs);
        return mul(mul(grad_reshaped, mask), tenzor::sign(input));
    };

    if (ord_ == "1")    return {induced_grad(1.0)};
    if (ord_ == "-1")   return {induced_grad(-1.0)};
    if (ord_ == "2")    return {induced_grad(2.0)};
    if (ord_ == "-2")   return {induced_grad(-2.0)};
    if (ord_ == "inf")  return {induced_grad(std::numeric_limits<double>::infinity())};
    if (ord_ == "-inf") return {induced_grad(-std::numeric_limits<double>::infinity())};

    throw std::runtime_error(
        "NormBackward_Linalg::backward: unsupported norm order '" + ord_ +
        "'. Supported: 'fro', 'nuc', '1', '-1', '2', '-2', 'inf', '-inf'.");
}

// SlogdetBackward implementation
// Forward: (sign, logabsdet) = slogdet(A)
// Backward: dL/dA = dL/d(logabsdet) * A^{-T}
// sign gradient is zero (discrete)
auto SlogdetBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SlogdetBackward::forward should not be called directly");
}

auto SlogdetBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Slogdet returns (sign, logabsdet) with identical shapes. A lone incoming
    // grad is assumed to be grad_logabsdet since sign is piecewise-constant
    // (mathematically zero gradient). saved_tensors_[0] carries A^{-1}, not
    // the forward outputs, so we reconstruct prototypes from A^{-1}'s leading
    // batch shape.
    const auto& inv_A = saved_tensors_[0];     // A^{-1}, (..., N, N)
    std::vector<int64_t> out_shape(inv_A.shape().begin(), inv_A.shape().end() - 2);
    auto proto = zeros(out_shape, inv_A.dtype(), inv_A.device());
    grad_outputs = pad_tuple_grad_outputs(std::move(grad_outputs), {proto, proto}, /*default_slot=*/1);
    const auto& grad_logabsdet = grad_outputs[1];

    auto ndim = inv_A.ndim();
    auto inv_At = transpose(inv_A, ndim - 2, ndim - 1);

    // Reshape grad_logabsdet to broadcast with inv_At
    auto gd_shape = std::vector<int64_t>(grad_logabsdet.shape().begin(), grad_logabsdet.shape().end());
    while (gd_shape.size() < static_cast<size_t>(ndim)) {
        gd_shape.push_back(1);
    }
    auto grad_expanded = reshape(grad_logabsdet, gd_shape);

    auto grad_A = mul(grad_expanded, inv_At);
    return {grad_A};
}

// SvdBackward implementation
// Forward: (U, S, Vh) = svd(A)
// Backward: complex formula using F matrix
// For A = U @ diag(S) @ Vh:
//   F_{ij} = 1/(s_j^2 - s_i^2) for i != j, 0 on diagonal
//   dL/dA = U @ (diag(dL/dS) + (F * (U^T @ dL/dU - dL/dVh^T @ Vh^T @ diag(S))) @ ... ) @ Vh
// Simplified approach for thin SVD:
auto SvdBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SvdBackward::forward should not be called directly");
}

auto SvdBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& U = saved_tensors_[0];        // U, (..., M, K)
    const auto& S = saved_tensors_[1];        // S, (..., K)
    const auto& Vh = saved_tensors_[2];       // Vh, (..., K, N)

    // audit-2026-05-03 — when output_slot_ is set, only one of {U, S, Vh}
    // received an upstream gradient (engine collapse safe). Re-route the
    // single grad to its proper slot.
    Tensor grad_U_t, grad_S_t, grad_Vh_t;
    if (output_slot_ == 0) {
        grad_U_t = grad_outputs[0];
        grad_S_t = zeros_like(S);
        grad_Vh_t = zeros_like(Vh);
    } else if (output_slot_ == 1) {
        grad_U_t = zeros_like(U);
        grad_S_t = grad_outputs[0];
        grad_Vh_t = zeros_like(Vh);
    } else if (output_slot_ == 2) {
        grad_U_t = zeros_like(U);
        grad_S_t = zeros_like(S);
        grad_Vh_t = grad_outputs[0];
    } else {
        // Legacy combined path.
        grad_outputs = pad_tuple_grad_outputs(std::move(grad_outputs), {U, S, Vh});
        grad_U_t = grad_outputs[0];
        grad_S_t = grad_outputs[1];
        grad_Vh_t = grad_outputs[2];
    }
    const auto& grad_U = grad_U_t;
    const auto& grad_S = grad_S_t;
    const auto& grad_Vh = grad_Vh_t;

    auto ndim = U.ndim();
    auto K = S.shape()[S.ndim() - 1];

    // Construct diagonal matrix from S
    auto S_diag = diag(S);  // (..., K, K)

    // U^T and V (= Vh^T)
    auto Ut = transpose(U, ndim - 2, ndim - 1);
    auto V = transpose(Vh, Vh.ndim() - 2, Vh.ndim() - 1);

    // Construct F matrix: F_{ij} = 1/(s_j^2 - s_i^2) for i != j
    // F is K x K
    auto S_sq = mul(S, S);  // (..., K)

    // B.8: batch-aware F matrix.
    // For S shaped (..., K), build F shaped (..., K, K) with
    //   F[..., i, j] = 1 / (s_j^2 - s_i^2)   for i != j
    //   F[..., i, i] = 0
    // Using unsqueeze + broadcast preserves all leading batch dims so
    // higher-order autograd through batched SVD works correctly.
    // S_row varies over j (last dim of the (..., 1, K) broadcast), S_col
    // varies over i (last-but-one dim of the (..., K, 1) broadcast).
    auto S_row = unsqueeze(S_sq, S_sq.ndim() - 1);   // (..., 1, K)  varies over j
    auto S_col = unsqueeze(S_sq, S_sq.ndim());       // (..., K, 1)  varies over i
    auto diffs = sub(S_row, S_col);                  // diffs[..., i, j] = s_j^2 - s_i^2

    // Clamp small |s_j^2 - s_i^2| to epsilon for near-degenerate singular
    // values. eps_t/mask shapes broadcast against the batched diffs.
    auto eps_val   = detail::dtype_epsilon(S.dtype());
    auto eps_t     = full({K, K}, eps_val, S.dtype(), S.device());
    auto abs_diffs = abs(diffs);
    auto safe_diffs = where(lt(abs_diffs, eps_t), mul(sign(diffs), eps_t), diffs);

    // Replace diagonal with 1 to avoid division by zero (broadcasts over batch).
    auto mask = eye(K, std::nullopt, S.dtype(), S.device());
    safe_diffs = add(safe_diffs, mask);

    // F = 1/safe_diffs, then zero out diagonal
    auto F = reciprocal(safe_diffs);
    auto anti_mask = sub(ones({K, K}, S.dtype(), S.device()), mask);
    F = mul(F, anti_mask);  // Zero out diagonal

    // Core SVD backward formula:
    // dL/dA = U @ (diag(dL/dS) + F * (Ut @ dL/dU) @ S_diag + S_diag @ F * (dL/dVh @ V)) @ Vh
    //       + (I - U @ Ut) @ dL/dU @ diag(1/S) @ Vh
    //       + U @ diag(1/S) @ dL/dVh @ (I - V @ Vt)

    // Simplified formula for full-rank case:
    // dA = U @ (diag(grad_S) + F * (Ut @ grad_U - (grad_Vh @ V)^T) @ S_diag) @ Vh
    //    + ... (projector terms for non-square)

    // Compute Ut @ grad_U
    auto UtgU = matmul(Ut, grad_U);  // (K, K)

    // Compute grad_Vh @ V
    auto gVhV = matmul(grad_Vh, V);  // (K, K)
    auto gVhVt = transpose(gVhV, gVhV.ndim() - 2, gVhV.ndim() - 1);

    // Symmetric part: F * (Ut @ grad_U - (grad_Vh @ V)^T) = F * (UtgU - gVhVt)
    auto skew = sub(UtgU, gVhVt);
    auto F_skew = mul(F, skew);  // (K, K)

    // F_skew @ S_diag
    auto term = matmul(F_skew, S_diag);

    // Add S_diag @ F * (gVhVt - UtgU)^T ... actually let's use the symmetric formulation
    // The correct formula from Ionescu et al. 2015:
    // dA = U @ (diag(grad_S) + (F * (Ut@gU)) @ S + S @ (F * (gVh@V))) @ Vh

    // audit-2026-05-03 — corrected SVD backward formula. The U-derivative
    // and V-derivative contributions go through the ANTISYMMETRIC part of
    // U^T grad_U and grad_Vh V (the symmetric parts are absorbed by the
    // U^T U = I and V^T V = I constraints). The previous formulation used
    // F ⊙ (U^T grad_U) directly, which mixed the symmetric (constrained)
    // component into the gradient and broke gradcheck.
    // Antisymmetric components:
    //   skew_U = U^T grad_U - (U^T grad_U)^T
    //   skew_V = V^T grad_V - (V^T grad_V)^T = (grad_Vh V)^T - (grad_Vh V) = gVhVt - gVhV
    // (Because Vh = V^T, grad_V = grad_Vh^T, so V^T grad_V = (grad_Vh V)^T.)
    auto UtgU_t = transpose(UtgU, UtgU.ndim() - 2, UtgU.ndim() - 1);
    auto skew_U = sub(UtgU, UtgU_t);
    auto skew_V = sub(gVhVt, gVhV);

    auto F_UtgU = mul(F, skew_U);
    auto F_gVhV = mul(F, skew_V);

    auto term1 = diag(grad_S);             // diag(grad_S), (K, K)
    auto term2 = matmul(F_UtgU, S_diag);   // F*(skew_U) @ S
    auto term3 = matmul(S_diag, F_gVhV);   // S @ F*(skew_V)

    auto middle = add(add(term1, term2), term3);  // (K, K)

    auto grad_A = matmul(matmul(U, middle), Vh);  // (..., M, N)

    // audit-2026-05-03 — projector terms for non-square U, V (full_matrices
    // = false with M ≠ N). For tall U (M > K): add (I - U U^T) · grad_U · S^{-1} · V^T.
    // For wide V (N > K): add U · S^{-1} · grad_Vh · (I - V V^T).
    int64_t M = U.shape()[ndim - 2];
    int64_t N = Vh.shape()[Vh.ndim() - 1];
    auto S_inv = reciprocal(S);
    auto S_inv_diag = diag(S_inv);
    if (M > K) {
        // (I - U U^T) is (M, M)
        auto eye_M = eye(M, std::nullopt, U.dtype(), U.device());
        auto UUt = matmul(U, Ut);                    // (M, M)
        auto proj_M = sub(eye_M, UUt);               // (M, M)
        // proj_M @ grad_U @ S^{-1} @ Vh
        auto extra = matmul(matmul(matmul(proj_M, grad_U), S_inv_diag), Vh);
        grad_A = add(grad_A, extra);
    }
    if (N > K) {
        // (I - V V^T) is (N, N)
        auto eye_N = eye(N, std::nullopt, U.dtype(), U.device());
        auto VVt = matmul(V, transpose(V, V.ndim() - 2, V.ndim() - 1));
        auto proj_N = sub(eye_N, VVt);
        auto extra = matmul(matmul(matmul(U, S_inv_diag), grad_Vh), proj_N);
        grad_A = add(grad_A, extra);
    }

    return {grad_A};
}

// QrBackward implementation
// Forward: (Q, R) = qr(A) where A = Q @ R
// Backward formula (from Seeger et al.):
//   M = R @ grad_R^T - grad_Q^T @ Q
//   copyltu(M) = tril(M) + tril(M, -1)^T   (symmetrize lower triangle)
//   dL/dA = (grad_Q + Q @ copyltu(M)) @ R^{-T}
auto QrBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("QrBackward::forward should not be called directly");
}

auto QrBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Force contiguous layout on the saved forwards and incoming grads: the
    // tuple-padding helper above fabricates zero-tensors, and any caller may
    // have produced a non-contiguous view of grad_R. The backward formula
    // below assumes row-major element access via matmul; a non-contiguous
    // grad_R would be re-interpreted in column-major and yield a transposed
    // analytical gradient (the "ana[j,i] = num[i,j]" failure mode seen in
    // gradcheck when we hand it a non-contiguous input).
    const auto Q = saved_tensors_[0].contiguous();   // Q, (..., M, N)
    const auto R = saved_tensors_[1].contiguous();   // R, (..., N, N)
    grad_outputs = pad_tuple_grad_outputs(std::move(grad_outputs), {Q, R});
    const auto grad_Q = grad_outputs[0].contiguous();   // dL/dQ, (..., M, N)
    const auto grad_R = grad_outputs[1].contiguous();   // dL/dR, (..., N, N)

    auto ndim = R.ndim();
    auto Rt = transpose(R, ndim - 2, ndim - 1);
    auto Qt = transpose(Q, Q.ndim() - 2, Q.ndim() - 1);

    // audit-2026-05-03 — corrected to PyTorch's convention:
    //   M = R · grad_R^T - grad_Q^T · Q
    // Previous code had `matmul(Qt, grad_Q)` = Q^T · grad_Q which is the
    // TRANSPOSE of grad_Q^T · Q — the formula's copyltu pattern then fired
    // on the wrong triangle, breaking the QR-Q gradcheck on every backend.
    auto grad_Rt = transpose(grad_R, ndim - 2, ndim - 1);
    auto grad_Qt = transpose(grad_Q, grad_Q.ndim() - 2, grad_Q.ndim() - 1);
    auto M = sub(matmul(R, grad_Rt), matmul(grad_Qt, Q));

    // copyltu(M) = tril(M) + tril(M, -1)^T
    auto M_tril = tril(M);
    auto M_strict_lower = tril(M, -1);
    auto M_strict_lower_t = transpose(M_strict_lower, ndim - 2, ndim - 1);
    auto copyltu_M = add(M_tril, M_strict_lower_t);

    // dL/dA = (grad_Q + Q @ copyltu_M) @ R^{-T}
    auto Q_copyltu = matmul(Q, copyltu_M);
    auto rhs = add(grad_Q, Q_copyltu);

    // Solve R^T @ X^T = rhs^T  =>  X = (R^{-T} @ rhs^T)^T...
    // Actually: rhs @ R^{-T} = solve(R, rhs^T)^T
    auto rhs_t = transpose(rhs, rhs.ndim() - 2, rhs.ndim() - 1);
    auto solve_result = tenzor::linalg::solve(R, rhs_t);
    auto grad_A = transpose(solve_result, solve_result.ndim() - 2, solve_result.ndim() - 1);

    // Materialise a contiguous row-major view — `transpose` returns a strided
    // view, which gradcheck's element-wise pointer walk interprets as
    // column-major, producing a spurious shape-correct-but-values-transposed
    // comparison failure.
    return {grad_A.contiguous()};
}

// EighBackward implementation
// Forward: (W, V) = eigh(A) where A = V @ diag(W) @ V^T
// Backward:
//   F_{ij} = 1/(w_j - w_i) for i != j, 0 on diagonal
//   dL/dA = V @ (F * (V^T @ dL/dV) + diag(dL/dW)) @ V^T
auto EighBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("EighBackward::forward should not be called directly");
}

auto EighBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& W = saved_tensors_[0];      // eigenvalues, (..., N)
    const auto& V = saved_tensors_[1];      // eigenvectors, (..., N, N)
    grad_outputs = pad_tuple_grad_outputs(std::move(grad_outputs), {W, V});
    const auto& grad_W = grad_outputs[0];   // dL/dW, (..., N)
    const auto& grad_V = grad_outputs[1];   // dL/dV, (..., N, N)

    auto N = W.shape()[W.ndim() - 1];

    // B.8: batch-aware F matrix. For W shaped (..., N), build F shaped
    //   (..., N, N) via unsqueeze+broadcast: F[..., i, j] = 1/(w_j - w_i)
    //   for i != j, 0 on diagonal. This makes higher-order autograd
    //   through batched eigh produce correct gradients.
    // W_row varies over j (last dim of the (..., 1, N) broadcast); W_col
    // varies over i (last-but-one dim of the (..., N, 1) broadcast).
    auto W_row = unsqueeze(W, W.ndim() - 1);  // (..., 1, N)  varies over j
    auto W_col = unsqueeze(W, W.ndim());      // (..., N, 1)  varies over i
    auto diffs = sub(W_row, W_col);           // diffs[..., i, j] = w_j - w_i

    // Clamp small |w_j - w_i| to epsilon to avoid singularity from degenerate eigenvalues
    auto eps_val = detail::dtype_epsilon(W.dtype());
    auto eps_t = full({N, N}, eps_val, W.dtype(), W.device());
    auto abs_diffs = abs(diffs);
    auto safe_diffs = where(lt(abs_diffs, eps_t), mul(sign(diffs), eps_t), diffs);

    // Replace diagonal with 1 to avoid division by zero
    auto mask = eye(N, std::nullopt, W.dtype(), W.device());
    safe_diffs = add(safe_diffs, mask);

    auto F = reciprocal(safe_diffs);
    auto anti_mask = sub(ones({N, N}, W.dtype(), W.device()), mask);
    F = mul(F, anti_mask);  // Zero out diagonal

    auto Vt = transpose(V, V.ndim() - 2, V.ndim() - 1);

    // V^T @ dL/dV
    auto VtgV = matmul(Vt, grad_V);  // (N, N)

    // F * (V^T @ dL/dV) + diag(dL/dW)
    // GG.7: diag_embed is batch-aware (works for (..., N) grad_W); the
    // earlier diag(grad_W) threw "diag: input must be 1D or 2D" on batched
    // input. diag_embed(grad_W, 0, -2, -1) places grad_W on the main
    // diagonal of a (..., N, N) tensor for any batch shape.
    auto middle = add(mul(F, VtgV), linalg::diag_embed(grad_W, 0, -2, -1));  // (..., N, N)

    // dL/dA = V @ middle @ V^T
    auto grad_A = matmul(matmul(V, middle), Vt);

    // Symmetrize the result (since A is symmetric)
    auto grad_At = transpose(grad_A, grad_A.ndim() - 2, grad_A.ndim() - 1);
    grad_A = mul(add(grad_A, grad_At), 0.5);

    return {grad_A};
}

// EigvalshBackward implementation
// Forward: W = eigvalsh(A)
// Backward: dL/dA = V @ diag(dL/dW) @ V^T
//           where V was computed via eigh during forward
auto EigvalshBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("EigvalshBackward::forward should not be called directly");
}

auto EigvalshBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad_W = grad_outputs[0];   // dL/dW, (..., N)
    const auto& V = saved_tensors_[0];      // eigenvectors, (..., N, N)

    auto Vt = transpose(V, V.ndim() - 2, V.ndim() - 1);

    // dL/dA = V @ diag(dL/dW) @ V^T
    // GG.7: diag_embed is batch-aware; the earlier diag(grad_W) threw on
    // batched input (ndim > 2).
    auto grad_diag = linalg::diag_embed(grad_W, 0, -2, -1);  // (..., N, N)
    auto grad_A = matmul(matmul(V, grad_diag), Vt);

    // Symmetrize (since A is symmetric)
    auto grad_At = transpose(grad_A, grad_A.ndim() - 2, grad_A.ndim() - 1);
    grad_A = mul(add(grad_A, grad_At), 0.5);

    return {grad_A};
}

// ============================================================================
// SpMMBackward
// ============================================================================

auto SpMMBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("SpMMBackward::forward should not be called directly");
}

auto SpMMBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Y = S @ D  =>  grad_D = S^T @ grad_Y
    // sparse_transposed_ = S^T stored as SparseTensor (K, M)
    if (!sparse_transposed_.has_value()) {
        throw std::runtime_error("SpMMBackward: sparse_transposed_ not set");
    }
    const auto& grad_output = grad_outputs[0];  // shape (M, N)

    // grad_D = S^T @ grad_Y using sparse::spmm, shape (K, N)
    auto grad_dense = sparse::spmm(*sparse_transposed_, grad_output);

    return {grad_dense};
}

// ============================================================================
// SpMVBackward
// ============================================================================

auto SpMVBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("SpMVBackward::forward should not be called directly");
}

auto SpMVBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // y = S @ v  =>  grad_v = S^T @ grad_y
    // sparse_transposed_ = S^T stored as SparseTensor (K, M)
    if (!sparse_transposed_.has_value()) {
        throw std::runtime_error("SpMVBackward: sparse_transposed_ not set");
    }
    const auto& grad_y = grad_outputs[0];  // shape (M,)

    // grad_v = S^T @ grad_y using sparse::spmv, shape (K,)
    auto grad_v = sparse::spmv(*sparse_transposed_, grad_y);

    return {grad_v};
}

// ============================================================================
// Linalg backward_with_variables implementations (higher-order gradients)
// ============================================================================

auto DetBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dA = dL/dy * det(A) * A^{-T}
    // Use saved Variables if available, otherwise wrap saved Tensors
    Variable det_val, inv_A;
    if (has_saved_variables()) {
        require_saved_variables(2);
        det_val = saved_variables_[0];
        inv_A = saved_variables_[1];
    } else {
        require_saved_tensors(2);
        det_val = Variable(saved_tensors_[0], false);
        inv_A = Variable(saved_tensors_[1], false);
    }

    auto ndim = inv_A.tensor().ndim();
    auto inv_At = tenzor::transpose(inv_A, ndim - 2, ndim - 1);

    // grad * det(A) — expand for broadcasting with matrix shape
    auto grad_det = grad_outputs[0] * det_val;
    auto gd_shape = std::vector<int64_t>(grad_det.shape().begin(), grad_det.shape().end());
    gd_shape.push_back(1);
    gd_shape.push_back(1);
    auto grad_det_expanded = tenzor::reshape(grad_det, gd_shape);

    auto grad_A = grad_det_expanded * inv_At;
    return {grad_A};
}

auto InvBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dA = -Y^T @ dL/dY @ Y^T  where Y = A^{-1}
    Variable inv_A;
    if (has_saved_variables()) {
        require_saved_variables(1);
        inv_A = saved_variables_[0];
    } else {
        require_saved_tensors(1);
        inv_A = Variable(saved_tensors_[0], false);
    }

    auto ndim = inv_A.tensor().ndim();
    auto inv_At = tenzor::transpose(inv_A, ndim - 2, ndim - 1);

    auto temp = tenzor::matmul(inv_At, grad_outputs[0]);
    auto result = tenzor::matmul(temp, inv_At);
    auto grad_A = tenzor::neg(result);

    return {grad_A};
}

auto SolveBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dB = solve(A^T, dL/dX), dL/dA = -dL/dB @ X^T
    Variable A, X;
    if (has_saved_variables()) {
        require_saved_variables(2);
        A = saved_variables_[0];
        X = saved_variables_[1];
    } else {
        require_saved_tensors(2);
        A = Variable(saved_tensors_[0], false);
        X = Variable(saved_tensors_[1], false);
    }

    auto ndim = A.tensor().ndim();
    auto At = tenzor::transpose(A, ndim - 2, ndim - 1);

    auto grad_B = tenzor::solve(At, grad_outputs[0]);

    auto x_ndim = X.tensor().ndim();
    auto Xt = tenzor::transpose(X, x_ndim - 2, x_ndim - 1);
    auto grad_A = tenzor::neg(tenzor::matmul(grad_B, Xt));

    return {grad_A, grad_B};
}

auto NormBackward_Linalg::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    require_saved_tensors(2);
    const Tensor& input = saved_tensors_[0];
    const Tensor& norm_val = saved_tensors_[1];

    if (ord_ == "fro") {
        // Frobenius: dL/dA = dL/dy * A / norm(A) — graph-preserving form so
        // `create_graph=true` works (audit-9 R.q).
        auto scale = grad_outputs[0] * tenzor::reciprocal(Variable(norm_val, false));
        auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
        auto scale_shape = std::vector<int64_t>(scale.shape().begin(), scale.shape().end());
        while (scale_shape.size() < input_shape.size()) {
            scale_shape.push_back(1);
        }
        auto scale_expanded = tenzor::reshape(scale, scale_shape);
        return {scale_expanded * Variable(input, false)};
    }

    // For the remaining ords the closed-form gradient factors as
    //   grad_input = grad_lifted * deriv(A, ord)
    // where `deriv` depends only on saved tensors. Compute `deriv` at tensor
    // level by feeding a ones-grad through the tensor backward (same trick
    // as LinalgMatrixNormBackward::backward_with_variables), then compose the
    // final multiply at Variable level so grad_fn flows through grad_outputs.
    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // The norm result shape == norm_val.shape() (whatever batch dims exist).
    auto norm_shape = std::vector<int64_t>(norm_val.shape().begin(), norm_val.shape().end());
    Tensor ones_grad = tenzor::ones(norm_shape, input.dtype(), input.device());
    Tensor deriv = backward({ones_grad})[0];  // shape == input.shape()
    Variable deriv_var(deriv, /*requires_grad=*/false);

    // Lift grad_outputs[0] (shape norm_shape) to input.shape() by appending
    // trailing 1s then broadcasting; both ops preserve grad_fn.
    auto grad_lifted_shape = norm_shape;
    while (grad_lifted_shape.size() < input_shape.size()) {
        grad_lifted_shape.push_back(1);
    }
    Variable grad_reshaped = tenzor::reshape(grad_outputs[0], grad_lifted_shape);
    Variable grad_expanded = tenzor::expand(grad_reshaped, input_shape);

    return {grad_expanded * deriv_var};
}

auto SlogdetBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dA = dL/d(logabsdet) * A^{-T} (sign gradient is zero)
    Variable inv_A;
    if (has_saved_variables()) {
        require_saved_variables(1);
        inv_A = saved_variables_[0];
    } else {
        require_saved_tensors(1);
        inv_A = Variable(saved_tensors_[0], false);
    }

    auto ndim = inv_A.tensor().ndim();
    auto inv_At = tenzor::transpose(inv_A, ndim - 2, ndim - 1);

    auto gd_shape = std::vector<int64_t>(grad_outputs[1].shape().begin(), grad_outputs[1].shape().end());
    while (gd_shape.size() < static_cast<size_t>(ndim)) {
        gd_shape.push_back(1);
    }
    auto grad_expanded = tenzor::reshape(grad_outputs[1], gd_shape);

    return {grad_expanded * inv_At};
}

auto EigvalshBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dA = V @ diag(dL/dW) @ V^T, symmetrized
    // V is a saved tensor (constant), grad_W is a Variable for higher-order grad tracking
    const auto& V_tensor = saved_tensors_[0];  // eigenvectors, (..., N, N)
    auto V = Variable(V_tensor, false);

    auto ndim = V_tensor.ndim();
    auto Vt = tenzor::transpose(V, ndim - 2, ndim - 1);

    // Variable-level diag and matmul
    auto grad_diag = tenzor::diag(grad_outputs[0]);  // (N, N)
    auto grad_A = tenzor::matmul(tenzor::matmul(V, grad_diag), Vt);

    // Symmetrize (since A is symmetric)
    auto grad_At = tenzor::transpose(grad_A, ndim - 2, ndim - 1);
    grad_A = (grad_A + grad_At) * 0.5;

    return {grad_A};
}

auto CholeskyBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dA = L^{-T} @ phi(L^T @ grad_L) @ L^{-1}, symmetrized
    // where phi(S) = tril(S) with diagonal halved
    // Use Variable-level ops for higher-order gradient support

    auto grad_L_var = grad_outputs[0];
    const auto& L_tensor = saved_tensors_[0];
    auto L = Variable(L_tensor, false);

    auto ndim = L_tensor.ndim();

    if (upper_) {
        // If upper triangular was returned, transpose to work with lower
        grad_L_var = tenzor::transpose(grad_L_var, ndim - 2, ndim - 1);
        auto L_lower = tenzor::transpose(L, ndim - 2, ndim - 1);

        auto Lt = tenzor::transpose(L_lower, ndim - 2, ndim - 1);

        // S = L^T @ grad_L  (Variable-level matmul)
        auto S = tenzor::matmul(Lt, grad_L_var);

        // phi(S): tril(S) with diagonal halved
        // phi(S) = tril(S, -1) + 0.5 * diag(diag(S))
        // Equivalently: phi(S) = tril(S) - 0.5 * (tril(S) - tril(S, -1))
        auto S_tril = tenzor::tril(S);
        auto S_strict_lower = tenzor::tril(S, -1);
        auto phi_S = S_strict_lower + (S_tril - S_strict_lower) * 0.5;

        // grad_A = L^{-T} @ phi_S @ L^{-1}
        // = solve(L^T, phi_S), then solve(L^T, result^T)^T
        auto temp = tenzor::solve(Lt, phi_S);
        auto grad_A = tenzor::solve(Lt, tenzor::transpose(temp, ndim - 2, ndim - 1));
        grad_A = tenzor::transpose(grad_A, ndim - 2, ndim - 1);

        // Symmetrize
        auto grad_At = tenzor::transpose(grad_A, ndim - 2, ndim - 1);
        grad_A = (grad_A + grad_At) * 0.5;

        return {grad_A};
    }

    auto Lt = tenzor::transpose(L, ndim - 2, ndim - 1);

    // S = L^T @ grad_L
    auto S = tenzor::matmul(Lt, grad_L_var);

    // phi(S): tril(S) with diagonal halved
    auto S_tril = tenzor::tril(S);
    auto S_strict_lower = tenzor::tril(S, -1);
    auto phi_S = S_strict_lower + (S_tril - S_strict_lower) * 0.5;

    // grad_A = L^{-T} @ phi_S @ L^{-1}
    auto temp = tenzor::solve(Lt, phi_S);
    auto grad_A = tenzor::solve(Lt, tenzor::transpose(temp, ndim - 2, ndim - 1));
    grad_A = tenzor::transpose(grad_A, ndim - 2, ndim - 1);

    // Symmetrize
    auto grad_At = tenzor::transpose(grad_A, ndim - 2, ndim - 1);
    grad_A = (grad_A + grad_At) * 0.5;

    return {grad_A};
}

auto SvdBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // SVD backward using Variable-level ops for higher-order gradient support.
    // Formula from Ionescu et al. 2015:
    //   dA = U @ (diag(grad_S) + (F * (Ut@gU)) @ S + S @ (F * (gVh@V))) @ Vh
    // where F_{ij} = 1/(s_j^2 - s_i^2) for i != j, 0 on diagonal

    auto grad_U_var = grad_outputs[0];
    auto grad_S_var = grad_outputs[1];
    auto grad_Vh_var = grad_outputs[2];

    // Wrap saved tensors as non-grad Variables
    auto U = Variable(saved_tensors_[0], false);   // (..., M, K)
    auto S = Variable(saved_tensors_[1], false);   // (..., K)
    auto Vh = Variable(saved_tensors_[2], false);  // (..., K, N)

    auto ndim = saved_tensors_[0].ndim();
    auto K = saved_tensors_[1].shape()[saved_tensors_[1].ndim() - 1];

    // Compute F matrix from singular values (constant — no grad needed)
    // F_{ij} = 1/(s_j^2 - s_i^2) for i != j, 0 on diagonal
    const auto& S_tensor = saved_tensors_[1];
    auto S_sq = mul(S_tensor, S_tensor);
    auto S_row = reshape(S_sq, {1, K});
    auto S_col = reshape(S_sq, {K, 1});
    auto diffs = sub(S_row, S_col);

    auto eps_val = detail::dtype_epsilon(S_tensor.dtype());
    auto eps_t = full({K, K}, eps_val, S_tensor.dtype(), S_tensor.device());
    auto abs_diffs = abs(diffs);
    auto safe_diffs = where(lt(abs_diffs, eps_t), mul(sign(diffs), eps_t), diffs);

    auto mask = eye(K, std::nullopt, S_tensor.dtype(), S_tensor.device());
    safe_diffs = add(safe_diffs, mask);
    auto F_tensor = reciprocal(safe_diffs);
    auto anti_mask = sub(ones({K, K}, S_tensor.dtype(), S_tensor.device()), mask);
    F_tensor = mul(F_tensor, anti_mask);

    auto F = Variable(F_tensor, false);

    // S_diag as Variable
    auto S_diag = tenzor::diag(S);  // (..., K, K)

    // U^T and V
    auto Ut = tenzor::transpose(U, ndim - 2, ndim - 1);
    auto V = tenzor::transpose(Vh, saved_tensors_[2].ndim() - 2, saved_tensors_[2].ndim() - 1);

    // Variable-level computations
    auto UtgU = tenzor::matmul(Ut, grad_U_var);       // (K, K)
    auto gVhV = tenzor::matmul(grad_Vh_var, V);       // (K, K)

    auto F_UtgU = F * UtgU;
    auto F_gVhV = F * gVhV;

    auto term1 = tenzor::diag(grad_S_var);             // diag(grad_S), (K, K)
    auto term2 = tenzor::matmul(F_UtgU, S_diag);       // F*(Ut@gU) @ S
    auto term3 = tenzor::matmul(S_diag, F_gVhV);       // S @ F*(gVh@V)

    auto middle = term1 + term2 + term3;

    auto grad_A = tenzor::matmul(tenzor::matmul(U, middle), Vh);

    return {grad_A};
}

auto QrBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // QR backward using Variable-level ops for higher-order gradient support.
    // Formula from Seeger et al.:
    //   M = R @ grad_R^T - grad_Q^T @ Q
    //   copyltu(M) = tril(M) + tril(M, -1)^T
    //   dL/dA = (grad_Q + Q @ copyltu(M)) @ R^{-T}

    auto grad_Q_var = grad_outputs[0];
    auto grad_R_var = grad_outputs[1];

    auto Q = Variable(saved_tensors_[0], false);
    auto R = Variable(saved_tensors_[1], false);

    auto ndim = saved_tensors_[1].ndim();
    auto Rt = tenzor::transpose(R, ndim - 2, ndim - 1);
    auto Qt = tenzor::transpose(Q, saved_tensors_[0].ndim() - 2, saved_tensors_[0].ndim() - 1);

    // M = R @ grad_R^T - grad_Q^T @ Q
    auto grad_Rt_var = tenzor::transpose(grad_R_var, ndim - 2, ndim - 1);
    auto M = tenzor::matmul(R, grad_Rt_var) - tenzor::matmul(Qt, grad_Q_var);

    // copyltu(M) = tril(M) + tril(M, -1)^T
    auto M_tril = tenzor::tril(M);
    auto M_strict_lower = tenzor::tril(M, -1);
    auto M_strict_lower_t = tenzor::transpose(M_strict_lower, ndim - 2, ndim - 1);
    auto copyltu_M = M_tril + M_strict_lower_t;

    // dL/dA = (grad_Q + Q @ copyltu_M) @ R^{-T}
    auto Q_copyltu = tenzor::matmul(Q, copyltu_M);
    auto rhs = grad_Q_var + Q_copyltu;

    // rhs @ R^{-T} = solve(R, rhs^T)^T
    auto rhs_t = tenzor::transpose(rhs, rhs.tensor().ndim() - 2, rhs.tensor().ndim() - 1);
    auto solve_result = tenzor::solve(R, rhs_t);
    auto grad_A = tenzor::transpose(solve_result, solve_result.tensor().ndim() - 2, solve_result.tensor().ndim() - 1);

    return {grad_A};
}

auto EighBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Eigh backward using Variable-level ops for higher-order gradient support.
    // Formula: dL/dA = V @ (F * (V^T @ dL/dV) + diag(dL/dW)) @ V^T, symmetrized
    // F_{ij} = 1/(w_j - w_i) for i != j, 0 on diagonal

    auto grad_W_var = grad_outputs[0];
    auto grad_V_var = grad_outputs[1];

    const auto& W_tensor = saved_tensors_[0];  // eigenvalues, (..., N)
    const auto& V_tensor = saved_tensors_[1];  // eigenvectors, (..., N, N)

    auto V = Variable(V_tensor, false);
    auto N = W_tensor.shape()[W_tensor.ndim() - 1];

    // Compute F matrix (constant — from eigenvalues)
    auto W_row = reshape(W_tensor, {1, N});
    auto W_col = reshape(W_tensor, {N, 1});
    auto diffs = sub(W_row, W_col);

    auto eps_val = detail::dtype_epsilon(W_tensor.dtype());
    auto eps_t = full({N, N}, eps_val, W_tensor.dtype(), W_tensor.device());
    auto abs_diffs = abs(diffs);
    auto safe_diffs = where(lt(abs_diffs, eps_t), mul(sign(diffs), eps_t), diffs);

    auto mask = eye(N, std::nullopt, W_tensor.dtype(), W_tensor.device());
    safe_diffs = add(safe_diffs, mask);
    auto F_tensor = reciprocal(safe_diffs);
    auto anti_mask = sub(ones({N, N}, W_tensor.dtype(), W_tensor.device()), mask);
    F_tensor = mul(F_tensor, anti_mask);

    auto F = Variable(F_tensor, false);

    auto ndim = V_tensor.ndim();
    auto Vt = tenzor::transpose(V, ndim - 2, ndim - 1);

    // V^T @ dL/dV (Variable-level)
    auto VtgV = tenzor::matmul(Vt, grad_V_var);

    // F * (V^T @ dL/dV) + diag(dL/dW)
    auto middle = F * VtgV + tenzor::diag(grad_W_var);

    // dL/dA = V @ middle @ V^T
    auto grad_A = tenzor::matmul(tenzor::matmul(V, middle), Vt);

    // Symmetrize
    auto grad_At = tenzor::transpose(grad_A, ndim - 2, ndim - 1);
    grad_A = (grad_A + grad_At) * 0.5;

    return {grad_A};
}

// LUBackward implementation
// Forward: (L, U, pivots) = lu(A) where P @ L @ U = A
// Backward: grad_A = P^T @ (tril(grad_L, -1) @ U + L @ triu(grad_U))
// The gradient must respect the constraints: L is unit lower triangular, U is upper triangular.
// grad for pivots is not computed (discrete, non-differentiable).
auto LUBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LUBackward::forward should not be called directly");
}

auto LUBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // audit-2026-05-03 — full closed-form LU backward.
    //
    // Forward (with partial pivoting): PA = LU where L is unit lower
    // triangular and U is upper triangular.
    // Differential: dA = P^T (dL · U + L · dU).
    // Adjoint:
    //   Φ = tril(L^T grad_L, -1) + triu(grad_U U^T, 0)
    //   grad_A = P^T · L^{-T} · Φ · U^{-T}
    // (Derivation: Walter "Structured Matrix Differentiation" / Giles
    //  matrix-derivative cookbook. Verified against numerical 2x2.)
    //
    // output_slot_ identifies which of {L=0, U=1} this instance handles;
    // the engine collapses per-output grads otherwise.
    require_saved_tensors(2);
    const auto& L = saved_tensors_[0];
    const auto& U = saved_tensors_[1];

    Tensor grad_L, grad_U;
    if (output_slot_ == 0) {
        grad_L = grad_outputs[0];
        grad_U = zeros_like(U);
    } else if (output_slot_ == 1) {
        grad_L = zeros_like(L);
        grad_U = grad_outputs[0];
    } else {
        // Legacy combined path (output_slot == -1): pad to 2 entries.
        while (grad_outputs.size() < 2) {
            grad_outputs.push_back(zeros_like(saved_tensors_[grad_outputs.size()]));
        }
        grad_L = grad_outputs[0];
        grad_U = grad_outputs[1];
    }


    int64_t n = L.shape().back();
    auto eye_n = tenzor::eye(n, std::nullopt, L.dtype(), L.device());

    // Φ = tril(L^T grad_L, -1) + triu(grad_U U^T, 0)
    auto LT = transpose(L, -1, -2);
    auto UT = transpose(U, -1, -2);
    auto phi_lower = tril(matmul(LT, grad_L), -1);
    auto phi_upper = triu(matmul(grad_U, UT), 0);
    auto phi = add(phi_lower, phi_upper);

    // L^{-1} and U^{-1} via triangular solves. The Vulkan solve_triangular
    // shader for unitriangular=true Float64 has a heap-corrupting bug
    // (audit-2026-05-03 — investigated separately); compute inverses on CPU
    // and move back when on Vulkan.
    auto run_solve = [&](const Tensor& T, bool upper, bool uni) {
        if (T.device().type == Device::Type::Vulkan) {
            auto T_cpu = T.to(Device::cpu()).contiguous();
            auto eye_cpu = ::tenzor::eye(n, std::nullopt, T.dtype(), Device::cpu());
            auto inv_cpu = tenzor::linalg::solve_triangular(T_cpu, eye_cpu, upper, uni);
            return inv_cpu.to(T.device());
        }
        return tenzor::linalg::solve_triangular(T, eye_n, upper, uni);
    };
    auto L_inv = run_solve(L, /*upper=*/false, /*unitriangular=*/true);
    auto LT_inv = transpose(L_inv, -1, -2);
    auto U_inv = run_solve(U, /*upper=*/true, /*unitriangular=*/false);
    auto UT_inv = transpose(U_inv, -1, -2);

    auto grad_A = matmul(matmul(LT_inv, phi), UT_inv);

    // Apply P^T if pivots were saved. The pivot encoding differs across
    // backends: LAPACK / CPU return 1-indexed values in [1..n]; Vulkan's
    // runBlockedLU writes 0-indexed values in [0..n-1]. Detect by max value
    // and normalise to 0-based row indices.
    if (saved_tensors_.size() > 2) {
        const auto& pivots = saved_tensors_[2];
        Tensor pivots_cpu = pivots.to(Device::cpu()).contiguous();
        if (pivots_cpu.ndim() == 1 && pivots_cpu.dtype() == DType::Int32) {
            const int32_t* piv_data = pivots_cpu.data<int32_t>();
            int64_t M = pivots_cpu.shape()[0];

            // Detect convention: if any value equals M, it's 1-indexed.
            bool one_indexed = false;
            for (int64_t i = 0; i < M; ++i) {
                if (piv_data[i] >= static_cast<int32_t>(M)) { one_indexed = true; break; }
                if (piv_data[i] < 0) { one_indexed = true; break; }
            }
            std::vector<int32_t> piv0(M);
            for (int64_t i = 0; i < M; ++i) {
                piv0[i] = one_indexed ? (piv_data[i] - 1) : piv_data[i];
            }

            bool identity = true;
            for (int64_t i = 0; i < M; ++i) {
                if (piv0[i] != static_cast<int32_t>(i)) { identity = false; break; }
            }
            if (!identity) {
                Tensor g_cpu = grad_A.to(Device::cpu()).contiguous();
                int64_t Nc = g_cpu.shape().back();

                auto do_swap = [&](auto* mat) {
                    for (int64_t i = M - 1; i >= 0; --i) {
                        int64_t target = static_cast<int64_t>(piv0[i]);
                        if (target != i) {
                            for (int64_t k = 0; k < Nc; ++k) {
                                std::swap(mat[i * Nc + k], mat[target * Nc + k]);
                            }
                        }
                    }
                };
                if (g_cpu.dtype() == DType::Float64) {
                    do_swap(g_cpu.data<double>());
                } else {
                    do_swap(g_cpu.data<float>());
                }
                grad_A = g_cpu.to(L.device());
            }
        }
    }

    return {grad_A};
}

auto LUBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable L, U;
    if (has_saved_variables()) {
        require_saved_variables(2);
        L = saved_variables_[0];
        U = saved_variables_[1];
    } else {
        require_saved_tensors(2);
        L = Variable(saved_tensors_[0], false);
        U = Variable(saved_tensors_[1], false);
    }

    // Strictly lower part of grad_L (L has unit diagonal, not differentiable there)
    auto grad_L_strict = tenzor::tril(grad_outputs[0], -1);
    auto grad_U_upper = tenzor::triu(grad_outputs[1]);

    auto term1 = tenzor::matmul(grad_L_strict, U);
    auto term2 = tenzor::matmul(L, grad_U_upper);
    auto grad_A = term1 + term2;

    return {grad_A};
}

// ============================================================================
// CholeskySolveBackward
// ============================================================================
// Forward: X = cholesky_solve(B, L) = L^{-T} L^{-1} B   (lower)
// Backward:
//   grad_B = cholesky_solve(grad_X, L)
//   grad_L = -tril(L^{-T} @ S @ L^{-1})
//     where S = grad_X @ X^T + X @ grad_X^T  (symmetrized)

auto CholeskySolveBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CholeskySolveBackward::forward should not be called directly");
}

auto CholeskySolveBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad_X = grad_outputs[0];   // dLoss/dX, (..., N, K)
    const auto& X = saved_tensors_[0];      // solution X, (..., N, K)
    const auto& L = saved_tensors_[1];      // Cholesky factor, (..., N, N)

    auto ndim = X.ndim();

    // grad_B = cholesky_solve(grad_X, L, upper)
    auto grad_B = tenzor::linalg::cholesky_solve(grad_X, L, upper_);

    // grad_L: S = grad_X @ X^T + X @ grad_X^T (symmetrized outer product)
    auto Xt = transpose(X, ndim - 2, ndim - 1);
    auto grad_Xt = transpose(grad_X, ndim - 2, ndim - 1);
    auto S = add(matmul(grad_X, Xt), matmul(X, grad_Xt));

    // grad_L = -tril(L^{-T} @ S @ L^{-1})
    // For lower: L^{-1} @ S via solve_triangular(L, S, lower)
    //            then L^{-T} @ result via solve_triangular(L^T, result, upper)
    auto L_ndim = L.ndim();
    if (!upper_) {
        auto temp = tenzor::linalg::solve_triangular(L, S, false);
        auto Lt = transpose(L, L_ndim - 2, L_ndim - 1);
        auto grad_L_full = tenzor::linalg::solve_triangular(Lt, temp, true);
        auto grad_L = neg(tril(grad_L_full));
        return {grad_B, grad_L};
    } else {
        auto Ut = transpose(L, L_ndim - 2, L_ndim - 1);
        auto temp = tenzor::linalg::solve_triangular(Ut, S, false);
        auto grad_U_full = tenzor::linalg::solve_triangular(L, temp, true);
        auto grad_U = neg(triu(grad_U_full));
        return {grad_B, grad_U};
    }
}

// Audit B.3: real higher-order backward via Variable-level composition.
// The Tensor-level backward uses two nested `solve_triangular` calls that
// compose to `(L L^T)^{-1} S = cholesky_solve(S, L)` (for the lower case,
// and the symmetric formula `(U^T U)^{-1} S = cholesky_solve(S, U, upper)`
// for the upper case). Re-expressing in those terms gives a backward that
// is built entirely from Variable-level ops — `cholesky_solve`, `matmul`,
// `transpose`, `tril`/`triu`, `neg` — so reverse-mode autograd over this
// computation produces the correct second-order gradient.
auto CholeskySolveBackward::backward_with_variables(std::vector<Variable> grad_outputs)
    -> std::vector<Variable> {
    require_saved_tensors(2);
    auto grad_X = grad_outputs[0];
    // Treat saved {X, L} as non-grad-tracking Variables — they were fixed
    // when the forward ran. The higher-order graph passes through grad_X.
    auto X = Variable(saved_tensors_[0], false);
    auto L = Variable(saved_tensors_[1], false);

    int64_t ndim = X.tensor().ndim();

    // grad_B = cholesky_solve(grad_X, L, upper) — Variable-level.
    auto grad_B = tenzor::cholesky_solve(grad_X, L, upper_);

    // S = grad_X @ X^T + X @ grad_X^T (symmetrized rank-2 outer product).
    auto Xt = tenzor::transpose(X, ndim - 2, ndim - 1);
    auto grad_Xt = tenzor::transpose(grad_X, ndim - 2, ndim - 1);
    auto S = tenzor::matmul(grad_X, Xt) + tenzor::matmul(X, grad_Xt);

    // grad_L = -tril(cholesky_solve(S, L, upper=false))  for lower;
    // grad_U = -triu(cholesky_solve(S, U, upper=true))   for upper.
    // cholesky_solve(M, L, upper=false) computes (L L^T)^{-1} M = L^{-T} L^{-1} M
    // which matches the Tensor-level double-triangular-solve chain.
    auto grad_L_full = tenzor::cholesky_solve(S, L, upper_);
    Variable grad_L = upper_ ? tenzor::neg(tenzor::triu(grad_L_full))
                              : tenzor::neg(tenzor::tril(grad_L_full));
    return {grad_B, grad_L};
}

// ============================================================================
// LUSolveBackward — audit-2026-05-03 Phase 8.
// Forward: X = lu_solve(LU, pivots, B). Treats LU/pivots as fixed.
// Backward: dL/dB = solve(A^T, dL/dX) where A = lu_to_dense(LU, pivots).
// Saves the reconstructed A for backward.
// ============================================================================
auto LUSolveBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LUSolveBackward::forward should not be called directly");
}

auto LUSolveBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& A = saved_tensors_[0];
    auto ndim = A.ndim();
    auto At = transpose(A, ndim - 2, ndim - 1).contiguous();
    return {tenzor::linalg::solve(At, grad)};
}

// ============================================================================
// EigBackward — audit-2026-05-03 Phase 8 (non-symmetric eigendecomposition).
// Forward: A → (W_real, W_imag, V).
// Backward (audit item A.10 — full coverage):
//   Real eigenvalues:
//     dA = V^{-T} (diag(dW) + (V^T dV) ∘ F) V^T,   F[i,j] = 1/(W[j]-W[i])
//   Complex eigenvalues:
//     The same Mike Giles closed form, but in complex arithmetic.
//     LAPACK geev returns V in *packed real* form for complex-conjugate
//     pairs — columns (j, j+1) hold (Re(v_j), Im(v_j)) with v_{j+1}=conj(v_j).
//     We unpack to a true Complex64/Complex128 V, run the formula, then
//     take Re(grad_A) (A is real).
//
//   The complex backward is routed through CPU regardless of the saved
//   tensors' device.  This is *explicit per-op routing*, not a fallback:
//   our GPU backends do not yet implement the complex linalg primitives
//   (cinv / cmatmul / cdiv) needed by this gradient path.  The forward
//   stays on GPU; only the gradient takes the CPU detour.  When GPU
//   complex linalg lands, drop the shuttle.
// ============================================================================
auto EigBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("EigBackward::forward should not be called directly");
}

namespace {

// Unpack LAPACK geev's packed-real eigenvector matrix into a true complex
// eigenvector matrix.  V_packed shape (n, n) on a single batch; W_imag
// shape (n,).  Returns Complex64/Complex128 matrix (depending on input
// dtype) with column j = the j-th complex eigenvector.
//
// For W_imag[j] == 0:  V_complex[:, j] = V_packed[:, j] + 0i.
// For W_imag[j]  > 0:  V_complex[:, j]   = V_packed[:, j] + i V_packed[:, j+1]
//                      V_complex[:, j+1] = V_packed[:, j] - i V_packed[:, j+1]
// For W_imag[j]  < 0:  already paired by column j-1 (sanity-checked).
template <typename Real>
void unpack_complex_eigenvectors_kernel(
        const Real* V_packed,           // (n*n) row-major
        const Real* W_imag,             // (n)
        std::complex<Real>* V_complex,  // (n*n) row-major
        int64_t n) {
    int64_t j = 0;
    while (j < n) {
        if (W_imag[j] == Real(0)) {
            for (int64_t i = 0; i < n; ++i) {
                V_complex[i * n + j] =
                    std::complex<Real>(V_packed[i * n + j], Real(0));
            }
            ++j;
        } else if (W_imag[j] > Real(0)) {
            // Conjugate pair (j, j+1).  geev guarantees j+1 < n and
            // W_imag[j+1] == -W_imag[j].
            if (j + 1 >= n) {
                throw std::runtime_error(
                    "EigBackward: geev returned a positive W_imag at the "
                    "final column — eigenvalue array is malformed.");
            }
            for (int64_t i = 0; i < n; ++i) {
                Real re = V_packed[i * n + j];
                Real im = V_packed[i * n + j + 1];
                V_complex[i * n + j]       = std::complex<Real>(re,  im);
                V_complex[i * n + j + 1]   = std::complex<Real>(re, -im);
            }
            j += 2;
        } else {
            // Should have been consumed by the previous step.
            throw std::runtime_error(
                "EigBackward: geev returned an unpaired negative W_imag entry "
                "— eigenvalue array is malformed.");
        }
    }
}

// Pullback of the real packed gradient grad_V_packed (n, n) to a complex
// grad_V_complex (n, n) consistent with V_complex above.
//
// Forward (packing):  V_packed[:, j]   = Re(V_complex[:, j])
//                     V_packed[:, j+1] = Im(V_complex[:, j])
//                     V_complex[:, j+1] = conj(V_complex[:, j])      (constraint)
//
// Wirtinger (conjugate-grad convention) pullback honouring the
// conjugate-pair constraint:
//   grad_V_complex[:, j]   = (1/2)(grad_V_packed[:, j] + i grad_V_packed[:, j+1])
//   grad_V_complex[:, j+1] = conj(grad_V_complex[:, j])
//
// For a purely-real column (W_imag[j] == 0):
//   grad_V_complex[:, j] = grad_V_packed[:, j] + 0i.
template <typename Real>
void unpack_complex_grad_V_kernel(
        const Real* gV_packed,
        const Real* W_imag,
        std::complex<Real>* gV_complex,
        int64_t n) {
    int64_t j = 0;
    while (j < n) {
        if (W_imag[j] == Real(0)) {
            for (int64_t i = 0; i < n; ++i) {
                gV_complex[i * n + j] =
                    std::complex<Real>(gV_packed[i * n + j], Real(0));
            }
            ++j;
        } else if (W_imag[j] > Real(0)) {
            if (j + 1 >= n) {
                throw std::runtime_error(
                    "EigBackward: malformed W_imag (orphan positive at column "
                    "boundary) in grad_V unpacking.");
            }
            for (int64_t i = 0; i < n; ++i) {
                Real re = Real(0.5) * gV_packed[i * n + j];
                Real im = Real(0.5) * gV_packed[i * n + j + 1];
                std::complex<Real> g(re, im);
                gV_complex[i * n + j]     = g;
                gV_complex[i * n + j + 1] = std::conj(g);
            }
            j += 2;
        } else {
            throw std::runtime_error(
                "EigBackward: malformed W_imag in grad_V unpacking.");
        }
    }
}

// Pullback of (grad_W_real, grad_W_imag) to a complex grad_w honouring
// the conjugate-pair symmetry of LAPACK eigenvalues.
//
// w_j = α + iβ, w_{j+1} = α - iβ.  For a real-valued downstream cost:
//   grad_w[j]   = (1/2)((grad_Wr[j] + grad_Wr[j+1])
//                       + i(grad_Wi[j] - grad_Wi[j+1]))
//   grad_w[j+1] = conj(grad_w[j])
// Real eigenvalue (β=0): grad_w[j] = grad_Wr[j] + i grad_Wi[j].
template <typename Real>
void unpack_complex_grad_W_kernel(
        const Real* gWr,
        const Real* gWi,
        const Real* W_imag,
        std::complex<Real>* gW_complex,
        int64_t n) {
    int64_t j = 0;
    while (j < n) {
        if (W_imag[j] == Real(0)) {
            gW_complex[j] = std::complex<Real>(gWr[j], gWi[j]);
            ++j;
        } else if (W_imag[j] > Real(0)) {
            if (j + 1 >= n) {
                throw std::runtime_error(
                    "EigBackward: malformed W_imag (orphan positive at column "
                    "boundary) in grad_W unpacking.");
            }
            Real re = Real(0.5) * (gWr[j] + gWr[j + 1]);
            Real im = Real(0.5) * (gWi[j] - gWi[j + 1]);
            std::complex<Real> g(re, im);
            gW_complex[j]     = g;
            gW_complex[j + 1] = std::conj(g);
            j += 2;
        } else {
            throw std::runtime_error(
                "EigBackward: malformed W_imag in grad_W unpacking.");
        }
    }
}

// Build complex tensors V_complex, grad_V_complex, grad_w_complex on CPU
// from the packed real saves.  Output dtype is Complex64 (Real=float) or
// Complex128 (Real=double).  Operates on a single (non-batched) (n, n)
// matrix — matches the rest of EigBackward which only supports
// non-batched eig.
template <typename Real, DType CDtype>
auto build_complex_inputs_typed(
        const Tensor& V_packed_cpu,    // (n, n), real dtype
        const Tensor& W_imag_cpu,      // (n,)
        const Tensor& grad_V_cpu,      // (n, n), real
        const Tensor& grad_Wr_cpu,     // (n,)
        const Tensor& grad_Wi_cpu)     // (n,)
        -> std::tuple<Tensor, Tensor, Tensor> {
    const int64_t n = V_packed_cpu.shape().back();

    auto V_packed_c = V_packed_cpu.contiguous();
    auto W_imag_c   = W_imag_cpu.contiguous();
    auto grad_V_c   = grad_V_cpu.contiguous();
    auto grad_Wr_c  = grad_Wr_cpu.contiguous();
    auto grad_Wi_c  = grad_Wi_cpu.contiguous();

    auto V_complex      = empty({n, n}, CDtype, Device::cpu());
    auto grad_V_complex = empty({n, n}, CDtype, Device::cpu());
    auto grad_W_complex = empty({n},    CDtype, Device::cpu());

    const Real* V_packed_data = V_packed_c.data<Real>();
    const Real* W_imag_data   = W_imag_c.data<Real>();
    const Real* gV_data       = grad_V_c.data<Real>();
    const Real* gWr_data      = grad_Wr_c.data<Real>();
    const Real* gWi_data      = grad_Wi_c.data<Real>();

    auto* Vc_data  = reinterpret_cast<std::complex<Real>*>(V_complex.storage()->data())
                     + V_complex.offset();
    auto* gVc_data = reinterpret_cast<std::complex<Real>*>(grad_V_complex.storage()->data())
                     + grad_V_complex.offset();
    auto* gWc_data = reinterpret_cast<std::complex<Real>*>(grad_W_complex.storage()->data())
                     + grad_W_complex.offset();

    unpack_complex_eigenvectors_kernel<Real>(V_packed_data, W_imag_data, Vc_data, n);
    unpack_complex_grad_V_kernel<Real>(gV_data, W_imag_data, gVc_data, n);
    unpack_complex_grad_W_kernel<Real>(gWr_data, gWi_data, W_imag_data, gWc_data, n);

    return {V_complex, grad_V_complex, grad_W_complex};
}

} // namespace

auto EigBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Audit item A.10: full coverage of both real- and complex-eigenvalue
    // backward.  See the formula and conjugate-pair derivation above.

    if (saved_tensors_.size() < 3) {
        throw std::runtime_error(
            "EigBackward: expected 3 saved tensors (W_real, W_imag, V); "
            "got " + std::to_string(saved_tensors_.size()) + ". "
            "This usually means an older eig() forward path wired the "
            "Function — rebuild the autograd graph.");
    }
    const auto& W_real = saved_tensors_[0];
    const auto& W_imag = saved_tensors_[1];
    const auto& V      = saved_tensors_[2];

    // Pad missing grad outputs with zeros so we can index uniformly.
    while (grad_outputs.size() < 3) {
        const auto& tmpl = (grad_outputs.size() < 2) ? W_real : V;
        grad_outputs.push_back(zeros_like(tmpl));
    }
    const auto& grad_W_real = grad_outputs[0];
    const auto& grad_W_imag = grad_outputs[1];
    const auto& grad_V      = grad_outputs[2];

    // Decide between the real-eigenvalue fast path and the complex path.
    // Threshold matches scipy.linalg's "treat as real" convention; LAPACK
    // returns exactly 0 for purely-real eigenvalues, so anything above
    // 1e-12 indicates a genuine complex-conjugate pair.
    constexpr double kImagThreshold = 1e-12;
    double imag_norm = 0.0;
    {
        auto imag_max = tenzor::max(tenzor::abs(W_imag));
        if (imag_max.dtype() == DType::Float64) {
            imag_norm = imag_max.item<double>();
        } else {
            imag_norm = static_cast<double>(imag_max.item<float>());
        }
    }

    const bool complex_path = (imag_norm > kImagThreshold);

    if (!complex_path) {
        // Sanity: a purely-real forward must have ~zero grad_W_imag, else
        // the caller is asking for a derivative through a non-existent
        // imaginary path.
        auto grad_imag_max = tenzor::max(tenzor::abs(grad_W_imag));
        double grad_imag_norm = (grad_imag_max.dtype() == DType::Float64)
            ? grad_imag_max.item<double>()
            : static_cast<double>(grad_imag_max.item<float>());
        if (grad_imag_norm > kImagThreshold) {
            throw std::runtime_error(
                "EigBackward: non-zero grad_W_imag passed for a "
                "real-eigenvalue forward — gradient flow inconsistent.");
        }

        // Real-eigenvalue closed-form backward (Mike Giles 2008).
        const auto ndim = V.ndim();
        const auto n = V.shape().back();

        auto W_col = unsqueeze(W_real, ndim - 1);  // (..., n, 1)
        auto W_row = unsqueeze(W_real, ndim - 2);  // (..., 1, n)
        auto diff = sub(W_row, W_col);             // (..., n, n)

        auto ones_n = tenzor::full({n}, 1.0, W_real.dtype(), W_real.device());
        auto eye_n = tenzor::diag(ones_n);
        auto ones_nn = tenzor::full({n, n}, 1.0, W_real.dtype(), W_real.device());
        auto off_diag_mask = tenzor::sub(ones_nn, eye_n);

        // Lorentzian-broadened reciprocal of eigenvalue gaps:
        //   F_ij = (W_j - W_i) / ((W_j - W_i)^2 + eps^2)   (i != j),   F_ii = 0.
        // Equals the exact 1/(W_j - W_i) for well-separated eigenvalues but stays
        // bounded by 1/(2 eps) at (near-)degenerate roots — where the eigenvector
        // derivative is genuinely singular. This prevents the Inf/NaN that the
        // naive reciprocal produced from reaching the downstream linalg::solve
        // (which hard-aborts on non-finite input), and matches the regularized
        // eig/eigh backward used by PyTorch/JAX for repeated eigenvalues. eps is
        // scaled to the spectrum so the exact gradient is recovered whenever the
        // gaps are resolvable in the working precision.
        const double mach = (W_real.dtype() == DType::Float64)
            ? 2.220446049250313e-16 : 1.1920929e-7;
        double w_scale = 0.0;
        {
            auto wmax = tenzor::max(tenzor::abs(W_real));
            w_scale = (wmax.dtype() == DType::Float64)
                ? wmax.item<double>() : static_cast<double>(wmax.item<float>());
        }
        const double eps_value = std::sqrt(mach) * (1.0 + std::abs(w_scale));
        auto eps_t = tenzor::full({1}, eps_value, W_real.dtype(), W_real.device());
        auto denom = tenzor::add(tenzor::mul(diff, diff), tenzor::mul(eps_t, eps_t));
        auto F = tenzor::mul(tenzor::div(diff, denom), off_diag_mask);

        auto Vt = transpose(V, ndim - 2, ndim - 1);

        auto VT_gradV = matmul(Vt, grad_V);
        auto eigvec_term = mul(VT_gradV, F);

        auto diag_grad_W = diag(grad_W_real);
        auto inner = tenzor::add(diag_grad_W, eigvec_term);

        auto rhs = matmul(inner, Vt);
        return {tenzor::linalg::solve(Vt.contiguous(), rhs)};
    }

    // ---- Complex-eigenvalue backward (CPU-only routing) ----
    //
    // We do not have GPU complex linalg primitives yet, so route the
    // entire complex gradient computation through CPU.  This is *not* a
    // CPU fallback for a GPU op — the eig forward stays on its original
    // device.  Only the gradient hop is CPU-bound until the GPU backends
    // grow complex inv / matmul / div.
    //
    // We currently support only non-batched eig (ndim == 2), matching the
    // real-path constraint above.  A genuine batched complex backward
    // would loop the unpacking and complex matmul per batch.

    if (V.ndim() != 2) {
        throw std::runtime_error(
            "EigBackward: complex-eigenvalue backward currently supports "
            "only non-batched matrices (ndim == 2); got ndim=" +
            std::to_string(V.ndim()) + ".  File an issue to add batched "
            "complex eig backward.");
    }

    const Device orig_device = V.device();
    const DType  real_dtype  = V.dtype();

    // Shuttle every input tensor to CPU.
    Tensor W_real_cpu   = W_real.to(Device::cpu());
    Tensor W_imag_cpu   = W_imag.to(Device::cpu());
    Tensor V_cpu        = V.to(Device::cpu());
    Tensor grad_Wr_cpu  = grad_W_real.to(Device::cpu());
    Tensor grad_Wi_cpu  = grad_W_imag.to(Device::cpu());
    Tensor grad_V_cpu   = grad_V.to(Device::cpu());

    // Promote everything to the LAPACK working precision (Float32 or
    // Float64) — the saved tensors are already that dtype, but the grad
    // outputs may have been padded with zeros_like (which preserves dtype
    // anyway).  Stay consistent with the saved real_dtype.
    W_real_cpu  = W_real_cpu.to(real_dtype);
    W_imag_cpu  = W_imag_cpu.to(real_dtype);
    V_cpu       = V_cpu.to(real_dtype);
    grad_Wr_cpu = grad_Wr_cpu.to(real_dtype);
    grad_Wi_cpu = grad_Wi_cpu.to(real_dtype);
    grad_V_cpu  = grad_V_cpu.to(real_dtype);

    Tensor V_complex_t, grad_V_complex_t, grad_W_complex_t;

    if (real_dtype == DType::Float64) {
        auto [Vc, gVc, gWc] = build_complex_inputs_typed<double, DType::Complex128>(
            V_cpu, W_imag_cpu, grad_V_cpu, grad_Wr_cpu, grad_Wi_cpu);
        V_complex_t      = Vc;
        grad_V_complex_t = gVc;
        grad_W_complex_t = gWc;
    } else if (real_dtype == DType::Float32) {
        auto [Vc, gVc, gWc] = build_complex_inputs_typed<float, DType::Complex64>(
            V_cpu, W_imag_cpu, grad_V_cpu, grad_Wr_cpu, grad_Wi_cpu);
        V_complex_t      = Vc;
        grad_V_complex_t = gVc;
        grad_W_complex_t = gWc;
    } else {
        // Lower-precision real dtypes are widened to Float32 by LAPACK; we
        // shouldn't reach here unless the forward changed.
        throw std::runtime_error(
            "EigBackward: complex backward requires Float32 or Float64 "
            "saved tensors; got dtype index " +
            std::to_string(static_cast<int>(real_dtype)) + ".");
    }

    // Build complex W (vector of n complex eigenvalues) via tenzor::complex.
    Tensor W_complex = tenzor::complex(W_real_cpu, W_imag_cpu);

    const int64_t n = V_complex_t.shape().back();

    // F[i, j] = 1 / (W[j] - W[i]), 0 on diagonal — all complex.
    auto W_col = unsqueeze(W_complex, 1);  // (n, 1)
    auto W_row = unsqueeze(W_complex, 0);  // (1, n)
    auto diff = sub(W_row, W_col);         // (n, n) complex

    // Lorentzian-broadened complex reciprocal of eigenvalue gaps:
    //   F_ij = conj(W_j - W_i) / (|W_j - W_i|^2 + eps^2)   (i != j),   F_ii = 0.
    // This is the exact 1/(W_j - W_i) for well-separated eigenvalues but stays
    // bounded by 1/(2 eps) at (near-)degenerate roots, preventing the Inf/NaN
    // that the naive reciprocal would push into the downstream complex
    // linalg::solve (which hard-aborts on non-finite input). Mirrors the
    // real-eigenvalue path's regularization.
    auto diff_re = tenzor::real(diff);
    auto diff_im = tenzor::imag(diff);
    auto mag2 = tenzor::add(tenzor::mul(diff_re, diff_re),
                            tenzor::mul(diff_im, diff_im));   // (n, n) real |diff|^2
    double w_scale_c = 0.0;
    {
        auto wr_max = tenzor::max(tenzor::abs(W_real_cpu));
        auto wi_max = tenzor::max(tenzor::abs(W_imag_cpu));
        auto scal = [](const Tensor& t) -> double {
            return (t.dtype() == DType::Float64) ? t.item<double>()
                                                 : static_cast<double>(t.item<float>());
        };
        w_scale_c = scal(wr_max) + scal(wi_max);
    }
    const double mach_c = (real_dtype == DType::Float64)
        ? 2.220446049250313e-16 : 1.1920929e-7;
    const double eps_c = std::sqrt(mach_c) * (1.0 + std::abs(w_scale_c));
    auto denom_c = tenzor::add(mag2,
        tenzor::full({1}, eps_c * eps_c, real_dtype, Device::cpu()));
    // F = conj(diff) / denom_c  (complex / real).  Build as complex(re/denom, im_conj/denom).
    auto inv_denom = tenzor::div(
        tenzor::full({1}, 1.0, real_dtype, Device::cpu()), denom_c);   // (n, n) real
    auto F_re = tenzor::mul(diff_re, inv_denom);
    auto F_im = tenzor::mul(tenzor::neg(diff_im), inv_denom);
    auto F = tenzor::complex(F_re, F_im);

    // Zero the diagonal of F (eigenvalue self-gap is identically zero).
    auto ones_n = tenzor::full({n}, 1.0, real_dtype, Device::cpu());
    auto zeros_n = tenzor::full({n}, 0.0, real_dtype, Device::cpu());
    auto eye_n_complex = tenzor::complex(tenzor::diag(ones_n), tenzor::diag(zeros_n));
    auto ones_nn_real = tenzor::full({n, n}, 1.0, real_dtype, Device::cpu());
    auto zeros_nn_real = tenzor::full({n, n}, 0.0, real_dtype, Device::cpu());
    auto ones_nn_complex = tenzor::complex(ones_nn_real, zeros_nn_real);
    auto off_diag_mask = tenzor::sub(ones_nn_complex, eye_n_complex);
    F = tenzor::mul(F, off_diag_mask);

    // The Mike Giles formula for the *non-symmetric* eig of a real
    // matrix A producing complex (W, V) uses the plain transpose, not
    // the Hermitian (conjugate) transpose:
    //
    //   grad_A = V^{-T} (diag(grad_W) + F ∘ (V^T grad_V)) V^T
    //
    // (See Giles 2008 §3 and the DLR matrix-calculus notes; the
    // conjugation that one might expect from "complex math" is already
    // baked into the conjugate-Wirtinger convention by which the
    // upstream gradients arrive — we must NOT conjugate again here.)
    auto V_T = transpose(V_complex_t, 0, 1).contiguous();

    // (V^T grad_V) ∘ F
    auto VT_gradV = matmul(V_T, grad_V_complex_t);
    auto eigvec_term = mul(VT_gradV, F);

    // diag(grad_W) is complex; tenzor::diag is dtype-agnostic byte copy
    // on CPU, so it works for Complex64/128.
    auto diag_grad_W = diag(grad_W_complex_t);
    auto inner = tenzor::add(diag_grad_W, eigvec_term);

    // grad_A_complex = V^{-T} @ inner @ V^T
    //                = solve(V^T, inner @ V^T)
    auto rhs = matmul(inner, V_T);
    auto grad_A_complex = tenzor::linalg::solve(V_T, rhs);

    // A is real → take the real part.
    auto grad_A_cpu = tenzor::real(grad_A_complex).to(real_dtype);

    // Move back to the original device.
    if (orig_device.type != Device::Type::CPU) {
        return {grad_A_cpu.to(orig_device)};
    }
    return {grad_A_cpu};
}

} // namespace tenzor
