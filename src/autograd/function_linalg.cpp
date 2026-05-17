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
    const auto& grad = grad_outputs[0];       // dL/dy, scalar
    const auto& input = saved_tensors_[0];    // A
    const auto& norm_val = saved_tensors_[1]; // norm(A), scalar

    if (ord_ == "fro") {
        // dL/dA = dL/dy * A / norm(A)
        // Reshape grad and norm_val to broadcast with A
        auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());

        // grad / norm_val is a scalar ratio
        auto scale = div(grad, norm_val);

        // Expand scale to match input shape
        auto scale_shape = std::vector<int64_t>(scale.shape().begin(), scale.shape().end());
        while (scale_shape.size() < input_shape.size()) {
            scale_shape.push_back(1);
        }
        auto scale_expanded = reshape(scale, scale_shape);

        auto grad_A = mul(scale_expanded, input);
        return {grad_A};
    }

    // For non-Frobenius norms, return zeros (unsupported)
    auto grad_A = zeros_like(input);
    return {grad_A};
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
    auto middle = add(mul(F, VtgV), diag(grad_W));  // (N, N)

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
    auto grad_diag = diag(grad_W);  // (N, N)
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
    // Frobenius: dL/dA = dL/dy * A / norm(A)
    Variable input, norm_val;
    if (has_saved_variables()) {
        require_saved_variables(2);
        input = saved_variables_[0];
        norm_val = saved_variables_[1];
    } else {
        require_saved_tensors(2);
        input = Variable(saved_tensors_[0], false);
        norm_val = Variable(saved_tensors_[1], false);
    }

    if (ord_ == "fro") {
        auto scale = grad_outputs[0] * tenzor::reciprocal(norm_val);
        auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
        auto scale_shape = std::vector<int64_t>(scale.shape().begin(), scale.shape().end());
        while (scale_shape.size() < input_shape.size()) {
            scale_shape.push_back(1);
        }
        auto scale_expanded = tenzor::reshape(scale, scale_shape);
        return {scale_expanded * input};
    }

    // Unsupported norm order — return zeros (no gradient)
    return {Variable(zeros_like(input.tensor()), false)};
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
// Backward (eigenvalue path only, real eigenvalues):
//   dA = V^{-T} @ diag(dW_real) @ V^T
// ============================================================================
auto EigBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("EigBackward::forward should not be called directly");
}

auto EigBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Pad to 3 grad outputs (W_real, W_imag, V) so we can safely index even
    // when the engine only fills the first one.
    if (grad_outputs.size() < 3) {
        const auto& W = saved_tensors_[0];
        const auto& V = saved_tensors_[1];
        if (grad_outputs.size() < 1) {
            grad_outputs.push_back(zeros_like(W));
        }
        while (grad_outputs.size() < 2) {
            grad_outputs.push_back(zeros_like(W));
        }
        while (grad_outputs.size() < 3) {
            grad_outputs.push_back(zeros_like(V));
        }
    }
    const auto& grad_W_real = grad_outputs[0];
    const auto& V = saved_tensors_[1];
    auto ndim = V.ndim();
    auto diag_dW = diag(grad_W_real);
    auto Vt = transpose(V, ndim - 2, ndim - 1);
    auto rhs = matmul(diag_dW, Vt);
    return {tenzor::linalg::solve(Vt.contiguous(), rhs)};
}

} // namespace tenzor
