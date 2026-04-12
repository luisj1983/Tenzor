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
    const auto& grad = grad_outputs[0];          // dL/dy, scalar or (...,)
    const auto& det_val = saved_tensors_[0];      // det(A), same shape as grad
    const auto& inv_A = saved_tensors_[1];        // A^{-1}, (..., N, N)

    auto ndim = inv_A.ndim();
    // A^{-T} = transpose of inverse
    auto inv_At = transpose(inv_A, ndim - 2, ndim - 1);

    // grad * det(A) is scalar (per batch), need to broadcast to matrix shape
    auto grad_det = mul(grad, det_val);  // (...,)

    // Reshape grad_det to broadcast with inv_At: add two trailing dims
    auto gd_shape = std::vector<int64_t>(grad_det.shape().begin(), grad_det.shape().end());
    gd_shape.push_back(1);
    gd_shape.push_back(1);
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
    // grad_outputs[0] = dL/d(sign) -- ignored (zero gradient)
    // grad_outputs[1] = dL/d(logabsdet)
    const auto& grad_logabsdet = grad_outputs[1];
    const auto& inv_A = saved_tensors_[0];     // A^{-1}, (..., N, N)

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
    const auto& grad_U = grad_outputs[0];     // dL/dU, (..., M, K)
    const auto& grad_S = grad_outputs[1];     // dL/dS, (..., K)
    const auto& grad_Vh = grad_outputs[2];    // dL/dVh, (..., K, N)

    const auto& U = saved_tensors_[0];        // U, (..., M, K)
    const auto& S = saved_tensors_[1];        // S, (..., K)
    const auto& Vh = saved_tensors_[2];       // Vh, (..., K, N)

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

    // Build F matrix element-wise
    // For simplicity, compute on CPU
    auto s_data = S.contiguous();
    auto f_shape = std::vector<int64_t>(S.shape().begin(), S.shape().end());
    f_shape[f_shape.size() - 1] = K;
    f_shape.push_back(K);

    // Create F as zeros
    // Simplified: we need batch-aware F construction
    // For now, compute F for last two dims
    auto F_tensor = zeros({K, K}, S.dtype(), S.device());

    // Fill F: this requires element access, do it with a simpler approach
    // Use outer products of S
    // s_j^2 - s_i^2 = (s_j - s_i)(s_j + s_i)
    // Reshape S for broadcasting: S_row (1, K), S_col (K, 1)
    auto S_row = reshape(S_sq, {1, K});
    auto S_col = reshape(S_sq, {K, 1});
    auto diffs = sub(S_row, S_col);  // (K, K): diffs[i][j] = s_j^2 - s_i^2

    // Clamp small |s_j^2 - s_i^2| to epsilon for near-degenerate singular values
    auto eps_val = detail::dtype_epsilon(S.dtype());
    auto eps_t = full({K, K}, eps_val, S.dtype(), S.device());
    auto abs_diffs = abs(diffs);
    auto safe_diffs = where(lt(abs_diffs, eps_t), mul(sign(diffs), eps_t), diffs);

    // Replace diagonal with 1 to avoid division by zero
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

    auto F_UtgU = mul(F, UtgU);
    auto F_gVhV = mul(F, gVhV);

    auto term1 = diag(grad_S);             // diag(grad_S), (K, K)
    auto term2 = matmul(F_UtgU, S_diag);   // F*(Ut@gU) @ S
    auto term3 = matmul(S_diag, F_gVhV);   // S @ F*(gVh@V)

    auto middle = add(add(term1, term2), term3);  // (K, K)

    auto grad_A = matmul(matmul(U, middle), Vh);  // (..., M, N)

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
    const auto& grad_Q = grad_outputs[0];   // dL/dQ, (..., M, N)
    const auto& grad_R = grad_outputs[1];   // dL/dR, (..., N, N)

    const auto& Q = saved_tensors_[0];      // Q, (..., M, N)
    const auto& R = saved_tensors_[1];      // R, (..., N, N)

    auto ndim = R.ndim();
    auto Rt = transpose(R, ndim - 2, ndim - 1);
    auto Qt = transpose(Q, Q.ndim() - 2, Q.ndim() - 1);

    // M = R @ grad_R^T - grad_Q^T @ Q
    auto grad_Rt = transpose(grad_R, ndim - 2, ndim - 1);
    auto M = sub(matmul(R, grad_Rt), matmul(Qt, grad_Q));

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

    return {grad_A};
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
    const auto& grad_W = grad_outputs[0];   // dL/dW, (..., N)
    const auto& grad_V = grad_outputs[1];   // dL/dV, (..., N, N)

    const auto& W = saved_tensors_[0];      // eigenvalues, (..., N)
    const auto& V = saved_tensors_[1];      // eigenvectors, (..., N, N)

    auto N = W.shape()[W.ndim() - 1];

    // Construct F matrix: F_{ij} = 1/(w_j - w_i) for i != j, 0 on diagonal
    auto W_row = reshape(W, {1, N});
    auto W_col = reshape(W, {N, 1});
    auto diffs = sub(W_row, W_col);  // (N, N): diffs[i][j] = w_j - w_i

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
    const auto& grad_L = grad_outputs[0];   // (..., N, N)
    const auto& grad_U = grad_outputs[1];   // (..., N, N)
    // grad_outputs[2] is for pivots -- ignored (non-differentiable)

    const auto& L = saved_tensors_[0];      // (..., N, N) unit lower triangular
    const auto& U = saved_tensors_[1];      // (..., N, N) upper triangular

    // L is unit lower triangular: only the strictly lower part of grad_L contributes
    auto grad_L_strict = tril(grad_L, -1);

    // U is upper triangular: only the upper part of grad_U contributes
    auto grad_U_upper = triu(grad_U);

    // grad_A = grad_L_strict @ U + L @ grad_U_upper
    // (Ignoring permutation P for simplicity -- the gradient flows through P^T
    //  but P is a constant permutation, so P^T @ grad = reorder rows of grad.
    //  For the common case where backward() is called after lu() in autograd,
    //  the permutation is handled by the saved_tensors order.)
    auto term1 = matmul(grad_L_strict, U);
    auto term2 = matmul(L, grad_U_upper);
    auto grad_A = add(term1, term2);

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

} // namespace tenzor
