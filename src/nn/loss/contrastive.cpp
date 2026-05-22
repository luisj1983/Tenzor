/**
 * @file contrastive.cpp
 * @brief Implementation of contrastive loss functions (InfoNCE, NT-Xent, TripletLoss)
 */

#include "tenzor/nn/loss/contrastive.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/autograd/ops.hpp"
#include <cmath>
#include <limits>

namespace tenzor::nn {

// ============================================================================
// Helpers
// ============================================================================

namespace {

// L2-normalize along the last dimension
auto l2_normalize(const Variable& input) -> Variable {
    // norm = sqrt(sum(x^2, dim=-1, keepdim=true) + eps)
    auto squared = input * input;
    auto sum_sq = sum(squared, -1, /*keepdim=*/true);
    auto eps_tensor = full({1}, 1e-8f, input.dtype(), input.device());
    Variable eps_var(eps_tensor, false);
    auto norm_val = sum_sq + eps_var;

    // sqrt via: x * rsqrt(x) = sqrt(x), but we only have sqrt on tensors
    // Use: exp(0.5 * log(norm_val))
    auto half = Variable(full({1}, 0.5f, input.dtype(), input.device()), false);
    auto sqrt_norm = exp(half * log(norm_val));

    return input / sqrt_norm;
}

// Compute pairwise cosine similarity matrix
auto cosine_similarity_matrix(const Variable& a, const Variable& b) -> Variable {
    // a: (N, D), b: (M, D)
    // Result: (N, M) cosine similarity matrix
    auto a_norm = l2_normalize(a);
    auto b_norm = l2_normalize(b);

    // a_norm @ b_norm^T
    auto b_t = permute(b_norm, {1, 0});
    return matmul(a_norm, b_t);
}

// Pairwise distance (Lp norm).
// Renamed from `pairwise_distance` to avoid ambiguity with the public
// tenzor::pairwise_distance(Variable, Variable, double) introduced in
// audit E.7 batch 6. The two implementations differ slightly in
// epsilon handling and shape (1-D vector vs. (B,)); existing TripletLoss
// callers want this exact behaviour, so we keep the local helper but
// give it an unambiguous name.
auto local_pairwise_distance(const Variable& a, const Variable& b, double p) -> Variable {
    // ||a - b||_p per sample
    auto diff = a - b;

    if (p == 2.0) {
        // L2 distance: sqrt(sum(diff^2))
        auto sq = diff * diff;
        auto sum_sq = sum(sq, -1, /*keepdim=*/false);
        auto eps_tensor = full({1}, 1e-8f, a.dtype(), a.device());
        Variable eps_var(eps_tensor, false);
        auto safe_sum = sum_sq + eps_var;
        auto half = Variable(full({1}, 0.5f, a.dtype(), a.device()), false);
        return exp(half * log(safe_sum));
    } else if (p == 1.0) {
        // L1 distance: sum(|diff|)
        auto abs_diff = abs(diff);
        return sum(abs_diff, -1, /*keepdim=*/false);
    } else {
        // General Lp: (sum(|diff|^p))^(1/p)
        auto abs_diff = abs(diff);
        auto p_tensor = Variable(full({1}, static_cast<float>(p), a.dtype(), a.device()), false);
        auto inv_p = Variable(full({1}, static_cast<float>(1.0 / p), a.dtype(), a.device()), false);
        // |diff|^p = exp(p * log(|diff| + eps))
        auto eps_tensor = full({1}, 1e-8f, a.dtype(), a.device());
        Variable eps_var(eps_tensor, false);
        auto pow_p = exp(p_tensor * log(abs_diff + eps_var));
        auto sum_pow = sum(pow_p, -1, /*keepdim=*/false);
        // sum^(1/p) = exp((1/p) * log(sum))
        auto eps2 = Variable(full({1}, 1e-8f, a.dtype(), a.device()), false);
        return exp(inv_p * log(sum_pow + eps2));
    }
}

// Create scalar Variable
auto scalar_var(float value, DType dtype, Device device) -> Variable {
    return Variable(full({1}, value, dtype, device), false);
}

} // anonymous namespace

// ============================================================================
// InfoNCE Loss
// ============================================================================

InfoNCELoss::InfoNCELoss(double temperature, Reduction reduction)
    : temperature_(temperature), reduction_(reduction) {}

auto InfoNCELoss::forward(const Variable& queries, const Variable& keys) -> Variable {
    // queries: (N, D), keys: (N, D)
    // Positive pairs: (query_i, key_i) for same i
    // Negative pairs: all other combinations

    auto batch_shape = queries.shape();
    int64_t batch_size = batch_shape[0];

    // Compute cosine similarity matrix: (N, N)
    auto sim_matrix = cosine_similarity_matrix(queries, keys);

    // Scale by temperature
    auto temp = scalar_var(static_cast<float>(temperature_), queries.dtype(), queries.device());
    auto logits = sim_matrix / temp;

    // Labels: positive pairs are on the diagonal (label[i] = i)
    // This is equivalent to cross-entropy with targets = [0, 1, 2, ..., N-1]
    Tensor labels = arange(0.0f, static_cast<float>(batch_size), 1.0f,
                          DType::Int64, queries.device());

    // Cross entropy loss: -log(softmax(logits)[i, labels[i]])
    CrossEntropyLoss ce_loss(reduction_);
    return ce_loss.forward(logits, labels);
}

// ============================================================================
// NT-Xent Loss
// ============================================================================

NTXentLoss::NTXentLoss(double temperature, Reduction reduction)
    : temperature_(temperature), reduction_(reduction) {}

auto NTXentLoss::forward(const Variable& z_i, const Variable& z_j) -> Variable {
    // z_i: (N, D), z_j: (N, D)
    // Concatenate: z = [z_i; z_j], shape (2N, D)
    // Positive pairs: (z_i[k], z_j[k]) and (z_j[k], z_i[k]) for all k

    auto batch_shape = z_i.shape();
    int64_t batch_size = batch_shape[0];

    // Concatenate both views
    auto z = cat({z_i, z_j}, 0);  // (2N, D)

    // Compute full similarity matrix: (2N, 2N)
    auto sim_matrix = cosine_similarity_matrix(z, z);

    // Scale by temperature
    auto temp = scalar_var(static_cast<float>(temperature_), z_i.dtype(), z_i.device());
    auto logits = sim_matrix / temp;

    // Mask out self-similarity on the diagonal with -inf
    // Create labels: for sample i in [0, N), positive is at i+N; for i in [N, 2N), positive is at i-N
    // labels = [N, N+1, ..., 2N-1, 0, 1, ..., N-1]
    int64_t total = 2 * batch_size;

    // Create the diagonal mask: fill diagonal with a large finite negative
    // value. We deliberately do NOT use `-inf` here because softmax /
    // cross-entropy backward on some backends (notably Vulkan, audit-2026-05-03)
    // surface as `0 / 0 = NaN` gradients at masked positions; the standard
    // numerically-safe trick used by PyTorch's MultiheadAttention etc. is to
    // mask with a large finite negative whose `exp(.)` rounds to 0 in the
    // forward direction but produces well-defined zero gradient. -1e4 is
    // small enough to fit in Float16 (`max ≈ 65504`) and large enough that
    // `exp(-1e4)` underflows to 0 in Float32+.
    //
    // Previously this path hardcoded `mask.data<float>()`, which throws a
    // "type mismatch" error whenever z_i has any non-Float32 dtype. Build
    // the mask in Float32 on CPU, then cast/transfer to the target dtype
    // and device — correct for every floating dtype we support.
    constexpr float kMaskNeg = -1e4f;
    Tensor cpu_mask_f32 = zeros({total, total}, DType::Float32, Device::cpu());
    {
        float* mask_data = cpu_mask_f32.data<float>();
        for (int64_t i = 0; i < total; ++i) {
            mask_data[i * total + i] = kMaskNeg;
        }
    }
    Tensor mask = cpu_mask_f32.to(z_i.dtype()).to(z_i.device());

    Variable mask_var(mask, false);
    logits = logits + mask_var;

    // Labels: positive pair indices
    // First N samples have positives at indices N..2N-1
    // Next N samples have positives at indices 0..N-1
    Tensor labels_tensor({total}, DType::Int64, Device::cpu());
    auto* label_data = labels_tensor.data<int64_t>();
    for (int64_t i = 0; i < batch_size; ++i) {
        label_data[i] = batch_size + i;           // z_i[i] -> z_j[i]
        label_data[batch_size + i] = i;           // z_j[i] -> z_i[i]
    }
    if (z_i.device() != Device::cpu()) {
        labels_tensor = labels_tensor.to(z_i.device());
    }

    // Cross entropy over full 2N samples
    CrossEntropyLoss ce_loss(reduction_);
    return ce_loss.forward(logits, labels_tensor);
}

// ============================================================================
// Triplet Loss
// ============================================================================

TripletLoss::TripletLoss(double margin, double p, bool swap, Reduction reduction)
    : margin_(margin), p_(p), swap_(swap), reduction_(reduction) {}

auto TripletLoss::forward(const Variable& anchor,
                          const Variable& positive,
                          const Variable& negative) -> Variable {
    // Compute distances
    auto dist_ap = local_pairwise_distance(anchor, positive, p_);  // (N,)
    auto dist_an = local_pairwise_distance(anchor, negative, p_);  // (N,)

    if (swap_) {
        // Distance-swapped variant: use min(d(a,n), d(p,n))
        auto dist_pn = local_pairwise_distance(positive, negative, p_);
        // Element-wise min: min(dist_an, dist_pn) = (dist_an + dist_pn - |dist_an - dist_pn|) / 2
        auto diff = dist_an - dist_pn;
        auto abs_diff = abs(diff);
        auto two_var = scalar_var(2.0f, anchor.dtype(), anchor.device());
        dist_an = (dist_an + dist_pn - abs_diff) / two_var;
    }

    // loss = max(0, d(a,p) - d(a,n) + margin)
    auto margin_var = scalar_var(static_cast<float>(margin_), anchor.dtype(), anchor.device());
    auto loss_raw = dist_ap - dist_an + margin_var;

    // ReLU: max(0, x) = (x + |x|) / 2
    auto abs_loss = abs(loss_raw);
    auto two_var = scalar_var(2.0f, anchor.dtype(), anchor.device());
    auto loss_unreduced = (loss_raw + abs_loss) / two_var;

    switch (reduction_) {
        case Reduction::None:
            return loss_unreduced;
        case Reduction::Mean:
            return mean(loss_unreduced);
        case Reduction::Sum:
            return sum(loss_unreduced);
        default:
            return mean(loss_unreduced);
    }
}

// ============================================================================
// Functional implementations
// ============================================================================

auto info_nce_loss(const Variable& queries, const Variable& keys,
                  double temperature, Reduction reduction) -> Variable {
    InfoNCELoss loss(temperature, reduction);
    return loss.forward(queries, keys);
}

auto nt_xent_loss(const Variable& z_i, const Variable& z_j,
                 double temperature, Reduction reduction) -> Variable {
    NTXentLoss loss(temperature, reduction);
    return loss.forward(z_i, z_j);
}

auto triplet_loss(const Variable& anchor,
                 const Variable& positive,
                 const Variable& negative,
                 double margin, double p, bool swap,
                 Reduction reduction) -> Variable {
    TripletLoss loss(margin, p, swap, reduction);
    return loss.forward(anchor, positive, negative);
}

} // namespace tenzor::nn
