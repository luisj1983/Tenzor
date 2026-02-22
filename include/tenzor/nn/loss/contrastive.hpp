/**
 * @file contrastive.hpp
 * @brief Contrastive loss functions for self-supervised and metric learning
 *
 * Provides InfoNCE, NT-Xent, and TripletLoss for contrastive representation learning.
 * These losses train models to produce embeddings where similar items are close
 * and dissimilar items are far apart in the embedding space.
 */

#pragma once

#include "../module.hpp"
#include "losses.hpp"
#include <string>

namespace tenzor {
namespace nn {

/**
 * @brief InfoNCE (Noise Contrastive Estimation) Loss
 *
 * Computes the InfoNCE loss used in contrastive learning (e.g., SimCLR, CLIP, MoCo):
 *
 * \f[
 * \mathcal{L} = -\log\frac{\exp(\text{sim}(q, k^+) / \tau)}
 *                          {\sum_{i=0}^{K}\exp(\text{sim}(q, k_i) / \tau)}
 * \f]
 *
 * where sim(a, b) = a^T b / (||a|| * ||b||) is cosine similarity,
 * k+ is the positive key, and K includes negative keys.
 *
 * **Input format:**
 * - queries: (N, D) - query embeddings
 * - keys: (N, D) - positive key embeddings (matched with queries)
 * The loss treats all non-matching pairs within the batch as negatives.
 *
 * **Use Cases:**
 * - Self-supervised learning (SimCLR, MoCo, CLIP)
 * - Representation learning
 * - Multimodal alignment (text-image matching)
 *
 * @param temperature Temperature scaling factor (default: 0.07)
 * @param reduction How to reduce the loss (default: Mean)
 *
 * @code
 * auto criterion = InfoNCELoss(0.07);
 * auto query_emb = encoder(augmented_view_1);  // (N, D)
 * auto key_emb = encoder(augmented_view_2);    // (N, D)
 * auto loss = criterion(query_emb, key_emb);
 * @endcode
 *
 * @see NTXentLoss for symmetric variant
 */
class InfoNCELoss {
public:
    explicit InfoNCELoss(double temperature = 0.07,
                        Reduction reduction = Reduction::Mean);

    /**
     * @brief Compute InfoNCE loss.
     *
     * @param queries Query embeddings (N, D)
     * @param keys Positive key embeddings (N, D) - row i matches query i
     * @return Loss value
     */
    auto forward(const Variable& queries, const Variable& keys) -> Variable;

    auto operator()(const Variable& queries, const Variable& keys) -> Variable {
        return forward(queries, keys);
    }

private:
    double temperature_;
    Reduction reduction_;
};

/**
 * @brief NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
 *
 * Symmetric contrastive loss used in SimCLR:
 *
 * \f[
 * \mathcal{L} = \frac{1}{2N}\sum_{k=1}^{N}\left[\ell(2k-1, 2k) + \ell(2k, 2k-1)\right]
 * \f]
 *
 * where:
 * \f[
 * \ell(i, j) = -\log\frac{\exp(\text{sim}(z_i, z_j)/\tau)}
 *                         {\sum_{k \neq i}\exp(\text{sim}(z_i, z_k)/\tau)}
 * \f]
 *
 * NT-Xent is symmetric: both views are treated as queries and keys.
 * This typically gives better performance than one-directional InfoNCE.
 *
 * **Input format:**
 * - z_i: (N, D) - embeddings from first augmented view
 * - z_j: (N, D) - embeddings from second augmented view
 *
 * **Use Cases:**
 * - SimCLR training
 * - Self-supervised visual representation learning
 * - Symmetric contrastive objectives
 *
 * @param temperature Temperature scaling factor (default: 0.5)
 * @param reduction How to reduce the loss (default: Mean)
 *
 * @code
 * auto criterion = NTXentLoss(0.5);
 * auto z_i = projector(encoder(augmented_1));  // (N, D)
 * auto z_j = projector(encoder(augmented_2));  // (N, D)
 * auto loss = criterion(z_i, z_j);
 * @endcode
 *
 * @see InfoNCELoss for asymmetric variant
 */
class NTXentLoss {
public:
    explicit NTXentLoss(double temperature = 0.5,
                       Reduction reduction = Reduction::Mean);

    /**
     * @brief Compute NT-Xent loss.
     *
     * @param z_i Embeddings from first view (N, D)
     * @param z_j Embeddings from second view (N, D)
     * @return Loss value
     */
    auto forward(const Variable& z_i, const Variable& z_j) -> Variable;

    auto operator()(const Variable& z_i, const Variable& z_j) -> Variable {
        return forward(z_i, z_j);
    }

private:
    double temperature_;
    Reduction reduction_;
};

/**
 * @brief Triplet Margin Loss
 *
 * Computes the triplet loss with margin:
 *
 * \f[
 * \mathcal{L} = \max(0, \|a - p\|_p - \|a - n\|_p + \text{margin})
 * \f]
 *
 * where a = anchor, p = positive, n = negative.
 *
 * Trains the model so that the anchor is closer to the positive than
 * to the negative by at least the margin distance.
 *
 * **Input format:**
 * - anchor: (N, D) - anchor embeddings
 * - positive: (N, D) - positive embeddings (similar to anchor)
 * - negative: (N, D) - negative embeddings (dissimilar to anchor)
 *
 * **Use Cases:**
 * - Face recognition/verification (FaceNet)
 * - Image retrieval
 * - Metric learning
 * - Siamese networks
 *
 * **Hard Mining:**
 * For best results, use hard negative/positive mining strategies
 * to select informative triplets.
 *
 * @param margin Minimum desired distance gap (default: 1.0)
 * @param p Norm degree for distance computation (default: 2.0 for L2)
 * @param swap If true, use distance-swapped triplet loss (default: false)
 * @param reduction How to reduce the loss (default: Mean)
 *
 * @code
 * auto criterion = TripletLoss(1.0);
 * auto anchor = encoder(anchor_images);    // (N, D)
 * auto pos = encoder(positive_images);     // (N, D)
 * auto neg = encoder(negative_images);     // (N, D)
 * auto loss = criterion(anchor, pos, neg);
 * @endcode
 *
 * @see InfoNCELoss for batch-level contrastive alternative
 */
class TripletLoss {
public:
    explicit TripletLoss(double margin = 1.0,
                        double p = 2.0,
                        bool swap = false,
                        Reduction reduction = Reduction::Mean);

    /**
     * @brief Compute triplet loss.
     *
     * @param anchor Anchor embeddings (N, D)
     * @param positive Positive embeddings (N, D)
     * @param negative Negative embeddings (N, D)
     * @return Loss value
     */
    auto forward(const Variable& anchor,
                const Variable& positive,
                const Variable& negative) -> Variable;

    auto operator()(const Variable& anchor,
                   const Variable& positive,
                   const Variable& negative) -> Variable {
        return forward(anchor, positive, negative);
    }

private:
    double margin_;
    double p_;
    bool swap_;
    Reduction reduction_;
};

// ============================================================================
// Functional contrastive losses
// ============================================================================

/** @brief Functional InfoNCE loss computation */
auto info_nce_loss(const Variable& queries, const Variable& keys,
                  double temperature = 0.07,
                  Reduction reduction = Reduction::Mean) -> Variable;

/** @brief Functional NT-Xent loss computation */
auto nt_xent_loss(const Variable& z_i, const Variable& z_j,
                 double temperature = 0.5,
                 Reduction reduction = Reduction::Mean) -> Variable;

/** @brief Functional triplet loss computation */
auto triplet_loss(const Variable& anchor,
                 const Variable& positive,
                 const Variable& negative,
                 double margin = 1.0,
                 double p = 2.0,
                 bool swap = false,
                 Reduction reduction = Reduction::Mean) -> Variable;

} // namespace nn
} // namespace tenzor
