/**
 * @file gradient_compression.hpp
 * @brief Gradient compression strategies for bandwidth-efficient distributed training
 *
 * Provides pluggable gradient compression algorithms that reduce communication
 * volume during all-reduce in distributed data parallel training. Compressors
 * can be attached to a DistributedDataParallel instance to transparently
 * compress gradients before all-reduce and decompress afterwards.
 *
 * Supported strategies:
 * - FP16Compressor: Cast gradients to Float16 before all-reduce (2x bandwidth savings)
 * - TopKCompressor: Keep only top-K% largest gradient values (with error feedback)
 *
 * @code
 * // Create a compressor and attach to DDP
 * auto compressor = std::make_unique<FP16Compressor>();
 * // In the all-reduce path:
 * auto compressed = compressor->compress(gradient);
 * pg.all_reduce(compressed.data, ReduceOp::SUM);
 * auto decompressed = compressor->decompress(compressed);
 * @endcode
 */

#pragma once

#include "../core/tensor.hpp"
#include "../core/dtype.hpp"
#include <memory>
#include <string>
#include <mutex>
#include <unordered_map>

namespace tenzor::distributed {

/**
 * @brief Metadata and compressed data for a gradient tensor.
 *
 * Holds the compressed representation of a gradient along with metadata
 * needed to reconstruct the original tensor (shape, dtype, device, etc.).
 * The exact contents depend on the compression algorithm used.
 */
struct CompressedGradient {
    /** @brief Compressed gradient data (may be a different dtype or sparse) */
    Tensor data;

    /** @brief Original shape of the gradient before compression */
    std::vector<int64_t> original_shape;

    /** @brief Original dtype of the gradient before compression */
    DType original_dtype{DType::Float32};

    /** @brief Original device of the gradient */
    Device original_device{Device::cpu()};

    /** @brief Number of elements in the original gradient */
    int64_t original_numel{0};

    /** @brief Compression ratio achieved (compressed_size / original_size) */
    float compression_ratio{1.0f};
};

/**
 * @brief Abstract base class for gradient compression algorithms.
 *
 * Gradient compressors reduce the communication volume during distributed
 * all-reduce by compressing gradient tensors before transmission and
 * decompressing after reception. Implementations must be stateless with
 * respect to the gradient data itself (any internal state like error
 * feedback buffers is keyed per-parameter, not per-call).
 *
 * Thread safety: compress() and decompress() must be safe to call from
 * multiple threads concurrently for different gradient tensors.
 */
class GradientCompressor {
public:
    virtual ~GradientCompressor() = default;

    /**
     * @brief Compress a gradient tensor for communication.
     *
     * Transforms the gradient into a more compact representation suitable
     * for bandwidth-efficient all-reduce. The input gradient may be modified
     * in-place by some compressors (e.g., TopK zeros out small values).
     *
     * @param gradient The gradient tensor to compress (may be modified in-place)
     * @return CompressedGradient containing the compressed data and metadata
     */
    virtual auto compress(Tensor& gradient) -> CompressedGradient = 0;

    /**
     * @brief Decompress a gradient back to its original representation.
     *
     * Reconstructs the gradient tensor from its compressed form. The
     * returned tensor will have the same shape and dtype as the original
     * gradient that was passed to compress().
     *
     * @param compressed The compressed gradient to decompress
     * @return Decompressed gradient tensor in the original dtype and shape
     */
    virtual auto decompress(CompressedGradient& compressed) -> Tensor = 0;

    /**
     * @brief Get the name of this compression algorithm.
     *
     * @return Human-readable name (e.g., "FP16", "TopK")
     */
    virtual auto name() const -> std::string = 0;

    /**
     * @brief Reset any internal state (e.g., error feedback buffers).
     *
     * Should be called when starting a new training run or when the
     * model parameters change. The default implementation is a no-op.
     */
    virtual auto reset() -> void {}
};

/**
 * @brief FP16 gradient compression via dtype casting.
 *
 * Compresses gradients by casting from their original dtype (typically
 * Float32 or Float64) to Float16 before all-reduce, then casting back
 * after all-reduce. This achieves a 2x bandwidth reduction for Float32
 * gradients with minimal accuracy impact for most training scenarios.
 *
 * This is the simplest and most widely-used gradient compression strategy,
 * commonly combined with loss scaling in mixed-precision training.
 *
 * Notes:
 * - Float16 has a limited dynamic range (~6e-8 to 65504). Very small
 *   gradients may underflow to zero. Consider using a GradScaler if
 *   gradient magnitudes are consistently small.
 * - If the gradient is already Float16 or BFloat16, compress() is a no-op.
 * - Gradients on any device (CPU, CUDA, etc.) are supported.
 */
class FP16Compressor : public GradientCompressor {
public:
    FP16Compressor() = default;
    ~FP16Compressor() override = default;

    /**
     * @brief Compress gradient by casting to Float16.
     *
     * @param gradient Input gradient tensor (any floating-point dtype)
     * @return CompressedGradient with Float16 data and original dtype metadata
     */
    auto compress(Tensor& gradient) -> CompressedGradient override;

    /**
     * @brief Decompress by casting back to the original dtype.
     *
     * @param compressed CompressedGradient with Float16 data
     * @return Tensor in the original dtype and shape
     */
    auto decompress(CompressedGradient& compressed) -> Tensor override;

    auto name() const -> std::string override { return "FP16"; }
};

/**
 * @brief Top-K sparsification gradient compression with error feedback.
 *
 * Keeps only the top K% of gradient values (by absolute magnitude) and
 * zeros out the rest. This achieves high compression ratios (e.g., 99%
 * with ratio=0.01) while preserving convergence through error feedback:
 * the values that were zeroed out are accumulated into a residual buffer
 * and added to the next iteration's gradient before sparsification.
 *
 * The error feedback mechanism (also called "memory" or "momentum
 * correction") ensures that no gradient information is permanently lost --
 * small gradients that are repeatedly zeroed out will accumulate in the
 * residual until they become large enough to be transmitted.
 *
 * Algorithm:
 * 1. Add residual from previous iteration: gradient += residual
 * 2. Flatten gradient and compute absolute values
 * 3. Find the K-th largest absolute value as threshold
 * 4. Create mask: keep values where |grad| >= threshold
 * 5. Update residual: residual = gradient * (1 - mask)  [zeroed-out values]
 * 6. Apply mask: gradient = gradient * mask
 *
 * Reference: "Deep Gradient Compression" (Lin et al., 2018)
 *
 * @note The compressed gradient is still a full-size tensor with zeros.
 *       A true sparse representation would require sparse tensor support.
 *       The bandwidth saving comes from the fact that zero values compress
 *       well with additional entropy coding, or can be transmitted as
 *       sparse index-value pairs in a custom all-reduce implementation.
 */
class TopKCompressor : public GradientCompressor {
public:
    /**
     * @brief Construct TopK compressor with given sparsification ratio.
     *
     * @param ratio Fraction of gradient values to keep (e.g., 0.01 = top 1%).
     *              Must be in range (0.0, 1.0].
     * @throws std::invalid_argument if ratio is not in (0.0, 1.0]
     */
    explicit TopKCompressor(float ratio = 0.01f);

    ~TopKCompressor() override = default;

    /**
     * @brief Compress gradient by keeping only top-K% values.
     *
     * Applies error feedback (adds accumulated residuals from previous
     * iterations), then sparsifies by zeroing out all but the top-K%
     * largest values by absolute magnitude. Zeroed-out values are saved
     * into the residual buffer for the next iteration.
     *
     * The gradient is identified by its data pointer for residual tracking.
     * If the gradient shape or device changes between calls, the residual
     * buffer for that parameter is reset.
     *
     * @param gradient Input gradient tensor (modified in-place: small values zeroed)
     * @return CompressedGradient with sparsified data
     */
    auto compress(Tensor& gradient) -> CompressedGradient override;

    /**
     * @brief Decompress a TopK-compressed gradient (no-op).
     *
     * Since the compressed gradient is already in full tensor form (with
     * zeros for pruned values), decompression simply returns the data
     * as-is, reshaped to the original shape if needed.
     *
     * @param compressed CompressedGradient with sparsified data
     * @return The gradient tensor (unchanged from compressed.data)
     */
    auto decompress(CompressedGradient& compressed) -> Tensor override;

    auto name() const -> std::string override { return "TopK"; }

    /**
     * @brief Reset all error feedback residual buffers.
     *
     * Clears accumulated residuals for all tracked parameters. Should be
     * called when starting a new training run or after model parameter
     * changes that invalidate the residual state.
     */
    auto reset() -> void override;

    /**
     * @brief Get the sparsification ratio.
     *
     * @return Fraction of gradient values kept (e.g., 0.01 = top 1%)
     */
    auto ratio() const -> float { return ratio_; }

private:
    /**
     * @brief Move the residual stored under @p old_key to @p new_key.
     *
     * Called when compress() reassigns the caller's gradient to a freshly
     * allocated buffer (the fallback paths), which changes data_ptr(). The
     * residual must follow the new pointer so the next iteration's lookup
     * hits and the error-feedback accumulation is preserved.
     */
    auto rekey_residual(const void* old_key, const void* new_key) -> void;

    /**
     * @brief Re-key the residual without taking residuals_mutex_.
     *
     * Body of rekey_residual() minus the lock. The caller MUST already hold
     * residuals_mutex_. compress() invokes this from within its own locked
     * region; calling the public (locking) rekey_residual() there would
     * re-lock the non-recursive mutex and deadlock.
     */
    auto rekey_residual_locked(const void* old_key, const void* new_key) -> void;

    /** @brief Fraction of gradient values to keep */
    float ratio_;

    /**
     * @brief Error feedback residual buffers, keyed by gradient data pointer.
     *
     * Each parameter's gradient accumulates its own residual so that
     * zeroed-out values from one iteration are added back in the next.
     * The key is the raw data pointer of the gradient tensor, which is
     * stable across training iterations for a given parameter.
     */
    std::unordered_map<const void*, Tensor> residuals_;
    // Guards all residuals_ read/modify/rekey operations — the base-class contract
    // permits concurrent compress() calls, and the map is otherwise unsynchronized.
    mutable std::mutex residuals_mutex_;
};

} // namespace tenzor::distributed
