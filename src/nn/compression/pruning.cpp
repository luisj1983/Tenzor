/**
 * @file pruning.cpp
 * @brief Implementation of model pruning algorithms
 */

#include "tenzor/nn/compression/pruning.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_set>

namespace tenzor {
namespace nn {
namespace compression {

// =============================================================================
// PruningMask Implementation
// =============================================================================

auto PruningMask::apply(const Tensor& weights) const -> Tensor {
    return weights * mask;
}

auto PruningMask::compute_sparsity() const -> float {
    // Move to CPU first if on GPU, then convert to Float32 for processing
    Tensor mask_cpu = mask;
    if (mask.device() != Device::cpu()) {
        mask_cpu = mask.to(Device::cpu());
    }
    if (mask_cpu.dtype() != DType::Float32) {
        mask_cpu = mask_cpu.to(DType::Float32);
    }
    auto mask_data = mask_cpu.data<float>();
    int64_t total = mask.numel();
    int64_t zeros = 0;

    for (int64_t i = 0; i < total; ++i) {
        if (mask_data[i] == 0.0f) {
            zeros++;
        }
    }

    return static_cast<float>(zeros) / static_cast<float>(total);
}

// =============================================================================
// PruningConfig Implementation
// =============================================================================

auto PruningConfig::get_current_sparsity() const -> float {
    if (schedule == PruningSchedule::OneShot) {
        return target_sparsity;
    }

    if (current_iteration >= num_iterations) {
        return target_sparsity;
    }

    float progress = static_cast<float>(current_iteration) / static_cast<float>(num_iterations);

    if (schedule == PruningSchedule::Iterative) {
        // Linear schedule
        return target_sparsity * progress;
    } else {
        // Polynomial schedule (cubic)
        return target_sparsity * std::pow(progress, 3.0f);
    }
}

// =============================================================================
// Importance Scoring
// =============================================================================

auto compute_importance(const Tensor& weights, ImportanceCriterion criterion) -> Tensor {
    switch (criterion) {
        case ImportanceCriterion::L1: {
            // L1 norm: absolute values
            return abs(weights);
        }

        case ImportanceCriterion::L2: {
            // L2 norm: squared values
            return weights * weights;
        }

        case ImportanceCriterion::L1Norm: {
            // L1 normalized by count: |w| / numel()
            auto abs_weights = abs(weights);
            return abs_weights / static_cast<float>(weights.numel());
        }

        case ImportanceCriterion::L2Norm: {
            // L2 normalized by count: w^2 / numel()
            auto squared = weights * weights;
            return squared / static_cast<float>(weights.numel());
        }

        default:
            throw std::runtime_error("Unknown importance criterion");
    }
}

auto create_mask_from_importance(const Tensor& importance, float sparsity) -> Tensor {
    // Create mask by thresholding importance scores
    int64_t total = importance.numel();
    int64_t num_to_prune = static_cast<int64_t>(total * sparsity);

    // Move to CPU first if on GPU, then convert to Float32 for processing
    Tensor imp_cpu = importance;
    if (importance.device() != Device::cpu()) {
        imp_cpu = importance.to(Device::cpu());
    }
    if (imp_cpu.dtype() != DType::Float32) {
        imp_cpu = imp_cpu.to(DType::Float32);
    }

    // Copy importance scores to vector for sorting
    std::vector<std::pair<float, int64_t>> scores;
    scores.reserve(total);

    auto imp_data = imp_cpu.data<float>();
    for (int64_t i = 0; i < total; ++i) {
        scores.emplace_back(imp_data[i], i);
    }

    // Sort by importance (ascending - lowest importance first)
    std::sort(scores.begin(), scores.end());

    // Create mask as Float32 first for processing
    auto shape_span = importance.shape();
    std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
    Tensor mask_f32(shape_vec, DType::Float32, Device::cpu());
    auto mask_data = mask_f32.data<float>();

    // Initialize all to 1 (keep)
    for (int64_t i = 0; i < total; ++i) {
        mask_data[i] = 1.0f;
    }

    // Set lowest importance weights to 0 (prune)
    for (int64_t i = 0; i < num_to_prune; ++i) {
        mask_data[scores[i].second] = 0.0f;
    }

    // Convert mask to original dtype and device
    Tensor mask = mask_f32;
    if (importance.dtype() != DType::Float32) {
        mask = mask_f32.to(importance.dtype());
    }
    if (importance.device() != Device::cpu()) {
        mask = mask.to(importance.device());
    }

    return mask;
}

// =============================================================================
// Unstructured Pruning
// =============================================================================

auto prune_unstructured(
    std::shared_ptr<Module> module,
    float sparsity,
    ImportanceCriterion criterion,
    bool global_pruning
) -> PruningConfig {
    PruningConfig config;
    config.target_sparsity = sparsity;
    config.criterion = criterion;
    config.schedule = PruningSchedule::OneShot;

    auto named_params = module->named_parameters();

    if (global_pruning) {
        // Global pruning: compute threshold across all layers
        std::vector<float> all_importances;

        // Collect all importance scores
        for (auto& [name, param] : named_params) {
            if (name.find("weight") != std::string::npos) {
                auto importance = compute_importance(param->tensor(), criterion);
                // Move to CPU first if on GPU, then convert to Float32 for data access
                Tensor imp_cpu = importance;
                if (importance.device() != Device::cpu()) {
                    imp_cpu = importance.to(Device::cpu());
                }
                if (imp_cpu.dtype() != DType::Float32) {
                    imp_cpu = imp_cpu.to(DType::Float32);
                }
                auto imp_data = imp_cpu.data<float>();
                for (int64_t i = 0; i < importance.numel(); ++i) {
                    all_importances.push_back(imp_data[i]);
                }
            }
        }

        // Find global threshold
        std::sort(all_importances.begin(), all_importances.end());
        int64_t threshold_idx = static_cast<int64_t>(all_importances.size() * sparsity);
        float threshold = all_importances[threshold_idx];

        // Create masks based on global threshold
        for (auto& [name, param] : named_params) {
            if (name.find("weight") != std::string::npos) {
                auto importance = compute_importance(param->tensor(), criterion);
                // Move to CPU first if on GPU, then convert to Float32 for data access
                Tensor imp_cpu = importance;
                if (importance.device() != Device::cpu()) {
                    imp_cpu = importance.to(Device::cpu());
                }
                if (imp_cpu.dtype() != DType::Float32) {
                    imp_cpu = imp_cpu.to(DType::Float32);
                }
                auto imp_data = imp_cpu.data<float>();

                // Create mask as Float32 on CPU first
                auto shape_span = importance.shape();
                std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
                Tensor mask_f32(shape_vec, DType::Float32, Device::cpu());
                auto mask_data = mask_f32.data<float>();

                for (int64_t i = 0; i < importance.numel(); ++i) {
                    mask_data[i] = (imp_data[i] > threshold) ? 1.0f : 0.0f;
                }

                // Convert mask to original dtype and device
                Tensor mask = mask_f32;
                if (importance.dtype() != DType::Float32) {
                    mask = mask_f32.to(importance.dtype());
                }
                if (importance.device() != Device::cpu()) {
                    mask = mask.to(importance.device());
                }

                PruningMask pm;
                pm.mask = mask;
                pm.layer_name = name;
                pm.current_sparsity = sparsity;
                config.masks[name] = pm;
            }
        }
    } else {
        // Layer-wise pruning: compute threshold per layer
        for (auto& [name, param] : named_params) {
            if (name.find("weight") != std::string::npos) {
                auto importance = compute_importance(param->tensor(), criterion);
                auto mask = create_mask_from_importance(importance, sparsity);

                PruningMask pm;
                pm.mask = mask;
                pm.layer_name = name;
                pm.current_sparsity = sparsity;
                config.masks[name] = pm;
            }
        }
    }

    return config;
}

auto prune_iterative(
    std::shared_ptr<Module> module,
    float target_sparsity,
    int num_iterations,
    PruningSchedule schedule,
    ImportanceCriterion criterion
) -> PruningConfig {
    PruningConfig config;
    config.target_sparsity = target_sparsity;
    config.criterion = criterion;
    config.schedule = schedule;
    config.num_iterations = num_iterations;
    config.current_iteration = 0;

    // Initialize masks with zeros (no pruning yet)
    auto named_params = module->named_parameters();
    for (auto& [name, param] : named_params) {
        if (name.find("weight") != std::string::npos) {
            // Convert std::span to std::vector for constructor
            auto shape_span = param->tensor().shape();
            std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
            Tensor mask(shape_vec, param->tensor().dtype(), param->tensor().device());
            mask.fill_(1.0f);  // Start with no pruning

            PruningMask pm;
            pm.mask = mask;
            pm.layer_name = name;
            pm.current_sparsity = 0.0f;
            config.masks[name] = pm;
        }
    }

    return config;
}

// =============================================================================
// Structured Pruning
// =============================================================================

auto prune_channels(
    std::shared_ptr<Module> module,
    float sparsity,
    ImportanceCriterion criterion
) -> std::shared_ptr<Module> {
    // Structured channel pruning: remove entire output channels from Conv2d layers

    // Try to cast to Conv2d
    auto conv = std::dynamic_pointer_cast<Conv2d>(module);
    if (!conv) {
        // Not a Conv2d layer, apply unstructured pruning fallback
        auto config = prune_unstructured(module, sparsity, criterion, false);
        apply_pruning_masks(module, config);
        return module;
    }

    // Get the weight parameter
    auto named_params = conv->named_parameters();
    Tensor weight;
    bool has_bias = false;
    Tensor bias;

    for (auto& [name, param] : named_params) {
        if (name.find("weight") != std::string::npos) {
            weight = param->tensor();
        } else if (name.find("bias") != std::string::npos) {
            has_bias = true;
            bias = param->tensor();
        }
    }

    // Weight shape: [out_channels, in_channels_per_group, kernel_h, kernel_w]
    auto weight_shape = weight.shape();
    int64_t out_channels = weight_shape[0];
    int64_t in_channels_per_group = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Compute importance score for each output channel
    std::vector<std::pair<float, int64_t>> channel_importance;
    channel_importance.reserve(out_channels);

    // Move to CPU first if on GPU, then convert to Float32 for data access
    Tensor weight_cpu = weight;
    if (weight.device() != Device::cpu()) {
        weight_cpu = weight.to(Device::cpu());
    }
    if (weight_cpu.dtype() != DType::Float32) {
        weight_cpu = weight_cpu.to(DType::Float32);
    }
    auto* weight_data = weight_cpu.data<float>();

    for (int64_t oc = 0; oc < out_channels; ++oc) {
        float importance = 0.0f;
        int64_t channel_size = in_channels_per_group * kernel_h * kernel_w;
        int64_t channel_offset = oc * channel_size;

        // Compute importance based on criterion
        if (criterion == ImportanceCriterion::L1) {
            for (int64_t i = 0; i < channel_size; ++i) {
                importance += std::abs(weight_data[channel_offset + i]);
            }
        } else { // L2
            for (int64_t i = 0; i < channel_size; ++i) {
                float val = weight_data[channel_offset + i];
                importance += val * val;
            }
            importance = std::sqrt(importance);
        }

        channel_importance.emplace_back(importance, oc);
    }

    // Sort by importance (ascending - lowest importance first)
    std::sort(channel_importance.begin(), channel_importance.end());

    // Determine how many channels to keep (at least 1)
    // Calculate directly to match test expectations
    int64_t channels_to_keep = static_cast<int64_t>(out_channels * (1.0f - sparsity));
    channels_to_keep = std::max(channels_to_keep, static_cast<int64_t>(1));
    int64_t channels_to_prune = out_channels - channels_to_keep;

    // Collect indices of channels to keep (skip the lowest importance ones)
    std::vector<int64_t> keep_indices;
    keep_indices.reserve(channels_to_keep);
    for (int64_t i = channels_to_prune; i < out_channels; ++i) {
        keep_indices.push_back(channel_importance[i].second);
    }
    std::sort(keep_indices.begin(), keep_indices.end());

    // Create new Conv2d with reduced out_channels
    // Get original Conv2d parameters (need to extract from the module)
    // Since we don't have direct access to internal params, we'll create with defaults
    // and copy the weights manually

    // Extract original parameters (we need to infer them from weight shape)
    int64_t in_channels_total = in_channels_per_group; // assuming groups=1 for simplicity
    int64_t groups = 1; // default

    auto pruned_conv = std::make_shared<Conv2d>(
        in_channels_total,
        channels_to_keep,
        kernel_h,
        1,  // stride (default)
        0,  // padding (default)
        1,  // dilation (default)
        groups,
        has_bias
    );

    // Copy weights for kept channels
    auto pruned_params = pruned_conv->named_parameters();
    Tensor new_weight;
    Tensor new_bias;

    for (auto& [name, param] : pruned_params) {
        if (name.find("weight") != std::string::npos) {
            new_weight = param->tensor();
        } else if (name.find("bias") != std::string::npos) {
            new_bias = param->tensor();
        }
    }

    // Move new_weight to CPU first if on GPU, then convert to Float32 for data access
    Tensor new_weight_cpu = new_weight;
    if (new_weight.device() != Device::cpu()) {
        new_weight_cpu = new_weight.to(Device::cpu());
    }
    if (new_weight_cpu.dtype() != DType::Float32) {
        new_weight_cpu = new_weight_cpu.to(DType::Float32);
    }
    auto* new_weight_data = new_weight_cpu.data<float>();
    int64_t channel_size = in_channels_per_group * kernel_h * kernel_w;

    for (int64_t i = 0; i < channels_to_keep; ++i) {
        int64_t old_channel_idx = keep_indices[i];
        int64_t old_offset = old_channel_idx * channel_size;
        int64_t new_offset = i * channel_size;

        std::copy_n(weight_data + old_offset, channel_size, new_weight_data + new_offset);
    }

    // Copy bias if present
    if (has_bias) {
        // Move bias to CPU first if on GPU, then convert to Float32
        Tensor bias_cpu = bias;
        if (bias.device() != Device::cpu()) {
            bias_cpu = bias.to(Device::cpu());
        }
        if (bias_cpu.dtype() != DType::Float32) {
            bias_cpu = bias_cpu.to(DType::Float32);
        }
        // Move new_bias to CPU first if on GPU, then convert to Float32
        Tensor new_bias_cpu = new_bias;
        if (new_bias.device() != Device::cpu()) {
            new_bias_cpu = new_bias.to(Device::cpu());
        }
        if (new_bias_cpu.dtype() != DType::Float32) {
            new_bias_cpu = new_bias_cpu.to(DType::Float32);
        }
        auto* bias_data = bias_cpu.data<float>();
        auto* new_bias_data = new_bias_cpu.data<float>();

        for (int64_t i = 0; i < channels_to_keep; ++i) {
            new_bias_data[i] = bias_data[keep_indices[i]];
        }
    }

    return pruned_conv;
}

auto prune_filters(
    std::shared_ptr<Module> module,
    float sparsity,
    ImportanceCriterion criterion
) -> std::shared_ptr<Module> {
    // Filter pruning: similar to channel pruning but removes entire filters
    // For Conv2d, this is essentially the same as channel pruning (removes output channels)
    // The distinction is conceptual: filters refer to the 3D kernels, channels to output feature maps

    return prune_channels(module, sparsity, criterion);
}

auto prune_layers(
    std::shared_ptr<Module> module,
    int num_layers,
    ImportanceCriterion criterion
) -> std::shared_ptr<Module> {
    // Layer pruning implementation based on layer importance
    //
    // Algorithm:
    // 1. Compute layer importance scores based on weight magnitudes
    // 2. Identify least important layers
    // 3. Create pruning masks that zero out all weights in those layers
    // 4. Apply masks to effectively remove layers
    //
    // Note: This doesn't physically remove layers from the module structure,
    // but zeros all their weights, making them non-functional (identity-like).
    // For true architectural pruning, the module structure would need to be rebuilt.

    auto named_params = module->named_parameters();

    if (named_params.empty()) {
        return module;  // Nothing to prune
    }

    // Compute importance score for each layer based on weight magnitudes
    std::vector<std::pair<std::string, float>> layer_importance;
    std::unordered_map<std::string, std::string> param_to_layer;

    for (auto& [name, param] : named_params) {
        if (name.find("weight") == std::string::npos) continue;

        // Extract layer name (everything before ".weight")
        size_t weight_pos = name.find(".weight");
        std::string layer_name = (weight_pos != std::string::npos)
            ? name.substr(0, weight_pos)
            : name;

        param_to_layer[name] = layer_name;

        // Compute layer importance using the specified criterion
        auto importance = compute_importance(param->tensor(), criterion);
        // Move to CPU first if on GPU, then convert to Float32 for data access
        Tensor imp_cpu = importance;
        if (importance.device() != Device::cpu()) {
            imp_cpu = importance.to(Device::cpu());
        }
        if (imp_cpu.dtype() != DType::Float32) {
            imp_cpu = imp_cpu.to(DType::Float32);
        }
        auto imp_data = imp_cpu.data<float>();

        // Aggregate importance across all weights in the layer
        float total_importance = 0.0f;
        for (int64_t i = 0; i < importance.numel(); ++i) {
            total_importance += imp_data[i];
        }

        // Normalize by number of parameters
        float avg_importance = total_importance / importance.numel();

        // Check if this layer already has an importance score
        bool found = false;
        for (auto& [ln, score] : layer_importance) {
            if (ln == layer_name) {
                score += avg_importance;  // Accumulate if multiple weight tensors
                found = true;
                break;
            }
        }

        if (!found) {
            layer_importance.emplace_back(layer_name, avg_importance);
        }
    }

    // Limit num_layers to available layers
    int layers_to_prune = std::min(num_layers, static_cast<int>(layer_importance.size()));

    if (layers_to_prune <= 0) {
        return module;  // Nothing to prune
    }

    // Sort layers by importance (ascending - least important first)
    std::sort(layer_importance.begin(), layer_importance.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // Identify layers to prune (least important ones)
    std::unordered_set<std::string> layers_to_remove;
    for (int i = 0; i < layers_to_prune; ++i) {
        layers_to_remove.insert(layer_importance[i].first);
    }

    // Create pruning masks that zero out all weights in selected layers
    PruningConfig config;
    config.target_sparsity = static_cast<float>(layers_to_prune) / layer_importance.size();
    config.criterion = criterion;
    config.schedule = PruningSchedule::OneShot;

    for (auto& [name, param] : named_params) {
        auto it = param_to_layer.find(name);
        if (it == param_to_layer.end()) continue;

        const std::string& layer_name = it->second;

        // Create mask as Float32 first for processing
        auto shape_span = param->tensor().shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor mask_f32(shape_vec, DType::Float32, Device::cpu());

        auto mask_data = mask_f32.data<float>();
        int64_t numel = mask_f32.numel();

        if (layers_to_remove.find(layer_name) != layers_to_remove.end()) {
            // Zero out this layer completely
            for (int64_t i = 0; i < numel; ++i) {
                mask_data[i] = 0.0f;
            }
        } else {
            // Keep this layer
            for (int64_t i = 0; i < numel; ++i) {
                mask_data[i] = 1.0f;
            }
        }

        // Convert mask to original dtype and device
        Tensor mask = mask_f32;
        if (param->tensor().dtype() != DType::Float32) {
            mask = mask_f32.to(param->tensor().dtype());
        }
        if (param->tensor().device() != Device::cpu()) {
            mask = mask.to(param->tensor().device());
        }

        PruningMask pm;
        pm.mask = mask;
        pm.layer_name = name;
        pm.current_sparsity = layers_to_remove.find(layer_name) != layers_to_remove.end() ? 1.0f : 0.0f;
        config.masks[name] = pm;
    }

    // Apply the masks to the module
    apply_pruning_masks(module, config);

    return module;
}

// =============================================================================
// Mask Management
// =============================================================================

auto apply_pruning_masks(
    std::shared_ptr<Module> module,
    const PruningConfig& config
) -> void {
    auto named_params = module->named_parameters();

    for (auto& [name, param] : named_params) {
        auto it = config.masks.find(name);
        if (it != config.masks.end()) {
            const auto& mask = it->second;
            // Apply mask to weights
            auto masked_weights = mask.apply(param->tensor());
            // Update parameter (in-place)
            param->tensor() = masked_weights;
        }
    }
}

auto register_pruning_auto_reapply(
    optim::Optimizer& optimizer,
    std::shared_ptr<Module> module,
    PruningConfig config
) -> uint64_t {
    // Audit G.10: register an optimizer post-step hook that reapplies
    // the pruning masks after every parameter update.  Without this,
    // the optimizer silently un-prunes the zeroed-out positions on
    // each step (each gradient step adds a small delta back into
    // them).
    //
    // The hook captures the module by shared_ptr (so it stays alive),
    // the config by value (so subsequent mutations of the caller's
    // config object don't change what the hook applies — semantically
    // the masks are snapshotted at registration time), and dispatches
    // through the existing apply_pruning_masks entry point.
    //
    // S.14 (supersedes P.7): hook idempotence keyed on the optimizer's
    // own step_count(). The previous P.7 implementation kept a hook-local
    // atomic that ticked once per invocation — under threaded DataParallel
    // multiple concurrent step()s would each tick the counter and race
    // through apply_pruning_masks(), defeating de-duplication. Reading
    // step_count() from the base Optimizer gives a stable per-iteration
    // key shared across all hooks on the same optimizer instance, so
    // re-entrant calls in the same step coalesce correctly.
    auto last_applied_step = std::make_shared<std::atomic<uint64_t>>(
        std::numeric_limits<uint64_t>::max());
    return optimizer.register_post_step_hook(
        [module, config = std::move(config), last_applied_step,
         optimizer_ptr = &optimizer]() mutable {
            const uint64_t iter = optimizer_ptr->step_count();
            uint64_t prev = last_applied_step->load();
            if (prev != std::numeric_limits<uint64_t>::max() && prev >= iter) {
                // Already re-applied at this (or a later) step — skip.
                return;
            }
            // CAS so concurrent invocations in the same step coalesce.
            if (!last_applied_step->compare_exchange_strong(prev, iter)) {
                return;
            }
            apply_pruning_masks(module, config);
        });
}

auto finalize_pruning(
    std::shared_ptr<Module> module,
    const PruningConfig& config
) -> std::shared_ptr<Module> {
    // Apply masks one final time
    apply_pruning_masks(module, config);

    // In a full implementation, this would:
    // 1. Actually remove zero weights from storage
    // 2. Convert to sparse tensor format
    // 3. Optimize memory layout

    return module;
}

auto remove_pruning(
    [[maybe_unused]] std::shared_ptr<Module> module,
    PruningConfig& config
) -> void {
    // Clear all masks
    config.masks.clear();
    config.current_sparsity = 0.0f;
}

// =============================================================================
// Analysis and Utilities
// =============================================================================

auto compute_sparsity(const std::shared_ptr<Module>& module) -> float {
    auto params = module->parameters();

    int64_t total_params = 0;
    int64_t zero_params = 0;

    for (auto& param : params) {
        // Move to CPU first if on GPU, then convert to Float32 for data access
        Tensor tensor_cpu = param->tensor();
        if (param->tensor().device() != Device::cpu()) {
            tensor_cpu = param->tensor().to(Device::cpu());
        }
        if (tensor_cpu.dtype() != DType::Float32) {
            tensor_cpu = tensor_cpu.to(DType::Float32);
        }
        auto data = tensor_cpu.data<float>();
        int64_t numel = param->tensor().numel();
        total_params += numel;

        for (int64_t i = 0; i < numel; ++i) {
            if (std::abs(data[i]) < 1e-8f) {
                zero_params++;
            }
        }
    }

    if (total_params == 0) return 0.0f;
    return static_cast<float>(zero_params) / static_cast<float>(total_params);
}

auto analyze_layer_sparsity(
    const std::shared_ptr<Module>& module
) -> std::unordered_map<std::string, float> {
    std::unordered_map<std::string, float> layer_sparsity;

    auto named_params = module->named_parameters();
    for (auto& [name, param] : named_params) {
        // Move to CPU first if on GPU, then convert to Float32 for data access
        Tensor tensor_cpu = param->tensor();
        if (param->tensor().device() != Device::cpu()) {
            tensor_cpu = param->tensor().to(Device::cpu());
        }
        if (tensor_cpu.dtype() != DType::Float32) {
            tensor_cpu = tensor_cpu.to(DType::Float32);
        }
        auto data = tensor_cpu.data<float>();
        int64_t numel = param->tensor().numel();
        int64_t zeros = 0;

        for (int64_t i = 0; i < numel; ++i) {
            if (std::abs(data[i]) < 1e-8f) {
                zeros++;
            }
        }

        layer_sparsity[name] = static_cast<float>(zeros) / static_cast<float>(numel);
    }

    return layer_sparsity;
}

auto compute_compression_ratio(
    const std::shared_ptr<Module>& original_module,
    const std::shared_ptr<Module>& pruned_module
) -> float {
    auto orig_params = original_module->parameters();
    auto pruned_params = pruned_module->parameters();

    int64_t orig_count = 0;
    int64_t pruned_nonzero = 0;

    for (auto& param : orig_params) {
        orig_count += param->tensor().numel();
    }

    for (auto& param : pruned_params) {
        // Move to CPU first if on GPU, then convert to Float32 for data access
        Tensor tensor_cpu = param->tensor();
        if (param->tensor().device() != Device::cpu()) {
            tensor_cpu = param->tensor().to(Device::cpu());
        }
        if (tensor_cpu.dtype() != DType::Float32) {
            tensor_cpu = tensor_cpu.to(DType::Float32);
        }
        auto data = tensor_cpu.data<float>();
        for (int64_t i = 0; i < param->tensor().numel(); ++i) {
            if (std::abs(data[i]) >= 1e-8f) {
                pruned_nonzero++;
            }
        }
    }

    if (pruned_nonzero == 0) return INFINITY;
    return static_cast<float>(orig_count) / static_cast<float>(pruned_nonzero);
}

auto estimate_flops_reduction(
    const std::shared_ptr<Module>& module,
    [[maybe_unused]] const std::vector<int64_t>& input_shape
) -> float {
    // Simplified FLOP estimation
    float sparsity = compute_sparsity(module);
    // FLOPs reduction approximately equals sparsity for dense operations
    return sparsity;
}

auto sensitivity_analysis(
    std::shared_ptr<Module> module,
    std::function<float(std::shared_ptr<Module>)> validation_fn,
    const std::vector<float>& sparsity_levels
) -> std::unordered_map<std::string, std::vector<float>> {
    std::unordered_map<std::string, std::vector<float>> results;

    // Get baseline accuracy
    float baseline = validation_fn(module);

    auto named_params = module->named_parameters();

    // Test each layer individually
    for (auto& [name, param] : named_params) {
        if (name.find("weight") == std::string::npos) continue;

        std::vector<float> accuracy_drops;

        // Save original weights
        auto original_weights = param->tensor().clone();

        for (float sparsity : sparsity_levels) {
            // Prune this layer
            auto importance = compute_importance(param->tensor(), ImportanceCriterion::L1);
            auto mask_tensor = create_mask_from_importance(importance, sparsity);

            // Create PruningMask and apply it
            PruningMask pm;
            pm.mask = mask_tensor;
            pm.layer_name = name;
            pm.current_sparsity = sparsity;
            param->tensor() = pm.apply(param->tensor());

            // Evaluate
            float acc = validation_fn(module);
            accuracy_drops.push_back(baseline - acc);

            // Restore weights
            param->tensor() = original_weights.clone();
        }

        results[name] = accuracy_drops;
    }

    return results;
}

// =============================================================================
// Lottery Ticket Hypothesis
// =============================================================================

auto find_lottery_ticket(
    std::shared_ptr<Module> module,
    [[maybe_unused]] const std::unordered_map<std::string, Tensor>& initial_weights,
    float target_sparsity,
    int num_rounds
) -> PruningConfig {
    PruningConfig config;
    config.target_sparsity = target_sparsity;
    config.schedule = PruningSchedule::Iterative;
    config.num_iterations = num_rounds;

    float current_sparsity = 0.0f;
    float sparsity_step = target_sparsity / num_rounds;

    for (int round = 0; round < num_rounds; ++round) {
        current_sparsity += sparsity_step;

        // Prune to current sparsity level
        auto round_config = prune_unstructured(
            module,
            current_sparsity,
            ImportanceCriterion::L1,
            true
        );

        // Merge masks with config
        for (auto& [name, mask] : round_config.masks) {
            config.masks[name] = mask;
        }

        // Reset remaining weights to initialization (caller's responsibility to retrain)
    }

    return config;
}

} // namespace compression
} // namespace nn
} // namespace tenzor
