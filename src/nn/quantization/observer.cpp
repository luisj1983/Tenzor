/**
 * @file observer.cpp
 * @brief Implementation of quantization observers
 */

#include "tenzor/nn/quantization/observer.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include <algorithm>
#include <limits>
#include <cmath>

namespace tenzor {
namespace nn {
namespace quantization {

// ============================================================================
// MinMaxObserver
// ============================================================================

MinMaxObserver::MinMaxObserver(bool per_channel, int64_t axis)
    : per_channel_(per_channel), axis_(axis) {}

auto MinMaxObserver::observe(const Tensor& tensor) -> void {
    // Convert to Float32 for reduction (device-agnostic, dispatches to GPU kernel)
    Tensor tensor_f32 = tensor;
    if (tensor.dtype() != DType::Float32) {
        tensor_f32 = tensor.to(DType::Float32);
    }

    // Helper: reshape to [num_channels, -1] for per-channel reduction
    auto per_channel_reduce = [&](const Tensor& t, bool is_min) -> Tensor {
        auto shape = t.shape();
        int64_t num_channels = shape[axis_];
        int64_t rest = t.numel() / num_channels;
        Tensor reshaped;
        if (axis_ == 0) {
            reshaped = t.reshape({num_channels, rest});
        } else {
            reshaped = t.transpose(0, axis_).contiguous().reshape({num_channels, rest});
        }
        return is_min ? tenzor::min(reshaped, 1, false) : tenzor::max(reshaped, 1, false);
    };

    if (!has_data_) {
        // First observation - compute min/max using dispatched reductions
        if (per_channel_) {
            // Per-channel: reduce all non-channel dims
            min_val_ = per_channel_reduce(tensor_f32, true).to(Device::cpu());
            max_val_ = per_channel_reduce(tensor_f32, false).to(Device::cpu());
        } else {
            // Global min/max
            min_val_ = tenzor::min(tensor_f32).to(Device::cpu());
            max_val_ = tenzor::max(tensor_f32).to(Device::cpu());
        }
        has_data_ = true;
    } else {
        // Update existing min/max
        if (per_channel_) {
            auto new_min = per_channel_reduce(tensor_f32, true).to(Device::cpu());
            auto new_max = per_channel_reduce(tensor_f32, false).to(Device::cpu());

            // Element-wise min/max with existing values
            auto shape = min_val_.shape();
            int64_t num_channels = shape[0];
            float* min_data = min_val_.data<float>();
            float* max_data = max_val_.data<float>();
            const float* new_min_data = new_min.data<float>();
            const float* new_max_data = new_max.data<float>();

            for (int64_t c = 0; c < num_channels; ++c) {
                min_data[c] = std::min(min_data[c], new_min_data[c]);
                max_data[c] = std::max(max_data[c], new_max_data[c]);
            }
        } else {
            auto new_min = tenzor::min(tensor_f32).to(Device::cpu());
            auto new_max = tenzor::max(tensor_f32).to(Device::cpu());

            float* min_data = min_val_.data<float>();
            float* max_data = max_val_.data<float>();
            min_data[0] = std::min(min_data[0], new_min.item<float>());
            max_data[0] = std::max(max_data[0], new_max.item<float>());
        }
    }
}

auto MinMaxObserver::calculate_qparams(QuantDType dtype, QuantizationScheme scheme)
    -> QuantizationParams {
    if (!has_data_) {
        throw std::runtime_error("Cannot calculate qparams without observed data");
    }

    return compute_quantization_params(min_val_, max_val_, dtype, scheme);
}

auto MinMaxObserver::reset() -> void {
    has_data_ = false;
    min_val_ = Tensor();
    max_val_ = Tensor();
}

// ============================================================================
// MovingAverageMinMaxObserver
// ============================================================================

MovingAverageMinMaxObserver::MovingAverageMinMaxObserver(float momentum,
                                                         bool per_channel,
                                                         int64_t axis)
    : momentum_(momentum), per_channel_(per_channel), axis_(axis) {
    if (momentum < 0.0f || momentum > 1.0f) {
        throw std::runtime_error("Momentum must be in [0, 1]");
    }
}

auto MovingAverageMinMaxObserver::observe(const Tensor& tensor) -> void {
    // Convert to Float32 for reduction (device-agnostic)
    Tensor tensor_f32 = tensor;
    if (tensor.dtype() != DType::Float32) {
        tensor_f32 = tensor.to(DType::Float32);
    }

    // Helper: reshape to [num_channels, -1] for per-channel reduction
    auto per_channel_reduce = [&](const Tensor& t, bool is_min) -> Tensor {
        auto shape = t.shape();
        int64_t num_channels = shape[axis_];
        int64_t rest = t.numel() / num_channels;
        Tensor reshaped;
        if (axis_ == 0) {
            reshaped = t.reshape({num_channels, rest});
        } else {
            reshaped = t.transpose(0, axis_).contiguous().reshape({num_channels, rest});
        }
        return is_min ? tenzor::min(reshaped, 1, false) : tenzor::max(reshaped, 1, false);
    };

    if (!has_data_) {
        // First observation - compute min/max using dispatched reductions
        if (per_channel_) {
            min_val_ = per_channel_reduce(tensor_f32, true).to(Device::cpu());
            max_val_ = per_channel_reduce(tensor_f32, false).to(Device::cpu());
        } else {
            min_val_ = tenzor::min(tensor_f32).to(Device::cpu());
            max_val_ = tenzor::max(tensor_f32).to(Device::cpu());
        }
        has_data_ = true;
    } else {
        // Update with exponential moving average
        if (per_channel_) {
            auto new_min = per_channel_reduce(tensor_f32, true).to(Device::cpu());
            auto new_max = per_channel_reduce(tensor_f32, false).to(Device::cpu());

            // EMA update on CPU (small per-channel vectors)
            auto shape = min_val_.shape();
            int64_t num_channels = shape[0];
            float* min_data = min_val_.data<float>();
            float* max_data = max_val_.data<float>();
            const float* new_min_data = new_min.data<float>();
            const float* new_max_data = new_max.data<float>();

            for (int64_t c = 0; c < num_channels; ++c) {
                min_data[c] = momentum_ * min_data[c] + (1.0f - momentum_) * new_min_data[c];
                max_data[c] = momentum_ * max_data[c] + (1.0f - momentum_) * new_max_data[c];
            }
        } else {
            float new_min_v = tenzor::min(tensor_f32).to(Device::cpu()).item<float>();
            float new_max_v = tenzor::max(tensor_f32).to(Device::cpu()).item<float>();

            float* min_data = min_val_.data<float>();
            float* max_data = max_val_.data<float>();

            min_data[0] = momentum_ * min_data[0] + (1.0f - momentum_) * new_min_v;
            max_data[0] = momentum_ * max_data[0] + (1.0f - momentum_) * new_max_v;
        }
    }
}

auto MovingAverageMinMaxObserver::calculate_qparams(QuantDType dtype, QuantizationScheme scheme)
    -> QuantizationParams {
    if (!has_data_) {
        throw std::runtime_error("Cannot calculate qparams without observed data");
    }

    return compute_quantization_params(min_val_, max_val_, dtype, scheme);
}

auto MovingAverageMinMaxObserver::reset() -> void {
    has_data_ = false;
    min_val_ = Tensor();
    max_val_ = Tensor();
}

// ============================================================================
// HistogramObserver
// ============================================================================

HistogramObserver::HistogramObserver(int64_t num_bins,
                                    float percentile_low,
                                    float percentile_high)
    : num_bins_(num_bins),
      percentile_low_(percentile_low),
      percentile_high_(percentile_high) {
    histogram_.resize(num_bins, 0);
}

auto HistogramObserver::update_histogram(const Tensor& tensor) -> void {
    // Convert to Float32 for histogram binning
    Tensor tensor_f32 = tensor;
    if (tensor.dtype() != DType::Float32) {
        tensor_f32 = tensor.to(DType::Float32);
    }

    if (total_count_ == 0) {
        // First update - find initial range using dispatched reductions (GPU-safe)
        min_val_ = tenzor::min(tensor_f32).to(Device::cpu()).item<float>();
        max_val_ = tenzor::max(tensor_f32).to(Device::cpu()).item<float>();

        // Expand range slightly to avoid edge cases
        float range = max_val_ - min_val_;
        min_val_ -= range * 0.01f;
        max_val_ += range * 0.01f;
    }

    // Transfer to CPU for histogram binning (per-element access required)
    if (tensor_f32.device() != Device::cpu()) {
        tensor_f32 = tensor_f32.to(Device::cpu());
    }
    const float* data = tensor_f32.data<const float>();
    int64_t n = tensor_f32.numel();

    // Update histogram. When a value falls outside [min_val_, max_val_] we
    // must expand the histogram range AND re-bin the existing counts: each
    // old bin's midpoint maps into the new range. Without this redistribution
    // step (the previous "simplified" behaviour), old counts stay stuck at
    // their old indices and silently corrupt the calibration distribution
    // for any non-stationary input.
    float bin_width = (max_val_ - min_val_) / num_bins_;

    for (int64_t i = 0; i < n; ++i) {
        float val = data[i];

        // Update range if needed
        if (val < min_val_ || val > max_val_) {
            // Compute expanded range with a small slack so future values
            // close to the new boundary still bin cleanly.
            float new_min = std::min(min_val_, val);
            float new_max = std::max(max_val_, val);
            float range = new_max - new_min;
            new_min -= range * 0.01f;
            new_max += range * 0.01f;
            float new_bin_width = (new_max - new_min) / num_bins_;

            // Re-bin existing histogram counts using each old bin's midpoint.
            // The midpoint of old bin k is old_min + (k + 0.5) * old_bin_width;
            // route that count into whatever bin index it falls into under
            // the new (wider) range. This is the standard streaming-histogram
            // resize step and preserves total_count exactly.
            std::vector<int64_t> new_histogram(num_bins_, 0);
            const float old_min = min_val_;
            const float old_bin_width = bin_width;
            if (new_bin_width > 0.0f) {
                for (int64_t k = 0; k < num_bins_; ++k) {
                    int64_t count = histogram_[k];
                    if (count == 0) continue;
                    float midpoint = old_min + (static_cast<float>(k) + 0.5f) * old_bin_width;
                    int64_t new_bin = static_cast<int64_t>((midpoint - new_min) / new_bin_width);
                    new_bin = std::clamp(new_bin, int64_t(0), num_bins_ - 1);
                    new_histogram[new_bin] += count;
                }
            } else {
                // Degenerate range (all values identical so far); funnel all
                // existing counts into bin 0 of the new histogram.
                int64_t carry = 0;
                for (int64_t k = 0; k < num_bins_; ++k) carry += histogram_[k];
                new_histogram[0] = carry;
            }

            histogram_.swap(new_histogram);
            min_val_ = new_min;
            max_val_ = new_max;
            bin_width = new_bin_width;
        }

        // Add to histogram
        if (bin_width > 0.0f) {
            int64_t bin = static_cast<int64_t>((val - min_val_) / bin_width);
            bin = std::clamp(bin, int64_t(0), num_bins_ - 1);
            histogram_[bin]++;
        } else {
            histogram_[0]++;
        }
    }

    total_count_ += n;
}

auto HistogramObserver::compute_percentile(float percentile) const -> float {
    // Compute the target cumulative count in double precision to avoid float
    // rounding error for large total_count_, and floor at 1 so a truncated-to-zero
    // target does not return at i=0 when bin 0 is empty (cumulative=0 >= 0).
    int64_t target_count = std::max<int64_t>(
        1, std::llround(static_cast<double>(total_count_) * percentile));
    int64_t cumulative = 0;

    for (int64_t i = 0; i < num_bins_; ++i) {
        cumulative += histogram_[i];
        if (cumulative >= target_count) {
            float bin_width = (max_val_ - min_val_) / num_bins_;
            return min_val_ + (i + 0.5f) * bin_width;
        }
    }

    return max_val_;
}

auto HistogramObserver::observe(const Tensor& tensor) -> void {
    update_histogram(tensor);
}

auto HistogramObserver::calculate_qparams(QuantDType dtype, QuantizationScheme scheme)
    -> QuantizationParams {
    if (total_count_ == 0) {
        throw std::runtime_error("Cannot calculate qparams without observed data");
    }

    // Compute clipped range using percentiles
    float min_clip = compute_percentile(percentile_low_);
    float max_clip = compute_percentile(percentile_high_);

    Tensor min({1}, DType::Float32, Device::cpu());
    Tensor max({1}, DType::Float32, Device::cpu());
    min.fill_(min_clip);
    max.fill_(max_clip);

    return compute_quantization_params(min, max, dtype, scheme);
}

auto HistogramObserver::get_qrange() const -> std::pair<float, float> {
    // Same percentile-clipped range calculate_qparams() uses internally,
    // exposed so the per-channel wrapper reuses identical outlier rejection.
    if (total_count_ == 0) {
        throw std::runtime_error("Cannot compute qrange without observed data");
    }
    return {compute_percentile(percentile_low_), compute_percentile(percentile_high_)};
}

auto HistogramObserver::reset() -> void {
    std::fill(histogram_.begin(), histogram_.end(), 0);
    min_val_ = 0.0f;
    max_val_ = 0.0f;
    total_count_ = 0;
}

auto HistogramObserver::get_histogram() const
    -> std::tuple<std::vector<float>, std::vector<int64_t>> {
    std::vector<float> bin_edges;
    float bin_width = (max_val_ - min_val_) / num_bins_;

    for (int64_t i = 0; i <= num_bins_; ++i) {
        bin_edges.push_back(min_val_ + i * bin_width);
    }

    return {bin_edges, histogram_};
}

// ============================================================================
// PerChannelHistogramObserver
// ============================================================================

PerChannelHistogramObserver::PerChannelHistogramObserver(int64_t axis,
                                                         int64_t num_bins,
                                                         float percentile_low,
                                                         float percentile_high)
    : axis_(axis),
      num_bins_(num_bins),
      percentile_low_(percentile_low),
      percentile_high_(percentile_high) {}

auto PerChannelHistogramObserver::observe(const Tensor& tensor) -> void {
    // Convert to Float32 for data access if necessary
    Tensor tensor_f32 = tensor;
    if (tensor.dtype() != DType::Float32) {
        tensor_f32 = tensor.to(DType::Float32);
    }
    if (tensor_f32.device() != Device::cpu()) {
        tensor_f32 = tensor_f32.to(Device::cpu());
    }

    auto shape = tensor_f32.shape();
    int64_t num_channels = shape[axis_];

    // Initialize observers if needed
    if (channel_observers_.empty()) {
        for (int64_t i = 0; i < num_channels; ++i) {
            channel_observers_.push_back(
                std::make_unique<HistogramObserver>(num_bins_, percentile_low_, percentile_high_)
            );
        }
    }

    // Extract and observe each channel
    int64_t channel_size = tensor_f32.numel() / num_channels;
    const float* data = tensor_f32.data<const float>();

    for (int64_t c = 0; c < num_channels; ++c) {
        // Create temporary tensor for this channel on CPU
        Tensor channel_data({channel_size}, DType::Float32, Device::cpu());
        float* ch_ptr = channel_data.data<float>();

        for (int64_t i = 0; i < channel_size; ++i) {
            ch_ptr[i] = data[c * channel_size + i];
        }

        channel_observers_[c]->observe(channel_data);
    }
}

auto PerChannelHistogramObserver::calculate_qparams(QuantDType dtype, QuantizationScheme scheme)
    -> QuantizationParams {
    if (channel_observers_.empty()) {
        throw std::runtime_error("Cannot calculate qparams without observed data");
    }

    int64_t num_channels = channel_observers_.size();
    Tensor min({num_channels}, DType::Float32, Device::cpu());
    Tensor max({num_channels}, DType::Float32, Device::cpu());

    float* min_data = min.data<float>();
    float* max_data = max.data<float>();

    for (int64_t c = 0; c < num_channels; ++c) {
        // Use each channel histogram's percentile-clipped [min, max]. The
        // previous code wrote the per-channel *scale* into BOTH min and max,
        // producing min==max and therefore degenerate (wrong) qparams.
        auto [ch_min, ch_max] = channel_observers_[c]->get_qrange();
        min_data[c] = ch_min;
        max_data[c] = ch_max;
    }

    auto params = compute_quantization_params(min, max, dtype, scheme);
    params.axis = axis_;
    return params;
}

auto PerChannelHistogramObserver::reset() -> void {
    for (auto& obs : channel_observers_) {
        obs->reset();
    }
    channel_observers_.clear();
}

// ============================================================================
// Factory
// ============================================================================

auto make_observer(QuantizationScheme scheme, bool use_histogram, int64_t axis)
    -> std::unique_ptr<Observer> {
    bool per_channel = (scheme == QuantizationScheme::PerChannelSymmetric ||
                       scheme == QuantizationScheme::PerChannelAsymmetric);

    if (use_histogram) {
        if (per_channel) {
            return std::make_unique<PerChannelHistogramObserver>(axis);
        } else {
            return std::make_unique<HistogramObserver>();
        }
    } else {
        return std::make_unique<MinMaxObserver>(per_channel, axis);
    }
}

} // namespace quantization
} // namespace nn
} // namespace tenzor
