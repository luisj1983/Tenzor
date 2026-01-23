/**
 * @file observer.cpp
 * @brief Implementation of quantization observers
 */

#include "tenzor/nn/quantization/observer.hpp"
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
    // Convert to Float32 for data access if necessary
    Tensor tensor_f32 = tensor;
    if (tensor.dtype() != DType::Float32) {
        tensor_f32 = tensor.to(DType::Float32);
    }
    if (tensor_f32.device() != Device::cpu()) {
        tensor_f32 = tensor_f32.to(Device::cpu());
    }
    const float* data = tensor_f32.data<const float>();
    int64_t n = tensor_f32.numel();

    // Track original device for later
    auto original_device = tensor.device();

    if (!has_data_) {
        // First observation - initialize min/max on CPU
        if (per_channel_) {
            auto shape = tensor.shape();
            int64_t num_channels = shape[axis_];
            min_val_ = Tensor({num_channels}, DType::Float32, Device::cpu());
            max_val_ = Tensor({num_channels}, DType::Float32, Device::cpu());

            float* min_data = min_val_.data<float>();
            float* max_data = max_val_.data<float>();

            int64_t channel_size = n / num_channels;

            for (int64_t c = 0; c < num_channels; ++c) {
                float ch_min = data[c * channel_size];
                float ch_max = data[c * channel_size];

                for (int64_t i = 0; i < channel_size; ++i) {
                    float val = data[c * channel_size + i];
                    ch_min = std::min(ch_min, val);
                    ch_max = std::max(ch_max, val);
                }

                min_data[c] = ch_min;
                max_data[c] = ch_max;
            }

            // Keep on CPU (small scalars, accessed from CPU)
        } else {
            min_val_ = Tensor({1}, DType::Float32, Device::cpu());
            max_val_ = Tensor({1}, DType::Float32, Device::cpu());

            float min_v = data[0];
            float max_v = data[0];
            for (int64_t i = 1; i < n; ++i) {
                min_v = std::min(min_v, data[i]);
                max_v = std::max(max_v, data[i]);
            }

            min_val_.fill_(min_v);
            max_val_.fill_(max_v);
        }
        has_data_ = true;
    } else {
        // Update existing min/max
        if (per_channel_) {
            auto shape = tensor.shape();
            int64_t num_channels = shape[axis_];
            int64_t channel_size = n / num_channels;

            // Move to CPU for data access
            Tensor min_cpu = min_val_;
            Tensor max_cpu = max_val_;
            if (min_cpu.device() != Device::cpu()) {
                min_cpu = min_cpu.to(Device::cpu());
            }
            if (max_cpu.device() != Device::cpu()) {
                max_cpu = max_cpu.to(Device::cpu());
            }

            float* min_data = min_cpu.data<float>();
            float* max_data = max_cpu.data<float>();

            for (int64_t c = 0; c < num_channels; ++c) {
                for (int64_t i = 0; i < channel_size; ++i) {
                    float val = data[c * channel_size + i];
                    min_data[c] = std::min(min_data[c], val);
                    max_data[c] = std::max(max_data[c], val);
                }
            }

            // Keep on CPU (small scalars, accessed from CPU)
            min_val_ = min_cpu;
            max_val_ = max_cpu;
        } else {
            // Move to CPU for data access
            Tensor min_cpu = min_val_;
            Tensor max_cpu = max_val_;
            if (min_cpu.device() != Device::cpu()) {
                min_cpu = min_cpu.to(Device::cpu());
            }
            if (max_cpu.device() != Device::cpu()) {
                max_cpu = max_cpu.to(Device::cpu());
            }

            float* min_data = min_cpu.data<float>();
            float* max_data = max_cpu.data<float>();

            for (int64_t i = 0; i < n; ++i) {
                min_data[0] = std::min(min_data[0], data[i]);
                max_data[0] = std::max(max_data[0], data[i]);
            }

            // Keep on CPU (small scalars, accessed from CPU)
            min_val_ = min_cpu;
            max_val_ = max_cpu;
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
    // Convert to Float32 for data access if necessary
    Tensor tensor_f32 = tensor;
    if (tensor.dtype() != DType::Float32) {
        tensor_f32 = tensor.to(DType::Float32);
    }
    if (tensor_f32.device() != Device::cpu()) {
        tensor_f32 = tensor_f32.to(Device::cpu());
    }
    const float* data = tensor_f32.data<const float>();
    int64_t n = tensor_f32.numel();

    // Track original device for later
    auto original_device = tensor.device();

    if (!has_data_) {
        // Initialize with first observation on CPU
        if (per_channel_) {
            auto shape = tensor.shape();
            int64_t num_channels = shape[axis_];
            min_val_ = Tensor({num_channels}, DType::Float32, Device::cpu());
            max_val_ = Tensor({num_channels}, DType::Float32, Device::cpu());

            float* min_data = min_val_.data<float>();
            float* max_data = max_val_.data<float>();

            int64_t channel_size = n / num_channels;

            for (int64_t c = 0; c < num_channels; ++c) {
                float ch_min = data[c * channel_size];
                float ch_max = data[c * channel_size];

                for (int64_t i = 0; i < channel_size; ++i) {
                    float val = data[c * channel_size + i];
                    ch_min = std::min(ch_min, val);
                    ch_max = std::max(ch_max, val);
                }

                min_data[c] = ch_min;
                max_data[c] = ch_max;
            }

            // Keep on CPU (small scalars, accessed from CPU)
        } else {
            min_val_ = Tensor({1}, DType::Float32, Device::cpu());
            max_val_ = Tensor({1}, DType::Float32, Device::cpu());

            float min_v = data[0];
            float max_v = data[0];
            for (int64_t i = 1; i < n; ++i) {
                min_v = std::min(min_v, data[i]);
                max_v = std::max(max_v, data[i]);
            }

            min_val_.fill_(min_v);
            max_val_.fill_(max_v);
        }
        has_data_ = true;
    } else {
        // Update with exponential moving average
        if (per_channel_) {
            auto shape = tensor.shape();
            int64_t num_channels = shape[axis_];
            int64_t channel_size = n / num_channels;

            // Move to CPU for data access
            Tensor min_cpu = min_val_;
            Tensor max_cpu = max_val_;
            if (min_cpu.device() != Device::cpu()) {
                min_cpu = min_cpu.to(Device::cpu());
            }
            if (max_cpu.device() != Device::cpu()) {
                max_cpu = max_cpu.to(Device::cpu());
            }

            float* min_data = min_cpu.data<float>();
            float* max_data = max_cpu.data<float>();

            for (int64_t c = 0; c < num_channels; ++c) {
                float ch_min = data[c * channel_size];
                float ch_max = data[c * channel_size];

                for (int64_t i = 0; i < channel_size; ++i) {
                    float val = data[c * channel_size + i];
                    ch_min = std::min(ch_min, val);
                    ch_max = std::max(ch_max, val);
                }

                min_data[c] = momentum_ * min_data[c] + (1.0f - momentum_) * ch_min;
                max_data[c] = momentum_ * max_data[c] + (1.0f - momentum_) * ch_max;
            }

            // Keep on CPU (small scalars, accessed from CPU)
            min_val_ = min_cpu;
            max_val_ = max_cpu;
        } else {
            float min_v = data[0];
            float max_v = data[0];
            for (int64_t i = 1; i < n; ++i) {
                min_v = std::min(min_v, data[i]);
                max_v = std::max(max_v, data[i]);
            }

            // Move to CPU for data access
            Tensor min_cpu = min_val_;
            Tensor max_cpu = max_val_;
            if (min_cpu.device() != Device::cpu()) {
                min_cpu = min_cpu.to(Device::cpu());
            }
            if (max_cpu.device() != Device::cpu()) {
                max_cpu = max_cpu.to(Device::cpu());
            }

            float* min_data = min_cpu.data<float>();
            float* max_data = max_cpu.data<float>();

            min_data[0] = momentum_ * min_data[0] + (1.0f - momentum_) * min_v;
            max_data[0] = momentum_ * max_data[0] + (1.0f - momentum_) * max_v;

            // Keep on CPU (small scalars, accessed from CPU)
            min_val_ = min_cpu;
            max_val_ = max_cpu;
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
    // Convert to Float32 for data access if necessary
    Tensor tensor_f32 = tensor;
    if (tensor.dtype() != DType::Float32) {
        tensor_f32 = tensor.to(DType::Float32);
    }
    if (tensor_f32.device() != Device::cpu()) {
        tensor_f32 = tensor_f32.to(Device::cpu());
    }
    const float* data = tensor_f32.data<const float>();
    int64_t n = tensor_f32.numel();

    if (total_count_ == 0) {
        // First update - find initial range
        min_val_ = data[0];
        max_val_ = data[0];
        for (int64_t i = 1; i < n; ++i) {
            min_val_ = std::min(min_val_, data[i]);
            max_val_ = std::max(max_val_, data[i]);
        }

        // Expand range slightly to avoid edge cases
        float range = max_val_ - min_val_;
        min_val_ -= range * 0.01f;
        max_val_ += range * 0.01f;
    }

    // Update histogram
    float bin_width = (max_val_ - min_val_) / num_bins_;

    for (int64_t i = 0; i < n; ++i) {
        float val = data[i];

        // Update range if needed
        if (val < min_val_ || val > max_val_) {
            // Need to rebuild histogram with new range
            float new_min = std::min(min_val_, val);
            float new_max = std::max(max_val_, val);
            float range = new_max - new_min;
            new_min -= range * 0.01f;
            new_max += range * 0.01f;

            // Rebuild histogram (simplified - in production, would redistribute existing bins)
            min_val_ = new_min;
            max_val_ = new_max;
            bin_width = (max_val_ - min_val_) / num_bins_;
        }

        // Add to histogram
        int64_t bin = static_cast<int64_t>((val - min_val_) / bin_width);
        bin = std::clamp(bin, int64_t(0), num_bins_ - 1);
        histogram_[bin]++;
    }

    total_count_ += n;
}

auto HistogramObserver::compute_percentile(float percentile) const -> float {
    int64_t target_count = static_cast<int64_t>(total_count_ * percentile);
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
        auto ch_params = channel_observers_[c]->calculate_qparams(dtype, scheme);
        // Move scale to CPU for data access
        Tensor scale_cpu = ch_params.scale;
        if (scale_cpu.device() != Device::cpu()) {
            scale_cpu = scale_cpu.to(Device::cpu());
        }
        min_data[c] = scale_cpu.data<const float>()[0];  // Simplified
        max_data[c] = scale_cpu.data<const float>()[0];
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
