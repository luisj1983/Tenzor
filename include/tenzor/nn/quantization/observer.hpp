/**
 * @file observer.hpp
 * @brief Statistical observers for quantization calibration
 *
 * Provides observer classes for collecting activation statistics during
 * model execution to determine optimal quantization parameters for
 * post-training quantization (PTQ).
 */

#pragma once

#include <memory>
#include <vector>
#include <cmath>
#include "quantize.hpp"
#include "../../core/tensor.hpp"

namespace tenzor {
namespace nn {
namespace quantization {

/**
 * @brief Base class for quantization observers.
 *
 * Observers collect statistics from tensors (typically activations during
 * forward passes) to determine optimal quantization parameters. Different
 * observer strategies make different trade-offs between accuracy and simplicity.
 */
class Observer {
public:
    virtual ~Observer() = default;

    /**
     * @brief Process a tensor to update statistics.
     *
     * @param tensor Tensor to observe (typically activation)
     */
    virtual auto observe(const Tensor& tensor) -> void = 0;

    /**
     * @brief Calculate quantization parameters from collected statistics.
     *
     * @param dtype Target quantized data type
     * @param scheme Quantization scheme
     * @return Computed quantization parameters
     */
    virtual auto calculate_qparams(QuantDType dtype, QuantizationScheme scheme)
        -> QuantizationParams = 0;

    /**
     * @brief Reset observer statistics.
     */
    virtual auto reset() -> void = 0;

    /**
     * @brief Check if observer has collected any data.
     */
    virtual auto has_data() const -> bool = 0;
};

/**
 * @brief Min-max observer for quantization calibration.
 *
 * Tracks the minimum and maximum values observed across all tensors.
 * Simple and fast, but sensitive to outliers.
 *
 * Quantization parameters are computed as:
 * - Symmetric: scale = max(|min|, |max|) / (quant_max - quant_min)
 * - Asymmetric: scale = (max - min) / (quant_max - quant_min),
 *               zero_point = round(-min / scale)
 *
 * @code
 * MinMaxObserver observer;
 *
 * // Collect statistics during calibration
 * for (auto& batch : calibration_data) {
 *     auto output = model.forward(batch);
 *     observer.observe(output.tensor());
 * }
 *
 * // Calculate quantization parameters
 * auto qparams = observer.calculate_qparams(QuantDType::INT8,
 *                                          QuantizationScheme::PerTensorSymmetric);
 * @endcode
 */
class MinMaxObserver : public Observer {
public:
    MinMaxObserver() = default;

    /**
     * @brief Construct observer with optional per-channel support.
     *
     * @param per_channel If true, track min/max per channel
     * @param axis Channel axis for per-channel observation (default: 0)
     */
    explicit MinMaxObserver(bool per_channel, int64_t axis = 0);

    auto observe(const Tensor& tensor) -> void override;
    auto calculate_qparams(QuantDType dtype, QuantizationScheme scheme)
        -> QuantizationParams override;
    auto reset() -> void override;
    auto has_data() const -> bool override { return has_data_; }

    /**
     * @brief Get observed minimum value(s).
     */
    auto get_min() const -> const Tensor& { return min_val_; }

    /**
     * @brief Get observed maximum value(s).
     */
    auto get_max() const -> const Tensor& { return max_val_; }

private:
    Tensor min_val_;           ///< Minimum observed value(s)
    Tensor max_val_;           ///< Maximum observed value(s)
    bool has_data_{false};     ///< Whether any data has been observed
    bool per_channel_{false};  ///< Per-channel observation flag
    int64_t axis_{0};          ///< Channel axis for per-channel mode
};

/**
 * @brief Moving average min-max observer.
 *
 * Similar to MinMaxObserver but uses exponential moving average to
 * reduce sensitivity to outliers and adapt to changing distributions.
 *
 * Update rule:
 * min_avg = momentum * min_avg + (1 - momentum) * current_min
 * max_avg = momentum * max_avg + (1 - momentum) * current_max
 *
 * @code
 * MovingAverageMinMaxObserver observer(0.99);  // Heavy smoothing
 * @endcode
 */
class MovingAverageMinMaxObserver : public Observer {
public:
    /**
     * @brief Construct moving average observer.
     *
     * @param momentum Momentum for exponential moving average (0.0-1.0, default: 0.9)
     * @param per_channel Per-channel observation flag (default: false)
     * @param axis Channel axis for per-channel mode (default: 0)
     */
    explicit MovingAverageMinMaxObserver(float momentum = 0.9f,
                                        bool per_channel = false,
                                        int64_t axis = 0);

    auto observe(const Tensor& tensor) -> void override;
    auto calculate_qparams(QuantDType dtype, QuantizationScheme scheme)
        -> QuantizationParams override;
    auto reset() -> void override;
    auto has_data() const -> bool override { return has_data_; }

private:
    Tensor min_val_;           ///< Moving average of minimum
    Tensor max_val_;           ///< Moving average of maximum
    float momentum_;           ///< EMA momentum
    bool has_data_{false};     ///< Whether any data has been observed
    bool per_channel_{false};  ///< Per-channel observation flag
    int64_t axis_{0};          ///< Channel axis
};

/**
 * @brief Histogram observer for quantization calibration.
 *
 * Maintains a histogram of observed values and uses percentile-based
 * clipping to handle outliers better than min-max observers.
 *
 * The histogram method:
 * 1. Builds histogram of observed values
 * 2. Clips extreme percentiles (e.g., 0.01% and 99.99%)
 * 3. Computes quantization params from clipped range
 *
 * This approach is more robust to outliers but more computationally expensive.
 *
 * @code
 * HistogramObserver observer(2048);  // 2048 histogram bins
 * @endcode
 */
class HistogramObserver : public Observer {
public:
    /**
     * @brief Construct histogram observer.
     *
     * @param num_bins Number of histogram bins (default: 2048)
     * @param percentile_low Lower percentile for clipping (default: 0.0001 = 0.01%)
     * @param percentile_high Upper percentile for clipping (default: 0.9999 = 99.99%)
     */
    explicit HistogramObserver(int64_t num_bins = 2048,
                              float percentile_low = 0.0001f,
                              float percentile_high = 0.9999f);

    auto observe(const Tensor& tensor) -> void override;
    auto calculate_qparams(QuantDType dtype, QuantizationScheme scheme)
        -> QuantizationParams override;
    auto reset() -> void override;
    auto has_data() const -> bool override { return total_count_ > 0; }

    /**
     * @brief Get histogram data for analysis.
     *
     * @return Tuple of (bin_edges, bin_counts)
     */
    auto get_histogram() const -> std::tuple<std::vector<float>, std::vector<int64_t>>;

private:
    int64_t num_bins_;              ///< Number of histogram bins
    float percentile_low_;          ///< Lower clipping percentile
    float percentile_high_;         ///< Upper clipping percentile
    std::vector<int64_t> histogram_; ///< Histogram bin counts
    float min_val_{0.0f};           ///< Minimum observed value
    float max_val_{0.0f};           ///< Maximum observed value
    int64_t total_count_{0};        ///< Total number of values observed

    /**
     * @brief Update histogram with new tensor values.
     */
    auto update_histogram(const Tensor& tensor) -> void;

    /**
     * @brief Compute percentile from histogram.
     */
    auto compute_percentile(float percentile) const -> float;
};

/**
 * @brief Per-channel histogram observer.
 *
 * Maintains separate histograms for each channel, useful for weight
 * quantization where different output channels may have very different ranges.
 */
class PerChannelHistogramObserver : public Observer {
public:
    /**
     * @brief Construct per-channel histogram observer.
     *
     * @param axis Channel axis (typically 0 for weight tensors)
     * @param num_bins Number of histogram bins per channel (default: 2048)
     * @param percentile_low Lower percentile for clipping (default: 0.0001)
     * @param percentile_high Upper percentile for clipping (default: 0.9999)
     */
    explicit PerChannelHistogramObserver(int64_t axis,
                                        int64_t num_bins = 2048,
                                        float percentile_low = 0.0001f,
                                        float percentile_high = 0.9999f);

    auto observe(const Tensor& tensor) -> void override;
    auto calculate_qparams(QuantDType dtype, QuantizationScheme scheme)
        -> QuantizationParams override;
    auto reset() -> void override;
    auto has_data() const -> bool override { return !channel_observers_.empty(); }

private:
    int64_t axis_;                                         ///< Channel axis
    int64_t num_bins_;                                     ///< Bins per histogram
    float percentile_low_;                                 ///< Lower percentile
    float percentile_high_;                                ///< Upper percentile
    std::vector<std::unique_ptr<HistogramObserver>> channel_observers_;  ///< Per-channel observers
};

/**
 * @brief KL-divergence observer for quantization calibration.
 *
 * Uses KL divergence between the reference floating-point distribution and
 * the simulated quantized distribution to find the optimal clipping threshold.
 * Generally produces better accuracy than min-max or percentile approaches,
 * at the cost of higher calibration time.
 *
 * Algorithm:
 * 1. Collect histogram of observed values
 * 2. For each candidate threshold (from num_quantized_bins to num_bins):
 *    a. Simulate quantization at that threshold
 *    b. Compute KL(reference || quantized)
 * 3. Select threshold that minimizes KL divergence
 */
class KLDivergenceObserver : public Observer {
public:
    explicit KLDivergenceObserver(int64_t num_bins = 2048, int64_t num_quantized_bins = 128);

    auto observe(const Tensor& tensor) -> void override;
    auto calculate_qparams(QuantDType dtype, QuantizationScheme scheme)
        -> QuantizationParams override;
    auto reset() -> void override;
    auto has_data() const -> bool override { return total_count_ > 0; }

private:
    int64_t num_bins_;
    int64_t num_quantized_bins_;
    std::vector<float> histogram_;
    float min_val_{0.0f};
    float max_val_{0.0f};
    int64_t total_count_{0};

    auto find_optimal_threshold() const -> float;
};

/**
 * @brief Percentile-based observer for better outlier handling.
 *
 * Instead of using absolute min/max, uses configurable percentiles
 * (e.g., 0.1% and 99.9%) to determine the quantization range.
 * This reduces the impact of outlier values on quantization quality.
 */
class PercentileObserver : public Observer {
public:
    explicit PercentileObserver(double lower_percentile = 0.001,
                                 double upper_percentile = 0.999);

    auto observe(const Tensor& tensor) -> void override;
    auto calculate_qparams(QuantDType dtype, QuantizationScheme scheme)
        -> QuantizationParams override;
    auto reset() -> void override;
    auto has_data() const -> bool override { return !collected_values_.empty(); }

private:
    double lower_percentile_;
    double upper_percentile_;
    std::vector<float> collected_values_;  ///< Reservoir sample of observed values
    static constexpr size_t kMaxSamples = 100000;
};

/**
 * @brief MSE-minimizing observer.
 *
 * Finds scale and zero_point that minimize the mean squared error between
 * original and quantized-then-dequantized values via grid search over
 * candidate ranges.
 */
class MSEObserver : public Observer {
public:
    explicit MSEObserver(int64_t num_candidates = 100);

    auto observe(const Tensor& tensor) -> void override;
    auto calculate_qparams(QuantDType dtype, QuantizationScheme scheme)
        -> QuantizationParams override;
    auto reset() -> void override;
    auto has_data() const -> bool override { return !collected_values_.empty(); }

private:
    int64_t num_candidates_;
    std::vector<float> collected_values_;
    static constexpr size_t kMaxSamples = 100000;
};

/**
 * @brief Create appropriate observer for quantization scheme.
 *
 * Factory function that creates the right observer type based on the
 * quantization scheme and other parameters.
 *
 * @param scheme Quantization scheme
 * @param use_histogram Use histogram-based observer (default: false)
 * @param axis Channel axis for per-channel schemes (default: 0)
 * @return Observer instance
 */
auto make_observer(QuantizationScheme scheme,
                  bool use_histogram = false,
                  int64_t axis = 0) -> std::unique_ptr<Observer>;

} // namespace quantization
} // namespace nn
} // namespace tenzor
