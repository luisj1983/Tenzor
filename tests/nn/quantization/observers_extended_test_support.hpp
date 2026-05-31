/**
 * @file observers_extended_test_support.hpp
 * @brief Test-only quantization observers (KLDivergence / Percentile / MSE).
 *
 * These observer classes were previously declared in
 * include/tenzor/nn/quantization/observer.hpp and defined in
 * src/nn/quantization/observer.cpp, but they are never constructed by
 * make_observer() nor used anywhere in production — their only users were the
 * extended observer tests. They were relocated here.
 *
 * Header-only (inline methods) so test_observers_extended.cpp can include it
 * without any CMake changes. Lives in namespace tenzor::nn::quantization so the
 * test call sites (which do `using namespace tenzor::nn::quantization;`) remain
 * unchanged.
 */

#pragma once

#include <tenzor/nn/quantization/observer.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/creation.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace nn {
namespace quantization {

/**
 * @brief KL-divergence observer for quantization calibration.
 *
 * Uses KL divergence between the reference floating-point distribution and
 * the simulated quantized distribution to find the optimal clipping threshold.
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

// ============================================================================
// KLDivergenceObserver
// ============================================================================

inline KLDivergenceObserver::KLDivergenceObserver(int64_t num_bins, int64_t num_quantized_bins)
    : num_bins_(num_bins), num_quantized_bins_(num_quantized_bins),
      histogram_(num_bins, 0.0f) {}

inline auto KLDivergenceObserver::observe(const Tensor& tensor) -> void {
    auto cpu_tensor = tensor.to(Device::cpu()).to(DType::Float32);
    const float* data = cpu_tensor.data<float>();
    int64_t n = cpu_tensor.numel();

    if (n == 0) return;

    // Update min/max
    for (int64_t i = 0; i < n; ++i) {
        if (total_count_ == 0 && i == 0) {
            min_val_ = max_val_ = data[0];
        }
        min_val_ = std::min(min_val_, data[i]);
        max_val_ = std::max(max_val_, data[i]);
    }

    // Build/update histogram
    float range = max_val_ - min_val_;
    if (range <= 0.0f) range = 1.0f;
    float bin_width = range / static_cast<float>(num_bins_);

    for (int64_t i = 0; i < n; ++i) {
        int64_t bin = static_cast<int64_t>((data[i] - min_val_) / bin_width);
        bin = std::clamp(bin, int64_t(0), num_bins_ - 1);
        histogram_[bin] += 1.0f;
    }
    total_count_ += n;
}

inline auto KLDivergenceObserver::find_optimal_threshold() const -> float {
    // Search over candidate thresholds to minimize KL divergence
    float best_kl = std::numeric_limits<float>::max();
    float best_threshold = max_val_;
    float range = max_val_ - min_val_;
    if (range <= 0.0f) return max_val_;

    float bin_width = range / static_cast<float>(num_bins_);

    for (int64_t num_active_bins = num_quantized_bins_; num_active_bins <= num_bins_; ++num_active_bins) {
        float threshold = min_val_ + static_cast<float>(num_active_bins) * bin_width;

        // Simulate quantization: map num_active_bins -> num_quantized_bins
        float q_bin_width = static_cast<float>(num_active_bins) / static_cast<float>(num_quantized_bins_);

        // Build simulated quantized distribution
        std::vector<float> q_dist(num_active_bins, 0.0f);
        for (int64_t i = 0; i < num_active_bins && i < num_bins_; ++i) {
            int64_t q_bin = static_cast<int64_t>(static_cast<float>(i) / q_bin_width);
            q_bin = std::clamp(q_bin, int64_t(0), num_quantized_bins_ - 1);
            q_dist[static_cast<int64_t>(q_bin * q_bin_width)] += histogram_[i];
        }

        // Spread quantized bins back to original resolution
        std::vector<float> expanded(num_active_bins, 0.0f);
        for (int64_t qb = 0; qb < num_quantized_bins_; ++qb) {
            int64_t start = static_cast<int64_t>(qb * q_bin_width);
            int64_t end = static_cast<int64_t>((qb + 1) * q_bin_width);
            end = std::min(end, static_cast<int64_t>(num_active_bins));
            int64_t count = 0;
            for (int64_t i = start; i < end; ++i) {
                if (histogram_[i] > 0) ++count;
            }
            if (count > 0) {
                float val = q_dist[start] / static_cast<float>(count);
                for (int64_t i = start; i < end; ++i) {
                    if (histogram_[i] > 0) expanded[i] = val;
                }
            }
        }

        // Compute KL divergence: sum(p * log(p / q))
        float kl = 0.0f;
        for (int64_t i = 0; i < num_active_bins && i < num_bins_; ++i) {
            float p = histogram_[i];
            float q = expanded[i];
            if (p > 0.0f && q > 0.0f) {
                kl += p * std::log(p / q);
            }
        }

        if (kl < best_kl) {
            best_kl = kl;
            best_threshold = threshold;
        }
    }

    return best_threshold;
}

inline auto KLDivergenceObserver::calculate_qparams(QuantDType dtype, QuantizationScheme scheme)
    -> QuantizationParams {
    float threshold = find_optimal_threshold();
    float abs_max = std::max(std::abs(min_val_), threshold);

    int32_t qmin, qmax;
    if (dtype == QuantDType::INT8) {
        qmin = -128; qmax = 127;
    } else if (dtype == QuantDType::UINT8) {
        qmin = 0; qmax = 255;
    } else {
        qmin = -8; qmax = 7;  // INT4
    }

    float scale_val;
    int32_t zp_val;

    if (scheme == QuantizationScheme::PerTensorSymmetric ||
        scheme == QuantizationScheme::PerChannelSymmetric) {
        scale_val = abs_max / static_cast<float>(qmax);
        zp_val = 0;
    } else {
        float range = threshold - min_val_;
        scale_val = range / static_cast<float>(qmax - qmin);
        zp_val = qmin - static_cast<int32_t>(std::round(min_val_ / scale_val));
    }

    if (scale_val <= 0.0f) scale_val = 1.0f;

    // Construct QuantizationParams with Tensor scale and zero_point
    Tensor scale_tensor = tenzor::full({1}, scale_val, DType::Float32, Device::cpu());
    Tensor zp_tensor = tenzor::full({1}, static_cast<float>(zp_val), DType::Int32, Device::cpu());
    return QuantizationParams(std::move(scale_tensor), std::move(zp_tensor), dtype, scheme);
}

inline auto KLDivergenceObserver::reset() -> void {
    std::fill(histogram_.begin(), histogram_.end(), 0.0f);
    min_val_ = max_val_ = 0.0f;
    total_count_ = 0;
}

// ============================================================================
// PercentileObserver
// ============================================================================

inline PercentileObserver::PercentileObserver(double lower_percentile, double upper_percentile)
    : lower_percentile_(lower_percentile), upper_percentile_(upper_percentile) {}

inline auto PercentileObserver::observe(const Tensor& tensor) -> void {
    auto cpu_tensor = tensor.device() != Device::cpu() ? tensor.to(Device::cpu()) : tensor;
    auto contiguous = cpu_tensor.contiguous();
    const float* data = contiguous.data<float>();
    int64_t n = contiguous.numel();

    // Reservoir sampling to cap memory usage
    for (int64_t i = 0; i < n; ++i) {
        if (collected_values_.size() < kMaxSamples) {
            collected_values_.push_back(data[i]);
        } else {
            // Reservoir sampling: replace with decreasing probability
            size_t total = collected_values_.size() + i;
            size_t j = static_cast<size_t>(std::rand()) % total;
            if (j < kMaxSamples) {
                collected_values_[j] = data[i];
            }
        }
    }
}

inline auto PercentileObserver::calculate_qparams(QuantDType dtype, QuantizationScheme scheme)
    -> QuantizationParams {
    if (collected_values_.empty()) {
        throw std::runtime_error("PercentileObserver: no data observed");
    }

    // Sort to find percentiles
    std::vector<float> sorted = collected_values_;
    std::sort(sorted.begin(), sorted.end());

    size_t lo_idx = static_cast<size_t>(lower_percentile_ * (sorted.size() - 1));
    size_t hi_idx = static_cast<size_t>(upper_percentile_ * (sorted.size() - 1));
    float min_val = sorted[lo_idx];
    float max_val = sorted[hi_idx];

    // Compute scale and zero_point using standard formulas
    int64_t qmin = (dtype == QuantDType::INT8) ? -128 : 0;
    int64_t qmax = (dtype == QuantDType::INT8) ? 127 : 255;
    if (dtype == QuantDType::INT4) { qmin = -8; qmax = 7; }
    if (dtype == QuantDType::UINT4) { qmin = 0; qmax = 15; }

    float scale_val = (max_val - min_val) / static_cast<float>(qmax - qmin);
    if (scale_val == 0.0f) scale_val = 1.0f;
    int32_t zp_val = static_cast<int32_t>(std::round(static_cast<float>(qmin) - min_val / scale_val));
    zp_val = std::clamp(zp_val, static_cast<int32_t>(qmin), static_cast<int32_t>(qmax));

    if (scheme == QuantizationScheme::PerTensorSymmetric ||
        scheme == QuantizationScheme::PerChannelSymmetric) {
        float abs_max = std::max(std::abs(min_val), std::abs(max_val));
        scale_val = abs_max / static_cast<float>(qmax);
        if (scale_val == 0.0f) scale_val = 1.0f;
        zp_val = 0;
    }

    Tensor scale_tensor = tenzor::full({1}, scale_val, DType::Float32, Device::cpu());
    Tensor zp_tensor = tenzor::full({1}, static_cast<float>(zp_val), DType::Int32, Device::cpu());
    return QuantizationParams(std::move(scale_tensor), std::move(zp_tensor), dtype, scheme);
}

inline auto PercentileObserver::reset() -> void {
    collected_values_.clear();
}

// ============================================================================
// MSEObserver
// ============================================================================

inline MSEObserver::MSEObserver(int64_t num_candidates) : num_candidates_(num_candidates) {}

inline auto MSEObserver::observe(const Tensor& tensor) -> void {
    auto cpu_tensor = tensor.device() != Device::cpu() ? tensor.to(Device::cpu()) : tensor;
    auto contiguous = cpu_tensor.contiguous();
    const float* data = contiguous.data<float>();
    int64_t n = contiguous.numel();

    for (int64_t i = 0; i < n; ++i) {
        if (collected_values_.size() < kMaxSamples) {
            collected_values_.push_back(data[i]);
        } else {
            size_t total = collected_values_.size() + i;
            size_t j = static_cast<size_t>(std::rand()) % total;
            if (j < kMaxSamples) {
                collected_values_[j] = data[i];
            }
        }
    }
}

inline auto MSEObserver::calculate_qparams(QuantDType dtype, QuantizationScheme scheme)
    -> QuantizationParams {
    if (collected_values_.empty()) {
        throw std::runtime_error("MSEObserver: no data observed");
    }

    int64_t qmin = (dtype == QuantDType::INT8) ? -128 : 0;
    int64_t qmax = (dtype == QuantDType::INT8) ? 127 : 255;
    if (dtype == QuantDType::INT4) { qmin = -8; qmax = 7; }
    if (dtype == QuantDType::UINT4) { qmin = 0; qmax = 15; }

    float abs_min = *std::min_element(collected_values_.begin(), collected_values_.end());
    float abs_max = *std::max_element(collected_values_.begin(), collected_values_.end());

    float best_scale = 1.0f;
    int32_t best_zp = 0;
    double best_mse = std::numeric_limits<double>::max();

    // Grid search over candidate scale values
    for (int64_t c = 1; c <= num_candidates_; ++c) {
        float fraction = static_cast<float>(c) / static_cast<float>(num_candidates_);
        float candidate_max = abs_max * fraction;
        float candidate_min = abs_min * fraction;

        float scale_val;
        int32_t zp_val;
        if (scheme == QuantizationScheme::PerTensorSymmetric ||
            scheme == QuantizationScheme::PerChannelSymmetric) {
            float sym_max = std::max(std::abs(candidate_min), std::abs(candidate_max));
            scale_val = sym_max / static_cast<float>(qmax);
            zp_val = 0;
        } else {
            scale_val = (candidate_max - candidate_min) / static_cast<float>(qmax - qmin);
            zp_val = static_cast<int32_t>(std::round(static_cast<float>(qmin) - candidate_min / scale_val));
            zp_val = std::clamp(zp_val, static_cast<int32_t>(qmin), static_cast<int32_t>(qmax));
        }

        if (scale_val <= 0.0f) continue;

        // Compute MSE for this candidate
        double mse = 0.0;
        for (float v : collected_values_) {
            int32_t q = static_cast<int32_t>(std::round(v / scale_val)) + zp_val;
            q = std::clamp(q, static_cast<int32_t>(qmin), static_cast<int32_t>(qmax));
            float dequant = static_cast<float>(q - zp_val) * scale_val;
            double diff = static_cast<double>(v - dequant);
            mse += diff * diff;
        }
        mse /= static_cast<double>(collected_values_.size());

        if (mse < best_mse) {
            best_mse = mse;
            best_scale = scale_val;
            best_zp = zp_val;
        }
    }

    Tensor scale_tensor = tenzor::full({1}, best_scale, DType::Float32, Device::cpu());
    Tensor zp_tensor = tenzor::full({1}, static_cast<float>(best_zp), DType::Int32, Device::cpu());
    return QuantizationParams(std::move(scale_tensor), std::move(zp_tensor), dtype, scheme);
}

inline auto MSEObserver::reset() -> void {
    collected_values_.clear();
}

} // namespace quantization
} // namespace nn
} // namespace tenzor
