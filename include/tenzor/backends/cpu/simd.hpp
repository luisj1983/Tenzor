/**
 * @file simd.hpp
 * @brief SIMD optimizations for CPU backend with runtime feature detection
 *
 * Provides vectorized implementations of common operations using:
 * - AVX-512 (512-bit vectors, 16 floats)
 * - AVX2 (256-bit vectors, 8 floats)
 * - SSE4.2 (128-bit vectors, 4 floats)
 * - Scalar fallback
 *
 * Features automatic runtime CPU detection and dispatch to best available implementation.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace tenzor {
namespace cpu {

/**
 * @brief CPU instruction set features
 */
enum class CPUFeature : uint32_t {
    None     = 0,
    SSE      = 1 << 0,
    SSE2     = 1 << 1,
    SSE3     = 1 << 2,
    SSSE3    = 1 << 3,
    SSE41    = 1 << 4,
    SSE42    = 1 << 5,
    AVX      = 1 << 6,
    AVX2     = 1 << 7,
    FMA      = 1 << 8,
    AVX512F  = 1 << 9,
    AVX512DQ = 1 << 10,
    AVX512BW = 1 << 11,
    AVX512VL = 1 << 12,
    AVX512VNNI = 1 << 13,
};

inline CPUFeature operator|(CPUFeature a, CPUFeature b) {
    return static_cast<CPUFeature>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline CPUFeature operator&(CPUFeature a, CPUFeature b) {
    return static_cast<CPUFeature>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

inline bool has_feature(CPUFeature features, CPUFeature test) {
    return static_cast<uint32_t>(features & test) != 0;
}

/**
 * @brief CPU feature detector using CPUID
 *
 * Detects available CPU features at runtime and provides
 * dispatch to optimal SIMD implementation.
 */
class CPUInfo {
public:
    /**
     * @brief Get singleton instance
     */
    static auto get() -> const CPUInfo&;

    /**
     * @brief Get detected CPU features
     */
    auto features() const -> CPUFeature { return features_; }

    /**
     * @brief Check if specific feature is available
     */
    auto has(CPUFeature feature) const -> bool {
        return has_feature(features_, feature);
    }

    /**
     * @brief Get CPU vendor string
     */
    auto vendor() const -> const std::string& { return vendor_; }

    /**
     * @brief Get CPU brand string
     */
    auto brand() const -> const std::string& { return brand_; }

    /**
     * @brief Get human-readable feature list
     */
    auto feature_string() const -> std::string;

    /**
     * @brief Check if AVX-512 is available
     */
    auto has_avx512() const -> bool {
        return has(CPUFeature::AVX512F);
    }

    /**
     * @brief Check if AVX2 is available
     */
    auto has_avx2() const -> bool {
        return has(CPUFeature::AVX2);
    }

    /**
     * @brief Check if SSE4.2 is available
     */
    auto has_sse42() const -> bool {
        return has(CPUFeature::SSE42);
    }

    /**
     * @brief Check if AVX512-VNNI is available
     */
    auto has_avx512_vnni() const -> bool {
        return has(CPUFeature::AVX512VNNI);
    }

private:
    CPUInfo();

    CPUFeature features_{CPUFeature::None};
    std::string vendor_;
    std::string brand_;

    /**
     * @brief Execute CPUID instruction
     */
    static auto cpuid(uint32_t leaf, uint32_t subleaf = 0)
        -> std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;

    /**
     * @brief Detect CPU features using CPUID
     */
    auto detect_features() -> void;
};

/**
 * @brief SIMD dispatch for vectorized operations
 *
 * Provides runtime dispatch to best available SIMD implementation
 * based on detected CPU features.
 */
namespace simd {

/**
 * @brief Vectorized element-wise add: out[i] = a[i] + b[i]
 *
 * @param a First input array
 * @param b Second input array
 * @param out Output array
 * @param size Number of elements
 */
auto add(const float* a, const float* b, float* out, size_t size) -> void;

/**
 * @brief Vectorized element-wise subtract: out[i] = a[i] - b[i]
 */
auto sub(const float* a, const float* b, float* out, size_t size) -> void;

/**
 * @brief Vectorized element-wise multiply: out[i] = a[i] * b[i]
 */
auto mul(const float* a, const float* b, float* out, size_t size) -> void;

/**
 * @brief Vectorized element-wise divide: out[i] = a[i] / b[i]
 */
auto div(const float* a, const float* b, float* out, size_t size) -> void;

/**
 * @brief Vectorized square root: out[i] = sqrt(a[i])
 */
auto sqrt(const float* a, float* out, size_t size) -> void;

/**
 * @brief Vectorized exponential: out[i] = exp(a[i])
 */
auto exp(const float* a, float* out, size_t size) -> void;

/**
 * @brief Vectorized natural logarithm: out[i] = log(a[i])
 */
auto log(const float* a, float* out, size_t size) -> void;

/**
 * @brief Vectorized ReLU: out[i] = max(0, a[i])
 */
auto relu(const float* a, float* out, size_t size) -> void;

/**
 * @brief Vectorized sigmoid: out[i] = 1 / (1 + exp(-a[i]))
 */
auto sigmoid(const float* a, float* out, size_t size) -> void;

/**
 * @brief Vectorized tanh: out[i] = tanh(a[i])
 */
auto tanh(const float* a, float* out, size_t size) -> void;

/**
 * @brief Vectorized GELU: out[i] = a[i] * Phi(a[i])
 *
 * where Phi(x) is the cumulative distribution function of the standard normal distribution.
 * Uses approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 */
auto gelu(const float* a, float* out, size_t size) -> void;

/**
 * @brief Vectorized fused multiply-add: out[i] = a[i] * b[i] + c[i]
 */
auto fma(const float* a, const float* b, const float* c, float* out, size_t size) -> void;

} // namespace simd

/**
 * @brief Scalar implementations (fallback)
 */
namespace scalar {

auto add(const float* a, const float* b, float* out, size_t size) -> void;
auto sub(const float* a, const float* b, float* out, size_t size) -> void;
auto mul(const float* a, const float* b, float* out, size_t size) -> void;
auto div(const float* a, const float* b, float* out, size_t size) -> void;
auto sqrt(const float* a, float* out, size_t size) -> void;
auto exp(const float* a, float* out, size_t size) -> void;
auto log(const float* a, float* out, size_t size) -> void;
auto relu(const float* a, float* out, size_t size) -> void;
auto sigmoid(const float* a, float* out, size_t size) -> void;
auto tanh(const float* a, float* out, size_t size) -> void;
auto gelu(const float* a, float* out, size_t size) -> void;
auto fma(const float* a, const float* b, const float* c, float* out, size_t size) -> void;

} // namespace scalar

/**
 * @brief AVX2 implementations (256-bit vectors)
 */
namespace avx2 {

auto add(const float* a, const float* b, float* out, size_t size) -> void;
auto sub(const float* a, const float* b, float* out, size_t size) -> void;
auto mul(const float* a, const float* b, float* out, size_t size) -> void;
auto div(const float* a, const float* b, float* out, size_t size) -> void;
auto sqrt(const float* a, float* out, size_t size) -> void;
auto exp(const float* a, float* out, size_t size) -> void;
auto log(const float* a, float* out, size_t size) -> void;
auto relu(const float* a, float* out, size_t size) -> void;
auto sigmoid(const float* a, float* out, size_t size) -> void;
auto tanh(const float* a, float* out, size_t size) -> void;
auto gelu(const float* a, float* out, size_t size) -> void;
auto fma(const float* a, const float* b, const float* c, float* out, size_t size) -> void;

} // namespace avx2

/**
 * @brief AVX-512 implementations (512-bit vectors)
 */
namespace avx512 {

auto add(const float* a, const float* b, float* out, size_t size) -> void;
auto sub(const float* a, const float* b, float* out, size_t size) -> void;
auto mul(const float* a, const float* b, float* out, size_t size) -> void;
auto div(const float* a, const float* b, float* out, size_t size) -> void;
auto sqrt(const float* a, float* out, size_t size) -> void;
auto exp(const float* a, float* out, size_t size) -> void;
auto log(const float* a, float* out, size_t size) -> void;
auto relu(const float* a, float* out, size_t size) -> void;
auto sigmoid(const float* a, float* out, size_t size) -> void;
auto tanh(const float* a, float* out, size_t size) -> void;
auto gelu(const float* a, float* out, size_t size) -> void;
auto fma(const float* a, const float* b, const float* c, float* out, size_t size) -> void;

} // namespace avx512

} // namespace cpu
} // namespace tenzor
