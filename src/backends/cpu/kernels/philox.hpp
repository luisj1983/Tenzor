/**
 * @file philox.hpp
 * @brief Philox 4x32-10 counter-based PRNG shared across CPU kernels.
 *
 * Counter-based RNGs are ideal for parallel tensor generation:
 *   - No shared mutable state between threads.
 *   - Deterministic: philox(seed, element_index) always produces the same value
 *     regardless of how many OMP threads are running.
 *   - Statistically excellent (passes BigCrush).
 *   - Fast: ~10 multiply-and-XOR rounds per 4 uint32 outputs.
 *
 * Usage:
 *   // Uniform float in [0, 1)
 *   float v = philox_uniform_f32(seed, element_index);
 *
 *   // Standard normal: one value is produced per element index from a single
 *   // Philox invocation (no Box-Muller pairing or index striding required).
 *   float v = philox_normal_f32(seed, element_index);
 */

#pragma once

#include <cstdint>
#include <cmath>

namespace tenzor::cpu::philox {

// ============================================================================
// Philox 4x32-10 core
// ============================================================================

struct Philox4x32 {
    uint32_t counter[4];
    uint32_t key[2];

    static void philox_round(uint32_t* ctr, const uint32_t* key) noexcept {
        constexpr uint64_t M0 = 0xD2511F53ULL;
        constexpr uint64_t M1 = 0xCD9E8D57ULL;

        uint64_t prod0 = M0 * static_cast<uint64_t>(ctr[0]);
        uint64_t prod1 = M1 * static_cast<uint64_t>(ctr[2]);

        uint32_t hi0 = static_cast<uint32_t>(prod0 >> 32);
        uint32_t lo0 = static_cast<uint32_t>(prod0);
        uint32_t hi1 = static_cast<uint32_t>(prod1 >> 32);
        uint32_t lo1 = static_cast<uint32_t>(prod1);

        ctr[0] = hi1 ^ ctr[1] ^ key[0];
        ctr[1] = lo1;
        ctr[2] = hi0 ^ ctr[3] ^ key[1];
        ctr[3] = lo0;
    }

    static void bump_key(uint32_t* key) noexcept {
        constexpr uint32_t W0 = 0x9E3779B9U;  // golden ratio
        constexpr uint32_t W1 = 0xBB67AE85U;  // sqrt(3) - 1
        key[0] += W0;
        key[1] += W1;
    }

    void generate(uint32_t output[4]) const noexcept {
        uint32_t ctr[4] = {counter[0], counter[1], counter[2], counter[3]};
        uint32_t k[2]   = {key[0], key[1]};
        for (int r = 0; r < 10; ++r) {
            philox_round(ctr, k);
            bump_key(k);
        }
        output[0] = ctr[0];
        output[1] = ctr[1];
        output[2] = ctr[2];
        output[3] = ctr[3];
    }

    static float uint32_to_uniform(uint32_t x) noexcept {
        // Top 24 bits → mantissa of float in [0, 1)
        return static_cast<float>(x >> 8) * (1.0f / 16777216.0f);
    }
};

// ============================================================================
// Convenience helpers: keyed by (seed, element_index)
// ============================================================================

/**
 * @brief Generate one uniform float in [0,1) for element at index `idx`.
 *
 * The counter encodes the element index in {counter[0], counter[1]} and zero
 * elsewhere. The key encodes the seed in {key[0], key[1]}.
 * Result is fully determined by (seed, idx) — independent of thread count.
 */
inline float philox_uniform_f32(uint64_t seed, int64_t idx) noexcept {
    Philox4x32 rng;
    rng.counter[0] = static_cast<uint32_t>(idx & 0xFFFFFFFFULL);
    rng.counter[1] = static_cast<uint32_t>((idx >> 32) & 0xFFFFFFFFULL);
    rng.counter[2] = 0u;
    rng.counter[3] = 0u;
    rng.key[0] = static_cast<uint32_t>(seed & 0xFFFFFFFFULL);
    rng.key[1] = static_cast<uint32_t>((seed >> 32) & 0xFFFFFFFFULL);

    uint32_t out[4];
    rng.generate(out);
    return Philox4x32::uint32_to_uniform(out[0]);
}

/**
 * @brief Generate one standard normal sample for element at index `idx`.
 *
 * Uses Box-Muller on two uniform values from the same Philox invocation
 * (out[0] and out[1]). This means every pair (idx, idx+1) shares the same
 * Philox call internally, but the result is still deterministic.
 */
inline float philox_normal_f32(uint64_t seed, int64_t idx) noexcept {
    Philox4x32 rng;
    rng.counter[0] = static_cast<uint32_t>(idx & 0xFFFFFFFFULL);
    rng.counter[1] = static_cast<uint32_t>((idx >> 32) & 0xFFFFFFFFULL);
    rng.counter[2] = 0u;
    rng.counter[3] = 0u;
    rng.key[0] = static_cast<uint32_t>(seed & 0xFFFFFFFFULL);
    rng.key[1] = static_cast<uint32_t>((seed >> 32) & 0xFFFFFFFFULL);

    uint32_t out[4];
    rng.generate(out);

    float u1 = Philox4x32::uint32_to_uniform(out[0]);
    float u2 = Philox4x32::uint32_to_uniform(out[1]);
    // Clamp u1 away from 0 to avoid log(0)
    if (u1 < 1e-37f) u1 = 1e-37f;
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
}

/**
 * @brief Generate one uniform double in [0,1) for element at index `idx`.
 */
inline double philox_uniform_f64(uint64_t seed, int64_t idx) noexcept {
    Philox4x32 rng;
    rng.counter[0] = static_cast<uint32_t>(idx & 0xFFFFFFFFULL);
    rng.counter[1] = static_cast<uint32_t>((idx >> 32) & 0xFFFFFFFFULL);
    rng.counter[2] = 0u;
    rng.counter[3] = 0u;
    rng.key[0] = static_cast<uint32_t>(seed & 0xFFFFFFFFULL);
    rng.key[1] = static_cast<uint32_t>((seed >> 32) & 0xFFFFFFFFULL);

    uint32_t out[4];
    rng.generate(out);
    // Combine two 32-bit values into a 53-bit mantissa for double
    uint64_t bits = (static_cast<uint64_t>(out[0]) << 21) | (static_cast<uint64_t>(out[1]) >> 11);
    return static_cast<double>(bits) * (1.0 / 9007199254740992.0);  // 1 / 2^53
}

/**
 * @brief Generate one standard normal double for element at index `idx`.
 */
inline double philox_normal_f64(uint64_t seed, int64_t idx) noexcept {
    // Build the Box-Muller uniform pair from the four independent outputs of a
    // single Philox block (out[0..3]), mirroring philox_normal_f32. The prior
    // implementation drew u1 and u2 from two separate single-block calls that
    // shared the same counter and only XOR'd the key, correlating the pair and
    // producing a non-Gaussian distribution.
    Philox4x32 rng;
    rng.counter[0] = static_cast<uint32_t>(idx & 0xFFFFFFFFULL);
    rng.counter[1] = static_cast<uint32_t>((idx >> 32) & 0xFFFFFFFFULL);
    rng.counter[2] = 0u;
    rng.counter[3] = 0u;
    rng.key[0] = static_cast<uint32_t>(seed & 0xFFFFFFFFULL);
    rng.key[1] = static_cast<uint32_t>((seed >> 32) & 0xFFFFFFFFULL);

    uint32_t out[4];
    rng.generate(out);
    uint64_t bits1 = (static_cast<uint64_t>(out[0]) << 21) | (static_cast<uint64_t>(out[1]) >> 11);
    uint64_t bits2 = (static_cast<uint64_t>(out[2]) << 21) | (static_cast<uint64_t>(out[3]) >> 11);
    double u1 = static_cast<double>(bits1) * (1.0 / 9007199254740992.0);
    double u2 = static_cast<double>(bits2) * (1.0 / 9007199254740992.0);
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

} // namespace tenzor::cpu::philox
