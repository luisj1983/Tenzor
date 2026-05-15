/**
 * @file philox_dropout.cpp
 * @brief Deterministic Philox4x32-10 dropout mask — CPU host implementation.
 *
 * Audit F13/F22-followup: Tensor-level helper used by the FlashAttention
 * composed-ops dropout fallback (and its corresponding backward) so the
 * `seed` / `offset` tensors returned by the forward can be replayed
 * bit-exactly in the backward.
 *
 * The Philox4x32-10 round + constants match `src/autograd/function_attention.cpp`
 * exactly, so any backward path that already understands that counter
 * convention can replay the masks this helper produces.
 */

#include "tenzor/ops/philox_dropout.hpp"
#include "tenzor/ops/creation.hpp"

#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace tenzor {

namespace {

// =========================================================================
// Philox4x32-10 — mirrors src/autograd/function_attention.cpp.
// =========================================================================

constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;
constexpr uint32_t kPhiloxKeyXor = 0x1BD11BDAu;

inline void philox_round(uint32_t ctr[4], const uint32_t key[2]) {
    uint64_t prod0 = uint64_t(kPhiloxM0) * uint64_t(ctr[0]);
    uint64_t prod1 = uint64_t(kPhiloxM1) * uint64_t(ctr[2]);
    uint32_t hi0 = static_cast<uint32_t>(prod0 >> 32);
    uint32_t lo0 = static_cast<uint32_t>(prod0);
    uint32_t hi1 = static_cast<uint32_t>(prod1 >> 32);
    uint32_t lo1 = static_cast<uint32_t>(prod1);
    uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
    uint32_t new1 = lo1;
    uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
    uint32_t new3 = lo0;
    ctr[0] = new0; ctr[1] = new1; ctr[2] = new2; ctr[3] = new3;
}

inline float philox_uniform_f32(uint32_t c0, uint32_t c1, uint32_t c2,
                                  uint32_t c3, uint32_t seed_low) {
    uint32_t ctr[4] = {c0, c1, c2, c3};
    uint32_t k[2] = {seed_low, seed_low ^ kPhiloxKeyXor};
    for (int r = 0; r < 10; ++r) {
        philox_round(ctr, k);
        if (r < 9) { k[0] += kPhiloxW0; k[1] += kPhiloxW1; }
    }
    // 24-bit mantissa precision: match the CUDA/ROCm convention.
    return static_cast<float>(ctr[0] >> 8) * (1.0f / 16777216.0f);
}

// Fill a flat float buffer with the pre-scaled keep-or-zero mask. `bh_dim`,
// `sq_dim`, `sk_dim` decode the FA `[BH, Sq, Sk]` counter; for other ranks
// the helper degenerates to the plain `(idx, 0, 0, 0)` linear convention.
void fill_mask_f32(float* out,
                    int64_t total,
                    int64_t bh_dim,
                    int64_t sq_dim,
                    int64_t sk_dim,
                    float p,
                    uint32_t seed_low,
                    uint64_t offset) {
    const float keep_scale = (p < 1.0f) ? 1.0f / (1.0f - p) : 0.0f;
    if (bh_dim > 0 && sq_dim > 0 && sk_dim > 0 && bh_dim * sq_dim * sk_dim == total) {
        // Attention-shape Philox addressing.
        for (int64_t bh = 0; bh < bh_dim; ++bh) {
            for (int64_t qi = 0; qi < sq_dim; ++qi) {
                for (int64_t ki = 0; ki < sk_dim; ++ki) {
                    uint32_t c0 = static_cast<uint32_t>(bh);
                    uint32_t c1 = static_cast<uint32_t>(qi);
                    uint32_t c2 = static_cast<uint32_t>(ki);
                    uint32_t c3 = static_cast<uint32_t>(offset & 0xFFFFFFFFu);
                    float u = philox_uniform_f32(c0, c1, c2, c3, seed_low);
                    int64_t idx = (bh * sq_dim + qi) * sk_dim + ki;
                    out[idx] = (u >= p) ? keep_scale : 0.0f;
                }
            }
        }
    } else {
        // Generic linear addressing fallback.
        for (int64_t i = 0; i < total; ++i) {
            uint32_t c0 = static_cast<uint32_t>(i);
            uint32_t c3 = static_cast<uint32_t>(offset & 0xFFFFFFFFu);
            float u = philox_uniform_f32(c0, 0, 0, c3, seed_low);
            out[i] = (u >= p) ? keep_scale : 0.0f;
        }
    }
}

}  // namespace

auto philox_dropout_mask(const std::vector<int64_t>& shape,
                          double p,
                          uint64_t seed,
                          uint64_t offset,
                          DType dtype) -> Tensor {
    if (p < 0.0 || p >= 1.0) {
        throw std::invalid_argument(
            "philox_dropout_mask: dropout probability must be in [0, 1), got " +
            std::to_string(p));
    }
    int64_t total = 1;
    for (auto d : shape) total *= d;

    // Decode attention shape: prefer 4-D [B, H, Sq, Sk] (flatten to [BH, Sq, Sk]).
    int64_t bh = 0, sq = 0, sk = 0;
    if (shape.size() == 4) {
        bh = shape[0] * shape[1];
        sq = shape[2];
        sk = shape[3];
    } else if (shape.size() == 3) {
        bh = shape[0];
        sq = shape[1];
        sk = shape[2];
    }

    // Generate as Float32 first; narrow to the target dtype below.
    Tensor mask_f32(std::vector<int64_t>(shape.begin(), shape.end()),
                     DType::Float32, Device::cpu());
    fill_mask_f32(mask_f32.data<float>(), total, bh, sq, sk,
                   static_cast<float>(p),
                   static_cast<uint32_t>(seed & 0xFFFFFFFFu),
                   offset);

    if (dtype == DType::Float32) return mask_f32;
    return mask_f32.to(dtype);
}

auto new_philox_stream() -> PhiloxStream {
    static thread_local std::random_device rd;
    static thread_local std::mt19937_64 gen{rd()};

    uint64_t seed_v = gen();
    uint64_t offset_v = 0;  // The per-element counters in `philox_dropout_mask`
                            // already disambiguate elements; offset=0 is the
                            // canonical "first call" sentinel.

    Tensor seed_t({1}, DType::Int64, Device::cpu());
    Tensor offset_t({1}, DType::Int64, Device::cpu());
    seed_t.data<int64_t>()[0]   = static_cast<int64_t>(seed_v);
    offset_t.data<int64_t>()[0] = static_cast<int64_t>(offset_v);
    return {seed_t, offset_t};
}

}  // namespace tenzor
