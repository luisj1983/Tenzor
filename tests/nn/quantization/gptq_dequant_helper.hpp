#pragma once

// Test-only reconstruction helper for GPTQ-quantized layers.
//
// GPTQResult (include/tenzor/nn/quantization/gptq.hpp) holds a packed INT4 (or
// Int8) weight plus per-group `scales` / `zeros`. Dequant is the standard
// affine rule W_recon[o,j] = (q[o,j] - zeros[o,g]) * scales[o,g], g =
// j/group_size. With desc_act=true the columns are in permuted (activation)
// order and `perm` maps permuted->original; these helpers assume the default
// desc_act=false (perm empty), which the round-trip tests use.
//
// Unlike AWQ, GPTQ performs Hessian-based error compensation, so the per-
// element weight error is *not* bounded by a half-quant-step; the quantity it
// minimises is the output reconstruction error. Tests therefore bound the
// aggregate relative Frobenius error of the reconstructed weight, which a
// correct quantizer keeps small and a garbage/zero quantizer cannot.

#include <cmath>
#include <cstdint>
#include <vector>

#include <tenzor/tenzor.hpp>
#include <tenzor/nn/quantization/gptq.hpp>

namespace tenzor {
namespace testing {
namespace gptq_detail {

inline int sign_extend_nibble(uint8_t nibble) {
    int v = nibble & 0x0F;
    if (v & 0x08) v -= 16;
    return v;
}

// Reconstruct the dequantized weight (row-major, out_features x in_features).
// Requires desc_act=false (perm empty).
inline std::vector<float> reconstruct_gptq(
    const ::tenzor::nn::quantization::GPTQResult& r, int group_size, bool sym) {
    using ::tenzor::DType;
    using ::tenzor::Device;

    const int64_t out_features = r.scales.size(0);
    const int64_t num_groups   = r.scales.size(1);
    const int64_t in_features  = r.in_features;

    auto scales_cpu = r.scales.to(Device::cpu()).to(DType::Float32);
    auto zeros_cpu  = r.zeros.to(Device::cpu()).to(DType::Float32);
    const float* scales_p = scales_cpu.template data<float>();
    const float* zeros_p  = zeros_cpu.template data<float>();

    auto packed_cpu = r.packed_weight.to(Device::cpu());

    std::vector<int> q(static_cast<size_t>(out_features * in_features), 0);
    if (packed_cpu.dtype() == DType::UInt8 && sym) {
        // INT4 packed, signed two's-complement nibbles.
        const int64_t packed_cols = (in_features + 1) / 2;
        const uint8_t* src = static_cast<const uint8_t*>(packed_cpu.data_ptr());
        for (int64_t o = 0; o < out_features; ++o) {
            for (int64_t c = 0; c < in_features; c += 2) {
                uint8_t byte = src[o * packed_cols + c / 2];
                q[o * in_features + c] = sign_extend_nibble(byte & 0x0F);
                if (c + 1 < in_features) {
                    q[o * in_features + c + 1] = sign_extend_nibble((byte >> 4) & 0x0F);
                }
            }
        }
    } else if (packed_cpu.dtype() == DType::UInt8) {
        // INT4 packed, unsigned nibbles.
        const int64_t packed_cols = (in_features + 1) / 2;
        const uint8_t* src = static_cast<const uint8_t*>(packed_cpu.data_ptr());
        for (int64_t o = 0; o < out_features; ++o) {
            for (int64_t c = 0; c < in_features; c += 2) {
                uint8_t byte = src[o * packed_cols + c / 2];
                q[o * in_features + c] = byte & 0x0F;
                if (c + 1 < in_features) {
                    q[o * in_features + c + 1] = (byte >> 4) & 0x0F;
                }
            }
        }
    } else {
        auto q8 = packed_cpu.to(DType::Int32);
        const int32_t* src = static_cast<const int32_t*>(q8.data_ptr());
        for (int64_t i = 0; i < out_features * in_features; ++i) q[i] = src[i];
    }

    std::vector<float> recon(static_cast<size_t>(out_features * in_features), 0.0f);
    for (int64_t o = 0; o < out_features; ++o) {
        for (int64_t j = 0; j < in_features; ++j) {
            int64_t g = j / group_size;
            float scale = scales_p[o * num_groups + g];
            float zero  = zeros_p[o * num_groups + g];
            recon[o * in_features + j] =
                (static_cast<float>(q[o * in_features + j]) - zero) * scale;
        }
    }
    return recon;
}

// Relative Frobenius error ||recon - ref|| / ||ref|| between a reconstructed
// weight and the original Float32 reference tensor.
inline double relative_frobenius_error(const std::vector<float>& recon,
                                       const ::tenzor::Tensor& ref) {
    auto ref_cpu = ref.to(::tenzor::Device::cpu()).to(::tenzor::DType::Float32);
    const float* ref_p = ref_cpu.template data<float>();
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < recon.size(); ++i) {
        double d = static_cast<double>(recon[i]) - static_cast<double>(ref_p[i]);
        num += d * d;
        den += static_cast<double>(ref_p[i]) * static_cast<double>(ref_p[i]);
    }
    if (den == 0.0) return num == 0.0 ? 0.0 : 1.0;
    return std::sqrt(num / den);
}

}  // namespace gptq_detail
}  // namespace testing
}  // namespace tenzor
