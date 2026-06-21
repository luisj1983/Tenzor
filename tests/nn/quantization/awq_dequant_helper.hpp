#pragma once

// Test-only reconstruction helpers for AWQ-quantized layers.
//
// The AWQ quantizer (include/tenzor/nn/quantization/awq.hpp) returns an
// AWQResult with a *packed* INT4 (or Int8) weight plus per-group `scales`,
// per-group `zeros`, and per-input-channel `act_scales`. The header documents
// the exact dequant rule:
//
//   W_recon[o,j] = (q[o,j] - zeros[o,g]) * scales[o,g] / act_scales[j]
//
// where g = j / group_size and q is the (sign-extended) integer code. These
// helpers unpack and apply that formula so round-trip tests can compare the
// reconstructed weight against the original Float32 reference, and bound the
// residual by the analytic half-quant-step (the only error a *correct*
// quantizer may introduce).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <tenzor/tenzor.hpp>
#include <tenzor/nn/quantization/awq.hpp>

namespace tenzor {
namespace testing {
namespace awq_detail {

// Sign-extend a 4-bit nibble (two's complement) to a signed int.
inline int sign_extend_nibble(uint8_t nibble) {
    int v = nibble & 0x0F;
    if (v & 0x08) v -= 16;  // bit-3 set => negative
    return v;
}

// Reconstruct the dequantized weight matrix (out_features x in_features) from
// an AWQResult produced with the given group_size. Returns a flat row-major
// std::vector<float>. Assumes bits==4 (packed UInt8) or bits==8 (Int8) and
// symmetric quantization (zeros == 0), the AWQQuantizer default.
inline std::vector<float> reconstruct_awq(
    const ::tenzor::nn::quantization::AWQResult& r, int group_size) {
    using ::tenzor::DType;
    using ::tenzor::Device;

    const int64_t out_features = r.scales.size(0);
    const int64_t num_groups   = r.scales.size(1);
    const int64_t in_features  = r.in_features;

    auto scales_cpu = r.scales.to(Device::cpu()).to(DType::Float32);
    auto zeros_cpu  = r.zeros.to(Device::cpu()).to(DType::Float32);
    auto act_cpu    = r.act_scales.to(Device::cpu()).to(DType::Float32);
    const float* scales_p = scales_cpu.template data<float>();
    const float* zeros_p  = zeros_cpu.template data<float>();
    const float* act_p     = act_cpu.template data<float>();

    auto packed_cpu = r.quantized_weight.to(Device::cpu());

    // Recover integer codes q[o,j].
    std::vector<int> q(static_cast<size_t>(out_features * in_features), 0);
    if (packed_cpu.dtype() == DType::UInt8) {
        // INT4 packed: low nibble = even column, high nibble = odd column.
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
    } else {
        // INT8: stored directly.
        auto q8 = packed_cpu.to(DType::Int32);
        const int32_t* src = static_cast<const int32_t*>(q8.data_ptr());
        for (int64_t i = 0; i < out_features * in_features; ++i) {
            q[i] = src[i];
        }
    }

    std::vector<float> recon(static_cast<size_t>(out_features * in_features), 0.0f);
    for (int64_t o = 0; o < out_features; ++o) {
        for (int64_t j = 0; j < in_features; ++j) {
            int64_t g = j / group_size;
            float scale = scales_p[o * num_groups + g];
            float zero  = zeros_p[o * num_groups + g];
            float aj    = act_p[j];
            float dq = (static_cast<float>(q[o * in_features + j]) - zero) * scale;
            // Undo the per-channel AWQ pre-scaling.
            recon[o * in_features + j] = (aj != 0.0f) ? (dq / aj) : dq;
        }
    }
    return recon;
}

}  // namespace awq_detail
}  // namespace testing
}  // namespace tenzor
