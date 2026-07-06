/**
 * @file mps_ctc_loss.mm
 * @brief MPS/Metal wrapper for CTC (Connectionist Temporal Classification) loss.
 *
 * Mirrors src/backends/cuda/kernels/ctc.cu — one threadgroup per batch
 * element runs log-domain forward-backward DP. Authored per the audit-4
 * W.3 spec; not exercised by the Linux build (mps backend is Apple-only).
 *
 *   inputs:  [log_probs (T, N, C) Float32,
 *             targets (N, S_max) Int32,
 *             input_lengths (N,) Int32,
 *             target_lengths (N,) Int32]
 *   attrs:   Blank (int, default 0), ZeroInfinity (bool, default false)
 *   outputs: [loss_per_sample (N,) Float32, raw_grad (T, N, C) Float32]
 */

#include "../mps_backend.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "../mps_cmd_check.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor::mps {

extern id<MTLDevice> g_device;
extern id<MTLCommandQueue> g_command_queue;
id<MTLComputePipelineState> get_pipeline(const std::string& name);
id<MTLBuffer> get_buffer(const Tensor& tensor);
void ensure_initialized();

struct CTCParamsMPS {
    int32_t T_max;
    int32_t N;
    int32_t C;
    int32_t S_max;
    int32_t L_max;
    int32_t blank;
    int32_t zero_infinity;
    int32_t _pad;
};

auto ctc_loss_forward_kernel(const Tensor& log_probs,
                             const Tensor& targets,
                             const Tensor& input_lengths,
                             const Tensor& target_lengths,
                             int64_t blank,
                             bool zero_infinity) -> std::vector<Tensor> {
    // The Metal shader is Float32-only (Metal has no native double). To match
    // CUDA/ROCm/OneAPI — which compute Float64 natively and widen Float16/
    // BFloat16 to Float32 then narrow — MPS widens every non-Float32 log_probs
    // dtype to Float32, runs the DP, then narrows both outputs (loss + raw
    // grad) back to the original dtype so precision metadata is preserved.
    const DType dt = log_probs.dtype();
    if (dt == DType::Float64 || dt == DType::Float16 || dt == DType::BFloat16) {
        auto out = ctc_loss_forward_kernel(log_probs.to(DType::Float32), targets,
                                           input_lengths, target_lengths, blank,
                                           zero_infinity);
        for (auto& t : out) t = t.to(dt);
        return out;
    }
    if (dt != DType::Float32) {
        throw std::invalid_argument(
            "ctc_loss_forward (MPS): log_probs must be "
            "Float16/BFloat16/Float32/Float64");
    }
    if (targets.dtype() != DType::Int32 ||
        input_lengths.dtype() != DType::Int32 ||
        target_lengths.dtype() != DType::Int32) {
        throw std::invalid_argument(
            "ctc_loss_forward (MPS): targets / input_lengths / target_lengths "
            "must be Int32");
    }
    auto lp_shape = log_probs.shape();
    if (lp_shape.size() != 3) {
        throw std::invalid_argument(
            "ctc_loss_forward (MPS): log_probs must be 3D (T, N, C)");
    }
    int64_t T_max = lp_shape[0];
    int64_t N = lp_shape[1];
    int64_t C = lp_shape[2];

    auto tgt_shape = targets.shape();
    int64_t S_max = (tgt_shape.size() >= 2) ? tgt_shape[1]
                  : (tgt_shape.size() == 1) ? tgt_shape[0] : 0;
    int64_t L_max = 2 * S_max + 1;

    Tensor loss_out({N}, DType::Float32, log_probs.device());
    Tensor grad_out({T_max, N, C}, DType::Float32, log_probs.device());

    int64_t alpha_elems = N * T_max * L_max;
    Tensor alpha_buf({alpha_elems}, DType::Float32, log_probs.device());
    Tensor beta_buf({alpha_elems}, DType::Float32, log_probs.device());

    if (N == 0 || T_max == 0 || C == 0) {
        return {loss_out, grad_out};
    }

    ensure_initialized();
    auto pipeline = get_pipeline("ctc_forward_backward_kernel");

    id<MTLBuffer> buf_lp    = get_buffer(log_probs);
    id<MTLBuffer> buf_tgt   = get_buffer(targets);
    id<MTLBuffer> buf_il    = get_buffer(input_lengths);
    id<MTLBuffer> buf_tl    = get_buffer(target_lengths);
    id<MTLBuffer> buf_alpha = get_buffer(alpha_buf);
    id<MTLBuffer> buf_beta  = get_buffer(beta_buf);
    id<MTLBuffer> buf_loss  = get_buffer(loss_out);
    id<MTLBuffer> buf_grad  = get_buffer(grad_out);

    CTCParamsMPS params{};
    params.T_max = static_cast<int32_t>(T_max);
    params.N = static_cast<int32_t>(N);
    params.C = static_cast<int32_t>(C);
    params.S_max = static_cast<int32_t>(S_max);
    params.L_max = static_cast<int32_t>(L_max);
    params.blank = static_cast<int32_t>(blank);
    params.zero_infinity = zero_infinity ? 1 : 0;

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_lp offset:0 atIndex:0];
    [encoder setBuffer:buf_tgt offset:0 atIndex:1];
    [encoder setBuffer:buf_il offset:0 atIndex:2];
    [encoder setBuffer:buf_tl offset:0 atIndex:3];
    [encoder setBuffer:buf_alpha offset:0 atIndex:4];
    [encoder setBuffer:buf_beta offset:0 atIndex:5];
    [encoder setBuffer:buf_loss offset:0 atIndex:6];
    [encoder setBuffer:buf_grad offset:0 atIndex:7];
    [encoder setBytes:&params length:sizeof(params) atIndex:8];

    constexpr NSUInteger threads_per_group = 128;
    NSUInteger tg = std::min(
        static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
        threads_per_group);

    MTLSize threadgroups = MTLSizeMake(static_cast<NSUInteger>(N), 1, 1);
    MTLSize tg_size = MTLSizeMake(tg, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:tg_size];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    return {loss_out, grad_out};
}

} // namespace tenzor::mps
