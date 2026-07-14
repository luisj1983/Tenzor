/**
 * @file mps_attention.mm
 * @brief Host-side dispatch for Metal attention compute shaders
 *
 * Dispatches FlashAttention (forward + backward), FusedAttention,
 * and GatherRelativePositionBias to Metal GPU kernels.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "../mps_backend.hpp"
#include "../mps_buffer_util.h"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "../mps_cmd_check.h"

namespace tenzor::mps {

namespace {

// These are shared with mps_elementwise.mm — in a real build they would
// live in a shared translation unit. For now, we duplicate the minimal
// infrastructure (the linker deduplicates the statics via inline/weak).

static id<MTLDevice> g_attn_device = nil;
static id<MTLLibrary> g_attn_library = nil;
static id<MTLCommandQueue> g_attn_queue = nil;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_attn_pipelines;

void ensure_attn_initialized() {
    if (g_attn_device == nil) {
        g_attn_device = MTLCreateSystemDefaultDevice();
        if (!g_attn_device) {
            throw std::runtime_error("MPS attention: No Metal device available");
        }
        g_attn_queue = [g_attn_device newCommandQueue];

        NSError* error = nil;
        g_attn_library = [g_attn_device newDefaultLibrary];
        if (!g_attn_library) {
            NSString* path = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
            if (path) {
                g_attn_library = [g_attn_device newLibraryWithFile:path error:&error];
            }
        }
        if (!g_attn_library) {
            throw std::runtime_error("MPS attention: Failed to load Metal shader library");
        }
    }
}

id<MTLComputePipelineState> get_attn_pipeline(const std::string& name) {
    auto it = g_attn_pipelines.find(name);
    if (it != g_attn_pipelines.end()) return it->second;

    ensure_attn_initialized();

    NSString* func_name = [NSString stringWithUTF8String:name.c_str()];
    id<MTLFunction> func = [g_attn_library newFunctionWithName:func_name];
    if (!func) {
        throw std::runtime_error("MPS attention: Shader not found: " + name);
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [g_attn_device newComputePipelineStateWithFunction:func error:&error];
    if (!pipeline) {
        throw std::runtime_error("MPS attention: Failed to create pipeline for: " + name +
                                  " - " + [[error localizedDescription] UTF8String]);
    }

    g_attn_pipelines[name] = pipeline;
    return pipeline;
}

id<MTLBuffer> get_attn_buffer(const Tensor& tensor) {
    // Route through the shared resolver: reuse the allocator's pooled buffer for
    // allocation-base tensors, and materialize a kept-alive contiguous copy for
    // offset views instead of wrapping a (likely unaligned) view pointer with
    // newBufferWithBytesNoCopy — which would return nil and crash on encode.
    return mps_buffer_for(g_attn_device, tensor);
}

id<MTLBuffer> make_attn_buffer(size_t bytes) {
    return [g_attn_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

static std::string attn_shader_for_dtype(const std::string& base, DType dtype) {
    switch (dtype) {
        case DType::Float32: return base;
        case DType::Float16: return base + "_f16";
        default:             return base;
    }
}

// AttentionParams struct layout must match the Metal shader
struct AttentionParams {
    uint32_t batch_heads;
    uint32_t seq_len_q;
    uint32_t seq_len_k;
    uint32_t head_dim;
    float scale;
    uint32_t causal;
};

struct GatherPosBiasParams {
    uint32_t num_positions;
    uint32_t num_heads;
    uint32_t seq_len;
};

} // anonymous namespace

// ============================================================================
// Flash Attention Forward
// Input: Q, K, V each (B*H, N, d) or (BH, N, d)
// Output: (BH, N, d)
// ============================================================================

Tensor mps_flash_attention_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                                    float scale, bool causal,
                                    [[maybe_unused]] float dropout_p,
                                    [[maybe_unused]] bool is_training) {
    ensure_attn_initialized();

    auto q_shape = Q.shape();
    // Support both 3D (BH, N, d) and 4D (B, H, N, d) layouts
    int64_t batch_heads, seq_len_q, head_dim, seq_len_k;
    Tensor Kb = K, Vb = V;
    if (q_shape.size() == 4) {
        batch_heads = q_shape[0] * q_shape[1];
        seq_len_q = q_shape[2];
        head_dim = q_shape[3];
        // JIT-R197: GQA/MQA head-broadcast for K/V, mirroring the fix already
        // applied to the CUDA/ROCm/OneAPI/Vulkan OpId::FlashAttention
        // registrations. The Metal kernel below shares a single `bh` grid
        // index across Q and K/V buffers, so without materializing K/V up to
        // Q's head count first, H_kv != H_q would have the kernel read K/V
        // buffers using a stride derived from H_q heads while the buffer
        // itself only holds H_kv heads worth of data (out-of-bounds / wrong
        // data for every kv-group after the first).
        int64_t h = q_shape[1];
        int64_t h_kv = K.shape()[1];
        if (h_kv != h) {
            if (h % h_kv != 0) {
                throw std::invalid_argument(
                    "MPS FlashAttention: H_q must be a multiple of H_kv; got " +
                    std::to_string(h) + " and " + std::to_string(h_kv));
            }
            int64_t b = q_shape[0];
            int64_t reps = h / h_kv;
            int64_t sk = K.shape()[2];
            int64_t d = K.shape()[3];
            int64_t dv = V.shape()[3];
            Tensor Kc = K.is_contiguous() ? K : K.contiguous();
            Tensor Vc = V.is_contiguous() ? V : V.contiguous();
            Tensor Ku = Kc.unsqueeze(2);
            Tensor Vu = Vc.unsqueeze(2);
            Tensor Ke = Ku.expand({b, h_kv, reps, sk, d});
            Tensor Ve = Vu.expand({b, h_kv, reps, sk, dv});
            Kb = Ke.contiguous().reshape({b, h, sk, d});
            Vb = Ve.contiguous().reshape({b, h, sk, dv});
        }
        seq_len_k = Kb.shape()[2];
    } else {
        batch_heads = q_shape[0];
        seq_len_q = q_shape[1];
        head_dim = q_shape[2];
        seq_len_k = Kb.shape()[1];
    }

    // Output same shape as Q
    std::vector<int64_t> out_shape(q_shape.begin(), q_shape.end());
    Tensor output(out_shape, Q.dtype(), Q.device());

    AttentionParams params;
    params.batch_heads = static_cast<uint32_t>(batch_heads);
    params.seq_len_q = static_cast<uint32_t>(seq_len_q);
    params.seq_len_k = static_cast<uint32_t>(seq_len_k);
    params.head_dim = static_cast<uint32_t>(head_dim);
    params.scale = scale;
    params.causal = causal ? 1 : 0;

    auto pipeline = get_attn_pipeline(attn_shader_for_dtype("flash_attention_forward", Q.dtype()));
    id<MTLBuffer> buf_q = get_attn_buffer(Q);
    id<MTLBuffer> buf_k = get_attn_buffer(Kb);
    id<MTLBuffer> buf_v = get_attn_buffer(Vb);
    id<MTLBuffer> buf_out = get_attn_buffer(output);

    id<MTLCommandBuffer> cmd = [g_attn_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_q offset:0 atIndex:0];
    [encoder setBuffer:buf_k offset:0 atIndex:1];
    [encoder setBuffer:buf_v offset:0 atIndex:2];
    [encoder setBuffer:buf_out offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(AttentionParams) atIndex:4];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(seq_len_q),
                               static_cast<NSUInteger>(batch_heads), 1);
    NSUInteger tg_x = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                               static_cast<NSUInteger>(seq_len_q));
    MTLSize threads = MTLSizeMake(tg_x, 1, 1);

    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    return output;
}

// ============================================================================
// Flash Attention Backward
// Inputs: dO, Q, K, V, O
// Returns: {dQ, dK, dV}
// ============================================================================

std::vector<Tensor> mps_flash_attention_backward(const Tensor& dO, const Tensor& Q,
                                                   const Tensor& K, const Tensor& V,
                                                   const Tensor& O,
                                                   float scale, bool causal) {
    // JIT-R197: no GQA head-broadcast here, matching the established
    // CUDA/ROCm/OneAPI/Vulkan OpId::FlashAttentionBackward precedent -- those
    // registrations assume K/V already carry Q's head count. Backward is
    // only ever invoked with the Q-head-width K/V that forward itself was
    // called with (repeat_kv already materializes GQA/MQA to full H before
    // any flash/composed path per src/nn/layers/gqa_attention.cpp and the
    // composed_attention_backward head-count assertion in
    // src/autograd/function_attention.cpp); no other backend reduces dK/dV
    // from a broadcast width back down to H_kv here either.
    ensure_attn_initialized();

    auto q_shape = Q.shape();
    int64_t batch_heads, seq_len_q, head_dim, seq_len_k;
    if (q_shape.size() == 4) {
        batch_heads = q_shape[0] * q_shape[1];
        seq_len_q = q_shape[2];
        head_dim = q_shape[3];
        seq_len_k = K.shape()[2];
    } else {
        batch_heads = q_shape[0];
        seq_len_q = q_shape[1];
        head_dim = q_shape[2];
        seq_len_k = K.shape()[1];
    }

    AttentionParams params;
    params.batch_heads = static_cast<uint32_t>(batch_heads);
    params.seq_len_q = static_cast<uint32_t>(seq_len_q);
    params.seq_len_k = static_cast<uint32_t>(seq_len_k);
    params.head_dim = static_cast<uint32_t>(head_dim);
    params.scale = scale;
    params.causal = causal ? 1 : 0;

    // Allocate outputs
    std::vector<int64_t> q_out_shape(q_shape.begin(), q_shape.end());
    auto k_shape = K.shape();
    std::vector<int64_t> k_out_shape(k_shape.begin(), k_shape.end());

    Tensor dQ(q_out_shape, Q.dtype(), Q.device());
    Tensor dK(k_out_shape, Q.dtype(), Q.device());
    Tensor dV(k_out_shape, Q.dtype(), Q.device());

    // Allocate D buffer (row sums for backward: BH * N floats)
    // D is always float regardless of input dtype for precision
    size_t d_bytes = static_cast<size_t>(batch_heads * seq_len_q) * sizeof(float);
    id<MTLBuffer> buf_D = make_attn_buffer(d_bytes);

    id<MTLBuffer> buf_dO = get_attn_buffer(dO);
    id<MTLBuffer> buf_Q = get_attn_buffer(Q);
    id<MTLBuffer> buf_K = get_attn_buffer(K);
    id<MTLBuffer> buf_V = get_attn_buffer(V);
    id<MTLBuffer> buf_O = get_attn_buffer(O);
    id<MTLBuffer> buf_dQ = get_attn_buffer(dQ);
    id<MTLBuffer> buf_dK = get_attn_buffer(dK);
    id<MTLBuffer> buf_dV = get_attn_buffer(dV);

    // Pass 1: Compute D[i] = sum(dO[i] * O[i]) per row
    {
        auto pipeline = get_attn_pipeline(attn_shader_for_dtype("flash_attention_backward_rowsum", Q.dtype()));
        id<MTLCommandBuffer> cmd = [g_attn_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_dO offset:0 atIndex:0];
        [encoder setBuffer:buf_O offset:0 atIndex:1];
        [encoder setBuffer:buf_D offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(AttentionParams) atIndex:3];

        MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(seq_len_q),
                                   static_cast<NSUInteger>(batch_heads), 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(seq_len_q));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    }

    // Pass 2: Compute dQ
    {
        auto pipeline = get_attn_pipeline(attn_shader_for_dtype("flash_attention_backward_dq", Q.dtype()));
        id<MTLCommandBuffer> cmd = [g_attn_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_dO offset:0 atIndex:0];
        [encoder setBuffer:buf_Q offset:0 atIndex:1];
        [encoder setBuffer:buf_K offset:0 atIndex:2];
        [encoder setBuffer:buf_V offset:0 atIndex:3];
        [encoder setBuffer:buf_O offset:0 atIndex:4];
        [encoder setBuffer:buf_D offset:0 atIndex:5];
        [encoder setBuffer:buf_dQ offset:0 atIndex:6];
        [encoder setBytes:&params length:sizeof(AttentionParams) atIndex:7];

        MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(seq_len_q),
                                   static_cast<NSUInteger>(batch_heads), 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(seq_len_q));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    }

    // Pass 3: Compute dK, dV
    {
        auto pipeline = get_attn_pipeline(attn_shader_for_dtype("flash_attention_backward_dkv", Q.dtype()));
        id<MTLCommandBuffer> cmd = [g_attn_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_dO offset:0 atIndex:0];
        [encoder setBuffer:buf_Q offset:0 atIndex:1];
        [encoder setBuffer:buf_K offset:0 atIndex:2];
        [encoder setBuffer:buf_V offset:0 atIndex:3];
        [encoder setBuffer:buf_O offset:0 atIndex:4];
        [encoder setBuffer:buf_D offset:0 atIndex:5];
        [encoder setBuffer:buf_dK offset:0 atIndex:6];
        [encoder setBuffer:buf_dV offset:0 atIndex:7];
        [encoder setBytes:&params length:sizeof(AttentionParams) atIndex:8];

        MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(seq_len_k),
                                   static_cast<NSUInteger>(batch_heads), 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(seq_len_k));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    }

    return {dQ, dK, dV};
}

// ============================================================================
// Fused Attention
// Same algorithm as flash attention forward; shared kernel
// ============================================================================

Tensor mps_fused_attention_kernel(const Tensor& Q, const Tensor& K, const Tensor& V,
                                   float scale, bool causal) {
    return mps_flash_attention_forward(Q, K, V, scale, causal, 0.0f, false);
}

// ============================================================================
// Gather Relative Position Bias
// bias_table: (num_rel_positions, num_heads)
// rel_pos_index: (num_positions,) — flat indices
// Returns: {output} where output is (num_heads, num_positions)
// ============================================================================

Tensor mps_gather_relative_position_bias(const Tensor& bias_table,
                                           const Tensor& rel_pos_index,
                                           int64_t num_positions,
                                           int64_t num_heads) {
    ensure_attn_initialized();

    std::vector<int64_t> out_shape = {num_heads, num_positions};
    Tensor output(out_shape, bias_table.dtype(), bias_table.device());

    GatherPosBiasParams params;
    params.num_positions = static_cast<uint32_t>(num_positions);
    params.num_heads = static_cast<uint32_t>(num_heads);
    params.seq_len = 0; // unused in current kernel

    auto pipeline = get_attn_pipeline(attn_shader_for_dtype("gather_relative_position_bias", bias_table.dtype()));
    id<MTLBuffer> buf_table = get_attn_buffer(bias_table);
    id<MTLBuffer> buf_index = get_attn_buffer(rel_pos_index);
    id<MTLBuffer> buf_out = get_attn_buffer(output);

    id<MTLCommandBuffer> cmd = [g_attn_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_table offset:0 atIndex:0];
    [encoder setBuffer:buf_index offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(GatherPosBiasParams) atIndex:3];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(num_positions),
                               static_cast<NSUInteger>(num_heads), 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(num_positions));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    return output;
}

} // namespace tenzor::mps
