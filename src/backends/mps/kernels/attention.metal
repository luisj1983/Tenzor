/**
 * @file attention.metal
 * @brief Metal compute shaders for Transformer attention operations
 *
 * Provides GPU kernels for FlashAttention (forward + backward),
 * FusedAttention, and GatherRelativePositionBias.
 * Uses online softmax (running max + running sum) for numerical stability.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Parameter structs
// ============================================================================

struct AttentionParams {
    uint batch_heads; // B * H
    uint seq_len_q;   // N (query length)
    uint seq_len_k;   // M (key/value length)
    uint head_dim;    // d
    float scale;      // 1/sqrt(d) or user-provided
    uint causal;      // 0 or 1
};

struct GatherPosBiasParams {
    uint num_positions; // total position pairs
    uint num_heads;     // number of attention heads
    uint seq_len;       // sequence length (sqrt of num_positions typically)
};

// ============================================================================
// Flash Attention Forward (Float32)
// Online softmax: numerically stable, single-pass per query row
// Grid: (seq_len_q, batch_heads, 1)
// Each thread computes one output row: O[bh][q][:] for one query position
// ============================================================================

kernel void flash_attention_forward(
    device const float* Q       [[buffer(0)]],  // (BH, N, d)
    device const float* K       [[buffer(1)]],  // (BH, M, d)
    device const float* V       [[buffer(2)]],  // (BH, M, d)
    device float* output        [[buffer(3)]],  // (BH, N, d)
    constant AttentionParams& p [[buffer(4)]],
    uint2 tid                   [[thread_position_in_grid]])
{
    uint q_idx = tid.x;  // query position
    uint bh    = tid.y;  // batch * head index

    if (q_idx >= p.seq_len_q || bh >= p.batch_heads) return;

    uint d = p.head_dim;
    float scale = p.scale;

    // Pointers into Q, K, V for this batch-head
    device const float* q_row = Q + (bh * p.seq_len_q + q_idx) * d;
    device const float* k_base = K + bh * p.seq_len_k * d;
    device const float* v_base = V + bh * p.seq_len_k * d;
    device float* o_row = output + (bh * p.seq_len_q + q_idx) * d;

    // Online softmax variables
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Temporary accumulator for output (on-chip)
    // We accumulate weighted V in float, using online softmax correction
    // For large head_dim, this stays in registers/threadgroup memory
    // Metal threads have ~256 registers, enough for d <= 256
    float acc[256]; // head_dim capped at 256 for register allocation
    for (uint i = 0; i < d && i < 256; ++i) acc[i] = 0.0f;

    uint kv_end = p.seq_len_k;
    if (p.causal) {
        kv_end = min(q_idx + 1, p.seq_len_k);
    }

    for (uint k_idx = 0; k_idx < kv_end; ++k_idx) {
        // Compute Q[q] . K[k] * scale
        float dot = 0.0f;
        device const float* k_row = k_base + k_idx * d;
        for (uint i = 0; i < d; ++i) {
            dot += q_row[i] * k_row[i];
        }
        dot *= scale;

        // Online softmax update
        float prev_max = row_max;
        row_max = max(row_max, dot);
        float correction = exp(prev_max - row_max);
        row_sum = row_sum * correction + exp(dot - row_max);

        // Correct existing accumulator and add new V contribution
        device const float* v_row = v_base + k_idx * d;
        float w = exp(dot - row_max);
        for (uint i = 0; i < d && i < 256; ++i) {
            acc[i] = acc[i] * correction + w * v_row[i];
        }
    }

    // Normalize by sum
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (uint i = 0; i < d && i < 256; ++i) {
        o_row[i] = acc[i] * inv_sum;
    }
}

// ============================================================================
// Flash Attention Forward (Float16)
// Compute in float for numerical stability, I/O in half
// ============================================================================

kernel void flash_attention_forward_f16(
    device const half* Q        [[buffer(0)]],
    device const half* K        [[buffer(1)]],
    device const half* V        [[buffer(2)]],
    device half* output         [[buffer(3)]],
    constant AttentionParams& p [[buffer(4)]],
    uint2 tid                   [[thread_position_in_grid]])
{
    uint q_idx = tid.x;
    uint bh    = tid.y;

    if (q_idx >= p.seq_len_q || bh >= p.batch_heads) return;

    uint d = p.head_dim;
    float scale = p.scale;

    device const half* q_row = Q + (bh * p.seq_len_q + q_idx) * d;
    device const half* k_base = K + bh * p.seq_len_k * d;
    device const half* v_base = V + bh * p.seq_len_k * d;
    device half* o_row = output + (bh * p.seq_len_q + q_idx) * d;

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc[256];
    for (uint i = 0; i < d && i < 256; ++i) acc[i] = 0.0f;

    uint kv_end = p.seq_len_k;
    if (p.causal) {
        kv_end = min(q_idx + 1, p.seq_len_k);
    }

    for (uint k_idx = 0; k_idx < kv_end; ++k_idx) {
        float dot = 0.0f;
        device const half* k_row = k_base + k_idx * d;
        for (uint i = 0; i < d; ++i) {
            dot += float(q_row[i]) * float(k_row[i]);
        }
        dot *= scale;

        float prev_max = row_max;
        row_max = max(row_max, dot);
        float correction = exp(prev_max - row_max);
        row_sum = row_sum * correction + exp(dot - row_max);

        device const half* v_row = v_base + k_idx * d;
        float w = exp(dot - row_max);
        for (uint i = 0; i < d && i < 256; ++i) {
            acc[i] = acc[i] * correction + w * float(v_row[i]);
        }
    }

    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (uint i = 0; i < d && i < 256; ++i) {
        o_row[i] = half(acc[i] * inv_sum);
    }
}

// ============================================================================
// Flash Attention Backward (Float32)
// Computes dQ, dK, dV given dO, Q, K, V, O
// Grid: (seq_len_q, batch_heads, 1) for dQ pass
// Then (seq_len_k, batch_heads, 1) for dK/dV pass
// We split into two kernels for clarity.
// ============================================================================

// Pass 1: Compute per-row D[i] = sum(dO[i] * O[i]) needed for both dQ and dK
kernel void flash_attention_backward_rowsum(
    device const float* dO      [[buffer(0)]],  // (BH, N, d)
    device const float* O       [[buffer(1)]],  // (BH, N, d)
    device float* D             [[buffer(2)]],  // (BH, N) — per-row dot product
    constant AttentionParams& p [[buffer(3)]],
    uint2 tid                   [[thread_position_in_grid]])
{
    uint q_idx = tid.x;
    uint bh    = tid.y;
    if (q_idx >= p.seq_len_q || bh >= p.batch_heads) return;

    uint d = p.head_dim;
    uint offset = (bh * p.seq_len_q + q_idx) * d;
    float sum = 0.0f;
    for (uint i = 0; i < d; ++i) {
        sum += dO[offset + i] * O[offset + i];
    }
    D[bh * p.seq_len_q + q_idx] = sum;
}

// Pass 2: Compute dQ, dK, dV
// Each thread handles one (q_idx, bh) pair, iterates over all K positions
// to compute dQ[bh][q_idx] and accumulate dK/dV contributions atomically.
// For simplicity, we use a per-query approach for dQ and a separate
// per-key approach for dK/dV.

// dQ kernel: one thread per query position
kernel void flash_attention_backward_dq(
    device const float* dO      [[buffer(0)]],  // (BH, N, d)
    device const float* Q       [[buffer(1)]],  // (BH, N, d)
    device const float* K       [[buffer(2)]],  // (BH, M, d)
    device const float* V       [[buffer(3)]],  // (BH, M, d)
    device const float* O       [[buffer(4)]],  // (BH, N, d)
    device const float* D       [[buffer(5)]],  // (BH, N)
    device float* dQ            [[buffer(6)]],  // (BH, N, d)
    constant AttentionParams& p [[buffer(7)]],
    uint2 tid                   [[thread_position_in_grid]])
{
    uint q_idx = tid.x;
    uint bh    = tid.y;
    if (q_idx >= p.seq_len_q || bh >= p.batch_heads) return;

    uint d = p.head_dim;
    float scale = p.scale;

    device const float* q_row = Q + (bh * p.seq_len_q + q_idx) * d;
    device const float* k_base = K + bh * p.seq_len_k * d;
    device const float* v_base = V + bh * p.seq_len_k * d;
    device const float* do_row = dO + (bh * p.seq_len_q + q_idx) * d;
    device float* dq_row = dQ + (bh * p.seq_len_q + q_idx) * d;
    float d_i = D[bh * p.seq_len_q + q_idx];

    // First: recompute softmax for this row (online softmax)
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    uint kv_end = p.seq_len_k;
    if (p.causal) kv_end = min(q_idx + 1, p.seq_len_k);

    // Compute attention weights
    for (uint k_idx = 0; k_idx < kv_end; ++k_idx) {
        float dot = 0.0f;
        device const float* k_row = k_base + k_idx * d;
        for (uint i = 0; i < d; ++i) dot += q_row[i] * k_row[i];
        dot *= scale;
        row_max = max(row_max, dot);
    }
    for (uint k_idx = 0; k_idx < kv_end; ++k_idx) {
        float dot = 0.0f;
        device const float* k_row = k_base + k_idx * d;
        for (uint i = 0; i < d; ++i) dot += q_row[i] * k_row[i];
        dot *= scale;
        row_sum += exp(dot - row_max);
    }
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    // Accumulate dQ
    float dq_acc[256];
    for (uint i = 0; i < d && i < 256; ++i) dq_acc[i] = 0.0f;

    for (uint k_idx = 0; k_idx < kv_end; ++k_idx) {
        device const float* k_row = k_base + k_idx * d;
        device const float* v_row = v_base + k_idx * d;

        float dot = 0.0f;
        for (uint i = 0; i < d; ++i) dot += q_row[i] * k_row[i];
        dot *= scale;
        float attn = exp(dot - row_max) * inv_sum;

        // dS = dO @ V^T element for this (q, k) pair
        float ds = 0.0f;
        for (uint i = 0; i < d; ++i) ds += do_row[i] * v_row[i];

        // dQ += (ds - D[q]) * attn * scale * K[k]
        float coeff = (ds - d_i) * attn * scale;
        for (uint i = 0; i < d && i < 256; ++i) {
            dq_acc[i] += coeff * k_row[i];
        }
    }

    for (uint i = 0; i < d && i < 256; ++i) {
        dq_row[i] = dq_acc[i];
    }
}

// dK/dV kernel: one thread per key position
kernel void flash_attention_backward_dkv(
    device const float* dO      [[buffer(0)]],  // (BH, N, d)
    device const float* Q       [[buffer(1)]],  // (BH, N, d)
    device const float* K       [[buffer(2)]],  // (BH, M, d)
    device const float* V       [[buffer(3)]],  // (BH, M, d)
    device const float* O       [[buffer(4)]],  // (BH, N, d)
    device const float* D       [[buffer(5)]],  // (BH, N)
    device float* dK            [[buffer(6)]],  // (BH, M, d)
    device float* dV            [[buffer(7)]],  // (BH, M, d)
    constant AttentionParams& p [[buffer(8)]],
    uint2 tid                   [[thread_position_in_grid]])
{
    uint k_idx = tid.x;
    uint bh    = tid.y;
    if (k_idx >= p.seq_len_k || bh >= p.batch_heads) return;

    uint d = p.head_dim;
    float scale = p.scale;

    device const float* k_row = K + (bh * p.seq_len_k + k_idx) * d;
    device const float* v_row = V + (bh * p.seq_len_k + k_idx) * d;
    device const float* q_base = Q + bh * p.seq_len_q * d;
    device const float* do_base = dO + bh * p.seq_len_q * d;
    device const float* d_base = D + bh * p.seq_len_q;
    device float* dk_row = dK + (bh * p.seq_len_k + k_idx) * d;
    device float* dv_row = dV + (bh * p.seq_len_k + k_idx) * d;

    float dk_acc[256];
    float dv_acc[256];
    for (uint i = 0; i < d && i < 256; ++i) { dk_acc[i] = 0.0f; dv_acc[i] = 0.0f; }

    // For causal: only queries q >= k_idx contribute
    uint q_start = p.causal ? k_idx : 0;

    for (uint q_idx = q_start; q_idx < p.seq_len_q; ++q_idx) {
        device const float* q_row = q_base + q_idx * d;
        device const float* do_row = do_base + q_idx * d;

        // Need to recompute softmax for row q_idx to get attn[q_idx][k_idx]
        // This is the expensive part of backward -- recomputation
        float row_max = -INFINITY;
        float row_sum = 0.0f;
        uint kv_end = p.causal ? min(q_idx + 1, p.seq_len_k) : p.seq_len_k;

        // Check if k_idx is within causal range for this q
        if (p.causal && k_idx > q_idx) continue;

        for (uint j = 0; j < kv_end; ++j) {
            float dot = 0.0f;
            device const float* kj = K + (bh * p.seq_len_k + j) * d;
            for (uint i = 0; i < d; ++i) dot += q_row[i] * kj[i];
            dot *= scale;
            row_max = max(row_max, dot);
        }
        for (uint j = 0; j < kv_end; ++j) {
            float dot = 0.0f;
            device const float* kj = K + (bh * p.seq_len_k + j) * d;
            for (uint i = 0; i < d; ++i) dot += q_row[i] * kj[i];
            dot *= scale;
            row_sum += exp(dot - row_max);
        }
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

        // Compute attn[q_idx][k_idx]
        float dot_qk = 0.0f;
        for (uint i = 0; i < d; ++i) dot_qk += q_row[i] * k_row[i];
        dot_qk *= scale;
        float attn = exp(dot_qk - row_max) * inv_sum;

        // dV += attn * dO[q]
        for (uint i = 0; i < d && i < 256; ++i) {
            dv_acc[i] += attn * do_row[i];
        }

        // dK contribution: (ds - D[q]) * attn * scale * Q[q]
        float ds = 0.0f;
        for (uint i = 0; i < d; ++i) ds += do_row[i] * v_row[i];
        float d_q = d_base[q_idx];
        float coeff = (ds - d_q) * attn * scale;
        for (uint i = 0; i < d && i < 256; ++i) {
            dk_acc[i] += coeff * q_row[i];
        }
    }

    for (uint i = 0; i < d && i < 256; ++i) {
        dk_row[i] = dk_acc[i];
        dv_row[i] = dv_acc[i];
    }
}

// Float16 backward variants
kernel void flash_attention_backward_rowsum_f16(
    device const half* dO       [[buffer(0)]],
    device const half* O        [[buffer(1)]],
    device float* D             [[buffer(2)]],  // keep D in float for precision
    constant AttentionParams& p [[buffer(3)]],
    uint2 tid                   [[thread_position_in_grid]])
{
    uint q_idx = tid.x;
    uint bh    = tid.y;
    if (q_idx >= p.seq_len_q || bh >= p.batch_heads) return;

    uint d = p.head_dim;
    uint offset = (bh * p.seq_len_q + q_idx) * d;
    float sum = 0.0f;
    for (uint i = 0; i < d; ++i) {
        sum += float(dO[offset + i]) * float(O[offset + i]);
    }
    D[bh * p.seq_len_q + q_idx] = sum;
}

kernel void flash_attention_backward_dq_f16(
    device const half* dO       [[buffer(0)]],
    device const half* Q        [[buffer(1)]],
    device const half* K        [[buffer(2)]],
    device const half* V        [[buffer(3)]],
    device const half* O        [[buffer(4)]],
    device const float* D       [[buffer(5)]],
    device half* dQ             [[buffer(6)]],
    constant AttentionParams& p [[buffer(7)]],
    uint2 tid                   [[thread_position_in_grid]])
{
    uint q_idx = tid.x;
    uint bh    = tid.y;
    if (q_idx >= p.seq_len_q || bh >= p.batch_heads) return;

    uint d = p.head_dim;
    float scale = p.scale;

    device const half* q_row = Q + (bh * p.seq_len_q + q_idx) * d;
    device const half* k_base = K + bh * p.seq_len_k * d;
    device const half* v_base = V + bh * p.seq_len_k * d;
    device const half* do_row = dO + (bh * p.seq_len_q + q_idx) * d;
    device half* dq_row = dQ + (bh * p.seq_len_q + q_idx) * d;
    float d_i = D[bh * p.seq_len_q + q_idx];

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    uint kv_end = p.causal ? min(q_idx + 1, p.seq_len_k) : p.seq_len_k;

    for (uint ki = 0; ki < kv_end; ++ki) {
        float dot = 0.0f;
        device const half* kr = k_base + ki * d;
        for (uint i = 0; i < d; ++i) dot += float(q_row[i]) * float(kr[i]);
        dot *= scale;
        row_max = max(row_max, dot);
    }
    for (uint ki = 0; ki < kv_end; ++ki) {
        float dot = 0.0f;
        device const half* kr = k_base + ki * d;
        for (uint i = 0; i < d; ++i) dot += float(q_row[i]) * float(kr[i]);
        dot *= scale;
        row_sum += exp(dot - row_max);
    }
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    float dq_acc[256];
    for (uint i = 0; i < d && i < 256; ++i) dq_acc[i] = 0.0f;

    for (uint ki = 0; ki < kv_end; ++ki) {
        device const half* kr = k_base + ki * d;
        device const half* vr = v_base + ki * d;
        float dot = 0.0f;
        for (uint i = 0; i < d; ++i) dot += float(q_row[i]) * float(kr[i]);
        dot *= scale;
        float attn = exp(dot - row_max) * inv_sum;
        float ds = 0.0f;
        for (uint i = 0; i < d; ++i) ds += float(do_row[i]) * float(vr[i]);
        float coeff = (ds - d_i) * attn * scale;
        for (uint i = 0; i < d && i < 256; ++i) dq_acc[i] += coeff * float(kr[i]);
    }

    for (uint i = 0; i < d && i < 256; ++i) dq_row[i] = half(dq_acc[i]);
}

kernel void flash_attention_backward_dkv_f16(
    device const half* dO       [[buffer(0)]],
    device const half* Q        [[buffer(1)]],
    device const half* K        [[buffer(2)]],
    device const half* V        [[buffer(3)]],
    device const half* O        [[buffer(4)]],
    device const float* D       [[buffer(5)]],
    device half* dK             [[buffer(6)]],
    device half* dV             [[buffer(7)]],
    constant AttentionParams& p [[buffer(8)]],
    uint2 tid                   [[thread_position_in_grid]])
{
    uint k_idx = tid.x;
    uint bh    = tid.y;
    if (k_idx >= p.seq_len_k || bh >= p.batch_heads) return;

    uint d = p.head_dim;
    float scale = p.scale;

    device const half* k_row = K + (bh * p.seq_len_k + k_idx) * d;
    device const half* v_row = V + (bh * p.seq_len_k + k_idx) * d;
    device const half* q_base = Q + bh * p.seq_len_q * d;
    device const half* do_base = dO + bh * p.seq_len_q * d;
    device const float* d_base = D + bh * p.seq_len_q;
    device half* dk_row = dK + (bh * p.seq_len_k + k_idx) * d;
    device half* dv_row = dV + (bh * p.seq_len_k + k_idx) * d;

    float dk_acc[256];
    float dv_acc[256];
    for (uint i = 0; i < d && i < 256; ++i) { dk_acc[i] = 0.0f; dv_acc[i] = 0.0f; }

    uint q_start = p.causal ? k_idx : 0;

    for (uint q_idx = q_start; q_idx < p.seq_len_q; ++q_idx) {
        device const half* q_row = q_base + q_idx * d;
        device const half* do_row = do_base + q_idx * d;
        uint kv_end = p.causal ? min(q_idx + 1, p.seq_len_k) : p.seq_len_k;
        if (p.causal && k_idx > q_idx) continue;

        float row_max = -INFINITY;
        float row_sum = 0.0f;
        for (uint j = 0; j < kv_end; ++j) {
            float dot = 0.0f;
            device const half* kj = K + (bh * p.seq_len_k + j) * d;
            for (uint i = 0; i < d; ++i) dot += float(q_row[i]) * float(kj[i]);
            dot *= scale;
            row_max = max(row_max, dot);
        }
        for (uint j = 0; j < kv_end; ++j) {
            float dot = 0.0f;
            device const half* kj = K + (bh * p.seq_len_k + j) * d;
            for (uint i = 0; i < d; ++i) dot += float(q_row[i]) * float(kj[i]);
            dot *= scale;
            row_sum += exp(dot - row_max);
        }
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

        float dot_qk = 0.0f;
        for (uint i = 0; i < d; ++i) dot_qk += float(q_row[i]) * float(k_row[i]);
        dot_qk *= scale;
        float attn = exp(dot_qk - row_max) * inv_sum;

        for (uint i = 0; i < d && i < 256; ++i) dv_acc[i] += attn * float(do_row[i]);

        float ds = 0.0f;
        for (uint i = 0; i < d; ++i) ds += float(do_row[i]) * float(v_row[i]);
        float d_q = d_base[q_idx];
        float coeff = (ds - d_q) * attn * scale;
        for (uint i = 0; i < d && i < 256; ++i) dk_acc[i] += coeff * float(q_row[i]);
    }

    for (uint i = 0; i < d && i < 256; ++i) {
        dk_row[i] = half(dk_acc[i]);
        dv_row[i] = half(dv_acc[i]);
    }
}

// ============================================================================
// Fused Attention (same as flash forward — shared implementation)
// For API compatibility, FusedAttention uses the same kernel
// ============================================================================

// fused_attention_forward is an alias — the host dispatch will call
// flash_attention_forward directly since they share the same algorithm.

// ============================================================================
// Gather Relative Position Bias
// bias_table: (num_rel_positions, num_heads)
// rel_pos_index: (num_positions,) — flat index into bias_table
// output: (num_heads, num_positions) or (1, num_heads, seq_len, seq_len)
// Grid: (num_positions, num_heads, 1)
// ============================================================================

kernel void gather_relative_position_bias(
    device const float* bias_table   [[buffer(0)]],  // (num_rel_positions, num_heads)
    device const int* rel_pos_index  [[buffer(1)]],   // (num_positions,)
    device float* output             [[buffer(2)]],   // (num_heads, num_positions)
    constant GatherPosBiasParams& p  [[buffer(3)]],
    uint2 tid                        [[thread_position_in_grid]])
{
    uint pos_idx = tid.x;
    uint head    = tid.y;
    if (pos_idx >= p.num_positions || head >= p.num_heads) return;

    int table_row = rel_pos_index[pos_idx];
    // bias_table is (num_rel_positions, num_heads), row-major
    output[head * p.num_positions + pos_idx] = bias_table[table_row * p.num_heads + head];
}

kernel void gather_relative_position_bias_f16(
    device const half* bias_table    [[buffer(0)]],
    device const int* rel_pos_index  [[buffer(1)]],
    device half* output              [[buffer(2)]],
    constant GatherPosBiasParams& p  [[buffer(3)]],
    uint2 tid                        [[thread_position_in_grid]])
{
    uint pos_idx = tid.x;
    uint head    = tid.y;
    if (pos_idx >= p.num_positions || head >= p.num_heads) return;

    int table_row = rel_pos_index[pos_idx];
    output[head * p.num_positions + pos_idx] = bias_table[table_row * p.num_heads + head];
}
