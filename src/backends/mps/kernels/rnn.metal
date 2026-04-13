/**
 * @file rnn.metal
 * @brief Metal compute shaders for RNN gate activations and state updates
 *
 * These kernels handle the element-wise gate activation and state update
 * portions of LSTM and GRU cells. The expensive linear transforms (GEMMs)
 * are handled by MPSMatrixMultiplication on the host side.
 *
 * Each kernel operates on pre-computed gate values (output of GEMM + bias)
 * and applies the non-linear activation functions + state transitions.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// LSTM gate activations + cell/hidden state update (Float32)
// ============================================================================

/// After GEMM: gates = input @ W_ih^T + hidden @ W_hh^T + bias_ih + bias_hh
/// This kernel applies sigmoid/tanh activations and computes new cell/hidden.
///
/// Thread grid: batch_size * hidden_size (one thread per hidden unit per batch)
kernel void lstm_gates_kernel(
    device const float* gates     [[buffer(0)]],  // (B, 4*H) pre-activation gates
    device const float* old_cell  [[buffer(1)]],  // (B, H)
    device float* new_cell        [[buffer(2)]],  // (B, H)
    device float* new_hidden      [[buffer(3)]],  // (B, H)
    constant uint& hidden_size    [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint b = tid / hidden_size;
    uint h = tid % hidden_size;
    uint base = b * 4 * hidden_size;

    float i_gate = 1.0f / (1.0f + exp(-gates[base + h]));
    float f_gate = 1.0f / (1.0f + exp(-gates[base + hidden_size + h]));
    float g_gate = tanh(gates[base + 2 * hidden_size + h]);
    float o_gate = 1.0f / (1.0f + exp(-gates[base + 3 * hidden_size + h]));

    float nc = f_gate * old_cell[b * hidden_size + h] + i_gate * g_gate;
    new_cell[b * hidden_size + h] = nc;
    new_hidden[b * hidden_size + h] = o_gate * tanh(nc);
}

// ============================================================================
// LSTM backward: recompute gates, backprop through activations (Float32)
// ============================================================================

/// Computes gradients for LSTM cell backward pass.
/// Recomputes gate activations from pre-activation gate values, then
/// backpropagates through the cell/hidden state computation.
///
/// Outputs per-element gate gradients (d_gates) for subsequent weight gradient
/// GEMM on the host side.
///
/// Thread grid: batch_size * hidden_size
kernel void lstm_backward_gates_kernel(
    device const float* gates       [[buffer(0)]],  // (B, 4*H) pre-activation gates
    device const float* old_cell    [[buffer(1)]],  // (B, H)
    device const float* grad_hy     [[buffer(2)]],  // (B, H)
    device const float* grad_cy     [[buffer(3)]],  // (B, H)
    device float* d_gates           [[buffer(4)]],  // (B, 4*H) output: gate gradients
    device float* grad_cx_out       [[buffer(5)]],  // (B, H) output: grad w.r.t. old cell
    constant uint& hidden_size      [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint b = tid / hidden_size;
    uint h = tid % hidden_size;
    uint base = b * 4 * hidden_size;

    // Recompute gate activations
    float i_val = 1.0f / (1.0f + exp(-gates[base + h]));
    float f_val = 1.0f / (1.0f + exp(-gates[base + hidden_size + h]));
    float g_val = tanh(gates[base + 2 * hidden_size + h]);
    float o_val = 1.0f / (1.0f + exp(-gates[base + 3 * hidden_size + h]));

    float c_new = f_val * old_cell[b * hidden_size + h] + i_val * g_val;
    float tanh_c = tanh(c_new);

    float dh = grad_hy[b * hidden_size + h];
    float dc = grad_cy[b * hidden_size + h];

    // Backprop through hidden = o * tanh(c)
    float d_o = dh * tanh_c;
    dc += dh * o_val * (1.0f - tanh_c * tanh_c);

    // Backprop through cell = f * old_cell + i * g
    float d_f = dc * old_cell[b * hidden_size + h];
    float d_i = dc * g_val;
    float d_g = dc * i_val;
    grad_cx_out[b * hidden_size + h] = dc * f_val;

    // Backprop through gate activations (sigmoid' = s*(1-s), tanh' = 1-t^2)
    d_gates[base + h]                    = d_i * i_val * (1.0f - i_val);
    d_gates[base + hidden_size + h]      = d_f * f_val * (1.0f - f_val);
    d_gates[base + 2 * hidden_size + h]  = d_g * (1.0f - g_val * g_val);
    d_gates[base + 3 * hidden_size + h]  = d_o * o_val * (1.0f - o_val);
}

// ============================================================================
// GRU gate activations + hidden state update (Float32)
// ============================================================================

/// GRU computation after GEMMs:
///   gates_ih = input @ W_ih^T + bias_ih  (3*H values: r_ih, z_ih, n_ih)
///   gates_hh = hidden @ W_hh^T + bias_hh (3*H values: r_hh, z_hh, n_hh)
///
///   r = sigmoid(r_ih + r_hh)    // reset gate
///   z = sigmoid(z_ih + z_hh)    // update gate
///   n = tanh(n_ih + r * n_hh)   // new gate
///   new_hidden = (1 - z) * n + z * old_hidden
///
/// Thread grid: batch_size * hidden_size
kernel void gru_gates_kernel(
    device const float* gates_ih     [[buffer(0)]],  // (B, 3*H) input-hidden gates
    device const float* gates_hh     [[buffer(1)]],  // (B, 3*H) hidden-hidden gates
    device const float* old_hidden   [[buffer(2)]],  // (B, H)
    device float* new_hidden         [[buffer(3)]],  // (B, H)
    constant uint& hidden_size       [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint b = tid / hidden_size;
    uint h = tid % hidden_size;
    uint base = b * 3 * hidden_size;

    float r = 1.0f / (1.0f + exp(-(gates_ih[base + h] + gates_hh[base + h])));
    float z = 1.0f / (1.0f + exp(-(gates_ih[base + hidden_size + h] +
                                     gates_hh[base + hidden_size + h])));
    float n = tanh(gates_ih[base + 2 * hidden_size + h] +
                   r * gates_hh[base + 2 * hidden_size + h]);

    float old_h = old_hidden[b * hidden_size + h];
    new_hidden[b * hidden_size + h] = (1.0f - z) * n + z * old_h;
}

// ============================================================================
// GRU backward: recompute gates, backprop through activations (Float32)
// ============================================================================

/// Computes gradients for GRU cell backward pass.
/// Outputs gate gradients for ih and hh paths separately, since GRU
/// applies the reset gate to the hidden-hidden path before the n-gate.
///
/// Thread grid: batch_size * hidden_size
kernel void gru_backward_gates_kernel(
    device const float* gates_ih     [[buffer(0)]],  // (B, 3*H) input-hidden gates
    device const float* gates_hh     [[buffer(1)]],  // (B, 3*H) hidden-hidden gates
    device const float* old_hidden   [[buffer(2)]],  // (B, H)
    device const float* grad_hy      [[buffer(3)]],  // (B, H)
    device float* d_gates_ih         [[buffer(4)]],  // (B, 3*H) output: ih gate grads
    device float* d_gates_hh         [[buffer(5)]],  // (B, 3*H) output: hh gate grads
    device float* grad_hx_out        [[buffer(6)]],  // (B, H) output: grad w.r.t. old hidden
    constant uint& hidden_size       [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint b = tid / hidden_size;
    uint h = tid % hidden_size;
    uint base = b * 3 * hidden_size;

    // Recompute gate activations
    float r = 1.0f / (1.0f + exp(-(gates_ih[base + h] + gates_hh[base + h])));
    float z = 1.0f / (1.0f + exp(-(gates_ih[base + hidden_size + h] +
                                     gates_hh[base + hidden_size + h])));
    float n_pre = gates_ih[base + 2 * hidden_size + h] +
                  r * gates_hh[base + 2 * hidden_size + h];
    float n = tanh(n_pre);

    float old_h = old_hidden[b * hidden_size + h];
    float dh = grad_hy[b * hidden_size + h];

    // hy = (1-z)*n + z*old_h
    float dn = dh * (1.0f - z);
    float dz = dh * (old_h - n);
    float d_hx = dh * z;  // direct gradient to old_hidden

    // Backprop through n = tanh(n_pre)
    float dn_pre = dn * (1.0f - n * n);

    // Backprop through r: n_pre = n_ih + r * n_hh
    float dr = dn_pre * gates_hh[base + 2 * hidden_size + h];

    // Backprop through sigmoid activations
    float dz_pre = dz * z * (1.0f - z);
    float dr_pre = dr * r * (1.0f - r);

    // ih gate gradients (for d_input GEMM)
    d_gates_ih[base + h]                    = dr_pre;
    d_gates_ih[base + hidden_size + h]      = dz_pre;
    d_gates_ih[base + 2 * hidden_size + h]  = dn_pre;

    // hh gate gradients (for d_hx GEMM)
    // r and z gates: same pre-activation gradients
    d_gates_hh[base + h]                    = dr_pre;
    d_gates_hh[base + hidden_size + h]      = dz_pre;
    // n_hh gate: multiply by r (chain rule through r*n_hh)
    d_gates_hh[base + 2 * hidden_size + h]  = dn_pre * r;

    grad_hx_out[b * hidden_size + h] = d_hx;
}

// ============================================================================
// Bias addition kernel (add bias to GEMM output, in-place)
// ============================================================================

/// Adds a 1-D bias vector to each row of a 2-D matrix.
/// Grid: rows * cols (one thread per element)
kernel void add_bias_kernel(
    device float* output          [[buffer(0)]],  // (rows, cols) - modified in-place
    device const float* bias      [[buffer(1)]],  // (cols,)
    constant uint& cols           [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    output[tid] += bias[tid % cols];
}

// ============================================================================
// Float16 variants
// ============================================================================

kernel void lstm_gates_kernel_f16(
    device const half* gates     [[buffer(0)]],
    device const half* old_cell  [[buffer(1)]],
    device half* new_cell        [[buffer(2)]],
    device half* new_hidden      [[buffer(3)]],
    constant uint& hidden_size   [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint b = tid / hidden_size;
    uint h = tid % hidden_size;
    uint base = b * 4 * hidden_size;

    // Compute in float for numerical stability
    float i_gate = 1.0f / (1.0f + exp(-float(gates[base + h])));
    float f_gate = 1.0f / (1.0f + exp(-float(gates[base + hidden_size + h])));
    float g_gate = tanh(float(gates[base + 2 * hidden_size + h]));
    float o_gate = 1.0f / (1.0f + exp(-float(gates[base + 3 * hidden_size + h])));

    float nc = f_gate * float(old_cell[b * hidden_size + h]) + i_gate * g_gate;
    new_cell[b * hidden_size + h] = half(nc);
    new_hidden[b * hidden_size + h] = half(o_gate * tanh(nc));
}

kernel void lstm_backward_gates_kernel_f16(
    device const half* gates       [[buffer(0)]],
    device const half* old_cell    [[buffer(1)]],
    device const half* grad_hy     [[buffer(2)]],
    device const half* grad_cy     [[buffer(3)]],
    device half* d_gates           [[buffer(4)]],
    device half* grad_cx_out       [[buffer(5)]],
    constant uint& hidden_size     [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint b = tid / hidden_size;
    uint h = tid % hidden_size;
    uint base = b * 4 * hidden_size;

    float i_val = 1.0f / (1.0f + exp(-float(gates[base + h])));
    float f_val = 1.0f / (1.0f + exp(-float(gates[base + hidden_size + h])));
    float g_val = tanh(float(gates[base + 2 * hidden_size + h]));
    float o_val = 1.0f / (1.0f + exp(-float(gates[base + 3 * hidden_size + h])));

    float c_new = f_val * float(old_cell[b * hidden_size + h]) + i_val * g_val;
    float tanh_c = tanh(c_new);

    float dh = float(grad_hy[b * hidden_size + h]);
    float dc = float(grad_cy[b * hidden_size + h]);

    float d_o = dh * tanh_c;
    dc += dh * o_val * (1.0f - tanh_c * tanh_c);

    float d_f = dc * float(old_cell[b * hidden_size + h]);
    float d_i = dc * g_val;
    float d_g = dc * i_val;
    grad_cx_out[b * hidden_size + h] = half(dc * f_val);

    d_gates[base + h]                    = half(d_i * i_val * (1.0f - i_val));
    d_gates[base + hidden_size + h]      = half(d_f * f_val * (1.0f - f_val));
    d_gates[base + 2 * hidden_size + h]  = half(d_g * (1.0f - g_val * g_val));
    d_gates[base + 3 * hidden_size + h]  = half(d_o * o_val * (1.0f - o_val));
}

kernel void gru_gates_kernel_f16(
    device const half* gates_ih     [[buffer(0)]],
    device const half* gates_hh     [[buffer(1)]],
    device const half* old_hidden   [[buffer(2)]],
    device half* new_hidden         [[buffer(3)]],
    constant uint& hidden_size      [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint b = tid / hidden_size;
    uint h = tid % hidden_size;
    uint base = b * 3 * hidden_size;

    float r = 1.0f / (1.0f + exp(-(float(gates_ih[base + h]) + float(gates_hh[base + h]))));
    float z = 1.0f / (1.0f + exp(-(float(gates_ih[base + hidden_size + h]) +
                                     float(gates_hh[base + hidden_size + h]))));
    float n = tanh(float(gates_ih[base + 2 * hidden_size + h]) +
                   r * float(gates_hh[base + 2 * hidden_size + h]));

    float old_h = float(old_hidden[b * hidden_size + h]);
    new_hidden[b * hidden_size + h] = half((1.0f - z) * n + z * old_h);
}

kernel void gru_backward_gates_kernel_f16(
    device const half* gates_ih     [[buffer(0)]],
    device const half* gates_hh     [[buffer(1)]],
    device const half* old_hidden   [[buffer(2)]],
    device const half* grad_hy      [[buffer(3)]],
    device half* d_gates_ih         [[buffer(4)]],
    device half* d_gates_hh         [[buffer(5)]],
    device half* grad_hx_out        [[buffer(6)]],
    constant uint& hidden_size      [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint b = tid / hidden_size;
    uint h = tid % hidden_size;
    uint base = b * 3 * hidden_size;

    float r = 1.0f / (1.0f + exp(-(float(gates_ih[base + h]) + float(gates_hh[base + h]))));
    float z = 1.0f / (1.0f + exp(-(float(gates_ih[base + hidden_size + h]) +
                                     float(gates_hh[base + hidden_size + h]))));
    float n_pre = float(gates_ih[base + 2 * hidden_size + h]) +
                  r * float(gates_hh[base + 2 * hidden_size + h]);
    float n = tanh(n_pre);

    float old_h = float(old_hidden[b * hidden_size + h]);
    float dh = float(grad_hy[b * hidden_size + h]);

    float dn = dh * (1.0f - z);
    float dz = dh * (old_h - n);
    float d_hx = dh * z;

    float dn_pre = dn * (1.0f - n * n);
    float dr = dn_pre * float(gates_hh[base + 2 * hidden_size + h]);

    float dz_pre = dz * z * (1.0f - z);
    float dr_pre = dr * r * (1.0f - r);

    d_gates_ih[base + h]                    = half(dr_pre);
    d_gates_ih[base + hidden_size + h]      = half(dz_pre);
    d_gates_ih[base + 2 * hidden_size + h]  = half(dn_pre);

    d_gates_hh[base + h]                    = half(dr_pre);
    d_gates_hh[base + hidden_size + h]      = half(dz_pre);
    d_gates_hh[base + 2 * hidden_size + h]  = half(dn_pre * r);

    grad_hx_out[b * hidden_size + h] = half(d_hx);
}

kernel void add_bias_kernel_f16(
    device half* output          [[buffer(0)]],
    device const half* bias      [[buffer(1)]],
    constant uint& cols          [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    output[tid] = half(float(output[tid]) + float(bias[tid % cols]));
}
