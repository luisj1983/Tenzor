/**
 * @file rnn_sequence.cu
 * @brief Full-sequence LSTM/GRU forward kernels for CUDA
 *
 * Implements single-layer, multi-layer, and bidirectional variants.
 * Uses cuBLAS for weight projections and existing fused cell kernels
 * for element-wise gate computation. All ops stay on a single CUDA
 * stream with no per-timestep synchronization.
 */

#ifdef TENZOR_CUDA_AVAILABLE

#include <cuda_runtime.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "cuda_common.cuh"
#include <cublas_v2.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "../cuda_stream_pool.hpp"

namespace tenzor {
namespace cuda {

#ifdef TENZOR_HAS_CUDNN
// Defined in cudnn_rnn.cu (kept there so cuDNN-typed args resolve in the TU
// that includes <cudnn.h> first — avoids an NVCC mangling mismatch on
// cudnnForwardMode_t across TUs). Single fused cuDNN inference kernel; Float32.
std::vector<Tensor> lstm_forward_cudnn_inference(
    const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
    const Tensor& bias_ih, const Tensor& bias_hh,
    const Tensor& h0, const Tensor& c0);
std::vector<Tensor> lstm_multi_layer_forward_cudnn_inference(
    const Tensor& input,
    const std::vector<Tensor>& W_ih_list,
    const std::vector<Tensor>& W_hh_list,
    const std::vector<Tensor>& bias_list,
    const Tensor& h0, const Tensor& c0);
std::vector<Tensor> gru_forward_cudnn_inference(
    const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
    const Tensor& bias_ih, const Tensor& h0, const Tensor& bias_hh);
#endif

// Logistic sigmoid. Matches the unclamped CPU reference (rnn_kernels.cpp) and
// gru.cu's 1/(1+exp(-x)) so saturated gates / their gradients agree across
// backends. A previous clamp to [-20,20] diverged from CPU at saturated gates
// (and forced the gradient to exactly 0 there); IEEE handles the extremes
// gracefully (exp(-x) -> +inf gives sigma -> 0, exp(-x) -> 0 gives sigma -> 1).
template<typename T>
__device__ __forceinline__ T stable_sigmoid(T x) {
    return T(1) / (T(1) + exp(-x));
}

// Forward declarations from other CUDA kernels
auto cublas_matmul(const Tensor& a, const Tensor& b) -> Tensor;
auto add_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;

// From lstm.cu
auto lstm_cell_forward_kernel(
    const Tensor& gates,
    const Tensor& c_prev,
    int64_t batch_size,
    int64_t hidden_size,
    cudaStream_t stream) -> std::pair<Tensor, Tensor>;

// ============================================================================
// Bias addition kernel (row-wise broadcast)
// ============================================================================

template<typename T>
__global__ void add_bias_kernel(
    T* __restrict__ output,
    const T* __restrict__ bias,
    int64_t batch_size,
    int64_t feature_size
) {
    int64_t total = batch_size * feature_size;
    // Grid-stride loop so a grid clamped to the device's maxGridSize still
    // covers tensors whose element count exceeds grid_size * block_size.
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        int64_t col = idx % feature_size;
        output[idx] += bias[col];
    }
}

template<typename T>
static void launch_add_bias(T* output, const T* bias,
                            int64_t batch, int64_t features, cudaStream_t stream) {
    int64_t total = batch * features;
    int block = 256;
    // Compute the grid in 64-bit then clamp to the device's maxGridSize so the
    // launch dimension never overflows the int grid argument for very large
    // total = (seq_len*batch) * gate_size. The kernel uses a grid-stride loop,
    // so a clamped grid still covers all elements.
    int64_t grid64 = (total + block - 1) / block;
    int device_id = 0;
    cudaGetDevice(&device_id);
    int max_grid_x = 65535;
    cudaDeviceGetAttribute(&max_grid_x, cudaDevAttrMaxGridDimX, device_id);
    int grid = static_cast<int>(std::min<int64_t>(grid64, static_cast<int64_t>(max_grid_x)));
    if (grid < 1) grid = 1;
    add_bias_kernel<<<grid, block, 0, stream>>>(output, bias, batch, features);
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

// Compute a grid dimension for `total` elements at `block` threads, in 64-bit,
// clamped to the device's maxGridSize so the int launch argument never
// overflows. Callers must pair this with a grid-stride kernel so a clamped grid
// still covers every element.
static int rnn_clamped_grid(int64_t total, int block) {
    int64_t grid64 = (total + block - 1) / block;
    int device_id = 0;
    cudaGetDevice(&device_id);
    int max_grid_x = 65535;
    cudaDeviceGetAttribute(&max_grid_x, cudaDevAttrMaxGridDimX, device_id);
    int grid = static_cast<int>(std::min<int64_t>(grid64, static_cast<int64_t>(max_grid_x)));
    if (grid < 1) grid = 1;
    return grid;
}

// ============================================================================
// GRU fused cell kernel for sequence ops
// ============================================================================

template<typename T>
__global__ void gru_cell_fused_kernel(
    const T* __restrict__ gates_ih, // (batch, 3*hidden) — W_ih @ x + bias_ih
    const T* __restrict__ gates_hh, // (batch, 3*hidden) — W_hh @ h + bias_hh
    const T* __restrict__ h_prev,
    T* __restrict__ h_out,
    int64_t batch_size,
    int64_t hidden_size
) {
    int64_t total = batch_size * hidden_size;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        int64_t b = idx / hidden_size;
        int64_t h = idx % hidden_size;

        // Gate offsets
        int64_t base_ih = b * 3 * hidden_size + h;
        int64_t base_hh = b * 3 * hidden_size + h;

        T r_ih = gates_ih[base_ih];
        T z_ih = gates_ih[base_ih + hidden_size];
        T n_ih = gates_ih[base_ih + 2 * hidden_size];

        T r_hh = gates_hh[base_hh];
        T z_hh = gates_hh[base_hh + hidden_size];
        T n_hh = gates_hh[base_hh + 2 * hidden_size];

        // Reset gate: r = sigmoid(r_ih + r_hh)
        T r = stable_sigmoid(r_ih + r_hh);
        // Update gate: z = sigmoid(z_ih + z_hh)
        T z = stable_sigmoid(z_ih + z_hh);
        // New gate: n = tanh(n_ih + r * n_hh)
        T n = tanh(n_ih + r * n_hh);

        T hp = h_prev[idx];
        h_out[idx] = (T(1) - z) * n + z * hp;
    }
}

// ============================================================================
// Helper: compute gates = x @ W^T  (using cublas_matmul with transposed weight)
// We store weights as (out_features, in_features) and matmul as (batch, in) @ (in, out)
// So we need to transpose W first, or use cublas directly.
// For simplicity, use cublas_matmul which handles 2D matmul.
// W is (out, in), W^T is (in, out), x is (batch, in), result is (batch, out)
// ============================================================================

// ============================================================================
// LSTM Forward (single layer, full sequence)
// ============================================================================

auto lstm_forward_cuda(
    const Tensor& input,     // (seq_len, batch, input_size)
    const Tensor& W_ih,      // (4*hidden, input_size)
    const Tensor& W_hh,      // (4*hidden, hidden)
    const Tensor& bias_ih,   // (4*hidden) or empty
    const Tensor& bias_hh,   // (4*hidden) or empty
    const Tensor& h0,        // (batch, hidden)
    const Tensor& c0         // (batch, hidden)
) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t seq_len = shape[0];
    int64_t batch = shape[1];
    int64_t input_size = shape[2];
    int64_t hidden = h0.shape()[1];
    int64_t gate_size = 4 * hidden;

    // The portable per-timestep path below only launches the LSTM cell and the
    // bias adds for Float32/Float64. A Float16/BFloat16 LSTM would otherwise
    // silently drop bias_ih/bias_hh (the launch_add_bias branches have no
    // half/bf16 case). Widen to Float32 up-front (on-GPU), run the real LSTM,
    // and narrow the {output, h_n, c_n} results back, mirroring gru_forward_cuda.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto results = lstm_forward_cuda(
            input.to(DType::Float32),
            W_ih.to(DType::Float32),
            W_hh.to(DType::Float32),
            bias_ih.numel() > 0 ? bias_ih.to(DType::Float32) : bias_ih,
            bias_hh.numel() > 0 ? bias_hh.to(DType::Float32) : bias_hh,
            h0.to(DType::Float32),
            c0.to(DType::Float32));
        for (auto& r : results) {
            r = r.to(orig);
        }
        return results;
    }

#ifdef TENZOR_HAS_CUDNN
    // Fast path: one fused cuDNN RNN kernel replaces the per-timestep cuBLAS +
    // cell-kernel loop below (~2*seq_len kernel launches whose overhead is why
    // PyTorch's cuDNN LSTM was several times faster). Float32 inference only;
    // other dtypes / gradient-tracking fall through to the portable loop.
    if (input.dtype() == DType::Float32) {
        return lstm_forward_cudnn_inference(input, W_ih, W_hh,
                                            bias_ih, bias_hh, h0, c0);
    }
#endif

    int device_id = input.device().index;
    auto stream_guard = CUDAStreamPool::instance().acquire_guard(device_id);
    cudaStream_t stream = stream_guard.get();

    // Output: (seq_len, batch, hidden)
    Tensor output({seq_len, batch, hidden}, input.dtype(), input.device());

    Tensor h_prev = h0.contiguous();
    Tensor c_prev = c0.contiguous();

    bool has_bias_ih = bias_ih.numel() > 0;
    bool has_bias_hh = bias_hh.numel() > 0;

    // Pre-transpose weights once (avoids per-timestep transpose)
    Tensor W_ih_t = W_ih.transpose(0, 1).contiguous();  // (input_size, 4*hidden)
    Tensor W_hh_t = W_hh.transpose(0, 1).contiguous();  // (hidden, 4*hidden)

    // Pre-compute all input gates: (seq_len*batch, input_size) @ (input_size, 4*hidden) -> (seq_len*batch, 4*hidden)
    Tensor input_2d = input.reshape({seq_len * batch, input_size});
    Tensor all_gates_ih = cublas_matmul(input_2d, W_ih_t);  // (seq_len*batch, 4*hidden)

    // Add input bias to all gates at once
    if (has_bias_ih) {
        if (input.dtype() == DType::Float32) {
            launch_add_bias(all_gates_ih.data<float>(), bias_ih.data<float>(), seq_len * batch, gate_size, stream);
        } else if (input.dtype() == DType::Float64) {
            launch_add_bias(all_gates_ih.data<double>(), bias_ih.data<double>(), seq_len * batch, gate_size, stream);
        }
    }

    // Reshape to (seq_len, batch, gate_size) for zero-copy per-timestep slicing
    Tensor all_gates_ih_3d = all_gates_ih.reshape({seq_len, batch, gate_size});

    size_t hidden_step_bytes = batch * hidden * dtype_size(input.dtype());

    // Pre-allocate hidden-to-hidden gate buffer to avoid per-timestep allocation
    Tensor gates_hh_buf({batch, gate_size}, input.dtype(), input.device());

    for (int64_t t = 0; t < seq_len; ++t) {
        // Zero-copy view into pre-computed input gates for this timestep: (batch, 4*hidden)
        Tensor gates_ih_t = all_gates_ih_3d.slice(0, t, t + 1).squeeze(0);

        // Hidden-to-hidden: h_prev @ W_hh_t (pre-transposed)
        Tensor gates_hh = cublas_matmul(h_prev, W_hh_t);

        // Combine input and hidden gates
        Tensor gates = add_kernel(gates_ih_t, gates_hh, stream);

        // Add hidden bias
        if (has_bias_hh && input.dtype() == DType::Float32) {
            launch_add_bias(gates.data<float>(), bias_hh.data<float>(), batch, gate_size, stream);
        }
        if (has_bias_hh && input.dtype() == DType::Float64) {
            launch_add_bias(gates.data<double>(), bias_hh.data<double>(), batch, gate_size, stream);
        }

        // LSTM cell: apply activations and compute h, c
        auto [h_out, c_out] = lstm_cell_forward_kernel(gates, c_prev, batch, hidden, stream);

        // Copy h_out to output[t] (async D2D on same stream, no sync needed)
        // audit V.17: surface cudaMemcpyAsync errors.
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(
            static_cast<char*>(output.data_ptr()) + t * hidden_step_bytes,
            h_out.data_ptr(),
            hidden_step_bytes,
            cudaMemcpyDeviceToDevice,
            stream));

        h_prev = h_out;
        c_prev = c_out;
    }

    // All per-timestep work (cuBLAS, cell kernels, async D2D copies into
    // `output`) was queued on the acquired pool stream. The StreamGuard only
    // releases the stream back to the pool on destruction — it does NOT
    // synchronize. Sync here so the returned tensors are safe for a consumer
    // running on a different (default/dispatch) stream, matching GRU's contract.
    TENZOR_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {output, h_prev, c_prev};
}

// ============================================================================
// GRU Forward (single layer, full sequence)
// ============================================================================

// PyTorch GRU formula:
//   r = sigmoid(W_ir@x + b_ir + W_hr@h + b_hr)
//   z = sigmoid(W_iz@x + b_iz + W_hz@h + b_hz)
//   n = tanh(W_in@x + b_in + r * (W_hn@h + b_hn))
//   h_out = (1 - z) * n + z * h_prev
//
// The kernel applies both biases at the correct places: the r/z gates fold
// b_i* and b_h* together (sigmoid(gate_ih + gate_hh)), and the n gate keeps
// b_hn inside `r * (...)` because gru_cell_fused_kernel computes
// n = tanh(n_ih + r * n_hh) where n_hh already contains b_hn. The signature
// accepts bias = bias_ih (size 3*hidden) and bias_hh (size 3*hidden). When
// bias_hh is empty it simply means there is no hidden bias (b_h* = 0), which
// is the mathematically exact result for that configuration — NOT an
// approximation. So this matches PyTorch's GRU formula exactly in all cases.
auto gru_forward_cuda(
    const Tensor& input,     // (seq_len, batch, input_size)
    const Tensor& W_ih,      // (3*hidden, input_size)
    const Tensor& W_hh,      // (3*hidden, hidden)
    const Tensor& bias,      // bias_ih: (3*hidden) or empty
    const Tensor& h0,        // (batch, hidden)
    const Tensor& bias_hh    // bias_hh: (3*hidden) or empty
) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t seq_len = shape[0];
    int64_t batch = shape[1];
    int64_t input_size = shape[2];
    int64_t hidden = h0.shape()[1];
    int64_t gate_size = 3 * hidden;

    // The portable per-timestep path below only launches gru_cell_fused_kernel
    // for Float32/Float64, and the cuDNN fast path is Float32-only. A
    // Float16/BFloat16 GRU would otherwise skip the bias add, launch no cell
    // kernel, and copy the never-written (uninitialized) h_out into output —
    // silent garbage. Widen to Float32 up-front (on-GPU), run the real GRU, and
    // narrow results back to the input dtype.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto results = gru_forward_cuda(
            input.to(DType::Float32),
            W_ih.to(DType::Float32),
            W_hh.to(DType::Float32),
            bias.numel() > 0 ? bias.to(DType::Float32) : bias,
            h0.to(DType::Float32),
            bias_hh.numel() > 0 ? bias_hh.to(DType::Float32) : bias_hh);
        for (auto& r : results) {
            r = r.to(orig);
        }
        return results;
    }

#ifdef TENZOR_HAS_CUDNN
    // Fast path: one fused cuDNN GRU kernel (CUDNN_GRU + DOUBLE_BIAS matches
    // PyTorch's formula). Float32 inference only; the earlier divergence was an
    // nn-layer weight-wiring bug (now fixed via named accessors), not a cuDNN
    // convention mismatch.
    if (input.dtype() == DType::Float32) {
        return gru_forward_cudnn_inference(input, W_ih, W_hh, bias, h0, bias_hh);
    }
#endif

    int device_id = input.device().index;
    auto stream_guard = CUDAStreamPool::instance().acquire_guard(device_id);
    cudaStream_t stream = stream_guard.get();

    Tensor output({seq_len, batch, hidden}, input.dtype(), input.device());
    // Phase 8.5 root-cause fix: explicit two-buffer ping-pong instead of
    // std::swap on Tensor handles. The previous swap pattern was correct
    // in principle but tripped a CUDA driver "invalid argument" on
    // cudaMemcpyAsync at t=1 with the BenchShape inputs (batch=32,
    // hidden=256, seq=128). Suspected cause: `Tensor` swap rebinds the
    // shared_ptr-style storage handle and the next iteration's
    // `h_out.data_ptr()` returns a pointer that the driver doesn't
    // recognise as part of the current memcpy domain.
    //
    // Allocating two distinct ping-pong buffers up front and indexing
    // them by `t & 1` gives the driver stable, never-rebinding device
    // pointers across the whole loop.
    Tensor h_buf[2] = {
        h0.clone(),
        Tensor({batch, hidden}, input.dtype(), input.device()),
    };

    bool has_bias = bias.numel() > 0;

    // Pre-transpose weights once
    Tensor W_ih_t = W_ih.transpose(0, 1).contiguous();  // (input_size, 3*hidden)
    Tensor W_hh_t = W_hh.transpose(0, 1).contiguous();  // (hidden, 3*hidden)

    // Pre-compute all input gates: (seq_len*batch, input_size) @ (input_size, 3*hidden)
    Tensor input_2d = input.reshape({seq_len * batch, input_size});
    Tensor all_gates_ih = cublas_matmul(input_2d, W_ih_t);  // (seq_len*batch, 3*hidden)

    // Add bias to all input gates at once
    if (has_bias) {
        if (input.dtype() == DType::Float32) {
            launch_add_bias(all_gates_ih.data<float>(), bias.data<float>(), seq_len * batch, gate_size, stream);
        } else if (input.dtype() == DType::Float64) {
            launch_add_bias(all_gates_ih.data<double>(), bias.data<double>(), seq_len * batch, gate_size, stream);
        }
    }

    // Reshape to (seq_len, batch, gate_size) for zero-copy per-timestep slicing
    Tensor all_gates_ih_3d = all_gates_ih.reshape({seq_len, batch, gate_size});

    size_t hidden_step_bytes = batch * hidden * dtype_size(input.dtype());

    // Pre-compute GRU cell launch config (constant across timesteps)
    int64_t total = batch * hidden;
    int block = 256;
    int grid = rnn_clamped_grid(total, block);

    for (int64_t t = 0; t < seq_len; ++t) {
        // Zero-copy view into pre-computed input gates for this timestep.
        // The slice+squeeze on a contiguous 3D tensor is contiguous; the
        // explicit .contiguous() copy is a defensive guard against any
        // backend that might surface a non-base data_ptr for sliced views.
        Tensor gates_ih_t = all_gates_ih_3d.slice(0, t, t + 1).squeeze(0).contiguous();

        Tensor& h_prev = h_buf[t & 1];
        Tensor& h_out  = h_buf[(t + 1) & 1];

        // Hidden-to-hidden: h_prev @ W_hh_t (pre-transposed)
        Tensor gates_hh = cublas_matmul(h_prev, W_hh_t);

        // Add bias_hh to gates_hh so the kernel's gates_hh slots already
        // include b_hr / b_hz / b_hn — matches PyTorch's per-gate bias
        // placement (b_hn ends up under `r * (...)` automatically because
        // the kernel multiplies n_hh by r).
        if (bias_hh.numel() > 0) {
            if (input.dtype() == DType::Float32) {
                launch_add_bias(gates_hh.data<float>(), bias_hh.data<float>(),
                                batch, gate_size, stream);
            } else if (input.dtype() == DType::Float64) {
                launch_add_bias(gates_hh.data<double>(), bias_hh.data<double>(),
                                batch, gate_size, stream);
            }
        }

        // GRU cell — write into the next-step buffer.
        if (input.dtype() == DType::Float32) {
            gru_cell_fused_kernel<float><<<grid, block, 0, stream>>>(
                gates_ih_t.data<float>(), gates_hh.data<float>(),
                h_prev.data<float>(), h_out.data<float>(),
                batch, hidden);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else if (input.dtype() == DType::Float64) {
            gru_cell_fused_kernel<double><<<grid, block, 0, stream>>>(
                gates_ih_t.data<double>(), gates_hh.data<double>(),
                h_prev.data<double>(), h_out.data<double>(),
                batch, hidden);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }

        // audit V.17.
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(
            static_cast<char*>(output.data_ptr()) + t * hidden_step_bytes,
            h_out.data_ptr(),
            hidden_step_bytes,
            cudaMemcpyDeviceToDevice,
            stream));
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    }

    // Final hidden state lives in h_buf[seq_len & 1] after the last iteration
    // wrote into the (t+1)&1 buffer.
    TENZOR_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {output, h_buf[seq_len & 1]};
}

// ============================================================================
// Multi-layer LSTM Forward
// ============================================================================

auto lstm_multi_layer_forward_cuda(
    const Tensor& input,
    const std::vector<Tensor>& W_ih_list,
    const std::vector<Tensor>& W_hh_list,
    const std::vector<Tensor>& bias_list,
    const Tensor& h0,    // (num_layers, batch, hidden)
    const Tensor& c0     // (num_layers, batch, hidden)
) -> std::vector<Tensor> {
    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    auto shape = input.shape();
    int64_t seq_len = shape[0];
    int64_t batch = shape[1];
    int64_t hidden = h0.shape()[2];

#ifdef TENZOR_HAS_CUDNN
    // Fast path: one fused cuDNN call covering all stacked layers, instead of a
    // per-layer loop of separate cuDNN calls. Float32 inference only.
    if (input.dtype() == DType::Float32) {
        return lstm_multi_layer_forward_cudnn_inference(
            input, W_ih_list, W_hh_list, bias_list, h0, c0);
    }
#endif

    int device_id = input.device().index;
    auto stream_guard = CUDAStreamPool::instance().acquire_guard(device_id);
    cudaStream_t stream = stream_guard.get();

    // Final hidden/cell states: (num_layers, batch, hidden)
    Tensor h_n({num_layers, batch, hidden}, input.dtype(), input.device());
    Tensor c_n({num_layers, batch, hidden}, input.dtype(), input.device());

    Tensor layer_input = input;
    size_t layer_bytes = batch * hidden * dtype_size(input.dtype());

    for (int64_t l = 0; l < num_layers; ++l) {
        // Extract h0/c0 for this layer
        Tensor h_l = h0.slice(0, l, l + 1).squeeze(0).contiguous();
        Tensor c_l = c0.slice(0, l, l + 1).squeeze(0).contiguous();

        // Split bias into bias_ih and bias_hh if present
        Tensor bias_ih, bias_hh;
        if (bias_list[l].numel() > 0) {
            // bias_list[l] is combined (8*hidden for LSTM: 4*hidden bias_ih + 4*hidden bias_hh)
            int64_t half = bias_list[l].numel() / 2;
            bias_ih = bias_list[l].slice(0, 0, half).contiguous();
            bias_hh = bias_list[l].slice(0, half, half + half).contiguous();
        }

        auto result = lstm_forward_cuda(
            layer_input, W_ih_list[l], W_hh_list[l],
            bias_ih, bias_hh, h_l, c_l);

        layer_input = result[0];  // output becomes input for next layer

        // Copy final h, c to h_n[l], c_n[l]
        // audit V.17.
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(
            static_cast<char*>(h_n.data_ptr()) + l * layer_bytes,
            result[1].data_ptr(), layer_bytes,
            cudaMemcpyDeviceToDevice, stream));
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(
            static_cast<char*>(c_n.data_ptr()) + l * layer_bytes,
            result[2].data_ptr(), layer_bytes,
            cudaMemcpyDeviceToDevice, stream));
    }

    // Sync before returning so consumers on a different stream observe the
    // async D2D copies into h_n/c_n (matches single-layer lstm_forward_cuda);
    // the StreamGuard dtor only releases the slot, it does not synchronize.
    TENZOR_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {layer_input, h_n, c_n};
}

// ============================================================================
// Multi-layer GRU Forward
// ============================================================================

auto gru_multi_layer_forward_cuda(
    const Tensor& input,
    const std::vector<Tensor>& W_ih_list,
    const std::vector<Tensor>& W_hh_list,
    const std::vector<Tensor>& bias_list,
    const Tensor& h0    // (num_layers, batch, hidden)
) -> std::vector<Tensor> {
    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    auto shape = input.shape();
    int64_t seq_len = shape[0];
    int64_t batch = shape[1];
    int64_t hidden = h0.shape()[2];

    int device_id = input.device().index;
    auto stream_guard = CUDAStreamPool::instance().acquire_guard(device_id);
    cudaStream_t stream = stream_guard.get();

    Tensor h_n({num_layers, batch, hidden}, input.dtype(), input.device());
    Tensor layer_input = input;
    size_t layer_bytes = batch * hidden * dtype_size(input.dtype());

    for (int64_t l = 0; l < num_layers; ++l) {
        Tensor h_l = h0.slice(0, l, l + 1).squeeze(0).contiguous();

        auto result = gru_forward_cuda(
            layer_input, W_ih_list[l], W_hh_list[l],
            bias_list[l], h_l, Tensor{});

        layer_input = result[0];

        // audit V.17.
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(
            static_cast<char*>(h_n.data_ptr()) + l * layer_bytes,
            result[1].data_ptr(), layer_bytes,
            cudaMemcpyDeviceToDevice, stream));
    }

    // Sync before returning so consumers on a different stream observe the
    // async D2D copies into h_n (matches single-layer gru_forward_cuda).
    TENZOR_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {layer_input, h_n};
}

// ============================================================================
// Bidirectional LSTM Forward
// ============================================================================

// Kernel to concatenate forward and backward LSTM outputs along the hidden dimension
template<typename T>
__global__ void bilstm_concat_kernel(
    const T* __restrict__ fwd_output,   // (seq_len, batch, hidden)
    const T* __restrict__ bwd_output,   // (seq_len, batch, hidden)
    T* __restrict__ output,             // (seq_len, batch, 2*hidden)
    int64_t seq_len,
    int64_t batch,
    int64_t hidden
) {
    int64_t total = seq_len * batch * hidden;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        int64_t h = idx % hidden;
        int64_t rem = idx / hidden;
        int64_t b = rem % batch;
        int64_t t = rem / batch;

        int64_t src_offset = (t * batch + b) * hidden + h;
        int64_t dst_base = (t * batch + b) * 2 * hidden;

        output[dst_base + h] = fwd_output[src_offset];
        output[dst_base + hidden + h] = bwd_output[src_offset];
    }
}

// Kernel to reverse a sequence along dim 0: output[t] = input[seq_len-1-t]
template<typename T>
__global__ void reverse_sequence_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t seq_len,
    int64_t batch,
    int64_t hidden
) {
    int64_t total = seq_len * batch * hidden;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        int64_t step_size = batch * hidden;
        int64_t t = idx / step_size;
        int64_t offset = idx % step_size;
        int64_t t_rev = seq_len - 1 - t;

        output[t_rev * step_size + offset] = input[t * step_size + offset];
    }
}

auto bilstm_forward_cuda(
    const Tensor& input,
    const Tensor& W_ih_fwd, const Tensor& W_hh_fwd,
    const Tensor& bias_ih_fwd, const Tensor& bias_hh_fwd,
    const Tensor& W_ih_bwd, const Tensor& W_hh_bwd,
    const Tensor& bias_ih_bwd, const Tensor& bias_hh_bwd,
    const Tensor& h0,    // (2, batch, hidden)
    const Tensor& c0     // (2, batch, hidden)
) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t seq_len = shape[0];
    int64_t batch = shape[1];
    int64_t hidden = h0.shape()[2];

    int device_id = input.device().index;
    auto stream_guard = CUDAStreamPool::instance().acquire_guard(device_id);
    cudaStream_t stream = stream_guard.get();

    // Forward direction
    Tensor h0_fwd = h0.slice(0, 0, 1).squeeze(0).contiguous();
    Tensor c0_fwd = c0.slice(0, 0, 1).squeeze(0).contiguous();
    auto fwd_result = lstm_forward_cuda(
        input, W_ih_fwd, W_hh_fwd, bias_ih_fwd, bias_hh_fwd, h0_fwd, c0_fwd);

    // Backward direction: reverse input, run LSTM, reverse output
    Tensor input_rev({seq_len, batch, shape[2]}, input.dtype(), input.device());
    int64_t total = seq_len * batch * shape[2];
    int block = 256;
    int grid = rnn_clamped_grid(total, block);

    if (input.dtype() == DType::Float32) {
        reverse_sequence_kernel<float><<<grid, block, 0, stream>>>(
            input.data<float>(), input_rev.data<float>(),
            seq_len, batch, shape[2]);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        reverse_sequence_kernel<double><<<grid, block, 0, stream>>>(
            input.data<double>(), input_rev.data<double>(),
            seq_len, batch, shape[2]);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    }

    // input_rev was produced by the reverse kernel on `stream`; lstm_forward_cuda
    // reads it via cuBLAS on a (potentially different) handle stream. Synchronize
    // so the reversal is complete before that read — otherwise the backward LSTM
    // consumes a partially-written buffer (race / wrong results).
    TENZOR_CUDA_CHECK(cudaStreamSynchronize(stream));

    Tensor h0_bwd = h0.slice(0, 1, 2).squeeze(0).contiguous();
    Tensor c0_bwd = c0.slice(0, 1, 2).squeeze(0).contiguous();
    auto bwd_result = lstm_forward_cuda(
        input_rev, W_ih_bwd, W_hh_bwd, bias_ih_bwd, bias_hh_bwd, h0_bwd, c0_bwd);

    // Reverse backward output
    Tensor bwd_output_rev({seq_len, batch, hidden}, input.dtype(), input.device());
    int64_t total_out = seq_len * batch * hidden;
    grid = rnn_clamped_grid(total_out, block);

    if (input.dtype() == DType::Float32) {
        reverse_sequence_kernel<float><<<grid, block, 0, stream>>>(
            bwd_result[0].data<float>(), bwd_output_rev.data<float>(),
            seq_len, batch, hidden);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        reverse_sequence_kernel<double><<<grid, block, 0, stream>>>(
            bwd_result[0].data<double>(), bwd_output_rev.data<double>(),
            seq_len, batch, hidden);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    }

    // Concatenate forward and backward outputs along hidden dim: (seq_len, batch, 2*hidden)
    Tensor output({seq_len, batch, 2 * hidden}, input.dtype(), input.device());

    {
        int64_t concat_total = seq_len * batch * hidden;
        int concat_block = 256;
        int concat_grid = rnn_clamped_grid(concat_total, concat_block);

        if (input.dtype() == DType::Float32) {
            bilstm_concat_kernel<float><<<concat_grid, concat_block, 0, stream>>>(
                fwd_result[0].data<float>(), bwd_output_rev.data<float>(),
                output.data<float>(), seq_len, batch, hidden);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else if (input.dtype() == DType::Float64) {
            bilstm_concat_kernel<double><<<concat_grid, concat_block, 0, stream>>>(
                fwd_result[0].data<double>(), bwd_output_rev.data<double>(),
                output.data<double>(), seq_len, batch, hidden);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }
    }

    // Stack h_n: (2, batch, hidden)
    Tensor h_n({2, batch, hidden}, input.dtype(), input.device());
    Tensor c_n({2, batch, hidden}, input.dtype(), input.device());
    size_t state_bytes = batch * hidden * dtype_size(input.dtype());

    // audit V.17.
    TENZOR_CUDA_CHECK(cudaMemcpyAsync(h_n.data_ptr(), fwd_result[1].data_ptr(),
                    state_bytes, cudaMemcpyDeviceToDevice, stream));
    TENZOR_CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(h_n.data_ptr()) + state_bytes,
                    bwd_result[1].data_ptr(),
                    state_bytes, cudaMemcpyDeviceToDevice, stream));

    TENZOR_CUDA_CHECK(cudaMemcpyAsync(c_n.data_ptr(), fwd_result[2].data_ptr(),
                    state_bytes, cudaMemcpyDeviceToDevice, stream));
    TENZOR_CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(c_n.data_ptr()) + state_bytes,
                    bwd_result[2].data_ptr(),
                    state_bytes, cudaMemcpyDeviceToDevice, stream));

    // Sync before returning so consumers on a different stream observe the
    // async D2D copies into h_n/c_n (matches single-layer lstm_forward_cuda).
    TENZOR_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {output, h_n, c_n};
}

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_CUDA_AVAILABLE
