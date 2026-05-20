/**
 * @file mps_rnn.mm
 * @brief Host-side dispatch for Metal RNN compute shaders
 *
 * Implements LSTM and GRU cell/sequence/multi-layer/bidirectional operations
 * natively on Metal. Uses MPSMatrixMultiplication for the expensive linear
 * transforms (GEMMs) and custom Metal compute shaders for the gate activations
 * and state updates (element-wise, cheap).
 *
 * This replaces the CPU-roundtrip fallbacks previously registered for all
 * RNN OpIds in the MPS kernel registry.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "../mps_backend.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "../mps_cmd_check.h"

namespace tenzor::mps {

namespace {

// ---------------------------------------------------------------------------
// Metal infrastructure (self-contained, same pattern as mps_elementwise.mm)
// Each translation unit maintains its own pipeline cache but shares the
// same underlying Metal device via MTLCreateSystemDefaultDevice().
// ---------------------------------------------------------------------------
static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLCommandQueue> g_command_queue = nil;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_pipelines;

void ensure_initialized() {
    if (g_device == nil) {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            throw std::runtime_error("MPS RNN: No Metal device available");
        }
        g_command_queue = [g_device newCommandQueue];

        NSError* error = nil;
        g_library = [g_device newDefaultLibrary];
        if (!g_library) {
            NSString* path = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
            if (path) {
                g_library = [g_device newLibraryWithFile:path error:&error];
            }
        }
        if (!g_library) {
            throw std::runtime_error("MPS RNN: Failed to load Metal shader library");
        }
    }
}

id<MTLComputePipelineState> get_pipeline(const std::string& name) {
    auto it = g_pipelines.find(name);
    if (it != g_pipelines.end()) return it->second;

    ensure_initialized();

    NSString* func_name = [NSString stringWithUTF8String:name.c_str()];
    id<MTLFunction> func = [g_library newFunctionWithName:func_name];
    if (!func) {
        throw std::runtime_error("MPS RNN: Shader function not found: " + name);
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
    if (!pipeline) {
        throw std::runtime_error("MPS RNN: Failed to create pipeline for: " + name +
                                  " - " + [[error localizedDescription] UTF8String]);
    }

    g_pipelines[name] = pipeline;
    return pipeline;
}

id<MTLBuffer> get_buffer(const Tensor& tensor) {
    size_t bytes = tensor.numel() * dtype_size(tensor.dtype());
    return [g_device newBufferWithBytesNoCopy:const_cast<void*>(tensor.data_ptr())
                                       length:bytes
                                      options:MTLResourceStorageModeShared
                                  deallocator:nil];
}

// ---------------------------------------------------------------------------
// Helper: shader name for dtype
// ---------------------------------------------------------------------------
static std::string rnn_shader_name(const std::string& base, DType dtype) {
    return (dtype == DType::Float16) ? base + "_f16" : base;
}

// ---------------------------------------------------------------------------
// Helper: MPSDataType for tensor dtype
// ---------------------------------------------------------------------------
static MPSDataType mps_data_type(DType dtype) {
    switch (dtype) {
        case DType::Float32: return MPSDataTypeFloat32;
        case DType::Float16: return MPSDataTypeFloat16;
        default: return MPSDataTypeFloat32;
    }
}

// ---------------------------------------------------------------------------
// Helper: element size in bytes
// ---------------------------------------------------------------------------
static size_t elem_size(DType dtype) {
    return (dtype == DType::Float16) ? 2 : 4;
}

// ---------------------------------------------------------------------------
// Helper: GEMM via MPSMatrixMultiplication
//   C = alpha * A @ B + beta * C
//   A: (M, K), B: (K, N) -> C: (M, N)
// ---------------------------------------------------------------------------
static void mps_gemm(id<MTLBuffer> buf_a, id<MTLBuffer> buf_b, id<MTLBuffer> buf_c,
                      int64_t M, int64_t K, int64_t N,
                      float alpha, float beta, DType dtype,
                      bool transpose_b = false) {
    ensure_initialized();

    auto mps_dt = mps_data_type(dtype);
    size_t es = elem_size(dtype);

    int64_t B_rows = transpose_b ? N : K;
    int64_t B_cols = transpose_b ? K : N;

    MPSMatrixDescriptor* desc_a = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M columns:K
                        rowBytes:K * es dataType:mps_dt];
    MPSMatrixDescriptor* desc_b = [MPSMatrixDescriptor
        matrixDescriptorWithRows:B_rows columns:B_cols
                        rowBytes:B_cols * es dataType:mps_dt];
    MPSMatrixDescriptor* desc_c = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M columns:N
                        rowBytes:N * es dataType:mps_dt];

    MPSMatrix* mat_a = [[MPSMatrix alloc] initWithBuffer:buf_a descriptor:desc_a];
    MPSMatrix* mat_b = [[MPSMatrix alloc] initWithBuffer:buf_b descriptor:desc_b];
    MPSMatrix* mat_c = [[MPSMatrix alloc] initWithBuffer:buf_c descriptor:desc_c];

    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_device
         transposeLeft:false
        transposeRight:transpose_b
            resultRows:M
         resultColumns:N
       interiorColumns:K
                 alpha:alpha
                  beta:beta];

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    [mm encodeToCommandBuffer:cmd leftMatrix:mat_a rightMatrix:mat_b resultMatrix:mat_c];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
}

// ---------------------------------------------------------------------------
// Helper: dispatch a compute kernel with given buffers and setBytes
// ---------------------------------------------------------------------------
static void dispatch_compute(const std::string& shader,
                              const std::vector<id<MTLBuffer>>& buffers,
                              uint32_t scalar_val, int scalar_index,
                              size_t num_threads) {
    auto pipeline = get_pipeline(shader);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    for (int i = 0; i < static_cast<int>(buffers.size()); ++i) {
        [enc setBuffer:buffers[i] offset:0 atIndex:i];
    }
    [enc setBytes:&scalar_val length:sizeof(scalar_val) atIndex:scalar_index];

    MTLSize grid = MTLSizeMake(num_threads, 1, 1);
    NSUInteger tg = std::min(
        static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
        static_cast<NSUInteger>(num_threads));
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
}

// ---------------------------------------------------------------------------
// Helper: add bias in-place to a (rows, cols) buffer
// ---------------------------------------------------------------------------
static void add_bias_inplace(id<MTLBuffer> buf, id<MTLBuffer> bias_buf,
                              uint32_t cols, size_t total_elems, DType dtype) {
    dispatch_compute(rnn_shader_name("add_bias_kernel", dtype),
                     {buf, bias_buf}, cols, 2, total_elems);
}

} // anonymous namespace

// ============================================================================
// LSTM Cell Forward
// ============================================================================

std::vector<Tensor> mps_lstm_cell_forward_kernel(
    const Tensor& input, const Tensor& hx, const Tensor& cx,
    const Tensor& weight_ih, const Tensor& weight_hh,
    const Tensor& bias_ih, const Tensor& bias_hh)
{
    ensure_initialized();

    // No Float64 on Metal
    if (input.dtype() == DType::Float64) {
        throw std::runtime_error("MPS RNN: Float64 not supported on Metal");
    }

    DType dtype = input.dtype();
    size_t es = elem_size(dtype);

    auto in_shape = input.shape();
    auto hx_shape = hx.shape();
    int64_t batch = in_shape[0];
    int64_t input_size = in_shape[1];
    int64_t hidden_size = hx_shape[1];
    size_t gate_elems = batch * 4 * hidden_size;
    size_t bh_elems = batch * hidden_size;

    // Allocate gates buffer (B, 4*H)
    id<MTLBuffer> buf_gates = [g_device newBufferWithLength:gate_elems * es
                                                    options:MTLResourceStorageModeShared];

    // Zero-initialize gates
    memset([buf_gates contents], 0, gate_elems * es);

    id<MTLBuffer> buf_input = get_buffer(input);
    id<MTLBuffer> buf_hx = get_buffer(hx);
    id<MTLBuffer> buf_cx = get_buffer(cx);
    id<MTLBuffer> buf_w_ih = get_buffer(weight_ih);
    id<MTLBuffer> buf_w_hh = get_buffer(weight_hh);

    // gates = input @ W_ih^T  (B, input_size) @ (input_size, 4*H) = (B, 4*H)
    // W_ih is (4*H, input_size), transposed
    mps_gemm(buf_input, buf_w_ih, buf_gates,
             batch, input_size, 4 * hidden_size,
             1.0f, 0.0f, dtype, /*transpose_b=*/true);

    // gates += hx @ W_hh^T  (B, hidden_size) @ (hidden_size, 4*H)
    // W_hh is (4*H, hidden_size), transposed
    mps_gemm(buf_hx, buf_w_hh, buf_gates,
             batch, hidden_size, 4 * hidden_size,
             1.0f, 1.0f, dtype, /*transpose_b=*/true);

    // Add biases
    if (bias_ih.numel() > 0) {
        id<MTLBuffer> buf_bih = get_buffer(bias_ih);
        add_bias_inplace(buf_gates, buf_bih,
                         static_cast<uint32_t>(4 * hidden_size), gate_elems, dtype);
    }
    if (bias_hh.numel() > 0) {
        id<MTLBuffer> buf_bhh = get_buffer(bias_hh);
        add_bias_inplace(buf_gates, buf_bhh,
                         static_cast<uint32_t>(4 * hidden_size), gate_elems, dtype);
    }

    // Allocate outputs
    Tensor new_hidden({batch, hidden_size}, dtype, input.device());
    Tensor new_cell({batch, hidden_size}, dtype, input.device());

    id<MTLBuffer> buf_new_cell = get_buffer(new_cell);
    id<MTLBuffer> buf_new_hidden = get_buffer(new_hidden);

    // Apply gate activations + cell/hidden update
    dispatch_compute(
        rnn_shader_name("lstm_gates_kernel", dtype),
        {buf_gates, buf_cx, buf_new_cell, buf_new_hidden},
        static_cast<uint32_t>(hidden_size), 4,
        bh_elems);

    return {new_hidden, new_cell};
}

// ============================================================================
// LSTM Cell Backward
// ============================================================================

std::vector<Tensor> mps_lstm_cell_backward_kernel(
    const Tensor& grad_hy, const Tensor& grad_cy,
    const Tensor& input, const Tensor& hx, const Tensor& cx,
    const Tensor& hy, const Tensor& cy,
    const Tensor& weight_ih, const Tensor& weight_hh,
    const Tensor& bias_ih, const Tensor& bias_hh)
{
    ensure_initialized();

    if (grad_hy.dtype() == DType::Float64) {
        throw std::runtime_error("MPS RNN: Float64 not supported on Metal");
    }

    DType dtype = grad_hy.dtype();
    size_t es = elem_size(dtype);

    int64_t batch = grad_hy.shape()[0];
    int64_t hidden_size = grad_hy.shape()[1];
    int64_t input_size = input.shape()[1];
    size_t gate_elems = batch * 4 * hidden_size;
    size_t bh_elems = batch * hidden_size;

    // Step 1: Recompute pre-activation gates (same GEMM as forward)
    id<MTLBuffer> buf_gates = [g_device newBufferWithLength:gate_elems * es
                                                    options:MTLResourceStorageModeShared];
    memset([buf_gates contents], 0, gate_elems * es);

    id<MTLBuffer> buf_input = get_buffer(input);
    id<MTLBuffer> buf_hx = get_buffer(hx);
    id<MTLBuffer> buf_w_ih = get_buffer(weight_ih);
    id<MTLBuffer> buf_w_hh = get_buffer(weight_hh);

    mps_gemm(buf_input, buf_w_ih, buf_gates,
             batch, input_size, 4 * hidden_size,
             1.0f, 0.0f, dtype, true);
    mps_gemm(buf_hx, buf_w_hh, buf_gates,
             batch, hidden_size, 4 * hidden_size,
             1.0f, 1.0f, dtype, true);

    if (bias_ih.numel() > 0) {
        add_bias_inplace(buf_gates, get_buffer(bias_ih),
                         static_cast<uint32_t>(4 * hidden_size), gate_elems, dtype);
    }
    if (bias_hh.numel() > 0) {
        add_bias_inplace(buf_gates, get_buffer(bias_hh),
                         static_cast<uint32_t>(4 * hidden_size), gate_elems, dtype);
    }

    // Step 2: Compute gate gradients via Metal shader
    id<MTLBuffer> buf_d_gates = [g_device newBufferWithLength:gate_elems * es
                                                      options:MTLResourceStorageModeShared];
    Tensor grad_cx_out({batch, hidden_size}, dtype, grad_hy.device());
    id<MTLBuffer> buf_grad_cx = get_buffer(grad_cx_out);

    dispatch_compute(
        rnn_shader_name("lstm_backward_gates_kernel", dtype),
        {buf_gates, get_buffer(cx), get_buffer(grad_hy), get_buffer(grad_cy),
         buf_d_gates, buf_grad_cx},
        static_cast<uint32_t>(hidden_size), 6,
        bh_elems);

    // Step 3: Compute input/hidden gradients via GEMM
    // grad_input = d_gates @ W_ih  (B, 4*H) @ (4*H, input_size) = (B, input_size)
    Tensor grad_input({batch, input_size}, dtype, grad_hy.device());
    id<MTLBuffer> buf_grad_input = get_buffer(grad_input);
    mps_gemm(buf_d_gates, buf_w_ih, buf_grad_input,
             batch, 4 * hidden_size, input_size,
             1.0f, 0.0f, dtype, false);

    // grad_hx = d_gates @ W_hh  (B, 4*H) @ (4*H, hidden_size) = (B, hidden_size)
    Tensor grad_hx({batch, hidden_size}, dtype, grad_hy.device());
    id<MTLBuffer> buf_grad_hx = get_buffer(grad_hx);
    mps_gemm(buf_d_gates, buf_w_hh, buf_grad_hx,
             batch, 4 * hidden_size, hidden_size,
             1.0f, 0.0f, dtype, false);

    // Step 4: Compute weight gradients via GEMM
    // grad_w_ih = d_gates^T @ input  (4*H, B) @ (B, input_size) = (4*H, input_size)
    Tensor grad_w_ih({4 * hidden_size, input_size}, dtype, grad_hy.device());
    id<MTLBuffer> buf_grad_w_ih = get_buffer(grad_w_ih);

    // For d_gates^T @ input, we compute input^T @ d_gates then transpose conceptually
    // Actually: (4H, B) @ (B, IS) - we need to transpose d_gates
    // MPSMatrixMultiplication with transposeLeft=true: A^T @ B
    {
        auto mps_dt = mps_data_type(dtype);
        MPSMatrixDescriptor* desc_dg = [MPSMatrixDescriptor
            matrixDescriptorWithRows:batch columns:4*hidden_size
                            rowBytes:4*hidden_size*es dataType:mps_dt];
        MPSMatrixDescriptor* desc_in = [MPSMatrixDescriptor
            matrixDescriptorWithRows:batch columns:input_size
                            rowBytes:input_size*es dataType:mps_dt];
        MPSMatrixDescriptor* desc_out = [MPSMatrixDescriptor
            matrixDescriptorWithRows:4*hidden_size columns:input_size
                            rowBytes:input_size*es dataType:mps_dt];

        MPSMatrix* mat_dg = [[MPSMatrix alloc] initWithBuffer:buf_d_gates descriptor:desc_dg];
        MPSMatrix* mat_in = [[MPSMatrix alloc] initWithBuffer:buf_input descriptor:desc_in];
        MPSMatrix* mat_out = [[MPSMatrix alloc] initWithBuffer:buf_grad_w_ih descriptor:desc_out];

        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device transposeLeft:true transposeRight:false
                resultRows:4*hidden_size resultColumns:input_size interiorColumns:batch
                     alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        [mm encodeToCommandBuffer:cmd leftMatrix:mat_dg rightMatrix:mat_in resultMatrix:mat_out];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    }

    // grad_w_hh = d_gates^T @ hx  (4*H, B) @ (B, hidden_size)
    Tensor grad_w_hh({4 * hidden_size, hidden_size}, dtype, grad_hy.device());
    id<MTLBuffer> buf_grad_w_hh = get_buffer(grad_w_hh);
    {
        auto mps_dt = mps_data_type(dtype);
        MPSMatrixDescriptor* desc_dg = [MPSMatrixDescriptor
            matrixDescriptorWithRows:batch columns:4*hidden_size
                            rowBytes:4*hidden_size*es dataType:mps_dt];
        MPSMatrixDescriptor* desc_hx = [MPSMatrixDescriptor
            matrixDescriptorWithRows:batch columns:hidden_size
                            rowBytes:hidden_size*es dataType:mps_dt];
        MPSMatrixDescriptor* desc_out = [MPSMatrixDescriptor
            matrixDescriptorWithRows:4*hidden_size columns:hidden_size
                            rowBytes:hidden_size*es dataType:mps_dt];

        MPSMatrix* mat_dg = [[MPSMatrix alloc] initWithBuffer:buf_d_gates descriptor:desc_dg];
        MPSMatrix* mat_hx = [[MPSMatrix alloc] initWithBuffer:buf_hx descriptor:desc_hx];
        MPSMatrix* mat_out = [[MPSMatrix alloc] initWithBuffer:buf_grad_w_hh descriptor:desc_out];

        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device transposeLeft:true transposeRight:false
                resultRows:4*hidden_size resultColumns:hidden_size interiorColumns:batch
                     alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        [mm encodeToCommandBuffer:cmd leftMatrix:mat_dg rightMatrix:mat_hx resultMatrix:mat_out];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    }

    // Step 5: Bias gradients = sum of d_gates over batch dimension
    // For simplicity, reduce on CPU from the d_gates buffer (small data)
    Tensor grad_bias_ih({4 * hidden_size}, dtype, grad_hy.device());
    Tensor grad_bias_hh({4 * hidden_size}, dtype, grad_hy.device());

    if (dtype == DType::Float32) {
        const float* dg = static_cast<const float*>([buf_d_gates contents]);
        float* gbih = grad_bias_ih.data<float>();
        float* gbhh = grad_bias_hh.data<float>();
        for (int64_t g = 0; g < 4 * hidden_size; ++g) {
            float sum = 0.0f;
            for (int64_t b = 0; b < batch; ++b) {
                sum += dg[b * 4 * hidden_size + g];
            }
            gbih[g] = sum;
            gbhh[g] = sum;
        }
    } else {
        // Float16: accumulate in float
        const uint16_t* dg = static_cast<const uint16_t*>([buf_d_gates contents]);
        // Use a simple float accumulation then write back
        auto h2f = [](uint16_t h) -> float {
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1f;
            uint32_t mant = h & 0x3ff;
            if (exp == 0) return sign ? -0.0f : 0.0f;
            if (exp == 31) return sign ? -INFINITY : INFINITY;
            float val = std::ldexp(1.0f + mant / 1024.0f, static_cast<int>(exp) - 15);
            return sign ? -val : val;
        };
        auto f2h = [](float f) -> uint16_t {
            uint32_t bits;
            std::memcpy(&bits, &f, sizeof(bits));
            uint32_t s = (bits >> 16) & 0x8000;
            int32_t e = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
            uint32_t m = bits & 0x7fffff;
            if (e <= 0) return static_cast<uint16_t>(s);
            if (e >= 31) return static_cast<uint16_t>(s | 0x7c00);
            return static_cast<uint16_t>(s | (e << 10) | (m >> 13));
        };
        uint16_t* gbih = reinterpret_cast<uint16_t*>(grad_bias_ih.data_ptr());
        uint16_t* gbhh = reinterpret_cast<uint16_t*>(grad_bias_hh.data_ptr());
        for (int64_t g = 0; g < 4 * hidden_size; ++g) {
            float sum = 0.0f;
            for (int64_t b = 0; b < batch; ++b) {
                sum += h2f(dg[b * 4 * hidden_size + g]);
            }
            gbih[g] = f2h(sum);
            gbhh[g] = f2h(sum);
        }
    }

    return {grad_input, grad_hx, grad_cx_out, grad_w_ih, grad_w_hh, grad_bias_ih, grad_bias_hh};
}

// ============================================================================
// LSTM Forward (single layer, full sequence)
// ============================================================================

std::vector<Tensor> mps_lstm_forward_kernel(
    const Tensor& input,
    const Tensor& W_ih, const Tensor& W_hh,
    const Tensor& bias_ih, const Tensor& bias_hh,
    const Tensor& h0, const Tensor& c0)
{
    ensure_initialized();

    if (input.dtype() == DType::Float64) {
        throw std::runtime_error("MPS RNN: Float64 not supported on Metal");
    }

    DType dtype = input.dtype();
    size_t es = elem_size(dtype);

    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch = input_shape[1];
    int64_t input_size = input_shape[2];
    int64_t hidden_size = h0.shape()[1];

    size_t gate_elems = batch * 4 * hidden_size;
    size_t bh_elems = batch * hidden_size;

    // Pre-compute all input projections at once:
    // all_ih = input_2d @ W_ih^T  where input_2d is (seq_len*batch, input_size)
    // Result: (seq_len*batch, 4*H)
    Tensor input_contig = input.contiguous();
    id<MTLBuffer> buf_input_2d = get_buffer(input_contig);
    id<MTLBuffer> buf_w_ih = get_buffer(W_ih);
    id<MTLBuffer> buf_w_hh = get_buffer(W_hh);

    size_t all_ih_elems = seq_len * batch * 4 * hidden_size;
    id<MTLBuffer> buf_all_ih = [g_device newBufferWithLength:all_ih_elems * es
                                                     options:MTLResourceStorageModeShared];
    memset([buf_all_ih contents], 0, all_ih_elems * es);

    mps_gemm(buf_input_2d, buf_w_ih, buf_all_ih,
             seq_len * batch, input_size, 4 * hidden_size,
             1.0f, 0.0f, dtype, true);

    // Add input bias to all timesteps at once
    if (bias_ih.numel() > 0) {
        add_bias_inplace(buf_all_ih, get_buffer(bias_ih),
                         static_cast<uint32_t>(4 * hidden_size), all_ih_elems, dtype);
    }

    // Allocate output
    Tensor output({seq_len, batch, hidden_size}, dtype, input.device());
    Tensor h_n({batch, hidden_size}, dtype, input.device());
    Tensor c_n({batch, hidden_size}, dtype, input.device());

    // Copy initial states
    Tensor h_cur = h0.contiguous();
    Tensor c_cur = c0.contiguous();

    // Per-timestep gate buffer (reused)
    id<MTLBuffer> buf_gates = [g_device newBufferWithLength:gate_elems * es
                                                    options:MTLResourceStorageModeShared];

    float* output_base = (dtype == DType::Float32) ? output.data<float>() : nullptr;
    void* output_ptr = output.data_ptr();

    for (int64_t t = 0; t < seq_len; ++t) {
        // Copy pre-computed ih gates for this timestep
        size_t offset = t * batch * 4 * hidden_size * es;
        memcpy([buf_gates contents],
               static_cast<const uint8_t*>([buf_all_ih contents]) + offset,
               gate_elems * es);

        // Add hidden projection: gates += h_cur @ W_hh^T
        id<MTLBuffer> buf_h = get_buffer(h_cur);
        mps_gemm(buf_h, buf_w_hh, buf_gates,
                 batch, hidden_size, 4 * hidden_size,
                 1.0f, 1.0f, dtype, true);

        // Add hidden bias
        if (bias_hh.numel() > 0) {
            add_bias_inplace(buf_gates, get_buffer(bias_hh),
                             static_cast<uint32_t>(4 * hidden_size), gate_elems, dtype);
        }

        // Allocate new h/c for this step
        Tensor h_next({batch, hidden_size}, dtype, input.device());
        Tensor c_next({batch, hidden_size}, dtype, input.device());

        id<MTLBuffer> buf_c_cur = get_buffer(c_cur);
        id<MTLBuffer> buf_h_next = get_buffer(h_next);
        id<MTLBuffer> buf_c_next = get_buffer(c_next);

        // Apply gate activations + state update
        dispatch_compute(
            rnn_shader_name("lstm_gates_kernel", dtype),
            {buf_gates, buf_c_cur, buf_c_next, buf_h_next},
            static_cast<uint32_t>(hidden_size), 4,
            bh_elems);

        // Copy hidden state to output[:, t, :]
        memcpy(static_cast<uint8_t*>(output_ptr) + t * bh_elems * es,
               h_next.data_ptr(), bh_elems * es);

        h_cur = h_next;
        c_cur = c_next;
    }

    // Copy final states
    memcpy(h_n.data_ptr(), h_cur.data_ptr(), bh_elems * es);
    memcpy(c_n.data_ptr(), c_cur.data_ptr(), bh_elems * es);

    return {output, h_n, c_n};
}

// ============================================================================
// LSTM Multi-Layer Forward
// ============================================================================

std::vector<Tensor> mps_lstm_multilayer_forward_kernel(
    const Tensor& input,
    const std::vector<Tensor>& W_ih_list,
    const std::vector<Tensor>& W_hh_list,
    const std::vector<Tensor>& bias_list,
    const Tensor& h0, const Tensor& c0)
{
    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch = input_shape[1];
    int64_t hidden_size = h0.shape()[2];  // h0: (num_layers, batch, hidden)
    DType dtype = input.dtype();
    size_t es = elem_size(dtype);
    size_t bh = batch * hidden_size;

    Tensor h0_contig = h0.contiguous();
    Tensor c0_contig = c0.contiguous();

    Tensor layer_input = input;
    std::vector<Tensor> h_states, c_states;

    for (int64_t l = 0; l < num_layers; ++l) {
        // Extract per-layer initial states
        Tensor h0_l({batch, hidden_size}, dtype, input.device());
        Tensor c0_l({batch, hidden_size}, dtype, input.device());
        memcpy(h0_l.data_ptr(),
               static_cast<const uint8_t*>(h0_contig.data_ptr()) + l * bh * es,
               bh * es);
        memcpy(c0_l.data_ptr(),
               static_cast<const uint8_t*>(c0_contig.data_ptr()) + l * bh * es,
               bh * es);

        // Bias: multilayer uses combined bias, pass as bias_ih with empty bias_hh
        Tensor bias_ih_l = (!bias_list.empty() && bias_list[l].numel() > 0)
                           ? bias_list[l] : Tensor({0}, dtype, input.device());
        Tensor bias_hh_empty({0}, dtype, input.device());

        auto layer_out = mps_lstm_forward_kernel(
            layer_input, W_ih_list[l], W_hh_list[l],
            bias_ih_l, bias_hh_empty, h0_l, c0_l);

        h_states.push_back(layer_out[1]);
        c_states.push_back(layer_out[2]);
        layer_input = layer_out[0];
    }

    // Stack outputs
    Tensor output({seq_len, batch, hidden_size}, dtype, input.device());
    memcpy(output.data_ptr(), layer_input.data_ptr(),
           seq_len * bh * es);

    Tensor h_n({num_layers, batch, hidden_size}, dtype, input.device());
    Tensor c_n({num_layers, batch, hidden_size}, dtype, input.device());
    for (int64_t l = 0; l < num_layers; ++l) {
        memcpy(static_cast<uint8_t*>(h_n.data_ptr()) + l * bh * es,
               h_states[l].data_ptr(), bh * es);
        memcpy(static_cast<uint8_t*>(c_n.data_ptr()) + l * bh * es,
               c_states[l].data_ptr(), bh * es);
    }

    return {output, h_n, c_n};
}

// ============================================================================
// Bidirectional LSTM Forward
// ============================================================================

std::vector<Tensor> mps_bilstm_forward_kernel(
    const Tensor& input,
    const Tensor& W_ih_fwd, const Tensor& W_hh_fwd,
    const Tensor& bias_ih_fwd, const Tensor& bias_hh_fwd,
    const Tensor& W_ih_bwd, const Tensor& W_hh_bwd,
    const Tensor& bias_ih_bwd, const Tensor& bias_hh_bwd,
    const Tensor& h0, const Tensor& c0)
{
    DType dtype = input.dtype();
    size_t es = elem_size(dtype);

    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch = input_shape[1];
    int64_t hidden_size = h0.shape()[2];  // h0: (2, batch, hidden)
    size_t bh = batch * hidden_size;

    Tensor h0_contig = h0.contiguous();
    Tensor c0_contig = c0.contiguous();

    // Extract per-direction initial states
    Tensor h0_fwd({batch, hidden_size}, dtype, input.device());
    Tensor c0_fwd({batch, hidden_size}, dtype, input.device());
    Tensor h0_bwd({batch, hidden_size}, dtype, input.device());
    Tensor c0_bwd({batch, hidden_size}, dtype, input.device());

    memcpy(h0_fwd.data_ptr(), h0_contig.data_ptr(), bh * es);
    memcpy(c0_fwd.data_ptr(), c0_contig.data_ptr(), bh * es);
    memcpy(h0_bwd.data_ptr(),
           static_cast<const uint8_t*>(h0_contig.data_ptr()) + bh * es, bh * es);
    memcpy(c0_bwd.data_ptr(),
           static_cast<const uint8_t*>(c0_contig.data_ptr()) + bh * es, bh * es);

    // Forward direction
    auto fwd_out = mps_lstm_forward_kernel(
        input, W_ih_fwd, W_hh_fwd, bias_ih_fwd, bias_hh_fwd, h0_fwd, c0_fwd);

    // Backward direction: reverse the input sequence
    // Create reversed input: flip along dim 0
    Tensor input_contig = input.contiguous();
    Tensor input_rev({seq_len, batch, input_shape[2]}, dtype, input.device());
    size_t step_bytes = batch * input_shape[2] * es;
    for (int64_t t = 0; t < seq_len; ++t) {
        memcpy(static_cast<uint8_t*>(input_rev.data_ptr()) + t * step_bytes,
               static_cast<const uint8_t*>(input_contig.data_ptr()) + (seq_len - 1 - t) * step_bytes,
               step_bytes);
    }

    auto bwd_out = mps_lstm_forward_kernel(
        input_rev, W_ih_bwd, W_hh_bwd, bias_ih_bwd, bias_hh_bwd, h0_bwd, c0_bwd);

    // Concatenate outputs: (seq, batch, 2*hidden)
    // Forward output is fwd_out[0]: (seq, batch, hidden)
    // Backward output is bwd_out[0]: (seq, batch, hidden) but reversed
    Tensor output({seq_len, batch, 2 * hidden_size}, dtype, input.device());
    size_t h_bytes = hidden_size * es;

    for (int64_t t = 0; t < seq_len; ++t) {
        for (int64_t b = 0; b < batch; ++b) {
            size_t out_offset = (t * batch + b) * 2 * hidden_size * es;
            size_t fwd_offset = (t * batch + b) * hidden_size * es;
            size_t bwd_offset = ((seq_len - 1 - t) * batch + b) * hidden_size * es;

            memcpy(static_cast<uint8_t*>(output.data_ptr()) + out_offset,
                   static_cast<const uint8_t*>(fwd_out[0].data_ptr()) + fwd_offset,
                   h_bytes);
            memcpy(static_cast<uint8_t*>(output.data_ptr()) + out_offset + h_bytes,
                   static_cast<const uint8_t*>(bwd_out[0].data_ptr()) + bwd_offset,
                   h_bytes);
        }
    }

    // Stack final states: (2, batch, hidden)
    Tensor h_n({2, batch, hidden_size}, dtype, input.device());
    Tensor c_n({2, batch, hidden_size}, dtype, input.device());

    memcpy(h_n.data_ptr(), fwd_out[1].data_ptr(), bh * es);
    memcpy(static_cast<uint8_t*>(h_n.data_ptr()) + bh * es,
           bwd_out[1].data_ptr(), bh * es);
    memcpy(c_n.data_ptr(), fwd_out[2].data_ptr(), bh * es);
    memcpy(static_cast<uint8_t*>(c_n.data_ptr()) + bh * es,
           bwd_out[2].data_ptr(), bh * es);

    return {output, h_n, c_n};
}

// ============================================================================
// GRU Cell Forward
// ============================================================================

std::vector<Tensor> mps_gru_cell_forward_kernel(
    const Tensor& input, const Tensor& hx,
    const Tensor& weight_ih, const Tensor& weight_hh,
    const Tensor& bias_ih, const Tensor& bias_hh)
{
    ensure_initialized();

    if (input.dtype() == DType::Float64) {
        throw std::runtime_error("MPS RNN: Float64 not supported on Metal");
    }

    DType dtype = input.dtype();
    size_t es = elem_size(dtype);

    int64_t batch = input.shape()[0];
    int64_t input_size = input.shape()[1];
    int64_t hidden_size = hx.shape()[1];

    size_t ih_elems = batch * 3 * hidden_size;
    size_t bh_elems = batch * hidden_size;

    // Compute input-hidden gates: gates_ih = input @ W_ih^T + bias_ih
    id<MTLBuffer> buf_gates_ih = [g_device newBufferWithLength:ih_elems * es
                                                       options:MTLResourceStorageModeShared];
    memset([buf_gates_ih contents], 0, ih_elems * es);

    mps_gemm(get_buffer(input), get_buffer(weight_ih), buf_gates_ih,
             batch, input_size, 3 * hidden_size,
             1.0f, 0.0f, dtype, true);

    if (bias_ih.numel() > 0) {
        add_bias_inplace(buf_gates_ih, get_buffer(bias_ih),
                         static_cast<uint32_t>(3 * hidden_size), ih_elems, dtype);
    }

    // Compute hidden-hidden gates: gates_hh = hx @ W_hh^T + bias_hh
    id<MTLBuffer> buf_gates_hh = [g_device newBufferWithLength:ih_elems * es
                                                       options:MTLResourceStorageModeShared];
    memset([buf_gates_hh contents], 0, ih_elems * es);

    mps_gemm(get_buffer(hx), get_buffer(weight_hh), buf_gates_hh,
             batch, hidden_size, 3 * hidden_size,
             1.0f, 0.0f, dtype, true);

    if (bias_hh.numel() > 0) {
        add_bias_inplace(buf_gates_hh, get_buffer(bias_hh),
                         static_cast<uint32_t>(3 * hidden_size), ih_elems, dtype);
    }

    // Apply gate activations + hidden update
    Tensor new_hidden({batch, hidden_size}, dtype, input.device());
    id<MTLBuffer> buf_new_hidden = get_buffer(new_hidden);

    dispatch_compute(
        rnn_shader_name("gru_gates_kernel", dtype),
        {buf_gates_ih, buf_gates_hh, get_buffer(hx), buf_new_hidden},
        static_cast<uint32_t>(hidden_size), 4,
        bh_elems);

    // GRU cell returns {new_hidden} wrapped in vector (CPU returns single Tensor
    // but dispatch table expects vector)
    return {new_hidden};
}

// ============================================================================
// GRU Cell Backward
// ============================================================================

std::vector<Tensor> mps_gru_cell_backward_kernel(
    const Tensor& grad_hy, const Tensor& input, const Tensor& hx,
    const Tensor& weight_ih, const Tensor& weight_hh,
    const Tensor& bias_ih, const Tensor& bias_hh)
{
    ensure_initialized();

    if (grad_hy.dtype() == DType::Float64) {
        throw std::runtime_error("MPS RNN: Float64 not supported on Metal");
    }

    DType dtype = grad_hy.dtype();
    size_t es = elem_size(dtype);

    int64_t batch = grad_hy.shape()[0];
    int64_t hidden_size = grad_hy.shape()[1];
    int64_t input_size = input.shape()[1];
    size_t gate3_elems = batch * 3 * hidden_size;
    size_t bh_elems = batch * hidden_size;

    // Step 1: Recompute gates (same as forward)
    id<MTLBuffer> buf_gates_ih = [g_device newBufferWithLength:gate3_elems * es
                                                       options:MTLResourceStorageModeShared];
    memset([buf_gates_ih contents], 0, gate3_elems * es);
    mps_gemm(get_buffer(input), get_buffer(weight_ih), buf_gates_ih,
             batch, input_size, 3 * hidden_size, 1.0f, 0.0f, dtype, true);
    if (bias_ih.numel() > 0) {
        add_bias_inplace(buf_gates_ih, get_buffer(bias_ih),
                         static_cast<uint32_t>(3 * hidden_size), gate3_elems, dtype);
    }

    id<MTLBuffer> buf_gates_hh = [g_device newBufferWithLength:gate3_elems * es
                                                       options:MTLResourceStorageModeShared];
    memset([buf_gates_hh contents], 0, gate3_elems * es);
    mps_gemm(get_buffer(hx), get_buffer(weight_hh), buf_gates_hh,
             batch, hidden_size, 3 * hidden_size, 1.0f, 0.0f, dtype, true);
    if (bias_hh.numel() > 0) {
        add_bias_inplace(buf_gates_hh, get_buffer(bias_hh),
                         static_cast<uint32_t>(3 * hidden_size), gate3_elems, dtype);
    }

    // Step 2: Compute gate gradients
    id<MTLBuffer> buf_d_gates_ih = [g_device newBufferWithLength:gate3_elems * es
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_d_gates_hh = [g_device newBufferWithLength:gate3_elems * es
                                                         options:MTLResourceStorageModeShared];
    Tensor grad_hx_out({batch, hidden_size}, dtype, grad_hy.device());

    dispatch_compute(
        rnn_shader_name("gru_backward_gates_kernel", dtype),
        {buf_gates_ih, buf_gates_hh, get_buffer(hx), get_buffer(grad_hy),
         buf_d_gates_ih, buf_d_gates_hh, get_buffer(grad_hx_out)},
        static_cast<uint32_t>(hidden_size), 7,
        bh_elems);

    // Step 3: Compute input gradient
    // grad_input = d_gates_ih @ W_ih  (B, 3H) @ (3H, IS)
    Tensor grad_input({batch, input_size}, dtype, grad_hy.device());
    mps_gemm(buf_d_gates_ih, get_buffer(weight_ih), get_buffer(grad_input),
             batch, 3 * hidden_size, input_size, 1.0f, 0.0f, dtype, false);

    // grad_hx += d_gates_hh[:2H] @ W_hh[:2H]  for r,z gates (direct hx path)
    // Plus existing grad_hx_out from the shader (z * dh)
    // For simplicity, compute full d_gates_hh @ W_hh and add to grad_hx_out
    id<MTLBuffer> buf_grad_hx_tmp = [g_device newBufferWithLength:bh_elems * es
                                                           options:MTLResourceStorageModeShared];
    memset([buf_grad_hx_tmp contents], 0, bh_elems * es);

    // For r,z gates: d_gates_hh[:, :2H] @ W_hh[:2H, :]
    // We need a partial GEMM. Use full d_gates_hh @ W_hh but only the first 2H columns
    // contribute directly. However, the n-gate hh gradients already account for r*hx
    // in the shader. So we can do the full GEMM.
    mps_gemm(buf_d_gates_hh, get_buffer(weight_hh), buf_grad_hx_tmp,
             batch, 3 * hidden_size, hidden_size, 1.0f, 0.0f, dtype, false);

    // Add tmp gradient to grad_hx_out (element-wise add)
    {
        auto pipeline = get_pipeline(rnn_shader_name("add_bias_kernel", dtype));
        // Repurpose add_bias_kernel: just add element-wise since cols=bh_elems and rows=1
        // Actually we need a simple vector add. Use the in-place add from elementwise.
        // Simpler: do it on CPU for the small hidden state
        if (dtype == DType::Float32) {
            float* dst = grad_hx_out.data<float>();
            const float* src = static_cast<const float*>([buf_grad_hx_tmp contents]);
            for (size_t i = 0; i < bh_elems; ++i) dst[i] += src[i];
        } else {
            uint16_t* dst = reinterpret_cast<uint16_t*>(grad_hx_out.data_ptr());
            const uint16_t* src = static_cast<const uint16_t*>([buf_grad_hx_tmp contents]);
            auto h2f = [](uint16_t h) -> float {
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp = (h >> 10) & 0x1f;
                uint32_t mant = h & 0x3ff;
                if (exp == 0) return sign ? -0.0f : 0.0f;
                if (exp == 31) return sign ? -INFINITY : INFINITY;
                float val = std::ldexp(1.0f + mant / 1024.0f, static_cast<int>(exp) - 15);
                return sign ? -val : val;
            };
            auto f2h = [](float f) -> uint16_t {
                uint32_t bits;
                std::memcpy(&bits, &f, sizeof(bits));
                uint32_t s = (bits >> 16) & 0x8000;
                int32_t e = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
                uint32_t m = bits & 0x7fffff;
                if (e <= 0) return static_cast<uint16_t>(s);
                if (e >= 31) return static_cast<uint16_t>(s | 0x7c00);
                return static_cast<uint16_t>(s | (e << 10) | (m >> 13));
            };
            for (size_t i = 0; i < bh_elems; ++i) {
                dst[i] = f2h(h2f(dst[i]) + h2f(src[i]));
            }
        }
    }

    // Step 4: Weight gradients
    auto mps_dt = mps_data_type(dtype);

    // grad_w_ih = d_gates_ih^T @ input
    Tensor grad_w_ih({3 * hidden_size, input_size}, dtype, grad_hy.device());
    {
        MPSMatrixDescriptor* desc_dg = [MPSMatrixDescriptor
            matrixDescriptorWithRows:batch columns:3*hidden_size
                            rowBytes:3*hidden_size*es dataType:mps_dt];
        MPSMatrixDescriptor* desc_in = [MPSMatrixDescriptor
            matrixDescriptorWithRows:batch columns:input_size
                            rowBytes:input_size*es dataType:mps_dt];
        MPSMatrixDescriptor* desc_out = [MPSMatrixDescriptor
            matrixDescriptorWithRows:3*hidden_size columns:input_size
                            rowBytes:input_size*es dataType:mps_dt];

        MPSMatrix* mat_dg = [[MPSMatrix alloc] initWithBuffer:buf_d_gates_ih descriptor:desc_dg];
        MPSMatrix* mat_in = [[MPSMatrix alloc] initWithBuffer:get_buffer(input) descriptor:desc_in];
        MPSMatrix* mat_out = [[MPSMatrix alloc] initWithBuffer:get_buffer(grad_w_ih) descriptor:desc_out];

        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device transposeLeft:true transposeRight:false
                resultRows:3*hidden_size resultColumns:input_size interiorColumns:batch
                     alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        [mm encodeToCommandBuffer:cmd leftMatrix:mat_dg rightMatrix:mat_in resultMatrix:mat_out];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    }

    // grad_w_hh = d_gates_hh^T @ hx
    Tensor grad_w_hh({3 * hidden_size, hidden_size}, dtype, grad_hy.device());
    {
        MPSMatrixDescriptor* desc_dg = [MPSMatrixDescriptor
            matrixDescriptorWithRows:batch columns:3*hidden_size
                            rowBytes:3*hidden_size*es dataType:mps_dt];
        MPSMatrixDescriptor* desc_hx = [MPSMatrixDescriptor
            matrixDescriptorWithRows:batch columns:hidden_size
                            rowBytes:hidden_size*es dataType:mps_dt];
        MPSMatrixDescriptor* desc_out = [MPSMatrixDescriptor
            matrixDescriptorWithRows:3*hidden_size columns:hidden_size
                            rowBytes:hidden_size*es dataType:mps_dt];

        MPSMatrix* mat_dg = [[MPSMatrix alloc] initWithBuffer:buf_d_gates_hh descriptor:desc_dg];
        MPSMatrix* mat_hx = [[MPSMatrix alloc] initWithBuffer:get_buffer(hx) descriptor:desc_hx];
        MPSMatrix* mat_out = [[MPSMatrix alloc] initWithBuffer:get_buffer(grad_w_hh) descriptor:desc_out];

        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device transposeLeft:true transposeRight:false
                resultRows:3*hidden_size resultColumns:hidden_size interiorColumns:batch
                     alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        [mm encodeToCommandBuffer:cmd leftMatrix:mat_dg rightMatrix:mat_hx resultMatrix:mat_out];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    }

    // Step 5: Bias gradients = sum of d_gates over batch
    Tensor grad_bias_ih({3 * hidden_size}, dtype, grad_hy.device());
    Tensor grad_bias_hh({3 * hidden_size}, dtype, grad_hy.device());

    if (dtype == DType::Float32) {
        const float* dg_ih = static_cast<const float*>([buf_d_gates_ih contents]);
        const float* dg_hh = static_cast<const float*>([buf_d_gates_hh contents]);
        float* gbih = grad_bias_ih.data<float>();
        float* gbhh = grad_bias_hh.data<float>();
        for (int64_t g = 0; g < 3 * hidden_size; ++g) {
            float sum_ih = 0.0f, sum_hh = 0.0f;
            for (int64_t b = 0; b < batch; ++b) {
                sum_ih += dg_ih[b * 3 * hidden_size + g];
                sum_hh += dg_hh[b * 3 * hidden_size + g];
            }
            gbih[g] = sum_ih;
            gbhh[g] = sum_hh;
        }
    } else {
        auto h2f = [](uint16_t h) -> float {
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp_v = (h >> 10) & 0x1f;
            uint32_t mant = h & 0x3ff;
            if (exp_v == 0) return sign ? -0.0f : 0.0f;
            if (exp_v == 31) return sign ? -INFINITY : INFINITY;
            float val = std::ldexp(1.0f + mant / 1024.0f, static_cast<int>(exp_v) - 15);
            return sign ? -val : val;
        };
        auto f2h = [](float f) -> uint16_t {
            uint32_t bits;
            std::memcpy(&bits, &f, sizeof(bits));
            uint32_t s = (bits >> 16) & 0x8000;
            int32_t e = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
            uint32_t m = bits & 0x7fffff;
            if (e <= 0) return static_cast<uint16_t>(s);
            if (e >= 31) return static_cast<uint16_t>(s | 0x7c00);
            return static_cast<uint16_t>(s | (e << 10) | (m >> 13));
        };
        const uint16_t* dg_ih = static_cast<const uint16_t*>([buf_d_gates_ih contents]);
        const uint16_t* dg_hh = static_cast<const uint16_t*>([buf_d_gates_hh contents]);
        uint16_t* gbih = reinterpret_cast<uint16_t*>(grad_bias_ih.data_ptr());
        uint16_t* gbhh = reinterpret_cast<uint16_t*>(grad_bias_hh.data_ptr());
        for (int64_t g = 0; g < 3 * hidden_size; ++g) {
            float sum_ih = 0.0f, sum_hh = 0.0f;
            for (int64_t b = 0; b < batch; ++b) {
                sum_ih += h2f(dg_ih[b * 3 * hidden_size + g]);
                sum_hh += h2f(dg_hh[b * 3 * hidden_size + g]);
            }
            gbih[g] = f2h(sum_ih);
            gbhh[g] = f2h(sum_hh);
        }
    }

    return {grad_input, grad_hx_out, grad_w_ih, grad_w_hh, grad_bias_ih, grad_bias_hh};
}

// ============================================================================
// GRU Forward (single layer, full sequence)
// ============================================================================

std::vector<Tensor> mps_gru_forward_kernel(
    const Tensor& input,
    const Tensor& W_ih, const Tensor& W_hh,
    const Tensor& bias,
    const Tensor& h0)
{
    ensure_initialized();

    if (input.dtype() == DType::Float64) {
        throw std::runtime_error("MPS RNN: Float64 not supported on Metal");
    }

    DType dtype = input.dtype();
    size_t es = elem_size(dtype);

    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch = input_shape[1];
    int64_t input_size = input_shape[2];
    int64_t hidden_size = h0.shape()[1];

    size_t gate3_elems = batch * 3 * hidden_size;
    size_t bh_elems = batch * hidden_size;

    // Pre-compute all input projections: (seq*B, IS) @ (IS, 3H) = (seq*B, 3H)
    Tensor input_contig = input.contiguous();
    size_t all_ih_elems = seq_len * gate3_elems;
    id<MTLBuffer> buf_all_ih = [g_device newBufferWithLength:all_ih_elems * es
                                                     options:MTLResourceStorageModeShared];
    memset([buf_all_ih contents], 0, all_ih_elems * es);

    mps_gemm(get_buffer(input_contig), get_buffer(W_ih), buf_all_ih,
             seq_len * batch, input_size, 3 * hidden_size,
             1.0f, 0.0f, dtype, true);

    // GRU uses combined bias split differently from LSTM:
    // bias = bias_ih + bias_hh for r,z gates; n gate has separate treatment
    // For the dispatch interface, GRU forward receives a single combined bias.
    // We add it to the ih path.
    if (bias.numel() > 0) {
        // The combined bias has 3*hidden_size elements. Add to ih gates.
        add_bias_inplace(buf_all_ih, get_buffer(bias),
                         static_cast<uint32_t>(3 * hidden_size), all_ih_elems, dtype);
    }

    // Allocate output
    Tensor output({seq_len, batch, hidden_size}, dtype, input.device());
    Tensor h_n({batch, hidden_size}, dtype, input.device());

    Tensor h_cur = h0.contiguous();

    id<MTLBuffer> buf_gates_ih = [g_device newBufferWithLength:gate3_elems * es
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_gates_hh = [g_device newBufferWithLength:gate3_elems * es
                                                       options:MTLResourceStorageModeShared];

    for (int64_t t = 0; t < seq_len; ++t) {
        // Copy pre-computed ih gates for this timestep
        size_t offset = t * gate3_elems * es;
        memcpy([buf_gates_ih contents],
               static_cast<const uint8_t*>([buf_all_ih contents]) + offset,
               gate3_elems * es);

        // Compute hh gates: h_cur @ W_hh^T
        memset([buf_gates_hh contents], 0, gate3_elems * es);
        mps_gemm(get_buffer(h_cur), get_buffer(W_hh), buf_gates_hh,
                 batch, hidden_size, 3 * hidden_size,
                 1.0f, 0.0f, dtype, true);

        // Apply gate activations + hidden update
        Tensor h_next({batch, hidden_size}, dtype, input.device());

        dispatch_compute(
            rnn_shader_name("gru_gates_kernel", dtype),
            {buf_gates_ih, buf_gates_hh, get_buffer(h_cur), get_buffer(h_next)},
            static_cast<uint32_t>(hidden_size), 4,
            bh_elems);

        // Copy to output
        memcpy(static_cast<uint8_t*>(output.data_ptr()) + t * bh_elems * es,
               h_next.data_ptr(), bh_elems * es);

        h_cur = h_next;
    }

    memcpy(h_n.data_ptr(), h_cur.data_ptr(), bh_elems * es);

    return {output, h_n};
}

// ============================================================================
// GRU Multi-Layer Forward
// ============================================================================

std::vector<Tensor> mps_gru_multilayer_forward_kernel(
    const Tensor& input,
    const std::vector<Tensor>& W_ih_list,
    const std::vector<Tensor>& W_hh_list,
    const std::vector<Tensor>& bias_list,
    const Tensor& h0)
{
    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch = input_shape[1];
    int64_t hidden_size = h0.shape()[2];  // h0: (num_layers, batch, hidden)
    DType dtype = input.dtype();
    size_t es = elem_size(dtype);
    size_t bh = batch * hidden_size;

    Tensor h0_contig = h0.contiguous();
    Tensor layer_input = input;
    std::vector<Tensor> h_states;

    for (int64_t l = 0; l < num_layers; ++l) {
        Tensor h0_l({batch, hidden_size}, dtype, input.device());
        memcpy(h0_l.data_ptr(),
               static_cast<const uint8_t*>(h0_contig.data_ptr()) + l * bh * es,
               bh * es);

        Tensor bias_l = (!bias_list.empty() && bias_list[l].numel() > 0)
                        ? bias_list[l] : Tensor({0}, dtype, input.device());

        auto layer_out = mps_gru_forward_kernel(
            layer_input, W_ih_list[l], W_hh_list[l], bias_l, h0_l);

        h_states.push_back(layer_out[1]);
        layer_input = layer_out[0];
    }

    Tensor output({seq_len, batch, hidden_size}, dtype, input.device());
    memcpy(output.data_ptr(), layer_input.data_ptr(), seq_len * bh * es);

    Tensor h_n({num_layers, batch, hidden_size}, dtype, input.device());
    for (int64_t l = 0; l < num_layers; ++l) {
        memcpy(static_cast<uint8_t*>(h_n.data_ptr()) + l * bh * es,
               h_states[l].data_ptr(), bh * es);
    }

    return {output, h_n};
}

} // namespace tenzor::mps
