// Full-sequence LSTM / GRU / Elman-RNN forward and backward built on
// cuDNN v8's RNN API (`cudnnRNNForward`, `cudnnRNNBackwardData_v8`,
// `cudnnRNNBackwardWeights_v8`). The descriptor wrappers live in
// `tenzor/backend/cudnn_wrapper.hpp` (`RNNDescriptor`, `RNNDataDescriptor`,
// `DropoutDescriptor`).
//
// Weight packing strategy: cuDNN owns a single opaque weight space. The
// per-layer / per-direction weights handed in by `nn::LSTM`/`nn::GRU`/`nn::RNN`
// are copied into the packed buffer via `cudnnGetRNNWeightParams`, which
// hands back the per-matrix pointer and a `cudnnTensorDescriptor_t`
// describing the expected layout. The same mechanism is used to scatter
// the gradient buffer back into per-layer tensors.

#ifdef TENZOR_HAS_CUDNN

#include "tenzor/backend/cudnn_wrapper.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "cuda_error.hpp"
#include "cuda_stream_pool.hpp"
#include "kernels/cuda_launch_utils.cuh"  // tenzor::cuda::CudaDeviceGuard
#include "tenzor/backend/cuda_config.hpp"

#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor {
namespace cuda {

namespace {

cudnnDataType_t to_cudnn_dtype_rnn(DType dtype) {
    switch (dtype) {
        case DType::Float32:  return CUDNN_DATA_FLOAT;
        case DType::Float64:  return CUDNN_DATA_DOUBLE;
        case DType::Float16:  return CUDNN_DATA_HALF;
        case DType::BFloat16: return CUDNN_DATA_BFLOAT16;
        default:
            throw std::runtime_error(
                std::string("cuDNN RNN: unsupported dtype ") +
                std::string(dtype_name(dtype)));
    }
}

// TF32 for Float32 RNN matmuls, gated by the unified backend toggle
// (tenzor::cuda::matmul::allow_tf32() / TENZOR_DISABLE_TF32) that conv2d and
// matmul already honor. Default is ON, matching PyTorch
// (torch.backends.cudnn.allow_tf32=True): TF32 tensor-core math (~1.3x faster)
// with FP32 fallback. Float64 always uses full precision.
cudnnMathType_t rnn_math_type(DType dtype) {
    if (dtype == DType::Float32 && ::tenzor::cuda::matmul::allow_tf32()) {
        return CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
    }
    return CUDNN_DEFAULT_MATH;
}

cudnnDataType_t math_prec_for(DType dtype) {
    // For half / bfloat16 inputs we still accumulate in Float32 (matches
    // PyTorch and what the cuDNN sample code recommends). Float64 stays
    // Float64. Float32 stays Float32.
    if (dtype == DType::Float64) {
        return CUDNN_DATA_DOUBLE;
    }
    return CUDNN_DATA_FLOAT;
}

int gates_per_cell(cudnnRNNMode_t mode) {
    switch (mode) {
        case CUDNN_RNN_RELU: return 1;
        case CUDNN_RNN_TANH: return 1;
        case CUDNN_GRU:      return 3;
        case CUDNN_LSTM:     return 4;
    }
    return 0;
}

// Build the host-side per-sample sequence length array (all equal to
// `max_seq_length`; cuDNN requires it even when no truncation occurs) and
// allocate a matching device buffer for `cudnnRNNForward`'s
// `devSeqLengths`.
struct SeqLengthBuffers {
    std::vector<int> host;
    Tensor device_buf;  // int32 tensor on the same CUDA device
    const int* host_ptr() const { return host.data(); }
    int32_t* device_ptr() {
        return reinterpret_cast<int32_t*>(device_buf.data_ptr());
    }
};

SeqLengthBuffers make_seq_lengths(int batch_size, int max_seq_length,
                                  Device device, cudaStream_t stream) {
    SeqLengthBuffers buf;
    buf.host.assign(static_cast<size_t>(batch_size), max_seq_length);
    buf.device_buf = Tensor({batch_size}, DType::Int32, device);
    CUDA_CHECK(cudaMemcpyAsync(
        buf.device_buf.data_ptr(),
        buf.host.data(),
        sizeof(int) * static_cast<size_t>(batch_size),
        cudaMemcpyHostToDevice,
        stream));
    return buf;
}

// Number of pseudo-layers (layer * direction) cuDNN exposes.
int num_pseudo_layers(int num_layers, bool bidirectional) {
    return num_layers * (bidirectional ? 2 : 1);
}

// cuDNN expects 8 weight params per LSTM pseudo-layer (4 W_ih, 4 W_hh) when
// projection is unused, plus 1 W_hr when projection is in use. The bias is
// addressed via the same indices via the `bDesc`/`bAddr` pair.
//
// Index convention from the cuDNN docs:
//   LSTM: gate order = i, f, c (~g), o.
//     linLayerID 0..3 → W_ih for gates i, f, g, o
//     linLayerID 4..7 → W_hh for gates i, f, g, o
//     linLayerID 8    → W_hr (projection)
//   GRU:  gate order = r, z, n.
//     linLayerID 0..2 → W_ih for r, z, n
//     linLayerID 3..5 → W_hh for r, z, n
//   RNN:  linLayerID 0 → W_ih, 1 → W_hh.

int lin_layer_ih(cudnnRNNMode_t mode, int gate) {
    (void)mode;
    return gate;
}

int lin_layer_hh(cudnnRNNMode_t mode, int gate) {
    return gates_per_cell(mode) + gate;
}

int lin_layer_hr(cudnnRNNMode_t mode) {
    // Only meaningful for LSTM with projection.
    return 2 * gates_per_cell(mode);
}

// Compute the number of bytes a `cudnnTensorDescriptor_t` describes,
// taking dtype and the up-to-3 dims into account.
size_t descriptor_bytes(cudnnTensorDescriptor_t desc) {
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    constexpr int kMaxDims = 3;
    int nb_dims = 0;
    int dims[kMaxDims] = {0, 0, 0};
    int strides[kMaxDims] = {0, 0, 0};
    CUDNN_CHECK(cudnnGetTensorNdDescriptor(
        desc, kMaxDims, &dtype, &nb_dims, dims, strides));
    size_t elems = 1;
    for (int i = 0; i < nb_dims; ++i) {
        elems *= static_cast<size_t>(dims[i]);
    }
    size_t dtype_bytes = 0;
    switch (dtype) {
        case CUDNN_DATA_DOUBLE:   dtype_bytes = 8; break;
        case CUDNN_DATA_FLOAT:    dtype_bytes = 4; break;
        case CUDNN_DATA_HALF:     dtype_bytes = 2; break;
        case CUDNN_DATA_BFLOAT16: dtype_bytes = 2; break;
        default:
            throw std::runtime_error("cuDNN RNN: unexpected weight dtype");
    }
    return elems * dtype_bytes;
}

// Pack per-layer weights/biases into cuDNN's opaque weight space buffer.
// Per pseudo-layer, the loop walks every (linLayerID, weight-or-bias) slot
// and copies the corresponding tensor into the device pointer cuDNN hands
// back. Empty input tensors are written as zeros so callers can omit
// biases without segfaulting.
void pack_weights(cudnnRNNDescriptor_t rnn_desc,
                  size_t weight_space_size,
                  void* weight_space,
                  cudnnRNNMode_t mode,
                  int num_layers,
                  bool bidirectional,
                  const std::vector<Tensor>& W_ih,
                  const std::vector<Tensor>& W_hh,
                  const std::vector<Tensor>& b_ih,
                  const std::vector<Tensor>& b_hh,
                  const std::vector<Tensor>& W_hr,
                  bool has_projection,
                  cudaStream_t stream) {
    auto cudnn_handle = CuDNNHandle::get();
    const int pseudo_layers = num_pseudo_layers(num_layers, bidirectional);
    const int gates = gates_per_cell(mode);
    const bool needs_lstm_hr = (mode == CUDNN_LSTM) && has_projection;

    if (static_cast<int>(W_ih.size()) != pseudo_layers ||
        static_cast<int>(W_hh.size()) != pseudo_layers) {
        throw std::runtime_error(
            "cuDNN RNN: weight vector size mismatch (expected "
            "num_layers * num_directions entries)");
    }
    if (!b_ih.empty() && static_cast<int>(b_ih.size()) != pseudo_layers) {
        throw std::runtime_error("cuDNN RNN: bias_ih vector size mismatch");
    }
    if (!b_hh.empty() && static_cast<int>(b_hh.size()) != pseudo_layers) {
        throw std::runtime_error("cuDNN RNN: bias_hh vector size mismatch");
    }
    if (needs_lstm_hr && static_cast<int>(W_hr.size()) != pseudo_layers) {
        throw std::runtime_error("cuDNN RNN: W_hr vector size mismatch");
    }

    // Zero the entire weight buffer first so that gates with omitted biases
    // (or unused projection slots) are deterministic.
    CUDA_CHECK(cudaMemsetAsync(weight_space, 0, weight_space_size, stream));

    cudnnTensorDescriptor_t m_desc = nullptr;
    cudnnTensorDescriptor_t b_desc = nullptr;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc));

    auto copy_in = [&](void* dst, const Tensor& src, size_t expected_bytes) {
        if (src.numel() == 0) return;  // leave the zero memset in place
        const size_t src_bytes =
            static_cast<size_t>(src.numel()) * dtype_size(src.dtype());
        if (src_bytes != expected_bytes) {
            throw std::runtime_error(
                "cuDNN RNN: weight slot size mismatch (got " +
                std::to_string(src_bytes) + " bytes, cuDNN expected " +
                std::to_string(expected_bytes) + ")");
        }
        CUDA_CHECK(cudaMemcpyAsync(dst, src.data_ptr(), expected_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    };

    for (int pl = 0; pl < pseudo_layers; ++pl) {
        // Input-to-hidden weights: gate-major in cuDNN. Per-gate slot, but
        // PyTorch / Tenzor passes the gate-stacked W_ih (rows = gates *
        // hidden). Copy per gate so we don't depend on a contiguous stride.
        //
        // S.8: lifetime contract for the cudaMemcpyAsync source buffers.
        // The four references below alias entries of W_ih / W_hh / b_ih /
        // b_hh, which are `const std::vector<Tensor>&` parameters owned by
        // the caller. All async D2D copies enqueued in this loop are placed
        // on `stream`; cuDNN's RNN forward / backward call further down the
        // same stream observes them stream-ordered, and the caller keeps
        // the source vectors alive at least until after `cudnnRNNForward`
        // returns (see pack_weights() call site in rnn_forward()). The
        // ternary on b_ih / b_hh emptiness binds to a temporary `Tensor{}`
        // whose lifetime is the loop iteration; that branch has
        // `numel() == 0`, so no memcpy is enqueued from a dangling pointer.
        const Tensor& wih = W_ih[pl];
        const Tensor& whh = W_hh[pl];
        const Tensor& bih = b_ih.empty() ? Tensor{} : b_ih[pl];
        const Tensor& bhh = b_hh.empty() ? Tensor{} : b_hh[pl];

        if (wih.numel() == 0 || whh.numel() == 0) {
            throw std::runtime_error(
                "cuDNN RNN: input-to-hidden and hidden-to-hidden weights are required");
        }
        if (wih.ndim() != 2 || whh.ndim() != 2) {
            throw std::runtime_error("cuDNN RNN: weight tensors must be 2-D");
        }
        const int64_t per_gate_in = wih.shape()[1];
        const int64_t per_gate_hh = whh.shape()[1];
        const size_t elem_bytes = dtype_size(wih.dtype());

        for (int g = 0; g < gates; ++g) {
            void* m_addr = nullptr;
            void* b_addr = nullptr;
            // W_ih slot
            CUDNN_CHECK(cudnnGetRNNWeightParams(
                cudnn_handle, rnn_desc, pl, weight_space_size, weight_space,
                lin_layer_ih(mode, g), m_desc, &m_addr, b_desc, &b_addr));
            const size_t expected_wih = descriptor_bytes(m_desc);
            const size_t row_bytes_in = static_cast<size_t>(per_gate_in) * elem_bytes;
            const char* wih_row_ptr = static_cast<const char*>(wih.data_ptr())
                + static_cast<size_t>(g) *
                  static_cast<size_t>(wih.shape()[0] / gates) * row_bytes_in;
            (void)wih_row_ptr;
            // For simplicity copy the full per-gate slab in one shot. This
            // matches cuDNN's expected per-gate row-major layout.
            const size_t per_gate_rows = static_cast<size_t>(wih.shape()[0]) /
                                         static_cast<size_t>(gates);
            const size_t per_gate_bytes = per_gate_rows * row_bytes_in;
            if (per_gate_bytes != expected_wih) {
                throw std::runtime_error(
                    "cuDNN RNN: gate slab size mismatch on input-to-hidden weights "
                    "(got " + std::to_string(per_gate_bytes) + " bytes, cuDNN expected " +
                    std::to_string(expected_wih) + " bytes)");
            }
            CUDA_CHECK(cudaMemcpyAsync(
                m_addr,
                static_cast<const char*>(wih.data_ptr()) +
                    static_cast<size_t>(g) * per_gate_bytes,
                per_gate_bytes, cudaMemcpyDeviceToDevice, stream));

            // Bias slot (input bias)
            if (bih.numel() > 0 && b_addr != nullptr) {
                const size_t expected_b = descriptor_bytes(b_desc);
                const size_t per_gate_b = static_cast<size_t>(bih.numel()) /
                                          static_cast<size_t>(gates) * elem_bytes;
                if (per_gate_b != expected_b) {
                    throw std::runtime_error(
                        "cuDNN RNN: gate slab size mismatch on input bias");
                }
                CUDA_CHECK(cudaMemcpyAsync(
                    b_addr,
                    static_cast<const char*>(bih.data_ptr()) +
                        static_cast<size_t>(g) * per_gate_b,
                    per_gate_b, cudaMemcpyDeviceToDevice, stream));
            }

            // W_hh slot
            CUDNN_CHECK(cudnnGetRNNWeightParams(
                cudnn_handle, rnn_desc, pl, weight_space_size, weight_space,
                lin_layer_hh(mode, g), m_desc, &m_addr, b_desc, &b_addr));
            const size_t expected_whh = descriptor_bytes(m_desc);
            const size_t row_bytes_hh = static_cast<size_t>(per_gate_hh) * elem_bytes;
            const size_t per_gate_rows_hh =
                static_cast<size_t>(whh.shape()[0]) / static_cast<size_t>(gates);
            const size_t per_gate_bytes_hh = per_gate_rows_hh * row_bytes_hh;
            if (per_gate_bytes_hh != expected_whh) {
                throw std::runtime_error(
                    "cuDNN RNN: gate slab size mismatch on hidden-to-hidden weights");
            }
            CUDA_CHECK(cudaMemcpyAsync(
                m_addr,
                static_cast<const char*>(whh.data_ptr()) +
                    static_cast<size_t>(g) * per_gate_bytes_hh,
                per_gate_bytes_hh, cudaMemcpyDeviceToDevice, stream));

            // Bias slot (recurrent bias)
            if (bhh.numel() > 0 && b_addr != nullptr) {
                const size_t expected_b = descriptor_bytes(b_desc);
                const size_t per_gate_b = static_cast<size_t>(bhh.numel()) /
                                          static_cast<size_t>(gates) * elem_bytes;
                if (per_gate_b != expected_b) {
                    throw std::runtime_error(
                        "cuDNN RNN: gate slab size mismatch on recurrent bias");
                }
                CUDA_CHECK(cudaMemcpyAsync(
                    b_addr,
                    static_cast<const char*>(bhh.data_ptr()) +
                        static_cast<size_t>(g) * per_gate_b,
                    per_gate_b, cudaMemcpyDeviceToDevice, stream));
            }
        }

        if (needs_lstm_hr) {
            void* m_addr = nullptr;
            void* b_addr = nullptr;
            CUDNN_CHECK(cudnnGetRNNWeightParams(
                cudnn_handle, rnn_desc, pl, weight_space_size, weight_space,
                lin_layer_hr(mode), m_desc, &m_addr, b_desc, &b_addr));
            copy_in(m_addr, W_hr[pl], descriptor_bytes(m_desc));
        }
    }

    cudnnDestroyTensorDescriptor(m_desc);
    cudnnDestroyTensorDescriptor(b_desc);
}

// Unpack a packed weight-gradient buffer back into per-layer tensors with
// the same layout as the forward call's inputs. The output vectors are
// sized to `num_layers * num_directions`.
void unpack_grads(cudnnRNNDescriptor_t rnn_desc,
                  size_t weight_space_size,
                  const void* weight_space_grad,
                  cudnnRNNMode_t mode,
                  int num_layers,
                  bool bidirectional,
                  int64_t input_size,
                  int64_t hidden_size,
                  int64_t proj_size,
                  DType dtype,
                  Device device,
                  std::vector<Tensor>& out_W_ih,
                  std::vector<Tensor>& out_W_hh,
                  std::vector<Tensor>& out_b_ih,
                  std::vector<Tensor>& out_b_hh,
                  std::vector<Tensor>& out_W_hr,
                  cudaStream_t stream) {
    const int pseudo_layers = num_pseudo_layers(num_layers, bidirectional);
    const int gates = gates_per_cell(mode);
    const bool has_proj = (mode == CUDNN_LSTM) && (proj_size > 0);
    const int64_t out_dim = (proj_size > 0 ? proj_size : hidden_size);

    auto cudnn_handle = CuDNNHandle::get();
    out_W_ih.assign(pseudo_layers, Tensor{});
    out_W_hh.assign(pseudo_layers, Tensor{});
    out_b_ih.assign(pseudo_layers, Tensor{});
    out_b_hh.assign(pseudo_layers, Tensor{});
    out_W_hr.assign(has_proj ? pseudo_layers : 0, Tensor{});

    cudnnTensorDescriptor_t m_desc = nullptr;
    cudnnTensorDescriptor_t b_desc = nullptr;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc));

    void* weight_space_mutable =
        const_cast<void*>(weight_space_grad);  // cuDNN's query takes void*

    const size_t elem_bytes = dtype_size(dtype);

    for (int pl = 0; pl < pseudo_layers; ++pl) {
        // For non-first pseudo layers, the input dim seen by cuDNN is
        // hidden * num_directions (the previous layer's output width), or
        // proj_size * num_directions when projection is in use.
        const int layer_idx = bidirectional ? pl / 2 : pl;
        const int64_t layer_input_size = (layer_idx == 0)
            ? input_size
            : (bidirectional ? 2 * out_dim : out_dim);
        // Allocate the destination tensors.
        out_W_ih[pl] = Tensor({gates * hidden_size, layer_input_size}, dtype, device);
        out_W_hh[pl] = Tensor({gates * hidden_size, out_dim}, dtype, device);
        out_b_ih[pl] = Tensor({gates * hidden_size}, dtype, device);
        out_b_hh[pl] = Tensor({gates * hidden_size}, dtype, device);
        if (has_proj) {
            out_W_hr[pl] = Tensor({proj_size, hidden_size}, dtype, device);
        }

        const size_t row_bytes_in = static_cast<size_t>(layer_input_size) * elem_bytes;
        const size_t row_bytes_hh = static_cast<size_t>(out_dim) * elem_bytes;

        for (int g = 0; g < gates; ++g) {
            void* m_addr = nullptr;
            void* b_addr = nullptr;
            // W_ih
            CUDNN_CHECK(cudnnGetRNNWeightParams(
                cudnn_handle, rnn_desc, pl, weight_space_size,
                weight_space_mutable, lin_layer_ih(mode, g),
                m_desc, &m_addr, b_desc, &b_addr));
            const size_t per_gate_bytes_in =
                static_cast<size_t>(hidden_size) * row_bytes_in;
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<char*>(out_W_ih[pl].data_ptr()) +
                    static_cast<size_t>(g) * per_gate_bytes_in,
                m_addr, per_gate_bytes_in, cudaMemcpyDeviceToDevice, stream));
            if (b_addr != nullptr) {
                const size_t per_gate_b =
                    static_cast<size_t>(hidden_size) * elem_bytes;
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<char*>(out_b_ih[pl].data_ptr()) +
                        static_cast<size_t>(g) * per_gate_b,
                    b_addr, per_gate_b, cudaMemcpyDeviceToDevice, stream));
            } else {
                CUDA_CHECK(cudaMemsetAsync(
                    out_b_ih[pl].data_ptr(), 0,
                    static_cast<size_t>(out_b_ih[pl].numel()) * elem_bytes,
                    stream));
            }

            // W_hh
            CUDNN_CHECK(cudnnGetRNNWeightParams(
                cudnn_handle, rnn_desc, pl, weight_space_size,
                weight_space_mutable, lin_layer_hh(mode, g),
                m_desc, &m_addr, b_desc, &b_addr));
            const size_t per_gate_bytes_hh =
                static_cast<size_t>(hidden_size) * row_bytes_hh;
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<char*>(out_W_hh[pl].data_ptr()) +
                    static_cast<size_t>(g) * per_gate_bytes_hh,
                m_addr, per_gate_bytes_hh, cudaMemcpyDeviceToDevice, stream));
            if (b_addr != nullptr) {
                const size_t per_gate_b =
                    static_cast<size_t>(hidden_size) * elem_bytes;
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<char*>(out_b_hh[pl].data_ptr()) +
                        static_cast<size_t>(g) * per_gate_b,
                    b_addr, per_gate_b, cudaMemcpyDeviceToDevice, stream));
            } else {
                CUDA_CHECK(cudaMemsetAsync(
                    out_b_hh[pl].data_ptr(), 0,
                    static_cast<size_t>(out_b_hh[pl].numel()) * elem_bytes,
                    stream));
            }
        }

        if (has_proj) {
            void* m_addr = nullptr;
            void* b_addr = nullptr;
            CUDNN_CHECK(cudnnGetRNNWeightParams(
                cudnn_handle, rnn_desc, pl, weight_space_size,
                weight_space_mutable, lin_layer_hr(mode),
                m_desc, &m_addr, b_desc, &b_addr));
            const size_t bytes =
                static_cast<size_t>(out_W_hr[pl].numel()) * elem_bytes;
            CUDA_CHECK(cudaMemcpyAsync(
                out_W_hr[pl].data_ptr(), m_addr, bytes,
                cudaMemcpyDeviceToDevice, stream));
        }
    }

    cudnnDestroyTensorDescriptor(m_desc);
    cudnnDestroyTensorDescriptor(b_desc);
}

// Shared driver for forward: handles dropout descriptor + workspace +
// reserve space allocation + the `cudnnRNNForward` invocation. The result
// `weight_space` tensor is owned by the caller; pass it on to the
// backward function.
struct ForwardResult {
    Tensor output;
    Tensor hy;
    Tensor cy;            // empty for non-LSTM
    Tensor reserve_space; // empty when fwd_mode == INFERENCE
    Tensor weight_space;
};

ForwardResult run_forward(
    const Tensor& input,
    const Tensor& hx,
    const Tensor& cx,  // may be empty for non-LSTM
    const std::vector<Tensor>& W_ih,
    const std::vector<Tensor>& W_hh,
    const std::vector<Tensor>& b_ih,
    const std::vector<Tensor>& b_hh,
    const std::vector<Tensor>& W_hr,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudnnRNNMode_t cell_mode,
    cudnnForwardMode_t fwd_mode,
    cudaStream_t stream) {

    if (input.ndim() != 3) {
        throw std::runtime_error("cuDNN RNN: input must be 3-D (seq, batch, feature)");
    }
    const int64_t seq_len = input.shape()[0];
    const int64_t batch = input.shape()[1];
    const int64_t input_size = input.shape()[2];
    const int64_t num_directions = bidirectional ? 2 : 1;
    const int64_t out_dim = (proj_size > 0 ? proj_size : hidden_size);
    const Device device = input.device();
    const DType dtype = input.dtype();

    // Make the tensor's device current so the device-keyed cuDNN handle and
    // workspace are fetched/allocated on the correct GPU. Restored on exit.
    CudaDeviceGuard dev_guard(device.index);
    auto handle = CuDNNHandle::get();
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    // Inter-layer dropout descriptor. cuDNN requires this even for a
    // single-layer model; we create one with the requested probability and
    // a deterministic seed (the layers above are responsible for
    // disabling/enabling dropout via `is_training()`).
    DropoutDescriptor dropout_desc;
    size_t dropout_state_size = 0;
    CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &dropout_state_size));
    Tensor dropout_state;
    void* dropout_state_ptr = nullptr;
    if (dropout_state_size > 0) {
        dropout_state = Tensor({static_cast<int64_t>(dropout_state_size)},
                               DType::UInt8, device);
        dropout_state_ptr = dropout_state.data_ptr();
    }
    dropout_desc.set(handle, dropout, dropout_state_ptr, dropout_state_size,
                     /*seed=*/0ULL);

    RNNDescriptor rnn_desc;
    const auto cudnn_dtype = to_cudnn_dtype_rnn(dtype);
    const auto math_prec = math_prec_for(dtype);
    if (cell_mode == CUDNN_LSTM) {
        rnn_desc.set_lstm(handle,
                          static_cast<int>(input_size),
                          static_cast<int>(hidden_size),
                          static_cast<int>(proj_size),
                          static_cast<int>(num_layers),
                          bidirectional,
                          dropout_desc.get(),
                          cudnn_dtype, math_prec, rnn_math_type(dtype));
    } else if (cell_mode == CUDNN_GRU) {
        rnn_desc.set_gru(handle,
                        static_cast<int>(input_size),
                        static_cast<int>(hidden_size),
                        static_cast<int>(num_layers),
                        bidirectional,
                        dropout_desc.get(),
                        cudnn_dtype, math_prec, rnn_math_type(dtype));
    } else {
        rnn_desc.set_rnn(handle,
                        static_cast<int>(input_size),
                        static_cast<int>(hidden_size),
                        static_cast<int>(num_layers),
                        bidirectional,
                        dropout_desc.get(),
                        cudnn_dtype, cell_mode, math_prec, rnn_math_type(dtype));
    }

    // Sequence layout: time-major unpacked. All samples have the same
    // length (caller is responsible for padding).
    auto seq_lens = make_seq_lengths(static_cast<int>(batch),
                                     static_cast<int>(seq_len), device, stream);

    RNNDataDescriptor x_desc;
    x_desc.set_seq_major(cudnn_dtype,
                         static_cast<int>(seq_len),
                         static_cast<int>(batch),
                         static_cast<int>(input_size),
                         seq_lens.host_ptr(), /*padding_fill=*/nullptr);

    RNNDataDescriptor y_desc;
    y_desc.set_seq_major(cudnn_dtype,
                         static_cast<int>(seq_len),
                         static_cast<int>(batch),
                         static_cast<int>(out_dim * num_directions),
                         seq_lens.host_ptr(), /*padding_fill=*/nullptr);

    // Hidden / cell tensor descriptors.
    // Shape per cuDNN: [num_layers * num_directions, batch, hidden_or_proj]
    TensorDescriptor h_desc;
    {
        int dims[3] = {
            static_cast<int>(num_layers * num_directions),
            static_cast<int>(batch),
            static_cast<int>(out_dim)};
        int strides[3] = {
            dims[1] * dims[2],
            dims[2],
            1};
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            h_desc.get(), cudnn_dtype, 3, dims, strides));
    }
    TensorDescriptor c_desc;
    {
        int dims[3] = {
            static_cast<int>(num_layers * num_directions),
            static_cast<int>(batch),
            static_cast<int>(hidden_size)};
        int strides[3] = {
            dims[1] * dims[2],
            dims[2],
            1};
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            c_desc.get(), cudnn_dtype, 3, dims, strides));
    }

    // Allocate output / states.
    Tensor output({seq_len, batch, out_dim * num_directions}, dtype, device);
    Tensor hy({num_layers * num_directions, batch, out_dim}, dtype, device);
    Tensor cy;
    if (cell_mode == CUDNN_LSTM) {
        cy = Tensor({num_layers * num_directions, batch, hidden_size}, dtype, device);
    }

    // Resolve initial hidden / cell. Empty -> zeros.
    Tensor hx_resolved = hx;
    if (!hx_resolved.numel()) {
        hx_resolved = Tensor({num_layers * num_directions, batch, out_dim}, dtype, device);
        CUDA_CHECK(cudaMemsetAsync(
            hx_resolved.data_ptr(), 0,
            static_cast<size_t>(hx_resolved.numel()) * dtype_size(dtype),
            stream));
    }
    Tensor cx_resolved = cx;
    if (cell_mode == CUDNN_LSTM && !cx_resolved.numel()) {
        cx_resolved = Tensor({num_layers * num_directions, batch, hidden_size}, dtype, device);
        CUDA_CHECK(cudaMemsetAsync(
            cx_resolved.data_ptr(), 0,
            static_cast<size_t>(cx_resolved.numel()) * dtype_size(dtype),
            stream));
    }

    // Allocate the weight space, pack inputs into it.
    size_t weight_space_size = 0;
    CUDNN_CHECK(cudnnGetRNNWeightSpaceSize(handle, rnn_desc.get(),
                                            &weight_space_size));
    Tensor weight_space({static_cast<int64_t>(weight_space_size)},
                        DType::UInt8, device);
    pack_weights(rnn_desc.get(), weight_space_size, weight_space.data_ptr(),
                 cell_mode, static_cast<int>(num_layers), bidirectional,
                 W_ih, W_hh, b_ih, b_hh, W_hr,
                 /*has_projection=*/proj_size > 0, stream);

    // Workspace + reserve space sizes.
    size_t work_space_size = 0;
    size_t reserve_space_size = 0;
    CUDNN_CHECK(cudnnGetRNNTempSpaceSizes(
        handle, rnn_desc.get(), fwd_mode, x_desc.get(),
        &work_space_size, &reserve_space_size));

    Tensor work_space;
    if (work_space_size > 0) {
        work_space = Tensor({static_cast<int64_t>(work_space_size)},
                            DType::UInt8, device);
    }
    Tensor reserve_space;
    if (fwd_mode == CUDNN_FWD_MODE_TRAINING && reserve_space_size > 0) {
        reserve_space = Tensor({static_cast<int64_t>(reserve_space_size)},
                               DType::UInt8, device);
    }

    CUDNN_CHECK(cudnnRNNForward(
        handle,
        rnn_desc.get(),
        fwd_mode,
        seq_lens.device_ptr(),
        x_desc.get(), input.data_ptr(),
        y_desc.get(), output.data_ptr(),
        h_desc.get(), hx_resolved.data_ptr(), hy.data_ptr(),
        (cell_mode == CUDNN_LSTM) ? c_desc.get() : nullptr,
        (cell_mode == CUDNN_LSTM) ? cx_resolved.data_ptr() : nullptr,
        (cell_mode == CUDNN_LSTM) ? cy.data_ptr() : nullptr,
        weight_space_size, weight_space.data_ptr(),
        work_space_size, work_space.numel() ? work_space.data_ptr() : nullptr,
        reserve_space_size,
        reserve_space.numel() ? reserve_space.data_ptr() : nullptr));

    ForwardResult res;
    res.output = output;
    res.hy = hy;
    res.cy = cy;
    res.reserve_space = reserve_space;
    res.weight_space = weight_space;
    return res;
}

// Shared driver for backward. Mirrors `run_forward` for descriptor setup,
// then calls `cudnnRNNBackwardData_v8` followed by
// `cudnnRNNBackwardWeights_v8`.
struct BackwardResult {
    Tensor grad_input;
    Tensor grad_hx;
    Tensor grad_cx;  // empty for non-LSTM
    Tensor grad_weight_space;
};

BackwardResult run_backward(
    const Tensor& grad_output,
    const Tensor& grad_hy,
    const Tensor& grad_cy,
    const Tensor& input,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& output,
    const Tensor& weight_space,
    const Tensor& reserve_space,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudnnRNNMode_t cell_mode,
    cudaStream_t stream) {

    if (input.ndim() != 3) {
        throw std::runtime_error("cuDNN RNN backward: input must be 3-D");
    }
    const int64_t seq_len = input.shape()[0];
    const int64_t batch = input.shape()[1];
    const int64_t input_size = input.shape()[2];
    const int64_t num_directions = bidirectional ? 2 : 1;
    const int64_t out_dim = (proj_size > 0 ? proj_size : hidden_size);
    const Device device = input.device();
    const DType dtype = input.dtype();

    // Make the tensor's device current so the device-keyed cuDNN handle and
    // workspace are fetched/allocated on the correct GPU. Restored on exit.
    CudaDeviceGuard dev_guard(device.index);
    auto handle = CuDNNHandle::get();
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    // Re-create dropout / RNN descriptors with the same settings as the
    // forward call. The reserve_space carries the random state across.
    DropoutDescriptor dropout_desc;
    size_t dropout_state_size = 0;
    CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &dropout_state_size));
    Tensor dropout_state;
    void* dropout_state_ptr = nullptr;
    if (dropout_state_size > 0) {
        dropout_state = Tensor({static_cast<int64_t>(dropout_state_size)},
                               DType::UInt8, device);
        dropout_state_ptr = dropout_state.data_ptr();
    }
    dropout_desc.set(handle, dropout, dropout_state_ptr, dropout_state_size,
                     /*seed=*/0ULL);

    RNNDescriptor rnn_desc;
    const auto cudnn_dtype = to_cudnn_dtype_rnn(dtype);
    const auto math_prec = math_prec_for(dtype);
    if (cell_mode == CUDNN_LSTM) {
        rnn_desc.set_lstm(handle,
                          static_cast<int>(input_size),
                          static_cast<int>(hidden_size),
                          static_cast<int>(proj_size),
                          static_cast<int>(num_layers),
                          bidirectional,
                          dropout_desc.get(),
                          cudnn_dtype, math_prec, rnn_math_type(dtype));
    } else if (cell_mode == CUDNN_GRU) {
        rnn_desc.set_gru(handle,
                        static_cast<int>(input_size),
                        static_cast<int>(hidden_size),
                        static_cast<int>(num_layers),
                        bidirectional,
                        dropout_desc.get(),
                        cudnn_dtype, math_prec, rnn_math_type(dtype));
    } else {
        rnn_desc.set_rnn(handle,
                        static_cast<int>(input_size),
                        static_cast<int>(hidden_size),
                        static_cast<int>(num_layers),
                        bidirectional,
                        dropout_desc.get(),
                        cudnn_dtype, cell_mode, math_prec, rnn_math_type(dtype));
    }

    auto seq_lens = make_seq_lengths(static_cast<int>(batch),
                                     static_cast<int>(seq_len), device, stream);

    RNNDataDescriptor x_desc;
    x_desc.set_seq_major(cudnn_dtype,
                         static_cast<int>(seq_len),
                         static_cast<int>(batch),
                         static_cast<int>(input_size),
                         seq_lens.host_ptr(), nullptr);
    RNNDataDescriptor y_desc;
    y_desc.set_seq_major(cudnn_dtype,
                         static_cast<int>(seq_len),
                         static_cast<int>(batch),
                         static_cast<int>(out_dim * num_directions),
                         seq_lens.host_ptr(), nullptr);

    TensorDescriptor h_desc;
    {
        int dims[3] = {
            static_cast<int>(num_layers * num_directions),
            static_cast<int>(batch),
            static_cast<int>(out_dim)};
        int strides[3] = {dims[1] * dims[2], dims[2], 1};
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            h_desc.get(), cudnn_dtype, 3, dims, strides));
    }
    TensorDescriptor c_desc;
    {
        int dims[3] = {
            static_cast<int>(num_layers * num_directions),
            static_cast<int>(batch),
            static_cast<int>(hidden_size)};
        int strides[3] = {dims[1] * dims[2], dims[2], 1};
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            c_desc.get(), cudnn_dtype, 3, dims, strides));
    }

    Tensor hx_resolved = hx;
    if (!hx_resolved.numel()) {
        hx_resolved = Tensor({num_layers * num_directions, batch, out_dim}, dtype, device);
        CUDA_CHECK(cudaMemsetAsync(hx_resolved.data_ptr(), 0,
            static_cast<size_t>(hx_resolved.numel()) * dtype_size(dtype),
            stream));
    }
    Tensor cx_resolved = cx;
    if (cell_mode == CUDNN_LSTM && !cx_resolved.numel()) {
        cx_resolved = Tensor({num_layers * num_directions, batch, hidden_size}, dtype, device);
        CUDA_CHECK(cudaMemsetAsync(cx_resolved.data_ptr(), 0,
            static_cast<size_t>(cx_resolved.numel()) * dtype_size(dtype),
            stream));
    }

    // Output buffers.
    Tensor grad_input({seq_len, batch, input_size}, dtype, device);
    Tensor grad_hx({num_layers * num_directions, batch, out_dim}, dtype, device);
    Tensor grad_cx;
    if (cell_mode == CUDNN_LSTM) {
        grad_cx = Tensor({num_layers * num_directions, batch, hidden_size}, dtype, device);
    }

    // Resolve grad_hy / grad_cy. cuDNN treats a NULL pointer as zero.
    const void* grad_hy_ptr = grad_hy.numel() ? grad_hy.data_ptr() : nullptr;
    const void* grad_cy_ptr =
        (cell_mode == CUDNN_LSTM && grad_cy.numel()) ? grad_cy.data_ptr() : nullptr;

    // Workspace sizes (same query as forward, but TRAINING mode since
    // backward requires the training-size scratch).
    size_t work_space_size = 0;
    size_t reserve_space_size_query = 0;
    CUDNN_CHECK(cudnnGetRNNTempSpaceSizes(
        handle, rnn_desc.get(), CUDNN_FWD_MODE_TRAINING, x_desc.get(),
        &work_space_size, &reserve_space_size_query));
    Tensor work_space;
    if (work_space_size > 0) {
        work_space = Tensor({static_cast<int64_t>(work_space_size)},
                            DType::UInt8, device);
    }

    // Data gradients first.
    CUDNN_CHECK(cudnnRNNBackwardData_v8(
        handle,
        rnn_desc.get(),
        seq_lens.device_ptr(),
        y_desc.get(), output.data_ptr(),
        grad_output.data_ptr(),
        x_desc.get(), grad_input.data_ptr(),
        h_desc.get(), hx_resolved.data_ptr(), grad_hy_ptr,
        grad_hx.data_ptr(),
        (cell_mode == CUDNN_LSTM) ? c_desc.get() : nullptr,
        (cell_mode == CUDNN_LSTM) ? cx_resolved.data_ptr() : nullptr,
        (cell_mode == CUDNN_LSTM) ? grad_cy_ptr : nullptr,
        (cell_mode == CUDNN_LSTM) ? grad_cx.data_ptr() : nullptr,
        static_cast<size_t>(weight_space.numel()),
        weight_space.data_ptr(),
        work_space_size, work_space.numel() ? work_space.data_ptr() : nullptr,
        static_cast<size_t>(reserve_space.numel()),
        // cuDNN scribbles the reserve space (random-state advance etc.)
        // even though the caller treats it as opaque; const_cast is
        // unavoidable here because Tensor::data_ptr() const returns const.
        const_cast<void*>(reserve_space.data_ptr())));

    // Weight gradients next. cudnnRNNBackwardWeights_v8 only supports
    // CUDNN_WGRAD_MODE_ADD (accumulate) — WGRAD_MODE_SET returns
    // CUDNN_STATUS_NOT_SUPPORTED — so we zero the buffer first, then accumulate.
    Tensor grad_weight_space({weight_space.numel()},
                             DType::UInt8, device);
    CUDA_CHECK(cudaMemsetAsync(grad_weight_space.data_ptr(), 0,
                               static_cast<size_t>(grad_weight_space.numel()), stream));
    CUDNN_CHECK(cudnnRNNBackwardWeights_v8(
        handle,
        rnn_desc.get(),
        CUDNN_WGRAD_MODE_ADD,
        seq_lens.device_ptr(),
        x_desc.get(), input.data_ptr(),
        h_desc.get(), hx_resolved.data_ptr(),
        y_desc.get(), output.data_ptr(),
        static_cast<size_t>(grad_weight_space.numel()),
        grad_weight_space.data_ptr(),
        work_space_size, work_space.numel() ? work_space.data_ptr() : nullptr,
        static_cast<size_t>(reserve_space.numel()),
        const_cast<void*>(reserve_space.data_ptr())));

    BackwardResult res;
    res.grad_input = grad_input;
    res.grad_hx = grad_hx;
    res.grad_cx = grad_cx;
    res.grad_weight_space = grad_weight_space;
    return res;
}

}  // anonymous namespace

// ============================================================================
// Public LSTM API
// ============================================================================

auto cudnn_lstm_forward(
    const Tensor& input,
    const Tensor& hx,
    const Tensor& cx,
    const std::vector<Tensor>& weights_ih,
    const std::vector<Tensor>& weights_hh,
    const std::vector<Tensor>& biases_ih,
    const std::vector<Tensor>& biases_hh,
    const std::vector<Tensor>& weights_hr,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudnnForwardMode_t fwd_mode,
    cudaStream_t stream) -> CudnnLSTMOutputs {

    auto raw = run_forward(input, hx, cx,
                           weights_ih, weights_hh, biases_ih, biases_hh,
                           weights_hr,
                           hidden_size, proj_size, num_layers,
                           bidirectional, dropout,
                           CUDNN_LSTM, fwd_mode, stream);
    CudnnLSTMOutputs out;
    out.output = raw.output;
    out.hy = raw.hy;
    out.cy = raw.cy;
    out.reserve_space = raw.reserve_space;
    out.weight_space = raw.weight_space;
    return out;
}

// Tensor-only inference wrapper for the registered single-layer LSTMForward
// kernel (see rnn_sequence.cu). Lives here so the cudnnForwardMode_t argument
// is resolved in the TU that includes <cudnn.h> first. Single layer, one
// direction, no projection. Returns {output (seq,batch,hidden),
// hy (batch,hidden), cy (batch,hidden)} to match the per-timestep kernel.
std::vector<Tensor> lstm_forward_cudnn_inference(
    const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
    const Tensor& bias_ih, const Tensor& bias_hh,
    const Tensor& h0, const Tensor& c0) {
    const int64_t batch = h0.shape()[0];
    const int64_t hidden = h0.shape()[1];

    auto guard = CUDAStreamPool::instance().acquire_guard(input.device().index);
    cudaStream_t stream = guard.get();

    Tensor hx = h0.contiguous().reshape({1, batch, hidden});
    Tensor cx = c0.contiguous().reshape({1, batch, hidden});
    Tensor b_ih = bias_ih.numel() > 0 ? bias_ih : Tensor{};
    Tensor b_hh = bias_hh.numel() > 0 ? bias_hh : Tensor{};

    auto out = cudnn_lstm_forward(
        input.contiguous(), hx, cx,
        {W_ih}, {W_hh}, {b_ih}, {b_hh}, /*weights_hr=*/{},
        hidden, /*proj_size=*/0, /*num_layers=*/1,
        /*bidirectional=*/false, /*dropout=*/0.0f,
        CUDNN_FWD_MODE_INFERENCE, stream);

    return { out.output,
             out.hy.reshape({batch, hidden}),
             out.cy.reshape({batch, hidden}) };
}

// GRU inference via cuDNN. cuDNN's CUDNN_GRU with CUDNN_RNN_DOUBLE_BIAS uses
// PyTorch's exact formula (n = tanh(W_in x + b_in + r*(W_hn h + b_hn))), so it
// matches the autograd cell-loop. Returns {output (seq,batch,hidden),
// hy (batch,hidden)}.
std::vector<Tensor> gru_forward_cudnn_inference(
    const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
    const Tensor& bias_ih, const Tensor& h0, const Tensor& bias_hh) {
    const int64_t batch = h0.shape()[0];
    const int64_t hidden = h0.shape()[1];

    auto guard = CUDAStreamPool::instance().acquire_guard(input.device().index);
    cudaStream_t stream = guard.get();

    Tensor hx = h0.contiguous().reshape({1, batch, hidden});
    Tensor b_ih = bias_ih.numel() > 0 ? bias_ih : Tensor{};
    Tensor b_hh = bias_hh.numel() > 0 ? bias_hh : Tensor{};

    auto out = cudnn_gru_forward(
        input.contiguous(), hx,
        {W_ih}, {W_hh}, {b_ih}, {b_hh},
        hidden, /*num_layers=*/1, /*bidirectional=*/false, /*dropout=*/0.0f,
        CUDNN_FWD_MODE_INFERENCE, stream);

    return { out.output, out.hy.reshape({batch, hidden}) };
}

// Multi-layer (stacked, unidirectional) LSTM inference in a single fused cuDNN
// call, replacing the previous per-layer loop of separate cuDNN calls. Each
// bias_list[l] is the combined 8*hidden bias (4*hidden bias_ih ++ 4*hidden
// bias_hh) — split exactly as the per-layer path does. h0/c0 are
// (num_layers, batch, hidden), which already matches cuDNN's hx/cx layout.
std::vector<Tensor> lstm_multi_layer_forward_cudnn_inference(
    const Tensor& input,
    const std::vector<Tensor>& W_ih_list,
    const std::vector<Tensor>& W_hh_list,
    const std::vector<Tensor>& bias_list,
    const Tensor& h0, const Tensor& c0) {
    const int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    const int64_t hidden = h0.shape()[2];

    auto guard = CUDAStreamPool::instance().acquire_guard(input.device().index);
    cudaStream_t stream = guard.get();

    std::vector<Tensor> biases_ih, biases_hh;
    biases_ih.reserve(num_layers);
    biases_hh.reserve(num_layers);
    for (int64_t l = 0; l < num_layers; ++l) {
        if (bias_list[l].numel() > 0) {
            const int64_t half = bias_list[l].numel() / 2;
            biases_ih.push_back(bias_list[l].slice(0, 0, half).contiguous());
            biases_hh.push_back(bias_list[l].slice(0, half, 2 * half).contiguous());
        } else {
            biases_ih.push_back(Tensor{});
            biases_hh.push_back(Tensor{});
        }
    }

    auto out = cudnn_lstm_forward(
        input.contiguous(), h0.contiguous(), c0.contiguous(),
        W_ih_list, W_hh_list, biases_ih, biases_hh, /*weights_hr=*/{},
        hidden, /*proj_size=*/0, num_layers,
        /*bidirectional=*/false, /*dropout=*/0.0f,
        CUDNN_FWD_MODE_INFERENCE, stream);

    // out.hy / out.cy are already (num_layers, batch, hidden).
    return { out.output, out.hy, out.cy };
}

// ---- Fused training forward/backward wrappers (single-layer, unidirectional,
// no projection). Tensor-only signatures so the registry TU can call them
// without the cudnnForwardMode_t mangling issue. States are (batch, hidden) on
// the boundary (matching the LSTMForward contract) and reshaped internally. ----
std::vector<Tensor> lstm_train_forward_cudnn(
    const Tensor& input, const Tensor& h0, const Tensor& c0,
    const Tensor& W_ih, const Tensor& W_hh,
    const Tensor& bias_ih, const Tensor& bias_hh) {
    const int64_t batch = h0.shape()[0];
    const int64_t hidden = h0.shape()[1];
    auto guard = CUDAStreamPool::instance().acquire_guard(input.device().index);
    cudaStream_t stream = guard.get();
    Tensor hx = h0.contiguous().reshape({1, batch, hidden});
    Tensor cx = c0.contiguous().reshape({1, batch, hidden});
    Tensor b_ih = bias_ih.numel() > 0 ? bias_ih : Tensor{};
    Tensor b_hh = bias_hh.numel() > 0 ? bias_hh : Tensor{};
    auto out = cudnn_lstm_forward(
        input.contiguous(), hx, cx, {W_ih}, {W_hh}, {b_ih}, {b_hh}, {},
        hidden, /*proj_size=*/0, /*num_layers=*/1, /*bidirectional=*/false,
        /*dropout=*/0.0f, CUDNN_FWD_MODE_TRAINING, stream);
    return { out.output, out.hy.reshape({batch, hidden}),
             out.cy.reshape({batch, hidden}), out.reserve_space, out.weight_space };
}

std::vector<Tensor> lstm_backward_cudnn_wrap(
    const Tensor& grad_out, const Tensor& grad_hy, const Tensor& grad_cy,
    const Tensor& input, const Tensor& h0, const Tensor& c0, const Tensor& output,
    const Tensor& weight_space, const Tensor& reserve_space,
    const Tensor& W_ih, const Tensor& W_hh,
    const Tensor& bias_ih, const Tensor& bias_hh) {
    const int64_t batch = h0.shape()[0];
    const int64_t hidden = h0.shape()[1];
    const int64_t input_size = W_ih.shape()[1];
    auto guard = CUDAStreamPool::instance().acquire_guard(input.device().index);
    cudaStream_t stream = guard.get();
    Tensor hx = h0.contiguous().reshape({1, batch, hidden});
    Tensor cx = c0.contiguous().reshape({1, batch, hidden});
    // Absent grads use a 0-size INITIALIZED tensor carrying the right
    // dtype/device so the downstream cudnn_lstm_backward call (ghy/gcy below)
    // receives a properly-typed handle on the correct device. (Tensor::numel()
    // is itself null-safe — it returns 0 when !impl_ — which is exactly why the
    // `grad_hy.numel() > 0` presence checks below work; the typed zero0 is for
    // device/dtype propagation, not to dodge a non-existent throw.)
    Tensor zero0({0}, grad_out.dtype(), grad_out.device());
    Tensor ghy = grad_hy.numel() > 0 ? grad_hy.contiguous().reshape({1, batch, hidden}) : zero0;
    Tensor gcy = grad_cy.numel() > 0 ? grad_cy.contiguous().reshape({1, batch, hidden}) : zero0;
    auto grads = cudnn_lstm_backward(
        grad_out.contiguous(), ghy, gcy, input.contiguous(), hx, cx,
        output.contiguous(), weight_space, reserve_space,
        hidden, /*proj_size=*/0, /*num_layers=*/1, /*bidirectional=*/false,
        /*dropout=*/0.0f, stream);
    std::vector<Tensor> gW_ih, gW_hh, gb_ih, gb_hh, gW_hr;
    cudnn_lstm_unpack_weight_grads(grads.grad_weight_space, input_size, hidden,
                                   /*proj_size=*/0, /*num_layers=*/1,
                                   /*bidirectional=*/false,
                                   gW_ih, gW_hh, gb_ih, gb_hh, gW_hr, stream);
    Tensor grad_hx = grads.grad_hx.numel() ? grads.grad_hx.reshape({batch, hidden}) : Tensor{};
    Tensor grad_cx = grads.grad_cx.numel() ? grads.grad_cx.reshape({batch, hidden}) : Tensor{};
    return { grads.grad_input, grad_hx, grad_cx,
             gW_ih.empty() ? Tensor{} : gW_ih[0],
             gW_hh.empty() ? Tensor{} : gW_hh[0],
             (bias_ih.numel() > 0 && !gb_ih.empty()) ? gb_ih[0] : Tensor{},
             (bias_hh.numel() > 0 && !gb_hh.empty()) ? gb_hh[0] : Tensor{} };
}

auto cudnn_lstm_backward(
    const Tensor& grad_output,
    const Tensor& grad_hy,
    const Tensor& grad_cy,
    const Tensor& input,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& output,
    const Tensor& weight_space,
    const Tensor& reserve_space,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudaStream_t stream) -> CudnnLSTMGrads {

    auto raw = run_backward(grad_output, grad_hy, grad_cy,
                            input, hx, cx, output,
                            weight_space, reserve_space,
                            hidden_size, proj_size, num_layers,
                            bidirectional, dropout, CUDNN_LSTM, stream);
    CudnnLSTMGrads g;
    g.grad_input = raw.grad_input;
    g.grad_hx = raw.grad_hx;
    g.grad_cx = raw.grad_cx;
    g.grad_weight_space = raw.grad_weight_space;
    return g;
}

void cudnn_lstm_unpack_weight_grads(
    const Tensor& weight_space_grad,
    int64_t input_size,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool bidirectional,
    std::vector<Tensor>& out_W_ih,
    std::vector<Tensor>& out_W_hh,
    std::vector<Tensor>& out_b_ih,
    std::vector<Tensor>& out_b_hh,
    std::vector<Tensor>& out_W_hr,
    cudaStream_t stream) {

    // Need an RNN descriptor matching the forward config to query layouts.
    auto handle = CuDNNHandle::get();
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    DropoutDescriptor dropout_desc;
    size_t dropout_state_size = 0;
    CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &dropout_state_size));
    Tensor dropout_state;
    void* dropout_state_ptr = nullptr;
    if (dropout_state_size > 0) {
        dropout_state = Tensor({static_cast<int64_t>(dropout_state_size)},
                               DType::UInt8, weight_space_grad.device());
        dropout_state_ptr = dropout_state.data_ptr();
    }
    // dropout=0 here; we only need the descriptor for weight layout queries.
    dropout_desc.set(handle, 0.0f, dropout_state_ptr, dropout_state_size, 0ULL);

    RNNDescriptor rnn_desc;
    rnn_desc.set_lstm(handle,
                      static_cast<int>(input_size),
                      static_cast<int>(hidden_size),
                      static_cast<int>(proj_size),
                      static_cast<int>(num_layers),
                      bidirectional,
                      dropout_desc.get(),
                      CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT);

    unpack_grads(rnn_desc.get(),
                 static_cast<size_t>(weight_space_grad.numel()),
                 weight_space_grad.data_ptr(),
                 CUDNN_LSTM,
                 static_cast<int>(num_layers),
                 bidirectional,
                 input_size, hidden_size, proj_size,
                 DType::Float32,
                 weight_space_grad.device(),
                 out_W_ih, out_W_hh, out_b_ih, out_b_hh, out_W_hr,
                 stream);
}

// ============================================================================
// Public GRU API
// ============================================================================

auto cudnn_gru_forward(
    const Tensor& input,
    const Tensor& hx,
    const std::vector<Tensor>& weights_ih,
    const std::vector<Tensor>& weights_hh,
    const std::vector<Tensor>& biases_ih,
    const std::vector<Tensor>& biases_hh,
    int64_t hidden_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudnnForwardMode_t fwd_mode,
    cudaStream_t stream) -> CudnnRNNOutputs {

    std::vector<Tensor> empty_hr;
    auto raw = run_forward(input, hx, /*cx=*/Tensor{},
                           weights_ih, weights_hh, biases_ih, biases_hh,
                           empty_hr,
                           hidden_size, /*proj_size=*/0, num_layers,
                           bidirectional, dropout,
                           CUDNN_GRU, fwd_mode, stream);
    CudnnRNNOutputs out;
    out.output = raw.output;
    out.hy = raw.hy;
    out.reserve_space = raw.reserve_space;
    out.weight_space = raw.weight_space;
    return out;
}

auto cudnn_gru_backward(
    const Tensor& grad_output,
    const Tensor& grad_hy,
    const Tensor& input,
    const Tensor& hx,
    const Tensor& output,
    const Tensor& weight_space,
    const Tensor& reserve_space,
    int64_t hidden_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudaStream_t stream) -> CudnnRNNGrads {

    auto raw = run_backward(grad_output, grad_hy, /*grad_cy=*/Tensor{},
                            input, hx, /*cx=*/Tensor{}, output,
                            weight_space, reserve_space,
                            hidden_size, /*proj_size=*/0, num_layers,
                            bidirectional, dropout, CUDNN_GRU, stream);
    CudnnRNNGrads g;
    g.grad_input = raw.grad_input;
    g.grad_hx = raw.grad_hx;
    g.grad_weight_space = raw.grad_weight_space;
    return g;
}

// ============================================================================
// Public Elman-RNN API
// ============================================================================

auto cudnn_rnn_forward(
    const Tensor& input,
    const Tensor& hx,
    const std::vector<Tensor>& weights_ih,
    const std::vector<Tensor>& weights_hh,
    const std::vector<Tensor>& biases_ih,
    const std::vector<Tensor>& biases_hh,
    int64_t hidden_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudnnRNNMode_t cell_mode,
    cudnnForwardMode_t fwd_mode,
    cudaStream_t stream) -> CudnnRNNOutputs {

    if (cell_mode != CUDNN_RNN_TANH && cell_mode != CUDNN_RNN_RELU) {
        throw std::runtime_error(
            "cudnn_rnn_forward: cell_mode must be CUDNN_RNN_TANH or CUDNN_RNN_RELU");
    }
    std::vector<Tensor> empty_hr;
    auto raw = run_forward(input, hx, /*cx=*/Tensor{},
                           weights_ih, weights_hh, biases_ih, biases_hh,
                           empty_hr,
                           hidden_size, /*proj_size=*/0, num_layers,
                           bidirectional, dropout,
                           cell_mode, fwd_mode, stream);
    CudnnRNNOutputs out;
    out.output = raw.output;
    out.hy = raw.hy;
    out.reserve_space = raw.reserve_space;
    out.weight_space = raw.weight_space;
    return out;
}

auto cudnn_rnn_backward(
    const Tensor& grad_output,
    const Tensor& grad_hy,
    const Tensor& input,
    const Tensor& hx,
    const Tensor& output,
    const Tensor& weight_space,
    const Tensor& reserve_space,
    int64_t hidden_size,
    int64_t num_layers,
    bool bidirectional,
    float dropout,
    cudnnRNNMode_t cell_mode,
    cudaStream_t stream) -> CudnnRNNGrads {

    if (cell_mode != CUDNN_RNN_TANH && cell_mode != CUDNN_RNN_RELU) {
        throw std::runtime_error(
            "cudnn_rnn_backward: cell_mode must be CUDNN_RNN_TANH or CUDNN_RNN_RELU");
    }
    auto raw = run_backward(grad_output, grad_hy, /*grad_cy=*/Tensor{},
                            input, hx, /*cx=*/Tensor{}, output,
                            weight_space, reserve_space,
                            hidden_size, /*proj_size=*/0, num_layers,
                            bidirectional, dropout, cell_mode, stream);
    CudnnRNNGrads g;
    g.grad_input = raw.grad_input;
    g.grad_hx = raw.grad_hx;
    g.grad_weight_space = raw.grad_weight_space;
    return g;
}

void cudnn_rnn_unpack_weight_grads(
    const Tensor& weight_space_grad,
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layers,
    bool bidirectional,
    cudnnRNNMode_t cell_mode,
    std::vector<Tensor>& out_W_ih,
    std::vector<Tensor>& out_W_hh,
    std::vector<Tensor>& out_b_ih,
    std::vector<Tensor>& out_b_hh,
    cudaStream_t stream) {

    if (cell_mode == CUDNN_LSTM) {
        throw std::runtime_error(
            "cudnn_rnn_unpack_weight_grads: use cudnn_lstm_unpack_weight_grads for LSTM");
    }

    auto handle = CuDNNHandle::get();
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    DropoutDescriptor dropout_desc;
    size_t dropout_state_size = 0;
    CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &dropout_state_size));
    Tensor dropout_state;
    void* dropout_state_ptr = nullptr;
    if (dropout_state_size > 0) {
        dropout_state = Tensor({static_cast<int64_t>(dropout_state_size)},
                               DType::UInt8, weight_space_grad.device());
        dropout_state_ptr = dropout_state.data_ptr();
    }
    dropout_desc.set(handle, 0.0f, dropout_state_ptr, dropout_state_size, 0ULL);

    RNNDescriptor rnn_desc;
    if (cell_mode == CUDNN_GRU) {
        rnn_desc.set_gru(handle,
                        static_cast<int>(input_size),
                        static_cast<int>(hidden_size),
                        static_cast<int>(num_layers),
                        bidirectional,
                        dropout_desc.get(),
                        CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT);
    } else {
        rnn_desc.set_rnn(handle,
                        static_cast<int>(input_size),
                        static_cast<int>(hidden_size),
                        static_cast<int>(num_layers),
                        bidirectional,
                        dropout_desc.get(),
                        CUDNN_DATA_FLOAT, cell_mode, CUDNN_DATA_FLOAT);
    }

    std::vector<Tensor> unused_hr;
    unpack_grads(rnn_desc.get(),
                 static_cast<size_t>(weight_space_grad.numel()),
                 weight_space_grad.data_ptr(),
                 cell_mode,
                 static_cast<int>(num_layers),
                 bidirectional,
                 input_size, hidden_size, /*proj_size=*/0,
                 DType::Float32,
                 weight_space_grad.device(),
                 out_W_ih, out_W_hh, out_b_ih, out_b_hh, unused_hr,
                 stream);
}

// ---- Fused GRU training forward/backward wrappers (single-layer,
// unidirectional). Tensor-only signatures so the registry TU can call them
// without the cudnnForwardMode_t mangling issue. h state is (batch, hidden)
// on the boundary (matching the GRUForward contract) and reshaped internally.
// Mirrors the LSTM wrappers; GRU has no cell state. ----
std::vector<Tensor> gru_train_forward_cudnn(
    const Tensor& input, const Tensor& h0,
    const Tensor& W_ih, const Tensor& W_hh,
    const Tensor& bias_ih, const Tensor& bias_hh) {
    const int64_t batch = h0.shape()[0];
    const int64_t hidden = h0.shape()[1];
    auto guard = CUDAStreamPool::instance().acquire_guard(input.device().index);
    cudaStream_t stream = guard.get();
    Tensor hx = h0.contiguous().reshape({1, batch, hidden});
    Tensor b_ih = bias_ih.numel() > 0 ? bias_ih : Tensor{};
    Tensor b_hh = bias_hh.numel() > 0 ? bias_hh : Tensor{};
    auto out = cudnn_gru_forward(
        input.contiguous(), hx, {W_ih}, {W_hh}, {b_ih}, {b_hh},
        hidden, /*num_layers=*/1, /*bidirectional=*/false, /*dropout=*/0.0f,
        CUDNN_FWD_MODE_TRAINING, stream);
    return { out.output, out.hy.reshape({batch, hidden}),
             out.reserve_space, out.weight_space };
}

std::vector<Tensor> gru_backward_cudnn_wrap(
    const Tensor& grad_out, const Tensor& grad_hy,
    const Tensor& input, const Tensor& h0, const Tensor& output,
    const Tensor& weight_space, const Tensor& reserve_space,
    const Tensor& W_ih, const Tensor& W_hh,
    const Tensor& bias_ih, const Tensor& bias_hh) {
    const int64_t batch = h0.shape()[0];
    const int64_t hidden = h0.shape()[1];
    const int64_t input_size = W_ih.shape()[1];
    auto guard = CUDAStreamPool::instance().acquire_guard(input.device().index);
    cudaStream_t stream = guard.get();
    Tensor hx = h0.contiguous().reshape({1, batch, hidden});
    // Absent grad uses a 0-size INITIALIZED tensor (run_backward calls .numel()).
    Tensor zero0({0}, grad_out.dtype(), grad_out.device());
    Tensor ghy = grad_hy.numel() > 0 ? grad_hy.contiguous().reshape({1, batch, hidden}) : zero0;
    auto grads = cudnn_gru_backward(
        grad_out.contiguous(), ghy, input.contiguous(), hx,
        output.contiguous(), weight_space, reserve_space,
        hidden, /*num_layers=*/1, /*bidirectional=*/false, /*dropout=*/0.0f, stream);
    std::vector<Tensor> gW_ih, gW_hh, gb_ih, gb_hh;
    cudnn_rnn_unpack_weight_grads(grads.grad_weight_space, input_size, hidden,
                                  /*num_layers=*/1, /*bidirectional=*/false, CUDNN_GRU,
                                  gW_ih, gW_hh, gb_ih, gb_hh, stream);
    Tensor grad_hx = grads.grad_hx.numel() ? grads.grad_hx.reshape({batch, hidden}) : Tensor{};
    return { grads.grad_input, grad_hx,
             gW_ih.empty() ? Tensor{} : gW_ih[0],
             gW_hh.empty() ? Tensor{} : gW_hh[0],
             (bias_ih.numel() > 0 && !gb_ih.empty()) ? gb_ih[0] : Tensor{},
             (bias_hh.numel() > 0 && !gb_hh.empty()) ? gb_hh[0] : Tensor{} };
}

}  // namespace cuda
}  // namespace tenzor

#endif  // TENZOR_HAS_CUDNN
