# Conv2d GPU Implementation Summary

## Overview
This document summarizes the implementation of GPU-native Conv2d operations to eliminate CPU fallbacks in the Tenzor tensor library.

## Files Created/Modified

### 1. Created: `/home/lee/Projects/Tenzor/src/backends/cuda/kernels/conv2d.cu`

**Status**: ✅ COMPLETE

This file contains:
- `im2col_kernel` - Converts 4D input (N,C,H,W) to 2D matrix for GEMM
- `col2im_kernel` - Reverse of im2col for backward pass (uses atomicAdd for gradient accumulation)
- `conv2d_forward_kernel` - Forward pass using im2col + cuBLAS SGEMM
- `conv2d_backward_kernel` - Backward pass computing gradients w.r.t input, weight, and bias

**Key Features**:
- All operations stay on GPU (no CPU transfers)
- Uses cuBLAS for efficient matrix multiplication
- Supports groups, padding, stride, dilation
- Handles bias addition via dedicated CUDA kernel
- Memory efficient with per-group processing

### 2. Modified: `/home/lee/Projects/Tenzor/src/backends/cuda/CMakeLists.txt`

**Status**: ✅ COMPLETE

Added `kernels/conv2d.cu` to the CUDA_BACKEND_SOURCES list.

### 3. To Modify: `/home/lee/Projects/Tenzor/src/backends/cuda/cuda_backend.cpp`

**Status**: ⚠️ PENDING - Add the following sections

#### A. Forward Declarations (add after line 76, before `} // namespace cuda`):

```cpp
    // Conv2d operations
    auto conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                               int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                               cudaStream_t stream) -> Tensor;
    auto conv2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                                int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                                bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias,
                                cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
```

#### B. Dispatcher Handlers (add before the final `else` clause in dispatch method, around line 766):

```cpp
            else if (op_name == "conv2d_forward") {
                // Expect inputs: [input, weight] or [input, weight, bias]
                if (inputs.size() < 2 || inputs.size() > 3) {
                    throw std::invalid_argument("conv2d_forward operation requires 2 or 3 inputs");
                }

                // Parse conv parameters from attributes
                int64_t stride = 1;
                int64_t padding = 0;
                int64_t dilation = 1;
                int64_t groups = 1;

                if (attrs.contains("stride")) {
                    stride = std::stoll(attrs.at("stride"));
                }
                if (attrs.contains("padding")) {
                    padding = std::stoll(attrs.at("padding"));
                }
                if (attrs.contains("dilation")) {
                    dilation = std::stoll(attrs.at("dilation"));
                }
                if (attrs.contains("groups")) {
                    groups = std::stoll(attrs.at("groups"));
                }

                const Tensor* bias = (inputs.size() == 3) ? &inputs[2] : nullptr;
                return {cuda::conv2d_forward_kernel(inputs[0], inputs[1], bias,
                                                     stride, padding, dilation, groups, stream)};
            }
            else if (op_name == "conv2d_backward") {
                // Expect inputs: [grad_output, input, weight]
                if (inputs.size() != 3) {
                    throw std::invalid_argument("conv2d_backward operation requires exactly 3 inputs");
                }

                // Parse conv parameters from attributes
                int64_t stride = 1;
                int64_t padding = 0;
                int64_t dilation = 1;
                int64_t groups = 1;
                bool compute_grad_input = true;
                bool compute_grad_weight = true;
                bool compute_grad_bias = true;

                if (attrs.contains("stride")) {
                    stride = std::stoll(attrs.at("stride"));
                }
                if (attrs.contains("padding")) {
                    padding = std::stoll(attrs.at("padding"));
                }
                if (attrs.contains("dilation")) {
                    dilation = std::stoll(attrs.at("dilation"));
                }
                if (attrs.contains("groups")) {
                    groups = std::stoll(attrs.at("groups"));
                }
                if (attrs.contains("compute_grad_input")) {
                    compute_grad_input = (attrs.at("compute_grad_input") == "1");
                }
                if (attrs.contains("compute_grad_weight")) {
                    compute_grad_weight = (attrs.at("compute_grad_weight") == "1");
                }
                if (attrs.contains("compute_grad_bias")) {
                    compute_grad_bias = (attrs.at("compute_grad_bias") == "1");
                }

                auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
                    inputs[0], inputs[1], inputs[2],
                    stride, padding, dilation, groups,
                    compute_grad_input, compute_grad_weight, compute_grad_bias,
                    stream
                );

                return {grad_input, grad_weight, grad_bias};
            }
```

### 4. To Modify: `/home/lee/Projects/Tenzor/src/nn/layers/conv.cpp`

**Status**: ⚠️ PENDING - Remove CPU fallbacks

#### Changes Required:

1. **Remove CPU fallback in `im2col` function** (lines 38-81):
   - Delete the CPU transfer code
   - Call backend dispatch for GPU tensors instead

2. **Remove CPU fallback in `col2im` function** (lines 98-136):
   - Delete the CPU transfer code
   - Call backend dispatch for GPU tensors instead

3. **Remove CPU fallback in `Conv2dBackward::backward`** (lines 160-430):
   - Delete CPU transfer and CPU computation
   - Call backend dispatch for GPU backward pass

4. **Remove CPU fallback in `Conv2d::forward`** (lines 506-648):
   - Delete CPU transfer and CPU computation
   - Call backend dispatch for GPU forward pass

#### Example GPU Path Implementation:

Replace CPU fallback sections with backend dispatch calls like:

```cpp
// In Conv2d::forward - replace lines 506-648
if (input.tensor().device().type == Device::Type::CUDA) {
    // Use GPU kernel via backend dispatch
    OpAttributes attrs;
    attrs["stride"] = std::to_string(stride_);
    attrs["padding"] = std::to_string(padding_);
    attrs["dilation"] = std::to_string(dilation_);
    attrs["groups"] = std::to_string(groups_);

    std::vector<Tensor> inputs = {input.tensor(), weight.tensor()};
    if (bias_it != parameters_.end()) {
        inputs.push_back(bias_it->second.tensor());
    }

    auto outputs = input.tensor().device().backend()->dispatch("conv2d_forward", inputs, attrs);
    output = outputs[0];
} else {
    // Keep CPU implementation as fallback
    // ... existing CPU code ...
}
```

## Architecture

### Memory Flow (Forward Pass):
```
Input (N,C,H,W) GPU
    ↓
im2col kernel → col_buffer (batch*out_h*out_w, C*kH*kW) GPU
    ↓
cuBLAS SGEMM: col_buffer @ weight^T → output_flat GPU
    ↓
Bias addition kernel → Output (N,C_out,H_out,W_out) GPU
```

### Memory Flow (Backward Pass):
```
grad_output (N,C_out,H_out,W_out) GPU
    ↓
[Gradient w.r.t input]
cuBLAS: grad_output @ weight → grad_col GPU
    ↓
col2im kernel → grad_input (N,C,H,W) GPU

[Gradient w.r.t weight]
im2col(input) → input_col GPU
    ↓
cuBLAS: grad_output^T @ input_col → grad_weight GPU

[Gradient w.r.t bias]
Sum reduction kernel → grad_bias GPU
```

## Performance Characteristics

- **No CPU-GPU transfers**: All operations stay on GPU
- **cuBLAS optimization**: Leverages highly optimized GEMM kernels
- **Memory efficient**: Per-group processing reduces peak memory
- **Asynchronous**: Uses CUDA streams for overlap
- **Scalable**: Grid-stride loop pattern handles any tensor size

## Testing

Build and test with:
```bash
cd /home/lee/Projects/Tenzor/build
make -j$(nproc)
ctest -R "Conv" --output-on-failure
```

## Verification Checklist

- [x] im2col kernel implemented
- [x] col2im kernel implemented
- [x] Forward pass implemented with cuBLAS
- [x] Backward pass implemented
- [x] CMakeLists.txt updated
- [ ] cuda_backend.cpp updated with forward declarations
- [ ] cuda_backend.cpp updated with dispatcher handlers
- [ ] conv.cpp updated to remove CPU fallbacks
- [ ] Tests passing

## Notes

- cuBLAS must be available (CMAKE will link it automatically if found)
- Supports all Conv2d parameters: stride, padding, dilation, groups, bias
- Thread-safe through separate cuBLAS handles per forward/backward call
- Compatible with existing autograd system
