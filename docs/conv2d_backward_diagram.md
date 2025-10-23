# Conv2d Backward Pass - Algorithm Visualization

## Forward Pass (for reference)
```
Input [N, C_in, H_in, W_in]
        ↓
    im2col → Col [C_in*K_h*K_w, H_out*W_out]
        ↓
    GEMM: Output = Weight × Col
        ↓
Output [N, C_out, H_out, W_out]
```

## Backward Pass Implementation

### 1. Grad Input Computation
```
Grad Output [N, C_out, H_out, W_out]
        ↓
    GEMM: Col = Weight^T × Grad_Output
        ↓
    Col [C_in*K_h*K_w, H_out*W_out]
        ↓
    col2im (scatter gradients)
        ↓
Grad Input [N, C_in, H_in, W_in]
```

**Matrix Dimensions:**
```
Weight: [C_out, C_in*K_h*K_w]
Weight^T: [C_in*K_h*K_w, C_out]
Grad_Output (reshaped): [C_out, H_out*W_out]
Result Col: [C_in*K_h*K_w, H_out*W_out]
```

**oneMKL Call:**
```cpp
gemm(queue, nontrans, trans,
     N_gemm=H_out*W_out, M=C_in*K_h*K_w, K=C_out,
     alpha=1.0,
     grad_out_batch[C_out, H_out*W_out], ld=N_gemm,
     weight[C_out, C_in*K_h*K_w], ld=M,
     beta=0.0,
     col[C_in*K_h*K_w, H_out*W_out], ld=N_gemm)
```

### 2. Grad Weight Computation
```
Input [N, C_in, H_in, W_in]
        ↓
    im2col → Col [C_in*K_h*K_w, H_out*W_out]
        ↓
    GEMM: Grad_Weight += Grad_Output × Col^T
        ↓
Grad Weight [C_out, C_in*K_h*K_w]

(Accumulate across all batches)
```

**Matrix Dimensions:**
```
Grad_Output (reshaped): [C_out, H_out*W_out]
Col: [C_in*K_h*K_w, H_out*W_out]
Col^T: [H_out*W_out, C_in*K_h*K_w]
Result: [C_out, C_in*K_h*K_w]
```

**oneMKL Call (per batch):**
```cpp
gemm(queue, trans, nontrans,
     N_gemm=C_in*K_h*K_w, M=C_out, K=H_out*W_out,
     alpha=1.0,
     col[C_in*K_h*K_w, H_out*W_out], ld=K,
     grad_out_batch[C_out, H_out*W_out], ld=K,
     beta=1.0,  // Accumulate!
     grad_weight[C_out, C_in*K_h*K_w], ld=N_gemm)
```

## Col2im Algorithm Details

### Input Processing
```
Col buffer organized as:
  [channel_0_kernel_0] [H_out*W_out values]
  [channel_0_kernel_1] [H_out*W_out values]
  ...
  [channel_C-1_kernel_K²-1] [H_out*W_out values]

For each position (c, kh, kw, h_out, w_out):
  1. Compute input coordinates:
     h_in = h_out * stride - pad + kh * dilation
     w_in = w_out * stride - pad + kw * dilation

  2. If valid (within bounds):
     grad_input[c, h_in, w_in] += col[index]
     (using atomic add for thread safety)
```

### Thread Organization
```
Total threads = C_in × K_h × K_w × H_out × W_out

Thread mapping (linearized index):
  w_out = index % W_out
  h_out = (index / W_out) % H_out
  kw = (index / (W_out * H_out)) % K_w
  kh = (index / (W_out * H_out * K_w)) % K_h
  c = index / (W_out * H_out * K_w * K_h)
```

## Im2col Algorithm (for reference)

### Input to Column Transformation
```
Image [C, H, W] → Col [C*K_h*K_w, H_out*W_out]

For each position (c, kh, kw, h_out, w_out):
  1. Compute input coordinates:
     h_in = h_out * stride - pad + kh * dilation
     w_in = w_out * stride - pad + kw * dilation

  2. Extract value:
     col[index] = image[c, h_in, w_in] if valid else 0
```

## Execution Paths

### Path 1: With oneMKL (Optimized)
```
✓ Hardware-optimized BLAS operations
✓ Vectorized execution
✓ Cache-friendly memory access
✓ Minimal synchronization overhead
✓ Recommended for production
```

### Path 2: Fallback (Portable)
```
✓ Pure SYCL implementation
✓ Works on any SYCL-compliant device
✓ Parallel execution via parallel_for
✓ Atomic operations for safety
✓ Useful for debugging/development
```

## Memory Access Patterns

### Grad Input (Col2im)
```
READ:  Sequential access to col buffer
WRITE: Scattered atomic writes to grad_input
SYNC:  Atomic operations ensure correctness
```

### Grad Weight
```
READ:  Sequential access to input (via col buffer)
READ:  Sequential access to grad_output
WRITE: Accumulated atomic writes to grad_weight
SYNC:  Batch accumulation requires atomics
```

## Performance Characteristics

### Time Complexity
- **Grad Input:** O(N × C_in × K_h × K_w × H_out × W_out)
- **Grad Weight:** O(N × C_out × C_in × K_h × K_w × H_out × W_out)

### Space Complexity
- **Col Buffer:** O(C_in × K_h × K_w × H_out × W_out) per batch
- **Grad Tensors:** O(input_size) + O(weight_size)

### Optimization Opportunities
1. **Workspace Reuse:** Share col buffer across backward calls
2. **Tiling:** Break large tensors into tiles for better cache utilization
3. **Asynchronous Execution:** Overlap GEMM with col2im/im2col
4. **Mixed Precision:** Use FP16 for intermediate computations
5. **Kernel Fusion:** Combine col2im with other operations

## Validation Checklist

- [x] Correct matrix dimensions in GEMM
- [x] Proper handling of stride, padding, dilation
- [x] Thread-safe atomic operations
- [x] Zero initialization where needed
- [x] Boundary checking in kernels
- [x] Batch accumulation for grad_weight
- [x] Memory layout consistency (NCHW)
- [x] Queue synchronization points
- [x] Error handling for edge cases
- [x] Documentation and comments
