/**
 * @file extended_codegen.cpp
 * @brief Implementation of extended GPU kernel generation
 */

#include "tenzor/jit/extended_codegen.hpp"
#include "tenzor/jit/codegen.hpp"
#include <sstream>

namespace tenzor {
namespace jit {

// ============================================================================
// Signature computation
// ============================================================================

auto ExtendedFusionGroup::compute_signature() -> std::string {
    std::ostringstream ss;
    ss << "xfuse_" << static_cast<int>(kind) << "_" << static_cast<int>(dtype);

    switch (kind) {
        case FusionKind::Reduction:
            ss << "_dim" << reduce_dim << "_pre" << pre_ops.size()
               << "_post" << post_ops.size();
            for (auto& op : pre_ops) ss << "_" << static_cast<int>(op.op);
            for (auto& op : post_ops) ss << "_" << static_cast<int>(op.op);
            break;
        case FusionKind::GemmEpilogue:
            ss << "_bias" << has_bias << "_act" << static_cast<int>(activation_type);
            break;
        case FusionKind::Softmax:
            ss << "_dim" << softmax_dim;
            break;
        case FusionKind::LayerNorm:
            ss << "_axis" << norm_axis << "_aff" << has_affine;
            break;
        case FusionKind::RMSNorm:
            ss << "_axis" << norm_axis << "_aff" << has_affine;
            break;
        case FusionKind::SmallMLP:
            ss << "_h" << hidden_dim << "_act" << static_cast<int>(mlp_activation);
            break;
        default:
            break;
    }

    signature = ss.str();
    return signature;
}

// ============================================================================
// Helpers
// ============================================================================

auto ExtendedKernelCodegen::dtype_to_cuda_type(DType dtype) -> std::string {
    switch (dtype) {
        case DType::Float32:  return "float";
        case DType::Float64:  return "double";
        case DType::Float16:  return "__half";
        case DType::BFloat16: return "__nv_bfloat16";
        default:              return "float";
    }
}

auto ExtendedKernelCodegen::activation_expr(OpType act, const std::string& var) -> std::string {
    switch (act) {
        case OpType::ReLU:    return var + " > 0 ? " + var + " : 0";
        case OpType::Sigmoid: return "1.0f / (1.0f + expf(-" + var + "))";
        case OpType::Tanh:    return "tanhf(" + var + ")";
        case OpType::GELU:
            return "0.5f * " + var + " * (1.0f + tanhf(0.7978845608f * ("
                   + var + " + 0.044715f * " + var + " * " + var + " * " + var + ")))";
        default:
            return var;
    }
}

// ============================================================================
// Dispatch
// ============================================================================

auto ExtendedKernelCodegen::generate(const ExtendedFusionGroup& group) -> std::string {
    switch (group.kind) {
        case FusionKind::Reduction:    return generate_reduction(group);
        case FusionKind::GemmEpilogue: return generate_gemm_epilogue(group);
        case FusionKind::Softmax:      return generate_softmax(group);
        case FusionKind::LayerNorm:    return generate_layer_norm(group);
        case FusionKind::RMSNorm:      return generate_rms_norm(group);
        case FusionKind::SmallMLP:     return generate_small_mlp(group);
        default:
            return "";
    }
}

// ============================================================================
// Reduction kernel: block-parallel with warp shuffles
// ============================================================================

auto ExtendedKernelCodegen::generate_reduction(const ExtendedFusionGroup& group) -> std::string {
    auto T = dtype_to_cuda_type(group.dtype);
    std::ostringstream ss;

    ss << R"(
extern "C" __global__ void fused_reduction_kernel(
    const )" << T << R"(* __restrict__ input,
    )" << T << R"(* __restrict__ output,
    int64_t outer_size, int64_t reduce_size, int64_t inner_size) {

    // Grid: (outer_size * inner_size) blocks, 256 threads each
    int idx = blockIdx.x;
    int outer = idx / inner_size;
    int inner = idx % inner_size;
    if (outer >= outer_size) return;

    // Each block reduces one (outer, inner) slice
    )" << T << R"( sum = 0;
    for (int r = threadIdx.x; r < reduce_size; r += blockDim.x) {
        )" << T << R"( val = input[outer * reduce_size * inner_size + r * inner_size + inner];
)";

    // Inline pre-reduction element-wise ops
    for (auto& op : group.pre_ops) {
        switch (op.op) {
            case ElemOp::Mul:    ss << "        val = val * val;\n"; break;
            case ElemOp::Abs:    ss << "        val = fabsf(val);\n"; break;
            case ElemOp::Exp:    ss << "        val = expf(val);\n"; break;
            case ElemOp::Neg:    ss << "        val = -val;\n"; break;
            case ElemOp::MulScalar:
                ss << "        val = val * " << op.scalar << "f;\n"; break;
            default: break;
        }
    }

    ss << R"(        sum += val;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Block-level reduction via shared memory
    __shared__ )" << T << R"( shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) shared[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < blockDim.x / warpSize) ? shared[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    if (threadIdx.x == 0) {
        )" << T << R"( result = sum;
)";

    // Inline post-reduction element-wise ops
    for (auto& op : group.post_ops) {
        switch (op.op) {
            case ElemOp::Sqrt:     ss << "        result = sqrtf(result);\n"; break;
            case ElemOp::Reciprocal: ss << "        result = 1.0f / result;\n"; break;
            case ElemOp::MulScalar:
                ss << "        result = result * " << op.scalar << "f;\n"; break;
            case ElemOp::AddScalar:
                ss << "        result = result + " << op.scalar << "f;\n"; break;
            default: break;
        }
    }

    ss << R"(        output[outer * inner_size + inner] = result;
    }
}
)";

    return ss.str();
}

// ============================================================================
// GEMM epilogue: bias + activation fused with matmul result
// ============================================================================

auto ExtendedKernelCodegen::generate_gemm_epilogue(const ExtendedFusionGroup& group) -> std::string {
    auto T = dtype_to_cuda_type(group.dtype);
    std::ostringstream ss;

    ss << R"(
extern "C" __global__ void fused_gemm_epilogue_kernel(
    )" << T << R"(* __restrict__ output,)";

    if (group.has_bias) {
        ss << R"(
    const )" << T << R"(* __restrict__ bias,)";
    }

    ss << R"(
    int64_t rows, int64_t cols) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    int col = idx % cols;
    )" << T << " val = output[idx];\n";

    if (group.has_bias) {
        ss << "    val = val + bias[col];\n";
    }

    if (group.has_activation) {
        ss << "    val = " << activation_expr(group.activation_type, "val") << ";\n";
    }

    ss << R"(    output[idx] = val;
}
)";

    return ss.str();
}

// ============================================================================
// Softmax: online 2-pass fused per-row
// ============================================================================

auto ExtendedKernelCodegen::generate_softmax(const ExtendedFusionGroup& group) -> std::string {
    auto T = dtype_to_cuda_type(group.dtype);
    std::ostringstream ss;

    ss << R"(
extern "C" __global__ void fused_softmax_kernel(
    const )" << T << R"(* __restrict__ input,
    )" << T << R"(* __restrict__ output,
    int64_t rows, int64_t cols) {

    // One block per row
    int row = blockIdx.x;
    if (row >= rows) return;

    const )" << T << R"(* row_input = input + row * cols;
    )" << T << R"(* row_output = output + row * cols;

    // Pass 1: find max (for numerical stability)
    )" << T << R"( thread_max = -1e30f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        )" << T << R"( val = row_input[c];
        thread_max = val > thread_max ? val : thread_max;
    }

    // Warp reduction for max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        )" << T << R"( other = __shfl_down_sync(0xffffffff, thread_max, offset);
        thread_max = other > thread_max ? other : thread_max;
    }

    __shared__ )" << T << R"( shared_max[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) shared_max[warp_id] = thread_max;
    __syncthreads();

    if (warp_id == 0) {
        thread_max = (lane < blockDim.x / warpSize) ? shared_max[lane] : -1e30f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            )" << T << R"( other = __shfl_down_sync(0xffffffff, thread_max, offset);
            thread_max = other > thread_max ? other : thread_max;
        }
        if (lane == 0) shared_max[0] = thread_max;
    }
    __syncthreads();
    )" << T << R"( row_max = shared_max[0];

    // Pass 2: compute exp(x - max) and sum
    )" << T << R"( thread_sum = 0;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        )" << T << R"( val = expf(row_input[c] - row_max);
        row_output[c] = val;  // Store intermediate exp values
        thread_sum += val;
    }

    // Warp reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    __shared__ )" << T << R"( shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = thread_sum;
    __syncthreads();

    if (warp_id == 0) {
        thread_sum = (lane < blockDim.x / warpSize) ? shared_sum[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        if (lane == 0) shared_sum[0] = thread_sum;
    }
    __syncthreads();
    )" << T << R"( inv_sum = 1.0f / shared_sum[0];

    // Pass 3: normalize
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        row_output[c] = row_output[c] * inv_sum;
    }
}
)";

    return ss.str();
}

// ============================================================================
// LayerNorm: Welford online mean+variance, single-kernel
// ============================================================================

auto ExtendedKernelCodegen::generate_layer_norm(const ExtendedFusionGroup& group) -> std::string {
    auto T = dtype_to_cuda_type(group.dtype);
    std::ostringstream ss;

    ss << R"(
extern "C" __global__ void fused_layer_norm_kernel(
    const )" << T << R"(* __restrict__ input,
    )" << T << R"(* __restrict__ output,)";

    if (group.has_affine) {
        ss << R"(
    const )" << T << R"(* __restrict__ gamma,
    const )" << T << R"(* __restrict__ beta,)";
    }

    ss << R"(
    int64_t outer_size, int64_t norm_size, float eps) {

    // One block per normalized instance
    int instance = blockIdx.x;
    if (instance >= outer_size) return;

    const )" << T << R"(* x = input + instance * norm_size;
    )" << T << R"(* y = output + instance * norm_size;

    // Welford online mean computation
    )" << T << R"( mean = 0;
    )" << T << R"( m2 = 0;
    int count = 0;

    for (int i = threadIdx.x; i < norm_size; i += blockDim.x) {
        count++;
        )" << T << R"( delta = x[i] - mean;
        mean += delta / count;
        )" << T << R"( delta2 = x[i] - mean;
        m2 += delta * delta2;
    }

    // Warp-level Welford merge
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        )" << T << R"( other_mean = __shfl_down_sync(0xffffffff, mean, offset);
        )" << T << R"( other_m2 = __shfl_down_sync(0xffffffff, m2, offset);
        int other_count = __shfl_down_sync(0xffffffff, count, offset);

        int total = count + other_count;
        if (total > 0) {
            )" << T << R"( delta = other_mean - mean;
            mean = (count * mean + other_count * other_mean) / total;
            m2 = m2 + other_m2 + delta * delta * count * other_count / total;
            count = total;
        }
    }

    // Block-level merge via shared memory
    __shared__ )" << T << R"( s_mean[32], s_m2[32];
    __shared__ int s_count[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        s_mean[warp_id] = mean;
        s_m2[warp_id] = m2;
        s_count[warp_id] = count;
    }
    __syncthreads();

    if (warp_id == 0) {
        int nwarps = blockDim.x / warpSize;
        mean = (lane < nwarps) ? s_mean[lane] : 0;
        m2 = (lane < nwarps) ? s_m2[lane] : 0;
        count = (lane < nwarps) ? s_count[lane] : 0;

        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            )" << T << R"( other_mean = __shfl_down_sync(0xffffffff, mean, offset);
            )" << T << R"( other_m2 = __shfl_down_sync(0xffffffff, m2, offset);
            int other_count = __shfl_down_sync(0xffffffff, count, offset);

            int total = count + other_count;
            if (total > 0) {
                )" << T << R"( delta = other_mean - mean;
                mean = (count * mean + other_count * other_mean) / total;
                m2 = m2 + other_m2 + delta * delta * count * other_count / total;
                count = total;
            }
        }

        if (lane == 0) {
            s_mean[0] = mean;
            s_m2[0] = m2 / count;  // variance
        }
    }
    __syncthreads();

    )" << T << R"( final_mean = s_mean[0];
    )" << T << R"( inv_std = rsqrtf(s_m2[0] + eps);

    // Normalize + affine
    for (int i = threadIdx.x; i < norm_size; i += blockDim.x) {
        )" << T << R"( val = (x[i] - final_mean) * inv_std;)";

    if (group.has_affine) {
        ss << R"(
        val = val * gamma[i] + beta[i];)";
    }

    ss << R"(
        y[i] = val;
    }
}
)";

    return ss.str();
}

// ============================================================================
// RMSNorm: square -> mean -> rsqrt -> mul
// ============================================================================

auto ExtendedKernelCodegen::generate_rms_norm(const ExtendedFusionGroup& group) -> std::string {
    auto T = dtype_to_cuda_type(group.dtype);
    std::ostringstream ss;

    ss << R"(
extern "C" __global__ void fused_rms_norm_kernel(
    const )" << T << R"(* __restrict__ input,
    )" << T << R"(* __restrict__ output,)";

    if (group.has_affine) {
        ss << R"(
    const )" << T << R"(* __restrict__ gamma,)";
    }

    ss << R"(
    int64_t outer_size, int64_t norm_size, float eps) {

    int instance = blockIdx.x;
    if (instance >= outer_size) return;

    const )" << T << R"(* x = input + instance * norm_size;
    )" << T << R"(* y = output + instance * norm_size;

    // Compute sum of squares
    )" << T << R"( sum_sq = 0;
    for (int i = threadIdx.x; i < norm_size; i += blockDim.x) {
        )" << T << R"( val = x[i];
        sum_sq += val * val;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ )" << T << R"( shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane < blockDim.x / warpSize) ? shared[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        if (lane == 0) shared[0] = sum_sq;
    }
    __syncthreads();

    )" << T << R"( rms_inv = rsqrtf(shared[0] / norm_size + eps);

    // Normalize
    for (int i = threadIdx.x; i < norm_size; i += blockDim.x) {
        )" << T << R"( val = x[i] * rms_inv;)";

    if (group.has_affine) {
        ss << R"(
        val = val * gamma[i];)";
    }

    ss << R"(
        y[i] = val;
    }
}
)";

    return ss.str();
}

// ============================================================================
// Small MLP: shared-memory tiled GEMM with fused activation
// ============================================================================

auto ExtendedKernelCodegen::generate_small_mlp(const ExtendedFusionGroup& group) -> std::string {
    auto T = dtype_to_cuda_type(group.dtype);
    std::ostringstream ss;

    // For small MLPs, we generate a two-stage kernel:
    // Stage 1: Y = activation(X @ W1 + b1)  (stored in shared memory)
    // Stage 2: Z = Y @ W2 + b2
    // This avoids writing the intermediate Y to global memory.

    ss << R"(
extern "C" __global__ void fused_small_mlp_kernel(
    const )" << T << R"(* __restrict__ input,     // [batch, in_dim]
    const )" << T << R"(* __restrict__ w1,         // [in_dim, hidden_dim]
    const )" << T << R"(* __restrict__ b1,         // [hidden_dim]
    const )" << T << R"(* __restrict__ w2,         // [hidden_dim, out_dim]
    const )" << T << R"(* __restrict__ b2,         // [out_dim]
    )" << T << R"(* __restrict__ output,           // [batch, out_dim]
    int64_t batch, int64_t in_dim, int64_t hidden_dim, int64_t out_dim) {

    // One block per batch element
    int b = blockIdx.x;
    if (b >= batch) return;

    extern __shared__ )" << T << R"( smem[];
    )" << T << R"(* hidden = smem;  // [hidden_dim]

    const )" << T << R"(* x = input + b * in_dim;

    // Stage 1: hidden = activation(x @ W1 + b1)
    for (int h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        )" << T << R"( acc = b1[h];
        for (int i = 0; i < in_dim; ++i) {
            acc += x[i] * w1[i * hidden_dim + h];
        }
        hidden[h] = )" << activation_expr(group.mlp_activation, "acc") << R"(;
    }
    __syncthreads();

    // Stage 2: output = hidden @ W2 + b2
    )" << T << R"(* out = output + b * out_dim;
    for (int o = threadIdx.x; o < out_dim; o += blockDim.x) {
        )" << T << R"( acc = b2[o];
        for (int h = 0; h < hidden_dim; ++h) {
            acc += hidden[h] * w2[h * out_dim + o];
        }
        out[o] = acc;
    }
}
)";

    return ss.str();
}

// ============================================================================
// Extended kernel execution
// ============================================================================

auto execute_extended_fused(const ExtendedFusionGroup& group,
                            const std::vector<Tensor>& inputs,
                            const std::vector<Tensor>& params) -> Tensor {
    // Generate source code
    auto source = ExtendedKernelCodegen::generate(group);
    if (source.empty()) {
        throw std::runtime_error("Extended codegen: unsupported fusion kind");
    }

    // Use existing KernelCache for compilation
    // For now, delegate to the element-wise infrastructure with a custom signature
    auto& cache = KernelCache::instance();

    // Create a FusionGroup wrapper for the cache key
    FusionGroup wrapper;
    wrapper.signature = group.signature.empty()
        ? const_cast<ExtendedFusionGroup&>(group).compute_signature()
        : group.signature;
    wrapper.dtype = group.dtype;

    auto kernel = cache.get_or_compile(wrapper);
    if (!kernel) {
        throw std::runtime_error("Extended codegen: failed to compile kernel: " + wrapper.signature);
    }

    // Determine output shape based on fusion kind
    if (inputs.empty()) {
        throw std::runtime_error("Extended codegen: no inputs provided");
    }

    // Output shape depends on the fusion kind
    auto output_shape_span = inputs[0].shape();
    std::vector<int64_t> output_shape(output_shape_span.begin(), output_shape_span.end());
    int64_t numel = inputs[0].numel();

    // Allocate output
    Tensor output(output_shape, group.dtype, inputs[0].device());

    // Build input pointer list
    std::vector<const void*> input_ptrs;
    for (auto& t : inputs) input_ptrs.push_back(t.data_ptr());
    for (auto& t : params) input_ptrs.push_back(t.data_ptr());

    // Launch
    kernel->launch(input_ptrs, output.data_ptr(), numel, nullptr);

    return output;
}

} // namespace jit
} // namespace tenzor
