/**
 * @file extended_codegen.cpp
 * @brief Implementation of extended GPU kernel generation
 */

#include "tenzor/jit/extended_codegen.hpp"
#include "tenzor/jit/codegen.hpp"
#include "tenzor/ops/math.hpp"   // tenzor::matmul for the GEMM-epilogue fusion
#include <sstream>
#include <algorithm>

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
            ss << "_dim" << reduce_dim << "_rk" << static_cast<int>(reduce_kind)
               << "_pre" << pre_ops.size()
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

auto ExtendedKernelCodegen::compute_type(DType dtype) -> std::string {
    // Compute/accumulate in double only for Float64; Float32 and the 16-bit
    // types compute in float (the half types are promoted to float for math).
    return (dtype == DType::Float64) ? "double" : "float";
}

auto ExtendedKernelCodegen::literal_suffix(DType dtype) -> std::string {
    return (dtype == DType::Float64) ? "" : "f";
}

auto ExtendedKernelCodegen::fn_for(const std::string& base, DType dtype) -> std::string {
    // double overloads are unsuffixed (exp, sqrt, rsqrt, fabs); the float
    // variants take the 'f' suffix (expf, sqrtf, rsqrtf, fabsf).
    return (dtype == DType::Float64) ? base : base + "f";
}

auto ExtendedKernelCodegen::activation_expr(OpType act, const std::string& var,
                                            DType dtype) -> std::string {
    const std::string F = literal_suffix(dtype);
    const std::string tanh_fn = fn_for("tanh", dtype);
    const std::string exp_fn = fn_for("exp", dtype);
    switch (act) {
        case OpType::ReLU:    return var + " > 0 ? " + var + " : 0";
        case OpType::Sigmoid:
            return "1.0" + F + " / (1.0" + F + " + " + exp_fn + "(-" + var + "))";
        case OpType::Tanh:    return tanh_fn + "(" + var + ")";
        case OpType::GELU:
            return "0.5" + F + " * " + var + " * (1.0" + F + " + " + tanh_fn +
                   "(0.7978845608" + F + " * (" + var + " + 0.044715" + F + " * " +
                   var + " * " + var + " * " + var + ")))";
        default:
            return var;
    }
}

// ============================================================================
// Dispatch
// ============================================================================

auto ExtendedKernelCodegen::generate(const ExtendedFusionGroup& group) -> std::string {
    std::string body;
    switch (group.kind) {
        case FusionKind::Reduction:    body = generate_reduction(group); break;
        case FusionKind::GemmEpilogue: body = generate_gemm_epilogue(group); break;
        case FusionKind::Softmax:      body = generate_softmax(group); break;
        case FusionKind::LayerNorm:    body = generate_layer_norm(group); break;
        case FusionKind::RMSNorm:      body = generate_rms_norm(group); break;
        case FusionKind::SmallMLP:     body = generate_small_mlp(group); break;
        default:                       return "";
    }
    if (body.empty()) return "";
    // NVRTC/hiprtc has no system headers, and the extended kernel signatures use
    // int64_t (the element-wise codegen uses `long long`). Provide the typedefs
    // so the generated source compiles at runtime.
    static const char* kPreamble =
        "typedef long long int64_t;\n"
        "typedef unsigned long long uint64_t;\n";
    return std::string(kPreamble) + body;
}

// ============================================================================
// Reduction kernel: block-parallel with warp shuffles
// ============================================================================

auto ExtendedKernelCodegen::generate_reduction(const ExtendedFusionGroup& group) -> std::string {
    auto T = dtype_to_cuda_type(group.dtype);
    auto C = compute_type(group.dtype);          // accumulate in float/double
    const std::string F = literal_suffix(group.dtype);
    const std::string abs_fn = fn_for("fabs", group.dtype);
    const std::string exp_fn = fn_for("exp", group.dtype);
    const std::string sqrt_fn = fn_for("sqrt", group.dtype);
    const std::string max_fn = fn_for("fmax", group.dtype);
    const std::string min_fn = fn_for("fmin", group.dtype);
    std::ostringstream ss;

    // Reduction kind drives the accumulator identity, the per-element combine,
    // the warp/block tree-reduction step, and the finalization. Sum/Mean fold
    // with addition (Mean additionally divides by reduce_size); Max/Min fold
    // with fmax/fmin from a -inf/+inf identity.
    const bool is_max = (group.reduce_kind == OpType::Max);
    const bool is_min = (group.reduce_kind == OpType::Min);
    const bool is_mean = (group.reduce_kind == OpType::Mean);
    const std::string ident =
        is_max ? std::string("-1e30") + F :
        is_min ? std::string("1e30") + F  :
                 std::string("0");
    // combine(acc, x) -> updated acc, for the element loop and the tree steps.
    auto combine = [&](const std::string& acc, const std::string& x) -> std::string {
        if (is_max) return acc + " = " + max_fn + "(" + acc + ", " + x + ");";
        if (is_min) return acc + " = " + min_fn + "(" + acc + ", " + x + ");";
        return acc + " += " + x + ";";
    };

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

    // Each block reduces one (outer, inner) slice. Accumulate in the compute
    // type (float for the 16-bit storage types) and narrow to T only on store.
    )" << C << R"( sum = )" << ident << R"(;
    for (int r = threadIdx.x; r < reduce_size; r += blockDim.x) {
        )" << C << R"( val = static_cast<)" << C << R"(>(input[outer * reduce_size * inner_size + r * inner_size + inner]);
)";

    // Inline pre-reduction element-wise ops
    for (auto& op : group.pre_ops) {
        switch (op.op) {
            case ElemOp::Mul:    ss << "        val = val * val;\n"; break;
            case ElemOp::Abs:    ss << "        val = " << abs_fn << "(val);\n"; break;
            case ElemOp::Exp:    ss << "        val = " << exp_fn << "(val);\n"; break;
            case ElemOp::Neg:    ss << "        val = -val;\n"; break;
            case ElemOp::MulScalar:
                ss << "        val = val * " << op.scalar << F << ";\n"; break;
            default: break;
        }
    }

    ss << "        " << combine("sum", "val") << "\n";
    ss << R"(    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
)";
    ss << "        " << combine("sum", "__shfl_down_sync(0xffffffff, sum, offset)") << "\n";
    ss << R"(    }

    // Block-level reduction via shared memory
    __shared__ )" << C << R"( shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) shared[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < blockDim.x / warpSize) ? shared[lane] : )" << ident << R"(;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
)";
    ss << "            " << combine("sum", "__shfl_down_sync(0xffffffff, sum, offset)") << "\n";
    ss << R"(        }
    }

    if (threadIdx.x == 0) {
        )" << C << R"( result = sum;
)";
    // Mean finalizes the summed accumulator by dividing by the element count.
    if (is_mean) {
        ss << "        result = result / static_cast<" << C << ">(reduce_size);\n";
    }

    // Inline post-reduction element-wise ops
    for (auto& op : group.post_ops) {
        switch (op.op) {
            case ElemOp::Sqrt:       ss << "        result = " << sqrt_fn << "(result);\n"; break;
            case ElemOp::Reciprocal: ss << "        result = 1.0" << F << " / result;\n"; break;
            case ElemOp::MulScalar:
                ss << "        result = result * " << op.scalar << F << ";\n"; break;
            case ElemOp::AddScalar:
                ss << "        result = result + " << op.scalar << F << ";\n"; break;
            default: break;
        }
    }

    ss << "        output[outer * inner_size + inner] = static_cast<" << T << ">(result);\n";
    ss << R"(    }
}
)";

    return ss.str();
}

// ============================================================================
// GEMM epilogue: bias + activation fused with matmul result
// ============================================================================

auto ExtendedKernelCodegen::generate_gemm_epilogue(const ExtendedFusionGroup& group) -> std::string {
    auto T = dtype_to_cuda_type(group.dtype);
    auto C = compute_type(group.dtype);  // float for f32/f16/bf16, double for f64
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
    // Promote to the compute type so bias add + activation run in float/double
    // (correct for the 16-bit storage types), then narrow back to T on store.
    )" << C << " val = static_cast<" << C << ">(output[idx]);\n";

    if (group.has_bias) {
        ss << "    val = val + static_cast<" << C << ">(bias[col]);\n";
    }

    if (group.has_activation) {
        ss << "    val = " << activation_expr(group.activation_type, "val", group.dtype) << ";\n";
    }

    ss << "    output[idx] = static_cast<" << T << ">(val);\n";
    ss << R"(}
)";

    return ss.str();
}

// ============================================================================
// Softmax: online 2-pass fused per-row
// ============================================================================

auto ExtendedKernelCodegen::generate_softmax(const ExtendedFusionGroup& group) -> std::string {
    auto T = dtype_to_cuda_type(group.dtype);
    auto C = compute_type(group.dtype);          // reductions/exp in float/double
    const std::string F = literal_suffix(group.dtype);
    const std::string exp_fn = fn_for("exp", group.dtype);
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

    // Pass 1: find max (for numerical stability). All reductions run in the
    // compute type (float for the 16-bit storage types); only loads/stores touch
    // T.
    )" << C << R"( thread_max = -1e30)" << F << R"(;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        )" << C << R"( val = static_cast<)" << C << R"(>(row_input[c]);
        thread_max = val > thread_max ? val : thread_max;
    }

    // Warp reduction for max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        )" << C << R"( other = __shfl_down_sync(0xffffffff, thread_max, offset);
        thread_max = other > thread_max ? other : thread_max;
    }

    __shared__ )" << C << R"( shared_max[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) shared_max[warp_id] = thread_max;
    __syncthreads();

    if (warp_id == 0) {
        thread_max = (lane < blockDim.x / warpSize) ? shared_max[lane] : -1e30)" << F << R"(;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            )" << C << R"( other = __shfl_down_sync(0xffffffff, thread_max, offset);
            thread_max = other > thread_max ? other : thread_max;
        }
        if (lane == 0) shared_max[0] = thread_max;
    }
    __syncthreads();
    )" << C << R"( row_max = shared_max[0];

    // Pass 2: compute exp(x - max) and sum
    )" << C << R"( thread_sum = 0;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        )" << C << R"( val = )" << exp_fn << R"((static_cast<)" << C << R"(>(row_input[c]) - row_max);
        row_output[c] = static_cast<)" << T << R"(>(val);  // Store intermediate exp values
        thread_sum += val;
    }

    // Warp reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    __shared__ )" << C << R"( shared_sum[32];
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
    )" << C << R"( inv_sum = 1.0)" << F << R"( / shared_sum[0];

    // Pass 3: normalize
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        row_output[c] = static_cast<)" << T << R"(>(static_cast<)" << C << R"(>(row_output[c]) * inv_sum);
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
    auto C = compute_type(group.dtype);  // Welford state in float/double
    const std::string rsqrt_fn = fn_for("rsqrt", group.dtype);
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

    // Welford online mean computation (accumulate in compute type; the 16-bit
    // storage types are promoted to float for the running mean/variance).
    )" << C << R"( mean = 0;
    )" << C << R"( m2 = 0;
    int count = 0;

    for (int i = threadIdx.x; i < norm_size; i += blockDim.x) {
        count++;
        )" << C << R"( xi = static_cast<)" << C << R"(>(x[i]);
        )" << C << R"( delta = xi - mean;
        mean += delta / count;
        )" << C << R"( delta2 = xi - mean;
        m2 += delta * delta2;
    }

    // Warp-level Welford merge
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        )" << C << R"( other_mean = __shfl_down_sync(0xffffffff, mean, offset);
        )" << C << R"( other_m2 = __shfl_down_sync(0xffffffff, m2, offset);
        int other_count = __shfl_down_sync(0xffffffff, count, offset);

        int total = count + other_count;
        if (total > 0) {
            )" << C << R"( delta = other_mean - mean;
            mean = (count * mean + other_count * other_mean) / total;
            m2 = m2 + other_m2 + delta * delta * count * other_count / total;
            count = total;
        }
    }

    // Block-level merge via shared memory
    __shared__ )" << C << R"( s_mean[32], s_m2[32];
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
            )" << C << R"( other_mean = __shfl_down_sync(0xffffffff, mean, offset);
            )" << C << R"( other_m2 = __shfl_down_sync(0xffffffff, m2, offset);
            int other_count = __shfl_down_sync(0xffffffff, count, offset);

            int total = count + other_count;
            if (total > 0) {
                )" << C << R"( delta = other_mean - mean;
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

    )" << C << R"( final_mean = s_mean[0];
    )" << C << R"( inv_std = )" << rsqrt_fn << R"((s_m2[0] + eps);

    // Normalize + affine
    for (int i = threadIdx.x; i < norm_size; i += blockDim.x) {
        )" << C << R"( val = (static_cast<)" << C << R"(>(x[i]) - final_mean) * inv_std;)";

    if (group.has_affine) {
        ss << R"(
        val = val * static_cast<)" << C << R"(>(gamma[i]) + static_cast<)" << C << R"(>(beta[i]);)";
    }

    ss << R"(
        y[i] = static_cast<)" << T << R"(>(val);
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
    auto C = compute_type(group.dtype);  // sum-of-squares in float/double
    const std::string rsqrt_fn = fn_for("rsqrt", group.dtype);
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

    // Compute sum of squares (accumulate in compute type; 16-bit storage types
    // are promoted to float for the reduction).
    )" << C << R"( sum_sq = 0;
    for (int i = threadIdx.x; i < norm_size; i += blockDim.x) {
        )" << C << R"( val = static_cast<)" << C << R"(>(x[i]);
        sum_sq += val * val;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ )" << C << R"( shared[32];
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

    )" << C << R"( rms_inv = )" << rsqrt_fn << R"((shared[0] / norm_size + eps);

    // Normalize
    for (int i = threadIdx.x; i < norm_size; i += blockDim.x) {
        )" << C << R"( val = static_cast<)" << C << R"(>(x[i]) * rms_inv;)";

    if (group.has_affine) {
        ss << R"(
        val = val * static_cast<)" << C << R"(>(gamma[i]);)";
    }

    ss << R"(
        y[i] = static_cast<)" << T << R"(>(val);
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
    auto C = compute_type(group.dtype);  // GEMM accumulation in float/double
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

    // Stage 1: hidden = activation(x @ W1 + b1). Accumulate the dot product in
    // the compute type (float for the 16-bit storage types) and narrow to T when
    // writing the shared-memory intermediate (whose layout is sized in T).
    for (int h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        )" << C << R"( acc = static_cast<)" << C << R"(>(b1[h]);
        for (int i = 0; i < in_dim; ++i) {
            acc += static_cast<)" << C << R"(>(x[i]) * static_cast<)" << C << R"(>(w1[i * hidden_dim + h]);
        }
        hidden[h] = static_cast<)" << T << R"(>()" << activation_expr(group.mlp_activation, "acc", group.dtype) << R"();
    }
    __syncthreads();

    // Stage 2: output = hidden @ W2 + b2
    )" << T << R"(* out = output + b * out_dim;
    for (int o = threadIdx.x; o < out_dim; o += blockDim.x) {
        )" << C << R"( acc = static_cast<)" << C << R"(>(b2[o]);
        for (int h = 0; h < hidden_dim; ++h) {
            acc += static_cast<)" << C << R"(>(hidden[h]) * static_cast<)" << C << R"(>(w2[h * out_dim + o]);
        }
        out[o] = static_cast<)" << T << R"(>(acc);
    }
}
)";

    return ss.str();
}

// ============================================================================
// Extended kernel execution
// ============================================================================

namespace {

// Entry-point symbol name emitted by each generate_* method, keyed by kind.
// Must stay in sync with the extern "C" __global__ names in the generators.
auto extended_kernel_name(FusionKind kind) -> std::string {
    switch (kind) {
        case FusionKind::Reduction:    return "fused_reduction_kernel";
        case FusionKind::GemmEpilogue: return "fused_gemm_epilogue_kernel";
        case FusionKind::Softmax:      return "fused_softmax_kernel";
        case FusionKind::LayerNorm:    return "fused_layer_norm_kernel";
        case FusionKind::RMSNorm:      return "fused_rms_norm_kernel";
        case FusionKind::SmallMLP:     return "fused_small_mlp_kernel";
    }
    return "";
}

} // namespace

auto execute_extended_fused(const ExtendedFusionGroup& group,
                            const std::vector<Tensor>& inputs,
                            const std::vector<Tensor>& params) -> Tensor {
    // Generate source code
    auto source = ExtendedKernelCodegen::generate(group);
    if (source.empty()) {
        throw std::runtime_error("Extended codegen: unsupported fusion kind");
    }

    // Compile the EXTENDED kernel source we just generated, keyed by the
    // extended group's signature. Previously this delegated to
    // get_or_compile(FusionGroup), which regenerates and compiles an unrelated
    // element-wise kernel and discards `source` entirely.
    auto& cache = KernelCache::instance();

    std::string signature = group.signature.empty()
        ? const_cast<ExtendedFusionGroup&>(group).compute_signature()
        : group.signature;
    std::string kernel_name = extended_kernel_name(group.kind);
    if (kernel_name.empty()) {
        throw std::runtime_error("Extended codegen: unsupported fusion kind");
    }

    auto kernel = cache.get_or_compile_source(signature, source, kernel_name);
    if (!kernel) {
        throw std::runtime_error("Extended codegen: failed to compile kernel: " + signature);
    }

    // Determine output shape based on fusion kind
    if (inputs.empty()) {
        throw std::runtime_error("Extended codegen: no inputs provided");
    }

    // ---- Per-kind output shape, launch geometry, and argument layout ----
    // The extended kernels do NOT share the element-wise [inputs..., output,
    // numel] ABI: each has a distinct signature (GEMM-epilogue puts output
    // first; norms take outer/norm_size+eps; reduction takes outer/reduce/inner;
    // MLP takes weights/biases + four dims) and a distinct launch shape. Build
    // the exact argument list and grid/block for each FusionKind.
    auto in_shape = std::vector<int64_t>(inputs[0].shape().begin(), inputs[0].shape().end());
    const int64_t ndim = static_cast<int64_t>(in_shape.size());
    const int64_t total = inputs[0].numel();
    auto axis_norm = [&](int axis) -> int64_t {
        int64_t a = axis;
        if (a < 0) a += ndim;
        if (a < 0) a = 0;
        if (a >= ndim) a = (ndim > 0 ? ndim - 1 : 0);
        return a;
    };
    auto suffix_prod = [&](int64_t from) -> int64_t {
        int64_t p = 1; for (int64_t d = from; d < ndim; ++d) p *= in_shape[d]; return p;
    };

    constexpr int kBlock = 256;
    Tensor output;
    std::vector<const void*> ptrs;   // buffer args (fully populated before &-taken)
    std::vector<int64_t> ivals;      // integer scalar args (fully populated first)
    float eps_val = group.eps;
    bool has_eps = false;
    int grid = 1;
    unsigned shared = 0;

    switch (group.kind) {
        case FusionKind::Softmax: {
            int64_t cols = std::max<int64_t>(1, suffix_prod(axis_norm(group.softmax_dim)));
            int64_t rows = total / cols;
            output = Tensor(in_shape, group.dtype, inputs[0].device());
            ptrs = {inputs[0].data_ptr(), output.data_ptr()};
            ivals = {rows, cols};
            grid = static_cast<int>(rows);
            break;
        }
        case FusionKind::LayerNorm:
        case FusionKind::RMSNorm: {
            int64_t norm_size = std::max<int64_t>(1, suffix_prod(axis_norm(group.norm_axis)));
            int64_t outer = total / norm_size;
            output = Tensor(in_shape, group.dtype, inputs[0].device());
            ptrs = {inputs[0].data_ptr(), output.data_ptr()};
            if (group.has_affine) {
                ptrs.push_back(params.at(0).data_ptr());           // gamma
                if (group.kind == FusionKind::LayerNorm) {
                    ptrs.push_back(params.at(1).data_ptr());       // beta
                }
            }
            ivals = {outer, norm_size};
            has_eps = true;
            grid = static_cast<int>(outer);
            break;
        }
        case FusionKind::Reduction: {
            int64_t a = axis_norm(group.reduce_dim);
            int64_t reduce_size = (ndim > 0) ? in_shape[a] : total;
            int64_t outer = 1; for (int64_t d = 0; d < a; ++d) outer *= in_shape[d];
            int64_t inner = suffix_prod(a + 1);
            std::vector<int64_t> out_shape;
            for (int64_t d = 0; d < ndim; ++d) {
                if (d == a) { if (group.keepdim) out_shape.push_back(1); }
                else out_shape.push_back(in_shape[d]);
            }
            output = Tensor(out_shape, group.dtype, inputs[0].device());
            ptrs = {inputs[0].data_ptr(), output.data_ptr()};
            ivals = {outer, reduce_size, inner};
            grid = static_cast<int>(outer * inner);
            break;
        }
        case FusionKind::GemmEpilogue: {
            // The epilogue kernel reads/writes a buffer that ALREADY holds the
            // GEMM result (it only adds bias + applies the activation). So we
            // must compute A @ B here and hand the product to the kernel as the
            // output buffer; the kernel then fuses bias/activation in place.
            // inputs[0] = A (lhs), inputs[1] = B (rhs); bias (if any) is a param.
            if (inputs.size() < 2) {
                throw std::runtime_error(
                    "execute_extended_fused: GemmEpilogue requires two matmul inputs (A, B)");
            }
            output = tenzor::matmul(inputs[0], inputs[1]);
            // Geometry is derived from the GEMM RESULT, not from inputs[0]:
            // the kernel grids over rows*cols of the product and indexes bias by
            // the last (column) dimension.
            auto out_shape = output.shape();
            const int64_t out_ndim = static_cast<int64_t>(out_shape.size());
            const int64_t out_total = output.numel();
            int64_t cols = std::max<int64_t>(1, out_ndim > 0 ? out_shape[out_ndim - 1] : out_total);
            int64_t rows = out_total / cols;
            ptrs = {output.data_ptr()};                            // output FIRST
            if (group.has_bias) ptrs.push_back(params.at(0).data_ptr());
            ivals = {rows, cols};
            grid = static_cast<int>((rows * cols + kBlock - 1) / kBlock);
            break;
        }
        case FusionKind::SmallMLP: {
            // params: {w1, b1, w2, b2}
            int64_t in_dim = std::max<int64_t>(1, ndim > 0 ? in_shape[ndim - 1] : total);
            int64_t batch = total / in_dim;
            int64_t hidden_dim = group.hidden_dim;
            int64_t out_dim = static_cast<int64_t>(params.at(3).numel());  // b2 = [out_dim]
            output = Tensor({batch, out_dim}, group.dtype, inputs[0].device());
            ptrs = {inputs[0].data_ptr(), params.at(0).data_ptr(), params.at(1).data_ptr(),
                    params.at(2).data_ptr(), params.at(3).data_ptr(), output.data_ptr()};
            ivals = {batch, in_dim, hidden_dim, out_dim};
            grid = static_cast<int>(batch);
            shared = static_cast<unsigned>(hidden_dim * static_cast<int64_t>(inputs[0].dtype_size()));
            break;
        }
        default:
            throw std::runtime_error("execute_extended_fused: unsupported fusion kind");
    }

    // Build the kernel argument array (pointers to the stable ptr/dim values),
    // in the order each kernel declares: buffers..., dims..., [eps].
    std::vector<void*> args;
    args.reserve(ptrs.size() + ivals.size() + 1);
    for (auto& p : ptrs) args.push_back(static_cast<void*>(&p));
    for (auto& v : ivals) args.push_back(static_cast<void*>(&v));
    if (has_eps) args.push_back(static_cast<void*>(&eps_val));

    kernel->launch_raw(args, grid, kBlock, shared, nullptr);

    return output;
}

} // namespace jit
} // namespace tenzor
