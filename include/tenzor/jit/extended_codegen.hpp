/**
 * @file extended_codegen.hpp
 * @brief Extended GPU kernel generation beyond element-wise fusion
 *
 * Generates fused CUDA/HIP kernels for:
 * - Reduction with pre/post element-wise ops (warp shuffle)
 * - GEMM epilogues (cuBLASLt callbacks or custom kernels)
 * - Softmax (online 2-pass, fused per-row)
 * - LayerNorm/RMSNorm (Welford online, single-kernel)
 * - Small MLPs (shared-memory tiled GEMM with fused activation)
 *
 * Integrates with the existing KernelCache for compilation/caching.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "../core/dtype.hpp"
#include "../core/tensor.hpp"
#include "codegen.hpp"
#include "pattern_matcher.hpp"

namespace tenzor {
namespace jit {

// ============================================================================
// Extended fusion group (carries richer metadata than FusionGroup)
// ============================================================================

/**
 * @brief Describes a fused computation beyond element-wise chains.
 *
 * Each FusionKind has specific fields populated; unused fields are zero/empty.
 */
struct ExtendedFusionGroup {
    FusionKind kind;
    DType dtype{DType::Float32};
    std::string signature;

    // --- Reduction fields ---
    int reduce_dim{-1};                  ///< Dimension to reduce over
    bool keepdim{false};
    /// Reduction operator the fused kernel must finalize as. Sum is the raw
    /// accumulator; Mean additionally divides by reduce_size; Max/Min use a
    /// max/min accumulation+reduction instead of a sum.
    OpType reduce_kind{OpType::Sum};
    std::vector<ElemStep> pre_ops;       ///< Element-wise ops before reduction
    std::vector<ElemStep> post_ops;      ///< Element-wise ops after reduction
    int num_inputs{1};

    // --- GEMM epilogue fields ---
    bool has_bias{false};
    bool has_activation{false};
    OpType activation_type{OpType::ReLU};  ///< ReLU, GELU, Sigmoid, Tanh

    // --- Softmax fields ---
    int softmax_dim{-1};                 ///< Dimension for softmax (typically last)

    // --- LayerNorm / RMSNorm fields ---
    int norm_axis{-1};                   ///< Normalization axis
    // double (not float) so a Float64 LayerNorm/RMSNorm receives a full-precision
    // eps in the `var + eps` denominator, matching the eager F64 kernel
    // (JIT-F014). The F32 kernel path narrows it back to float at marshal time.
    double eps{1e-5};
    bool has_affine{true};               ///< Has gamma/beta parameters

    // --- Small MLP fields ---
    int64_t hidden_dim{0};               ///< Hidden layer dimension
    OpType mlp_activation{OpType::GELU}; ///< Activation between linear layers

    /// Compute a cache key from the fusion metadata.
    auto compute_signature() -> std::string;
};

// ============================================================================
// Extended kernel code generation
// ============================================================================

/**
 * @brief Generates CUDA/HIP source code for extended fusion patterns.
 *
 * Each generate_* method returns complete kernel source code ready
 * for NVRTC/HIPRTC compilation.
 */
class ExtendedKernelCodegen {
public:
    /**
     * @brief Generate kernel for any extended fusion group.
     *
     * Dispatches to the appropriate generate_* method based on kind.
     *
     * @param group Extended fusion group description
     * @return CUDA/HIP kernel source code
     */
    static auto generate(const ExtendedFusionGroup& group) -> std::string;

    /**
     * @brief Generate a reduction kernel with optional pre/post ops.
     *
     * Implements block-parallel reduction with warp shuffles.
     * Pre-ops are applied before accumulation, post-ops after.
     *
     * @param group Must have kind == Reduction
     * @return Kernel source
     */
    static auto generate_reduction(const ExtendedFusionGroup& group) -> std::string;

    /**
     * @brief Generate a GEMM epilogue kernel.
     *
     * For cuBLASLt-capable devices, returns a descriptor for epilogue
     * configuration. For others, generates a custom kernel.
     *
     * @param group Must have kind == GemmEpilogue
     * @return Kernel source
     */
    static auto generate_gemm_epilogue(const ExtendedFusionGroup& group) -> std::string;

    /**
     * @brief Generate a fused softmax kernel.
     *
     * Online 2-pass algorithm: first pass computes max, second pass
     * computes exp(x-max)/sum. All fused into a single kernel per row.
     *
     * @param group Must have kind == Softmax
     * @return Kernel source
     */
    static auto generate_softmax(const ExtendedFusionGroup& group) -> std::string;

    /**
     * @brief Generate a fused LayerNorm kernel.
     *
     * Uses Welford's online algorithm for numerically stable
     * mean+variance computation in a single pass.
     *
     * @param group Must have kind == LayerNorm
     * @return Kernel source
     */
    static auto generate_layer_norm(const ExtendedFusionGroup& group) -> std::string;

    /**
     * @brief Generate a fused RMSNorm kernel.
     *
     * Computes root mean square and normalizes in a single kernel.
     *
     * @param group Must have kind == RMSNorm
     * @return Kernel source
     */
    static auto generate_rms_norm(const ExtendedFusionGroup& group) -> std::string;

    /**
     * @brief Generate a fused small MLP kernel.
     *
     * Shared-memory tiled GEMM with fused inter-layer activation.
     * Only profitable for hidden dim <= 4096.
     *
     * @param group Must have kind == SmallMLP
     * @return Kernel source
     */
    static auto generate_small_mlp(const ExtendedFusionGroup& group) -> std::string;

private:
    static auto dtype_to_cuda_type(DType dtype) -> std::string;
    static auto activation_expr(OpType act, const std::string& var, DType dtype)
        -> std::string;

    /// In-kernel math/accumulation type for a storage dtype. Float64 computes in
    /// double; Float32 and the 16-bit types (Float16/BFloat16) compute in float
    /// — half-precision accumulation is numerically wrong and the float math
    /// intrinsics don't apply to __half/__nv_bfloat16 directly, so values are
    /// promoted to float for the math and narrowed back to T on store.
    static auto compute_type(DType dtype) -> std::string;

    /// Float-literal suffix matching compute_type(): "" for double, "f" for
    /// float (an 'f'-suffixed literal in a double kernel silently rounds to
    /// single precision; mirrors KernelCodegen::emit_op).
    static auto literal_suffix(DType dtype) -> std::string;

    /// Math-intrinsic name for compute_type(): e.g. fn_for("exp", dt) returns
    /// "exp" for double and "expf" for float.
    static auto fn_for(const std::string& base, DType dtype) -> std::string;

    /// Emit a scalar operand as a valid floating literal at FULL round-trip
    /// (max_digits10) precision plus the dtype's literal suffix. A plain
    /// `ss << scalar << F` uses the default ostream precision (~6 digits), so a
    /// Float64 kernel bakes in a single-precision-rounded constant and diverges
    /// from the eager/CPU result. Mirrors KernelCodegen::emit_op's fmt_double.
    static auto fmt_scalar(double v, DType dtype) -> std::string;
};

// ============================================================================
// Extended kernel execution
// ============================================================================

/**
 * @brief Execute a fused extended kernel on GPU tensors.
 *
 * Compiles (or retrieves cached) the kernel for the given group,
 * then launches it with the provided inputs.
 *
 * @param group Extended fusion group description
 * @param inputs Input tensors (varies by fusion kind)
 * @param params Additional parameter tensors (weights, biases, gamma, beta)
 * @return Output tensor
 */
auto execute_extended_fused(const ExtendedFusionGroup& group,
                            const std::vector<Tensor>& inputs,
                            const std::vector<Tensor>& params = {}) -> Tensor;

/**
 * @brief Total number of native fused GPU kernels launched.
 *
 * Incremented once per successful execute_extended_fused() launch (across all
 * FusionKinds and both NVRTC/HIPRTC backends). The JIT graph executor calls the
 * codegen path for fusion nodes, so a monotonically increasing count is the
 * observable proof that a fused node actually ran on the GPU codegen path rather
 * than silently falling back to eager ops. Tests read this to fail closed.
 */
auto extended_fused_launch_count() -> uint64_t;

} // namespace jit
} // namespace tenzor
