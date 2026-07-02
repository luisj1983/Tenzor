/**
 * @file compiler.hpp
 * @brief Graph optimization passes for JIT compilation
 *
 * Provides a suite of optimization passes that transform IR graphs
 * for improved performance. Includes:
 * - Operator fusion (Conv+BatchNorm, Conv+ReLU, etc.)
 * - Dead code elimination (DCE)
 * - Common subexpression elimination (CSE)
 * - Constant folding
 * - Algebraic simplification
 * - Memory planning (buffer reuse for intermediate values)
 */

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include "graph.hpp"
#include "fusion_cost_model.hpp"
#include "memory_planner.hpp"
#include "pattern_matcher.hpp"
#include "symbolic_shape_inference.hpp"
#include "../nn/module.hpp"
#include "../backend/cuda_graph.hpp"

namespace tenzor {
namespace jit {

/**
 * @brief Base class for optimization passes.
 *
 * All graph transformations inherit from this class and implement
 * the run() method to modify the graph in-place.
 */
class Pass {
public:
    virtual ~Pass() = default;

    /**
     * @brief Run optimization pass on graph.
     *
     * @param graph Graph to optimize (modified in-place)
     * @return true if graph was modified
     */
    virtual auto run(Graph& graph) -> bool = 0;

    /**
     * @brief Get pass name for logging.
     *
     * @return Pass identifier
     */
    virtual auto name() const -> std::string = 0;
};

/**
 * @brief Dead code elimination pass.
 *
 * Removes nodes that don't contribute to any graph output.
 * Works backwards from outputs, marking all reachable nodes,
 * then removes unreachable ones.
 *
 * Example:
 * @code
 * x = input
 * y = relu(x)
 * z = sigmoid(x)  # Dead if not used in output
 * output = y
 * @endcode
 */
class DeadCodeEliminationPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "DeadCodeElimination"; }

private:
    /**
     * @brief Mark all nodes reachable from outputs.
     *
     * @param graph Graph to analyze
     * @return Set of reachable node pointers
     */
    auto mark_reachable_nodes(const Graph& graph) -> std::unordered_set<Node*>;
};

/**
 * @brief Common subexpression elimination pass.
 *
 * Detects and merges duplicate computations. If two nodes have:
 * - Same operation type
 * - Same inputs
 * - Same attributes
 * Then they can be merged into one.
 *
 * Example:
 * @code
 * a = relu(x)
 * b = relu(x)  # Duplicate - can reuse 'a'
 * c = a + b    # Becomes: c = a + a
 * @endcode
 */
class CommonSubexpressionEliminationPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "CommonSubexpressionElimination"; }

private:
    /**
     * @brief Compute hash for a node (based on type, inputs, attrs).
     *
     * @param node Node to hash
     * @return Hash value
     */
    auto compute_node_hash(const Node& node) -> size_t;

    /**
     * @brief Check if two nodes are equivalent.
     *
     * @param a First node
     * @param b Second node
     * @return true if nodes can be merged
     */
    auto nodes_equivalent(const Node& a, const Node& b) -> bool;
};

/**
 * @brief Constant folding pass.
 *
 * Evaluates operations on constant inputs at compile time.
 * Replaces the operation with a constant node.
 *
 * Example:
 * @code
 * a = Constant(2.0)
 * b = Constant(3.0)
 * c = a + b           # Can be folded to Constant(5.0)
 * d = c * x           # Becomes: Constant(5.0) * x
 * @endcode
 */
class ConstantFoldingPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "ConstantFolding"; }

private:
    /**
     * @brief Check if all inputs to node are constants.
     *
     * @param node Node to check
     * @return true if node can be folded
     */
    auto can_fold(const Node& node) -> bool;

    /**
     * @brief Evaluate node with constant inputs.
     *
     * @param node Node to evaluate
     * @return Computed constant value
     */
    auto evaluate_constant(const Node& node) -> Tensor;
};

/**
 * @brief Operator fusion pass - Conv2d + BatchNorm2d.
 *
 * Fuses convolution and batch normalization into a single operation
 * by folding the batch norm parameters into the conv weights and biases.
 *
 * Formula:
 * @code
 * y = gamma * (conv(x) - mean) / sqrt(var + eps) + beta
 * Can be rewritten as:
 * y = conv(x, w', b') where:
 *   w' = gamma * w / sqrt(var + eps)
 *   b' = gamma * (b - mean) / sqrt(var + eps) + beta
 * @endcode
 *
 * Speedup: ~10-20% for inference
 */
class FuseConvBatchNormPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "FuseConvBatchNorm"; }

private:
    /**
     * @brief Fuse a conv+bn pair.
     *
     * @param conv_node Convolution node
     * @param bn_node Batch norm node
     * @param graph Graph containing nodes
     * @return true if fusion succeeded
     */
    auto fuse_pair(std::shared_ptr<Node> conv_node,
                   std::shared_ptr<Node> bn_node,
                   Graph& graph) -> bool;
};

/**
 * @brief Operator fusion pass - Conv2d + ReLU.
 *
 * Fuses convolution and ReLU activation into a single kernel.
 * Many backends (CUDA, oneDNN) have optimized fused kernels.
 *
 * Example:
 * @code
 * y = conv(x)
 * z = relu(y)
 * # Becomes:
 * z = conv_relu(x)
 * @endcode
 *
 * Speedup: ~5-15% for inference
 */
class FuseConvReluPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "FuseConvReLU"; }

private:
    auto fuse_pair(std::shared_ptr<Node> conv_node,
                   std::shared_ptr<Node> relu_node,
                   Graph& graph) -> bool;
};

/**
 * @brief Operator fusion pass - Linear + ReLU.
 *
 * Fuses fully-connected layer and ReLU activation.
 *
 * Example:
 * @code
 * y = linear(x)
 * z = relu(y)
 * # Becomes:
 * z = linear_relu(x)
 * @endcode
 */
class FuseLinearReluPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "FuseLinearReLU"; }

private:
    auto fuse_pair(std::shared_ptr<Node> linear_node,
                   std::shared_ptr<Node> relu_node,
                   Graph& graph) -> bool;
};

/**
 * @brief Operator fusion pass - MatMul + Add (bias pattern).
 *
 * Detects sequential MatMul(A, B) followed by Add(result, bias) where
 * bias is a 1D tensor, and fuses them into a single MatMul with a
 * fused_bias attribute. This avoids a separate element-wise kernel launch.
 *
 * Example:
 * @code
 * y = matmul(x, w)
 * z = y + bias       // bias is 1D
 * # Becomes:
 * z = matmul(x, w)   // with fused_bias = bias tensor
 * @endcode
 *
 * Speedup: ~5-10% for inference (eliminates separate Add kernel)
 */
class FuseMatMulAddPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "FuseMatMulAdd"; }

private:
    auto fuse_pair(std::shared_ptr<Node> matmul_node,
                   std::shared_ptr<Node> add_node,
                   Graph& graph) -> bool;
};

/**
 * @brief Operator fusion pass - Conv2d + BatchNorm2d + ReLU (triple fusion).
 *
 * Extends Conv+BN and Conv+ReLU patterns into a single triple fusion.
 * Detects Conv2d -> BatchNorm2d -> ReLU sequences and folds BN parameters
 * into convolution weights while also marking the fused ReLU activation.
 *
 * Example:
 * @code
 * y = conv2d(x, w, b)
 * z = batchnorm(y, gamma, beta, mean, var)
 * out = relu(z)
 * # Becomes:
 * out = conv2d(x, w', b')  // with fused_bn=true, fused_relu=true
 * @endcode
 *
 * Speedup: ~15-25% for inference (single kernel instead of three)
 */
class FuseConvBatchNormReluPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "FuseConvBatchNormReLU"; }

private:
    auto fuse_triple(std::shared_ptr<Node> conv_node,
                     std::shared_ptr<Node> bn_node,
                     std::shared_ptr<Node> relu_node,
                     Graph& graph) -> bool;
};

/**
 * @brief Operator fusion pass - LayerNorm + Activation.
 *
 * Fuses LayerNorm followed by ReLU or GELU into a single operation.
 * Many backends can apply the activation within the normalization kernel,
 * avoiding a separate memory pass.
 *
 * Example:
 * @code
 * y = layer_norm(x, ...)
 * z = gelu(y)
 * # Becomes:
 * z = layer_norm(x, ...)  // with fused_activation="gelu"
 * @endcode
 *
 * Speedup: ~5-15% for transformer inference
 */
class FuseLayerNormActivationPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "FuseLayerNormActivation"; }

private:
    auto fuse_pair(std::shared_ptr<Node> ln_node,
                   std::shared_ptr<Node> act_node,
                   Graph& graph) -> bool;
};

/**
 * @brief Operator fusion pass - Flash Attention.
 *
 * Detects the multi-head attention pattern:
 *   MatMul(Q, K^T) -> Scale -> (optional Mask + Add) -> Softmax -> MatMul(attn, V)
 * and fuses the entire sequence into a single FlashAttention node.
 *
 * Requirements:
 * - Q, K, V must be 3D (batch, seq, dim) or 4D (batch, heads, seq, dim)
 * - The Scale must be a scalar multiply (typically 1/sqrt(d_k))
 * - The MatMul(Q, K^T) output must not be used by other nodes (besides the chain)
 *
 * Example:
 * @code
 * attn_weights = matmul(Q, transpose(K))
 * attn_weights = attn_weights * scale
 * attn_weights = softmax(attn_weights)
 * output = matmul(attn_weights, V)
 * # Becomes:
 * output = flash_attention(Q, K, V, scale)
 * @endcode
 *
 * Speedup: ~2-4x for long sequences (reduced memory traffic)
 */
class FuseAttentionPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "FuseAttention"; }

private:
    /**
     * @brief Validate that tensors have compatible attention shapes.
     *
     * @param q_shape Q tensor shape
     * @param k_shape K tensor shape
     * @param v_shape V tensor shape
     * @return true if shapes are valid for attention
     */
    auto validate_attention_shapes(const std::vector<int64_t>& q_shape,
                                    const std::vector<int64_t>& k_shape,
                                    const std::vector<int64_t>& v_shape) -> bool;

    /**
     * @brief Fuse a matched attention pattern into a FlashAttention node.
     *
     * @param qk_matmul The MatMul(Q, K^T) node
     * @param scale_node The Scale (Mul) node
     * @param softmax_node The Softmax node
     * @param av_matmul The MatMul(attn, V) node
     * @param mask_add_node Optional mask Add node (nullptr if no mask)
     * @param graph Graph containing nodes
     * @return true if fusion succeeded
     */
    auto fuse_attention(std::shared_ptr<Node> qk_matmul,
                        std::shared_ptr<Node> scale_node,
                        std::shared_ptr<Node> softmax_node,
                        std::shared_ptr<Node> av_matmul,
                        std::shared_ptr<Node> mask_add_node,
                        Graph& graph) -> bool;
};

/**
 * @brief Operator fusion pass - Residual Add.
 *
 * Detects residual connection patterns where a value is added to the
 * output of a sublayer applied to that same value: x + sublayer(x).
 * Marks the Add node as a residual connection, enabling memory optimizations
 * such as in-place addition.
 *
 * Example:
 * @code
 * y = layer_norm(x)
 * z = linear(y)
 * out = x + z      // Residual connection
 * # Becomes:
 * out = x + z      // with residual=true attribute
 * @endcode
 */
class FuseResidualAddPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "FuseResidualAdd"; }

private:
    /**
     * @brief Check if a node is a recognized sublayer operation.
     *
     * Recognized sublayers include Linear, Conv2d, LayerNorm, BatchNorm2d,
     * MatMul, and any node with fused attributes.
     *
     * @param node Node to check
     * @return true if node is a sublayer
     */
    auto is_sublayer_op(const Node& node) -> bool;

    /**
     * @brief Check if a value is reachable from another value through a chain.
     *
     * Traces backward from target_value through producing nodes to see
     * if source_value appears as an input to any node in the chain.
     *
     * @param source_value The original input value ID
     * @param target_value The sublayer output value
     * @param max_depth Maximum chain depth to search
     * @return true if source feeds into the chain producing target
     */
    auto value_feeds_into(const std::string& source_value,
                          const std::shared_ptr<Value>& target_value,
                          int max_depth) -> bool;
};

/**
 * @brief Operator fusion pass - Feed-Forward Network.
 *
 * Detects the FFN pattern commonly found in transformers:
 *   Linear -> GELU/ReLU -> Linear
 * and fuses them into a single FusedFFN node.
 *
 * Example:
 * @code
 * h = linear1(x)
 * h = gelu(h)
 * y = linear2(h)
 * # Becomes:
 * y = fused_ffn(x)  // with weights from linear1, linear2
 * @endcode
 *
 * Speedup: ~10-20% for transformer inference (reduced memory traffic)
 */
class FuseFFNPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "FuseFFN"; }

private:
    /**
     * @brief Fuse a matched Linear -> Activation -> Linear pattern.
     *
     * @param linear1 First Linear node
     * @param act_node Activation node (GELU or ReLU)
     * @param linear2 Second Linear node
     * @param graph Graph containing nodes
     * @return true if fusion succeeded
     */
    auto fuse_triple(std::shared_ptr<Node> linear1,
                     std::shared_ptr<Node> act_node,
                     std::shared_ptr<Node> linear2,
                     Graph& graph) -> bool;
};

/**
 * @brief Shape guard insertion pass for dynamic shape support.
 *
 * Inserts ShapeGuard nodes at graph inputs that check tensor dimensions
 * at runtime. On a shape mismatch, a retrace flag is set so the caller
 * can re-trace the graph with the new shapes.
 *
 * Each ShapeGuard node stores the expected shape as a vector attribute
 * and produces its input unchanged as output (identity operation).
 *
 * Example:
 * @code
 * // Before:
 * input -> conv -> relu -> output
 * // After:
 * input -> ShapeGuard(expected=[1,3,224,224]) -> conv -> relu -> output
 * @endcode
 */
class ShapeGuardInsertionPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "ShapeGuardInsertion"; }

    /**
     * @brief Check if a retrace is needed (shape mismatch detected).
     *
     * @return true if runtime shapes did not match expected shapes
     */
    auto needs_retrace() const -> bool { return needs_retrace_; }

    /**
     * @brief Reset the retrace flag.
     */
    auto reset_retrace() -> void { needs_retrace_ = false; }

private:
    bool needs_retrace_{false};  ///< Set when a shape mismatch is detected
};

/**
 * @brief Symbolic shape tracing pass.
 *
 * Converts specified input dimensions from concrete to symbolic,
 * then propagates symbolic shapes through the graph using
 * SymbolicShapeInference.
 *
 * This pass enables dynamic shape support by marking certain
 * dimensions (e.g., batch size, sequence length) as symbolic
 * variables that can take different values at runtime.
 *
 * Example:
 * @code
 * SymbolicTracePass pass;
 * pass.mark_dynamic(0, 0, "batch");   // input 0, dim 0 is dynamic
 * pass.mark_dynamic(0, 1, "seq_len"); // input 0, dim 1 is dynamic
 * pass.run(graph);
 * // Now graph values have symbolic shapes with "batch" and "seq_len"
 * @endcode
 */
class SymbolicTracePass : public Pass {
public:
    /**
     * @brief Mark a dimension of an input as dynamic with a symbolic name.
     *
     * @param input_idx Index of the graph input (0-based)
     * @param dim Dimension index within the input's shape
     * @param name Symbolic name for this dimension (e.g., "batch")
     */
    auto mark_dynamic(int input_idx, int dim, const std::string& name) -> void;

    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "SymbolicTrace"; }

private:
    struct DynamicDim {
        int input_idx;
        int dim;
        std::string name;
    };
    std::vector<DynamicDim> dynamic_dims_;
};

/**
 * @brief Algebraic simplification pass.
 *
 * Applies algebraic identities to simplify expressions:
 * - x + 0 = x
 * - x * 1 = x
 * - x * 0 = 0
 * - x / 1 = x
 * - log(exp(x)) = x
 * - exp(log(x)) = x
 *
 * Example:
 * @code
 * y = x * 1.0    # Simplifies to: y = x
 * z = y + 0.0    # Simplifies to: z = y = x
 * @endcode
 */
class AlgebraicSimplificationPass : public Pass {
public:
    /// @param allow_unsafe_algebra Enable rewrites that change results for
    /// non-finite or out-of-domain inputs: exp(log x)=x (invalid for x<=0),
    /// log(exp x)=x (changes overflow to +Inf), and x*0=0 (IEEE-754 makes
    /// NaN*0 and Inf*0 yield NaN, not 0). These are gated OFF by default so a
    /// correctness-first compile preserves NaN/Inf/domain semantics, matching
    /// how XLA/PyTorch fence such transforms behind fast-math. The pure
    /// identity rewrites (x+0, x*1, x/1) are always applied — they are
    /// value-preserving for all inputs.
    explicit AlgebraicSimplificationPass(bool allow_unsafe_algebra = false)
        : allow_unsafe_algebra_(allow_unsafe_algebra) {}
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "AlgebraicSimplification"; }

private:
    bool allow_unsafe_algebra_ = false;

    /**
     * @brief Try to simplify a binary operation.
     *
     * @param node Operation node
     * @param graph Graph containing node
     * @return true if simplified
     */
    auto simplify_binary_op(std::shared_ptr<Node> node, Graph& graph) -> bool;

    /**
     * @brief Try to simplify a unary operation.
     *
     * @param node Operation node
     * @param graph Graph containing node
     * @return true if simplified
     */
    auto simplify_unary_op(std::shared_ptr<Node> node, Graph& graph) -> bool;
};

/**
 * @brief Strength reduction pass.
 *
 * Replaces expensive operations with cheaper equivalents:
 * - Div(x, const) -> Mul(x, 1/const)
 * - Pow(x, 2) -> Mul(x, x)
 * - Pow(x, 0.5) -> Sqrt(x)
 * - Mul(x, 2) -> Add(x, x)
 *
 * These transformations reduce the number of expensive arithmetic
 * operations (division, exponentiation) in favor of cheaper ones
 * (multiplication, addition, square root).
 */
class StrengthReductionPass : public Pass {
public:
    /// @param allow_unsafe_algebra Enable the Div(x,c)->Mul(x,1/c) rewrite.
    /// Multiplying by a precomputed reciprocal is NOT bit-identical to division
    /// (1/c is generally not exactly representable), so it is gated OFF by
    /// default and only applied under fast-math, mirroring
    /// AlgebraicSimplificationPass. The value-preserving rewrites (Pow(x,2),
    /// Pow(x,0.5), Mul(x,2)) are always applied.
    explicit StrengthReductionPass(bool allow_unsafe_algebra = false)
        : allow_unsafe_algebra_(allow_unsafe_algebra) {}
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "StrengthReduction"; }

private:
    bool allow_unsafe_algebra_ = false;
    auto reduce_node(std::shared_ptr<Node> node, Graph& graph) -> bool;
};

/**
 * @brief Reshape elimination pass.
 *
 * Removes redundant reshape operations:
 * - reshape(x, shape) where shape == x.shape
 * - reshape(reshape(x, s1), s2) -> reshape(x, s2)
 *
 * Example:
 * @code
 * y = reshape(x, [10, 20])
 * z = reshape(y, [200])
 * # Becomes:
 * z = reshape(x, [200])
 * @endcode
 */
class ReshapeEliminationPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "ReshapeElimination"; }
};

/**
 * @brief Loop unrolling pass.
 *
 * For Loop nodes with a small constant bound (<= 8), unrolls the loop
 * by inlining the body subgraph N times with remapped value IDs.
 */
class LoopUnrollingPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "LoopUnrolling"; }
    auto set_max_unroll(int64_t max) -> void { max_unroll_ = max; }
private:
    int64_t max_unroll_{8};
};

/**
 * @brief Loop-invariant code motion (LICM) pass.
 *
 * Hoists loop-invariant computations outside of Loop nodes.
 * A node is invariant if all its inputs come from outside the loop.
 */
class LICMPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "LICM"; }
};

/**
 * @brief Memory planning pass.
 *
 * Performs live range analysis and greedy buffer assignment to enable
 * memory reuse between non-overlapping intermediate values. This pass
 * should run after all other optimization passes, since it annotates
 * graph values with buffer assignments based on the final graph topology.
 *
 * This pass is not iterative -- it runs once and annotates the graph.
 * It does not structurally modify the graph (no nodes added or removed),
 * so it always returns false to prevent re-running of earlier passes.
 *
 * Example:
 * @code
 * MemoryPlanningPass pass;
 * pass.run(graph);
 * // Values now have buffer_id and buffer_offset set
 * auto plan = pass.memory_plan();
 * @endcode
 */
class MemoryPlanningPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "MemoryPlanning"; }

    /**
     * @brief Get the memory plan produced by the last run.
     *
     * @return Memory plan (empty if run() has not been called)
     */
    auto memory_plan() const -> const MemoryPlan& { return plan_; }

    /**
     * @brief Set alignment for buffer allocations.
     *
     * @param alignment Alignment in bytes (must be power of 2)
     */
    auto set_alignment(size_t alignment) -> void { alignment_ = alignment; }

private:
    MemoryPlan plan_;
    size_t alignment_{64};
};

/**
 * @brief Extended fusion pass for complex kernel patterns.
 *
 * Uses PatternMatcher to identify multi-node fusion opportunities
 * beyond simple pairwise fusion: reduction chains, GEMM epilogues,
 * softmax, LayerNorm, RMSNorm, and small MLPs. Matched patterns
 * are replaced with single fused-kernel nodes executed via
 * ExtendedKernelCodegen.
 *
 * Should run after individual fusion passes (Conv+BN, etc.) but
 * before memory planning.
 *
 * Speedup: 10-50% for transformer-heavy workloads
 */
class ExtendedFusionPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "ExtendedFusion"; }

    /**
     * @brief Set max hidden dimension for SmallMLP fusion.
     *
     * @param max_dim Maximum hidden dimension (default: 4096)
     */
    auto set_max_mlp_hidden_dim(int64_t max_dim) -> void { max_mlp_hidden_ = max_dim; }

    /**
     * @brief Set the target device type for fusion cost estimation.
     *
     * Configures the internal cost model to use device-specific heuristics
     * when evaluating fusion profitability.
     *
     * @param type Target device type
     */
    auto set_device_type(Device::Type type) -> void { cost_model_.set_device_type(type); }

private:
    int64_t max_mlp_hidden_{4096};
    FusionCostModel cost_model_;
};

/**
 * @brief Post-training quantization pass.
 *
 * Identifies Linear and Conv2d nodes with quantizable weights and
 * replaces them with quantized variants (QuantizedLinear, QuantizedConv2d).
 * Inserts quantize/dequantize nodes at graph boundaries.
 *
 * Supports INT8 symmetric quantization with per-tensor or per-channel scales.
 * Optionally runs a calibration phase to determine activation ranges.
 *
 * Typical speedup: 1.5-4x depending on hardware INT8 support.
 */
class QuantizationPass : public Pass {
public:
    enum class Mode {
        Dynamic,   ///< Quantize weights statically, activations dynamically at runtime
        Static     ///< Both weights and activations quantized using calibration data
    };

    explicit QuantizationPass(Mode mode = Mode::Dynamic) : mode_(mode) {}

    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "Quantization"; }

    /// Set the target dtype for quantized operations (default: Int8)
    auto set_target_dtype(DType dtype) -> void { target_dtype_ = dtype; }

private:
    Mode mode_;
    DType target_dtype_{DType::Int8};

    auto quantize_linear(std::shared_ptr<Node> node, Graph& graph) -> bool;
    auto quantize_conv2d(std::shared_ptr<Node> node, Graph& graph) -> bool;
    auto compute_scale_and_zero(const Tensor& weight) -> std::pair<float, int64_t>;
};

/**
 * @brief Sparse tensor optimization pass.
 *
 * Identifies nodes where weight tensors have high sparsity (>threshold)
 * and replaces dense operations with sparse equivalents (SparseSpMM).
 * Inserts DenseToSparse conversion nodes for sparse weights.
 *
 * Most effective for large language models with pruned weights.
 */
class SparsePass : public Pass {
public:
    explicit SparsePass(float sparsity_threshold = 0.9f)
        : threshold_(sparsity_threshold) {}

    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "SparseOptimization"; }

    /// Set minimum sparsity ratio to trigger sparse conversion (0.0 - 1.0)
    auto set_threshold(float threshold) -> void { threshold_ = threshold; }

private:
    float threshold_;

    auto compute_sparsity(const Tensor& weight) -> float;
    auto convert_to_sparse(std::shared_ptr<Node> node, Graph& graph) -> bool;
};

/**
 * @brief Layout optimization pass (NCHW -> NHWC).
 *
 * Automatically converts convolution and batch normalization nodes to
 * channels-last memory format on GPU devices that benefit from it
 * (e.g., Tensor Core utilization on NVIDIA/AMD GPUs).
 *
 * Inserts LayoutConvert pseudo-nodes at format boundaries to maintain
 * correctness when transitioning between channels-last and contiguous ops.
 */
class LayoutOptimizationPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "LayoutOptimization"; }

private:
    /// Check if a node benefits from channels-last format
    auto benefits_from_channels_last(OpType op) const -> bool;

    /// Check if a node is format-agnostic (element-wise, activation, etc.)
    auto is_format_agnostic(OpType op) const -> bool;
};

/**
 * @brief DType auto-optimization pass (compile-time AMP).
 *
 * Automatically inserts Cast nodes to use lower-precision types
 * (Float16/BFloat16) for compute-heavy operations while keeping
 * precision-sensitive operations in Float32.
 *
 * Only active on GPU-device graphs. Must be explicitly enabled
 * or triggered via "max-autotune" compilation mode.
 */
class DTypeOptimizationPass : public Pass {
public:
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "DTypeOptimization"; }

    /// Set target low-precision dtype (default: Float16)
    auto set_target_dtype(DType dtype) -> void { target_dtype_ = dtype; }

    /// Enable or disable the pass
    auto set_enabled(bool enabled) -> void { enabled_ = enabled; }

private:
    DType target_dtype_{DType::Float16};
    bool enabled_{true};

    /// Check if an op is compute-heavy (benefits from lower precision)
    auto is_compute_heavy(OpType op) const -> bool;

    /// Check if an op is stability-critical (must stay in Float32)
    auto is_stability_critical(OpType op) const -> bool;
};

/**
 * @brief Compiler that applies optimization passes.
 *
 * The compiler runs a sequence of passes to transform and optimize
 * the IR graph. Passes are applied in order until convergence or
 * max iterations is reached. After optimization converges, memory
 * planning is performed as a final step.
 *
 * @code
 * Compiler compiler;
 * compiler.add_pass(std::make_unique<DeadCodeEliminationPass>());
 * compiler.add_pass(std::make_unique<FuseConvBatchNormPass>());
 * compiler.add_pass(std::make_unique<FuseConvReluPass>());
 * compiler.optimize(graph);
 * auto plan = compiler.memory_plan();
 * @endcode
 */
class Compiler {
public:
    /**
     * @brief Construct compiler with default passes.
     *
     * @param enable_default_passes If true, adds standard optimization passes
     */
    explicit Compiler(bool enable_default_passes = true);

    /**
     * @brief Add optimization pass.
     *
     * @param pass Pass to add (compiler takes ownership)
     */
    auto add_pass(std::unique_ptr<Pass> pass) -> void;

    /**
     * @brief Optimize graph with all passes.
     *
     * Runs passes in order, repeating until no changes or max iterations.
     *
     * @param graph Graph to optimize
     * @param max_iterations Maximum number of pass iterations (default: 10)
     * @return Number of passes that made changes
     */
    auto optimize(Graph& graph, int max_iterations = 10) -> int;

    /**
     * @brief Get optimization statistics.
     *
     * @return Map of pass name -> number of times it made changes
     */
    auto get_stats() const -> const std::unordered_map<std::string, int>& {
        return pass_stats_;
    }

    /**
     * @brief Clear statistics.
     */
    auto clear_stats() -> void { pass_stats_.clear(); }

    /**
     * @brief Enable verbose logging.
     *
     * @param enable If true, print pass execution details
     */
    auto set_verbose(bool enable) -> void { verbose_ = enable; }

    /**
     * @brief Run memory planning after optimization.
     *
     * Performs live range analysis and greedy buffer assignment on the
     * optimized graph. Should be called after optimize() completes.
     * The resulting plan is stored and can be retrieved via memory_plan().
     *
     * @param graph Graph to plan memory for
     * @return The computed memory plan
     */
    auto plan_memory(Graph& graph) -> MemoryPlan;

    /**
     * @brief Get the memory plan from the last plan_memory() call.
     *
     * @return Memory plan (empty if plan_memory() has not been called)
     */
    auto memory_plan() const -> const MemoryPlan& { return memory_plan_; }

    /**
     * @brief Enable or disable memory planning during optimize().
     *
     * When enabled, memory planning runs automatically as the final
     * step of optimize(). By default, memory planning uses "auto" mode:
     * it is enabled when the graph exceeds a size threshold (50 nodes),
     * and disabled for smaller graphs where the planning overhead is
     * not worthwhile.
     *
     * @param enable If true, always run memory planning; if false, never run
     */
    auto set_memory_planning(bool enable) -> void {
        memory_planning_explicit_ = true;
        enable_memory_planning_ = enable;
    }

    /**
     * @brief Set the node count threshold for automatic memory planning.
     *
     * When memory planning is in auto mode (default), graphs with at
     * least this many nodes will have memory planning enabled.
     *
     * @param threshold Minimum number of nodes (default: 50)
     */
    auto set_memory_planning_threshold(size_t threshold) -> void {
        memory_planning_threshold_ = threshold;
    }

private:
    std::vector<std::unique_ptr<Pass>> passes_;               ///< Optimization passes
    std::unordered_map<std::string, int> pass_stats_;         ///< Pass execution stats
    bool verbose_{false};                                     ///< Verbose logging flag
    bool enable_memory_planning_{false};                      ///< Memory planning flag
    bool memory_planning_explicit_{false};                    ///< True if user explicitly set memory planning
    size_t memory_planning_threshold_{50};                    ///< Auto-enable threshold (node count)
    MemoryPlan memory_plan_;                                  ///< Cached memory plan

    /**
     * @brief Run single pass iteration.
     *
     * @param graph Graph to optimize
     * @return Number of passes that made changes
     */
    auto run_passes(Graph& graph) -> int;
};

/**
 * @brief Apply standard optimizations to graph.
 *
 * Convenience function that creates a compiler with default passes
 * and optimizes the graph.
 *
 * @param graph Graph to optimize
 * @return Number of optimizations applied
 *
 * @code
 * auto graph = trace(model, input);
 * optimize_graph(*graph);
 * graph->save("optimized_model.pt");
 * @endcode
 */
auto optimize_graph(Graph& graph) -> int;

/**
 * @brief A compiled (traced) module for optimized execution.
 *
 * CompiledModule wraps a traced IR graph with a high-level interface
 * for inference. It supports:
 * - Running the optimized graph with new inputs
 * - Saving/loading compiled models to/from disk
 * - Metadata storage for model versioning and provenance
 * - Access to the underlying graph for inspection
 *
 * Create a CompiledModule by tracing an nn::Module:
 * @code
 * auto model = std::make_shared<MyNetwork>();
 * Variable input(Tensor({1, 3, 224, 224}, DType::Float32, Device::cpu()));
 * auto compiled = CompiledModule::trace(model, input);
 * compiled->optimize_for_inference();
 * auto output = compiled->forward(new_input);
 * @endcode
 */
class CompiledModule {
public:
    /**
     * @brief Trace a module with an example input.
     *
     * Records all operations during the module's forward pass and
     * builds an IR graph that can be optimized and executed.
     *
     * @param module Module to trace (set to eval mode internally)
     * @param example_input Example input for shape and type inference
     * @return Compiled module wrapping the traced graph
     */
    static auto trace(std::shared_ptr<nn::Module> module,
                      const Variable& example_input) -> std::shared_ptr<CompiledModule>;

    /**
     * @brief Trace a module with a raw tensor input.
     *
     * Convenience overload that wraps the tensor in a Variable.
     *
     * @param module Module to trace
     * @param example_input Example input tensor
     * @return Compiled module wrapping the traced graph
     */
    static auto trace(std::shared_ptr<nn::Module> module,
                      const Tensor& example_input) -> std::shared_ptr<CompiledModule>;

    /**
     * @brief Execute the compiled graph with a Variable input.
     *
     * Runs the optimized graph using the provided input.
     *
     * @param input Input variable
     * @return Output variable
     */
    auto forward(const Variable& input) -> Variable;

    /**
     * @brief Execute the compiled graph with a raw Tensor input.
     *
     * Convenience overload that wraps the tensor in a Variable.
     *
     * @param input Input tensor
     * @return Output variable
     */
    auto forward(const Tensor& input) -> Variable;

    /**
     * @brief Execute the compiled graph with multiple inputs.
     *
     * @param inputs Input variables
     * @return Output variables
     */
    auto forward(const std::vector<Variable>& inputs) -> std::vector<Variable>;

    /**
     * @brief Apply inference optimizations to the graph.
     *
     * Runs the full suite of optimization passes: fusion, DCE, CSE,
     * constant folding, algebraic simplification, reshape elimination,
     * and memory planning.
     *
     * @return Number of optimizations applied
     */
    auto optimize_for_inference() -> int;

    /**
     * @brief Mark input dimensions as dynamic and propagate symbolic shapes.
     *
     * Creates a SymbolicTracePass with the specified dynamic dimension
     * configuration, runs it on the graph, and stores the configuration
     * for use during forward(). When forward() is called with inputs
     * that have symbolic dimensions in the graph, a SymbolicShapeEnvironment
     * is created to bind symbolic names to actual input dimension values.
     *
     * @param dynamic_dims Vector of {input_idx, dim, name} tuples specifying
     *        which dimensions are dynamic and their symbolic names
     *
     * @code
     * auto compiled = CompiledModule::trace(model, example_input);
     * compiled->mark_dynamic_dims({
     *     {0, 0, "batch"},      // input 0, dim 0 = dynamic "batch"
     *     {0, 1, "seq_len"},    // input 0, dim 1 = dynamic "seq_len"
     * });
     * compiled->optimize_for_inference();
     * // Now forward() supports variable batch/seq_len
     * @endcode
     */
    struct DynamicDimSpec {
        int input_idx;
        int dim;
        std::string name;
    };
    auto mark_dynamic_dims(const std::vector<DynamicDimSpec>& dynamic_dims) -> void;

    /**
     * @brief Check if this module has dynamic shape support enabled.
     *
     * @return true if mark_dynamic_dims() has been called with at least one dim
     */
    auto has_dynamic_shapes() const -> bool { return !dynamic_dims_.empty(); }

    /**
     * @brief Get the underlying IR graph.
     *
     * @return Shared pointer to the graph
     */
    auto graph() const -> std::shared_ptr<Graph> { return graph_; }

    /**
     * @brief Set the underlying IR graph.
     *
     * Used by compile() to inject a traced graph into a CompiledModule.
     *
     * @param g Graph to set
     */
    auto set_graph(std::shared_ptr<Graph> g) -> void { graph_ = std::move(g); }

    /**
     * @brief Get the memory plan for this module.
     *
     * Available after optimize_for_inference() has been called.
     *
     * @return Memory plan (empty if not yet optimized)
     */
    auto memory_plan() const -> const MemoryPlan& { return memory_plan_; }

    /**
     * @brief Set the memory plan for this module.
     *
     * @param plan Memory plan to store
     */
    auto set_memory_plan(MemoryPlan plan) -> void { memory_plan_ = std::move(plan); }

    /**
     * @brief Save compiled module to file.
     *
     * Serializes the graph and metadata to a binary file.
     *
     * @param path Output file path
     */
    auto save(const std::string& path) const -> void;

    /**
     * @brief Load compiled module from file.
     *
     * Deserializes the graph and metadata from a binary file.
     *
     * @param path Input file path
     * @return Loaded compiled module
     * @throws std::runtime_error if file is invalid or corrupted
     */
    static auto load(const std::string& path) -> std::shared_ptr<CompiledModule>;

    /**
     * @brief Add metadata key-value pair.
     *
     * @param key Metadata key
     * @param value Metadata value
     */
    auto add_metadata(const std::string& key, const std::string& value) -> void;

    /**
     * @brief Get metadata value by key.
     *
     * @param key Metadata key
     * @return Metadata value (empty string if not found)
     */
    auto get_metadata(const std::string& key) const -> std::string;

    /**
     * @brief Check if metadata key exists.
     *
     * @param key Metadata key
     * @return true if key exists
     */
    auto has_metadata(const std::string& key) const -> bool;

    /**
     * @brief Get all metadata.
     *
     * @return Map of all metadata key-value pairs
     */
    auto all_metadata() const -> const std::unordered_map<std::string, std::string>&;

    /**
     * @brief Capture a CUDA graph from the compiled module's forward pass.
     *
     * Records all GPU operations during a forward pass with the given sample
     * inputs into a CUDA graph that can be replayed for faster execution.
     * Input tensor shapes are recorded; subsequent replays must use the same shapes.
     *
     * All GPU memory allocations must happen before capture (use the caching
     * allocator). The sample inputs must reside on a CUDA device.
     *
     * @param sample_inputs Sample input tensors for shape/type inference
     * @throws std::runtime_error if CUDA is unavailable or capture fails
     */
    auto capture_cuda_graph(std::vector<Tensor> sample_inputs) -> void;

    /**
     * @brief Replay the captured CUDA graph with new input data.
     *
     * The input tensors must have the same shapes as those used during capture.
     * If no graph has been captured, returns false without executing.
     *
     * @param inputs Input tensors (data is read from these buffers)
     * @return true if the graph was replayed, false if no graph is captured
     * @throws std::runtime_error if input shapes don't match captured shapes
     */
    auto replay_cuda_graph(std::vector<Tensor>& inputs) -> bool;

    /**
     * @brief Get the captured graph's output tensors.
     *
     * The returned tensors alias the device buffers the captured graph writes
     * into, so after a replay_cuda_graph() call they hold the fresh results for
     * the most recent replay inputs. Returns an empty vector if nothing has been
     * captured.
     *
     * @return The captured output tensors (post-replay results live here).
     */
    auto replay_cuda_graph_outputs() const -> std::vector<Tensor>;

    /**
     * @brief Invalidate the captured CUDA graph.
     *
     * Call this when input shapes change or when the graph is no longer needed.
     * After invalidation, capture_cuda_graph() must be called again before replay.
     */
    auto invalidate_cuda_graph() -> void;

    /**
     * @brief Check if a CUDA graph has been captured and is ready for replay.
     *
     * @return true if a graph is captured
     */
    auto has_cuda_graph() const -> bool;

    /**
     * @brief Destructor.
     */
    ~CompiledModule();

    /**
     * @brief Constructor (prefer using static trace/load methods).
     */
    CompiledModule() = default;

    /**
     * @brief Constructor with an existing graph.
     *
     * @param graph Pre-built IR graph
     */
    explicit CompiledModule(std::shared_ptr<Graph> graph);

    /// Set the source module for retrace support (called by trace())
    auto set_source_module(std::shared_ptr<nn::Module> module) -> void {
        source_module_ = std::move(module);
    }

private:
    std::shared_ptr<Graph> graph_;                                   ///< IR graph
    std::unordered_map<std::string, std::string> metadata_;          ///< Metadata storage
    MemoryPlan memory_plan_;                                         ///< Memory plan for buffer reuse
    std::unique_ptr<CUDAGraph> cuda_graph_;                          ///< Captured CUDA graph for replay
    std::vector<std::vector<int64_t>> captured_shapes_;              ///< Input shapes at capture time
    /// The exact contiguous input Tensors whose device buffers the captured
    /// CUDA/HIP graph hard-codes. A captured graph re-runs verbatim over these
    /// pointers — it has no input-rebinding API — so replay() MUST first copy
    /// each fresh replay input's data into these buffers (device-to-device),
    /// otherwise replay returns the stale capture-time result. Retaining the
    /// Tensors here keeps their storage alive for the lifetime of the graph.
    std::vector<Tensor> captured_inputs_;
    /// The captured graph's output Tensors. Their device storage is the buffer
    /// the terminal nodes write into, so after a replay() they hold the fresh
    /// results. Exposed via replay_cuda_graph_outputs() so a replay is readable.
    std::vector<Tensor> captured_outputs_;
    std::shared_ptr<nn::Module> source_module_;                      ///< Source module for re-tracing
    std::unordered_map<std::string, std::shared_ptr<Graph>> shape_cache_;  ///< Cached graphs by shape key
    int retrace_count_{0};                                           ///< Number of retraces performed
    static constexpr int MAX_RETRACES = 8;                           ///< Maximum distinct shapes to cache
    std::vector<DynamicDimSpec> dynamic_dims_;                       ///< Dynamic dimension configuration
    Device traced_device_{Device::cpu()};                            ///< Device used at most recent trace
    DType  traced_dtype_{DType::Float32};                            ///< DType used at most recent trace

    /// Serialises the mutating paths of forward()/replay/capture so a single
    /// CompiledModule shared across inference threads (the natural server
    /// pattern) cannot race on graph_, shape_cache_, traced_device_/dtype_,
    /// cuda_graph_ or captured_shapes_. Recursive so a future internal call
    /// chain that re-enters a guarded method cannot self-deadlock. Mutable so it
    /// can be locked from const-correct accessors if added later.
    mutable std::recursive_mutex forward_mutex_;

    /// Compute cache key from input shapes
    static auto compute_shape_key(const Variable& input) -> std::string;
    static auto compute_shape_key(const std::vector<Variable>& inputs) -> std::string;
};

// ============================================================================
// Convenience free functions for working with CompiledModule
// ============================================================================

/**
 * @brief Apply inference optimizations to a compiled module.
 *
 * Convenience function matching common JIT API patterns.
 *
 * @param module Compiled module to optimize
 * @return Number of optimizations applied
 */
auto optimize_for_inference(std::shared_ptr<CompiledModule> module) -> int;

/**
 * @brief Save a compiled module to file.
 *
 * @param module Module to save
 * @param path Output file path
 */
auto save(const std::shared_ptr<CompiledModule>& module, const std::string& path) -> void;

/**
 * @brief Load a compiled module from file.
 *
 * @param path Input file path
 * @return Loaded compiled module
 */
auto load(const std::string& path) -> std::shared_ptr<CompiledModule>;

/**
 * @brief Add metadata to a compiled module.
 *
 * @param module Target module
 * @param key Metadata key
 * @param value Metadata value
 */
auto add_metadata(const std::shared_ptr<CompiledModule>& module,
                  const std::string& key, const std::string& value) -> void;

/**
 * @brief Get metadata from a compiled module.
 *
 * @param module Source module
 * @param key Metadata key
 * @return Metadata value (empty string if not found)
 */
auto get_metadata(const std::shared_ptr<CompiledModule>& module,
                  const std::string& key) -> std::string;

} // namespace jit
} // namespace tenzor
