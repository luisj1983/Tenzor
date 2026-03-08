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
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include "graph.hpp"
#include "memory_planner.hpp"
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
    auto run(Graph& graph) -> bool override;
    auto name() const -> std::string override { return "AlgebraicSimplification"; }

private:
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
     * @brief Get the underlying IR graph.
     *
     * @return Shared pointer to the graph
     */
    auto graph() const -> std::shared_ptr<Graph> { return graph_; }

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

private:
    std::shared_ptr<Graph> graph_;                                   ///< IR graph
    std::unordered_map<std::string, std::string> metadata_;          ///< Metadata storage
    MemoryPlan memory_plan_;                                         ///< Memory plan for buffer reuse
    std::unique_ptr<CUDAGraph> cuda_graph_;                          ///< Captured CUDA graph for replay
    std::vector<std::vector<int64_t>> captured_shapes_;              ///< Input shapes at capture time
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
