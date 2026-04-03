/**
 * @file mps_kernel_registry.mm
 * @brief MPS kernel registration for Tier 1 operations
 *
 * Registers Metal compute shader kernels with the dispatch table.
 * Tier 1 covers the essential ops needed for inference.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "mps_backend.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/core/tensor.hpp"

namespace tenzor::mps {

// Forward declarations of kernel functions (defined in kernels/mps_elementwise.mm)
Tensor mps_add_kernel(const Tensor& a, const Tensor& b);
Tensor mps_sub_kernel(const Tensor& a, const Tensor& b);
Tensor mps_mul_kernel(const Tensor& a, const Tensor& b);
Tensor mps_div_kernel(const Tensor& a, const Tensor& b);
Tensor mps_relu_kernel(const Tensor& input);
Tensor mps_sigmoid_kernel(const Tensor& input);
Tensor mps_neg_kernel(const Tensor& input);
Tensor mps_exp_kernel(const Tensor& input);
Tensor mps_log_kernel(const Tensor& input);
Tensor mps_matmul_kernel(const Tensor& a, const Tensor& b);
Tensor mps_linear_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias);
Tensor mps_embedding_kernel(const Tensor& weight, const Tensor& indices);
Tensor mps_softmax_kernel(const Tensor& input, int64_t dim);
Tensor mps_batch_norm_kernel(const Tensor& input, const Tensor& mean,
                              const Tensor& var, const Tensor& weight,
                              const Tensor& bias, float eps);
Tensor mps_layer_norm_kernel(const Tensor& input, const Tensor& weight,
                              const Tensor& bias, float eps);
Tensor mps_conv2d_kernel(const Tensor& input, const Tensor& weight,
                          int64_t stride_h, int64_t stride_w,
                          int64_t pad_h, int64_t pad_w, int64_t groups);

auto register_mps_kernels(BackendDispatchTable& table) -> void {
    // ================================================================
    // Tier 1: Arithmetic operations
    // ================================================================
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Add, mps_add_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Sub, mps_sub_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Mul, mps_mul_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Div, mps_div_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, MatMul, mps_matmul_kernel);

    // ================================================================
    // Tier 1: Activation functions
    // ================================================================
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, ReLU, mps_relu_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Sigmoid, mps_sigmoid_kernel);

    // ================================================================
    // Tier 1: Element-wise math
    // ================================================================
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Neg, mps_neg_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Exp, mps_exp_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Log, mps_log_kernel);

    // ================================================================
    // Tier 1: Linear (matmul + bias add)
    // ================================================================
    table.register_single_output_kernel(OpId::Linear,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            if (inputs.size() >= 3) {
                return mps_linear_kernel(inputs[0], inputs[1], inputs[2]);
            }
            return mps_matmul_kernel(inputs[0], inputs[1]);
        });

    // ================================================================
    // Tier 1: Embedding
    // ================================================================
    table.register_kernel(OpId::Embedding,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {mps_embedding_kernel(inputs[0], inputs[1])};
        });

    // ================================================================
    // Tier 1: Softmax
    // ================================================================
    table.register_kernel(OpId::Softmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return {mps_softmax_kernel(inputs[0], dim)};
        });

    // ================================================================
    // Tier 1: BatchNorm (inference path)
    // ================================================================
    table.register_kernel(OpId::BatchNorm2dForwardAffine,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            return {mps_batch_norm_kernel(inputs[0], inputs[1], inputs[2],
                                          inputs[3], inputs[4], eps)};
        });

    // ================================================================
    // Tier 1: LayerNorm
    // ================================================================
    table.register_kernel(OpId::FusedLayerNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            Tensor mean, inv_std; // placeholders for training path
            return {mps_layer_norm_kernel(inputs[0], inputs[1], inputs[2], eps),
                    mean, inv_std};
        });

    // ================================================================
    // Tier 1: Conv2d
    // ================================================================
    table.register_kernel(OpId::Conv2dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t sh = attrs.get_int(AttrKey::StrideH, 1);
            int64_t sw = attrs.get_int(AttrKey::StrideW, 1);
            int64_t ph = attrs.get_int(AttrKey::PaddingH, 0);
            int64_t pw = attrs.get_int(AttrKey::PaddingW, 0);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            return {mps_conv2d_kernel(inputs[0], inputs[1], sh, sw, ph, pw, groups)};
        });

    // TODO: Tier 2 (training) - backward ops, optimizer steps
    // TODO: Tier 3 (completeness) - RNN, 3D ops, FFT, sparse, linalg
}

} // namespace tenzor::mps

// Export function for dynamic loading
extern "C" {
    void register_kernels(tenzor::BackendDispatchTable* table) {
        if (table) {
            tenzor::mps::register_mps_kernels(*table);
        }
    }
}
