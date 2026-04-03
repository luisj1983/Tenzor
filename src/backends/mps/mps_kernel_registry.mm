/**
 * @file mps_kernel_registry.mm
 * @brief MPS kernel registration stub
 *
 * Registers Metal compute shader kernels with the dispatch table.
 * Initially provides basic operations; more kernels will be added
 * as Metal compute shaders are implemented.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "mps_backend.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/core/tensor.hpp"

namespace tenzor {
namespace mps {

auto register_mps_kernels(BackendDispatchTable& table) -> void {
    // TODO: Register Metal compute shader kernels as they are implemented.
    //
    // Priority order:
    // 1. Tier 1 (inference): Add, Sub, Mul, Div, MatMul, ReLU, Sigmoid,
    //    Softmax, Conv2d, LayerNorm, BatchNorm, Embedding, Linear
    // 2. Tier 2 (training): Backward ops, in-place ops, optimizer fused steps
    // 3. Tier 3 (completeness): RNN, 3D conv/pool, FFT, sparse, linalg
    //
    // For MatMul, use MPSMatrixMultiplication for best performance.
    // For Conv2d, use MPSCNNConvolution.
    // For element-wise ops, use custom .metal compute shaders.
}

} // namespace mps
} // namespace tenzor

// Export function for dynamic loading
extern "C" {
    void register_kernels(tenzor::BackendDispatchTable* table) {
        if (table) {
            tenzor::mps::register_mps_kernels(*table);
        }
    }
}
