#include "tenzor/backend/cuda_graph.hpp"

namespace tenzor {

// Default (weak) implementation of CUDAGraph::create for when the CUDA backend
// is not loaded. The CUDA backend .so provides the real implementation which
// overrides this via symbol interposition when dlopen'd with RTLD_GLOBAL.
// Since backends use RTLD_LOCAL, the Python bindings (which link against the
// core lib) need this fallback to avoid unresolved symbol errors at import time.
auto CUDAGraph::create(int32_t /*device_id*/) -> std::unique_ptr<CUDAGraph> {
    return nullptr;  // CUDA not available
}

} // namespace tenzor
