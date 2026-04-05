/**
 * @file runtime.cpp
 * @brief Implementation of the lite inference runtime
 */

#include "tenzor/lite/runtime.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace tenzor {
namespace lite {

// ============================================================================
// LiteTensor
// ============================================================================

LiteTensor::~LiteTensor() {
    if (owns_data && data) {
        std::free(data);
        data = nullptr;
    }
}

auto LiteTensor::numel() const -> int64_t {
    if (ndim == 0) return 0;
    int64_t n = 1;
    for (int32_t i = 0; i < ndim; ++i) {
        n *= shape[i];
    }
    return n;
}

auto LiteTensor::nbytes() const -> int64_t {
    return numel() * dtype_size(dtype);
}

// ============================================================================
// LiteAllocator
// ============================================================================

LiteAllocator::LiteAllocator(const std::vector<size_t>& pool_sizes, size_t alignment)
    : pool_sizes_(pool_sizes), alignment_(alignment) {
    pools_.reserve(pool_sizes.size());
    for (auto size : pool_sizes) {
        void* ptr = nullptr;
        if (size > 0) {
#ifdef _WIN32
            ptr = _aligned_malloc(size, alignment);
#else
            if (posix_memalign(&ptr, alignment, size) != 0) {
                ptr = nullptr;
            }
#endif
            if (!ptr) {
                // Clean up already allocated pools
                for (auto p : pools_) {
#ifdef _WIN32
                    _aligned_free(p);
#else
                    std::free(p);
#endif
                }
                throw std::runtime_error("LiteAllocator: failed to allocate " +
                                         std::to_string(size) + " bytes");
            }
            std::memset(ptr, 0, size);
        }
        pools_.push_back(ptr);
        total_bytes_ += size;
    }
}

LiteAllocator::~LiteAllocator() {
    for (auto ptr : pools_) {
        if (ptr) {
#ifdef _WIN32
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }
    }
}

auto LiteAllocator::get_buffer(size_t buffer_id, size_t offset) -> void* {
    if (buffer_id >= pools_.size()) {
        throw std::out_of_range("LiteAllocator: buffer_id out of range");
    }
    return static_cast<char*>(pools_[buffer_id]) + offset;
}

// ============================================================================
// LiteRuntime implementation
// ============================================================================

struct LiteRuntime::Impl {
    std::unique_ptr<LiteAllocator> allocator;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
    std::unordered_map<std::string, std::string> metadata;

    // In production: graph nodes, weight data, op dispatch table
    std::vector<uint8_t> model_data;  // Raw model bytes for now
};

LiteRuntime::~LiteRuntime() = default;

auto LiteRuntime::load(const std::string& path) -> std::unique_ptr<LiteRuntime> {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("LiteRuntime: cannot open file: " + path);
    }

    // Read file into memory
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size);

    return load(data.data(), data.size());
}

auto LiteRuntime::load(const void* data, size_t size) -> std::unique_ptr<LiteRuntime> {
    if (!data || size == 0) {
        throw std::runtime_error("LiteRuntime: empty model data");
    }

    // In production: parse TZLITE format header, extract graph, weights, memory plan
    // For now: create a basic runtime with placeholder data

    auto runtime = std::unique_ptr<LiteRuntime>(new LiteRuntime());
    runtime->impl_ = std::make_unique<Impl>();
    runtime->impl_->model_data.assign(
        static_cast<const uint8_t*>(data),
        static_cast<const uint8_t*>(data) + size);

    // Default: single pool of 1MB for intermediates
    runtime->impl_->allocator = std::make_unique<LiteAllocator>(
        std::vector<size_t>{1024 * 1024}, 64);

    return runtime;
}

auto LiteRuntime::forward(const LiteTensor& input) -> LiteTensor {
    // In production: execute the graph nodes in topological order,
    // dispatching each to the appropriate lite kernel, using
    // allocator buffers for intermediates.

    // Placeholder: return input as output
    LiteTensor output;
    output.ndim = input.ndim;
    output.dtype = input.dtype;
    output.shape = input.shape;
    output.strides = input.strides;
    output.owns_data = true;

    auto bytes = input.nbytes();
    output.data = std::malloc(bytes);
    if (output.data && input.data) {
        std::memcpy(output.data, input.data, bytes);
    }

    return output;
}

auto LiteRuntime::forward(const std::vector<LiteTensor>& inputs) -> std::vector<LiteTensor> {
    std::vector<LiteTensor> outputs;
    outputs.reserve(inputs.size());
    for (auto& input : inputs) {
        outputs.push_back(forward(input));
    }
    return outputs;
}

auto LiteRuntime::input_shapes() const -> std::vector<std::vector<int64_t>> {
    return impl_ ? impl_->input_shapes : std::vector<std::vector<int64_t>>{};
}

auto LiteRuntime::output_shapes() const -> std::vector<std::vector<int64_t>> {
    return impl_ ? impl_->output_shapes : std::vector<std::vector<int64_t>>{};
}

auto LiteRuntime::model_metadata(const std::string& key) const -> std::string {
    if (!impl_) return "";
    auto it = impl_->metadata.find(key);
    return it != impl_->metadata.end() ? it->second : "";
}

auto LiteRuntime::create_input(const std::vector<int64_t>& shape, DType dtype) -> LiteTensor {
    LiteTensor tensor;
    tensor.ndim = static_cast<int32_t>(std::min(shape.size(), static_cast<size_t>(kMaxDims)));
    tensor.dtype = dtype;
    tensor.owns_data = true;

    int64_t numel = 1;
    for (int32_t i = 0; i < tensor.ndim; ++i) {
        tensor.shape[i] = shape[i];
        numel *= shape[i];
    }

    // Compute strides (row-major)
    for (int32_t i = tensor.ndim - 1; i >= 0; --i) {
        tensor.strides[i] = (i == tensor.ndim - 1) ? 1 : tensor.strides[i + 1] * tensor.shape[i + 1];
    }

    auto bytes = numel * dtype_size(dtype);
    tensor.data = std::calloc(1, bytes);

    return tensor;
}

} // namespace lite
} // namespace tenzor
