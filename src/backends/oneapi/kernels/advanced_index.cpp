/**
 * @file advanced_index.cpp
 * @brief OneAPI/SYCL implementation of AdvancedIndex and AdvancedIndexPut.
 *
 * Mirrors src/backends/cuda/kernels/advanced.cu lines 1600-1924
 * (advanced_index_cuda_kernel / advanced_index_put_cuda_kernel). Replaces
 * the previous CPU-roundtrip fallback in oneapi_kernel_registry.cpp.
 *
 * Algorithm: one work-item per output element. Each work-item decodes its
 * position into a broadcast index (over indexed dims) and a pass index
 * (over passthrough dims), gathers per-dim index values from device
 * pointers, computes the source offset, then copies the element.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "oneapi_kernel_utils.hpp"
#include <sycl/sycl.hpp>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor {
namespace oneapi {

namespace {

constexpr int MAX_INDEX_DIMS = 16;

struct AdvancedIndexMeta {
    int num_indices;
    int src_ndim;
    int num_pass_dims;
    int64_t bc_numel;
    int64_t pass_numel;
    int64_t src_shape[MAX_INDEX_DIMS];
    int64_t src_strides[MAX_INDEX_DIMS];
    int pass_dims[MAX_INDEX_DIMS];
    int is_indexed[MAX_INDEX_DIMS];
};

struct PreparedAdvancedIndex {
    AdvancedIndexMeta meta;
    std::vector<int64_t> output_shape;
    int64_t total;
};

inline PreparedAdvancedIndex prepare_advanced_index(
    const Tensor& src,
    const Tensor* const* index_tensors,
    int64_t num_indices
) {
    PreparedAdvancedIndex out{};
    auto src_shape_span = src.shape();
    int64_t src_ndim = static_cast<int64_t>(src_shape_span.size());
    if (src_ndim > MAX_INDEX_DIMS) {
        throw std::runtime_error("AdvancedIndex: source ndim exceeds MAX_INDEX_DIMS");
    }
    if (num_indices > MAX_INDEX_DIMS) {
        throw std::runtime_error("AdvancedIndex: num_indices exceeds MAX_INDEX_DIMS");
    }

    out.meta.num_indices = static_cast<int>(num_indices);
    out.meta.src_ndim = static_cast<int>(src_ndim);

    for (int64_t i = 0; i < src_ndim; ++i) {
        out.meta.src_shape[i] = src_shape_span[i];
    }
    out.meta.src_strides[src_ndim - 1] = 1;
    for (int64_t d = src_ndim - 2; d >= 0; --d) {
        out.meta.src_strides[d] = out.meta.src_strides[d + 1] * src_shape_span[d + 1];
    }

    std::vector<int64_t> broadcast_shape;
    for (int i = 0; i < num_indices; ++i) {
        if (index_tensors[i] != nullptr && index_tensors[i]->numel() > 0) {
            out.meta.is_indexed[i] = 1;
            if (broadcast_shape.empty()) {
                auto s = index_tensors[i]->shape();
                broadcast_shape.assign(s.begin(), s.end());
            }
        } else {
            out.meta.is_indexed[i] = 0;
        }
    }
    if (broadcast_shape.empty()) {
        throw std::runtime_error("AdvancedIndex: at least one index tensor required");
    }

    out.output_shape = broadcast_shape;
    int pass_count = 0;
    for (int i = 0; i < num_indices; ++i) {
        if (!out.meta.is_indexed[i]) {
            out.output_shape.push_back(src_shape_span[i]);
            out.meta.pass_dims[pass_count++] = i;
        }
    }
    for (int64_t i = num_indices; i < src_ndim; ++i) {
        out.output_shape.push_back(src_shape_span[i]);
        out.meta.pass_dims[pass_count++] = static_cast<int>(i);
    }
    out.meta.num_pass_dims = pass_count;

    out.meta.bc_numel = 1;
    for (auto d : broadcast_shape) out.meta.bc_numel *= d;
    out.meta.pass_numel = 1;
    for (int k = 0; k < pass_count; ++k) {
        out.meta.pass_numel *= src_shape_span[out.meta.pass_dims[k]];
    }
    out.total = out.meta.bc_numel * out.meta.pass_numel;
    return out;
}

// Kernel name tags
template<typename T> class AdvancedIndexGatherKernel;
template<typename T> class AdvancedIndexPutKernelTag;

template<typename T>
auto launch_advanced_index_gather(
    const Tensor& src,
    const Tensor* const* index_tensors,
    int64_t num_indices,
    sycl::queue& queue
) -> Tensor {
    auto prep = prepare_advanced_index(src, index_tensors, num_indices);
    Tensor src_contig = src.contiguous();
    Tensor result(prep.output_shape, src.dtype(), src.device());
    if (prep.total == 0) return result;

    // Make each indexed tensor contiguous and collect device pointers
    std::vector<const int64_t*> host_ptrs(num_indices, nullptr);
    std::vector<Tensor> idx_contig(num_indices);
    for (int i = 0; i < num_indices; ++i) {
        if (prep.meta.is_indexed[i]) {
            idx_contig[i] = index_tensors[i]->contiguous();
            host_ptrs[i] = get_data_ptr<const int64_t>(idx_contig[i]);

            // Validate indices host-side and RAISE on out-of-range (matching the
            // CPU reference / PyTorch). The gather kernel below clamped OOB
            // indices into range, silently producing a wrong result instead of
            // an IndexError.
            const int64_t dim_size = prep.meta.src_shape[i];
            const int64_t n = idx_contig[i].numel();
            if (n > 0) {
                std::vector<int64_t> host_idx(static_cast<size_t>(n));
                queue.memcpy(host_idx.data(), host_ptrs[i],
                             static_cast<size_t>(n) * sizeof(int64_t)).wait();
                for (int64_t k = 0; k < n; ++k) {
                    int64_t v = host_idx[k];
                    if (v < 0) v += dim_size;
                    if (v < 0 || v >= dim_size) {
                        throw std::out_of_range(
                            "advanced indexing: index out of range for dimension");
                    }
                }
            }
        }
    }

    // Copy pointer array to device via USM
    const int64_t** d_idx_ptrs = sycl::malloc_device<const int64_t*>(
        static_cast<size_t>(num_indices), queue);
    queue.memcpy(d_idx_ptrs, host_ptrs.data(),
                 static_cast<size_t>(num_indices) * sizeof(const int64_t*)).wait();

    const T* src_ptr = get_data_ptr<const T>(src_contig);
    T* dst_ptr = get_data_ptr<T>(result);
    AdvancedIndexMeta meta = prep.meta;
    const int64_t total = prep.total;

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<AdvancedIndexGatherKernel<T>>(
            sycl::range<1>(static_cast<size_t>(total)),
            [=](sycl::id<1> gid) {
                int64_t out_idx = static_cast<int64_t>(gid[0]);
                int64_t p = (meta.pass_numel > 0) ? (out_idx % meta.pass_numel) : 0;
                int64_t bc = (meta.pass_numel > 0) ? (out_idx / meta.pass_numel) : out_idx;

                int64_t src_offset = 0;
                for (int i = 0; i < meta.num_indices; ++i) {
                    if (meta.is_indexed[i]) {
                        int64_t idx_val = d_idx_ptrs[i][bc];
                        if (idx_val < 0) idx_val += meta.src_shape[i];
                        // Bounds guard: an OOB user index (still <0 after one wrap,
                        // or >= dim size) would dereference out-of-bounds device
                        // memory. Clamp into range to keep the read in-bounds.
                        if (idx_val < 0) idx_val = 0;
                        else if (idx_val >= meta.src_shape[i]) idx_val = meta.src_shape[i] - 1;
                        src_offset += idx_val * meta.src_strides[i];
                    }
                }
                if (meta.num_pass_dims > 0) {
                    int64_t remaining = p;
                    for (int k = meta.num_pass_dims - 1; k >= 0; --k) {
                        int d = meta.pass_dims[k];
                        int64_t coord = remaining % meta.src_shape[d];
                        remaining /= meta.src_shape[d];
                        src_offset += coord * meta.src_strides[d];
                    }
                }
                dst_ptr[out_idx] = src_ptr[src_offset];
            });
    }).wait();

    sycl::free(d_idx_ptrs, queue);
    return result;
}

template<typename T>
auto launch_advanced_index_put(
    const Tensor& src,
    const Tensor& values,
    const Tensor* const* index_tensors,
    int64_t num_indices,
    sycl::queue& queue
) -> Tensor {
    auto prep = prepare_advanced_index(src, index_tensors, num_indices);
    Tensor result = src.clone();
    Tensor result_contig = result.contiguous();
    Tensor values_contig = values.contiguous();
    if (prep.total == 0) return result_contig;

    std::vector<const int64_t*> host_ptrs(num_indices, nullptr);
    std::vector<Tensor> idx_contig(num_indices);
    for (int i = 0; i < num_indices; ++i) {
        if (prep.meta.is_indexed[i]) {
            idx_contig[i] = index_tensors[i]->contiguous();
            host_ptrs[i] = get_data_ptr<const int64_t>(idx_contig[i]);

            // Validate indices host-side and RAISE on out-of-range (matching the
            // CPU reference / PyTorch). The put kernel below clamped OOB indices,
            // silently scattering to the wrong position instead of raising.
            const int64_t dim_size = prep.meta.src_shape[i];
            const int64_t n = idx_contig[i].numel();
            if (n > 0) {
                std::vector<int64_t> host_idx(static_cast<size_t>(n));
                queue.memcpy(host_idx.data(), host_ptrs[i],
                             static_cast<size_t>(n) * sizeof(int64_t)).wait();
                for (int64_t k = 0; k < n; ++k) {
                    int64_t v = host_idx[k];
                    if (v < 0) v += dim_size;
                    if (v < 0 || v >= dim_size) {
                        throw std::out_of_range(
                            "advanced indexing (put): index out of range for dimension");
                    }
                }
            }
        }
    }

    const int64_t** d_idx_ptrs = sycl::malloc_device<const int64_t*>(
        static_cast<size_t>(num_indices), queue);
    queue.memcpy(d_idx_ptrs, host_ptrs.data(),
                 static_cast<size_t>(num_indices) * sizeof(const int64_t*)).wait();

    T* dst_ptr = get_data_ptr<T>(result_contig);
    const T* val_ptr = get_data_ptr<const T>(values_contig);
    AdvancedIndexMeta meta = prep.meta;
    const int64_t total = prep.total;

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<AdvancedIndexPutKernelTag<T>>(
            sycl::range<1>(static_cast<size_t>(total)),
            [=](sycl::id<1> gid) {
                int64_t out_idx = static_cast<int64_t>(gid[0]);
                int64_t p = (meta.pass_numel > 0) ? (out_idx % meta.pass_numel) : 0;
                int64_t bc = (meta.pass_numel > 0) ? (out_idx / meta.pass_numel) : out_idx;

                int64_t dst_offset = 0;
                for (int i = 0; i < meta.num_indices; ++i) {
                    if (meta.is_indexed[i]) {
                        int64_t idx_val = d_idx_ptrs[i][bc];
                        if (idx_val < 0) idx_val += meta.src_shape[i];
                        // Bounds guard: an OOB user index would write out-of-bounds
                        // device memory. Clamp into range to keep the write in-bounds.
                        if (idx_val < 0) idx_val = 0;
                        else if (idx_val >= meta.src_shape[i]) idx_val = meta.src_shape[i] - 1;
                        dst_offset += idx_val * meta.src_strides[i];
                    }
                }
                if (meta.num_pass_dims > 0) {
                    int64_t remaining = p;
                    for (int k = meta.num_pass_dims - 1; k >= 0; --k) {
                        int d = meta.pass_dims[k];
                        int64_t coord = remaining % meta.src_shape[d];
                        remaining /= meta.src_shape[d];
                        dst_offset += coord * meta.src_strides[d];
                    }
                }
                dst_ptr[dst_offset] = val_ptr[out_idx];
            });
    }).wait();

    sycl::free(d_idx_ptrs, queue);
    return result_contig;
}

}  // namespace

auto advanced_index_oneapi_kernel(
    const Tensor& src,
    const std::vector<Tensor>& indices,
    int64_t num_indices,
    sycl::queue& queue
) -> Tensor {
    std::vector<const Tensor*> idx_ptrs(num_indices);
    for (int64_t i = 0; i < num_indices; ++i) idx_ptrs[i] = &indices[i];

    switch (src.dtype()) {
        case DType::Float32:
            return launch_advanced_index_gather<float>(src, idx_ptrs.data(), num_indices, queue);
        case DType::Float64:
            return launch_advanced_index_gather<double>(src, idx_ptrs.data(), num_indices, queue);
        case DType::Int32:
            return launch_advanced_index_gather<int32_t>(src, idx_ptrs.data(), num_indices, queue);
        case DType::Int64:
            return launch_advanced_index_gather<int64_t>(src, idx_ptrs.data(), num_indices, queue);
        case DType::Float16:
        case DType::BFloat16:
            // Pure gather — treat 16-bit floats as raw uint16_t for bit-copy
            return launch_advanced_index_gather<uint16_t>(src, idx_ptrs.data(), num_indices, queue);
        default:
            throw std::runtime_error("AdvancedIndex OneAPI: unsupported dtype");
    }
}

auto advanced_index_put_oneapi_kernel(
    const Tensor& src,
    const std::vector<Tensor>& indices,
    const Tensor& values,
    int64_t num_indices,
    sycl::queue& queue
) -> Tensor {
    std::vector<const Tensor*> idx_ptrs(num_indices);
    for (int64_t i = 0; i < num_indices; ++i) idx_ptrs[i] = &indices[i];

    switch (src.dtype()) {
        case DType::Float32:
            return launch_advanced_index_put<float>(src, values, idx_ptrs.data(), num_indices, queue);
        case DType::Float64:
            return launch_advanced_index_put<double>(src, values, idx_ptrs.data(), num_indices, queue);
        case DType::Int32:
            return launch_advanced_index_put<int32_t>(src, values, idx_ptrs.data(), num_indices, queue);
        case DType::Int64:
            return launch_advanced_index_put<int64_t>(src, values, idx_ptrs.data(), num_indices, queue);
        case DType::Float16:
        case DType::BFloat16:
            return launch_advanced_index_put<uint16_t>(src, values, idx_ptrs.data(), num_indices, queue);
        default:
            throw std::runtime_error("AdvancedIndexPut OneAPI: unsupported dtype");
    }
}

}  // namespace oneapi
}  // namespace tenzor
