/**
 * @file vulkan_ops_advanced_index.cpp
 * @brief Native Vulkan dispatch for AdvancedIndex / AdvancedIndexPut
 *        (NumPy-style fancy indexing). Replaces CPU fallback.
 *
 * Algorithm mirrors src/backends/cuda/kernels/advanced.cu. One thread per
 * output element decodes (broadcast_idx, pass_idx), reads per-dim index
 * values from a packed int64 index SSBO, applies negative-index wrapping,
 * and performs a raw-bit copy from src to dst (or vals to dst for put).
 *
 * Meta layout (int64 SSBO, shared by all shader variants):
 *   [0]             num_indices
 *   [1]             src_ndim
 *   [2]             num_pass_dims
 *   [3]             total_out
 *   [4]             pass_numel
 *   [5 .. 20]       src_shape[16]
 *   [21 .. 36]      src_strides[16]
 *   [37 .. 52]      pass_dims[16]
 *   [53 .. 68]      is_indexed[16]
 *   [69 .. 84]      idx_offset[16]   (start offset into packed index buffer)
 */

#include "vulkan_ops_common.hpp"

namespace tenzor {

namespace {

constexpr int kMaxIndexDims = 16;
constexpr int kMetaInt64Count = 85;  // 5 + 16*5

struct PreparedMeta {
    std::array<int64_t, kMetaInt64Count> meta{};
    std::vector<int64_t> output_shape;
    int64_t total{0};
    int64_t packed_idx_count{0};
    std::array<int64_t, kMaxIndexDims> idx_offset{};
    std::array<int, kMaxIndexDims> is_indexed{};
};

PreparedMeta prepare_meta(const Tensor& src, const std::vector<Tensor>& indices,
                          int64_t num_indices) {
    PreparedMeta prep;
    auto src_shape = src.shape();
    int64_t src_ndim = static_cast<int64_t>(src_shape.size());
    if (src_ndim > kMaxIndexDims) {
        throw std::runtime_error("Vulkan AdvancedIndex: source ndim exceeds 16");
    }
    if (num_indices > kMaxIndexDims) {
        throw std::runtime_error("Vulkan AdvancedIndex: num_indices exceeds 16");
    }

    prep.meta[0] = num_indices;          // num_indices
    prep.meta[1] = src_ndim;              // src_ndim
    // meta[2] num_pass_dims filled in below
    // meta[3] total_out filled in below
    // meta[4] pass_numel filled in below

    // src_shape[16] at offset 5
    for (int64_t i = 0; i < src_ndim; ++i) {
        prep.meta[5 + i] = src_shape[i];
    }
    // src_strides[16] at offset 21 — row-major
    if (src_ndim > 0) {
        prep.meta[21 + (src_ndim - 1)] = 1;
        for (int64_t d = src_ndim - 2; d >= 0; --d) {
            prep.meta[21 + d] = prep.meta[21 + d + 1] * src_shape[d + 1];
        }
    }

    // Identify is_indexed[] and broadcast shape
    std::vector<int64_t> broadcast_shape;
    for (int i = 0; i < num_indices; ++i) {
        if (i < static_cast<int>(indices.size()) && indices[i].numel() > 0) {
            prep.is_indexed[i] = 1;
            prep.meta[53 + i] = 1;  // is_indexed
            if (broadcast_shape.empty()) {
                auto s = indices[i].shape();
                broadcast_shape.assign(s.begin(), s.end());
            }
        } else {
            prep.is_indexed[i] = 0;
            prep.meta[53 + i] = 0;
        }
    }
    if (broadcast_shape.empty()) {
        throw std::runtime_error("Vulkan AdvancedIndex: at least one index tensor required");
    }

    // Output shape = broadcast_shape + passthrough dims
    prep.output_shape = broadcast_shape;
    int pass_count = 0;
    for (int i = 0; i < num_indices; ++i) {
        if (!prep.is_indexed[i]) {
            prep.output_shape.push_back(src_shape[i]);
            prep.meta[37 + pass_count] = i;  // pass_dims
            ++pass_count;
        }
    }
    for (int64_t i = num_indices; i < src_ndim; ++i) {
        prep.output_shape.push_back(src_shape[i]);
        prep.meta[37 + pass_count] = i;
        ++pass_count;
    }
    prep.meta[2] = pass_count;

    int64_t bc_numel = 1;
    for (auto d : broadcast_shape) bc_numel *= d;
    int64_t pass_numel = 1;
    for (int k = 0; k < pass_count; ++k) {
        pass_numel *= src_shape[prep.meta[37 + k]];
    }
    prep.meta[4] = pass_numel;
    prep.total = bc_numel * pass_numel;
    prep.meta[3] = prep.total;

    // Build idx_offset[] — each indexed dim stores bc_numel int64 values in the
    // packed index buffer, laid out back-to-back in dim order.
    int64_t running = 0;
    for (int i = 0; i < num_indices; ++i) {
        prep.meta[69 + i] = running;
        prep.idx_offset[i] = running;
        if (prep.is_indexed[i]) running += bc_numel;
    }
    prep.packed_idx_count = running;

    return prep;
}

// Advanced index gather/put is pure data movement, so the shader is selected by
// element BYTE WIDTH, not numeric type — every same-width dtype shares one shader
// (the f16/bf16 shaders are pure 16-bit bit-moves, _f64 a 64-bit move, etc.).
// This gives full dtype coverage including the sub-32-bit integer and Bool types
// that previously threw "unsupported dtype".
const char* select_gather_shader(DType dt) {
    switch (dt) {
        case DType::Int8: case DType::UInt8: case DType::Bool:
            return "advanced_index_gather_i8";    // 1-byte
        case DType::Float16: case DType::BFloat16:
        case DType::Int16: case DType::UInt16:
            return "advanced_index_gather_f16";   // 2-byte (pure 16-bit move)
        case DType::Float32: case DType::Int32: case DType::UInt32:
            return "advanced_index_gather";       // 4-byte
        case DType::Float64: case DType::Int64: case DType::UInt64:
            return "advanced_index_gather_f64";   // 8-byte
        default: break;
    }
    throw std::runtime_error("Vulkan AdvancedIndex: unsupported dtype");
}

const char* select_put_shader(DType dt) {
    switch (dt) {
        case DType::Int8: case DType::UInt8: case DType::Bool:
            return "advanced_index_put_i8";       // 1-byte
        case DType::Float16: case DType::BFloat16:
        case DType::Int16: case DType::UInt16:
            return "advanced_index_put_f16";      // 2-byte (pure 16-bit move)
        case DType::Float32: case DType::Int32: case DType::UInt32:
            return "advanced_index_put";          // 4-byte
        case DType::Float64: case DType::Int64: case DType::UInt64:
            return "advanced_index_put_f64";      // 8-byte
        default: break;
    }
    throw std::runtime_error("Vulkan AdvancedIndexPut: unsupported dtype");
}

}  // namespace

auto VulkanBackend::dispatchAdvancedIndex(const Tensor& src,
                                          const std::vector<Tensor>& indices,
                                          int64_t num_indices) -> Tensor {
    auto prep = prepare_meta(src, indices, num_indices);
    Tensor src_contig = src.contiguous();
    Tensor output(prep.output_shape, src.dtype(), src.device());

    if (prep.total == 0) return output;

    int32_t device_id = src.device().index;

    // Upload meta buffer (int64 storage — use Int64 tensor)
    Tensor meta_buf({static_cast<int64_t>(kMetaInt64Count)}, DType::Int64, src.device());
    copy(meta_buf.data_ptr(), prep.meta.data(),
         kMetaInt64Count * sizeof(int64_t), CopyKind::HostToDevice);

    // Build packed indices buffer on host, then upload.
    // Each indexed dim contributes bc_numel contiguous int64s.
    Tensor idx_buf;
    std::vector<Tensor> idx_contig_tensors(num_indices);
    if (prep.packed_idx_count > 0) {
        idx_buf = Tensor({prep.packed_idx_count}, DType::Int64, src.device());
        // Make sure each indexed tensor is contiguous (input may be expand()'d),
        // then copy its bc_numel Int64 values into the packed buffer at idx_offset[i].
        size_t element_bytes = sizeof(int64_t);
        for (int i = 0; i < num_indices; ++i) {
            if (!prep.is_indexed[i]) continue;
            Tensor contig = indices[i].contiguous();
            idx_contig_tensors[i] = contig;  // keep alive
            int64_t count = contig.numel();
            const void* src_ptr = contig.data_ptr();
            // device-to-device copy within the Vulkan device
            uint8_t* dst_base = static_cast<uint8_t*>(idx_buf.data_ptr());
            copy(dst_base + prep.idx_offset[i] * element_bytes,
                 src_ptr, count * element_bytes, CopyKind::DeviceToDevice);
        }
    } else {
        // Still need a non-null buffer for binding (any Int64 tensor works).
        idx_buf = Tensor({1}, DType::Int64, src.device());
    }

    synchronize(device_id);

    const char* shader_name = select_gather_shader(src.dtype());
    auto* pipeline = getPipeline(shader_name, device_id);

    size_t src_bytes = src_contig.numel() * src_contig.dtype_size();
    size_t dst_bytes = output.numel() * output.dtype_size();
    size_t meta_bytes = kMetaInt64Count * sizeof(int64_t);
    size_t idx_bytes = static_cast<size_t>(idx_buf.numel()) * sizeof(int64_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, src_contig.data_ptr()},
        {1, output.data_ptr()},
        {2, meta_buf.data_ptr()},
        {3, idx_buf.data_ptr()},
    };
    std::vector<size_t> sizes = {src_bytes, dst_bytes, meta_bytes, idx_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    uint32_t groups = div_wg(static_cast<uint64_t>(prep.total), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, groups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return output;
}

auto VulkanBackend::dispatchAdvancedIndexPut(const Tensor& src,
                                             const std::vector<Tensor>& indices,
                                             const Tensor& values,
                                             int64_t num_indices) -> Tensor {
    auto prep = prepare_meta(src, indices, num_indices);

    // Out-of-place semantics: clone src into a new contiguous tensor.
    Tensor result = src.contiguous().clone();
    Tensor values_contig = values.contiguous();

    if (prep.total == 0) return result;

    int32_t device_id = src.device().index;

    Tensor meta_buf({static_cast<int64_t>(kMetaInt64Count)}, DType::Int64, src.device());
    copy(meta_buf.data_ptr(), prep.meta.data(),
         kMetaInt64Count * sizeof(int64_t), CopyKind::HostToDevice);

    Tensor idx_buf;
    std::vector<Tensor> idx_contig_tensors(num_indices);
    if (prep.packed_idx_count > 0) {
        idx_buf = Tensor({prep.packed_idx_count}, DType::Int64, src.device());
        size_t element_bytes = sizeof(int64_t);
        for (int i = 0; i < num_indices; ++i) {
            if (!prep.is_indexed[i]) continue;
            Tensor contig = indices[i].contiguous();
            idx_contig_tensors[i] = contig;
            int64_t count = contig.numel();
            uint8_t* dst_base = static_cast<uint8_t*>(idx_buf.data_ptr());
            copy(dst_base + prep.idx_offset[i] * element_bytes,
                 contig.data_ptr(), count * element_bytes, CopyKind::DeviceToDevice);
        }
    } else {
        idx_buf = Tensor({1}, DType::Int64, src.device());
    }

    synchronize(device_id);

    // Contract (shared with CPU/CUDA backends): the put shader reads
    // vals_data[gid] for every output element gid in [0, prep.total), so the
    // values buffer MUST already contain at least prep.total elements. The op
    // layer is responsible for expanding/broadcasting `values` to the full
    // output element layout before dispatch. Guard here to turn a silent
    // out-of-bounds SSBO read (garbage writes / validation errors) into a
    // deterministic error if that contract is violated.
    if (values_contig.numel() < prep.total) {
        throw std::runtime_error(
            "Vulkan AdvancedIndexPut: values has " +
            std::to_string(values_contig.numel()) +
            " elements but the indexed assignment requires " +
            std::to_string(prep.total) +
            " (values must be broadcast/expanded to the output layout before dispatch)");
    }

    const char* shader_name = select_put_shader(src.dtype());
    auto* pipeline = getPipeline(shader_name, device_id);

    size_t result_bytes = result.numel() * result.dtype_size();
    size_t values_bytes = values_contig.numel() * values_contig.dtype_size();
    size_t meta_bytes = kMetaInt64Count * sizeof(int64_t);
    size_t idx_bytes = static_cast<size_t>(idx_buf.numel()) * sizeof(int64_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, result.data_ptr()},
        {1, values_contig.data_ptr()},
        {2, meta_buf.data_ptr()},
        {3, idx_buf.data_ptr()},
    };
    std::vector<size_t> sizes = {result_bytes, values_bytes, meta_bytes, idx_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    uint32_t groups = div_wg(static_cast<uint64_t>(prep.total), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, groups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return result;
}

}  // namespace tenzor
