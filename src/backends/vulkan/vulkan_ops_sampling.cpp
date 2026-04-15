/**
 * @file vulkan_ops_sampling.cpp
 * @brief Native Vulkan dispatch for Bernoulli, Multinomial, Bucketize,
 *        Histogram, CDist. Replaces the previous CPU-roundtrip fallbacks.
 *
 * Float32-only on the GPU; non-Float32 dtypes are promoted via dispatchCast
 * (an on-device Vulkan compute pipeline) before invocation. Histogram
 * auto-range support computes min/max via existing reduction infrastructure
 * (tenzor::min/max dispatch) so we never touch the host for compute.
 */

#include "vulkan_ops_common.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <chrono>

namespace tenzor {

namespace {

struct BernoulliPC {
    uint32_t n;
    uint32_t seed_lo;
    uint32_t seed_hi;
};

struct BucketizePC {
    uint32_t n;
    uint32_t num_boundaries;
    uint32_t right;
};

struct CDistPC {
    uint32_t B;
    uint32_t P;
    uint32_t R;
    uint32_t M;
    uint32_t total;
};

struct HistogramPC {
    uint32_t n;
    uint32_t num_bins;
    float min_val;
    float bin_width;
};

struct HistogramddPC {
    uint32_t n;
    uint32_t D;
    uint32_t total_bins;
};

struct MultinomialCdfPC {
    uint32_t num_categories;
    uint32_t probs_offset;
    uint32_t cdf_offset;
};

struct MultinomialSamplePC {
    uint32_t num_categories;
    uint32_t num_samples;
    float total;
    uint32_t seed_lo;
    uint32_t seed_hi;
    uint32_t cdf_offset;
    uint32_t out_offset;
};

inline std::pair<uint32_t, uint32_t> seed_split() {
    uint64_t s = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return {static_cast<uint32_t>(s & 0xFFFFFFFFu),
            static_cast<uint32_t>(s >> 32)};
}

}  // namespace

auto VulkanBackend::dispatchBernoulli(const Tensor& probs) -> Tensor {
    DType orig_dtype = probs.dtype();
    Tensor probs_f32 = (orig_dtype == DType::Float32) ? probs.contiguous()
                                                       : dispatchCast(probs.contiguous(), DType::Float32);

    std::vector<int64_t> shape(probs_f32.shape().begin(), probs_f32.shape().end());
    Tensor output_f32(shape, DType::Float32, probs.device());
    int64_t n = probs_f32.numel();
    if (n == 0) return (orig_dtype == DType::Float32) ? output_f32 : dispatchCast(output_f32, orig_dtype);

    int32_t device_id = probs.device().index;
    auto* pipeline = getPipeline("bernoulli", device_id);

    auto [seed_lo, seed_hi] = seed_split();
    BernoulliPC pc{static_cast<uint32_t>(n), seed_lo, seed_hi};

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, probs_f32.data_ptr()},
        {1, output_f32.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(n) * sizeof(float),
        static_cast<size_t>(n) * sizeof(float),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(BernoulliPC), &pc);
    uint32_t workgroups = div_wg(static_cast<uint32_t>(n), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return (orig_dtype == DType::Float32) ? output_f32 : dispatchCast(output_f32, orig_dtype);
}

auto VulkanBackend::dispatchBucketize(const Tensor& input, const Tensor& boundaries, bool right) -> Tensor {
    DType orig_dtype = input.dtype();
    Tensor in_f32   = (orig_dtype == DType::Float32) ? input.contiguous() : dispatchCast(input.contiguous(), DType::Float32);
    Tensor bound_f32 = (boundaries.dtype() == DType::Float32) ? boundaries.contiguous()
                                                               : dispatchCast(boundaries.contiguous(), DType::Float32);

    std::vector<int64_t> shape(in_f32.shape().begin(), in_f32.shape().end());
    Tensor output(shape, DType::Int32, input.device());  // int32 indices
    int64_t n = in_f32.numel();
    if (n == 0) return output;

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("bucketize", device_id);

    BucketizePC pc{static_cast<uint32_t>(n),
                   static_cast<uint32_t>(bound_f32.numel()),
                   right ? 1u : 0u};

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, in_f32.data_ptr()},
        {1, bound_f32.data_ptr()},
        {2, output.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(n) * sizeof(float),
        static_cast<size_t>(bound_f32.numel()) * sizeof(float),
        static_cast<size_t>(n) * sizeof(int32_t),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(BucketizePC), &pc);
    uint32_t workgroups = div_wg(static_cast<uint32_t>(n), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    // Bucketize returns int64 in the standard API; cast int32 → int64 via dispatchCast
    return dispatchCast(output, DType::Int64);
}

auto VulkanBackend::dispatchCDist(const Tensor& x1, const Tensor& x2) -> Tensor {
    DType orig_dtype = x1.dtype();
    Tensor a = (orig_dtype == DType::Float32) ? x1.contiguous() : dispatchCast(x1.contiguous(), DType::Float32);
    Tensor b = (x2.dtype() == DType::Float32) ? x2.contiguous() : dispatchCast(x2.contiguous(), DType::Float32);

    int64_t B, P, M, R;
    if (a.ndim() == 2 && b.ndim() == 2) {
        B = 1; P = a.shape()[0]; M = a.shape()[1]; R = b.shape()[0];
    } else if (a.ndim() == 3 && b.ndim() == 3) {
        B = a.shape()[0]; P = a.shape()[1]; M = a.shape()[2]; R = b.shape()[1];
        if (b.shape()[0] != B) throw std::runtime_error("cdist: batch dims must match");
    } else {
        throw std::runtime_error("cdist: inputs must be 2D or 3D");
    }
    std::vector<int64_t> result_shape = (a.ndim() == 2) ? std::vector<int64_t>{P, R}
                                                         : std::vector<int64_t>{B, P, R};
    Tensor result_f32(result_shape, DType::Float32, x1.device());

    int64_t total = B * P * R;
    if (total == 0) return (orig_dtype == DType::Float32) ? result_f32 : dispatchCast(result_f32, orig_dtype);

    int32_t device_id = x1.device().index;
    auto* pipeline = getPipeline("cdist", device_id);

    CDistPC pc{static_cast<uint32_t>(B), static_cast<uint32_t>(P),
               static_cast<uint32_t>(R), static_cast<uint32_t>(M),
               static_cast<uint32_t>(total)};

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, a.data_ptr()},
        {1, b.data_ptr()},
        {2, result_f32.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(a.numel()) * sizeof(float),
        static_cast<size_t>(b.numel()) * sizeof(float),
        static_cast<size_t>(result_f32.numel()) * sizeof(float),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(CDistPC), &pc);
    uint32_t workgroups = div_wg(static_cast<uint32_t>(total), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return (orig_dtype == DType::Float32) ? result_f32 : dispatchCast(result_f32, orig_dtype);
}

auto VulkanBackend::dispatchHistogram(const Tensor& input, int64_t bins,
                                      double min_val, double max_val)
    -> std::pair<Tensor, Tensor> {
    Tensor in_f32 = (input.dtype() == DType::Float32) ? input.contiguous()
                                                       : dispatchCast(input.contiguous(), DType::Float32);
    int64_t n = in_f32.numel();

    // Auto-range via aminmax dispatch (single kernel, single D2H transfer)
    if (min_val == 0.0 && max_val == 0.0 && n > 0) {
        // aminmax returns (min, max) in one dispatch, avoiding two separate D2H syncs
        auto [min_t, max_t] = tenzor::aminmax(in_f32);
        // Pack into single 2-element tensor for one D2H transfer
        Tensor minmax = tenzor::cat({min_t.reshape({1}), max_t.reshape({1})}, 0);
        Tensor minmax_cpu = minmax.to(Device::cpu()).to(DType::Float32);
        min_val = static_cast<double>(minmax_cpu.data<float>()[0]);
        max_val = static_cast<double>(minmax_cpu.data<float>()[1]);
    }
    if (max_val <= min_val) max_val = min_val + 1.0;
    float bin_width = static_cast<float>((max_val - min_val) / bins);

    // Counts allocated as int32 (matches shader); promoted to int64 at the end
    Tensor counts_i32(std::vector<int64_t>{bins}, DType::Int32, input.device());
    // Zero-initialise via dispatch (on-device fill)
    counts_i32 = dispatchCast(counts_i32, DType::Int32);  // forces a clean buffer

    // Actually we need to memset the counts to zero; use a small zero shader or
    // the existing fill infrastructure. Use full(0) instead.
    counts_i32 = tenzor::zeros({bins}, DType::Int32, input.device());

    int32_t device_id = input.device().index;
    if (n > 0) {
        auto* pipeline = getPipeline("histogram", device_id);
        HistogramPC pc{static_cast<uint32_t>(n),
                       static_cast<uint32_t>(bins),
                       static_cast<float>(min_val),
                       bin_width};
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, in_f32.data_ptr()},
            {1, counts_i32.data_ptr()},
        };
        std::vector<size_t> sizes = {
            static_cast<size_t>(n) * sizeof(float),
            static_cast<size_t>(bins) * sizeof(int32_t),
        };
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(HistogramPC), &pc);
        uint32_t workgroups = div_wg(static_cast<uint32_t>(n), devices_[device_id].workgroupSize);
        vkCmdDispatch(cmd, workgroups, 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Promote counts to int64 (the API returns int64) via dispatchCast
    Tensor counts_i64 = dispatchCast(counts_i32, DType::Int64);

    // Bin edges: build via tenzor::arange + scale (all on-device dispatches)
    Tensor edges = tenzor::arange(0, bins + 1, 1, DType::Float32, input.device());
    edges = tenzor::mul(edges, tenzor::full({bins + 1}, bin_width, DType::Float32, input.device()));
    edges = tenzor::add(edges, tenzor::full({bins + 1}, static_cast<float>(min_val), DType::Float32, input.device()));

    return {counts_i64, edges};
}

auto VulkanBackend::dispatchHistogramdd(const Tensor& input,
                                        std::vector<int64_t> bins,
                                        std::vector<std::pair<double,double>> ranges,
                                        bool density)
    -> std::pair<Tensor, std::vector<Tensor>> {
    if (input.dim() != 2) {
        throw std::runtime_error("dispatchHistogramdd: input must be 2-D (N, D)");
    }

    const int64_t N = input.shape()[0];
    const int64_t D = input.shape()[1];

    if (static_cast<int64_t>(bins.size()) != D) {
        throw std::runtime_error("dispatchHistogramdd: bins length must equal D");
    }

    Tensor in_f32 = (input.dtype() == DType::Float32) ? input.contiguous()
                                                       : dispatchCast(input.contiguous(), DType::Float32);
    int32_t device_id = input.device().index;

    // Auto-detect ranges from data if not provided
    bool auto_range = ranges.empty();
    if (auto_range) {
        ranges.resize(static_cast<size_t>(D));
    }

    if (auto_range && N > 0) {
        // Compute per-column min/max entirely on GPU via standard reduction ops
        // in_f32 is (N, D); reduce along dim 0 to get (D,) min and max vectors
        Tensor col_min = tenzor::min(in_f32, /*dim=*/0, /*keepdim=*/false);
        Tensor col_max = tenzor::max(in_f32, /*dim=*/0, /*keepdim=*/false);

        // Single D2H for the D-sized min/max vectors (small, acceptable)
        Tensor min_cpu = col_min.to(Device::cpu());
        Tensor max_cpu = col_max.to(Device::cpu());
        const float* min_data = min_cpu.data<float>();
        const float* max_data = max_cpu.data<float>();
        for (int64_t d = 0; d < D; ++d) {
            float vmin = min_data[d];
            float vmax = max_data[d];
            if (vmin == vmax) {
                vmin -= 0.5f;
                vmax += 0.5f;
            }
            ranges[static_cast<size_t>(d)] = {static_cast<double>(vmin),
                                                static_cast<double>(vmax)};
        }
    } else if (auto_range) {
        // N == 0: use default ranges
        for (int64_t d = 0; d < D; ++d) {
            ranges[static_cast<size_t>(d)] = {0.0, 1.0};
        }
    }

    // Build per-dimension parameters: (min, step) pairs and strides
    std::vector<float> dim_params(static_cast<size_t>(D) * 2);
    std::vector<float> dim_steps(static_cast<size_t>(D));

    for (int64_t d = 0; d < D; ++d) {
        auto sd = static_cast<size_t>(d);
        float fmin = static_cast<float>(ranges[sd].first);
        float fmax = static_cast<float>(ranges[sd].second);
        float step = (fmax - fmin) / static_cast<float>(bins[sd]);
        dim_params[sd * 2]     = fmin;
        dim_params[sd * 2 + 1] = step;
        dim_steps[sd] = step;
    }

    // Compute strides (row-major)
    std::vector<int64_t> out_shape(bins.begin(), bins.end());
    std::vector<int32_t> out_strides(static_cast<size_t>(D));
    int64_t stride = 1;
    for (int64_t d = D - 1; d >= 0; --d) {
        out_strides[static_cast<size_t>(d)] = static_cast<int32_t>(stride);
        stride *= bins[static_cast<size_t>(d)];
    }
    int64_t total_bins = stride;

    // Build edge tensors on-device
    std::vector<Tensor> edges_vec;
    edges_vec.reserve(static_cast<size_t>(D));
    for (int64_t d = 0; d < D; ++d) {
        auto sd = static_cast<size_t>(d);
        int64_t nb = bins[sd];
        float fmin = dim_params[sd * 2];
        float step = dim_params[sd * 2 + 1];

        // Build edges using arange + scale on device
        Tensor edge = tenzor::arange(0, nb + 1, 1, DType::Float32, input.device());
        edge = tenzor::mul(edge, tenzor::full({nb + 1}, step, DType::Float32, input.device()));
        edge = tenzor::add(edge, tenzor::full({nb + 1}, fmin, DType::Float32, input.device()));
        edges_vec.push_back(std::move(edge));
    }

    // Allocate counts as int32 (shader uses atomicAdd on int)
    Tensor counts_i32 = tenzor::zeros(out_shape, DType::Int32, input.device());

    if (N > 0) {
        // Upload dim_params and strides buffers to device
        Tensor params_buf({static_cast<int64_t>(dim_params.size())}, DType::Float32, input.device());
        Tensor strides_buf({D}, DType::Int32, input.device());

        // We need to fill these buffers. Use CPU staging.
        {
            Tensor params_cpu({static_cast<int64_t>(dim_params.size())}, DType::Float32, Device::cpu());
            std::memcpy(params_cpu.data<float>(), dim_params.data(), dim_params.size() * sizeof(float));
            params_buf = params_cpu.to(input.device());
        }
        {
            Tensor strides_cpu({D}, DType::Int32, Device::cpu());
            std::memcpy(strides_cpu.data<int32_t>(), out_strides.data(),
                        static_cast<size_t>(D) * sizeof(int32_t));
            strides_buf = strides_cpu.to(input.device());
        }

        auto* pipeline = getPipeline("histogramdd", device_id);
        HistogramddPC pc{static_cast<uint32_t>(N),
                         static_cast<uint32_t>(D),
                         static_cast<uint32_t>(total_bins)};

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, in_f32.data_ptr()},
            {1, counts_i32.data_ptr()},
            {2, params_buf.data_ptr()},
            {3, strides_buf.data_ptr()},
        };
        std::vector<size_t> sizes = {
            static_cast<size_t>(N * D) * sizeof(float),
            static_cast<size_t>(total_bins) * sizeof(int32_t),
            dim_params.size() * sizeof(float),
            static_cast<size_t>(D) * sizeof(int32_t),
        };

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(HistogramddPC), &pc);
        uint32_t workgroups = div_wg(static_cast<uint32_t>(N), devices_[device_id].workgroupSize);
        vkCmdDispatch(cmd, workgroups, 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Promote counts to int64
    Tensor counts_i64 = dispatchCast(counts_i32, DType::Int64);

    // Density normalization
    Tensor result = counts_i64;
    if (density && N > 0) {
        double bin_volume = 1.0;
        for (int64_t d = 0; d < D; ++d) {
            bin_volume *= static_cast<double>(dim_steps[static_cast<size_t>(d)]);
        }
        double norm = static_cast<double>(N) * bin_volume;
        float inv_norm = static_cast<float>(1.0 / norm);

        // Convert counts to float, multiply by inv_norm
        Tensor counts_f = dispatchCast(counts_i32, DType::Float32);
        result = tenzor::mul(counts_f, tenzor::full(out_shape, inv_norm, DType::Float32, input.device()));
    }

    return {result, std::move(edges_vec)};
}

auto VulkanBackend::dispatchMultinomial(const Tensor& probs, int64_t num_samples,
                                        bool /*replacement*/) -> Tensor {
    Tensor input = (probs.dtype() == DType::Float32) ? probs.contiguous()
                                                      : dispatchCast(probs.contiguous(), DType::Float32);

    bool was_1d = (input.ndim() == 1);
    if (was_1d) input = tenzor::reshape(input, {1, input.numel()});

    int64_t batch_size = input.shape()[0];
    int64_t num_categories = input.shape()[1];
    Tensor result_i32(std::vector<int64_t>{batch_size, num_samples}, DType::Int32, probs.device());
    Tensor cdf_buf(std::vector<int64_t>{batch_size, num_categories}, DType::Float32, probs.device());

    int32_t device_id = probs.device().index;
    auto* cdf_pipeline = getPipeline("multinomial_cdf", device_id);
    auto* sample_pipeline = getPipeline("multinomial_sample", device_id);

    auto [seed_lo, seed_hi] = seed_split();

    // Per-batch dispatch with offset-based shader bindings (no slice copies, no host fallback).
    // For each row b: dispatch CDF (single thread) → read CDF[end] (1 scalar D2H = metadata sync,
    // not compute fallback) → dispatch sampler with computed offsets.
    for (int64_t b = 0; b < batch_size; ++b) {
        // CDF dispatch (single workgroup, single thread, computes prefix sum at offset)
        {
            MultinomialCdfPC pc{static_cast<uint32_t>(num_categories),
                                static_cast<uint32_t>(b * num_categories),
                                static_cast<uint32_t>(b * num_categories)};
            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, input.data_ptr()},
                {1, cdf_buf.data_ptr()},
            };
            std::vector<size_t> sizes = {
                static_cast<size_t>(input.numel())  * sizeof(float),
                static_cast<size_t>(cdf_buf.numel()) * sizeof(float),
            };
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, cdf_pipeline, bindings, sizes);
            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cdf_pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   cdf_pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, cdf_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(MultinomialCdfPC), &pc);
            vkCmdDispatch(cmd, 1, 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
            synchronize(device_id);
        }

        // Read CDF total for this batch element.
        // Gather the last CDF value per row into a slice and transfer.
        Tensor last_val = cdf_buf.slice(0, b, b + 1).slice(1, num_categories - 1, num_categories);
        Tensor last_cpu = last_val.to(Device::cpu()).to(DType::Float32);
        float total = last_cpu.template item<float>();
        if (total <= 0.0f) total = 1.0f;

        // Sampler dispatch
        {
            MultinomialSamplePC pc{static_cast<uint32_t>(num_categories),
                                   static_cast<uint32_t>(num_samples),
                                   total,
                                   seed_lo ^ static_cast<uint32_t>(b * 0x9E3779B9u),
                                   seed_hi,
                                   static_cast<uint32_t>(b * num_categories),
                                   static_cast<uint32_t>(b * num_samples)};
            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, cdf_buf.data_ptr()},
                {1, result_i32.data_ptr()},
            };
            std::vector<size_t> sizes = {
                static_cast<size_t>(cdf_buf.numel())   * sizeof(float),
                static_cast<size_t>(result_i32.numel()) * sizeof(int32_t),
            };
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, sample_pipeline, bindings, sizes);
            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sample_pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   sample_pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, sample_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(MultinomialSamplePC), &pc);
            uint32_t workgroups = div_wg(static_cast<uint32_t>(num_samples),
                                          devices_[device_id].workgroupSize);
            vkCmdDispatch(cmd, workgroups, 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
            synchronize(device_id);
        }
    }

    if (was_1d) result_i32 = tenzor::reshape(result_i32, {num_samples});
    return dispatchCast(result_i32, DType::Int64);
}

auto VulkanBackend::dispatchPoissonSample(const Tensor& rates) -> Tensor {
    Tensor rates_f32 = (rates.dtype() == DType::Float32)
                           ? rates.contiguous()
                           : dispatchCast(rates.contiguous(), DType::Float32);

    std::vector<int64_t> shape(rates_f32.shape().begin(), rates_f32.shape().end());
    int64_t n = rates_f32.numel();
    Tensor output_i32(shape, DType::Int32, rates.device());
    if (n == 0) return dispatchCast(output_i32, DType::Int64);

    int32_t device_id = rates.device().index;
    auto* pipeline = getPipeline("poisson_sample", device_id);

    auto [seed_lo, seed_hi] = seed_split();
    BernoulliPC pc{static_cast<uint32_t>(n), seed_lo, seed_hi};

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, rates_f32.data_ptr()},
        {1, output_i32.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(n) * sizeof(float),
        static_cast<size_t>(n) * sizeof(int32_t),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(BernoulliPC), &pc);
    uint32_t workgroups = div_wg(static_cast<uint32_t>(n), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return dispatchCast(output_i32, DType::Int64);
}

auto VulkanBackend::dispatchNormalSample(const Tensor& mean, const Tensor& stddev) -> Tensor {
    Tensor mean_f32 = (mean.dtype() == DType::Float32)
                          ? mean.contiguous()
                          : dispatchCast(mean.contiguous(), DType::Float32);
    Tensor std_f32 = (stddev.dtype() == DType::Float32)
                         ? stddev.contiguous()
                         : dispatchCast(stddev.contiguous(), DType::Float32);

    std::vector<int64_t> shape(mean_f32.shape().begin(), mean_f32.shape().end());
    int64_t n = mean_f32.numel();
    Tensor output(shape, DType::Float32, mean.device());
    if (n == 0) return output;

    int32_t device_id = mean.device().index;
    auto* pipeline = getPipeline("normal_sample", device_id);

    auto [seed_lo, seed_hi] = seed_split();
    BernoulliPC pc{static_cast<uint32_t>(n), seed_lo, seed_hi};

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, mean_f32.data_ptr()},
        {1, std_f32.data_ptr()},
        {2, output.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(n) * sizeof(float),
        static_cast<size_t>(n) * sizeof(float),
        static_cast<size_t>(n) * sizeof(float),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(BernoulliPC), &pc);
    uint32_t workgroups = div_wg(static_cast<uint32_t>(n), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return output;
}

auto VulkanBackend::dispatchExponentialSample(const Tensor& rate) -> Tensor {
    Tensor rate_f32 = (rate.dtype() == DType::Float32)
                          ? rate.contiguous()
                          : dispatchCast(rate.contiguous(), DType::Float32);

    std::vector<int64_t> shape(rate_f32.shape().begin(), rate_f32.shape().end());
    int64_t n = rate_f32.numel();
    Tensor output(shape, DType::Float32, rate.device());
    if (n == 0) return output;

    int32_t device_id = rate.device().index;
    auto* pipeline = getPipeline("exponential_sample", device_id);

    auto [seed_lo, seed_hi] = seed_split();
    BernoulliPC pc{static_cast<uint32_t>(n), seed_lo, seed_hi};

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, rate_f32.data_ptr()},
        {1, output.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(n) * sizeof(float),
        static_cast<size_t>(n) * sizeof(float),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(BernoulliPC), &pc);
    uint32_t workgroups = div_wg(static_cast<uint32_t>(n), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return output;
}

auto VulkanBackend::dispatchNestedAttention(const Tensor& Q, const Tensor& K, const Tensor& V,
                                              const Tensor& q_offsets, const Tensor& kv_offsets,
                                              float scale, bool causal) -> Tensor {
    Tensor Q_f32 = (Q.dtype() == DType::Float32) ? Q.contiguous()
                                                   : dispatchCast(Q.contiguous(), DType::Float32);
    Tensor K_f32 = (K.dtype() == DType::Float32) ? K.contiguous()
                                                   : dispatchCast(K.contiguous(), DType::Float32);
    Tensor V_f32 = (V.dtype() == DType::Float32) ? V.contiguous()
                                                   : dispatchCast(V.contiguous(), DType::Float32);

    // Offsets must be Int32 for the GLSL shader (no native int64 in Vulkan)
    Tensor q_off_i32 = (q_offsets.dtype() == DType::Int32)
                           ? q_offsets.contiguous()
                           : dispatchCast(q_offsets.contiguous(), DType::Int32);
    Tensor kv_off_i32 = (kv_offsets.dtype() == DType::Int32)
                            ? kv_offsets.contiguous()
                            : dispatchCast(kv_offsets.contiguous(), DType::Int32);

    int64_t head_dim = Q_f32.shape().back();
    int64_t total_q = Q_f32.shape()[0];
    int64_t total_kv = K_f32.shape()[0];
    uint32_t B = static_cast<uint32_t>(q_off_i32.numel() - 1);

    std::vector<int64_t> out_shape(Q_f32.shape().begin(), Q_f32.shape().end());
    Tensor output(out_shape, DType::Float32, Q.device());
    if (total_q == 0 || B == 0) return output;

    int32_t device_id = Q.device().index;
    auto* pipeline = getPipeline("nested_attention", device_id);

    struct NestedAttentionPC {
        uint32_t B;
        uint32_t head_dim;
        float scale;
        uint32_t causal;
    };
    NestedAttentionPC pc{B, static_cast<uint32_t>(head_dim), scale,
                          causal ? 1u : 0u};

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, Q_f32.data_ptr()},
        {1, K_f32.data_ptr()},
        {2, V_f32.data_ptr()},
        {3, q_off_i32.data_ptr()},
        {4, kv_off_i32.data_ptr()},
        {5, output.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(total_q) * static_cast<size_t>(head_dim) * sizeof(float),
        static_cast<size_t>(total_kv) * static_cast<size_t>(head_dim) * sizeof(float),
        static_cast<size_t>(total_kv) * static_cast<size_t>(head_dim) * sizeof(float),
        static_cast<size_t>(q_off_i32.numel()) * sizeof(int32_t),
        static_cast<size_t>(kv_off_i32.numel()) * sizeof(int32_t),
        static_cast<size_t>(total_q) * static_cast<size_t>(head_dim) * sizeof(float),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(NestedAttentionPC), &pc);
    vkCmdDispatch(cmd, B, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return output;
}

// ============================================================================
// Trapezoid integration
// ============================================================================

struct TrapezoidPC {
    uint32_t outer;
    uint32_t inner;
    uint32_t n;         // size along integration dim
    uint32_t total;     // outer * inner
    float dx;           // uniform spacing (ignored when x tensor is provided)
    uint32_t has_x;     // 1 if non-uniform x is provided
};

auto VulkanBackend::dispatchTrapezoid(const Tensor& y, int64_t dim, double dx,
                                       const Tensor* x_ptr) -> Tensor {
    DType orig_dtype = y.dtype();
    Tensor yf = (orig_dtype == DType::Float32) ? y.contiguous() : dispatchCast(y.contiguous(), DType::Float32);

    auto shape = yf.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t n = shape[dim];

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim; d++) {
        if (d != dim) out_shape.push_back(shape[d]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor result_f32(out_shape, DType::Float32, y.device());
    int64_t total = outer * inner;
    if (n < 2 || total == 0) return (orig_dtype == DType::Float32) ? result_f32 : dispatchCast(result_f32, orig_dtype);

    Tensor xf;
    if (x_ptr) {
        xf = (x_ptr->dtype() == DType::Float32) ? x_ptr->contiguous() : dispatchCast(x_ptr->contiguous(), DType::Float32);
    }

    int32_t device_id = y.device().index;
    auto* pipeline = getPipeline("trapezoid", device_id);

    TrapezoidPC pc{static_cast<uint32_t>(outer), static_cast<uint32_t>(inner),
                   static_cast<uint32_t>(n), static_cast<uint32_t>(total),
                   static_cast<float>(dx), x_ptr ? 1u : 0u};

    std::vector<std::pair<uint32_t, const void*>> bindings;
    std::vector<size_t> sizes;
    bindings.push_back({0, yf.data_ptr()});
    sizes.push_back(static_cast<size_t>(yf.numel()) * sizeof(float));
    if (x_ptr) {
        bindings.push_back({1, xf.data_ptr()});
        sizes.push_back(static_cast<size_t>(xf.numel()) * sizeof(float));
    } else {
        // Bind y again as dummy for binding 1 (shader reads has_x flag)
        bindings.push_back({1, yf.data_ptr()});
        sizes.push_back(static_cast<size_t>(yf.numel()) * sizeof(float));
    }
    bindings.push_back({2, result_f32.data_ptr()});
    sizes.push_back(static_cast<size_t>(result_f32.numel()) * sizeof(float));

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(TrapezoidPC), &pc);
    uint32_t workgroups = div_wg(static_cast<uint32_t>(total), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return (orig_dtype == DType::Float32) ? result_f32 : dispatchCast(result_f32, orig_dtype);
}

// ============================================================================
// Cumulative trapezoid integration
// ============================================================================

struct CumulativeTrapezoidPC {
    uint32_t outer;
    uint32_t inner;
    uint32_t n;
    uint32_t total;
    float dx;
    uint32_t has_x;
};

auto VulkanBackend::dispatchCumulativeTrapezoid(const Tensor& y, int64_t dim, double dx,
                                                 const Tensor* x_ptr) -> Tensor {
    DType orig_dtype = y.dtype();
    Tensor yf = (orig_dtype == DType::Float32) ? y.contiguous() : dispatchCast(y.contiguous(), DType::Float32);

    auto shape = yf.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t n = shape[dim];

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = n - 1;

    if (n < 2) {
        out_shape[dim] = 0;
        return Tensor(out_shape, orig_dtype, y.device());
    }

    Tensor result_f32(out_shape, DType::Float32, y.device());
    int64_t total = outer * inner;
    if (total == 0) return (orig_dtype == DType::Float32) ? result_f32 : dispatchCast(result_f32, orig_dtype);

    Tensor xf;
    if (x_ptr) {
        xf = (x_ptr->dtype() == DType::Float32) ? x_ptr->contiguous() : dispatchCast(x_ptr->contiguous(), DType::Float32);
    }

    int32_t device_id = y.device().index;
    auto* pipeline = getPipeline("cumulative_trapezoid", device_id);

    CumulativeTrapezoidPC pc{static_cast<uint32_t>(outer), static_cast<uint32_t>(inner),
                              static_cast<uint32_t>(n), static_cast<uint32_t>(total),
                              static_cast<float>(dx), x_ptr ? 1u : 0u};

    std::vector<std::pair<uint32_t, const void*>> bindings;
    std::vector<size_t> sizes;
    bindings.push_back({0, yf.data_ptr()});
    sizes.push_back(static_cast<size_t>(yf.numel()) * sizeof(float));
    if (x_ptr) {
        bindings.push_back({1, xf.data_ptr()});
        sizes.push_back(static_cast<size_t>(xf.numel()) * sizeof(float));
    } else {
        bindings.push_back({1, yf.data_ptr()});
        sizes.push_back(static_cast<size_t>(yf.numel()) * sizeof(float));
    }
    bindings.push_back({2, result_f32.data_ptr()});
    sizes.push_back(static_cast<size_t>(result_f32.numel()) * sizeof(float));

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(CumulativeTrapezoidPC), &pc);
    uint32_t workgroups = div_wg(static_cast<uint32_t>(total), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return (orig_dtype == DType::Float32) ? result_f32 : dispatchCast(result_f32, orig_dtype);
}

// ============================================================================
// Numerical gradient
// ============================================================================

struct GradientPC {
    uint32_t outer;
    uint32_t inner;
    uint32_t n;
    uint32_t total;
    float spacing;
};

auto VulkanBackend::dispatchGradient(const Tensor& input, int64_t dim, double spacing) -> Tensor {
    DType orig_dtype = input.dtype();
    Tensor inf = (orig_dtype == DType::Float32) ? input.contiguous() : dispatchCast(input.contiguous(), DType::Float32);

    auto shape = inf.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t n = shape[dim];

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    Tensor result_f32(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32, input.device());
    int64_t total = outer * inner;
    if (n < 2 || total == 0) return (orig_dtype == DType::Float32) ? result_f32 : dispatchCast(result_f32, orig_dtype);

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("gradient", device_id);

    GradientPC pc{static_cast<uint32_t>(outer), static_cast<uint32_t>(inner),
                  static_cast<uint32_t>(n), static_cast<uint32_t>(total),
                  static_cast<float>(spacing)};

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, inf.data_ptr()},
        {1, result_f32.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(inf.numel()) * sizeof(float),
        static_cast<size_t>(result_f32.numel()) * sizeof(float),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(GradientPC), &pc);
    uint32_t workgroups = div_wg(static_cast<uint32_t>(total), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return (orig_dtype == DType::Float32) ? result_f32 : dispatchCast(result_f32, orig_dtype);
}

// ============================================================================
// Pairwise distance
// ============================================================================

struct PairwiseDistPC {
    uint32_t N;
    uint32_t D;
    float p;
};

auto VulkanBackend::dispatchPairwiseDistance(const Tensor& x1, const Tensor& x2, double p) -> Tensor {
    DType orig_dtype = x1.dtype();
    Tensor a = (orig_dtype == DType::Float32) ? x1.contiguous() : dispatchCast(x1.contiguous(), DType::Float32);
    Tensor b = (x2.dtype() == DType::Float32) ? x2.contiguous() : dispatchCast(x2.contiguous(), DType::Float32);

    int64_t N = a.shape()[0];
    int64_t D = a.shape()[1];

    Tensor result_f32({N}, DType::Float32, x1.device());
    if (N == 0) return (orig_dtype == DType::Float32) ? result_f32 : dispatchCast(result_f32, orig_dtype);

    int32_t device_id = x1.device().index;
    auto* pipeline = getPipeline("pairwise_distance", device_id);

    PairwiseDistPC pc{static_cast<uint32_t>(N), static_cast<uint32_t>(D), static_cast<float>(p)};

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, a.data_ptr()},
        {1, b.data_ptr()},
        {2, result_f32.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(a.numel()) * sizeof(float),
        static_cast<size_t>(b.numel()) * sizeof(float),
        static_cast<size_t>(N) * sizeof(float),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PairwiseDistPC), &pc);
    uint32_t workgroups = div_wg(static_cast<uint32_t>(N), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return (orig_dtype == DType::Float32) ? result_f32 : dispatchCast(result_f32, orig_dtype);
}

// ============================================================================
// Pdist (all-pairs pairwise distances)
// ============================================================================

struct PdistPC {
    uint32_t N;
    uint32_t D;
    uint32_t num_pairs;
    float p;
};

auto VulkanBackend::dispatchPdist(const Tensor& input, double p) -> Tensor {
    DType orig_dtype = input.dtype();
    Tensor inf = (orig_dtype == DType::Float32) ? input.contiguous() : dispatchCast(input.contiguous(), DType::Float32);

    int64_t N = inf.shape()[0];
    int64_t D = inf.shape()[1];
    int64_t num_pairs = N * (N - 1) / 2;

    Tensor result_f32({num_pairs}, DType::Float32, input.device());
    if (num_pairs == 0) return (orig_dtype == DType::Float32) ? result_f32 : dispatchCast(result_f32, orig_dtype);

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("pdist", device_id);

    PdistPC pc{static_cast<uint32_t>(N), static_cast<uint32_t>(D),
               static_cast<uint32_t>(num_pairs), static_cast<float>(p)};

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, inf.data_ptr()},
        {1, result_f32.data_ptr()},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(inf.numel()) * sizeof(float),
        static_cast<size_t>(num_pairs) * sizeof(float),
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PdistPC), &pc);
    uint32_t workgroups = div_wg(static_cast<uint32_t>(num_pairs), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return (orig_dtype == DType::Float32) ? result_f32 : dispatchCast(result_f32, orig_dtype);
}

}  // namespace tenzor
