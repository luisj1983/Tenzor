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

}  // namespace tenzor
