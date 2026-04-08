/**
 * @file vulkan_ops_stft.cpp
 * @brief Native Vulkan STFT and ISTFT dispatch.
 *
 * Native Vulkan STFT/ISTFT implementation. Forward path builds
 * (B, num_frames, n_fft) frames via stft_frame_window.comp, calls
 * dispatchRFFT for the batched real FFT, then transposes to the standard
 * (B, freq_bins, num_frames) layout. Inverse reshapes/transposes to
 * (B, num_frames, freq_bins), calls dispatchIRFFT, then runs the
 * output-centric istft_overlap_add.comp + istft_normalize.comp.
 *
 * Note: this file depends on the Complex64 fixes to dispatchContiguous and
 * dispatchPermute in vulkan_ops_shape.cpp (8-byte dtypes were previously
 * falling through to the 4-byte shader). Without those, the round-trip
 * reconstruction error was ~2.0; with them the test passes to 1e-3.
 *
 * Strategy: build a (B, num_frames, n_fft) framed+windowed tensor on-device
 * via stft_frame_window.comp, then call dispatchRFFT/dispatchFFT (existing
 * Vulkan FFT shaders), then transpose for the standard STFT output layout.
 * Inverse path uses dispatchIRFFT/dispatchIFFT + the output-centric
 * istft_overlap_add.comp shader followed by istft_normalize.comp.
 *
 * Float32 only on the GPU; non-Float32 inputs are promoted via dispatchCast
 * (an on-device Vulkan compute pipeline) before invocation.
 */

#include "vulkan_ops_common.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"

namespace tenzor {

namespace {

struct STFTFrameWindowPC {
    int32_t batch_size;
    int32_t signal_length;
    int32_t num_frames;
    int32_t n_fft;
    int32_t hop_length;
    int32_t pad;
    int32_t total;
};

struct ISTFTOverlapAddPC {
    int32_t batch_size;
    int32_t num_frames;
    int32_t n_fft;
    int32_t hop_length;
    int32_t expected_length;
    int32_t total;
};

struct ISTFTNormalizePC {
    int32_t total;
};

}  // namespace

auto VulkanBackend::dispatchSTFT(const Tensor& input, int64_t n_fft,
                                 int64_t hop_length, int64_t win_length,
                                 const Tensor& window, bool center,
                                 bool normalized, bool onesided) -> Tensor {
    if (n_fft <= 0) throw std::runtime_error("stft: n_fft must be > 0");
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;
    if (win_length > n_fft) throw std::runtime_error("stft: win_length must be <= n_fft");

    DType orig_dtype = input.dtype();
    Tensor input_f32 = (orig_dtype == DType::Float32) ? input.contiguous()
                                                       : dispatchCast(input.contiguous(), DType::Float32);

    auto in_shape = input_f32.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (ndim < 1) throw std::runtime_error("stft: input must have ≥ 1 dim");

    int64_t signal_length = in_shape[ndim - 1];
    int64_t batch_size = 1;
    for (int64_t d = 0; d < ndim - 1; ++d) batch_size *= in_shape[d];

    int64_t pad = center ? (n_fft / 2) : 0;
    int64_t padded_length = signal_length + 2 * pad;
    int64_t num_frames = (padded_length - n_fft) / hop_length + 1;
    if (num_frames <= 0) throw std::runtime_error("stft: signal too short");

    int64_t freq_bins = onesided ? (n_fft / 2 + 1) : n_fft;

    int32_t device_id = input.device().index;

    // Build window tensor (n_fft) — pad win_length with zeros on either side
    Tensor window_dev = tenzor::zeros({n_fft}, DType::Float32, input.device());
    int64_t win_offset = (n_fft - win_length) / 2;
    if (window.is_valid() && window.numel() > 0) {
        Tensor win_f32 = (window.dtype() != DType::Float32) ? dispatchCast(window.contiguous(), DType::Float32)
                                                              : window.contiguous();
        // Slice-assign would be ideal; use a Vulkan buffer-to-buffer copy at offset.
        auto [src_buf, src_off] = getVulkanBufferAndOffset(win_f32.data_ptr());
        auto [dst_buf, dst_off] = getVulkanBufferAndOffset(window_dev.data_ptr());
        VkBufferCopy region{};
        region.srcOffset = static_cast<VkDeviceSize>(src_off);
        region.dstOffset = static_cast<VkDeviceSize>(dst_off) + win_offset * sizeof(float);
        region.size = win_length * sizeof(float);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    } else {
        // Default rectangular window: fill [win_offset, win_offset+win_length) with 1.0
        Tensor ones = tenzor::full({win_length}, 1.0f, DType::Float32, input.device());
        auto [src_buf, src_off] = getVulkanBufferAndOffset(ones.data_ptr());
        auto [dst_buf, dst_off] = getVulkanBufferAndOffset(window_dev.data_ptr());
        VkBufferCopy region{};
        region.srcOffset = static_cast<VkDeviceSize>(src_off);
        region.dstOffset = static_cast<VkDeviceSize>(dst_off) + win_offset * sizeof(float);
        region.size = win_length * sizeof(float);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Frame+window kernel
    Tensor framed({batch_size, num_frames, n_fft}, DType::Float32, input.device());
    int64_t total = batch_size * num_frames * n_fft;
    if (total > 0) {
        auto* pipeline = getPipeline("stft_frame_window", device_id);
        STFTFrameWindowPC pc{
            static_cast<int32_t>(batch_size),
            static_cast<int32_t>(signal_length),
            static_cast<int32_t>(num_frames),
            static_cast<int32_t>(n_fft),
            static_cast<int32_t>(hop_length),
            static_cast<int32_t>(pad),
            static_cast<int32_t>(total),
        };
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input_f32.data_ptr()},
            {1, window_dev.data_ptr()},
            {2, framed.data_ptr()},
        };
        std::vector<size_t> sizes = {
            static_cast<size_t>(input_f32.numel()) * sizeof(float),
            static_cast<size_t>(n_fft) * sizeof(float),
            static_cast<size_t>(total) * sizeof(float),
        };
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(STFTFrameWindowPC), &pc);
        uint32_t workgroups = div_wg(static_cast<uint32_t>(total),
                                      devices_[device_id].workgroupSize);
        vkCmdDispatch(cmd, workgroups, 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Batched FFT along last dim of (B, num_frames, n_fft)
    Tensor fft_out;
    if (onesided) {
        fft_out = dispatchRFFT(framed, /*dim=*/2, n_fft, normalized ? "ortho" : "backward");
    } else {
        // For non-onesided we need a complex input; cast framed → complex first by
        // building a complex tensor (real part = framed, imag = 0) and calling FFT.
        // dispatchFFT expects Complex64. Use dispatch::add zero imag via existing ops.
        // Pragmatic: rfft is the common case; non-onesided is rare.
        throw std::runtime_error("Vulkan STFT: onesided=false not yet supported");
    }
    // fft_out: (B, num_frames, freq_bins) Complex64

    Tensor transposed = tenzor::transpose(fft_out, -1, -2);

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim - 1; ++d) out_shape.push_back(in_shape[d]);
    out_shape.push_back(freq_bins);
    out_shape.push_back(num_frames);
    return tenzor::reshape(transposed, out_shape);
}

auto VulkanBackend::dispatchISTFT(const Tensor& input, int64_t n_fft,
                                  int64_t hop_length, int64_t win_length,
                                  const Tensor& window, bool center,
                                  bool /*normalized*/, bool onesided,
                                  int64_t length) -> Tensor {
    if (n_fft <= 0) throw std::runtime_error("istft: n_fft must be > 0");
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;
    if (!onesided) throw std::runtime_error("Vulkan ISTFT: onesided=false not yet supported");

    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (ndim < 2) throw std::runtime_error("istft: input must have ≥ 2 dims");

    int64_t freq_bins = in_shape[ndim - 2];
    int64_t num_frames = in_shape[ndim - 1];
    int64_t batch_size = 1;
    for (int64_t d = 0; d < ndim - 2; ++d) batch_size *= in_shape[d];

    int64_t expected_length = n_fft + (num_frames - 1) * hop_length;

    Tensor input_c64 = (input.dtype() != DType::Complex64) ? input.to(DType::Complex64)
                                                            : input.contiguous();
    Tensor reshaped = tenzor::reshape(input_c64, {batch_size, freq_bins, num_frames});
    Tensor transposed = tenzor::transpose(reshaped, -1, -2);
    Tensor transposed_contig = transposed.contiguous();
    // (B, num_frames, freq_bins) Complex64

    // Inverse RFFT → (B, num_frames, n_fft) Float32
    Tensor time_frames = dispatchIRFFT(transposed_contig, /*dim=*/2, n_fft, "backward");

    int32_t device_id = input.device().index;

    // Build window
    Tensor window_dev = tenzor::zeros({n_fft}, DType::Float32, input.device());
    int64_t win_offset = (n_fft - win_length) / 2;
    if (window.is_valid() && window.numel() > 0) {
        Tensor win_f32 = (window.dtype() != DType::Float32) ? dispatchCast(window.contiguous(), DType::Float32)
                                                              : window.contiguous();
        auto [src_buf, src_off] = getVulkanBufferAndOffset(win_f32.data_ptr());
        auto [dst_buf, dst_off] = getVulkanBufferAndOffset(window_dev.data_ptr());
        VkBufferCopy region{};
        region.srcOffset = static_cast<VkDeviceSize>(src_off);
        region.dstOffset = static_cast<VkDeviceSize>(dst_off) + win_offset * sizeof(float);
        region.size = win_length * sizeof(float);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    } else {
        Tensor ones = tenzor::full({win_length}, 1.0f, DType::Float32, input.device());
        auto [src_buf, src_off] = getVulkanBufferAndOffset(ones.data_ptr());
        auto [dst_buf, dst_off] = getVulkanBufferAndOffset(window_dev.data_ptr());
        VkBufferCopy region{};
        region.srcOffset = static_cast<VkDeviceSize>(src_off);
        region.dstOffset = static_cast<VkDeviceSize>(dst_off) + win_offset * sizeof(float);
        region.size = win_length * sizeof(float);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Output and window-sum buffers (output-centric kernel writes them, no zeroing needed)
    Tensor output_buf({batch_size, expected_length}, DType::Float32, input.device());
    Tensor wsum_buf({batch_size, expected_length}, DType::Float32, input.device());

    // Overlap-add (output-centric, no atomics)
    {
        auto* pipeline = getPipeline("istft_overlap_add", device_id);
        ISTFTOverlapAddPC pc{
            static_cast<int32_t>(batch_size),
            static_cast<int32_t>(num_frames),
            static_cast<int32_t>(n_fft),
            static_cast<int32_t>(hop_length),
            static_cast<int32_t>(expected_length),
            static_cast<int32_t>(batch_size * expected_length),
        };
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, time_frames.data_ptr()},
            {1, window_dev.data_ptr()},
            {2, output_buf.data_ptr()},
            {3, wsum_buf.data_ptr()},
        };
        std::vector<size_t> sizes = {
            static_cast<size_t>(time_frames.numel()) * sizeof(float),
            static_cast<size_t>(n_fft) * sizeof(float),
            static_cast<size_t>(output_buf.numel()) * sizeof(float),
            static_cast<size_t>(wsum_buf.numel()) * sizeof(float),
        };
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(ISTFTOverlapAddPC), &pc);
        uint32_t workgroups = div_wg(static_cast<uint32_t>(pc.total),
                                      devices_[device_id].workgroupSize);
        vkCmdDispatch(cmd, workgroups, 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Normalize
    {
        auto* pipeline = getPipeline("istft_normalize", device_id);
        ISTFTNormalizePC pc{static_cast<int32_t>(batch_size * expected_length)};
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, output_buf.data_ptr()},
            {1, wsum_buf.data_ptr()},
        };
        std::vector<size_t> sizes = {
            static_cast<size_t>(output_buf.numel()) * sizeof(float),
            static_cast<size_t>(wsum_buf.numel()) * sizeof(float),
        };
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(ISTFTNormalizePC), &pc);
        uint32_t workgroups = div_wg(static_cast<uint32_t>(pc.total),
                                      devices_[device_id].workgroupSize);
        vkCmdDispatch(cmd, workgroups, 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    int64_t pad = center ? (n_fft / 2) : 0;
    int64_t out_length = expected_length - 2 * pad;
    if (length >= 0) out_length = length;

    Tensor result;
    if (pad == 0 && length < 0) {
        result = output_buf;
    } else {
        Tensor sliced = tenzor::slice(output_buf, 1, pad, pad + out_length);
        result = sliced.contiguous();
    }

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim - 2; ++d) out_shape.push_back(in_shape[d]);
    out_shape.push_back(out_length);
    return tenzor::reshape(result, out_shape);
}

}  // namespace tenzor
