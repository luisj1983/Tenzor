#include "tenzor/nn/utils/rnn_utils.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstring>

namespace tenzor::nn {

auto pack_padded_sequence(const Tensor& input, const Tensor& lengths,
                          bool batch_first, bool enforce_sorted)
    -> PackedSequence {
    if (lengths.ndim() != 1) {
        throw std::invalid_argument("lengths must be a 1D tensor");
    }

    // Get input dimensions
    Tensor seq_input = batch_first ? input.permute({1, 0, 2}) : input;
    // seq_input shape: (seq_len, batch, features)
    auto shape = seq_input.shape();
    int64_t seq_len = shape[0];
    int64_t batch_size = shape[1];
    int64_t features = shape[2];

    if (lengths.numel() != batch_size) {
        throw std::invalid_argument("lengths size must match batch dimension");
    }

    // Read lengths to CPU
    Tensor lengths_cpu = lengths.device().type != Device::Type::CPU
        ? lengths.to(Device::cpu()) : lengths;
    Tensor lengths_int64 = lengths_cpu.dtype() != DType::Int64
        ? lengths_cpu.to(DType::Int64) : lengths_cpu;

    std::vector<int64_t> len_vec(batch_size);
    auto* len_data = lengths_int64.data<int64_t>();
    for (int64_t i = 0; i < batch_size; ++i) {
        len_vec[i] = len_data[i];
        // The pack loop indexes seq_input along the time axis up to len_vec[i];
        // a length exceeding the real time extent (shape[0]) — or a negative
        // length — would read past the input buffer. Validate against seq_len.
        if (len_vec[i] < 0 || len_vec[i] > seq_len) {
            throw std::invalid_argument(
                "pack_padded_sequence: length " + std::to_string(len_vec[i]) +
                " at batch index " + std::to_string(i) +
                " is out of range [0, " + std::to_string(seq_len) + "]");
        }
    }

    // Create sort indices (descending by length). Use stable_sort so that
    // sequences with tied lengths keep their original relative order. This
    // matters for the enforce_sorted identity check below: a validly
    // non-increasing input with tied lengths (e.g. [5,3,3,1]) must yield the
    // identity permutation rather than being spuriously reordered by an
    // unstable sort, which would otherwise reject a legitimate batch.
    std::vector<int64_t> sorted_idx(batch_size);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                     [&](int64_t a, int64_t b) { return len_vec[a] > len_vec[b]; });

    if (enforce_sorted) {
        // Verify already sorted
        for (int64_t i = 0; i < batch_size; ++i) {
            if (sorted_idx[i] != i) {
                throw std::runtime_error(
                    "pack_padded_sequence: sequences must be sorted by length "
                    "in decreasing order when enforce_sorted=true");
            }
        }
    }

    // Sorted lengths
    std::vector<int64_t> sorted_lengths(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
        sorted_lengths[i] = len_vec[sorted_idx[i]];
    }

    // Compute batch_sizes: for each timestep, how many sequences are active
    int64_t actual_max_len = sorted_lengths[0];
    std::vector<int64_t> batch_sizes_vec(actual_max_len);
    for (int64_t t = 0; t < actual_max_len; ++t) {
        int64_t count = 0;
        for (int64_t b = 0; b < batch_size; ++b) {
            if (sorted_lengths[b] > t) ++count;
        }
        batch_sizes_vec[t] = count;
    }

    // Compute total elements
    int64_t total = 0;
    for (auto bs : batch_sizes_vec) total += bs;

    // Pack data: work on CPU to avoid device-pointer memcpy issues
    Device orig_device = input.device();
    Tensor seq_input_cpu = seq_input.to(Device::cpu()).contiguous();
    Tensor packed_data_cpu = zeros({total, features}, input.dtype(), Device::cpu());

    int64_t offset = 0;
    for (int64_t t = 0; t < actual_max_len; ++t) {
        int64_t bs = batch_sizes_vec[t];
        for (int64_t b = 0; b < bs; ++b) {
            // Copy seq_input_cpu[t, sorted_idx[b], :] to packed_data_cpu[offset + b, :]
            if (features > 0) {
                size_t elem_size = packed_data_cpu.dtype_size();
                int64_t src_offset = (t * batch_size + sorted_idx[b]) * features;
                int64_t dst_offset = (offset + b) * features;
                std::memcpy(
                    static_cast<char*>(packed_data_cpu.data_ptr()) + dst_offset * elem_size,
                    static_cast<const char*>(seq_input_cpu.data_ptr()) + src_offset * elem_size,
                    features * elem_size
                );
            }
        }
        offset += bs;
    }

    // Move packed data back to original device
    Tensor packed_data = (orig_device.type != Device::Type::CPU)
        ? packed_data_cpu.to(orig_device) : packed_data_cpu;

    // Create result tensors
    Tensor batch_sizes_tensor = zeros({actual_max_len}, DType::Int64, Device::cpu());
    auto* bs_data = batch_sizes_tensor.data<int64_t>();
    for (int64_t t = 0; t < actual_max_len; ++t) {
        bs_data[t] = batch_sizes_vec[t];
    }

    Tensor sorted_indices = zeros({batch_size}, DType::Int64, Device::cpu());
    Tensor unsorted_indices = zeros({batch_size}, DType::Int64, Device::cpu());
    auto* si_data = sorted_indices.data<int64_t>();
    auto* ui_data = unsorted_indices.data<int64_t>();
    for (int64_t i = 0; i < batch_size; ++i) {
        si_data[i] = sorted_idx[i];
        ui_data[sorted_idx[i]] = i;
    }

    return PackedSequence{
        std::move(packed_data),
        std::move(batch_sizes_tensor),
        std::move(sorted_indices),
        std::move(unsorted_indices)
    };
}

auto pad_packed_sequence(const PackedSequence& packed,
                         bool batch_first,
                         float padding_value,
                         int64_t total_length)
    -> std::pair<Tensor, Tensor> {

    auto bs_tensor = packed.batch_sizes;
    if (bs_tensor.numel() == 0) {
        throw std::invalid_argument(
            "pad_packed_sequence: batch_sizes is empty");
    }
    auto* bs_data = bs_tensor.data<int64_t>();
    int64_t max_seq_len = bs_tensor.numel();
    int64_t batch_size = bs_data[0];  // First timestep has all sequences
    int64_t features = packed.data.shape()[1];

    if (total_length >= 0) {
        if (total_length < max_seq_len) {
            throw std::invalid_argument("total_length must be >= max sequence length");
        }
        max_seq_len = total_length;
    }

    // Compute lengths from batch_sizes
    std::vector<int64_t> sorted_lengths(batch_size, 0);
    int64_t actual_len = bs_tensor.numel();
    for (int64_t t = 0; t < actual_len; ++t) {
        for (int64_t b = 0; b < bs_data[t]; ++b) {
            sorted_lengths[b] = t + 1;
        }
    }

    // Work on CPU to avoid device-pointer memcpy issues, then move back
    Device orig_device = packed.data.device();
    Tensor packed_cpu = packed.data.to(Device::cpu()).contiguous();
    size_t elem_size = packed_cpu.dtype_size();

    Tensor output_cpu;
    if (batch_first) {
        output_cpu = full({batch_size, max_seq_len, features}, padding_value,
                          packed.data.dtype(), Device::cpu());
    } else {
        output_cpu = full({max_seq_len, batch_size, features}, padding_value,
                          packed.data.dtype(), Device::cpu());
    }

    int64_t offset = 0;
    for (int64_t t = 0; t < actual_len; ++t) {
        int64_t bs = bs_data[t];
        for (int64_t b = 0; b < bs; ++b) {
            int64_t src_off = (offset + b) * features;
            if (batch_first) {
                int64_t dst_off = (b * max_seq_len + t) * features;
                std::memcpy(
                    static_cast<char*>(output_cpu.data_ptr()) + dst_off * elem_size,
                    static_cast<const char*>(packed_cpu.data_ptr()) + src_off * elem_size,
                    features * elem_size);
            } else {
                int64_t dst_off = (t * batch_size + b) * features;
                std::memcpy(
                    static_cast<char*>(output_cpu.data_ptr()) + dst_off * elem_size,
                    static_cast<const char*>(packed_cpu.data_ptr()) + src_off * elem_size,
                    features * elem_size);
            }
        }
        offset += bs;
    }

    // Unsort lengths back to original order
    Tensor lengths_tensor = zeros({batch_size}, DType::Int64, Device::cpu());
    auto* len_out = lengths_tensor.data<int64_t>();

    // Unsort output on CPU
    auto* si_data = packed.sorted_indices.data<int64_t>();
    if (batch_first) {
        Tensor unsorted_cpu = zeros_like(output_cpu);
        for (int64_t sorted_pos = 0; sorted_pos < batch_size; ++sorted_pos) {
            int64_t orig_pos = si_data[sorted_pos];
            len_out[orig_pos] = sorted_lengths[sorted_pos];
            std::memcpy(
                static_cast<char*>(unsorted_cpu.data_ptr()) + orig_pos * max_seq_len * features * elem_size,
                static_cast<char*>(output_cpu.data_ptr()) + sorted_pos * max_seq_len * features * elem_size,
                max_seq_len * features * elem_size);
        }
        output_cpu = unsorted_cpu;
    } else {
        Tensor unsorted_cpu = zeros_like(output_cpu);
        for (int64_t t = 0; t < max_seq_len; ++t) {
            for (int64_t sorted_pos = 0; sorted_pos < batch_size; ++sorted_pos) {
                int64_t orig_pos = si_data[sorted_pos];
                if (t == 0) len_out[orig_pos] = sorted_lengths[sorted_pos];
                std::memcpy(
                    static_cast<char*>(unsorted_cpu.data_ptr()) + (t * batch_size + orig_pos) * features * elem_size,
                    static_cast<char*>(output_cpu.data_ptr()) + (t * batch_size + sorted_pos) * features * elem_size,
                    features * elem_size);
            }
        }
        output_cpu = unsorted_cpu;
    }

    // Move output back to original device
    Tensor output_final = (orig_device.type != Device::Type::CPU)
        ? output_cpu.to(orig_device) : output_cpu;

    return {std::move(output_final), std::move(lengths_tensor)};
}

auto pack_sequence(const std::vector<Tensor>& sequences, bool enforce_sorted)
    -> PackedSequence {
    if (sequences.empty()) {
        throw std::invalid_argument("pack_sequence: empty sequence list");
    }

    // Get features from first sequence
    int64_t features = sequences[0].shape().back();
    int64_t batch_size = static_cast<int64_t>(sequences.size());

    // Find max length, validating each sequence's trailing feature width,
    // dtype and device against sequences[0]. memcpy below uses sequences[0]'s
    // feature width as the source row stride for every sequence; a mismatch
    // would over-read (smaller seq) or silently truncate (larger seq).
    int64_t max_len = 0;
    std::vector<int64_t> lengths(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
        const auto& s = sequences[i];
        if (s.ndim() != 2) {
            throw std::invalid_argument(
                "pack_sequence: each sequence must be 2D [length, features], got "
                + std::to_string(s.ndim()) + " dims at index " + std::to_string(i));
        }
        if (s.shape().back() != features) {
            throw std::invalid_argument(
                "pack_sequence: feature size mismatch at index " + std::to_string(i)
                + " (expected " + std::to_string(features) + ", got "
                + std::to_string(s.shape().back()) + ")");
        }
        if (s.dtype() != sequences[0].dtype() || s.device() != sequences[0].device()) {
            throw std::invalid_argument(
                "pack_sequence: dtype/device mismatch at index " + std::to_string(i));
        }
        lengths[i] = s.shape()[0];
        max_len = std::max(max_len, lengths[i]);
    }

    // Create padded tensor (batch_first format)
    Tensor padded = zeros({batch_size, max_len, features},
                          sequences[0].dtype(), sequences[0].device());

    // Copy each sequence
    for (int64_t i = 0; i < batch_size; ++i) {
        auto seq = sequences[i].contiguous();
        std::memcpy(
            static_cast<char*>(padded.data_ptr()) + i * max_len * features * padded.dtype_size(),
            seq.data_ptr(),
            lengths[i] * features * padded.dtype_size()
        );
    }

    // Create lengths tensor
    Tensor lengths_tensor = zeros({batch_size}, DType::Int64, Device::cpu());
    auto* len_data = lengths_tensor.data<int64_t>();
    for (int64_t i = 0; i < batch_size; ++i) {
        len_data[i] = lengths[i];
    }

    return pack_padded_sequence(padded, lengths_tensor, /*batch_first=*/true, enforce_sorted);
}

auto pad_sequence(const std::vector<Tensor>& sequences,
                  bool batch_first,
                  float padding_value)
    -> Tensor {
    if (sequences.empty()) {
        throw std::invalid_argument("pad_sequence: got empty list of sequences");
    }

    int64_t max_len = 0;
    int64_t batch_size = static_cast<int64_t>(sequences.size());

    // Trailing dimensions are taken from sequences[0] and assumed identical for
    // every sequence (the memcpy below uses trail_elems derived from
    // sequences[0] for all sources). Validate this so a mismatched trailing
    // shape cannot over-read a smaller source buffer or silently truncate.
    auto trailing = std::vector<int64_t>(
        sequences[0].shape().begin() + 1, sequences[0].shape().end());
    int64_t trail_elems = 1;
    for (auto d : trailing) trail_elems *= d;

    for (size_t i = 0; i < sequences.size(); ++i) {
        const auto& seq = sequences[i];
        if (seq.ndim() == 0) {
            throw std::invalid_argument("pad_sequence: sequences must have at least 1 dimension");
        }
        std::vector<int64_t> seq_trailing(
            seq.shape().begin() + 1, seq.shape().end());
        if (seq_trailing != trailing) {
            throw std::invalid_argument(
                "pad_sequence: trailing shape mismatch at index " + std::to_string(i)
                + " (all sequences must share sequences[0]'s trailing dimensions)");
        }
        if (seq.dtype() != sequences[0].dtype() || seq.device() != sequences[0].device()) {
            throw std::invalid_argument(
                "pad_sequence: dtype/device mismatch at index " + std::to_string(i));
        }
        max_len = std::max(max_len, seq.shape()[0]);
    }

    // Always build batch_first: (batch, max_len, *trailing)
    std::vector<int64_t> padded_shape = {batch_size, max_len};
    padded_shape.insert(padded_shape.end(), trailing.begin(), trailing.end());

    Tensor out = full(padded_shape, padding_value,
                      sequences[0].dtype(), sequences[0].device());

    // Copy each sequence using memcpy (same pattern as pack_sequence)
    int64_t row_stride = max_len * trail_elems;
    size_t elem_size = out.dtype_size();
    for (int64_t i = 0; i < batch_size; ++i) {
        auto seq = sequences[i].contiguous();
        int64_t length = seq.shape()[0];
        std::memcpy(
            static_cast<char*>(out.data_ptr()) + i * row_stride * elem_size,
            seq.data_ptr(),
            length * trail_elems * elem_size
        );
    }

    // Permute to (max_len, batch, *trailing) if not batch_first
    if (!batch_first) {
        std::vector<int64_t> perm = {1, 0};
        for (size_t d = 2; d < padded_shape.size(); ++d) {
            perm.push_back(static_cast<int64_t>(d));
        }
        out = out.permute(perm).contiguous();
    }

    return out;
}

} // namespace tenzor::nn
