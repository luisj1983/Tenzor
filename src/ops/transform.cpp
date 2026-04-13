#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/utils/profiling.hpp"
#include <sstream>
#include <cstring>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor {

// Helper function to convert shape vector to comma-separated string
static auto shape_to_string(const std::vector<int64_t>& shape) -> std::string {
    std::ostringstream oss;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ",";
        oss << shape[i];
    }
    return oss.str();
}

// Shape transformations (delegate to Tensor methods)

auto reshape(const Tensor& input, std::vector<int64_t> shape) -> Tensor {
    return input.reshape(std::move(shape));
}

auto view(const Tensor& input, std::vector<int64_t> shape) -> Tensor {
    return input.view(std::move(shape));
}

auto transpose(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor {
    return input.transpose(dim0, dim1);
}

auto permute(const Tensor& input, std::vector<int64_t> dims) -> Tensor {
    return input.permute(std::move(dims));
}

auto squeeze(const Tensor& input, std::optional<int64_t> dim) -> Tensor {
    return input.squeeze(dim);
}

auto unsqueeze(const Tensor& input, int64_t dim) -> Tensor {
    return input.unsqueeze(dim);
}

auto flatten(const Tensor& input, int64_t start_dim, int64_t end_dim) -> Tensor {
    return input.flatten(start_dim, end_dim);
}

auto contiguous(const Tensor& input) -> Tensor {
    return input.contiguous();
}

// Additional transform operations
auto cat(std::span<const Tensor> tensors, int64_t dim) -> Tensor {
    if (tensors.empty()) {
        throw std::runtime_error("Cannot concatenate empty tensor list");
    }

    if (tensors.size() == 1) {
        return tensors[0];
    }

    // Validate all tensors have compatible shapes
    auto first_shape = tensors[0].shape();
    int64_t ndim = first_shape.size();

    // Normalize dim
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Dimension out of range for concatenation");
    }

    // Check all tensors have same ndim and same shape except at dim
    int64_t total_size_at_dim = 0;
    for (const auto& t : tensors) {
        auto shape = t.shape();
        if (shape.size() != static_cast<size_t>(ndim)) {
            throw std::runtime_error("All tensors must have the same number of dimensions");
        }
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != dim && shape[i] != first_shape[i]) {
                throw std::runtime_error("All tensors must have the same shape except in the concatenation dimension");
            }
        }
        total_size_at_dim += shape[dim];
    }

    // Create output shape
    std::vector<int64_t> out_shape(first_shape.begin(), first_shape.end());
    out_shape[dim] = total_size_at_dim;

    // For non-CPU devices, use dispatcher to route to backend-specific implementation
    if (tensors[0].device().type != Device::Type::CPU) {
        NewOpAttributes attrs;
        attrs.set(AttrKey::Dim, dim);

        // Convert span to vector for dispatch
        std::vector<Tensor> tensor_vec(tensors.begin(), tensors.end());
        return dispatch(OpId::Cat, std::span<const Tensor>(tensor_vec), attrs)[0];
    }

    // Optimized CPU concatenation using memcpy + OpenMP
    auto output = zeros(out_shape, tensors[0].dtype(), Device::cpu());
    auto dtype = tensors[0].dtype();
    size_t elem_size = dtype_size(dtype);

    // Calculate dimensions for chunk-based copying:
    // outer_size = product of dims before `dim`
    // inner_size = product of dims after `dim` (contiguous chunk size)
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= out_shape[i];
    }

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= out_shape[i];
    }

    // Overflow check: out_shape[dim] * inner_size
    if (inner_size > 0 && out_shape[dim] > std::numeric_limits<int64_t>::max() / inner_size) {
        throw std::overflow_error("cat: output tensor size overflows int64_t");
    }
    int64_t out_slice_size = out_shape[dim] * inner_size;  // Full slice size for one outer index

    char* out_base = reinterpret_cast<char*>(output.data<uint8_t>());

    // Process each input tensor
    int64_t offset_at_dim = 0;
    for (const auto& t : tensors) {
        auto t_cont = t.is_contiguous() ? t : t.contiguous();
        const char* in_base = reinterpret_cast<const char*>(t_cont.data<uint8_t>());
        auto in_shape = t_cont.shape();
        int64_t in_dim_size = in_shape[dim];

        // Fast path: dim=0 and contiguous - single large memcpy per tensor
        if (dim == 0 && outer_size == 1) {
            int64_t total_bytes = t_cont.numel() * elem_size;
            int64_t out_offset = offset_at_dim * inner_size * elem_size;
            std::memcpy(out_base + out_offset, in_base, total_bytes);
        }
        // General case: chunk-based copying with OpenMP
        else {
            int64_t copy_chunk_bytes = inner_size * elem_size;
            int64_t total_chunks = outer_size * in_dim_size;

            // Flatten the nested loops and parallelize over all chunks
            #pragma omp parallel for if(total_chunks > 64)
            for (int64_t chunk = 0; chunk < total_chunks; ++chunk) {
                int64_t outer = chunk / in_dim_size;
                int64_t d = chunk % in_dim_size;

                // Input position: outer * (in_dim_size * inner_size) + d * inner_size
                int64_t in_offset = (outer * in_dim_size + d) * inner_size * elem_size;
                // Output position: outer * out_slice_size + (offset_at_dim + d) * inner_size
                int64_t out_offset = (outer * out_slice_size + (offset_at_dim + d) * inner_size) * elem_size;

                std::memcpy(out_base + out_offset, in_base + in_offset, copy_chunk_bytes);
            }
        }

        offset_at_dim += in_dim_size;
    }

    return output;
}

auto stack(std::span<const Tensor> tensors, int64_t dim) -> Tensor {
    if (tensors.empty()) {
        throw std::runtime_error("Cannot stack empty tensor list");
    }

    // All tensors must have the same shape
    auto first_shape = tensors[0].shape();
    for (size_t i = 1; i < tensors.size(); ++i) {
        auto shape = tensors[i].shape();
        if (shape.size() != first_shape.size()) {
            throw std::runtime_error("All tensors must have the same number of dimensions for stacking");
        }
        for (size_t j = 0; j < shape.size(); ++j) {
            if (shape[j] != first_shape[j]) {
                throw std::runtime_error("All tensors must have the same shape for stacking");
            }
        }
    }

    // Normalize dim
    int64_t ndim = first_shape.size();
    if (dim < 0) {
        dim += ndim + 1;
    }
    if (dim < 0 || dim > ndim) {
        throw std::runtime_error("Dimension out of range for stack");
    }

    // Dispatch to registered stack kernel for single-pass implementation
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    std::vector<Tensor> tensor_vec(tensors.begin(), tensors.end());
    return dispatch(OpId::Stack, std::span<const Tensor>(tensor_vec), attrs)[0];
}

auto split(const Tensor& input, int64_t split_size, int64_t dim) -> std::vector<Tensor> {
    if (split_size <= 0) {
        throw std::invalid_argument("Split size must be positive");
    }

    // Get input shape and normalize dimension
    auto shape = input.shape();
    int64_t ndim = shape.size();

    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Dimension out of range for split");
    }

    int64_t dim_size = shape[dim];

    // Calculate number of splits needed
    int64_t num_splits = (dim_size + split_size - 1) / split_size;  // Ceiling division

    // Split tensor into chunks using slice (creates views, no copying)
    std::vector<Tensor> result;
    result.reserve(num_splits);

    for (int64_t i = 0; i < num_splits; ++i) {
        int64_t start = i * split_size;
        int64_t end = std::min(start + split_size, dim_size);

        // Stop if we've exhausted the dimension (shouldn't happen with ceiling division)
        if (start >= dim_size) {
            break;
        }

        // Use slice to create a view (zero-copy)
        result.push_back(input.slice(dim, start, end));
    }

    return result;
}

auto split_with_sizes(const Tensor& input, const std::vector<int64_t>& split_sizes, int64_t dim) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Dimension out of range for split_with_sizes");
    }

    // Validate split sizes sum to dimension size
    int64_t total = 0;
    for (auto s : split_sizes) {
        if (s < 0) throw std::invalid_argument("split_with_sizes: sizes must be non-negative");
        total += s;
    }
    if (total != shape[dim]) {
        throw std::invalid_argument("split_with_sizes: sizes must sum to dimension size (" +
            std::to_string(total) + " != " + std::to_string(shape[dim]) + ")");
    }

    std::vector<Tensor> result;
    result.reserve(split_sizes.size());
    int64_t offset = 0;
    for (auto s : split_sizes) {
        result.push_back(input.slice(dim, offset, offset + s));
        offset += s;
    }
    return result;
}

auto chunk(const Tensor& input, int64_t chunks, int64_t dim) -> std::vector<Tensor> {
    if (chunks <= 0) {
        throw std::invalid_argument("Number of chunks must be positive");
    }

    // Get input shape and normalize dimension
    auto shape = input.shape();
    int64_t ndim = shape.size();

    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Dimension out of range for chunk");
    }

    int64_t dim_size = shape[dim];

    // Calculate chunk size (ceiling division)
    int64_t chunk_size = (dim_size + chunks - 1) / chunks;

    // Split tensor into chunks
    std::vector<Tensor> result;
    result.reserve(chunks);

    for (int64_t i = 0; i < chunks; ++i) {
        int64_t start = i * chunk_size;
        int64_t end = std::min(start + chunk_size, dim_size);

        // Stop if we've exhausted the dimension
        if (start >= dim_size) {
            break;
        }

        result.push_back(input.slice(dim, start, end));
    }

    return result;
}

auto repeat(const Tensor& input, std::vector<int64_t> repeats) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    // If repeats has more dims than input, unsqueeze leading dimensions
    Tensor padded = input;
    if (repeats.size() > static_cast<size_t>(ndim)) {
        for (size_t i = 0; i < repeats.size() - static_cast<size_t>(ndim); ++i) {
            padded = padded.unsqueeze(0);
        }
        shape = padded.shape();
        ndim = shape.size();
    }

    // Pad repeats with 1s at the front if needed
    while (repeats.size() < static_cast<size_t>(ndim)) {
        repeats.insert(repeats.begin(), 1);
    }

    // Calculate output shape
    std::vector<int64_t> out_shape(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        out_shape[i] = shape[i] * repeats[i];
    }

    // For non-CPU devices, use dispatcher
    if (padded.device().type != Device::Type::CPU) {
        NewOpAttributes attrs;
        attrs.set(AttrKey::Repeats, shape_to_string(repeats));
        std::vector<Tensor> inputs = {padded};
        return dispatch(OpId::Repeat, inputs, attrs)[0];
    }

    // CPU implementation: repeat elements along each dimension
    // Make input contiguous for easier indexing
    auto input_cont = padded.is_contiguous() ? padded : padded.contiguous();

    // Create output tensor
    auto output = empty(out_shape, padded.dtype(), Device::cpu());

    // Calculate total elements
    int64_t total_out = 1;
    for (auto s : out_shape) {
        total_out *= s;
    }

    // Calculate input strides (contiguous)
    std::vector<int64_t> in_strides(ndim);
    int64_t in_stride = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        in_strides[i] = in_stride;
        in_stride *= shape[i];
    }

    // Helper lambda to perform the copy for any dtype
    auto do_repeat = [&]<typename T>() {
        const T* input_data = input_cont.data<T>();
        T* output_data = output.data<T>();

        for (int64_t out_idx = 0; out_idx < total_out; ++out_idx) {
            // Calculate output coordinates
            int64_t temp = out_idx;
            std::vector<int64_t> out_coords(ndim);
            for (int64_t i = ndim - 1; i >= 0; --i) {
                out_coords[i] = temp % out_shape[i];
                temp /= out_shape[i];
            }

            // Map to input coordinates (divide by repeat factor)
            int64_t in_idx = 0;
            for (int64_t i = 0; i < ndim; ++i) {
                int64_t in_coord = out_coords[i] / repeats[i];
                in_idx += in_coord * in_strides[i];
            }

            output_data[out_idx] = input_data[in_idx];
        }
    };

    // Dispatch based on dtype
    switch (input.dtype()) {
        case DType::Float32: do_repeat.template operator()<float>(); break;
        case DType::Float64: do_repeat.template operator()<double>(); break;
        case DType::Float16: do_repeat.template operator()<Float16>(); break;
        case DType::BFloat16: do_repeat.template operator()<BFloat16>(); break;
        case DType::Int8: do_repeat.template operator()<int8_t>(); break;
        case DType::Int16: do_repeat.template operator()<int16_t>(); break;
        case DType::Int32: do_repeat.template operator()<int32_t>(); break;
        case DType::Int64: do_repeat.template operator()<int64_t>(); break;
        default:
            throw std::runtime_error("Unsupported dtype for repeat operation");
    }

    return output;
}

auto tile(const Tensor& input, std::vector<int64_t> reps) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    // Validate reps
    if (reps.empty()) {
        throw std::runtime_error("Tile repetitions cannot be empty");
    }

    // Determine output dimensions
    int64_t out_ndim = std::max(ndim, static_cast<int64_t>(reps.size()));

    // Pad shape and reps with 1s to match dimensions
    std::vector<int64_t> padded_shape(out_ndim, 1);
    std::vector<int64_t> padded_reps(out_ndim, 1);

    // Copy shape (right-aligned)
    int64_t shape_offset = out_ndim - ndim;
    for (int64_t i = 0; i < ndim; ++i) {
        padded_shape[shape_offset + i] = shape[i];
    }

    // Copy reps (right-aligned)
    int64_t reps_offset = out_ndim - static_cast<int64_t>(reps.size());
    for (size_t i = 0; i < reps.size(); ++i) {
        padded_reps[reps_offset + i] = reps[i];
    }

    // Calculate output shape
    std::vector<int64_t> out_shape(out_ndim);
    for (int64_t i = 0; i < out_ndim; ++i) {
        out_shape[i] = padded_shape[i] * padded_reps[i];
    }

    // For non-CPU devices, use dispatcher
    if (input.device().type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Reps, shape_to_string(reps));
        std::vector<Tensor> inputs = {input};
        return dispatch(OpId::Tile, inputs, attrs)[0];
    }

    // CPU implementation: tile entire tensor along each dimension
    // Make contiguous for easier indexing
    auto input_cont = input.is_contiguous() ? input : input.contiguous();

    // Create output tensor
    auto output = empty(out_shape, input.dtype(), Device::cpu());

    // Calculate total elements
    int64_t total_out = 1;
    for (auto s : out_shape) {
        total_out *= s;
    }

    // Calculate input strides (contiguous)
    std::vector<int64_t> in_strides(out_ndim);
    int64_t in_stride = 1;
    for (int64_t i = out_ndim - 1; i >= 0; --i) {
        in_strides[i] = in_stride;
        in_stride *= padded_shape[i];
    }

    // Helper lambda to perform the copy for any dtype
    auto do_tile = [&]<typename T>() {
        const T* input_data = input_cont.data<T>();
        T* output_data = output.data<T>();

        for (int64_t out_idx = 0; out_idx < total_out; ++out_idx) {
            // Calculate output coordinates
            int64_t temp = out_idx;
            std::vector<int64_t> out_coords(out_ndim);
            for (int64_t i = out_ndim - 1; i >= 0; --i) {
                out_coords[i] = temp % out_shape[i];
                temp /= out_shape[i];
            }

            // Map to input coordinates (modulo by input shape)
            int64_t in_idx = 0;
            for (int64_t i = 0; i < out_ndim; ++i) {
                int64_t in_coord = out_coords[i] % padded_shape[i];
                in_idx += in_coord * in_strides[i];
            }

            output_data[out_idx] = input_data[in_idx];
        }
    };

    // Dispatch based on dtype
    switch (input.dtype()) {
        case DType::Float32: do_tile.template operator()<float>(); break;
        case DType::Float64: do_tile.template operator()<double>(); break;
        case DType::Float16: do_tile.template operator()<Float16>(); break;
        case DType::BFloat16: do_tile.template operator()<BFloat16>(); break;
        case DType::Int8: do_tile.template operator()<int8_t>(); break;
        case DType::Int16: do_tile.template operator()<int16_t>(); break;
        case DType::Int32: do_tile.template operator()<int32_t>(); break;
        case DType::Int64: do_tile.template operator()<int64_t>(); break;
        default:
            throw std::runtime_error("Unsupported dtype for tile operation");
    }

    return output;
}

auto expand(const Tensor& input, std::vector<int64_t> shape) -> Tensor {
    auto input_shape = input.shape();

    // Validate expansion is possible
    if (shape.size() < input_shape.size()) {
        throw std::runtime_error("Expanded shape must have at least as many dimensions as input");
    }

    // Check if already the right shape
    bool same_shape = (shape.size() == input_shape.size());
    if (same_shape) {
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] != input_shape[i]) {
                same_shape = false;
                break;
            }
        }
    }
    if (same_shape) {
        return input;
    }

    // Use dispatcher for non-CPU devices (expand takes an input tensor)
    if (input.device().type != Device::Type::CPU) {
        NewOpAttributes attrs;
        attrs.set(AttrKey::Shape, shape_to_string(shape));

        std::vector<Tensor> inputs = {input};
        return dispatch(OpId::Expand, inputs, attrs)[0];
    }

    // CPU path: Manual implementation — ensure contiguous layout
    auto cont_input = input.is_contiguous() ? input : input.contiguous();
    auto input_shape_vec = std::vector<int64_t>(cont_input.shape().begin(), cont_input.shape().end());

    // Create output tensor on CPU
    auto output = zeros(shape, input.dtype(), Device::cpu());

    // Calculate strides
    std::vector<int64_t> input_strides(input_shape_vec.size());
    int64_t input_stride = 1;
    for (int i = input_shape_vec.size() - 1; i >= 0; --i) {
        input_strides[i] = input_stride;
        input_stride *= input_shape_vec[i];
    }

    // Calculate total output elements
    int64_t total_elements = 1;
    for (auto s : shape) {
        total_elements *= s;
    }

    // Fill output by replicating input - dtype-aware implementation
    if (cont_input.dtype() == DType::Float64) {
        const double* input_data = cont_input.data<double>();
        double* output_data = output.data<double>();

        for (int64_t out_idx = 0; out_idx < total_elements; ++out_idx) {
            int64_t temp = out_idx;
            int64_t in_idx = 0;

            int input_dim_offset = shape.size() - input_shape_vec.size();

            for (int i = shape.size() - 1; i >= 0; --i) {
                int64_t coord = temp % shape[i];
                temp /= shape[i];

                int input_dim = i - input_dim_offset;
                if (input_dim >= 0 && input_dim < static_cast<int>(input_shape_vec.size())) {
                    // Map to input dimension (handle size-1 dimensions)
                    if (input_shape_vec[input_dim] == 1) {
                        coord = 0;
                    } else if (input_shape_vec[input_dim] != shape[i]) {
                        throw std::runtime_error("Cannot expand dimension from non-1 size to different size");
                    }
                    in_idx += coord * input_strides[input_dim];
                }
            }

            output_data[out_idx] = input_data[in_idx];
        }
    } else if (cont_input.dtype() == DType::Float16) {
        const Float16* input_data = cont_input.data<Float16>();
        Float16* output_data = output.data<Float16>();

        for (int64_t out_idx = 0; out_idx < total_elements; ++out_idx) {
            int64_t temp = out_idx;
            int64_t in_idx = 0;

            int input_dim_offset = shape.size() - input_shape_vec.size();

            for (int i = shape.size() - 1; i >= 0; --i) {
                int64_t coord = temp % shape[i];
                temp /= shape[i];

                int input_dim = i - input_dim_offset;
                if (input_dim >= 0 && input_dim < static_cast<int>(input_shape_vec.size())) {
                    // Map to input dimension (handle size-1 dimensions)
                    if (input_shape_vec[input_dim] == 1) {
                        coord = 0;
                    } else if (input_shape_vec[input_dim] != shape[i]) {
                        throw std::runtime_error("Cannot expand dimension from non-1 size to different size");
                    }
                    in_idx += coord * input_strides[input_dim];
                }
            }

            output_data[out_idx] = input_data[in_idx];
        }
    } else {
        // Helper lambda for expand logic
        auto expand_impl = [&]<typename T>(T*) {
            const T* input_data = cont_input.data<T>();
            T* output_data = output.data<T>();

            for (int64_t out_idx = 0; out_idx < total_elements; ++out_idx) {
                int64_t temp = out_idx;
                int64_t in_idx = 0;

                int input_dim_offset = shape.size() - input_shape_vec.size();

                for (int i = shape.size() - 1; i >= 0; --i) {
                    int64_t coord = temp % shape[i];
                    temp /= shape[i];

                    int input_dim = i - input_dim_offset;
                    if (input_dim >= 0 && input_dim < static_cast<int>(input_shape_vec.size())) {
                        // Map to input dimension (handle size-1 dimensions)
                        if (input_shape_vec[input_dim] == 1) {
                            coord = 0;
                        } else if (input_shape_vec[input_dim] != shape[i]) {
                            throw std::runtime_error("Cannot expand dimension from non-1 size to different size");
                        }
                        in_idx += coord * input_strides[input_dim];
                    }
                }

                output_data[out_idx] = input_data[in_idx];
            }
        };

        // Dispatch based on dtype
        switch (cont_input.dtype()) {
            case DType::Float16:
                expand_impl(static_cast<Float16*>(nullptr));
                break;
            case DType::BFloat16:
                expand_impl(static_cast<BFloat16*>(nullptr));
                break;
            case DType::Float32:
                expand_impl(static_cast<float*>(nullptr));
                break;
            case DType::Float64:
                expand_impl(static_cast<double*>(nullptr));
                break;
            case DType::Int32:
                expand_impl(static_cast<int32_t*>(nullptr));
                break;
            case DType::Int64:
                expand_impl(static_cast<int64_t*>(nullptr));
                break;
            default:
                throw std::runtime_error("Unsupported dtype for expand");
        }
    }

    return output;
}

auto roll(const Tensor& input, int64_t shifts, int64_t dim) -> Tensor {
    // Normalize dimension
    int64_t ndim = input.ndim();
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Dimension out of range for roll");
    }

    int64_t dim_size = input.shape()[dim];
    if (dim_size == 0) {
        return input.clone();
    }

    // Normalize shifts to [0, dim_size)
    shifts = shifts % dim_size;
    if (shifts < 0) {
        shifts += dim_size;
    }

    if (shifts == 0) {
        return input.clone();
    }

    std::array<Tensor, 1> inputs = {input};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Shift, shifts);
    attrs.set(AttrKey::Dim, dim);
    return dispatch<OpId::Roll>(inputs, attrs)[0];
}

// =========================================================================
// Triangular, Diagonal, and Flip Operations
// =========================================================================

namespace {
// Helper: apply a per-element operation on the last two dims of a tensor
template<typename Func>
auto triangular_op(const Tensor& input, int64_t diagonal, Func&& should_keep) -> Tensor {
    if (input.ndim() < 2) {
        throw std::runtime_error("triu/tril requires at least 2D tensor");
    }

    auto cont = input.is_contiguous() ? input : input.contiguous();
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto result = zeros(shape_vec, input.dtype(), input.device());

    auto shape = cont.shape();
    int64_t rows = shape[input.ndim() - 2];
    int64_t cols = shape[input.ndim() - 1];
    int64_t batch_size = cont.numel() / (rows * cols);
    auto elem_size = dtype_size(input.dtype());

    const auto* src = static_cast<const uint8_t*>(cont.data_ptr());
    auto* dst = static_cast<uint8_t*>(result.data_ptr());

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                if (should_keep(i, j, diagonal)) {
                    int64_t idx = (b * rows * cols + i * cols + j) * elem_size;
                    std::memcpy(dst + idx, src + idx, elem_size);
                }
            }
        }
    }
    return result;
}
} // anonymous namespace

auto triu(const Tensor& input, int64_t diagonal) -> Tensor {
    if (input.device().type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Diagonal, diagonal);
        return dispatch(OpId::Triu, std::span<const Tensor>(&input, 1), attrs)[0];
    }
    return triangular_op(input, diagonal, [](int64_t i, int64_t j, int64_t d) {
        return j >= i + d;  // Keep upper triangle
    });
}

auto tril(const Tensor& input, int64_t diagonal) -> Tensor {
    if (input.device().type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Diagonal, diagonal);
        return dispatch(OpId::Tril, std::span<const Tensor>(&input, 1), attrs)[0];
    }
    return triangular_op(input, diagonal, [](int64_t i, int64_t j, int64_t d) {
        return j <= i + d;  // Keep lower triangle
    });
}

auto diag(const Tensor& input, int64_t diagonal) -> Tensor {
    if (input.ndim() == 1) {
        // 1D -> 2D: construct diagonal matrix
        int64_t n = input.shape()[0];
        int64_t abs_diag = std::abs(diagonal);
        int64_t size = n + abs_diag;
        auto result = zeros({size, size}, input.dtype(), input.device());

        if (input.device().type != Device::Type::CPU) {
            OpAttributes attrs;
            attrs.set(AttrKey::Diagonal, diagonal);
            return dispatch(OpId::Diag, std::span<const Tensor>(&input, 1), attrs)[0];
        }

        auto cont = input.is_contiguous() ? input : input.contiguous();
        auto elem_size = dtype_size(input.dtype());
        const auto* src = static_cast<const uint8_t*>(cont.data_ptr());
        auto* dst = static_cast<uint8_t*>(result.data_ptr());

        for (int64_t i = 0; i < n; ++i) {
            int64_t row = diagonal >= 0 ? i : i - diagonal;
            int64_t col = diagonal >= 0 ? i + diagonal : i;
            int64_t dst_idx = (row * size + col) * elem_size;
            int64_t src_idx = i * elem_size;
            std::memcpy(dst + dst_idx, src + src_idx, elem_size);
        }
        return result;
    } else if (input.ndim() == 2) {
        // 2D -> 1D: extract diagonal
        if (input.device().type != Device::Type::CPU) {
            OpAttributes attrs;
            attrs.set(AttrKey::Diagonal, diagonal);
            return dispatch(OpId::Diag, std::span<const Tensor>(&input, 1), attrs)[0];
        }

        auto cont = input.is_contiguous() ? input : input.contiguous();
        int64_t rows = input.shape()[0];
        int64_t cols = input.shape()[1];
        int64_t start_row = diagonal >= 0 ? 0 : -diagonal;
        int64_t start_col = diagonal >= 0 ? diagonal : 0;
        int64_t diag_len = std::min(rows - start_row, cols - start_col);
        if (diag_len <= 0) {
            return empty({0}, input.dtype(), input.device());
        }

        auto result = empty({diag_len}, input.dtype(), input.device());
        auto elem_size = dtype_size(input.dtype());
        const auto* src = static_cast<const uint8_t*>(cont.data_ptr());
        auto* dst = static_cast<uint8_t*>(result.data_ptr());

        for (int64_t i = 0; i < diag_len; ++i) {
            int64_t src_idx = ((start_row + i) * cols + (start_col + i)) * elem_size;
            int64_t dst_idx = i * elem_size;
            std::memcpy(dst + dst_idx, src + src_idx, elem_size);
        }
        return result;
    } else {
        throw std::runtime_error("diag: input must be 1D or 2D, got " +
                                 std::to_string(input.ndim()) + "D");
    }
}

auto trace(const Tensor& input) -> Tensor {
    if (input.ndim() < 2) {
        throw std::runtime_error("trace requires at least 2D tensor");
    }

    if (input.device().type != Device::Type::CPU) {
        return dispatch(OpId::Trace, std::span<const Tensor>(&input, 1), {})[0];
    }

    // Extract main diagonal and sum
    auto diagonal = diag(input, 0);
    return tenzor::sum(diagonal, std::nullopt, false);
}

auto flip(const Tensor& input, std::vector<int64_t> dims) -> Tensor {
    auto ndim = input.ndim();
    // Normalize negative dims
    for (auto& d : dims) {
        if (d < 0) d += ndim;
        if (d < 0 || d >= ndim) {
            throw std::runtime_error("flip: dimension " + std::to_string(d) + " out of range for " +
                                     std::to_string(ndim) + "D tensor");
        }
    }

    if (input.device().type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dims, shape_to_string(dims));
        return dispatch(OpId::Flip, std::span<const Tensor>(&input, 1), attrs)[0];
    }

    // Flip by reversing slices along each dimension
    auto result = input;
    for (auto dim : dims) {
        int64_t dim_size = result.shape()[dim];
        if (dim_size <= 1) continue;

        std::vector<Tensor> slices;
        slices.reserve(dim_size);
        for (int64_t i = dim_size - 1; i >= 0; --i) {
            slices.push_back(result.slice(dim, i, i + 1));
        }
        result = cat(slices, dim);
    }
    return result;
}

auto movedim(const Tensor& input, std::vector<int64_t> source, std::vector<int64_t> destination) -> Tensor {
    int64_t ndim = input.ndim();
    if (source.size() != destination.size()) {
        throw std::invalid_argument("movedim: source and destination must have the same length");
    }

    // Normalize negative dims
    for (auto& s : source) { if (s < 0) s += ndim; }
    for (auto& d : destination) { if (d < 0) d += ndim; }

    // Build the permutation
    // Start with all dims not in source, then place source dims at destination positions
    std::vector<int64_t> perm(ndim, -1);
    std::vector<bool> used_src(ndim, false);
    std::vector<bool> used_dst(ndim, false);

    // Place source[i] at destination[i]
    for (size_t i = 0; i < source.size(); ++i) {
        perm[destination[i]] = source[i];
        used_src[source[i]] = true;
        used_dst[destination[i]] = true;
    }

    // Fill remaining positions with remaining dims in order
    int64_t src_idx = 0;
    for (int64_t i = 0; i < ndim; ++i) {
        if (!used_dst[i]) {
            while (src_idx < ndim && used_src[src_idx]) ++src_idx;
            perm[i] = src_idx++;
        }
    }

    return tenzor::permute(input, perm);
}

auto swapaxes(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor {
    return tenzor::transpose(input, dim0, dim1);
}

// =========================================================================
// Repeat Interleave
// =========================================================================

auto repeat_interleave(const Tensor& input, int64_t repeats,
                       std::optional<int64_t> dim) -> Tensor {
    if (repeats < 0) {
        throw std::invalid_argument("repeat_interleave: repeats must be non-negative");
    }

    // If dim not specified, flatten input first
    Tensor src = dim.has_value() ? input : tenzor::flatten(input);
    int64_t actual_dim = dim.value_or(0);

    // Normalize negative dim
    int64_t ndim = src.ndim();
    if (actual_dim < 0) {
        actual_dim += ndim;
    }
    if (actual_dim < 0 || actual_dim >= ndim) {
        throw std::runtime_error("repeat_interleave: dimension out of range");
    }

    auto shape = src.shape();
    int64_t dim_size = shape[actual_dim];

    // Build output shape
    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[actual_dim] = dim_size * repeats;

    if (out_shape[actual_dim] == 0) {
        return zeros(out_shape, src.dtype(), src.device());
    }

    // Dispatch to backend
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, actual_dim);
    attrs.set(AttrKey::NumRepeats, repeats);

    std::vector<Tensor> inputs = {src};
    return dispatch(OpId::RepeatInterleave, std::span<const Tensor>(inputs), attrs)[0];
}

auto repeat_interleave(const Tensor& input, const Tensor& repeats,
                       std::optional<int64_t> dim) -> Tensor {
    if (repeats.ndim() != 1) {
        throw std::invalid_argument("repeat_interleave: repeats tensor must be 1D");
    }

    // If dim not specified, flatten input first
    Tensor src = dim.has_value() ? input : tenzor::flatten(input);
    int64_t actual_dim = dim.value_or(0);

    // Normalize negative dim
    int64_t ndim = src.ndim();
    if (actual_dim < 0) {
        actual_dim += ndim;
    }
    if (actual_dim < 0 || actual_dim >= ndim) {
        throw std::runtime_error("repeat_interleave: dimension out of range");
    }

    auto shape = src.shape();
    int64_t dim_size = shape[actual_dim];

    if (repeats.shape()[0] != dim_size) {
        throw std::invalid_argument(
            "repeat_interleave: repeats tensor size (" + std::to_string(repeats.shape()[0]) +
            ") must match dimension size (" + std::to_string(dim_size) + ")");
    }

    // Dispatch to backend — pass repeats tensor as second input, NumRepeats = -1 signals tensor mode
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, actual_dim);
    attrs.set(AttrKey::NumRepeats, static_cast<int64_t>(-1));

    std::vector<Tensor> inputs = {src, repeats};
    return dispatch(OpId::RepeatInterleave, std::span<const Tensor>(inputs), attrs)[0];
}

// =========================================================================
// New shape/transform operations for PyTorch parity (compositions)
// =========================================================================

auto ravel(const Tensor& input) -> Tensor {
    return reshape(input, {-1});
}

auto unflatten(const Tensor& input, int64_t dim, std::vector<int64_t> sizes) -> Tensor {
    const auto& shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;

    std::vector<int64_t> new_shape;
    for (int64_t i = 0; i < ndim; i++) {
        if (i == dim) {
            for (auto s : sizes) new_shape.push_back(s);
        } else {
            new_shape.push_back(shape[i]);
        }
    }
    return reshape(input, new_shape);
}

auto hstack(const std::vector<Tensor>& tensors) -> Tensor {
    if (tensors.empty()) throw std::runtime_error("hstack: empty tensor list");
    if (tensors[0].ndim() == 1) {
        return cat(std::span<const Tensor>(tensors), 0);
    }
    return cat(std::span<const Tensor>(tensors), 1);
}

auto vstack(const std::vector<Tensor>& tensors) -> Tensor {
    if (tensors.empty()) throw std::runtime_error("vstack: empty tensor list");
    // Ensure at least 2D
    std::vector<Tensor> reshaped;
    for (const auto& t : tensors) {
        if (t.ndim() == 1) {
            reshaped.push_back(reshape(t, {1, t.shape()[0]}));
        } else {
            reshaped.push_back(t);
        }
    }
    return cat(std::span<const Tensor>(reshaped), 0);
}

auto dstack(const std::vector<Tensor>& tensors) -> Tensor {
    if (tensors.empty()) throw std::runtime_error("dstack: empty tensor list");
    // Ensure at least 3D
    std::vector<Tensor> reshaped;
    for (const auto& t : tensors) {
        if (t.ndim() == 1) {
            reshaped.push_back(reshape(t, {1, t.shape()[0], 1}));
        } else if (t.ndim() == 2) {
            reshaped.push_back(reshape(t, {t.shape()[0], t.shape()[1], 1}));
        } else {
            reshaped.push_back(t);
        }
    }
    return cat(std::span<const Tensor>(reshaped), 2);
}

auto tensor_split(const Tensor& input, int64_t sections, int64_t dim) -> std::vector<Tensor> {
    const auto& shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t dim_size = shape[dim];
    int64_t chunk_size = (dim_size + sections - 1) / sections;

    std::vector<Tensor> result;
    for (int64_t i = 0; i < sections; i++) {
        int64_t start = i * chunk_size;
        int64_t end = std::min(start + chunk_size, dim_size);
        if (start >= dim_size) break;
        result.push_back(input.slice(dim, start, end));
    }
    return result;
}

auto tensor_split(const Tensor& input, std::vector<int64_t> indices, int64_t dim) -> std::vector<Tensor> {
    const auto& shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t dim_size = shape[dim];

    std::vector<Tensor> result;
    int64_t prev = 0;
    for (auto idx : indices) {
        if (idx > dim_size) idx = dim_size;
        result.push_back(input.slice(dim, prev, idx));
        prev = idx;
    }
    result.push_back(input.slice(dim, prev, dim_size));
    return result;
}

auto hsplit(const Tensor& input, int64_t sections) -> std::vector<Tensor> {
    if (input.ndim() == 1) return tensor_split(input, sections, 0);
    return tensor_split(input, sections, 1);
}

auto vsplit(const Tensor& input, int64_t sections) -> std::vector<Tensor> {
    return tensor_split(input, sections, 0);
}

auto dsplit(const Tensor& input, int64_t sections) -> std::vector<Tensor> {
    if (input.ndim() < 3) throw std::runtime_error("dsplit requires tensor with ndim >= 3");
    return tensor_split(input, sections, 2);
}

auto unbind(const Tensor& input, int64_t dim) -> std::vector<Tensor> {
    const auto& shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t n = shape[dim];

    std::vector<Tensor> result;
    result.reserve(n);
    for (int64_t i = 0; i < n; i++) {
        result.push_back(squeeze(input.slice(dim, i, i + 1), dim));
    }
    return result;
}

auto rot90(const Tensor& input, int64_t k, std::vector<int64_t> dims) -> Tensor {
    if (dims.size() != 2) throw std::runtime_error("rot90: dims must have exactly 2 elements");
    int64_t ndim = static_cast<int64_t>(input.shape().size());
    int64_t d0 = dims[0] < 0 ? dims[0] + ndim : dims[0];
    int64_t d1 = dims[1] < 0 ? dims[1] + ndim : dims[1];

    k = ((k % 4) + 4) % 4; // normalize to [0, 3]
    if (k == 0) return input;

    Tensor result = input;
    for (int64_t i = 0; i < k; i++) {
        result = flip(result, {d1});
        // Transpose d0 and d1
        std::vector<int64_t> perm;
        for (int64_t d = 0; d < static_cast<int64_t>(result.shape().size()); d++) {
            if (d == d0) perm.push_back(d1);
            else if (d == d1) perm.push_back(d0);
            else perm.push_back(d);
        }
        result = result.permute(perm);
    }
    return result;
}

auto argwhere(const Tensor& input) -> Tensor {
    // nonzero returns (nnz, ndim) tensor of indices
    return nonzero(input);
}

auto fliplr(const Tensor& input) -> Tensor {
    if (input.ndim() < 2) throw std::runtime_error("fliplr requires tensor with ndim >= 2");
    return flip(input, {1});
}

auto flipud(const Tensor& input) -> Tensor {
    if (input.ndim() < 1) throw std::runtime_error("flipud requires tensor with ndim >= 1");
    return flip(input, {0});
}

} // namespace tenzor
