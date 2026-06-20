#include "tenzor/core/tensor.hpp"
#include "../sycl_prefix_sum.hpp"
#include "oneapi_kernel_utils.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>
#include <vector>
#include "tenzor/ops/creation.hpp"

#ifdef TENZOR_HAS_ONEDPL
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#endif

namespace tenzor {
namespace oneapi {

// Kernel class declarations for SYCL
class GatherKernelFloat32;
class GatherKernelFloat64;
class GatherKernelFloat16;
class ScatterKernelFloat32;
class ScatterKernelFloat64;
class ScatterKernelFloat16;
class IndexSelectKernelFloat32;
class IndexSelectKernelFloat64;
class IndexSelectKernelFloat16;
class IndexSelectKernelInt32;
class IndexSelectKernelInt64;
class IndexSelectKernelBool;
class MaskedFillKernelFloat32;
class MaskedFillKernelFloat64;
class MaskedFillKernelFloat16;
class MaskedFillKernelInt32;
class MaskedFillKernelInt64;
class MaskedFillKernelBool;
class GatherKernelBFloat16;
class ScatterKernelBFloat16;
class IndexSelectKernelBFloat16;
class MaskedFillKernelBFloat16;
class IndexAddKernelF32;
class IndexCopyKernelF32;
class IndexFillKernelF32;
class ScatterReduceSumKernelF32;
class ScatterReduceProdKernelF32;
class ScatterReduceAmaxKernelF32;
class ScatterReduceAminKernelF32;
class ScatterReduceInitKernelF32;
class ScatterReduceMeanDivKernelF32;

// Kernel names for device-side masked_select and nonzero
class MaskedSelectPrefixSumUpSweep;
class MaskedSelectPrefixSumDownSweep;
class MaskedSelectScatterFloat32;
class MaskedSelectScatterFloat64;
class MaskedSelectScatterFloat16;
class MaskedSelectScatterBFloat16;
class NonzeroBinaryMaskFloat32;
class NonzeroBinaryMaskFloat64;
class NonzeroBinaryMaskFloat16;
class NonzeroBinaryMaskBFloat16;
class NonzeroBinaryMaskInt32;
class NonzeroBinaryMaskInt64;
class NonzeroBinaryMaskBool;
class NonzeroScatterIndices;


// Gather operation - collect values at specified indices along a dimension
auto gather_kernel(const Tensor& input_in, int64_t dim, const Tensor& index_in, sycl::queue& queue) -> Tensor {
    // The kernel indexes input via contiguous strides derived from shape and
    // reads the index by flat position, so both operands must be contiguous.
    Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();
    Tensor index = index_in.is_contiguous() ? index_in : index_in.contiguous();

    auto input_shape_span = input.shape();
    auto index_shape_span = index.shape();

    // Convert spans to vectors
    std::vector<int64_t> input_shape(input_shape_span.begin(), input_shape_span.end());
    std::vector<int64_t> index_shape(index_shape_span.begin(), index_shape_span.end());

    // Validate dimension
    if (dim < 0) dim += input_shape.size();
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::invalid_argument("Gather: invalid dimension");
    }

    // Output has same shape as index
    Tensor output(index_shape, input.dtype(), input.device());

    auto input_strides = calculate_strides(input_shape);
    auto index_strides = calculate_strides(index_shape);

    const int64_t numel = output.numel();

    // Copy vectors to arrays for device copyability (max 8 dimensions)
    int64_t input_shape_arr[8], input_strides_arr[8], index_strides_arr[8];
    const size_t ndims = index_shape.size();
    if (ndims > 8) throw std::invalid_argument("oneapi indexing: tensor rank > 8 is unsupported (on-device stride arrays are fixed at 8 dims)");
    for (size_t i = 0; i < ndims && i < 8; ++i) {
        input_shape_arr[i] = input_shape[i];
        input_strides_arr[i] = input_strides[i];
        index_strides_arr[i] = index_strides[i];
    }

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = get_data_ptr<const float>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        float* output_ptr = get_data_ptr<float>(output);

        queue.parallel_for<GatherKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            // Compute multi-dimensional index in output
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides_arr[d];
                temp %= index_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    // Use index tensor to determine coordinate
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* input_ptr = get_data_ptr<const double>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        double* output_ptr = get_data_ptr<double>(output);

        queue.parallel_for<GatherKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides_arr[d];
                temp %= index_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* input_ptr = get_data_ptr<const sycl::half>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        sycl::half* output_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<GatherKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides_arr[d];
                temp %= index_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* input_ptr = get_data_ptr<const uint16_t>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        uint16_t* output_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<GatherKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides_arr[d];
                temp %= index_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() != DType::QInt4x2 &&
             (input.dtype_size() == 1 || input.dtype_size() == 2 ||
              input.dtype_size() == 4 || input.dtype_size() == 8 ||
              input.dtype_size() == 16)) {
        // Generic gather for any remaining fixed-width dtype (Int16/UInt16/32/64,
        // Complex64/128): pure element movement keyed by byte width.
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        auto gen = [&]<typename U>(int64_t lanes) {
            const U* in_ptr = get_data_ptr<const U>(input);
            U* out_ptr = get_data_ptr<U>(output);
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx, input_idx = 0;
                for (size_t d = 0; d < ndims; ++d) {
                    int64_t coord = temp / index_strides_arr[d];
                    temp %= index_strides_arr[d];
                    if (static_cast<int64_t>(d) == dim) {
                        int64_t iv = index_ptr[flat_idx];
                        if (iv < 0) iv += input_shape_arr[d];
                        input_idx += iv * input_strides_arr[d];
                    } else {
                        input_idx += coord * input_strides_arr[d];
                    }
                }
                for (int64_t l = 0; l < lanes; ++l)
                    out_ptr[flat_idx * lanes + l] = in_ptr[input_idx * lanes + l];
            });
        };
        switch (input.dtype_size()) {
            case 1:  gen.template operator()<uint8_t>(1);  break;
            case 2:  gen.template operator()<uint16_t>(1); break;
            case 4:  gen.template operator()<uint32_t>(1); break;
            case 8:  gen.template operator()<uint64_t>(1); break;
            case 16: gen.template operator()<uint64_t>(2); break;
        }
    }
    else {
        throw std::runtime_error("Unsupported dtype for gather");
    }

    return output;
}

// Scatter operation - distribute values at specified indices along a dimension
auto scatter_kernel(const Tensor& input, int64_t dim, const Tensor& index_in,
                    const Tensor& src_in, sycl::queue& queue) -> Tensor {
    // index and src are read via flat row-major offsets derived from
    // index_shape, so both must be contiguous to read the correct elements.
    Tensor index = index_in.is_contiguous() ? index_in : index_in.contiguous();
    Tensor src = src_in.is_contiguous() ? src_in : src_in.contiguous();

    auto input_shape_span = input.shape();
    std::vector<int64_t> input_shape(input_shape_span.begin(), input_shape_span.end());

    // Validate dimension
    if (dim < 0) dim += input_shape.size();
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::invalid_argument("Scatter: invalid dimension");
    }

    // Create output as copy of input
    Tensor output(input_shape, input.dtype(), input.device());

    // Copy input to output first. Force a contiguous source — if `input` is a
    // non-contiguous view the raw memcpy would read the wrong layout, leaving
    // non-scattered positions holding stale/unrelated data and causing the
    // gradcheck's sum() to see non-deterministic output across calls with the
    // same input.
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const size_t bytes = input_cont.numel() * input_cont.dtype_size();
    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input_cont);
        float* out_ptr = get_data_ptr<float>(output);
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input_cont);
        double* out_ptr = get_data_ptr<double>(output);
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
    } else {
        // Any other dtype (int/uint/complex): the seed copy is pure bytes.
        queue.memcpy(const_cast<void*>(output.data_ptr()),
                     input_cont.data_ptr(), bytes).wait();
    }

    auto index_shape_span = index.shape();
    std::vector<int64_t> index_shape(index_shape_span.begin(), index_shape_span.end());
    auto input_strides = calculate_strides(input_shape);
    auto index_strides = calculate_strides(index_shape);

    const int64_t numel = index.numel();

    // Copy vectors to arrays for device copyability (max 8 dimensions)
    int64_t input_strides_arr[8], index_strides_arr[8];
    const size_t ndims = index_shape.size();
    if (ndims > 8) throw std::invalid_argument("oneapi indexing: tensor rank > 8 is unsupported (on-device stride arrays are fixed at 8 dims)");
    if (input_shape.size() != ndims) {
        throw std::invalid_argument("scatter: index and self must have the same number of dimensions");
    }
    for (size_t i = 0; i < ndims && i < 8; ++i) {
        input_strides_arr[i] = input_strides[i];
        index_strides_arr[i] = index_strides[i];
    }

    const int64_t input_dim_size = input_shape[dim];
    const int64_t index_dim_size = index_shape[dim];
    const int64_t idx_dim_stride = index_strides[dim];
    const int64_t ndim = static_cast<int64_t>(ndims);

    const int64_t* index_ptr = get_data_ptr<const int64_t>(index);

    // Deterministic scatter: one work-item per INDEX element. Addressing the
    // index/src buffers uses the INDEX's own (smaller-or-equal) extents and
    // strides — never the input's — so an index/src smaller than self along any
    // non-scatter dim is handled correctly with no out-of-bounds reads. Each
    // item decomposes its flat position into per-dim coords via index strides,
    // maps the non-scatter coords directly onto the output (they share dim
    // order, index.size(d) <= self.size(d)) and uses the index value for the
    // scatter dim. For PyTorch "last write wins" determinism with duplicate
    // targets, an item only writes if no LATER element along the index scatter
    // dim in the same column targets the same output slot.
    auto run = [&]<typename T>() {
        T* output_ptr = get_data_ptr<T>(output);
        const T* src_ptr = get_data_ptr<const T>(src);
        queue.parallel_for(
            sycl::range<1>(static_cast<size_t>(numel)),
            [=](sycl::id<1> id) {
                const int64_t flat = id[0];

                // Decompose flat index position into per-dim coords (index strides)
                // and accumulate the output offset (input strides). The scatter-dim
                // coordinate k and the target value v are tracked separately.
                int64_t remaining = flat;
                int64_t out_offset = 0;
                int64_t k = 0;
                for (int64_t d = 0; d < ndim; ++d) {
                    int64_t coord = remaining / index_strides_arr[d];
                    remaining %= index_strides_arr[d];
                    if (d == dim) {
                        k = coord;
                    } else {
                        out_offset += coord * input_strides_arr[d];
                    }
                }

                int64_t v = index_ptr[flat];
                if (v < 0) v += input_dim_size;

                // Last-write-wins: skip if a later k' in the same column targets v.
                for (int64_t kp = k + 1; kp < index_dim_size; ++kp) {
                    int64_t vp = index_ptr[flat + (kp - k) * idx_dim_stride];
                    if (vp < 0) vp += input_dim_size;
                    if (vp == v) return;  // a later element wins
                }

                out_offset += v * input_strides_arr[dim];
                output_ptr[out_offset] = src_ptr[flat];
            }).wait();
    };

    struct U128 { uint64_t a, b; };
    if (input.dtype() == DType::Float32)  run.template operator()<float>();
    else if (input.dtype() == DType::Float64) run.template operator()<double>();
    else if (input.dtype() == DType::Float16) run.template operator()<sycl::half>();
    else if (input.dtype() == DType::BFloat16) run.template operator()<uint16_t>();
    else if (input.dtype() == DType::Int8)   run.template operator()<int8_t>();
    else if (input.dtype() == DType::Int16)  run.template operator()<int16_t>();
    else if (input.dtype() == DType::Int32)  run.template operator()<int32_t>();
    else if (input.dtype() == DType::Int64)  run.template operator()<int64_t>();
    else if (input.dtype() == DType::UInt8)  run.template operator()<uint8_t>();
    else if (input.dtype() == DType::UInt16) run.template operator()<uint16_t>();
    else if (input.dtype() == DType::UInt32) run.template operator()<uint32_t>();
    else if (input.dtype() == DType::UInt64) run.template operator()<uint64_t>();
    else if (input.dtype() == DType::Bool)   run.template operator()<uint8_t>();
    else if (input.dtype() == DType::Complex64)  run.template operator()<uint64_t>();   // 8 bytes
    else if (input.dtype() == DType::Complex128) run.template operator()<U128>();        // 16 bytes
    else {
        throw std::runtime_error("Unsupported dtype for scatter");
    }

    return output;
}

// Index select - select elements along dimension using 1D index tensor
auto index_select_kernel(const Tensor& input_in, int64_t dim, const Tensor& index_in, sycl::queue& queue) -> Tensor {
    // The kernel indexes input via contiguous strides derived from shape and
    // reads the index by flat position, so both operands must be contiguous.
    Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();
    Tensor index = index_in.is_contiguous() ? index_in : index_in.contiguous();

    auto input_shape_span = input.shape();
    std::vector<int64_t> input_shape(input_shape_span.begin(), input_shape_span.end());

    // Validate dimension
    if (dim < 0) dim += input_shape.size();
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::invalid_argument("IndexSelect: invalid dimension");
    }

    // Output shape: replace dimension size with index size
    std::vector<int64_t> output_shape = input_shape;
    output_shape[dim] = index.numel();

    Tensor output(output_shape, input.dtype(), input.device());

    auto input_strides = calculate_strides(input_shape);
    auto output_strides = calculate_strides(output_shape);

    const int64_t numel = output.numel();

    // Copy vectors to arrays for device copyability (max 8 dimensions)
    int64_t input_shape_arr[8], input_strides_arr[8], output_strides_arr[8];
    const size_t ndims = output_shape.size();
    if (ndims > 8) throw std::invalid_argument("oneapi indexing: tensor rank > 8 is unsupported (on-device stride arrays are fixed at 8 dims)");
    for (size_t i = 0; i < ndims && i < 8; ++i) {
        input_shape_arr[i] = input_shape[i];
        input_strides_arr[i] = input_strides[i];
        output_strides_arr[i] = output_strides[i];
    }

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = get_data_ptr<const float>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        float* output_ptr = get_data_ptr<float>(output);

        queue.parallel_for<IndexSelectKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            // Compute multi-dimensional index in output
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / output_strides_arr[d];
                temp %= output_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    // Use index tensor
                    int64_t idx_val = index_ptr[coord];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* input_ptr = get_data_ptr<const double>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        double* output_ptr = get_data_ptr<double>(output);

        queue.parallel_for<IndexSelectKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / output_strides_arr[d];
                temp %= output_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[coord];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* input_ptr = get_data_ptr<const sycl::half>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        sycl::half* output_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<IndexSelectKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / output_strides_arr[d];
                temp %= output_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[coord];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* input_ptr = get_data_ptr<const uint16_t>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        uint16_t* output_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<IndexSelectKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / output_strides_arr[d];
                temp %= output_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[coord];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* input_ptr = get_data_ptr<const int32_t>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        int32_t* output_ptr = get_data_ptr<int32_t>(output);

        queue.parallel_for<IndexSelectKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;
            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / output_strides_arr[d];
                temp %= output_strides_arr[d];
                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[coord];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }
            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::Int64) {
        const int64_t* input_ptr = get_data_ptr<const int64_t>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        int64_t* output_ptr = get_data_ptr<int64_t>(output);

        queue.parallel_for<IndexSelectKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;
            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / output_strides_arr[d];
                temp %= output_strides_arr[d];
                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[coord];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }
            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::Bool) {
        const bool* input_ptr = get_data_ptr<const bool>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        bool* output_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<IndexSelectKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;
            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / output_strides_arr[d];
                temp %= output_strides_arr[d];
                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[coord];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }
            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() != DType::QInt4x2 &&
             (input.dtype_size() == 1 || input.dtype_size() == 2 ||
              input.dtype_size() == 4 || input.dtype_size() == 8 ||
              input.dtype_size() == 16)) {
        // Generic index_select for any remaining fixed-width dtype.
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        auto gen = [&]<typename U>(int64_t lanes) {
            const U* input_ptr = get_data_ptr<const U>(input);
            U* output_ptr = get_data_ptr<U>(output);
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx, input_idx = 0;
                for (size_t d = 0; d < ndims; ++d) {
                    int64_t coord = temp / output_strides_arr[d];
                    temp %= output_strides_arr[d];
                    if (static_cast<int64_t>(d) == dim) {
                        int64_t idx_val = index_ptr[coord];
                        if (idx_val < 0) idx_val += input_shape_arr[d];
                        input_idx += idx_val * input_strides_arr[d];
                    } else {
                        input_idx += coord * input_strides_arr[d];
                    }
                }
                for (int64_t l = 0; l < lanes; ++l)
                    output_ptr[flat_idx * lanes + l] = input_ptr[input_idx * lanes + l];
            });
        };
        switch (input.dtype_size()) {
            case 1:  gen.template operator()<uint8_t>(1);  break;
            case 2:  gen.template operator()<uint16_t>(1); break;
            case 4:  gen.template operator()<uint32_t>(1); break;
            case 8:  gen.template operator()<uint64_t>(1); break;
            case 16: gen.template operator()<uint64_t>(2); break;
        }
    }
    else {
        throw std::runtime_error("Unsupported dtype for index_select");
    }

    return output;
}

// Masked fill - fill elements where mask is true with value
auto masked_fill_kernel(const Tensor& input, const Tensor& mask_in, double value, sycl::queue& queue) -> Tensor {
    auto input_shape = input.shape();
    auto mask_shape = mask_in.shape();
    if (!std::equal(input_shape.begin(), input_shape.end(), mask_shape.begin(), mask_shape.end())) {
        throw std::invalid_argument("MaskedFill: input and mask must have same shape");
    }

    // Accept non-Bool masks (e.g. a Float32 attention mask) like the CPU
    // backend: any non-zero element is true. Normalize to Bool once so the
    // branches below (which read `const bool*`) don't reinterpret float/int
    // mask bytes as bool (which silently selects the wrong elements).
    Tensor mask_storage;
    const Tensor& mask = (mask_in.dtype() == DType::Bool)
        ? mask_in
        : (mask_storage = mask_in.to(DType::Bool));

    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = get_data_ptr<const float>(input);
        const bool* mask_ptr = get_data_ptr<const bool>(mask);
        float* output_ptr = get_data_ptr<float>(output);
        const float value_f = static_cast<float>(value);

        queue.parallel_for<MaskedFillKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            output_ptr[idx] = mask_ptr[idx] ? value_f : input_ptr[idx];
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* input_ptr = get_data_ptr<const double>(input);
        const bool* mask_ptr = get_data_ptr<const bool>(mask);
        double* output_ptr = get_data_ptr<double>(output);
        const double value_d = static_cast<double>(value);

        queue.parallel_for<MaskedFillKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            output_ptr[idx] = mask_ptr[idx] ? value_d : input_ptr[idx];
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* input_ptr = get_data_ptr<const sycl::half>(input);
        const bool* mask_ptr = get_data_ptr<const bool>(mask);
        sycl::half* output_ptr = get_data_ptr<sycl::half>(output);
        const sycl::half value_h = static_cast<sycl::half>(value);

        queue.parallel_for<MaskedFillKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            output_ptr[idx] = mask_ptr[idx] ? value_h : input_ptr[idx];
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* input_ptr = get_data_ptr<const uint16_t>(input);
        const bool* mask_ptr = get_data_ptr<const bool>(mask);
        uint16_t* output_ptr = get_data_ptr<uint16_t>(output);
        const uint16_t value_bf16 = f32_to_bf16(static_cast<float>(value));

        queue.parallel_for<MaskedFillKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            output_ptr[idx] = mask_ptr[idx] ? value_bf16 : input_ptr[idx];
        });
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* input_ptr = get_data_ptr<const int32_t>(input);
        const bool* mask_ptr = get_data_ptr<const bool>(mask);
        int32_t* output_ptr = get_data_ptr<int32_t>(output);
        const int32_t value_i = static_cast<int32_t>(value);

        queue.parallel_for<MaskedFillKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            output_ptr[idx] = mask_ptr[idx] ? value_i : input_ptr[idx];
        });
    }
    else if (input.dtype() == DType::Int64) {
        const int64_t* input_ptr = get_data_ptr<const int64_t>(input);
        const bool* mask_ptr = get_data_ptr<const bool>(mask);
        int64_t* output_ptr = get_data_ptr<int64_t>(output);
        const int64_t value_i = static_cast<int64_t>(value);

        queue.parallel_for<MaskedFillKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            output_ptr[idx] = mask_ptr[idx] ? value_i : input_ptr[idx];
        });
    }
    else if (input.dtype() == DType::Bool) {
        const bool* input_ptr = get_data_ptr<const bool>(input);
        const bool* mask_ptr = get_data_ptr<const bool>(mask);
        bool* output_ptr = get_data_ptr<bool>(output);
        const bool value_b = (value != 0.0);

        queue.parallel_for<MaskedFillKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            output_ptr[idx] = mask_ptr[idx] ? value_b : input_ptr[idx];
        });
    }
    else if (input.dtype() == DType::Complex64) {
        // Fill with value + 0i; elements are interleaved (real, imag) floats.
        const float* input_ptr = reinterpret_cast<const float*>(input.data_ptr());
        const bool* mask_ptr = get_data_ptr<const bool>(mask);
        float* output_ptr = reinterpret_cast<float*>(output.data_ptr());
        const float value_f = static_cast<float>(value);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const size_t b = idx[0] * 2;
            output_ptr[b]     = mask_ptr[idx] ? value_f : input_ptr[b];
            output_ptr[b + 1] = mask_ptr[idx] ? 0.0f : input_ptr[b + 1];
        });
    }
    else if (input.dtype() == DType::Complex128) {
        const double* input_ptr = reinterpret_cast<const double*>(input.data_ptr());
        const bool* mask_ptr = get_data_ptr<const bool>(mask);
        double* output_ptr = reinterpret_cast<double*>(output.data_ptr());
        const double value_d = value;
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const size_t b = idx[0] * 2;
            output_ptr[b]     = mask_ptr[idx] ? value_d : input_ptr[b];
            output_ptr[b + 1] = mask_ptr[idx] ? 0.0 : input_ptr[b + 1];
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for masked_fill");
    }

    return output;
}

// sycl_exclusive_prefix_sum is now in ../sycl_prefix_sum.hpp

// Masked select - select elements where mask is true
// Uses device-side prefix sum to avoid host roundtrips
auto masked_select_kernel(const Tensor& input_in, const Tensor& mask_in, sycl::queue& queue) -> Tensor {
    auto input_shape = input_in.shape();
    auto mask_shape = mask_in.shape();
    if (!std::equal(input_shape.begin(), input_shape.end(), mask_shape.begin(), mask_shape.end())) {
        throw std::invalid_argument("MaskedSelect: input and mask must have same shape");
    }

    // The kernel reads input and mask by flat (physical) index, so both must be
    // contiguous. The op layer only contiguifies on the broadcast path; an
    // equal-shape but non-contiguous (e.g. transposed) view would otherwise be
    // read in the wrong physical order.
    Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();

    const int64_t numel = input.numel();
    if (numel == 0) {
        return Tensor({0}, input.dtype(), input.device());
    }

    // Accept non-Bool masks (nonzero = true), matching the CPU backend.
    // .to(DType::Bool) yields a contiguous tensor; an already-Bool mask must be
    // contiguified explicitly since it is read by flat index.
    Tensor mask_storage;
    const Tensor& mask = (mask_in.dtype() == DType::Bool)
        ? (mask_in.is_contiguous() ? mask_in : (mask_storage = mask_in.contiguous()))
        : (mask_storage = mask_in.to(DType::Bool));
    const bool* mask_ptr = get_data_ptr<const bool>(mask);

    // Phase 1: Create int32 mask on device (bool -> 0/1)
    int32_t* d_mask_int = sycl::malloc_device<int32_t>(numel, queue);
    queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
        d_mask_int[i] = mask_ptr[i] ? 1 : 0;
    }).wait();

    // Phase 2: Exclusive prefix sum — returns total count
    // We need a copy for the scatter step (prefix sum modifies in-place)
    int32_t* d_offsets = sycl::malloc_device<int32_t>(numel, queue);
    queue.memcpy(d_offsets, d_mask_int, numel * sizeof(int32_t)).wait();

    int64_t true_count = sycl_exclusive_prefix_sum(d_offsets, numel, queue);

    // Create output
    Tensor output({true_count}, input.dtype(), input.device());

    if (true_count == 0) {
        sycl::free(d_mask_int, queue);
        sycl::free(d_offsets, queue);
        return output;
    }

    // Phase 3: Parallel scatter using prefix-sum offsets
    auto scatter_impl = [&](auto* out_ptr, const auto* in_ptr) {
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            if (d_mask_int[i]) {
                out_ptr[d_offsets[i]] = in_ptr[i];
            }
        }).wait();
    };

    if (input.dtype() == DType::Float32) {
        scatter_impl(get_data_ptr<float>(output), get_data_ptr<const float>(input));
    }
    else if (input.dtype() == DType::Float64) {
        scatter_impl(get_data_ptr<double>(output), get_data_ptr<const double>(input));
    }
    else if (input.dtype() == DType::Float16) {
        scatter_impl(get_data_ptr<sycl::half>(output), get_data_ptr<const sycl::half>(input));
    }
    else if (input.dtype() == DType::BFloat16) {
        scatter_impl(get_data_ptr<uint16_t>(output), get_data_ptr<const uint16_t>(input));
    }
    else {
        // Size-generic scatter for the remaining dtypes (integers, Bool,
        // Complex64/128): the gather is a bit-copy, so only element size
        // matters. Complex64 moves as one 8-byte word, Complex128 as a
        // 16-byte two-word struct.
        const size_t esz = dtype_size(input.dtype());
        if (esz == 1) {
            scatter_impl(reinterpret_cast<uint8_t*>(output.data_ptr()),
                         reinterpret_cast<const uint8_t*>(input.data_ptr()));
        } else if (esz == 2) {
            scatter_impl(reinterpret_cast<uint16_t*>(output.data_ptr()),
                         reinterpret_cast<const uint16_t*>(input.data_ptr()));
        } else if (esz == 4) {
            scatter_impl(reinterpret_cast<uint32_t*>(output.data_ptr()),
                         reinterpret_cast<const uint32_t*>(input.data_ptr()));
        } else if (esz == 8) {
            scatter_impl(reinterpret_cast<uint64_t*>(output.data_ptr()),
                         reinterpret_cast<const uint64_t*>(input.data_ptr()));
        } else if (esz == 16) {
            struct alignas(16) W16 { uint64_t lo, hi; };
            scatter_impl(reinterpret_cast<W16*>(output.data_ptr()),
                         reinterpret_cast<const W16*>(input.data_ptr()));
        } else {
            sycl::free(d_mask_int, queue);
            sycl::free(d_offsets, queue);
            throw std::runtime_error("Unsupported dtype for masked_select");
        }
    }

    sycl::free(d_mask_int, queue);
    sycl::free(d_offsets, queue);
    return output;
}

// Nonzero operation - find indices of non-zero elements
// Returns shape (num_nonzero, ndim) with Int64 dtype
// Uses device-side prefix sum to avoid host roundtrips
auto nonzero_kernel(const Tensor& input_in, sycl::queue& queue) -> Tensor {
    // The kernel reads input by flat (physical) index and decodes coordinates
    // with strides derived from shape (contiguous assumption), so a
    // non-contiguous (e.g. transposed) view must be made contiguous first.
    Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();

    const int64_t numel = input.numel();
    auto input_shape_span = input.shape();
    std::vector<int64_t> input_shape(input_shape_span.begin(), input_shape_span.end());
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    if (numel == 0) {
        return Tensor({0, ndim}, DType::Int64, input.device());
    }

    // Phase 1: Create binary mask on device (nonzero -> 1, zero -> 0)
    int32_t* d_mask = sycl::malloc_device<int32_t>(numel, queue);

    auto create_mask = [&](const auto* ptr, auto zero_val) {
        using ValT = std::remove_const_t<std::remove_pointer_t<decltype(ptr)>>;
        auto zv = zero_val;
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            d_mask[i] = (ptr[i] != zv) ? 1 : 0;
        }).wait();
    };

    // BFloat16 needs special comparison
    auto create_mask_bf16 = [&](const uint16_t* ptr) {
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            // BFloat16: zero is 0x0000
            d_mask[i] = (ptr[i] != 0) ? 1 : 0;
        }).wait();
    };

    // Float16 needs cast for zero comparison
    auto create_mask_f16 = [&](const sycl::half* ptr) {
        sycl::half zero_h{0.0f};
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            d_mask[i] = (ptr[i] != zero_h) ? 1 : 0;
        }).wait();
    };

    if (input.dtype() == DType::Float32) {
        create_mask(get_data_ptr<const float>(input), 0.0f);
    } else if (input.dtype() == DType::Float64) {
        create_mask(get_data_ptr<const double>(input), 0.0);
    } else if (input.dtype() == DType::Float16) {
        create_mask_f16(get_data_ptr<const sycl::half>(input));
    } else if (input.dtype() == DType::BFloat16) {
        create_mask_bf16(get_data_ptr<const uint16_t>(input));
    } else if (input.dtype() == DType::Int32) {
        create_mask(get_data_ptr<const int32_t>(input), int32_t{0});
    } else if (input.dtype() == DType::Int64) {
        create_mask(get_data_ptr<const int64_t>(input), int64_t{0});
    } else if (input.dtype() == DType::Bool) {
        const uint8_t* bool_ptr = reinterpret_cast<const uint8_t*>(get_data_ptr<const bool>(input));
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            d_mask[i] = bool_ptr[i] ? 1 : 0;
        }).wait();
    } else {
        sycl::free(d_mask, queue);
        throw std::runtime_error("nonzero: unsupported dtype");
    }

    // Phase 2: Exclusive prefix sum over mask — returns total nonzero count
    int32_t* d_offsets = sycl::malloc_device<int32_t>(numel, queue);
    queue.memcpy(d_offsets, d_mask, numel * sizeof(int32_t)).wait();

    int64_t nonzero_count = sycl_exclusive_prefix_sum(d_offsets, numel, queue);

    // Create output tensor of shape (nonzero_count, ndim)
    Tensor output({nonzero_count, ndim}, DType::Int64, input.device());

    if (nonzero_count == 0) {
        sycl::free(d_mask, queue);
        sycl::free(d_offsets, queue);
        return output;
    }

    // Phase 3: Parallel kernel computes multi-dimensional indices using strides
    // Copy strides to device
    int64_t* d_strides = sycl::malloc_device<int64_t>(ndim, queue);
    auto strides = calculate_strides(input_shape);
    queue.memcpy(d_strides, strides.data(), ndim * sizeof(int64_t)).wait();

    int64_t* output_ptr = get_data_ptr<int64_t>(output);

    queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        int64_t i = gid;
        if (d_mask[i]) {
            int64_t out_row = d_offsets[i];
            int64_t flat = i;
            for (int64_t d = 0; d < ndim; ++d) {
                output_ptr[out_row * ndim + d] = flat / d_strides[d];
                flat %= d_strides[d];
            }
        }
    }).wait();

    sycl::free(d_mask, queue);
    sycl::free(d_offsets, queue);
    sycl::free(d_strides, queue);

    return output;
}

// One-hot encoding operation
// Input: Int64 indices tensor of shape (N,)
// Output: Float tensor of shape (N, num_classes)
class OneHotKernel {};

auto one_hot_kernel(const Tensor& indices, int64_t num_classes, DType output_dtype,
                    sycl::queue& queue) -> Tensor {
    auto indices_shape = indices.shape();
    int64_t numel = indices.numel();

    // Output shape: indices_shape + [num_classes]
    std::vector<int64_t> output_shape(indices_shape.begin(), indices_shape.end());
    output_shape.push_back(num_classes);

    Tensor output(output_shape, output_dtype, indices.device());
    int64_t total = numel * num_classes;

    // Zero-initialize
    if (output_dtype == DType::Float32) {
        float* out_ptr = get_data_ptr<float>(output);
        queue.fill(out_ptr, 0.0f, total);

        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.parallel_for<OneHotKernel>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            int64_t cls = idx_ptr[i];
            if (cls >= 0 && cls < num_classes) {
                out_ptr[i * num_classes + cls] = 1.0f;
            }
        });
    }
    else if (output_dtype == DType::Float64) {
        double* out_ptr = get_data_ptr<double>(output);
        queue.fill(out_ptr, 0.0, total);

        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.parallel_for<class OneHotKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            int64_t cls = idx_ptr[i];
            if (cls >= 0 && cls < num_classes) {
                out_ptr[i * num_classes + cls] = 1.0;
            }
        });
    }
    else if (output_dtype == DType::Float16) {
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.fill(out_ptr, sycl::half(0.0f), total);

        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.parallel_for<class OneHotKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            int64_t cls = idx_ptr[i];
            if (cls >= 0 && cls < num_classes) {
                out_ptr[i * num_classes + cls] = sycl::half(1.0f);
            }
        });
    }
    else if (output_dtype == DType::BFloat16) {
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        const uint16_t zero_bf16 = f32_to_bf16(0.0f);
        queue.fill(out_ptr, zero_bf16, total);

        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);
        const uint16_t one_bf16 = f32_to_bf16(1.0f);

        queue.parallel_for<class OneHotKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            int64_t cls = idx_ptr[i];
            if (cls >= 0 && cls < num_classes) {
                out_ptr[i * num_classes + cls] = one_bf16;
            }
        });
    }
    else {
        throw std::runtime_error("one_hot: unsupported output dtype");
    }

    return output;
}

// ============================================================================
// Argsort
// ============================================================================

/**
 * @brief Returns indices that would sort a tensor along a given dimension.
 *
 * For each 1D slice along the specified dimension, computes the indices
 * that would sort the elements.
 *
 * @param input Input tensor of any shape
 * @param dim Dimension along which to sort (negative indexing supported)
 * @param descending If true, sort in descending order
 * @param queue SYCL queue for execution
 * @return Int64 tensor of same shape with sorted indices
 */
auto argsort_kernel(const Tensor& input, int64_t dim, bool descending, sycl::queue& queue) -> Tensor {
    const int64_t ndim = input.ndim();

    // Normalize dimension
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("argsort: dimension out of range");
    }

    auto input_shape = input.shape();
    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());

    // Output has same shape as input but with Int64 dtype
    Tensor output(shape_vec, DType::Int64, input.device());

    const int64_t dim_size = shape_vec[dim];

    if (dim_size == 0) {
        return output;
    }

    // Compute inner_size to decide contiguous vs transpose path
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= shape_vec[d];
    }

#ifdef TENZOR_HAS_ONEDPL
    // Device-side argsort for contiguous sort dimension (inner_size == 1)
    if (inner_size == 1) {
        int64_t total_elems = input.numel();
        int64_t outer_size = total_elems / (dim_size * inner_size);
        auto policy = ::oneapi::dpl::execution::make_device_policy(queue);

        auto device_argsort_impl = [&](const auto* in_ptr) {
            using T = std::remove_const_t<std::remove_pointer_t<decltype(in_ptr)>>;
            int64_t* idx_ptr = get_data_ptr<int64_t>(output);

            // Allocate temp device buffer for values (we sort values+indices together)
            T* tmp_vals = sycl::malloc_device<T>(total_elems, queue);
            queue.memcpy(tmp_vals, in_ptr, total_elems * sizeof(T)).wait();

            // Initialize indices: each slice gets 0..dim_size-1
            queue.parallel_for(sycl::range<1>(total_elems), [=](sycl::id<1> gid) {
                idx_ptr[gid] = static_cast<int64_t>(gid[0]) % dim_size;
            }).wait();

            // Sort each contiguous slice using oneDPL sort_by_key
            for (int64_t o = 0; o < outer_size; ++o) {
                T* slice_vals = tmp_vals + o * dim_size;
                int64_t* slice_idx = idx_ptr + o * dim_size;
                if (descending) {
                    ::oneapi::dpl::sort_by_key(policy, slice_vals, slice_vals + dim_size,
                                              slice_idx, std::greater<T>());
                } else {
                    ::oneapi::dpl::sort_by_key(policy, slice_vals, slice_vals + dim_size,
                                              slice_idx);
                }
            }

            sycl::free(tmp_vals, queue);
        };

        if (input.dtype() == DType::Float32) {
            device_argsort_impl(get_data_ptr<const float>(input));
        } else if (input.dtype() == DType::Float64) {
            device_argsort_impl(get_data_ptr<const double>(input));
        } else if (input.dtype() == DType::Int32) {
            device_argsort_impl(get_data_ptr<const int32_t>(input));
        } else if (input.dtype() == DType::Int64) {
            device_argsort_impl(get_data_ptr<const int64_t>(input));
        } else if (input.dtype() == DType::Float16) {
            // Upcast to Float32 for sorting
            Tensor input_f32 = input.to(DType::Float32);
            device_argsort_impl(get_data_ptr<const float>(input_f32));
        } else if (input.dtype() == DType::BFloat16) {
            Tensor input_f32 = input.to(DType::Float32);
            device_argsort_impl(get_data_ptr<const float>(input_f32));
        } else {
            throw std::runtime_error("argsort: unsupported dtype");
        }

        return output;
    }
    // Fall through: transpose so sort dim is last, argsort on device, transpose back
    {
        std::vector<int64_t> perm(ndim);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[dim], perm[ndim - 1]);

        std::vector<int64_t> inv_perm(ndim);
        for (int64_t i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;

        Tensor transposed = input.permute(perm).contiguous();
        output = argsort_kernel(transposed, ndim - 1, descending, queue);
        output = output.permute(inv_perm).contiguous();
    }

    return output;
#else
    // Without oneDPL, no device-side argsort is available
    throw std::runtime_error("argsort: oneDPL required for device-side argsort");
#endif
}

// ============================================================================
// ScatterAdd kernel - scatter with addition
// ============================================================================
class ScatterAddKernelFloat32;
class ScatterAddKernelFloat64;
class ScatterAddKernelFloat16;
class ScatterAddKernelFloat16Convert;
class ScatterAddKernelBFloat16;
class ScatterAddKernelBFloat16Convert;
class ScatterAddKernelInt32;
class ScatterAddKernelInt64;

auto scatter_add_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src,
                        sycl::queue& queue) -> Tensor {
    // Force operands contiguous: this kernel computes strides purely from
    // shape() and indexes data_ptr() by those strides, so a non-contiguous
    // self/index/src (transposed view, slice, or upstream grad from
    // gather/sort/take_along_dim/FFT backward) would otherwise read the wrong
    // elements. Mirror the sibling indexing kernels (gather/scatter/
    // index_select), which all force contiguity first.
    Tensor self_c  = self.is_contiguous()  ? self  : self.contiguous();
    Tensor index_c = index.is_contiguous() ? index : index.contiguous();
    Tensor src_c   = src.is_contiguous()   ? src   : src.contiguous();

    auto self_shape_span = self_c.shape();
    std::vector<int64_t> shape(self_shape_span.begin(), self_shape_span.end());
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    // Clone self as output (scatter_add modifies in-place semantically)
    Tensor output(shape, self_c.dtype(), self_c.device());
    queue.memcpy(const_cast<void*>(output.data_ptr()), self_c.data_ptr(),
                 self_c.numel() * self_c.dtype_size());

    // Host-side implementation for atomicity correctness
    int64_t idx_numel = index_c.numel();
    if (idx_numel == 0) return output;

    auto idx_shape = index_c.shape();
    std::vector<int64_t> idx_shape_vec(idx_shape.begin(), idx_shape.end());

    // Compute strides for self/output
    std::vector<int64_t> out_strides(ndim);
    { int64_t s = 1; for (int64_t i = ndim - 1; i >= 0; --i) { out_strides[i] = s; s *= shape[i]; } }
    std::vector<int64_t> idx_strides(ndim);
    { int64_t s = 1; for (int64_t i = ndim - 1; i >= 0; --i) { idx_strides[i] = s; s *= idx_shape_vec[i]; } }

    if (self_c.dtype() == DType::Float32) {
        float* out_ptr = get_data_ptr<float>(output);
        const float* src_ptr = get_data_ptr<const float>(src_c);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(index_c);

        std::array<int64_t, 8> d_out_strides{}, d_idx_strides{};
        for (int64_t d = 0; d < ndim; ++d) {
            d_out_strides[d] = out_strides[d];
            d_idx_strides[d] = idx_strides[d];
        }

        queue.parallel_for(sycl::range<1>(idx_numel), [=](sycl::id<1> id) {
            int64_t flat = id[0];
            int64_t remaining = flat;
            int64_t out_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                if (d == dim) {
                    out_offset += idx_ptr[flat] * d_out_strides[d];
                } else {
                    out_offset += coord * d_out_strides[d];
                }
            }
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                atomic_out(out_ptr[out_offset]);
            atomic_out.fetch_add(src_ptr[flat]);
        }).wait();
    } else if (self_c.dtype() == DType::Float64) {
        double* out_ptr = get_data_ptr<double>(output);
        const double* src_ptr = get_data_ptr<const double>(src_c);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(index_c);

        std::array<int64_t, 8> d_out_strides{}, d_idx_strides{};
        for (int64_t d = 0; d < ndim; ++d) {
            d_out_strides[d] = out_strides[d];
            d_idx_strides[d] = idx_strides[d];
        }

        queue.parallel_for(sycl::range<1>(idx_numel), [=](sycl::id<1> id) {
            int64_t flat = id[0];
            int64_t remaining = flat;
            int64_t out_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                if (d == dim) {
                    out_offset += idx_ptr[flat] * d_out_strides[d];
                } else {
                    out_offset += coord * d_out_strides[d];
                }
            }
            sycl::atomic_ref<double, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                atomic_out(out_ptr[out_offset]);
            atomic_out.fetch_add(src_ptr[flat]);
        }).wait();
    } else if (self_c.dtype() == DType::Int32) {
        int32_t* out_ptr = get_data_ptr<int32_t>(output);
        const int32_t* src_ptr = get_data_ptr<const int32_t>(src_c);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(index_c);

        std::array<int64_t, 8> d_out_strides{}, d_idx_strides{};
        for (int64_t d = 0; d < ndim; ++d) {
            d_out_strides[d] = out_strides[d];
            d_idx_strides[d] = idx_strides[d];
        }

        queue.parallel_for(sycl::range<1>(idx_numel), [=](sycl::id<1> id) {
            int64_t flat = id[0];
            int64_t remaining = flat;
            int64_t out_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                if (d == dim) {
                    out_offset += idx_ptr[flat] * d_out_strides[d];
                } else {
                    out_offset += coord * d_out_strides[d];
                }
            }
            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                atomic_out(out_ptr[out_offset]);
            atomic_out.fetch_add(src_ptr[flat]);
        }).wait();
    } else if (self_c.dtype() == DType::Int64) {
        int64_t* out_ptr = get_data_ptr<int64_t>(output);
        const int64_t* src_ptr = get_data_ptr<const int64_t>(src_c);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(index_c);

        std::array<int64_t, 8> d_out_strides{}, d_idx_strides{};
        for (int64_t d = 0; d < ndim; ++d) {
            d_out_strides[d] = out_strides[d];
            d_idx_strides[d] = idx_strides[d];
        }

        queue.parallel_for(sycl::range<1>(idx_numel), [=](sycl::id<1> id) {
            int64_t flat = id[0];
            int64_t remaining = flat;
            int64_t out_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                if (d == dim) {
                    out_offset += idx_ptr[flat] * d_out_strides[d];
                } else {
                    out_offset += coord * d_out_strides[d];
                }
            }
            sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                atomic_out(out_ptr[out_offset]);
            atomic_out.fetch_add(src_ptr[flat]);
        }).wait();
    } else if (self_c.dtype() == DType::UInt32 || self_c.dtype() == DType::UInt64) {
        std::array<int64_t, 8> d_out_strides{}, d_idx_strides{};
        for (int64_t d = 0; d < ndim; ++d) { d_out_strides[d] = out_strides[d]; d_idx_strides[d] = idx_strides[d]; }
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(index_c);
        const int64_t nd = ndim, dm = dim, inum = idx_numel;
        auto run_u = [&]<typename T>() {
            T* out_ptr = get_data_ptr<T>(output);
            const T* src_ptr = get_data_ptr<const T>(src_c);
            queue.parallel_for(sycl::range<1>(inum), [=](sycl::id<1> id) {
                int64_t flat = id[0], remaining = flat, out_offset = 0;
                for (int64_t d = 0; d < nd; ++d) {
                    int64_t coord = remaining / d_idx_strides[d];
                    remaining %= d_idx_strides[d];
                    out_offset += (d == dm) ? idx_ptr[flat] * d_out_strides[d] : coord * d_out_strides[d];
                }
                sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> atomic_out(out_ptr[out_offset]);
                atomic_out.fetch_add(src_ptr[flat]);
            }).wait();
        };
        if (self_c.dtype() == DType::UInt32) run_u.template operator()<uint32_t>();
        else run_u.template operator()<uint64_t>();
    } else if (self_c.dtype() == DType::Int16 || self_c.dtype() == DType::UInt16) {
        // 16-bit lacks atomic_ref support; accumulate in a 32-bit device buffer.
        const int64_t total = self_c.numel();
        int32_t* acc = sycl::malloc_device<int32_t>(total, queue);
        std::array<int64_t, 8> d_out_strides{}, d_idx_strides{};
        for (int64_t d = 0; d < ndim; ++d) { d_out_strides[d] = out_strides[d]; d_idx_strides[d] = idx_strides[d]; }
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(index_c);
        const int64_t nd = ndim, dm = dim, inum = idx_numel;
        auto run16 = [&]<typename T>() {
            T* out_ptr = get_data_ptr<T>(output);
            const T* src_ptr = get_data_ptr<const T>(src_c);
            queue.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i) { acc[i] = static_cast<int32_t>(out_ptr[i]); }).wait();
            queue.parallel_for(sycl::range<1>(inum), [=](sycl::id<1> id) {
                int64_t flat = id[0], remaining = flat, out_offset = 0;
                for (int64_t d = 0; d < nd; ++d) {
                    int64_t coord = remaining / d_idx_strides[d];
                    remaining %= d_idx_strides[d];
                    out_offset += (d == dm) ? idx_ptr[flat] * d_out_strides[d] : coord * d_out_strides[d];
                }
                sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> atomic_out(acc[out_offset]);
                atomic_out.fetch_add(static_cast<int32_t>(src_ptr[flat]));
            }).wait();
            queue.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i) { out_ptr[i] = static_cast<T>(acc[i]); }).wait();
        };
        if (self_c.dtype() == DType::Int16) run16.template operator()<int16_t>();
        else run16.template operator()<uint16_t>();
        sycl::free(acc, queue);
    } else if (self_c.dtype() == DType::Float16) {
        // Float16: use float32 accumulator since atomic_ref<half> not widely supported
        float* acc = sycl::malloc_device<float>(self_c.numel(), queue);
        const sycl::half* self_ptr = get_data_ptr<const sycl::half>(output);
        // Convert output to float32
        queue.parallel_for(sycl::range<1>(self_c.numel()), [=](sycl::id<1> i) {
            acc[i] = static_cast<float>(self_ptr[i]);
        }).wait();

        const sycl::half* src_ptr = get_data_ptr<const sycl::half>(src_c);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(index_c);
        std::array<int64_t, 8> d_out_strides{}, d_idx_strides{};
        for (int64_t d = 0; d < ndim; ++d) {
            d_out_strides[d] = out_strides[d];
            d_idx_strides[d] = idx_strides[d];
        }

        queue.parallel_for<ScatterAddKernelFloat16>(sycl::range<1>(idx_numel), [=](sycl::id<1> id) {
            int64_t flat = id[0];
            int64_t remaining = flat;
            int64_t out_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                if (d == dim) {
                    out_offset += idx_ptr[flat] * d_out_strides[d];
                } else {
                    out_offset += coord * d_out_strides[d];
                }
            }
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                atomic_out(acc[out_offset]);
            atomic_out.fetch_add(static_cast<float>(src_ptr[flat]));
        }).wait();

        // Convert back to half
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<ScatterAddKernelFloat16Convert>(sycl::range<1>(self_c.numel()), [=](sycl::id<1> i) {
            out_ptr[i] = sycl::half(acc[i]);
        }).wait();
        sycl::free(acc, queue);
    } else if (self_c.dtype() == DType::BFloat16) {
        float* acc = sycl::malloc_device<float>(self_c.numel(), queue);
        const uint16_t* self_ptr = get_data_ptr<const uint16_t>(output);
        queue.parallel_for(sycl::range<1>(self_c.numel()), [=](sycl::id<1> i) {
            acc[i] = bf16_to_f32(self_ptr[i]);
        }).wait();

        const uint16_t* src_ptr = get_data_ptr<const uint16_t>(src_c);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(index_c);
        std::array<int64_t, 8> d_out_strides{}, d_idx_strides{};
        for (int64_t d = 0; d < ndim; ++d) {
            d_out_strides[d] = out_strides[d];
            d_idx_strides[d] = idx_strides[d];
        }

        queue.parallel_for<ScatterAddKernelBFloat16>(sycl::range<1>(idx_numel), [=](sycl::id<1> id) {
            int64_t flat = id[0];
            int64_t remaining = flat;
            int64_t out_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                if (d == dim) {
                    out_offset += idx_ptr[flat] * d_out_strides[d];
                } else {
                    out_offset += coord * d_out_strides[d];
                }
            }
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                atomic_out(acc[out_offset]);
            atomic_out.fetch_add(bf16_to_f32(src_ptr[flat]));
        }).wait();

        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<ScatterAddKernelBFloat16Convert>(sycl::range<1>(self_c.numel()), [=](sycl::id<1> i) {
            out_ptr[i] = f32_to_bf16(acc[i]);
        }).wait();
        sycl::free(acc, queue);
    } else {
        throw std::runtime_error("scatter_add: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Put Operation
// ============================================================================
// Kernel name classes
class PutKernelFloat32;
class PutKernelFloat64;
class PutKernelInt32;
class PutKernelInt64;
class PutKernelFloat32Acc;
class PutKernelFloat64Acc;
class PutKernelInt32Acc;
class PutKernelInt64Acc;

auto put_kernel(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& source,
    bool accumulate,
    sycl::queue& queue
) -> Tensor {
    Tensor output = input.clone();

    int64_t num_indices = indices.numel();
    int64_t total_size = input.numel();

    if (num_indices == 0) return output;

    if (accumulate) {
        // Accumulate mode: use device-side parallel_for with atomic operations
        if (input.dtype() == DType::Float32) {
            float* out_ptr = get_data_ptr<float>(output);
            const float* src_ptr = get_data_ptr<const float>(source);
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

            queue.parallel_for<PutKernelFloat32Acc>(sycl::range<1>(num_indices), [=](sycl::id<1> id) {
                int64_t i = id[0];
                int64_t target_idx = idx_ptr[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                        atomic_out(out_ptr[target_idx]);
                    atomic_out.fetch_add(src_ptr[i]);
                }
            }).wait();
        } else if (input.dtype() == DType::Float64) {
            double* out_ptr = get_data_ptr<double>(output);
            const double* src_ptr = get_data_ptr<const double>(source);
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

            queue.parallel_for<PutKernelFloat64Acc>(sycl::range<1>(num_indices), [=](sycl::id<1> id) {
                int64_t i = id[0];
                int64_t target_idx = idx_ptr[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                        atomic_out(out_ptr[target_idx]);
                    atomic_out.fetch_add(src_ptr[i]);
                }
            }).wait();
        } else if (input.dtype() == DType::Int32) {
            int32_t* out_ptr = get_data_ptr<int32_t>(output);
            const int32_t* src_ptr = get_data_ptr<const int32_t>(source);
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

            queue.parallel_for<PutKernelInt32Acc>(sycl::range<1>(num_indices), [=](sycl::id<1> id) {
                int64_t i = id[0];
                int64_t target_idx = idx_ptr[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                        atomic_out(out_ptr[target_idx]);
                    atomic_out.fetch_add(src_ptr[i]);
                }
            }).wait();
        } else if (input.dtype() == DType::Int64) {
            int64_t* out_ptr = get_data_ptr<int64_t>(output);
            const int64_t* src_ptr = get_data_ptr<const int64_t>(source);
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

            queue.parallel_for<PutKernelInt64Acc>(sycl::range<1>(num_indices), [=](sycl::id<1> id) {
                int64_t i = id[0];
                int64_t target_idx = idx_ptr[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                        atomic_out(out_ptr[target_idx]);
                    atomic_out.fetch_add(src_ptr[i]);
                }
            }).wait();
        } else {
            throw std::runtime_error("put_kernel: unsupported dtype for accumulate mode");
        }
    } else {
        // Non-accumulate mode: safe to use parallel_for (last write wins semantics)
        if (input.dtype() == DType::Float32) {
            float* out_ptr = get_data_ptr<float>(output);
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);
            const float* src_ptr = get_data_ptr<const float>(source);

            queue.parallel_for<PutKernelFloat32>(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                int64_t target_idx = idx_ptr[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    out_ptr[target_idx] = src_ptr[i];
                }
            });
        } else if (input.dtype() == DType::Float64) {
            double* out_ptr = get_data_ptr<double>(output);
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);
            const double* src_ptr = get_data_ptr<const double>(source);

            queue.parallel_for<PutKernelFloat64>(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                int64_t target_idx = idx_ptr[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    out_ptr[target_idx] = src_ptr[i];
                }
            });
        } else if (input.dtype() == DType::Int32) {
            int32_t* out_ptr = get_data_ptr<int32_t>(output);
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);
            const int32_t* src_ptr = get_data_ptr<const int32_t>(source);

            queue.parallel_for<PutKernelInt32>(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                int64_t target_idx = idx_ptr[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    out_ptr[target_idx] = src_ptr[i];
                }
            });
        } else if (input.dtype() == DType::Int64) {
            int64_t* out_ptr = get_data_ptr<int64_t>(output);
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);
            const int64_t* src_ptr = get_data_ptr<const int64_t>(source);

            queue.parallel_for<PutKernelInt64>(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                int64_t target_idx = idx_ptr[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    out_ptr[target_idx] = src_ptr[i];
                }
            });
        } else {
            throw std::runtime_error("put_kernel: unsupported dtype");
        }
    }

    return output;
}

// ============================================================================
// SearchSorted: binary search per element in a sorted 1-D sequence
// ============================================================================

class SearchSortedKernelFloat32;
class SearchSortedKernelFloat64;
class SearchSortedKernelInt32;
class SearchSortedKernelInt64;

auto searchsorted_kernel(const Tensor& sorted_sequence, const Tensor& values,
                          bool right, sycl::queue& queue) -> Tensor {
    if (sorted_sequence.ndim() != 1) {
        throw std::runtime_error("searchsorted: sorted_sequence must be 1-D");
    }

    Tensor seq_cont = sorted_sequence.contiguous();
    Tensor val_cont = values.contiguous();
    int64_t seq_len = seq_cont.shape()[0];
    int64_t num_values = val_cont.numel();

    Tensor result(std::vector<int64_t>(values.shape().begin(), values.shape().end()),
                  DType::Int64, values.device());

    if (num_values == 0) return result;

    int64_t* out_ptr = get_data_ptr<int64_t>(result);

    auto launch_search = [&]<typename T, typename KernelName>(const T* seq_ptr, const T* val_ptr) {
        queue.parallel_for<KernelName>(sycl::range<1>(num_values), [=](sycl::id<1> i) {
            T v = val_ptr[i];
            int64_t lo = 0, hi = seq_len;
            while (lo < hi) {
                int64_t mid = lo + (hi - lo) / 2;
                bool go_right = right ? (seq_ptr[mid] <= v) : (seq_ptr[mid] < v);
                if (go_right) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            out_ptr[i] = lo;
        });
        queue.wait_and_throw();
    };

    switch (sorted_sequence.dtype()) {
        case DType::Float32:
            launch_search.template operator()<float, SearchSortedKernelFloat32>(
                get_data_ptr<const float>(seq_cont), get_data_ptr<const float>(val_cont));
            break;
        case DType::Float64:
            launch_search.template operator()<double, SearchSortedKernelFloat64>(
                get_data_ptr<const double>(seq_cont), get_data_ptr<const double>(val_cont));
            break;
        case DType::Int32:
            launch_search.template operator()<int32_t, SearchSortedKernelInt32>(
                get_data_ptr<const int32_t>(seq_cont), get_data_ptr<const int32_t>(val_cont));
            break;
        case DType::Int64:
            launch_search.template operator()<int64_t, SearchSortedKernelInt64>(
                get_data_ptr<const int64_t>(seq_cont), get_data_ptr<const int64_t>(val_cont));
            break;
        case DType::Float16:
        case DType::BFloat16: {
            auto seq_f32 = sorted_sequence.to(DType::Float32).contiguous();
            auto val_f32 = values.to(DType::Float32).contiguous();
            launch_search.template operator()<float, SearchSortedKernelFloat32>(
                get_data_ptr<const float>(seq_f32), get_data_ptr<const float>(val_f32));
            break;
        }
        default:
            throw std::runtime_error("searchsorted: unsupported dtype " +
                                     std::string(dtype_name(sorted_sequence.dtype())));
    }

    return result;
}

// ============================================================================
// IndexAdd - atomically adds source into output at indexed positions
// ============================================================================
auto index_add_kernel(const Tensor& input, int64_t dim, const Tensor& index,
                      const Tensor& source, sycl::queue& queue) -> Tensor {
    // Non-Float32: convert to Float32 and recurse
    if (input.dtype() != DType::Float32) {
        auto f32_in = input.to(DType::Float32);
        auto f32_src = source.to(DType::Float32);
        auto r = index_add_kernel(f32_in, dim, index, f32_src, queue);
        return r.to(input.dtype());
    }

    auto output = input.clone();
    int64_t ndim = output.shape().size();
    if (dim < 0) dim += ndim;
    auto shape = output.shape();
    int64_t dim_size = shape[dim];
    int64_t idx_n = index.numel();

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    int64_t total = outer * idx_n * inner;
    if (total == 0) return output;

    float* out_ptr = get_data_ptr<float>(output);
    const float* src_ptr = get_data_ptr<const float>(source);
    const int64_t* idx_ptr = get_data_ptr<const int64_t>(index);

    // Validate indices host-side (normalize negatives, throw on out-of-range)
    // matching the CPU reference, then normalize again inside the kernel.
    if (idx_n > 0) {
        std::vector<int64_t> host_idx(static_cast<size_t>(idx_n));
        queue.memcpy(host_idx.data(), idx_ptr,
                     static_cast<size_t>(idx_n) * sizeof(int64_t)).wait();
        for (int64_t i = 0; i < idx_n; ++i) {
            int64_t di = host_idx[i];
            if (di < 0) di += dim_size;
            if (di < 0 || di >= dim_size) {
                throw std::out_of_range("index_add: index " + std::to_string(host_idx[i]) +
                    " out of range [0, " + std::to_string(dim_size) + ")");
            }
        }
    }

    queue.parallel_for<IndexAddKernelF32>(sycl::range<1>(total), [=](sycl::id<1> tid) {
        int64_t id = static_cast<int64_t>(tid);
        int64_t o = id / (idx_n * inner);
        int64_t k = (id / inner) % idx_n;
        int64_t j = id % inner;
        int64_t di = idx_ptr[k];
        if (di < 0) di += dim_size;
        int64_t dst_offset = (o * dim_size + di) * inner + j;
        int64_t src_offset = (o * idx_n + k) * inner + j;
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space> ref(out_ptr[dst_offset]);
        ref.fetch_add(src_ptr[src_offset]);
    }).wait();

    return output;
}

// ============================================================================
// IndexCopy - copies source into output at indexed positions
// ============================================================================
auto index_copy_kernel(const Tensor& input, int64_t dim, const Tensor& index,
                       const Tensor& source, sycl::queue& queue) -> Tensor {
    // Non-Float32: convert to Float32 and recurse
    if (input.dtype() != DType::Float32) {
        auto f32_in = input.to(DType::Float32);
        auto f32_src = source.to(DType::Float32);
        auto r = index_copy_kernel(f32_in, dim, index, f32_src, queue);
        return r.to(input.dtype());
    }

    auto output = input.clone();
    int64_t ndim = output.shape().size();
    if (dim < 0) dim += ndim;
    auto shape = output.shape();
    int64_t dim_size = shape[dim];
    int64_t idx_n = index.numel();

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    int64_t total = outer * idx_n * inner;
    if (total == 0) return output;

    float* out_ptr = get_data_ptr<float>(output);
    const float* src_ptr = get_data_ptr<const float>(source);
    const int64_t* idx_ptr = get_data_ptr<const int64_t>(index);

    // Validate indices host-side (normalize negatives, throw on out-of-range)
    // matching the CPU reference, then normalize again inside the kernel.
    if (idx_n > 0) {
        std::vector<int64_t> host_idx(static_cast<size_t>(idx_n));
        queue.memcpy(host_idx.data(), idx_ptr,
                     static_cast<size_t>(idx_n) * sizeof(int64_t)).wait();
        for (int64_t i = 0; i < idx_n; ++i) {
            int64_t di = host_idx[i];
            if (di < 0) di += dim_size;
            if (di < 0 || di >= dim_size) {
                throw std::out_of_range("index_copy: index " + std::to_string(host_idx[i]) +
                    " out of range [0, " + std::to_string(dim_size) + ")");
            }
        }
    }

    queue.parallel_for<IndexCopyKernelF32>(sycl::range<1>(total), [=](sycl::id<1> tid) {
        int64_t id = static_cast<int64_t>(tid);
        int64_t o = id / (idx_n * inner);
        int64_t k = (id / inner) % idx_n;
        int64_t j = id % inner;
        int64_t di = idx_ptr[k];
        if (di < 0) di += dim_size;
        out_ptr[(o * dim_size + di) * inner + j] =
            src_ptr[(o * idx_n + k) * inner + j];
    }).wait();

    return output;
}

// ============================================================================
// IndexFill - fills output at indexed positions with a scalar value
// ============================================================================
auto index_fill_kernel(const Tensor& input, int64_t dim, const Tensor& index,
                       float value, sycl::queue& queue) -> Tensor {
    // Non-Float32: convert to Float32 and recurse
    if (input.dtype() != DType::Float32) {
        auto f32_in = input.to(DType::Float32);
        auto r = index_fill_kernel(f32_in, dim, index, value, queue);
        return r.to(input.dtype());
    }

    auto output = input.clone();
    int64_t ndim = output.shape().size();
    if (dim < 0) dim += ndim;
    auto shape = output.shape();
    int64_t dim_size = shape[dim];
    int64_t idx_n = index.numel();

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    int64_t total = outer * idx_n * inner;
    if (total == 0) return output;

    float* out_ptr = get_data_ptr<float>(output);
    const int64_t* idx_ptr = get_data_ptr<const int64_t>(index);
    float fill_val = value;

    // Validate indices host-side (normalize negatives, throw on out-of-range)
    // matching the CPU reference, then normalize again inside the kernel.
    if (idx_n > 0) {
        std::vector<int64_t> host_idx(static_cast<size_t>(idx_n));
        queue.memcpy(host_idx.data(), idx_ptr,
                     static_cast<size_t>(idx_n) * sizeof(int64_t)).wait();
        for (int64_t i = 0; i < idx_n; ++i) {
            int64_t di = host_idx[i];
            if (di < 0) di += dim_size;
            if (di < 0 || di >= dim_size) {
                throw std::out_of_range("index_fill: index " + std::to_string(host_idx[i]) +
                    " out of range for dim of size " + std::to_string(dim_size));
            }
        }
    }

    queue.parallel_for<IndexFillKernelF32>(sycl::range<1>(total), [=](sycl::id<1> tid) {
        int64_t id = static_cast<int64_t>(tid);
        int64_t o = id / (idx_n * inner);
        int64_t k = (id / inner) % idx_n;
        int64_t j = id % inner;
        int64_t di = idx_ptr[k];
        if (di < 0) di += dim_size;
        out_ptr[(o * dim_size + di) * inner + j] = fill_val;
    }).wait();

    return output;
}

// ============================================================================
// ScatterReduce - scatter with configurable reduction (sum/prod/mean/amax/amin)
// ============================================================================
auto scatter_reduce_kernel(const Tensor& input, int64_t dim, const Tensor& index,
                           const Tensor& source, const std::string& reduce,
                           bool include_self, sycl::queue& queue) -> Tensor {
    // Non-Float32: convert to Float32 and recurse
    if (input.dtype() != DType::Float32) {
        auto f32_in = input.to(DType::Float32);
        auto f32_src = source.to(DType::Float32);
        auto r = scatter_reduce_kernel(f32_in, dim, index, f32_src, reduce, include_self, queue);
        return r.to(input.dtype());
    }

    auto output = input.clone();
    int64_t ndim = output.shape().size();
    if (dim < 0) dim += ndim;
    auto shape = output.shape();

    // index and source share the SAME (index) shape, which may have SMALLER
    // trailing dims than self. We must therefore iterate index.numel() items
    // and stride index/source by the INDEX shape, while the destination offset
    // is computed with the OUTPUT (self) strides and the scatter-dim coordinate
    // replaced by idx_ptr[flat]. Mirrors scatter_add_kernel / the CPU reference;
    // deriving the iteration count or src indexing from self's collapsed
    // outer/inner (the old code) over-reads index/source out of bounds.
    Tensor index_c  = index.is_contiguous()  ? index  : index.contiguous();
    Tensor source_c = source.is_contiguous() ? source : source.contiguous();

    int64_t total = index_c.numel();
    if (total == 0) return output;

    auto idx_shape = index_c.shape();
    std::vector<int64_t> idx_shape_vec(idx_shape.begin(), idx_shape.end());

    // Output (self) strides and index strides, both row-major contiguous.
    std::array<int64_t, 8> d_out_strides{}, d_idx_strides{};
    { int64_t s = 1; for (int64_t i = ndim - 1; i >= 0; --i) { d_out_strides[i] = s; s *= shape[i]; } }
    { int64_t s = 1; for (int64_t i = ndim - 1; i >= 0; --i) { d_idx_strides[i] = s; s *= idx_shape_vec[i]; } }

    float* out_ptr = get_data_ptr<float>(output);
    const float* src_ptr = get_data_ptr<const float>(source_c);
    const int64_t* idx_ptr = get_data_ptr<const int64_t>(index_c);

    // Decode the flat index-space position into the full output offset:
    // for each dim, take the index-space coordinate, except along `dim` where
    // the destination coordinate is idx_ptr[flat]. Defined as a device lambda
    // re-derived inline in each kernel (SYCL lambdas cannot capture lambdas
    // cleanly across kernels, so the body is duplicated per reduce mode).

    // If !include_self, initialize touched positions to identity.
    if (!include_self) {
        float identity;
        if (reduce == "sum" || reduce == "mean") identity = 0.0f;
        else if (reduce == "prod") identity = 1.0f;
        else if (reduce == "amax") identity = -3.402823466e+38f;
        else if (reduce == "amin") identity = 3.402823466e+38f;
        else throw std::invalid_argument("scatter_reduce: unknown reduce mode '" + reduce + "'");

        queue.parallel_for<ScatterReduceInitKernelF32>(sycl::range<1>(total), [=](sycl::id<1> tid) {
            int64_t flat = static_cast<int64_t>(tid[0]);
            int64_t remaining = flat;
            int64_t dst_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                dst_offset += (d == dim ? idx_ptr[flat] : coord) * d_out_strides[d];
            }
            out_ptr[dst_offset] = identity;
        }).wait();
    }

    // Allocate count buffer for mean mode.
    int* count_ptr = nullptr;
    int64_t out_numel = output.numel();
    int* count_alloc = nullptr;
    if (reduce == "mean") {
        count_alloc = sycl::malloc_device<int>(out_numel, queue);
        queue.memset(count_alloc, 0, out_numel * sizeof(int)).wait();
        count_ptr = count_alloc;
    }

    if (reduce == "sum" || reduce == "mean") {
        int* cnt_ptr = count_ptr;
        bool is_mean = (reduce == "mean");
        queue.parallel_for<ScatterReduceSumKernelF32>(sycl::range<1>(total), [=](sycl::id<1> tid) {
            int64_t flat = static_cast<int64_t>(tid[0]);
            int64_t remaining = flat;
            int64_t dst_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                dst_offset += (d == dim ? idx_ptr[flat] : coord) * d_out_strides[d];
            }
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ref(out_ptr[dst_offset]);
            ref.fetch_add(src_ptr[flat]);
            if (is_mean && cnt_ptr) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> cnt_ref(cnt_ptr[dst_offset]);
                cnt_ref.fetch_add(1);
            }
        }).wait();
    } else if (reduce == "prod") {
        queue.parallel_for<ScatterReduceProdKernelF32>(sycl::range<1>(total), [=](sycl::id<1> tid) {
            int64_t flat = static_cast<int64_t>(tid[0]);
            int64_t remaining = flat;
            int64_t dst_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                dst_offset += (d == dim ? idx_ptr[flat] : coord) * d_out_strides[d];
            }
            float val = src_ptr[flat];
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ref(out_ptr[dst_offset]);
            float expected = ref.load();
            while (!ref.compare_exchange_weak(expected, expected * val)) {}
        }).wait();
    } else if (reduce == "amax") {
        queue.parallel_for<ScatterReduceAmaxKernelF32>(sycl::range<1>(total), [=](sycl::id<1> tid) {
            int64_t flat = static_cast<int64_t>(tid[0]);
            int64_t remaining = flat;
            int64_t dst_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                dst_offset += (d == dim ? idx_ptr[flat] : coord) * d_out_strides[d];
            }
            float val = src_ptr[flat];
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ref(out_ptr[dst_offset]);
            float expected = ref.load();
            while (val > expected) {
                if (ref.compare_exchange_weak(expected, val)) break;
            }
        }).wait();
    } else if (reduce == "amin") {
        queue.parallel_for<ScatterReduceAminKernelF32>(sycl::range<1>(total), [=](sycl::id<1> tid) {
            int64_t flat = static_cast<int64_t>(tid[0]);
            int64_t remaining = flat;
            int64_t dst_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                dst_offset += (d == dim ? idx_ptr[flat] : coord) * d_out_strides[d];
            }
            float val = src_ptr[flat];
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ref(out_ptr[dst_offset]);
            float expected = ref.load();
            while (val < expected) {
                if (ref.compare_exchange_weak(expected, val)) break;
            }
        }).wait();
    } else {
        if (count_alloc) sycl::free(count_alloc, queue);
        throw std::invalid_argument("scatter_reduce: unknown reduce mode '" + reduce + "'");
    }

    // For mean mode: divide by counts.
    //
    // Phase 7.6 mean fix (mirrors CUDA / ROCm): counts = number of scatters
    // touching this position; self is NOT counted in the scatter kernel.
    // With include_self=true the accumulator is `input + sum(scatters)`,
    // so divisor is `count + 1`. Untouched positions (count == 0) keep
    // their initial input value and are skipped.
    if (reduce == "mean" && count_alloc) {
        int incl = include_self ? 1 : 0;
        queue.parallel_for<ScatterReduceMeanDivKernelF32>(sycl::range<1>(out_numel), [=](sycl::id<1> tid) {
            int64_t i = static_cast<int64_t>(tid);
            int c = count_alloc[i];
            if (c <= 0) return;
            float divisor = (incl ? static_cast<float>(c + 1) : static_cast<float>(c));
            out_ptr[i] /= divisor;
        }).wait();
        sycl::free(count_alloc, queue);
    }

    return output;
}

// ============================================================================
// SYCL kernel class declarations for new ops
// ============================================================================
class TakeAlongDimKernelF32;
class TakeAlongDimKernelF64;
class TakeAlongDimKernelI32;
class TakeAlongDimKernelI64;
class MaskedScatterKernelF32;
class MaskedScatterKernelF64;
class MaskedScatterKernelI32;
class MaskedScatterKernelI64;
class MaskedScatterPrefixSumKernel;

// ============================================================================
// take_along_dim kernel (SYCL)
// ============================================================================

auto take_along_dim_kernel(const Tensor& input, const Tensor& indices, int64_t dim,
                           sycl::queue& queue) -> Tensor {
    auto in_shape = input.shape();
    auto idx_shape = indices.shape();
    int64_t ndim = in_shape.size();
    if (dim < 0) dim += ndim;

    if (ndim > 8) {
        throw std::runtime_error("take_along_dim OneAPI: max 8 dimensions supported");
    }

    Tensor output(std::vector<int64_t>(idx_shape.begin(), idx_shape.end()),
                  input.dtype(), input.device());
    int64_t numel = indices.numel();
    if (numel == 0) return output;

    int64_t in_dim_size = in_shape[dim];

    // Per-dimension decode requires the index shape and the INPUT's own
    // contiguous strides: input and index may differ on non-dim axes (PyTorch
    // broadcasts the index against the input), so every non-dim coordinate must
    // be re-linearised against the input strides — not the index extents.
    // Mirrors the CPU reference.
    std::array<int64_t, 8> idx_shape_arr{};
    std::array<int64_t, 8> in_strides{};
    {
        int64_t s = 1;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            in_strides[d] = s;
            s *= in_shape[d];
        }
        for (int64_t d = 0; d < ndim; ++d) idx_shape_arr[d] = idx_shape[d];
    }

    const int64_t* idx_ptr = indices.data<int64_t>();

    // Validate indices host-side (normalize negatives, throw on out-of-range),
    // matching the CPU reference (throwing inside a SYCL kernel is not possible).
    {
        std::vector<int64_t> host_idx(static_cast<size_t>(numel));
        queue.memcpy(host_idx.data(), idx_ptr,
                     static_cast<size_t>(numel) * sizeof(int64_t)).wait();
        for (int64_t i = 0; i < numel; ++i) {
            int64_t src_idx = host_idx[i];
            if (src_idx < 0) src_idx += in_dim_size;
            if (src_idx < 0 || src_idx >= in_dim_size) {
                throw std::out_of_range("take_along_dim: index out of range for dim");
            }
        }
    }

    const int64_t nd = ndim;
    const int64_t the_dim = dim;

    // Map an output/index flat position i (row-major over idx_shape) to the
    // matching input offset: decode each coordinate from idx_shape, substitute
    // the gathered index for the `dim` coordinate, and re-linearise every
    // coordinate against the input strides.
    auto in_offset_of = [=](int64_t i) {
        int64_t off = 0;
        int64_t rem = i;
        for (int64_t d = nd - 1; d >= 0; --d) {
            int64_t c = rem % idx_shape_arr[d];
            rem /= idx_shape_arr[d];
            if (d == the_dim) {
                int64_t src_idx = idx_ptr[i];
                if (src_idx < 0) src_idx += in_dim_size;
                off += src_idx * in_strides[d];
            } else {
                off += c * in_strides[d];
            }
        }
        return off;
    };

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = input.data<float>();
        float* out_ptr = output.data<float>();
        queue.parallel_for<TakeAlongDimKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> tid) {
            int64_t i = static_cast<int64_t>(tid);
            out_ptr[i] = in_ptr[in_offset_of(i)];
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = input.data<double>();
        double* out_ptr = output.data<double>();
        queue.parallel_for<TakeAlongDimKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> tid) {
            int64_t i = static_cast<int64_t>(tid);
            out_ptr[i] = in_ptr[in_offset_of(i)];
        }).wait();
    } else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = input.data<int32_t>();
        int32_t* out_ptr = output.data<int32_t>();
        queue.parallel_for<TakeAlongDimKernelI32>(sycl::range<1>(numel), [=](sycl::id<1> tid) {
            int64_t i = static_cast<int64_t>(tid);
            out_ptr[i] = in_ptr[in_offset_of(i)];
        }).wait();
    } else if (input.dtype() == DType::Int64) {
        const int64_t* in_ptr = input.data<int64_t>();
        int64_t* out_ptr = output.data<int64_t>();
        queue.parallel_for<TakeAlongDimKernelI64>(sycl::range<1>(numel), [=](sycl::id<1> tid) {
            int64_t i = static_cast<int64_t>(tid);
            out_ptr[i] = in_ptr[in_offset_of(i)];
        }).wait();
    } else {
        throw std::runtime_error("take_along_dim OneAPI: unsupported dtype");
    }

    return output;
}

// ============================================================================
// masked_scatter kernel — native SYCL with exclusive_scan prefix sum
// ============================================================================

auto masked_scatter_kernel(const Tensor& input, const Tensor& mask_in,
                           const Tensor& source, sycl::queue& queue) -> Tensor {
    int64_t numel = input.numel();
    int64_t src_numel = source.numel();

    // Clone input to output (preserves values where mask is false)
    Tensor output = input.clone();

    if (numel == 0) return output;

    // Accept non-Bool masks (nonzero = true), matching the CPU backend.
    Tensor mask_storage;
    const Tensor& mask = (mask_in.dtype() == DType::Bool)
        ? mask_in
        : (mask_storage = mask_in.to(DType::Bool));
    const bool* mask_ptr = get_data_ptr<const bool>(mask);

    // Step 1: Convert bool mask to int32 on device for prefix sum
    int32_t* mask_int = sycl::malloc_device<int32_t>(numel, queue);
    queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
        mask_int[i] = mask_ptr[i] ? 1 : 0;
    }).wait();

    // Step 2: Compute exclusive prefix sum of mask to get scatter indices
    // prefix_sum[i] = number of true values in mask[0..i-1]
    int32_t* prefix_sum = sycl::malloc_device<int32_t>(numel, queue);

    // Use a Blelloch-style work-efficient parallel scan for large arrays,
    // or a simple sequential scan on device for moderate sizes.
    // For production quality, we use a two-pass approach that works for all sizes.
    {
        // Pass 1: Compute block-level sums
        constexpr int64_t BLOCK_SIZE = 256;
        int64_t num_blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // For simplicity with SYCL, use a sequential scan kernel for the prefix sum.
        // This is O(n) work and runs as a single work-item, but is correct.
        // For very large tensors, a multi-pass parallel scan would be better,
        // but the scatter itself is the bottleneck, not the prefix sum.
        //
        // However, for better GPU utilization, we do a block-parallel approach:
        // 1. Each block computes local prefix sums
        // 2. Block totals are scanned
        // 3. Block offsets are added

        int32_t* block_totals = sycl::malloc_device<int32_t>(num_blocks, queue);
        int32_t* block_offsets = sycl::malloc_device<int32_t>(num_blocks, queue);

        // Phase 1: Local prefix sums within each block + compute block totals
        queue.parallel_for(sycl::range<1>(num_blocks), [=](sycl::id<1> bid) {
            int64_t block_start = bid * BLOCK_SIZE;
            int64_t block_end = sycl::min(block_start + BLOCK_SIZE, numel);
            int32_t running = 0;
            for (int64_t i = block_start; i < block_end; ++i) {
                prefix_sum[i] = running;
                running += mask_int[i];
            }
            block_totals[bid] = running;
        }).wait();

        // Phase 2: Exclusive scan of block totals (sequential — num_blocks is small)
        queue.single_task([=]() {
            int32_t running = 0;
            for (int64_t b = 0; b < num_blocks; ++b) {
                block_offsets[b] = running;
                running += block_totals[b];
            }
        }).wait();

        // Phase 3: Add block offsets to local prefix sums
        if (num_blocks > 1) {
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
                int64_t block_id = static_cast<int64_t>(i) / BLOCK_SIZE;
                prefix_sum[i] += block_offsets[block_id];
            }).wait();
        }

        sycl::free(block_totals, queue);
        sycl::free(block_offsets, queue);
    }

    // Total number of true mask entries = exclusive_prefix[last] + mask[last].
    // PyTorch / the CPU reference treat a source with fewer elements than the
    // true count as an error, so check before scattering.
    {
        int32_t last_prefix = 0, last_mask = 0;
        queue.memcpy(&last_prefix, prefix_sum + (numel - 1), sizeof(int32_t)).wait();
        queue.memcpy(&last_mask, mask_int + (numel - 1), sizeof(int32_t)).wait();
        int64_t true_count = static_cast<int64_t>(last_prefix) + static_cast<int64_t>(last_mask);
        if (true_count > src_numel) {
            sycl::free(mask_int, queue);
            sycl::free(prefix_sum, queue);
            throw std::runtime_error("masked_scatter: source has fewer elements than mask true count");
        }
    }

    // Step 3: Scatter source values using prefix sum indices
    if (input.dtype() == DType::Float32) {
        float* out_ptr = get_data_ptr<float>(output);
        const float* src_ptr = get_data_ptr<const float>(source);
        queue.parallel_for<MaskedScatterKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            if (mask_ptr[i]) {
                int32_t src_idx = prefix_sum[i];
                if (src_idx < src_numel) {
                    out_ptr[i] = src_ptr[src_idx];
                }
            }
        }).wait();
    } else if (input.dtype() == DType::Float64) {
        double* out_ptr = get_data_ptr<double>(output);
        const double* src_ptr = get_data_ptr<const double>(source);
        queue.parallel_for<MaskedScatterKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            if (mask_ptr[i]) {
                int32_t src_idx = prefix_sum[i];
                if (src_idx < src_numel) {
                    out_ptr[i] = src_ptr[src_idx];
                }
            }
        }).wait();
    } else if (input.dtype() == DType::Int32) {
        int32_t* out_ptr = get_data_ptr<int32_t>(output);
        const int32_t* src_ptr = get_data_ptr<const int32_t>(source);
        queue.parallel_for<MaskedScatterKernelI32>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            if (mask_ptr[i]) {
                int32_t src_idx = prefix_sum[i];
                if (src_idx < src_numel) {
                    out_ptr[i] = src_ptr[src_idx];
                }
            }
        }).wait();
    } else if (input.dtype() == DType::Int64) {
        int64_t* out_ptr = get_data_ptr<int64_t>(output);
        const int64_t* src_ptr = get_data_ptr<const int64_t>(source);
        queue.parallel_for<MaskedScatterKernelI64>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            if (mask_ptr[i]) {
                int32_t src_idx = prefix_sum[i];
                if (src_idx < src_numel) {
                    out_ptr[i] = src_ptr[src_idx];
                }
            }
        }).wait();
    } else {
        sycl::free(mask_int, queue);
        sycl::free(prefix_sum, queue);
        throw std::runtime_error("masked_scatter OneAPI: unsupported dtype");
    }

    sycl::free(mask_int, queue);
    sycl::free(prefix_sum, queue);

    return output;
}

// ============================================================================
// tril_indices / triu_indices — native SYCL GPU kernels
// ============================================================================

class TrilIndicesKernel;
class TriuIndicesKernel;

// Closed-form count of lower-triangular indices.
// Each row r contributes max(0, min(col, r + offset + 1)) elements.
static auto tril_count(int64_t row, int64_t col, int64_t offset) -> int64_t {
    int64_t n = 0;
    // Rows where the contribution is partial: r + offset + 1 < col, i.e. r < col - offset - 1
    // Rows where the contribution is full (col): r + offset + 1 >= col, i.e. r >= col - offset - 1
    // Rows where contribution is zero: r + offset + 1 <= 0, i.e. r < -offset

    int64_t r_start = std::max(int64_t(0), -offset);          // first row with nonzero count
    int64_t r_full  = std::max(int64_t(0), col - offset - 1); // first row contributing full col
    r_full = std::min(r_full, row);
    r_start = std::min(r_start, row);

    // Partial rows: r in [r_start, min(r_full, row)) contribute (r + offset + 1) each
    int64_t partial_end = std::min(r_full, row);
    if (partial_end > r_start) {
        // Sum of (r + offset + 1) for r in [r_start, partial_end)
        // = sum of (offset + 1 + r) = count*(offset+1) + sum(r)
        int64_t count = partial_end - r_start;
        int64_t first_val = r_start + offset + 1;
        int64_t last_val  = partial_end - 1 + offset + 1;
        n += count * (first_val + last_val) / 2;
    }
    // Full rows: r in [r_full, row) contribute col each
    if (row > r_full) {
        n += (row - r_full) * col;
    }
    return n;
}

// Closed-form count of upper-triangular indices.
// Each row r contributes max(0, col - max(0, r + offset)) elements.
static auto triu_count(int64_t row, int64_t col, int64_t offset) -> int64_t {
    int64_t n = 0;
    // Rows where contribution is full col: r + offset <= 0, i.e. r < -offset + 1
    // Rows where contribution is partial: 0 < r + offset < col
    // Rows where contribution is zero: r + offset >= col

    int64_t r_partial_start = std::max(int64_t(0), -offset + 1); // first row with partial count (if offset <= 0, some are full)
    // Actually: for r < max(0, 1 - offset), contribution is col (since max(0, r+offset) == 0)
    int64_t r_full_end = std::max(int64_t(0), 1 - offset);       // rows [0, r_full_end) contribute col
    r_full_end = std::min(r_full_end, row);

    int64_t r_zero_start = std::max(int64_t(0), col - offset);   // first row with zero contribution
    r_zero_start = std::min(r_zero_start, row);

    // Full rows: [0, r_full_end)
    n += r_full_end * col;

    // Partial rows: [r_full_end, r_zero_start) contribute (col - r - offset) each
    int64_t p_start = std::max(r_full_end, int64_t(0));
    int64_t p_end   = r_zero_start;
    if (p_end > p_start) {
        int64_t count = p_end - p_start;
        int64_t first_val = col - (p_start + offset);
        int64_t last_val  = col - (p_end - 1 + offset);
        n += count * (first_val + last_val) / 2;
    }
    return n;
}

auto tril_indices_kernel(int64_t row, int64_t col, int64_t offset, sycl::queue& queue) -> Tensor {
    int64_t n = tril_count(row, col, offset);
    if (n == 0) return tenzor::empty({2, 0}, DType::Int64, Device::oneapi(0));

    Tensor output = tenzor::empty({2, n}, DType::Int64, Device::oneapi(0));
    int64_t* out_ptr = get_data_ptr<int64_t>(output);

    queue.parallel_for<TrilIndicesKernel>(sycl::range<1>(n), [=](sycl::id<1> id) {
        int64_t idx = static_cast<int64_t>(id[0]);

        // Binary search: find row r such that cumulative count up to row r > idx
        // Cumulative count through row r = sum_{i=max(0,-offset)}^{r} min(col, i + offset + 1)
        int64_t lo = 0, hi = row - 1;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            // Compute cumulative count through row mid
            int64_t r_start = (-offset > 0) ? -offset : int64_t(0);
            int64_t cum = 0;
            if (mid >= r_start) {
                int64_t r_full = (col - offset - 1 > 0) ? col - offset - 1 : int64_t(0);
                int64_t partial_end = (r_full < mid + 1) ? r_full : mid + 1;
                if (partial_end > r_start) {
                    int64_t count = partial_end - r_start;
                    int64_t fv = r_start + offset + 1;
                    int64_t lv = partial_end - 1 + offset + 1;
                    cum += count * (fv + lv) / 2;
                }
                if (mid + 1 > r_full) {
                    int64_t full_start = (r_full > r_start) ? r_full : r_start;
                    if (mid + 1 > full_start) {
                        cum += (mid + 1 - full_start) * col;
                    }
                }
            }
            if (cum <= idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        int64_t r = lo;
        // Compute cumulative count through row (r - 1) to get column offset
        int64_t prev_cum = 0;
        if (r > 0) {
            int64_t r_start = (-offset > 0) ? -offset : int64_t(0);
            if (r > r_start) {
                int64_t r_full = (col - offset - 1 > 0) ? col - offset - 1 : int64_t(0);
                int64_t partial_end = (r_full < r) ? r_full : r;
                if (partial_end > r_start) {
                    int64_t count = partial_end - r_start;
                    int64_t fv = r_start + offset + 1;
                    int64_t lv = partial_end - 1 + offset + 1;
                    prev_cum += count * (fv + lv) / 2;
                }
                if (r > r_full) {
                    int64_t full_start = (r_full > r_start) ? r_full : r_start;
                    if (r > full_start) {
                        prev_cum += (r - full_start) * col;
                    }
                }
            }
        }
        int64_t c = idx - prev_cum;
        out_ptr[idx] = r;
        out_ptr[n + idx] = c;
    }).wait();

    return output;
}

auto triu_indices_kernel(int64_t row, int64_t col, int64_t offset, sycl::queue& queue) -> Tensor {
    int64_t n = triu_count(row, col, offset);
    if (n == 0) return tenzor::empty({2, 0}, DType::Int64, Device::oneapi(0));

    Tensor output = tenzor::empty({2, n}, DType::Int64, Device::oneapi(0));
    int64_t* out_ptr = get_data_ptr<int64_t>(output);

    queue.parallel_for<TriuIndicesKernel>(sycl::range<1>(n), [=](sycl::id<1> id) {
        int64_t idx = static_cast<int64_t>(id[0]);

        // Binary search: find row r such that cumulative count up to row r > idx
        int64_t lo = 0, hi = row - 1;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            // Compute cumulative count through row mid
            int64_t r_full_end = (1 - offset > 0) ? 1 - offset : int64_t(0);
            int64_t r_zero = (col - offset > 0) ? col - offset : int64_t(0);
            int64_t cum = 0;

            int64_t fe = (r_full_end < mid + 1) ? r_full_end : mid + 1;
            cum += fe * col;

            int64_t p_start = (r_full_end > 0) ? r_full_end : int64_t(0);
            int64_t p_end = (r_zero < mid + 1) ? r_zero : mid + 1;
            if (p_end > p_start) {
                int64_t count = p_end - p_start;
                int64_t fv = col - (p_start + offset);
                int64_t lv = col - (p_end - 1 + offset);
                cum += count * (fv + lv) / 2;
            }

            if (cum <= idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        int64_t r = lo;
        // Compute cumulative count through row (r - 1)
        int64_t prev_cum = 0;
        if (r > 0) {
            int64_t r_full_end = (1 - offset > 0) ? 1 - offset : int64_t(0);
            int64_t r_zero = (col - offset > 0) ? col - offset : int64_t(0);

            int64_t fe = (r_full_end < r) ? r_full_end : r;
            prev_cum += fe * col;

            int64_t p_start = (r_full_end > 0) ? r_full_end : int64_t(0);
            int64_t p_end = (r_zero < r) ? r_zero : r;
            if (p_end > p_start) {
                int64_t count = p_end - p_start;
                int64_t fv = col - (p_start + offset);
                int64_t lv = col - (p_end - 1 + offset);
                prev_cum += count * (fv + lv) / 2;
            }
        }
        int64_t c = idx - prev_cum;
        int64_t col_start = (r + offset > 0) ? r + offset : int64_t(0);
        out_ptr[idx] = r;
        out_ptr[n + idx] = col_start + c;
    }).wait();

    return output;
}

} // namespace oneapi
} // namespace tenzor
