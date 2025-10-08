#pragma once

#include "../core/tensor.hpp"

namespace tenzor {

// Indexing operations

// Slice
auto slice(const Tensor& input,
          int64_t dim,
          int64_t start,
          int64_t end,
          int64_t step = 1) -> Tensor;

// Index select
auto index_select(const Tensor& input,
                 int64_t dim,
                 const Tensor& index) -> Tensor;

// Gather
auto gather(const Tensor& input,
           int64_t dim,
           const Tensor& index) -> Tensor;

// Scatter
auto scatter(const Tensor& input,
            int64_t dim,
            const Tensor& index,
            const Tensor& src) -> Tensor;

// Masked select
auto masked_select(const Tensor& input, const Tensor& mask) -> Tensor;

// Masked fill
auto masked_fill(const Tensor& input, const Tensor& mask, float value) -> Tensor;

// Where (conditional selection)
auto where(const Tensor& condition,
          const Tensor& x,
          const Tensor& y) -> Tensor;

// Take
auto take(const Tensor& input, const Tensor& index) -> Tensor;

// Put
auto put(const Tensor& input, const Tensor& index, const Tensor& source) -> Tensor;

} // namespace tenzor
