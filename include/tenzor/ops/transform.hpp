#pragma once

#include <vector>
#include "../core/tensor.hpp"

namespace tenzor {

// Shape transformation operations

// Reshape
auto reshape(const Tensor& input, std::vector<int64_t> shape) -> Tensor;

// View (zero-copy reshape)
auto view(const Tensor& input, std::vector<int64_t> shape) -> Tensor;

// Transpose
auto transpose(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor;

// Permute
auto permute(const Tensor& input, std::vector<int64_t> dims) -> Tensor;

// Squeeze
auto squeeze(const Tensor& input, std::optional<int64_t> dim = std::nullopt) -> Tensor;

// Unsqueeze
auto unsqueeze(const Tensor& input, int64_t dim) -> Tensor;

// Flatten
auto flatten(const Tensor& input, int64_t start_dim = 0, int64_t end_dim = -1) -> Tensor;

// Concatenate
auto cat(std::span<const Tensor> tensors, int64_t dim = 0) -> Tensor;

// Stack
auto stack(std::span<const Tensor> tensors, int64_t dim = 0) -> Tensor;

// Split
auto split(const Tensor& input, int64_t split_size, int64_t dim = 0) -> std::vector<Tensor>;

// Chunk
auto chunk(const Tensor& input, int64_t chunks, int64_t dim = 0) -> std::vector<Tensor>;

// Repeat
auto repeat(const Tensor& input, std::vector<int64_t> repeats) -> Tensor;

// Tile
auto tile(const Tensor& input, std::vector<int64_t> reps) -> Tensor;

// Expand
auto expand(const Tensor& input, std::vector<int64_t> shape) -> Tensor;

// Contiguous
auto contiguous(const Tensor& input) -> Tensor;

} // namespace tenzor
