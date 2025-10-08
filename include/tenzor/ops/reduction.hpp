#pragma once

#include <optional>
#include <vector>
#include "../core/tensor.hpp"

namespace tenzor {

// Reduction operations

// Sum
auto sum(const Tensor& input,
        std::optional<int64_t> dim = std::nullopt,
        bool keepdim = false) -> Tensor;

// Mean
auto mean(const Tensor& input,
         std::optional<int64_t> dim = std::nullopt,
         bool keepdim = false) -> Tensor;

// Max
auto max(const Tensor& input,
        std::optional<int64_t> dim = std::nullopt,
        bool keepdim = false) -> Tensor;

// Min
auto min(const Tensor& input,
        std::optional<int64_t> dim = std::nullopt,
        bool keepdim = false) -> Tensor;

// Argmax
auto argmax(const Tensor& input,
           std::optional<int64_t> dim = std::nullopt,
           bool keepdim = false) -> Tensor;

// Argmin
auto argmin(const Tensor& input,
           std::optional<int64_t> dim = std::nullopt,
           bool keepdim = false) -> Tensor;

// Product
auto prod(const Tensor& input,
         std::optional<int64_t> dim = std::nullopt,
         bool keepdim = false) -> Tensor;

// Standard deviation
auto std(const Tensor& input,
        std::optional<int64_t> dim = std::nullopt,
        bool keepdim = false,
        bool unbiased = true) -> Tensor;

// Variance
auto var(const Tensor& input,
        std::optional<int64_t> dim = std::nullopt,
        bool keepdim = false,
        bool unbiased = true) -> Tensor;

// Norm
auto norm(const Tensor& input,
         float p = 2.0f,
         std::optional<int64_t> dim = std::nullopt,
         bool keepdim = false) -> Tensor;

} // namespace tenzor
