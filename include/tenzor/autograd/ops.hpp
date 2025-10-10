#pragma once

#include "variable.hpp"
#include "function.hpp"
#include <optional>

namespace tenzor {

// Variable-aware operations that preserve autograd graph

auto sum(const Variable& input,
         std::optional<int64_t> dim = std::nullopt,
         bool keepdim = false) -> Variable;

auto mean(const Variable& input,
          std::optional<int64_t> dim = std::nullopt,
          bool keepdim = false) -> Variable;

auto log(const Variable& input) -> Variable;

auto exp(const Variable& input) -> Variable;

auto neg(const Variable& input) -> Variable;

auto log_softmax(const Variable& input, int64_t dim) -> Variable;

} // namespace tenzor
