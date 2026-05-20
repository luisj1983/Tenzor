/**
 * @file named_tensor.cpp
 * @brief NamedTensor implementation — optional dimension names on a Tensor.
 */

#include "tenzor/core/named_tensor.hpp"
#include "tenzor/ops/reduction.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tenzor {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

NamedTensor::NamedTensor(Tensor tensor,
                         std::vector<std::optional<std::string>> names)
    : tensor_(std::move(tensor)), names_(std::move(names)) {
    if (static_cast<int64_t>(names_.size()) != tensor_.ndim()) {
        throw std::invalid_argument(
            "NamedTensor: names count (" + std::to_string(names_.size()) +
            ") does not match tensor ndim (" +
            std::to_string(tensor_.ndim()) + ")");
    }
    // Validate no duplicate names.
    for (size_t i = 0; i < names_.size(); ++i) {
        if (!names_[i].has_value()) continue;
        for (size_t j = i + 1; j < names_.size(); ++j) {
            if (names_[j].has_value() && *names_[i] == *names_[j]) {
                throw std::invalid_argument(
                    "NamedTensor: duplicate dimension name '" + *names_[i] +
                    "' at dims " + std::to_string(i) + " and " +
                    std::to_string(j));
            }
        }
    }
}

NamedTensor::NamedTensor(Tensor tensor)
    : tensor_(std::move(tensor)),
      names_(static_cast<size_t>(tensor_.ndim()), std::nullopt) {}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

auto NamedTensor::has_names() const -> bool {
    return std::ranges::any_of(
        names_, [](const auto& n) { return n.has_value(); });
}

auto NamedTensor::dim_index(const std::string& name) const -> int64_t {
    for (size_t i = 0; i < names_.size(); ++i) {
        if (names_[i].has_value() && *names_[i] == name) {
            return static_cast<int64_t>(i);
        }
    }
    throw std::invalid_argument(
        "NamedTensor: dimension name '" + name + "' not found");
}

// ---------------------------------------------------------------------------
// Rename / refine
// ---------------------------------------------------------------------------

auto NamedTensor::rename(
    std::vector<std::optional<std::string>> new_names) const -> NamedTensor {
    return NamedTensor(tensor_, std::move(new_names));
}

auto NamedTensor::refine_names(
    const std::vector<std::optional<std::string>>& new_names) const
    -> NamedTensor {
    if (static_cast<int64_t>(new_names.size()) != ndim()) {
        throw std::invalid_argument(
            "refine_names: expected " + std::to_string(ndim()) +
            " names, got " + std::to_string(new_names.size()));
    }

    std::vector<std::optional<std::string>> result(names_);
    for (size_t i = 0; i < result.size(); ++i) {
        if (!new_names[i].has_value()) {
            // New is unnamed — keep current.
            continue;
        }
        if (!result[i].has_value()) {
            // Current unnamed, new is named — adopt.
            result[i] = new_names[i];
        } else if (*result[i] != *new_names[i]) {
            throw std::invalid_argument(
                "refine_names: cannot refine dim " + std::to_string(i) +
                " from '" + *result[i] + "' to '" + *new_names[i] + "'");
        }
        // Both named and matching — nothing to do.
    }
    return NamedTensor(tensor_, std::move(result));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

auto NamedTensor::names_after_reduction(int64_t dim, bool keepdim) const
    -> std::vector<std::optional<std::string>> {
    std::vector<std::optional<std::string>> result;
    result.reserve(static_cast<size_t>(ndim()));

    for (int64_t i = 0; i < ndim(); ++i) {
        if (i == dim) {
            if (keepdim) {
                // Dimension kept as size-1 but unnamed after reduction.
                result.push_back(std::nullopt);
            }
            // Otherwise, skip the reduced dimension entirely.
        } else {
            result.push_back(names_[static_cast<size_t>(i)]);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Named reduction operations
// ---------------------------------------------------------------------------

auto NamedTensor::sum(const std::string& dim, bool keepdim) const
    -> NamedTensor {
    auto idx = dim_index(dim);
    auto out = tenzor::sum(tensor_, idx, keepdim);
    return NamedTensor(std::move(out), names_after_reduction(idx, keepdim));
}

auto NamedTensor::mean(const std::string& dim, bool keepdim) const
    -> NamedTensor {
    auto idx = dim_index(dim);
    auto out = tenzor::mean(tensor_, idx, keepdim);
    return NamedTensor(std::move(out), names_after_reduction(idx, keepdim));
}

auto NamedTensor::max(const std::string& dim) const -> NamedTensor {
    auto idx = dim_index(dim);
    auto out = tenzor::max(tensor_, idx, /*keepdim=*/false);
    return NamedTensor(std::move(out), names_after_reduction(idx, false));
}

auto NamedTensor::min(const std::string& dim) const -> NamedTensor {
    auto idx = dim_index(dim);
    auto out = tenzor::min(tensor_, idx, /*keepdim=*/false);
    return NamedTensor(std::move(out), names_after_reduction(idx, false));
}

// ---------------------------------------------------------------------------
// Named shape operations
// ---------------------------------------------------------------------------

auto NamedTensor::squeeze(const std::string& dim) const -> NamedTensor {
    auto idx = dim_index(dim);
    auto out = tensor_.squeeze(idx);

    // Build names: remove the squeezed dimension (only if size was 1).
    std::vector<std::optional<std::string>> new_names;
    auto sh = tensor_.shape();
    for (int64_t i = 0; i < ndim(); ++i) {
        if (i == idx && sh[static_cast<size_t>(i)] == 1) {
            continue;  // Dimension removed.
        }
        new_names.push_back(names_[static_cast<size_t>(i)]);
    }
    return NamedTensor(std::move(out), std::move(new_names));
}

auto NamedTensor::unsqueeze(const std::string& name, int64_t dim) const
    -> NamedTensor {
    auto out = tensor_.unsqueeze(dim);

    // Normalise negative dim, then insert the new name at that
    // position in a copy of the existing name vector (audit item I.5 —
    // removed dead scaffolding loop that the next two lines made
    // redundant anyway).
    int64_t effective_dim = dim < 0 ? ndim() + 1 + dim : dim;

    std::vector<std::optional<std::string>> new_names = names_;
    new_names.insert(
        new_names.begin() + effective_dim,
        std::make_optional(name));

    return NamedTensor(std::move(out), std::move(new_names));
}

auto NamedTensor::transpose(const std::string& dim0,
                            const std::string& dim1) const -> NamedTensor {
    auto i0 = dim_index(dim0);
    auto i1 = dim_index(dim1);
    auto out = tensor_.transpose(i0, i1);

    auto new_names = names_;
    std::swap(new_names[static_cast<size_t>(i0)],
              new_names[static_cast<size_t>(i1)]);
    return NamedTensor(std::move(out), std::move(new_names));
}

auto NamedTensor::align_to(
    const std::vector<std::string>& target_names) const -> NamedTensor {
    if (static_cast<int64_t>(target_names.size()) != ndim()) {
        throw std::invalid_argument(
            "align_to: target names count (" +
            std::to_string(target_names.size()) +
            ") does not match tensor ndim (" +
            std::to_string(ndim()) + ")");
    }

    // Build permutation: target_names[i] should come from dim_index(target_names[i]).
    std::vector<int64_t> perm;
    perm.reserve(target_names.size());
    for (const auto& tn : target_names) {
        perm.push_back(dim_index(tn));
    }

    auto out = tensor_.permute(perm);

    // Build output names in target order.
    std::vector<std::optional<std::string>> new_names;
    new_names.reserve(target_names.size());
    for (const auto& tn : target_names) {
        new_names.emplace_back(tn);
    }
    return NamedTensor(std::move(out), std::move(new_names));
}

} // namespace tenzor
