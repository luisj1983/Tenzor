#pragma once

#include "tensor.hpp"
#include <optional>
#include <string>
#include <vector>
#include <stdexcept>

namespace tenzor {

/**
 * @brief Lightweight wrapper adding optional dimension names to a Tensor.
 *
 * NamedTensor does not modify the core Tensor class. It holds a Tensor
 * alongside a vector of optional name strings, one per dimension. Operations
 * that accept dimension names resolve them to integer indices internally and
 * delegate to the underlying Tensor.
 *
 * @code
 *   auto t = NamedTensor(rand({3, 4, 5}), {"batch", "height", "width"});
 *   auto s = t.sum("height");  // Sum along the "height" dimension
 * @endcode
 */
class NamedTensor {
public:
    NamedTensor() = default;

    /// Create from tensor with dimension names.
    /// @throws std::invalid_argument if names.size() != tensor.ndim()
    NamedTensor(Tensor tensor, std::vector<std::optional<std::string>> names);

    /// Create from tensor without names (all dimensions unnamed).
    explicit NamedTensor(Tensor tensor);

    // -- Accessors ----------------------------------------------------------

    auto tensor() const -> const Tensor& { return tensor_; }
    auto tensor() -> Tensor& { return tensor_; }

    auto names() const -> const std::vector<std::optional<std::string>>& {
        return names_;
    }

    /// True if at least one dimension has a name.
    [[nodiscard]] auto has_names() const -> bool;

    /// Return the index of the dimension named @p name.
    /// @throws std::invalid_argument if no dimension has that name.
    [[nodiscard]] auto dim_index(const std::string& name) const -> int64_t;

    // -- Rename / refine ----------------------------------------------------

    /// Return a new NamedTensor with replaced names.
    [[nodiscard]] auto rename(
        std::vector<std::optional<std::string>> new_names) const -> NamedTensor;

    /// Refine names: set names where currently unnamed; error on conflict.
    [[nodiscard]] auto refine_names(
        const std::vector<std::optional<std::string>>& new_names) const
        -> NamedTensor;

    // -- Named reduction / shape ops ----------------------------------------

    [[nodiscard]] auto sum(const std::string& dim,
                           bool keepdim = false) const -> NamedTensor;
    [[nodiscard]] auto mean(const std::string& dim,
                            bool keepdim = false) const -> NamedTensor;
    [[nodiscard]] auto max(const std::string& dim) const -> NamedTensor;
    [[nodiscard]] auto min(const std::string& dim) const -> NamedTensor;

    [[nodiscard]] auto squeeze(const std::string& dim) const -> NamedTensor;

    /// Insert a new dimension with the given @p name at position @p dim.
    [[nodiscard]] auto unsqueeze(const std::string& name,
                                 int64_t dim) const -> NamedTensor;

    [[nodiscard]] auto transpose(const std::string& dim0,
                                 const std::string& dim1) const -> NamedTensor;

    /// Permute dimensions so they match @p target_names order.
    [[nodiscard]] auto align_to(
        const std::vector<std::string>& target_names) const -> NamedTensor;

    // -- Shape delegation ---------------------------------------------------

    auto shape() const { return tensor_.shape(); }
    auto ndim() const { return tensor_.ndim(); }
    auto dtype() const { return tensor_.dtype(); }
    auto device() const { return tensor_.device(); }

private:
    Tensor tensor_;
    std::vector<std::optional<std::string>> names_;

    /// Produce names after reducing dimension @p dim.
    [[nodiscard]] auto names_after_reduction(int64_t dim, bool keepdim) const
        -> std::vector<std::optional<std::string>>;
};

} // namespace tenzor
