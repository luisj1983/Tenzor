/**
 * @file symbolic_shape.hpp
 * @brief Symbolic dimension and shape types for dynamic shape support in JIT
 *
 * Provides SymbolicDim (a dimension that is either a concrete int64_t or a
 * symbolic string name) and SymbolicShape (a vector of SymbolicDim).
 * These types allow the JIT compiler to reason about dynamic shapes such as
 * variable batch sizes and sequence lengths.
 *
 * @code
 * SymbolicDim batch = SymbolicDim::symbolic("batch");
 * SymbolicDim channels = SymbolicDim::concrete(64);
 * SymbolicShape shape = {batch, channels, SymbolicDim::concrete(224), SymbolicDim::concrete(224)};
 *
 * // Arithmetic: concrete + concrete = concrete, symbolic stays symbolic
 * auto result = batch + channels;  // Still symbolic
 * auto c_result = channels + SymbolicDim::concrete(32);  // Concrete(96)
 * @endcode
 */

#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>
#include <sstream>
#include <stdexcept>

namespace tenzor {
namespace jit {

/**
 * @brief A dimension that is either concrete (int64_t) or symbolic (named).
 *
 * SymbolicDim represents a single dimension in a tensor shape. It can be:
 * - Concrete: a known integer value (e.g., 64, 224)
 * - Symbolic: a named variable (e.g., "batch", "seq_len")
 *
 * Symbolic dimensions propagate through arithmetic operations. When both
 * operands are concrete, the result is computed. When either operand is
 * symbolic, the result is a new symbolic expression.
 */
class SymbolicDim {
public:
    /**
     * @brief Create a concrete dimension.
     *
     * @param value Integer dimension value
     * @return Concrete SymbolicDim
     */
    static auto concrete(int64_t value) -> SymbolicDim {
        return SymbolicDim(value);
    }

    /**
     * @brief Create a symbolic (named) dimension.
     *
     * @param name Symbolic name (e.g., "batch", "seq_len")
     * @return Symbolic SymbolicDim
     */
    static auto symbolic(std::string name) -> SymbolicDim {
        return SymbolicDim(std::move(name));
    }

    /**
     * @brief Default constructor creates concrete dimension with value 0.
     */
    SymbolicDim() : dim_(int64_t{0}) {}

    /**
     * @brief Construct from integer (concrete dimension).
     *
     * @param value Dimension value
     */
    SymbolicDim(int64_t value) : dim_(value) {}  // NOLINT(google-explicit-constructor)

    /**
     * @brief Construct from string (symbolic dimension).
     *
     * @param name Symbolic name
     */
    SymbolicDim(std::string name) : dim_(std::move(name)) {}  // NOLINT(google-explicit-constructor)

    /**
     * @brief Check if this dimension is concrete (known integer).
     *
     * @return true if dimension has a concrete value
     */
    [[nodiscard]] auto is_concrete() const -> bool {
        return std::holds_alternative<int64_t>(dim_);
    }

    /**
     * @brief Check if this dimension is symbolic (named variable).
     *
     * @return true if dimension is a symbolic name
     */
    [[nodiscard]] auto is_symbolic() const -> bool {
        return std::holds_alternative<std::string>(dim_);
    }

    /**
     * @brief Get the concrete integer value.
     *
     * @return Concrete dimension value
     * @throws std::runtime_error if dimension is symbolic
     */
    [[nodiscard]] auto value() const -> int64_t {
        if (!is_concrete()) {
            throw std::runtime_error("Cannot get concrete value from symbolic dimension: " + name());
        }
        return std::get<int64_t>(dim_);
    }

    /**
     * @brief Get the symbolic name.
     *
     * @return Symbolic dimension name
     * @throws std::runtime_error if dimension is concrete
     */
    [[nodiscard]] auto name() const -> const std::string& {
        if (!is_symbolic()) {
            throw std::runtime_error("Cannot get name from concrete dimension");
        }
        return std::get<std::string>(dim_);
    }

    /**
     * @brief Get string representation.
     *
     * Returns the integer as a string for concrete dims, or the name for
     * symbolic dims.
     *
     * @return String representation
     */
    [[nodiscard]] auto to_string() const -> std::string {
        if (is_concrete()) {
            return std::to_string(std::get<int64_t>(dim_));
        }
        return std::get<std::string>(dim_);
    }

    /**
     * @brief Equality comparison.
     *
     * Two SymbolicDims are equal if they hold the same variant value.
     */
    auto operator==(const SymbolicDim& other) const -> bool {
        return dim_ == other.dim_;
    }

    auto operator!=(const SymbolicDim& other) const -> bool {
        return dim_ != other.dim_;
    }

    // ========================================================================
    // Symbolic arithmetic
    // ========================================================================

    /**
     * @brief Add two symbolic dimensions.
     *
     * If both are concrete, returns concrete sum.
     * If either is symbolic, returns a symbolic expression string.
     */
    friend auto operator+(const SymbolicDim& lhs, const SymbolicDim& rhs) -> SymbolicDim {
        if (lhs.is_concrete() && rhs.is_concrete()) {
            return SymbolicDim(lhs.value() + rhs.value());
        }
        return SymbolicDim(std::string("(" + lhs.to_string() + " + " + rhs.to_string() + ")"));
    }

    /**
     * @brief Subtract two symbolic dimensions.
     *
     * If both are concrete, returns concrete difference.
     * If either is symbolic, returns a symbolic expression string.
     */
    friend auto operator-(const SymbolicDim& lhs, const SymbolicDim& rhs) -> SymbolicDim {
        if (lhs.is_concrete() && rhs.is_concrete()) {
            return SymbolicDim(lhs.value() - rhs.value());
        }
        return SymbolicDim(std::string("(" + lhs.to_string() + " - " + rhs.to_string() + ")"));
    }

    /**
     * @brief Multiply two symbolic dimensions.
     *
     * If both are concrete, returns concrete product.
     * If either is symbolic, returns a symbolic expression string.
     */
    friend auto operator*(const SymbolicDim& lhs, const SymbolicDim& rhs) -> SymbolicDim {
        if (lhs.is_concrete() && rhs.is_concrete()) {
            return SymbolicDim(lhs.value() * rhs.value());
        }
        return SymbolicDim(std::string("(" + lhs.to_string() + " * " + rhs.to_string() + ")"));
    }

    /**
     * @brief Divide two symbolic dimensions.
     *
     * If both are concrete, returns concrete quotient (integer division).
     * If either is symbolic, returns a symbolic expression string.
     */
    friend auto operator/(const SymbolicDim& lhs, const SymbolicDim& rhs) -> SymbolicDim {
        if (lhs.is_concrete() && rhs.is_concrete()) {
            if (rhs.value() == 0) {
                throw std::runtime_error("Division by zero in symbolic dimension arithmetic");
            }
            return SymbolicDim(lhs.value() / rhs.value());
        }
        return SymbolicDim(std::string("(" + lhs.to_string() + " / " + rhs.to_string() + ")"));
    }

private:
    std::variant<int64_t, std::string> dim_;  ///< Either concrete value or symbolic name
};

/**
 * @brief A shape composed of symbolic dimensions.
 *
 * SymbolicShape is a vector of SymbolicDim, representing a tensor shape
 * where some dimensions may be symbolic (unknown at compile time).
 * This is used by the JIT compiler to support dynamic batching,
 * variable sequence lengths, and other dynamic shape scenarios.
 *
 * @code
 * SymbolicShape shape = {
 *     SymbolicDim::symbolic("batch"),
 *     SymbolicDim::concrete(3),
 *     SymbolicDim::concrete(224),
 *     SymbolicDim::concrete(224)
 * };
 *
 * bool dynamic = shape.has_symbolic_dims();  // true
 * auto rank = shape.rank();                  // 4
 * @endcode
 */
class SymbolicShape {
public:
    /**
     * @brief Default constructor creates empty shape.
     */
    SymbolicShape() = default;

    /**
     * @brief Construct from initializer list of SymbolicDim.
     *
     * @param dims Initializer list of symbolic dimensions
     */
    SymbolicShape(std::initializer_list<SymbolicDim> dims)
        : dims_(dims) {}

    /**
     * @brief Construct from vector of SymbolicDim.
     *
     * @param dims Vector of symbolic dimensions
     */
    explicit SymbolicShape(std::vector<SymbolicDim> dims)
        : dims_(std::move(dims)) {}

    /**
     * @brief Construct from a concrete shape (vector of int64_t).
     *
     * All dimensions are concrete.
     *
     * @param concrete_shape Vector of concrete dimension values
     */
    static auto from_concrete(const std::vector<int64_t>& concrete_shape) -> SymbolicShape {
        std::vector<SymbolicDim> dims;
        dims.reserve(concrete_shape.size());
        for (auto d : concrete_shape) {
            dims.emplace_back(d);
        }
        return SymbolicShape(std::move(dims));
    }

    /**
     * @brief Get number of dimensions (rank).
     *
     * @return Number of dimensions
     */
    [[nodiscard]] auto rank() const -> size_t { return dims_.size(); }

    /**
     * @brief Check if shape is empty (rank 0).
     *
     * @return true if shape has no dimensions
     */
    [[nodiscard]] auto empty() const -> bool { return dims_.empty(); }

    /**
     * @brief Access dimension by index (unchecked).
     *
     * @param idx Dimension index
     * @return Reference to SymbolicDim at index
     */
    auto operator[](size_t idx) const -> const SymbolicDim& { return dims_[idx]; }
    auto operator[](size_t idx) -> SymbolicDim& { return dims_[idx]; }

    /**
     * @brief Access dimension by index (checked).
     *
     * @param idx Dimension index
     * @return Reference to SymbolicDim at index
     * @throws std::out_of_range if index is out of bounds
     */
    [[nodiscard]] auto at(size_t idx) const -> const SymbolicDim& { return dims_.at(idx); }
    auto at(size_t idx) -> SymbolicDim& { return dims_.at(idx); }

    /**
     * @brief Check if any dimension is symbolic.
     *
     * @return true if at least one dimension is symbolic
     */
    [[nodiscard]] auto has_symbolic_dims() const -> bool {
        for (const auto& d : dims_) {
            if (d.is_symbolic()) return true;
        }
        return false;
    }

    /**
     * @brief Check if all dimensions are concrete.
     *
     * @return true if every dimension has a known integer value
     */
    [[nodiscard]] auto is_fully_concrete() const -> bool {
        return !has_symbolic_dims();
    }

    /**
     * @brief Convert to concrete shape.
     *
     * Only valid if all dimensions are concrete.
     *
     * @return Vector of concrete dimension values
     * @throws std::runtime_error if any dimension is symbolic
     */
    [[nodiscard]] auto to_concrete() const -> std::vector<int64_t> {
        std::vector<int64_t> result;
        result.reserve(dims_.size());
        for (const auto& d : dims_) {
            result.push_back(d.value());  // throws if symbolic
        }
        return result;
    }

    /**
     * @brief Append a dimension.
     *
     * @param dim Dimension to append
     */
    auto push_back(SymbolicDim dim) -> void { dims_.push_back(std::move(dim)); }

    /**
     * @brief Insert a dimension at a position.
     *
     * @param pos Position to insert at
     * @param dim Dimension to insert
     */
    auto insert(size_t pos, SymbolicDim dim) -> void {
        dims_.insert(dims_.begin() + static_cast<ptrdiff_t>(pos), std::move(dim));
    }

    /**
     * @brief Erase a dimension at a position.
     *
     * @param pos Position to erase
     */
    auto erase(size_t pos) -> void {
        dims_.erase(dims_.begin() + static_cast<ptrdiff_t>(pos));
    }

    /**
     * @brief Get iterators for range-based for loops.
     */
    auto begin() const { return dims_.begin(); }
    auto end() const { return dims_.end(); }
    auto begin() { return dims_.begin(); }
    auto end() { return dims_.end(); }

    /**
     * @brief Get the underlying dimension vector.
     *
     * @return Reference to dimension vector
     */
    [[nodiscard]] auto dims() const -> const std::vector<SymbolicDim>& { return dims_; }

    /**
     * @brief Get string representation.
     *
     * @return String like "(batch, 3, 224, 224)"
     */
    [[nodiscard]] auto to_string() const -> std::string {
        std::ostringstream oss;
        oss << "(";
        for (size_t i = 0; i < dims_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << dims_[i].to_string();
        }
        oss << ")";
        return oss.str();
    }

    /**
     * @brief Equality comparison.
     */
    auto operator==(const SymbolicShape& other) const -> bool {
        return dims_ == other.dims_;
    }

    auto operator!=(const SymbolicShape& other) const -> bool {
        return dims_ != other.dims_;
    }

private:
    std::vector<SymbolicDim> dims_;  ///< Dimensions (concrete or symbolic)
};

// ============================================================================
// Symbolic shape utilities
// ============================================================================

/**
 * @brief Broadcast two symbolic shapes.
 *
 * Applies NumPy-style broadcasting rules to symbolic shapes.
 * - If both dimensions are concrete, standard broadcasting applies.
 * - If both dimensions are the same symbolic name, the result is that symbol.
 * - If one dimension is concrete 1, the result is the other dimension.
 * - Otherwise, a new symbolic expression is created.
 *
 * @param shape1 First symbolic shape
 * @param shape2 Second symbolic shape
 * @return Broadcasted symbolic shape
 * @throws std::runtime_error if shapes are not broadcastable
 */
inline auto broadcast_symbolic_shapes(const SymbolicShape& shape1,
                                      const SymbolicShape& shape2) -> SymbolicShape {
    size_t ndim = std::max(shape1.rank(), shape2.rank());
    std::vector<SymbolicDim> result(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        size_t idx1 = shape1.rank() > i ? shape1.rank() - 1 - i : SIZE_MAX;
        size_t idx2 = shape2.rank() > i ? shape2.rank() - 1 - i : SIZE_MAX;

        SymbolicDim dim1 = (idx1 != SIZE_MAX) ? shape1[idx1] : SymbolicDim::concrete(1);
        SymbolicDim dim2 = (idx2 != SIZE_MAX) ? shape2[idx2] : SymbolicDim::concrete(1);

        // Both concrete: use standard broadcasting
        if (dim1.is_concrete() && dim2.is_concrete()) {
            int64_t v1 = dim1.value();
            int64_t v2 = dim2.value();
            if (v1 == v2 || v1 == 1 || v2 == 1) {
                result[ndim - 1 - i] = SymbolicDim::concrete(std::max(v1, v2));
            } else {
                throw std::runtime_error("Symbolic shapes are not broadcastable");
            }
        }
        // One is concrete 1: take the other
        else if (dim1.is_concrete() && dim1.value() == 1) {
            result[ndim - 1 - i] = dim2;
        }
        else if (dim2.is_concrete() && dim2.value() == 1) {
            result[ndim - 1 - i] = dim1;
        }
        // Both symbolic with same name: keep the symbol
        else if (dim1 == dim2) {
            result[ndim - 1 - i] = dim1;
        }
        // Otherwise: not provably broadcastable at compile time,
        // keep as symbolic expression (broadcast result)
        else {
            result[ndim - 1 - i] = SymbolicDim::symbolic(
                "broadcast(" + dim1.to_string() + ", " + dim2.to_string() + ")");
        }
    }

    return SymbolicShape(std::move(result));
}

} // namespace jit
} // namespace tenzor
