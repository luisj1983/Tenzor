/**
 * @file symbolic_shape.hpp
 * @brief Symbolic dimension and shape types for dynamic shape support in JIT
 *
 * Provides SymbolicDim (a dimension that is either a concrete int64_t, a
 * symbolic string name, or an expression tree) and SymbolicShape (a vector
 * of SymbolicDim). These types allow the JIT compiler to reason about dynamic
 * shapes such as variable batch sizes and sequence lengths.
 *
 * @code
 * SymbolicDim batch = SymbolicDim::symbolic("batch");
 * SymbolicDim channels = SymbolicDim::concrete(64);
 * SymbolicShape shape = {batch, channels, SymbolicDim::concrete(224), SymbolicDim::concrete(224)};
 *
 * // Arithmetic: concrete + concrete = concrete, symbolic builds AST
 * auto result = batch + channels;  // Expression node: Add(batch, 64)
 * auto c_result = channels + SymbolicDim::concrete(32);  // Concrete(96)
 *
 * // Resolve expressions with an environment
 * SymbolicShapeEnvironment env;
 * env.bind("batch", 32);
 * int64_t resolved = env.resolve(result);  // 32 + 64 = 96
 * @endcode
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace tenzor {
namespace jit {

// ============================================================================
// Forward declarations for recursive type
// ============================================================================

class SymbolicDim;

/// Binary operation type for symbolic expressions
enum class ExprOp : uint8_t { Add, Sub, Mul, Div };

/// Forward-declared; full definition after SymbolicDim (needs SymbolicDim to be complete)
struct SymbolicExpr;

// ============================================================================
// SymbolicDim
// ============================================================================

/**
 * @brief A dimension that is either concrete (int64_t), symbolic (named), or an expression.
 *
 * SymbolicDim represents a single dimension in a tensor shape. It can be:
 * - Concrete: a known integer value (e.g., 64, 224)
 * - Symbolic: a named variable (e.g., "batch", "seq_len")
 * - Expression: a binary operation on two SymbolicDims (e.g., batch + 32)
 *
 * Arithmetic operators build expression AST nodes with algebraic simplification
 * (constant folding, identity elimination). The SymbolicShapeEnvironment can
 * recursively evaluate expression trees by substituting bound variable values.
 */
class SymbolicDim {
public:
    /// Create a concrete dimension.
    static auto concrete(int64_t value) -> SymbolicDim {
        return SymbolicDim(value);
    }

    /// Create a symbolic (named) dimension.
    static auto symbolic(std::string name) -> SymbolicDim {
        return SymbolicDim(std::move(name));
    }

    /// Create an expression dimension (defined after SymbolicExpr).
    static auto expr(ExprOp op, SymbolicDim lhs, SymbolicDim rhs) -> SymbolicDim;

    /// Default constructor creates concrete dimension with value 0.
    SymbolicDim() : dim_(int64_t{0}) {}

    /// Construct from integer (concrete dimension).
    SymbolicDim(int64_t value) : dim_(value) {}  // NOLINT(google-explicit-constructor)

    /// Construct from string (symbolic dimension).
    SymbolicDim(std::string name) : dim_(std::move(name)) {}  // NOLINT(google-explicit-constructor)

    /// Check if this dimension is concrete (known integer).
    [[nodiscard]] auto is_concrete() const -> bool {
        return std::holds_alternative<int64_t>(dim_);
    }

    /// Check if this dimension is symbolic (not concrete — includes both names and expressions).
    [[nodiscard]] auto is_symbolic() const -> bool {
        return !is_concrete();
    }

    /// Check if this dimension is a named symbol (not concrete, not expression).
    [[nodiscard]] auto is_named_symbol() const -> bool {
        return std::holds_alternative<std::string>(dim_);
    }

    /// Check if this dimension is an expression tree node.
    [[nodiscard]] auto is_expr() const -> bool {
        return std::holds_alternative<std::shared_ptr<SymbolicExpr>>(dim_);
    }

    /// Get the concrete integer value. Throws if not concrete.
    [[nodiscard]] auto value() const -> int64_t {
        if (!is_concrete()) {
            throw std::runtime_error("Cannot get concrete value from symbolic dimension: " + to_string());
        }
        return std::get<int64_t>(dim_);
    }

    /// Get the symbolic name. Throws if not a named symbol.
    [[nodiscard]] auto name() const -> const std::string& {
        if (!std::holds_alternative<std::string>(dim_)) {
            throw std::runtime_error("Cannot get name from non-named dimension: " + to_string());
        }
        return std::get<std::string>(dim_);
    }

    /// Get the expression node. Throws if not an expression.
    [[nodiscard]] auto expr_node() const -> const SymbolicExpr&;  // defined after SymbolicExpr

    /// Get string representation (defined after SymbolicExpr for expression branch).
    [[nodiscard]] auto to_string() const -> std::string;

    /// Structural equality comparison (defined after SymbolicExpr).
    auto operator==(const SymbolicDim& other) const -> bool;

    auto operator!=(const SymbolicDim& other) const -> bool {
        return !(*this == other);
    }

    // ========================================================================
    // Symbolic arithmetic (defined after SymbolicExpr for AST construction)
    // ========================================================================

    friend auto operator+(const SymbolicDim& lhs, const SymbolicDim& rhs) -> SymbolicDim;
    friend auto operator-(const SymbolicDim& lhs, const SymbolicDim& rhs) -> SymbolicDim;
    friend auto operator*(const SymbolicDim& lhs, const SymbolicDim& rhs) -> SymbolicDim;
    friend auto operator/(const SymbolicDim& lhs, const SymbolicDim& rhs) -> SymbolicDim;

private:
    /// Either concrete value, symbolic name, or expression tree node
    std::variant<int64_t, std::string, std::shared_ptr<SymbolicExpr>> dim_;
};

// ============================================================================
// SymbolicExpr (full definition — SymbolicDim is now complete)
// ============================================================================

/**
 * @brief Binary expression node for symbolic dimension arithmetic.
 *
 * Represents operations like (batch + 32), ((H + 2*pad - kernel) / stride + 1).
 * The tree is built by SymbolicDim's arithmetic operators with algebraic
 * simplification at construction time.
 */
struct SymbolicExpr {
    ExprOp op;
    SymbolicDim lhs;
    SymbolicDim rhs;

    SymbolicExpr(ExprOp op_, SymbolicDim lhs_, SymbolicDim rhs_)
        : op(op_), lhs(std::move(lhs_)), rhs(std::move(rhs_)) {}

    auto operator==(const SymbolicExpr& other) const -> bool {
        return op == other.op && lhs == other.lhs && rhs == other.rhs;
    }

    auto operator!=(const SymbolicExpr& other) const -> bool {
        return !(*this == other);
    }
};

// ============================================================================
// Deferred SymbolicDim method definitions (need SymbolicExpr to be complete)
// ============================================================================

inline auto SymbolicDim::expr(ExprOp op, SymbolicDim lhs, SymbolicDim rhs) -> SymbolicDim {
    SymbolicDim result;
    result.dim_ = std::make_shared<SymbolicExpr>(op, std::move(lhs), std::move(rhs));
    return result;
}

inline auto SymbolicDim::expr_node() const -> const SymbolicExpr& {
    if (!is_expr()) {
        throw std::runtime_error("Cannot get expression node from non-expression dimension");
    }
    return *std::get<std::shared_ptr<SymbolicExpr>>(dim_);
}

inline auto SymbolicDim::to_string() const -> std::string {
    if (is_concrete()) {
        return std::to_string(std::get<int64_t>(dim_));
    }
    if (is_named_symbol()) {
        return std::get<std::string>(dim_);
    }
    // Expression: produce parenthesized infix format matching legacy output
    auto& e = expr_node();
    const char* op_str = nullptr;
    switch (e.op) {
        case ExprOp::Add: op_str = " + "; break;
        case ExprOp::Sub: op_str = " - "; break;
        case ExprOp::Mul: op_str = " * "; break;
        case ExprOp::Div: op_str = " / "; break;
    }
    return "(" + e.lhs.to_string() + op_str + e.rhs.to_string() + ")";
}

inline auto SymbolicDim::operator==(const SymbolicDim& other) const -> bool {
    if (dim_.index() != other.dim_.index()) return false;
    if (is_concrete()) return std::get<int64_t>(dim_) == std::get<int64_t>(other.dim_);
    if (is_named_symbol()) return std::get<std::string>(dim_) == std::get<std::string>(other.dim_);
    // Both expressions: structural equality
    return expr_node() == other.expr_node();
}

// Arithmetic operators with algebraic simplification

inline auto operator+(const SymbolicDim& lhs, const SymbolicDim& rhs) -> SymbolicDim {
    // Constant folding
    if (lhs.is_concrete() && rhs.is_concrete()) {
        return SymbolicDim(lhs.value() + rhs.value());
    }
    // Identity: x + 0 = x, 0 + x = x
    if (rhs.is_concrete() && rhs.value() == 0) return lhs;
    if (lhs.is_concrete() && lhs.value() == 0) return rhs;
    return SymbolicDim::expr(ExprOp::Add, lhs, rhs);
}

inline auto operator-(const SymbolicDim& lhs, const SymbolicDim& rhs) -> SymbolicDim {
    // Constant folding
    if (lhs.is_concrete() && rhs.is_concrete()) {
        return SymbolicDim(lhs.value() - rhs.value());
    }
    // Identity: x - 0 = x
    if (rhs.is_concrete() && rhs.value() == 0) return lhs;
    return SymbolicDim::expr(ExprOp::Sub, lhs, rhs);
}

inline auto operator*(const SymbolicDim& lhs, const SymbolicDim& rhs) -> SymbolicDim {
    // Constant folding
    if (lhs.is_concrete() && rhs.is_concrete()) {
        return SymbolicDim(lhs.value() * rhs.value());
    }
    // Zero: x * 0 = 0, 0 * x = 0
    if (rhs.is_concrete() && rhs.value() == 0) return SymbolicDim(int64_t{0});
    if (lhs.is_concrete() && lhs.value() == 0) return SymbolicDim(int64_t{0});
    // Identity: x * 1 = x, 1 * x = x
    if (rhs.is_concrete() && rhs.value() == 1) return lhs;
    if (lhs.is_concrete() && lhs.value() == 1) return rhs;
    return SymbolicDim::expr(ExprOp::Mul, lhs, rhs);
}

inline auto operator/(const SymbolicDim& lhs, const SymbolicDim& rhs) -> SymbolicDim {
    // Constant folding
    if (lhs.is_concrete() && rhs.is_concrete()) {
        if (rhs.value() == 0) {
            throw std::runtime_error("Division by zero in symbolic dimension arithmetic");
        }
        return SymbolicDim(lhs.value() / rhs.value());
    }
    // Identity: x / 1 = x
    if (rhs.is_concrete() && rhs.value() == 1) return lhs;
    // Zero numerator: 0 / x = 0
    if (lhs.is_concrete() && lhs.value() == 0) return SymbolicDim(int64_t{0});
    return SymbolicDim::expr(ExprOp::Div, lhs, rhs);
}

// ============================================================================
// SymbolicShape
// ============================================================================

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
    SymbolicShape() = default;

    SymbolicShape(std::initializer_list<SymbolicDim> dims)
        : dims_(dims) {}

    explicit SymbolicShape(std::vector<SymbolicDim> dims)
        : dims_(std::move(dims)) {}

    static auto from_concrete(const std::vector<int64_t>& concrete_shape) -> SymbolicShape {
        std::vector<SymbolicDim> dims;
        dims.reserve(concrete_shape.size());
        for (auto d : concrete_shape) {
            dims.emplace_back(d);
        }
        return SymbolicShape(std::move(dims));
    }

    [[nodiscard]] auto rank() const -> size_t { return dims_.size(); }
    [[nodiscard]] auto empty() const -> bool { return dims_.empty(); }

    auto operator[](size_t idx) const -> const SymbolicDim& { return dims_[idx]; }
    auto operator[](size_t idx) -> SymbolicDim& { return dims_[idx]; }

    [[nodiscard]] auto at(size_t idx) const -> const SymbolicDim& { return dims_.at(idx); }
    auto at(size_t idx) -> SymbolicDim& { return dims_.at(idx); }

    [[nodiscard]] auto has_symbolic_dims() const -> bool {
        for (const auto& d : dims_) {
            if (!d.is_concrete()) return true;
        }
        return false;
    }

    [[nodiscard]] auto is_fully_concrete() const -> bool {
        return !has_symbolic_dims();
    }

    [[nodiscard]] auto to_concrete() const -> std::vector<int64_t> {
        std::vector<int64_t> result;
        result.reserve(dims_.size());
        for (const auto& d : dims_) {
            result.push_back(d.value());
        }
        return result;
    }

    auto push_back(SymbolicDim dim) -> void { dims_.push_back(std::move(dim)); }

    auto insert(size_t pos, SymbolicDim dim) -> void {
        dims_.insert(dims_.begin() + static_cast<ptrdiff_t>(pos), std::move(dim));
    }

    auto erase(size_t pos) -> void {
        dims_.erase(dims_.begin() + static_cast<ptrdiff_t>(pos));
    }

    auto begin() const { return dims_.begin(); }
    auto end() const { return dims_.end(); }
    auto begin() { return dims_.begin(); }
    auto end() { return dims_.end(); }

    [[nodiscard]] auto dims() const -> const std::vector<SymbolicDim>& { return dims_; }

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

    auto operator==(const SymbolicShape& other) const -> bool {
        return dims_ == other.dims_;
    }

    auto operator!=(const SymbolicShape& other) const -> bool {
        return dims_ != other.dims_;
    }

private:
    std::vector<SymbolicDim> dims_;
};

// ============================================================================
// Symbolic shape utilities
// ============================================================================

/**
 * @brief Broadcast two symbolic shapes (NumPy-style).
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

        if (dim1.is_concrete() && dim2.is_concrete()) {
            int64_t v1 = dim1.value();
            int64_t v2 = dim2.value();
            if (v1 == v2 || v1 == 1 || v2 == 1) {
                result[ndim - 1 - i] = SymbolicDim::concrete(std::max(v1, v2));
            } else {
                throw std::runtime_error("Symbolic shapes are not broadcastable");
            }
        }
        else if (dim1.is_concrete() && dim1.value() == 1) {
            result[ndim - 1 - i] = dim2;
        }
        else if (dim2.is_concrete() && dim2.value() == 1) {
            result[ndim - 1 - i] = dim1;
        }
        else if (dim1 == dim2) {
            result[ndim - 1 - i] = dim1;
        }
        else {
            result[ndim - 1 - i] = SymbolicDim::symbolic(
                "broadcast(" + dim1.to_string() + ", " + dim2.to_string() + ")");
        }
    }

    return SymbolicShape(std::move(result));
}

// ============================================================================
// SymbolicShapeEnvironment
// ============================================================================

/**
 * @brief Environment for binding symbolic dimension names to concrete values at runtime.
 *
 * Supports resolving both named symbols and expression trees by recursive evaluation.
 *
 * @code
 * SymbolicShapeEnvironment env;
 * env.bind("batch", 32);
 *
 * auto expr = SymbolicDim::symbolic("batch") + SymbolicDim::concrete(10);
 * int64_t result = env.resolve(expr);  // 42
 * @endcode
 */
class SymbolicShapeEnvironment {
public:
    void bind(const std::string& name, int64_t value) {
        bindings_[name] = value;
    }

    void unbind(const std::string& name) {
        bindings_.erase(name);
    }

    void clear() {
        bindings_.clear();
    }

    auto is_bound(const std::string& name) const -> bool {
        return bindings_.find(name) != bindings_.end();
    }

    auto get(const std::string& name) const -> int64_t {
        auto it = bindings_.find(name);
        if (it == bindings_.end()) {
            throw std::runtime_error("SymbolicShapeEnvironment: unbound symbol '" + name + "'");
        }
        return it->second;
    }

    /**
     * @brief Resolve a symbolic dimension to a concrete value.
     *
     * Handles concrete values (passthrough), named symbols (lookup),
     * and expression trees (recursive evaluation).
     */
    auto resolve(const SymbolicDim& dim) const -> int64_t {
        if (dim.is_concrete()) {
            return dim.value();
        }
        if (dim.is_named_symbol()) {
            return get(dim.name());
        }
        // Expression: recursively evaluate
        auto& e = dim.expr_node();
        int64_t l = resolve(e.lhs);
        int64_t r = resolve(e.rhs);
        switch (e.op) {
            case ExprOp::Add: return l + r;
            case ExprOp::Sub: return l - r;
            case ExprOp::Mul: return l * r;
            case ExprOp::Div:
                if (r == 0) throw std::runtime_error("Division by zero resolving symbolic expression");
                return l / r;
        }
        return 0;  // unreachable
    }

    auto resolve(const SymbolicShape& shape) const -> std::vector<int64_t> {
        std::vector<int64_t> result;
        result.reserve(shape.rank());
        for (size_t i = 0; i < shape.rank(); ++i) {
            result.push_back(resolve(shape[i]));
        }
        return result;
    }

    /**
     * @brief Check if a single dimension can be fully resolved.
     */
    auto can_resolve_dim(const SymbolicDim& dim) const -> bool {
        if (dim.is_concrete()) return true;
        if (dim.is_named_symbol()) return is_bound(dim.name());
        // Expression: both children must be resolvable
        auto& e = dim.expr_node();
        return can_resolve_dim(e.lhs) && can_resolve_dim(e.rhs);
    }

    /**
     * @brief Check if all symbolic dimensions in a shape are bound.
     */
    auto can_resolve(const SymbolicShape& shape) const -> bool {
        for (size_t i = 0; i < shape.rank(); ++i) {
            if (!can_resolve_dim(shape[i])) return false;
        }
        return true;
    }

    auto size() const -> size_t { return bindings_.size(); }

    auto bindings() const -> const std::unordered_map<std::string, int64_t>& {
        return bindings_;
    }

private:
    std::unordered_map<std::string, int64_t> bindings_;
};

} // namespace jit
} // namespace tenzor
