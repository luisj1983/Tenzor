/**
 * @file error.hpp
 * @brief Exception classes and error handling
 *
 * Provides hierarchy of exception classes for different error categories,
 * with automatic source location tracking for debugging.
 */

#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <source_location>
#include <format>

namespace tenzor {

/**
 * @brief Base exception class for all Tenzor errors
 *
 * Automatically captures source location (file, line, function) where exception
 * was thrown for easier debugging.
 *
 * All derived exceptions inherit source location tracking.
 *
 * @code
 * throw TenzorException("Invalid tensor shape");
 * // Output: "tensor.cpp:42 in compute(): Invalid tensor shape"
 * @endcode
 */
class TenzorException : public std::runtime_error {
public:
    explicit TenzorException(const std::string& message,
                            const std::source_location& location = std::source_location::current())
        : std::runtime_error(format_message(message, location)),
          location_(location) {}

    auto location() const -> const std::source_location& {
        return location_;
    }

private:
    std::source_location location_;

    static auto format_message(const std::string& message,
                              const std::source_location& location) -> std::string {
        return std::format("{}:{} in {}: {}",
                          location.file_name(),
                          location.line(),
                          location.function_name(),
                          message);
    }
};

/** @brief Device-related errors (CUDA unavailable, invalid device ID, etc.) */
class DeviceException : public TenzorException {
    using TenzorException::TenzorException;
};

/** @brief Data type errors (unsupported dtype, type mismatch, etc.) */
class DTypeException : public TenzorException {
    using TenzorException::TenzorException;
};

/** @brief Shape mismatch and dimension errors */
class ShapeException : public TenzorException {
    using TenzorException::TenzorException;
};

/** @brief Backend and computation errors */
class BackendException : public TenzorException {
    using TenzorException::TenzorException;
};

/** @brief Memory allocation and management errors */
class MemoryException : public TenzorException {
    using TenzorException::TenzorException;
};

/** @brief Automatic differentiation errors */
class AutogradException : public TenzorException {
    using TenzorException::TenzorException;
};

// =====================================================================
// Python-parity exception types.
//
// These mirror Python's stdlib exception hierarchy so that pybind11 can
// translate them to the corresponding Python exception classes via
// py::register_exception in the bindings layer. User-facing throw sites
// in indexing/dtype-validation/range-check paths should use these to
// give Python users the idiomatic exception type rather than a generic
// RuntimeError.
// =====================================================================

/**
 * @brief Index out of range. Translated to Python's ``IndexError``.
 */
class IndexError : public TenzorException {
    using TenzorException::TenzorException;
};

/**
 * @brief Value out of valid range / invalid argument value. Translated to
 *        Python's ``ValueError``. Use for "argument was in the wrong domain
 *        but was the right type" — e.g. negative ``num_classes``, empty
 *        tensor where non-empty was required.
 */
class ValueError : public TenzorException {
    using TenzorException::TenzorException;
};

/**
 * @brief Type mismatch from a user-API perspective. Translated to Python's
 *        ``TypeError``. Distinct from the internal ``DTypeException``, which
 *        is for backend precondition violations; ``TypeError`` is what a
 *        Python user sees when passing the wrong tensor kind to an op.
 */
class TypeError : public TenzorException {
    using TenzorException::TenzorException;
};

/**
 * @brief Operation not implemented for the given input. Translated to
 *        Python's ``NotImplementedError``.
 */
class NotImplementedError : public TenzorException {
    using TenzorException::TenzorException;
};

/**
 * @brief Generic runtime error from a Tenzor op. Translated to Python's
 *        ``RuntimeError`` (which is also pybind11's default mapping for any
 *        unrecognised ``std::runtime_error``-derived exception).
 */
class RuntimeError : public TenzorException {
    using TenzorException::TenzorException;
};

/**
 * @brief Raised when a Function's backward() is invoked on an op whose
 *        forward is intrinsically non-differentiable (histogram counts,
 *        bin assignments, sort positions, discrete samples).  Audit E.7
 *        replaces the previous silent "no backward registered" behaviour
 *        — those ops now route through Function wrappers whose backward()
 *        throws this typed exception with the offending op name.
 *
 *        Recoverable contract: callers that want to support such ops in
 *        a gradient graph must wrap them in a custom Function that
 *        explicitly provides a surrogate gradient (e.g. straight-through
 *        estimator, Gumbel-softmax, etc.).
 */
class NonDifferentiable : public TenzorException {
    using TenzorException::TenzorException;
};

namespace error {

/**
 * @brief Raised when a distribution method (e.g. cdf, icdf, mean) is not
 *        mathematically defined for that distribution.
 *
 * Audit item E.5: distributions like Categorical, Multinomial, Dirichlet
 * have no canonical scalar order over their support and therefore no
 * meaningful cdf/icdf. These methods raise this typed exception with a
 * message that names the distribution and the method, rather than a
 * generic ``std::runtime_error``.
 */
class DistributionMethodUndefined : public TenzorException {
    using TenzorException::TenzorException;
};

} // namespace error

// Error checking macros
#define TENZOR_CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            throw ::tenzor::TenzorException(message); \
        } \
    } while (0)

#define TENZOR_CHECK_DEVICE(condition, message) \
    do { \
        if (!(condition)) { \
            throw ::tenzor::DeviceException(message); \
        } \
    } while (0)

#define TENZOR_CHECK_DTYPE(condition, message) \
    do { \
        if (!(condition)) { \
            throw ::tenzor::DTypeException(message); \
        } \
    } while (0)

#define TENZOR_CHECK_SHAPE(condition, message) \
    do { \
        if (!(condition)) { \
            throw ::tenzor::ShapeException(message); \
        } \
    } while (0)

} // namespace tenzor
