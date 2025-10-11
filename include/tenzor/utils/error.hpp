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
