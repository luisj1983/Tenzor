#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <source_location>
#include <format>

namespace tenzor {

// Base exception class
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

// Specific exception types
class DeviceException : public TenzorException {
    using TenzorException::TenzorException;
};

class DTypeException : public TenzorException {
    using TenzorException::TenzorException;
};

class ShapeException : public TenzorException {
    using TenzorException::TenzorException;
};

class BackendException : public TenzorException {
    using TenzorException::TenzorException;
};

class MemoryException : public TenzorException {
    using TenzorException::TenzorException;
};

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
