/**
 * @file metrics.hpp
 * @brief Convenience header for MetricsRegistry
 *
 * MetricsRegistry is defined in server.hpp as part of the unified serving
 * infrastructure. This header exists so that downstream code can include
 * the component by logical name without depending on the monolithic header
 * path directly.
 */

#pragma once

// MetricsRegistry is defined in server.hpp.
#include "server.hpp"
