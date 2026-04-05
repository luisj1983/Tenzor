/**
 * @file dynamic_batcher.hpp
 * @brief Convenience header for DynamicBatcher
 *
 * DynamicBatcher is defined in server.hpp as part of the unified serving
 * infrastructure. This header exists so that downstream code can include
 * the component by logical name without depending on the monolithic header
 * path directly.
 */

#pragma once

// DynamicBatcher is defined in server.hpp.
#include "server.hpp"
