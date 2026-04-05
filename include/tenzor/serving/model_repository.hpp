/**
 * @file model_repository.hpp
 * @brief Convenience header for ModelRepository
 *
 * ModelRepository is defined in server.hpp as part of the unified serving
 * infrastructure. This header exists so that downstream code can include
 * the component by logical name without depending on the monolithic header
 * path directly.
 */

#pragma once

// ModelRepository is defined in server.hpp.
#include "server.hpp"
