/**
 * @file lite_tensor.hpp
 * @brief Convenience header for LiteTensor
 *
 * LiteTensor is defined in runtime.hpp as part of the core lite runtime.
 * This header exists so that downstream code can include <tenzor/lite/lite_tensor.hpp>
 * when only the tensor type is needed, without having to know the actual location.
 */

#pragma once

#include "runtime.hpp"
