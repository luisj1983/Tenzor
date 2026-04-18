/**
 * @file conv3d_autograd.hpp
 * @brief Factory that hands out a Conv3dBackward autograd node.
 *
 * Conv3dBackward is defined inside src/nn/layers/conv.cpp. This factory
 * lets the functional-style wrapper (F::conv3d, in functional.cpp) wire the
 * same backward graph without making the class a public header symbol.
 * Parallels conv_autograd.hpp which exposes Conv2dBackward similarly.
 *
 * Internal to nn/.
 */

#pragma once

#include <memory>
#include <vector>

#include "tenzor/autograd/function.hpp"
#include "tenzor/core/tensor.hpp"

namespace tenzor::nn::internal {

// Builds a Conv3dBackward bound to isotropic stride/padding/dilation,
// already populated with tensors_to_save (input, weight, optional bias).
// The caller still needs to set_input_variables() and set_next_functions().
auto make_conv3d_backward(int64_t stride, int64_t padding, int64_t dilation,
                          int64_t groups,
                          std::vector<::tenzor::Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function>;

// Same pattern for ConvTranspose2d (isotropic stride/padding/dilation).
auto make_conv_transpose2d_backward(int64_t stride, int64_t padding,
                                    int64_t output_padding, int64_t dilation,
                                    int64_t groups,
                                    std::vector<::tenzor::Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function>;

} // namespace tenzor::nn::internal
