#pragma once

#include <vector>
#include <optional>
#include "../core/tensor.hpp"
#include "../core/dtype.hpp"
#include "../core/device.hpp"

namespace tenzor {

// Tensor creation operations

// Create tensor filled with zeros
auto zeros(std::vector<int64_t> shape,
          DType dtype = DType::Float32,
          Device device = Device::cpu()) -> Tensor;

// Create tensor filled with ones
auto ones(std::vector<int64_t> shape,
         DType dtype = DType::Float32,
         Device device = Device::cpu()) -> Tensor;

// Create tensor filled with specific value
auto full(std::vector<int64_t> shape,
         float value,
         DType dtype = DType::Float32,
         Device device = Device::cpu()) -> Tensor;

// Create uninitialized tensor
auto empty(std::vector<int64_t> shape,
          DType dtype = DType::Float32,
          Device device = Device::cpu()) -> Tensor;

// Create tensor with random values (uniform distribution)
auto rand(std::vector<int64_t> shape,
         DType dtype = DType::Float32,
         Device device = Device::cpu()) -> Tensor;

// Create tensor with random values (normal distribution)
auto randn(std::vector<int64_t> shape,
          DType dtype = DType::Float32,
          Device device = Device::cpu()) -> Tensor;

// Create range of values
auto arange(float start, float end, float step = 1.0f,
           DType dtype = DType::Float32,
           Device device = Device::cpu()) -> Tensor;

// Create linearly spaced values
auto linspace(float start, float end, int64_t steps,
             DType dtype = DType::Float32,
             Device device = Device::cpu()) -> Tensor;

// Create identity matrix
auto eye(int64_t n, std::optional<int64_t> m = std::nullopt,
        DType dtype = DType::Float32,
        Device device = Device::cpu()) -> Tensor;

// Create tensor from data
template<typename T>
auto from_data(const T* data, std::vector<int64_t> shape,
              Device device = Device::cpu()) -> Tensor;

// Create tensor like another tensor
auto zeros_like(const Tensor& tensor) -> Tensor;
auto ones_like(const Tensor& tensor) -> Tensor;
auto rand_like(const Tensor& tensor) -> Tensor;
auto randn_like(const Tensor& tensor) -> Tensor;

} // namespace tenzor
