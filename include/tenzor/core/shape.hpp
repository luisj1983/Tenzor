#pragma once

#include <vector>
#include <cstdint>
#include <span>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace tenzor {

// Shape utilities
class Shape {
public:
    using size_type = int64_t;

    Shape() = default;
    explicit Shape(std::vector<size_type> dims) : dims_(std::move(dims)) {}

    // Access
    auto operator[](size_t idx) const -> size_type { return dims_[idx]; }
    auto at(size_t idx) const -> size_type { return dims_.at(idx); }
    auto size() const -> size_t { return dims_.size(); }
    auto data() const -> const size_type* { return dims_.data(); }
    auto begin() const { return dims_.begin(); }
    auto end() const { return dims_.end(); }

    // Modification
    auto push_back(size_type dim) -> void { dims_.push_back(dim); }
    auto resize(size_t n) -> void { dims_.resize(n); }

    // Computation
    auto numel() const -> size_type {
        return std::accumulate(dims_.begin(), dims_.end(), size_type{1},
                             std::multiplies<size_type>{});
    }

    // Comparison
    auto operator==(const Shape& other) const -> bool {
        return dims_ == other.dims_;
    }

private:
    std::vector<size_type> dims_;
};

// Stride calculation
inline auto compute_strides(std::span<const int64_t> shape) -> std::vector<int64_t> {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) return strides;

    strides.back() = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// Broadcasting
inline auto broadcast_shapes(std::span<const int64_t> shape1,
                            std::span<const int64_t> shape2)
    -> std::vector<int64_t> {
    size_t ndim = std::max(shape1.size(), shape2.size());
    std::vector<int64_t> result(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        int64_t dim1 = i < shape1.size() ? shape1[shape1.size() - 1 - i] : 1;
        int64_t dim2 = i < shape2.size() ? shape2[shape2.size() - 1 - i] : 1;

        if (dim1 == dim2 || dim1 == 1 || dim2 == 1) {
            result[ndim - 1 - i] = std::max(dim1, dim2);
        } else {
            throw std::runtime_error("Shapes are not broadcastable");
        }
    }

    return result;
}

} // namespace tenzor
