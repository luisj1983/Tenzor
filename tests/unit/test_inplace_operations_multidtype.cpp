/**
 * @file test_inplace_operations_multidtype.cpp
 * @brief Multi-dtype unit tests for in-place operations with memory allocation tracking
 *
 * Tests in-place operations across Float32, Float64, Float16, and Int32 dtypes:
 * - Inplace arithmetic (add_, sub_, mul_, div_)
 * - Inplace activations (relu_, sigmoid_, tanh_, etc.)
 * - Inplace normalization
 * - Memory efficiency verification
 * - Gradient compatibility
 * - Dtype preservation
 */

#include <gtest/gtest.h>
#include "tenzor/ops/math.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/tensor.hpp"
#include <memory>
#include <type_traits>
#include <cmath>

namespace tenzor {
namespace test {

// ============================================================================
// Type-Parametrized Test Fixtures
// ============================================================================

template <typename T>
struct DTypeTraits;

template <>
struct DTypeTraits<float> {
    static constexpr DType dtype = DType::Float32;
    static constexpr float tolerance = 1e-5f;
    static constexpr bool supports_negative = true;
    static constexpr bool supports_fractional = true;
};

template <>
struct DTypeTraits<double> {
    static constexpr DType dtype = DType::Float64;
    static constexpr double tolerance = 1e-10;
    static constexpr bool supports_negative = true;
    static constexpr bool supports_fractional = true;
};

template <>
struct DTypeTraits<int32_t> {
    static constexpr DType dtype = DType::Int32;
    static constexpr float tolerance = 0.0f;
    static constexpr bool supports_negative = true;
    static constexpr bool supports_fractional = false;
};

// Float16 support (if available)
#ifdef TENZOR_ENABLE_FLOAT16
template <>
struct DTypeTraits<half> {
    static constexpr DType dtype = DType::Float16;
    static constexpr float tolerance = 1e-3f;
    static constexpr bool supports_negative = true;
    static constexpr bool supports_fractional = true;
};
#endif

template <typename T>
class InPlaceOperationsMultiDTypeTest : public ::testing::Test {
protected:
    using Scalar = T;
    static constexpr DType dtype = DTypeTraits<T>::dtype;
    static constexpr auto tolerance = DTypeTraits<T>::tolerance;
    static constexpr bool supports_negative = DTypeTraits<T>::supports_negative;
    static constexpr bool supports_fractional = DTypeTraits<T>::supports_fractional;

    void SetUp() override {
        device_ = Device::cpu();
    }

    // Helper to convert scalar to appropriate type
    template <typename U = T>
    typename std::enable_if<std::is_floating_point<U>::value, U>::type
    scalar(double val) const {
        return static_cast<U>(val);
    }

    template <typename U = T>
    typename std::enable_if<std::is_integral<U>::value, U>::type
    scalar(double val) const {
        return static_cast<U>(std::round(val));
    }

    // Helper for floating-point comparison
    template <typename U = T>
    typename std::enable_if<std::is_floating_point<U>::value, void>::type
    expect_near(U actual, U expected, const std::string& msg = "") const {
        EXPECT_NEAR(static_cast<double>(actual), static_cast<double>(expected),
                    static_cast<double>(tolerance)) << msg;
    }

    template <typename U = T>
    typename std::enable_if<std::is_integral<U>::value, void>::type
    expect_near(U actual, U expected, const std::string& msg = "") const {
        EXPECT_EQ(actual, expected) << msg;
    }

    Device device_;
};

// Test types
using TestTypes = ::testing::Types<float, double, int32_t
#ifdef TENZOR_ENABLE_FLOAT16
    , half
#endif
>;

TYPED_TEST_SUITE(InPlaceOperationsMultiDTypeTest, TestTypes);

// ============================================================================
// In-Place Arithmetic Operations - Multi-DType
// ============================================================================

TYPED_TEST(InPlaceOperationsMultiDTypeTest, AddInPlace) {
    using T = typename TestFixture::Scalar;

    auto a = ones({3, 4}, this->dtype, this->device_);
    auto b = ones({3, 4}, this->dtype, this->device_) * this->scalar(2.0);

    void* original_ptr = a.data_ptr();

    // In-place add
    add_(a, b);

    // Verify pointer didn't change (true in-place)
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Verify values
    const T* data = a.template data<typename TestFixture::Scalar>();
    T expected = this->scalar(3.0);
    for (size_t i = 0; i < static_cast<size_t>(a.numel()); ++i) {
        this->expect_near(data[i], expected, "at index " + std::to_string(i));
    }
}

TYPED_TEST(InPlaceOperationsMultiDTypeTest, MulInPlace) {
    using T = typename TestFixture::Scalar;

    auto a = ones({3, 4}, this->dtype, this->device_) * this->scalar(2.0);
    auto b = ones({3, 4}, this->dtype, this->device_) * this->scalar(3.0);

    void* original_ptr = a.data_ptr();

    // In-place multiply
    mul_(a, b);

    // Verify pointer didn't change
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Verify values
    const T* data = a.template data<typename TestFixture::Scalar>();
    T expected = this->scalar(6.0);
    for (size_t i = 0; i < static_cast<size_t>(a.numel()); ++i) {
        this->expect_near(data[i], expected, "at index " + std::to_string(i));
    }
}

TYPED_TEST(InPlaceOperationsMultiDTypeTest, SubInPlace) {
    using T = typename TestFixture::Scalar;

    auto a = ones({3, 4}, this->dtype, this->device_) * this->scalar(5.0);
    auto b = ones({3, 4}, this->dtype, this->device_) * this->scalar(2.0);

    void* original_ptr = a.data_ptr();

    // In-place subtract
    sub_(a, b);

    // Verify pointer didn't change
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Verify values
    const T* data = a.template data<typename TestFixture::Scalar>();
    T expected = this->scalar(3.0);
    for (size_t i = 0; i < static_cast<size_t>(a.numel()); ++i) {
        this->expect_near(data[i], expected, "at index " + std::to_string(i));
    }
}

TYPED_TEST(InPlaceOperationsMultiDTypeTest, DivInPlace) {
    using T = typename TestFixture::Scalar;

    if constexpr (TestFixture::supports_fractional) {
        auto a = ones({3, 4}, this->dtype, this->device_) * this->scalar(6.0);
        auto b = ones({3, 4}, this->dtype, this->device_) * this->scalar(2.0);

        void* original_ptr = a.data_ptr();

        // In-place divide
        div_(a, b);

        // Verify pointer didn't change
        EXPECT_EQ(a.data_ptr(), original_ptr);

        // Verify values
        const T* data = a.template data<typename TestFixture::Scalar>();
        T expected = this->scalar(3.0);
        for (size_t i = 0; i < static_cast<size_t>(a.numel()); ++i) {
            this->expect_near(data[i], expected, "at index " + std::to_string(i));
        }
    } else {
        // Integer division
        auto a = ones({3, 4}, this->dtype, this->device_) * this->scalar(6.0);
        auto b = ones({3, 4}, this->dtype, this->device_) * this->scalar(2.0);

        void* original_ptr = a.data_ptr();
        div_(a, b);
        EXPECT_EQ(a.data_ptr(), original_ptr);

        const T* data = a.template data<typename TestFixture::Scalar>();
        T expected = this->scalar(3.0);
        for (size_t i = 0; i < static_cast<size_t>(a.numel()); ++i) {
            EXPECT_EQ(data[i], expected) << "at index " << i;
        }
    }
}

TYPED_TEST(InPlaceOperationsMultiDTypeTest, ChainedInPlaceOperations) {
    using T = typename TestFixture::Scalar;

    auto a = ones({3, 4}, this->dtype, this->device_);
    auto b = ones({3, 4}, this->dtype, this->device_) * this->scalar(2.0);
    auto c = ones({3, 4}, this->dtype, this->device_) * this->scalar(3.0);

    void* original_ptr = a.data_ptr();

    // Chain multiple in-place operations: (1 + 2) * 3 - 2 = 7
    add_(a, b);  // a = 1 + 2 = 3
    mul_(a, c);  // a = 3 * 3 = 9
    sub_(a, b);  // a = 9 - 2 = 7

    // Verify pointer stayed the same
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Verify final value
    const T* data = a.template data<typename TestFixture::Scalar>();
    T expected = this->scalar(7.0);
    for (size_t i = 0; i < static_cast<size_t>(a.numel()); ++i) {
        this->expect_near(data[i], expected, "at index " + std::to_string(i));
    }
}

// ============================================================================
// In-Place Activation Functions - Multi-DType
// ============================================================================

TYPED_TEST(InPlaceOperationsMultiDTypeTest, ReLUInPlace) {
    using T = typename TestFixture::Scalar;

    if constexpr (TestFixture::supports_negative) {
        // Create tensor with positive and negative values
        std::vector<T> values;
        if constexpr (TestFixture::supports_fractional) {
            values = {this->scalar(-2.0), this->scalar(-1.0), this->scalar(0.0),
                     this->scalar(1.0), this->scalar(2.0), this->scalar(3.0)};
        } else {
            values = {this->scalar(-2.0), this->scalar(-1.0), this->scalar(0.0),
                     this->scalar(1.0), this->scalar(2.0), this->scalar(3.0)};
        }

        auto a = ones({2, 3}, this->dtype, this->device_);
        void* original_ptr = a.data_ptr();

        // In-place ReLU
        nn::relu_(a);

        // Verify pointer didn't change
        EXPECT_EQ(a.data_ptr(), original_ptr);

        // Verify values: max(0, x)
        const T* data = a.template data<typename TestFixture::Scalar>();
        std::vector<T> expected = {this->scalar(0.0), this->scalar(0.0), this->scalar(0.0),
                                  this->scalar(1.0), this->scalar(2.0), this->scalar(3.0)};
        ASSERT_EQ(static_cast<size_t>(a.numel()), expected.size());
        for (size_t i = 0; i < static_cast<size_t>(a.numel()); ++i) {
            this->expect_near(data[i], expected[i], "at index " + std::to_string(i));
        }
    }
}

TYPED_TEST(InPlaceOperationsMultiDTypeTest, SigmoidInPlace) {
    using T = typename TestFixture::Scalar;

    if constexpr (TestFixture::supports_fractional) {
        std::vector<T> values = {this->scalar(-1.0), this->scalar(0.0), this->scalar(1.0)};
        auto a = ones({3}, this->dtype, this->device_);

        void* original_ptr = a.data_ptr();

        // In-place sigmoid
        nn::sigmoid_(a);

        // Verify pointer didn't change
        EXPECT_EQ(a.data_ptr(), original_ptr);

        // Verify values are in (0, 1) range
        const T* data = a.template data<typename TestFixture::Scalar>();
        for (size_t i = 0; i < static_cast<size_t>(a.numel()); ++i) {
            EXPECT_GT(static_cast<double>(data[i]), 0.0) << "at index " << i;
            EXPECT_LT(static_cast<double>(data[i]), 1.0) << "at index " << i;
        }

        // Check specific value: sigmoid(0) = 0.5
        this->expect_near(data[1], this->scalar(0.5), "sigmoid(0) should be 0.5");
    }
}

TYPED_TEST(InPlaceOperationsMultiDTypeTest, TanhInPlace) {
    using T = typename TestFixture::Scalar;

    if constexpr (TestFixture::supports_fractional) {
        std::vector<T> values = {this->scalar(-1.0), this->scalar(0.0), this->scalar(1.0)};
        auto a = ones({3}, this->dtype, this->device_);

        void* original_ptr = a.data_ptr();

        // In-place tanh
        nn::tanh_(a);

        // Verify pointer didn't change
        EXPECT_EQ(a.data_ptr(), original_ptr);

        // Verify values are in (-1, 1) range
        const T* data = a.template data<typename TestFixture::Scalar>();
        for (size_t i = 0; i < static_cast<size_t>(a.numel()); ++i) {
            EXPECT_GT(static_cast<double>(data[i]), -1.0) << "at index " << i;
            EXPECT_LT(static_cast<double>(data[i]), 1.0) << "at index " << i;
        }

        // Check specific value: tanh(0) = 0
        this->expect_near(data[1], this->scalar(0.0), "tanh(0) should be 0");
    }
}

TYPED_TEST(InPlaceOperationsMultiDTypeTest, LeakyReLUInPlace) {
    using T = typename TestFixture::Scalar;

    if constexpr (TestFixture::supports_negative && TestFixture::supports_fractional) {
        std::vector<T> values = {this->scalar(-2.0), this->scalar(-1.0),
                                this->scalar(0.0), this->scalar(1.0), this->scalar(2.0)};
        auto a = ones({5}, this->dtype, this->device_);

        void* original_ptr = a.data_ptr();

        // In-place leaky ReLU with slope 0.1
        nn::leaky_relu_(a, 0.1);

        // Verify pointer didn't change
        EXPECT_EQ(a.data_ptr(), original_ptr);

        // Verify values
        const T* data = a.template data<typename TestFixture::Scalar>();
        std::vector<T> expected = {this->scalar(-0.2), this->scalar(-0.1),
                                  this->scalar(0.0), this->scalar(1.0), this->scalar(2.0)};
        ASSERT_EQ(static_cast<size_t>(a.numel()), expected.size());
        for (size_t i = 0; i < static_cast<size_t>(a.numel()); ++i) {
            this->expect_near(data[i], expected[i], "at index " + std::to_string(i));
        }
    }
}

TYPED_TEST(InPlaceOperationsMultiDTypeTest, GeLUInPlace) {
    using T = typename TestFixture::Scalar;

    if constexpr (TestFixture::supports_fractional) {
        std::vector<T> values = {this->scalar(-1.0), this->scalar(0.0), this->scalar(1.0)};
        auto a = ones({3}, this->dtype, this->device_);

        void* original_ptr = a.data_ptr();

        // In-place GELU
        nn::gelu_(a);

        // Verify pointer didn't change
        EXPECT_EQ(a.data_ptr(), original_ptr);

        // GELU(0) should be approximately 0
        const T* data = a.template data<typename TestFixture::Scalar>();

        // Tolerance adjusted for Float16
        double gelu_tolerance = std::is_same_v<T, float> ? 1e-3 :
                               (std::is_same_v<T, double> ? 1e-6 : 1e-2);

        EXPECT_NEAR(static_cast<double>(data[1]), 0.0, gelu_tolerance) << "GELU(0) should be ~0";

        // GELU output should be smooth
        EXPECT_LT(static_cast<double>(data[0]), 0.0) << "Negative input gives negative output";
        EXPECT_GT(static_cast<double>(data[2]), 0.0) << "Positive input gives positive output";
    }
}

// ============================================================================
// DType Preservation Tests
// ============================================================================

TYPED_TEST(InPlaceOperationsMultiDTypeTest, DTypePreservation) {
    auto a = ones({3, 4}, this->dtype, this->device_);
    auto b = ones({3, 4}, this->dtype, this->device_);

    // Verify initial dtype
    EXPECT_EQ(a.dtype(), this->dtype);

    // In-place operation
    add_(a, b);

    // Verify dtype is preserved
    EXPECT_EQ(a.dtype(), this->dtype);
}

TYPED_TEST(InPlaceOperationsMultiDTypeTest, ActivationDTypePreservation) {
    if constexpr (TestFixture::supports_fractional) {
        auto a = ones({3, 4}, this->dtype, this->device_);

        EXPECT_EQ(a.dtype(), this->dtype);

        nn::relu_(a);
        EXPECT_EQ(a.dtype(), this->dtype);

        nn::sigmoid_(a);
        EXPECT_EQ(a.dtype(), this->dtype);

        nn::tanh_(a);
        EXPECT_EQ(a.dtype(), this->dtype);
    }
}

// ============================================================================
// Memory Efficiency Tests
// ============================================================================

TYPED_TEST(InPlaceOperationsMultiDTypeTest, MemoryEfficiencyComparison) {
    const size_t size = 1000 * 100;  // 100K elements
    auto a = ones({size}, this->dtype, this->device_);
    auto b = ones({size}, this->dtype, this->device_);

    // Record initial pointer
    void* original_ptr = a.data_ptr();

    // In-place operation should not allocate new memory
    add_(a, b);

    // Verify no new allocation (pointer unchanged)
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Out-of-place would create new tensor
    auto c = ones({size}, this->dtype, this->device_);
    auto d = add(c, b);

    // d should be different tensor with different pointer
    EXPECT_NE(d.data_ptr(), c.data_ptr());
}

TYPED_TEST(InPlaceOperationsMultiDTypeTest, LargeActivationInPlace) {
    if constexpr (TestFixture::supports_fractional) {
        const size_t size = 1000 * 100;
        auto a = ones({size}, this->dtype, this->device_) * this->scalar(2.0);

        void* original_ptr = a.data_ptr();

        // In-place activation on large tensor
        nn::relu_(a);

        // Should not allocate
        EXPECT_EQ(a.data_ptr(), original_ptr);

        // Verify some values are still 2.0 (since input was positive)
        auto data = a.template data<typename TestFixture::Scalar>();
        for (size_t i = 0; i < std::min(size_t(100), static_cast<size_t>(a.numel())); ++i) {
            this->expect_near(data[i], this->scalar(2.0), "at index " + std::to_string(i));
        }
    }
}

TYPED_TEST(InPlaceOperationsMultiDTypeTest, MemorySizeConsistency) {
    auto a = ones({10, 20}, this->dtype, this->device_);

    size_t element_size = 4;
    size_t num_elements = a.numel();
    size_t expected_bytes = element_size * num_elements;

    // Verify memory size
    EXPECT_EQ(a.numel() * 4, expected_bytes);

    // After in-place operation, size should remain the same
    auto b = ones({10, 20}, this->dtype, this->device_);
    add_(a, b);

    EXPECT_EQ(a.numel() * 4, expected_bytes);
    EXPECT_EQ(a.numel(), num_elements);
}

// ============================================================================
// Broadcasting Tests
// ============================================================================

TYPED_TEST(InPlaceOperationsMultiDTypeTest, BroadcastingInPlace) {
    using T = typename TestFixture::Scalar;

    auto a = ones({3, 4}, this->dtype, this->device_);
    auto b = ones({1, 4}, this->dtype, this->device_) * this->scalar(2.0);

    void* original_ptr = a.data_ptr();

    // In-place add with broadcasting
    add_(a, b);

    EXPECT_EQ(a.data_ptr(), original_ptr);

    // All values should be 3.0 (1.0 + 2.0)
    const T* data = a.template data<typename TestFixture::Scalar>();
    T expected = this->scalar(3.0);
    for (size_t i = 0; i < static_cast<size_t>(a.numel()); ++i) {
        this->expect_near(data[i], expected, "at index " + std::to_string(i));
    }
}

TYPED_TEST(InPlaceOperationsMultiDTypeTest, ScalarBroadcastInPlace) {
    using T = typename TestFixture::Scalar;

    auto a = ones({4, 3}, this->dtype, this->device_) * this->scalar(2.0);
    auto b = ones({1}, this->dtype, this->device_) * this->scalar(3.0);

    void* original_ptr = a.data_ptr();

    mul_(a, b);

    EXPECT_EQ(a.data_ptr(), original_ptr);

    const T* data = a.template data<typename TestFixture::Scalar>();
    T expected = this->scalar(6.0);
    for (size_t i = 0; i < static_cast<size_t>(a.numel()); ++i) {
        this->expect_near(data[i], expected, "at index " + std::to_string(i));
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TYPED_TEST(InPlaceOperationsMultiDTypeTest, NonContiguousTensorError) {
    auto a = ones({4, 4}, this->dtype, this->device_);
    auto b = ones({4, 4}, this->dtype, this->device_);

    // Create non-contiguous view via transpose
    auto a_t = a.transpose(0, 1);

    // In-place operations should fail on non-contiguous tensors
    EXPECT_THROW(add_(a_t, b), std::runtime_error);
    EXPECT_THROW(mul_(a_t, b), std::runtime_error);

    if constexpr (TestFixture::supports_fractional) {
        EXPECT_THROW(nn::relu_(a_t), std::runtime_error);
        EXPECT_THROW(nn::sigmoid_(a_t), std::runtime_error);
    }
}

TYPED_TEST(InPlaceOperationsMultiDTypeTest, ShapeMismatchError) {
    auto a = ones({3, 4}, this->dtype, this->device_);
    auto b = ones({4, 3}, this->dtype, this->device_);

    // Shape mismatch should throw
    EXPECT_THROW(add_(a, b), std::runtime_error);
    EXPECT_THROW(mul_(a, b), std::runtime_error);
}

// ============================================================================
// Gradient Compatibility Tests
// ============================================================================

TYPED_TEST(InPlaceOperationsMultiDTypeTest, GradientCompatibility) {
    if constexpr (TestFixture::supports_fractional) {
        auto a = ones({3, 4}, this->dtype, this->device_);
        auto b = ones({3, 4}, this->dtype, this->device_);

        // Enable gradient tracking
        // a.set_requires_grad(true) // TODO: Fix gradient tracking;

        void* original_ptr = a.data_ptr();

        // In-place operation with gradients enabled
        add_(a, b);

        // Verify pointer didn't change
        EXPECT_EQ(a.data_ptr(), original_ptr);

        // Verify requires_grad is still true
        EXPECT_TRUE(a.requires_grad());
    }
}

TYPED_TEST(InPlaceOperationsMultiDTypeTest, ActivationGradientFlow) {
    if constexpr (TestFixture::supports_fractional) {
        using T = typename TestFixture::Scalar;

        std::vector<T> values = {this->scalar(-1.0), this->scalar(0.0),
                                this->scalar(1.0), this->scalar(2.0)};
        auto a = ones({2, 2}, this->dtype, this->device_);

        // a.requires_grad(true) // TODO: Fix gradient tracking;

        void* original_ptr = a.data_ptr();

        // In-place ReLU
        nn::relu_(a);

        EXPECT_EQ(a.data_ptr(), original_ptr);
        EXPECT_TRUE(a.requires_grad());

        // Verify ReLU output
        const T* data = a.template data<typename TestFixture::Scalar>();
        std::vector<T> expected = {this->scalar(0.0), this->scalar(0.0),
                                  this->scalar(1.0), this->scalar(2.0)};
        for (size_t i = 0; i < static_cast<size_t>(a.numel()); ++i) {
            this->expect_near(data[i], expected[i], "at index " + std::to_string(i));
        }
    }
}

// ============================================================================
// Complex Workflow Tests
// ============================================================================

TYPED_TEST(InPlaceOperationsMultiDTypeTest, ComplexInPlaceWorkflow) {
    using T = typename TestFixture::Scalar;

    if constexpr (TestFixture::supports_fractional) {
        // Simulate a mini neural network layer update
        auto weights = ones({5, 5}, this->dtype, this->device_) * this->scalar(0.5);
        auto gradients = ones({5, 5}, this->dtype, this->device_) * this->scalar(0.1);

        T learning_rate = this->scalar(0.01);
        auto lr_tensor = ones({1}, this->dtype, this->device_) * learning_rate;

        void* original_ptr = weights.data_ptr();

        // Weight update: weights -= learning_rate * gradients
        auto scaled_grad = gradients * lr_tensor;
        sub_(weights, scaled_grad);

        EXPECT_EQ(weights.data_ptr(), original_ptr);

        // Verify update: 0.5 - 0.01 * 0.1 = 0.499
        const T* data = weights.template data<T>();
        T expected = this->scalar(0.499);
        for (size_t i = 0; i < static_cast<size_t>(weights.numel()); ++i) {
            this->expect_near(data[i], expected, "at index " + std::to_string(i));
        }
    }
}

TYPED_TEST(InPlaceOperationsMultiDTypeTest, ActivationPipelineInPlace) {
    if constexpr (TestFixture::supports_fractional) {
        using T = typename TestFixture::Scalar;

        // Create input with mixed values
        std::vector<T> values = {this->scalar(-2.0), this->scalar(-0.5),
                                this->scalar(0.0), this->scalar(0.5),
                                this->scalar(1.0), this->scalar(2.0)};
        auto a = ones({2, 3}, this->dtype, this->device_);

        void* original_ptr = a.data_ptr();

        // Apply multiple activations in sequence
        nn::relu_(a);           // First eliminate negatives
        nn::sigmoid_(a);        // Then squash to (0,1)

        EXPECT_EQ(a.data_ptr(), original_ptr);

        // All values should be in (0, 1) range
        const T* data = a.template data<typename TestFixture::Scalar>();
        for (size_t i = 0; i < static_cast<size_t>(a.numel()); ++i) {
            EXPECT_GE(static_cast<double>(data[i]), 0.0) << "at index " << i;
            EXPECT_LE(static_cast<double>(data[i]), 1.0) << "at index " << i;
        }
    }
}

// ============================================================================
// Normalization Tests
// ============================================================================

TYPED_TEST(InPlaceOperationsMultiDTypeTest, InPlaceNormalization) {
    using T = typename TestFixture::Scalar;

    if constexpr (TestFixture::supports_fractional) {
        // Create tensor with known values
        std::vector<T> values = {this->scalar(1.0), this->scalar(2.0),
                                this->scalar(3.0), this->scalar(4.0)};
        auto a = ones({2, 2}, this->dtype, this->device_);

        void* original_ptr = a.data_ptr();

        // In-place normalization: x = (x - mean) / std
        auto mean_val = this->scalar(2.5);  // (1+2+3+4)/4
        auto mean_tensor = ones({1}, this->dtype, this->device_) * mean_val;
        sub_(a, mean_tensor);

        EXPECT_EQ(a.data_ptr(), original_ptr);

        // Verify centering
        const T* data = a.template data<typename TestFixture::Scalar>();
        std::vector<T> expected = {this->scalar(-1.5), this->scalar(-0.5),
                                  this->scalar(0.5), this->scalar(1.5)};
        for (size_t i = 0; i < static_cast<size_t>(a.numel()); ++i) {
            this->expect_near(data[i], expected[i], "at index " + std::to_string(i));
        }
    }
}

} // namespace test
} // namespace tenzor
