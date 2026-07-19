#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>
#include <limits>

using namespace tenzor;

/**
 * @file test_backend_ops_parameterized.cpp
 * @brief Parameterized tests that run on ALL backends (CPU, CUDA, OneAPI, ROCm, Vulkan)
 *
 * This ensures feature parity across all backends by running the same tests
 * with different device configurations.
 */

// ============================================================================
// Backend Configuration
// ============================================================================

struct BackendConfig {
    std::string name;
    Device::Type type;
    int device_id;
    bool is_available;

    std::string ToString() const {
        return name + "_" + std::to_string(device_id);
    }
};

// Required for gtest_discover_tests to show human-readable test names
void PrintTo(const BackendConfig& param, std::ostream* os) {
    *os << param.ToString();
}

// Helper to check if backend is available
// Note: For ROCm, we use a safer check that doesn't launch kernels
// because some ROCm configurations may crash on kernel launch
bool is_backend_available(Device::Type type) {
    try {
        auto backend = backend_registry().get_backend(type);
        if (!backend) {
            return false;
        }

        // Check if the backend reports having devices
        if (backend->device_count() == 0) {
            return false;
        }

        // For ROCm, check environment variable to opt-in (avoids crashes on broken setups)
        if (type == Device::Type::ROCm) {
            const char* enable_rocm = std::getenv("TENZOR_TEST_ROCM");
            if (!enable_rocm || std::string(enable_rocm) != "1") {
                return false;  // Skip ROCm tests unless explicitly enabled
            }
        }

        // For backends that might have driver issues, don't test with actual tensor ops
        // Just check that the backend is registered and has devices
        if (type == Device::Type::ROCm) {
            return true;  // Trust the device count, don't launch kernels
        }

        // For OneAPI, trust the device count — SYCL queue creation can segfault
        // during GTest's static initialization (test discovery phase)
        if (type == Device::Type::OneAPI) {
            return true;
        }

        // For CUDA and other backends, verify with actual tensor creation
        auto device = Device(type, 0);
        auto test_tensor = ones({2, 2}, DType::Float32, device);
        return true;
    } catch (...) {
        return false;
    }
}

// Get all possible backends for parameterized testing.
// Actual availability is checked in SetUp() after initialization,
// not here — this runs at static init time (INSTANTIATE_TEST_SUITE_P)
// and must not trigger heavy GPU backend initialization.
std::vector<BackendConfig> get_available_backends() {
    return {
        {"CPU",   Device::Type::CPU,   0, true},
        {"CUDA",  Device::Type::CUDA,  0, true},
        {"OneAPI", Device::Type::OneAPI, 0, true},
        {"ROCm",  Device::Type::ROCm,  0, true},
        {"Vulkan", Device::Type::Vulkan, 0, true},
    };
}

// ============================================================================
// Parameterized Test Fixture
// ============================================================================

class BackendOpsTest : public ::testing::TestWithParam<BackendConfig> {
protected:
    void SetUp() override {
        config_ = GetParam();
        device_ = Device(config_.type, config_.device_id);

        // Initialize Tenzor if not already initialized
        static bool initialized = false;
        if (!initialized) {
            tenzor::initialize();
            initialized = true;
        }

        // Skip if backend not available
        if (!is_backend_available(config_.type)) {
            GTEST_SKIP() << config_.name << " backend not available";
        }
    }

    void TearDown() override {
        // For GPU backends, synchronize to ensure all operations complete
        if (config_.is_available && config_.type != Device::Type::CPU) {
            try {
                // Create a dummy tensor to force synchronization
                auto dummy = ones({1}, DType::Float32, device_);
                auto dummy_cpu = dummy.to(Device::cpu());
            } catch (...) {
                // Ignore synchronization errors during teardown
            }
        }
    }

    // Helper to compare tensors with tolerance
    template<typename T>
    void expect_tensor_near(const Tensor& result, const std::vector<T>& expected,
                           T tolerance = static_cast<T>(1e-5)) {
        ASSERT_EQ(result.numel(), static_cast<int64_t>(expected.size()));

        // Copy to CPU if needed
        auto result_cpu = (result.device().type == Device::Type::CPU)
                         ? result
                         : result.to(Device::cpu());

        auto data = result_cpu.data<T>();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(static_cast<double>(data[i]),
                       static_cast<double>(expected[i]),
                       static_cast<double>(tolerance))
                << "Mismatch at index " << i << " on " << config_.name;
        }
    }

    BackendConfig config_;
    Device device_;
};

// ============================================================================
// Addition Tests
// ============================================================================

TEST_P(BackendOpsTest, AddFloat32_Basic) {
    auto a = ones({2, 3}, DType::Float32, device_);
    auto b = ones({2, 3}, DType::Float32, device_);

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);
    EXPECT_EQ(c.device().type, device_.type);

    expect_tensor_near<float>(c, {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});
}

TEST_P(BackendOpsTest, AddFloat32_DifferentValues) {
    auto a = ones({3, 2}, DType::Float32, device_);
    auto b = ones({3, 2}, DType::Float32, device_);

    // Initialize with specific values
    if (device_.type == Device::Type::CPU) {
        auto a_data = a.data<float>();
        auto b_data = b.data<float>();
        for (int i = 0; i < 6; i++) {
            a_data[i] = static_cast<float>(i + 1);  // [1, 2, 3, 4, 5, 6]
            b_data[i] = static_cast<float>(i * 2);  // [0, 2, 4, 6, 8, 10]
        }
    } else {
        // For GPU backends, create on CPU, initialize, then transfer
        auto a_cpu = ones({3, 2}, DType::Float32, Device::cpu());
        auto b_cpu = ones({3, 2}, DType::Float32, Device::cpu());

        auto a_data = a_cpu.data<float>();
        auto b_data = b_cpu.data<float>();
        for (int i = 0; i < 6; i++) {
            a_data[i] = static_cast<float>(i + 1);
            b_data[i] = static_cast<float>(i * 2);
        }

        a = a_cpu.to(device_);
        b = b_cpu.to(device_);
    }

    auto c = add(a, b);

    // Expected: [1, 4, 7, 10, 13, 16]
    expect_tensor_near<float>(c, {1.0f, 4.0f, 7.0f, 10.0f, 13.0f, 16.0f});
}

TEST_P(BackendOpsTest, AddFloat64_Basic) {
    auto a = ones({2, 2}, DType::Float64, device_);
    auto b = ones({2, 2}, DType::Float64, device_);

    auto c = add(a, b);

    EXPECT_EQ(c.dtype(), DType::Float64);
    expect_tensor_near<double>(c, {2.0, 2.0, 2.0, 2.0}, 1e-10);
}

// ============================================================================
// Subtraction Tests
// ============================================================================

TEST_P(BackendOpsTest, SubFloat32_Basic) {
    auto a = full({2, 3}, 5.0f, DType::Float32, device_);
    auto b = ones({2, 3}, DType::Float32, device_);

    auto c = sub(a, b);

    expect_tensor_near<float>(c, {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f});
}

TEST_P(BackendOpsTest, SubFloat32_ResultNegative) {
    auto a = full({2, 2}, 2.0f, DType::Float32, device_);
    auto b = full({2, 2}, 5.0f, DType::Float32, device_);

    auto c = sub(a, b);

    expect_tensor_near<float>(c, {-3.0f, -3.0f, -3.0f, -3.0f});
}

// ============================================================================
// Multiplication Tests
// ============================================================================

TEST_P(BackendOpsTest, MulFloat32_Basic) {
    auto a = full({2, 3}, 3.0f, DType::Float32, device_);
    auto b = full({2, 3}, 4.0f, DType::Float32, device_);

    auto c = mul(a, b);

    expect_tensor_near<float>(c, {12.0f, 12.0f, 12.0f, 12.0f, 12.0f, 12.0f});
}

TEST_P(BackendOpsTest, MulFloat32_WithZero) {
    auto a = full({2, 3}, 100.0f, DType::Float32, device_);
    auto b = zeros({2, 3}, DType::Float32, device_);

    auto c = mul(a, b);

    expect_tensor_near<float>(c, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
}

// ============================================================================
// Division Tests
// ============================================================================

TEST_P(BackendOpsTest, DivFloat32_Basic) {
    auto a = full({2, 3}, 12.0f, DType::Float32, device_);
    auto b = full({2, 3}, 4.0f, DType::Float32, device_);

    auto c = div(a, b);

    expect_tensor_near<float>(c, {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f});
}

TEST_P(BackendOpsTest, DivFloat32_Fractional) {
    auto a = ones({2, 2}, DType::Float32, device_);
    auto b = full({2, 2}, 3.0f, DType::Float32, device_);

    auto c = div(a, b);

    expect_tensor_near<float>(c, {0.33333333f, 0.33333333f, 0.33333333f, 0.33333333f}, 1e-6f);
}

// ============================================================================
// Matrix Multiplication Tests
// ============================================================================

TEST_P(BackendOpsTest, MatMulFloat32_2x3_3x2) {
    // Create test matrices on CPU, then transfer to target device
    auto a_cpu = ones({2, 3}, DType::Float32, Device::cpu());
    auto b_cpu = ones({3, 2}, DType::Float32, Device::cpu());

    auto a_data = a_cpu.data<float>();
    auto b_data = b_cpu.data<float>();

    // A = [[1, 2, 3], [4, 5, 6]]
    a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
    a_data[3] = 4.0f; a_data[4] = 5.0f; a_data[5] = 6.0f;

    // B = [[1, 2], [3, 4], [5, 6]]
    b_data[0] = 1.0f; b_data[1] = 2.0f;
    b_data[2] = 3.0f; b_data[3] = 4.0f;
    b_data[4] = 5.0f; b_data[5] = 6.0f;

    auto a = a_cpu.to(device_);
    auto b = b_cpu.to(device_);

    auto c = matmul(a, b);

    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);

    // Expected: [[22, 28], [49, 64]]
    expect_tensor_near<float>(c, {22.0f, 28.0f, 49.0f, 64.0f});
}

TEST_P(BackendOpsTest, MatMulFloat32_LargeMatrix) {
    const int M = 128;
    const int K = 128;
    const int N = 128;

    auto a = ones({M, K}, DType::Float32, device_);
    auto b = ones({K, N}, DType::Float32, device_);

    auto c = matmul(a, b);

    EXPECT_EQ(c.shape()[0], M);
    EXPECT_EQ(c.shape()[1], N);

    // Each element should be K (sum of K 1s)
    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();

    for (int i = 0; i < 100; i++) {
        EXPECT_FLOAT_EQ(c_data[i], static_cast<float>(K));
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

TEST_P(BackendOpsTest, SumFloat32_FullReduction) {
    auto a = ones({1000}, DType::Float32, device_);

    auto result = sum(a);

    auto result_cpu = result.to(Device::cpu());
    auto result_data = result_cpu.data<float>();

    EXPECT_NEAR(result_data[0], 1000.0f, 1.0f);
}

TEST_P(BackendOpsTest, MeanFloat32_Basic) {
    auto a = full({1000}, 5.0f, DType::Float32, device_);

    auto result = mean(a);

    auto result_cpu = result.to(Device::cpu());
    auto result_data = result_cpu.data<float>();

    EXPECT_NEAR(result_data[0], 5.0f, 1e-5f);
}

// ============================================================================
// Activation Functions
// ============================================================================

TEST_P(BackendOpsTest, ReLU_Forward_Float32) {
    auto a_cpu = ones({1000}, DType::Float32, Device::cpu());
    auto a_data = a_cpu.data<float>();

    for (int i = 0; i < 1000; i++) {
        a_data[i] = static_cast<float>(i - 500);
    }

    auto a = a_cpu.to(device_);
    auto output = nn::relu(Variable(a));

    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();

    for (int i = 0; i < 1000; i++) {
        float expected = std::max(0.0f, static_cast<float>(i - 500));
        EXPECT_FLOAT_EQ(output_data[i], expected);
    }
}

TEST_P(BackendOpsTest, Sigmoid_Forward_Float32) {
    auto a_cpu = ones({100}, DType::Float32, Device::cpu());
    auto a_data = a_cpu.data<float>();

    for (int i = 0; i < 100; i++) {
        a_data[i] = static_cast<float>((i - 50) * 0.1f);
    }

    auto a = a_cpu.to(device_);
    auto output = nn::sigmoid(Variable(a));

    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();

    for (int i = 0; i < 100; i++) {
        float x = static_cast<float>((i - 50) * 0.1f);
        float expected = 1.0f / (1.0f + std::exp(-x));
        EXPECT_NEAR(output_data[i], expected, 1e-5f);
    }
}

TEST_P(BackendOpsTest, Softmax_Forward_2D) {
    auto a_cpu = ones({32, 10}, DType::Float32, Device::cpu());
    auto a_data = a_cpu.data<float>();

    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 10; j++) {
            a_data[i * 10 + j] = static_cast<float>(j);
        }
    }

    auto a = a_cpu.to(device_);
    auto output = nn::softmax(Variable(a), 1);

    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();

    // Check that each row sums to 1.0
    for (int i = 0; i < 32; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 10; j++) {
            sum += output_data[i * 10 + j];
            EXPECT_GT(output_data[i * 10 + j], 0.0f);
            EXPECT_LE(output_data[i * 10 + j], 1.0f);
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Row " << i << " on " << config_.name;
    }
}

// ============================================================================
// Transform Operations
// ============================================================================

TEST_P(BackendOpsTest, Transpose_Float32) {
    auto a_cpu = ones({4, 5}, DType::Float32, Device::cpu());
    auto a_data = a_cpu.data<float>();

    for (int i = 0; i < 20; i++) {
        a_data[i] = static_cast<float>(i);
    }

    auto a = a_cpu.to(device_);
    auto b = transpose(a, 0, 1);

    EXPECT_EQ(b.shape()[0], 5);
    EXPECT_EQ(b.shape()[1], 4);

    auto b_cpu = b.to(Device::cpu());
    auto b_data = b_cpu.data<float>();

    // Verify transpose
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 4; j++) {
            EXPECT_FLOAT_EQ(b_data[i * 4 + j], a_data[j * 5 + i]);
        }
    }
}

TEST_P(BackendOpsTest, Reshape_Float32) {
    auto a_cpu = ones({2, 3, 4}, DType::Float32, Device::cpu());
    auto a_data = a_cpu.data<float>();

    for (int i = 0; i < 24; i++) {
        a_data[i] = static_cast<float>(i);
    }

    auto a = a_cpu.to(device_);
    auto b = reshape(a, {4, 6});

    EXPECT_EQ(b.shape()[0], 4);
    EXPECT_EQ(b.shape()[1], 6);
    EXPECT_EQ(b.numel(), 24);

    auto b_cpu = b.to(Device::cpu());
    auto b_data = b_cpu.data<float>();

    // Data should be the same
    for (int i = 0; i < 24; i++) {
        EXPECT_FLOAT_EQ(b_data[i], static_cast<float>(i));
    }
}

// ============================================================================
// Instantiate Tests for All Backends
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    BackendOpsTest,
    ::testing::ValuesIn(get_available_backends()),
    [](const ::testing::TestParamInfo<BackendConfig>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Print available backends
    auto backends = get_available_backends();
    std::cout << "\n=== Available Backends ===" << std::endl;
    for (const auto& backend : backends) {
        std::cout << "  - " << backend.name << " (device " << backend.device_id << ")" << std::endl;
    }
    std::cout << "=========================\n" << std::endl;

    return RUN_ALL_TESTS();
}
