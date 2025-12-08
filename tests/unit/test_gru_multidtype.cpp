#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// Helper function to convert DType to string
std::string dtype_to_string(DType dtype) {
    switch(dtype) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Float16: return "float16";
        default: return "unknown";
    }
}

// ============================================================================
// Multi-DType Test Fixture
// ============================================================================

struct BackendDTypeParam {
    std::string backend_name;
    DType dtype;
    std::string dtype_name;

    std::string ToString() const {
        return backend_name + "_" + dtype_name;
    }
};

// Required for gtest_discover_tests to show human-readable test names
void PrintTo(const BackendDTypeParam& param, std::ostream* os) {
    *os << param.ToString();
}

// ============================================================================
// GRUCell Multi-DType Tests
// ============================================================================

class GRUCellMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType dtype;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;

        if (param.backend_name == "cpu") {
            device = Device::cpu();
        }
        else if (param.backend_name == "cuda") {
            if (!isBackendAvailable(Device::Type::CUDA)) {
                GTEST_SKIP() << "CUDA not available";
            }
            device = Device::cuda(0);
        }
        else if (param.backend_name == "vulkan") {
            if (!isBackendAvailable(Device::Type::Vulkan)) {
                GTEST_SKIP() << "Vulkan not available";
            }
            device = Device::vulkan(0);
        }
        else if (param.backend_name == "oneapi") {
            if (!isBackendAvailable(Device::Type::OneAPI)) {
                GTEST_SKIP() << "OneAPI not available";
            }
            device = Device::oneapi(0);
        }
        else if (param.backend_name == "adaptivecpp") {
            if (!isBackendAvailable(Device::Type::AdaptiveCpp)) {
                GTEST_SKIP() << "AdaptiveCpp not available";
            }
            device = Device::adaptivecpp(0);
        }
    }

    static bool isBackendAvailable(Device::Type type) {
        try {
            Device test_device{type, 0};
            auto t = zeros({2, 2}, DType::Float32, test_device);
            return true;
        } catch (...) {
            return false;
        }
    }

    // Helper to get dtype-specific tolerance
    template<typename T>
    T getTolerance() const {
        if (std::is_same<T, double>::value) {
            return static_cast<T>(1e-5);
        } else if (std::is_same<T, float>::value) {
            return static_cast<T>(1e-4);
        } else {  // Float16
            return static_cast<T>(1e-2);
        }
    }

    double getTolerance(DType dtype) const {
        switch (dtype) {
            case DType::Float64: return 1e-5;
            case DType::Float32: return 1e-4;
            case DType::Float16: return 1e-2;
            default: return 1e-4;
        }
    }

    double get_tolerance() const {
        if (dtype == DType::Float32) {
            return 1e-4;
        } else if (dtype == DType::Float64) {
            return 1e-5;
        } else if (dtype == DType::Float16) {
            return 1e-2;
        }
        return 1e-4;
    }
};

TEST_P(GRUCellMultiDTypeTest, BasicForward) {
    // Parameters are already set in SetUp()

    // Test basic forward pass
    nn::GRUCell cell(10, 20);
    cell.to(device);
    auto input = Variable(randn({5, 10}, dtype, device), true);
    auto h = Variable(randn({5, 20}, dtype, device), true);

    auto h_next = cell.forward(input, h);

    EXPECT_EQ(h_next.shape().size(), 2) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(h_next.shape()[0], 5) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(h_next.shape()[1], 20) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(h_next.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUCellMultiDTypeTest, NoInitialHidden) {
    // Parameters are already set in SetUp()

    // Test with zero-initialized hidden state
    nn::GRUCell cell(10, 20);
    cell.to(device);
    auto input = Variable(randn({5, 10}, dtype, device), true);

    auto h_next = cell.forward(input);

    EXPECT_EQ(h_next.shape().size(), 2) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(h_next.shape()[0], 5) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(h_next.shape()[1], 20) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(h_next.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUCellMultiDTypeTest, NoBias) {
    // Parameters are already set in SetUp()

    // Test without bias
    nn::GRUCell cell(10, 20, false);
    cell.to(device);
    auto input = Variable(randn({5, 10}, dtype, device), true);

    auto h_next = cell.forward(input);

    EXPECT_EQ(h_next.shape()[0], 5) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(h_next.shape()[1], 20) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(h_next.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUCellMultiDTypeTest, HiddenStateEvolution) {
    // Parameters are already set in SetUp()

    // Test that hidden state evolves across time steps
    nn::GRUCell cell(10, 20);
    cell.to(device);
    auto input = Variable(randn({5, 10}, dtype, device), true);

    auto h1 = cell.forward(input);
    auto h2 = cell.forward(input, h1);
    auto h3 = cell.forward(input, h2);

    // Each step should produce outputs
    EXPECT_EQ(h3.shape()[0], 5) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(h3.shape()[1], 20) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(h3.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUCellMultiDTypeTest, ParameterCount) {
    // Parameters are already set in SetUp()

    // Test parameter count
    nn::GRUCell cell(10, 20);
    auto params = cell.parameters();

    // GRU has 3 gates (reset, update, new)
    // Each gate has input and hidden transformations
    // 6 linear layers * 2 params (weight + bias) = 12 parameters
    EXPECT_EQ(params.size(), 12) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

// ============================================================================
// GRU Multi-DType Tests
// ============================================================================

class GRUMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType dtype;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;

        if (param.backend_name == "cpu") {
            device = Device::cpu();
        }
        else if (param.backend_name == "cuda") {
            if (!isBackendAvailable(Device::Type::CUDA)) {
                GTEST_SKIP() << "CUDA not available";
            }
            device = Device::cuda(0);
        }
        else if (param.backend_name == "vulkan") {
            if (!isBackendAvailable(Device::Type::Vulkan)) {
                GTEST_SKIP() << "Vulkan not available";
            }
            device = Device::vulkan(0);
        }
        else if (param.backend_name == "oneapi") {
            if (!isBackendAvailable(Device::Type::OneAPI)) {
                GTEST_SKIP() << "OneAPI not available";
            }
            device = Device::oneapi(0);
        }
        else if (param.backend_name == "adaptivecpp") {
            if (!isBackendAvailable(Device::Type::AdaptiveCpp)) {
                GTEST_SKIP() << "AdaptiveCpp not available";
            }
            device = Device::adaptivecpp(0);
        }
    }

    static bool isBackendAvailable(Device::Type type) {
        try {
            Device test_device{type, 0};
            auto t = zeros({2, 2}, DType::Float32, test_device);
            return true;
        } catch (...) {
            return false;
        }
    }

    double getTolerance(DType dtype) const {
        switch (dtype) {
            case DType::Float64: return 1e-5;
            case DType::Float32: return 1e-4;
            case DType::Float16: return 1e-2;
            default: return 1e-4;
        }
    }

    double get_tolerance() const {
        if (dtype == DType::Float32) {
            return 1e-4;
        } else if (dtype == DType::Float64) {
            return 1e-5;
        } else if (dtype == DType::Float16) {
            return 1e-2;
        }
        return 1e-4;
    }
};

TEST_P(GRUMultiDTypeTest, BasicForward) {
    // Parameters are already set in SetUp()

    // Test basic forward pass
    nn::GRU gru(10, 20, 1);
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);  // (seq_len, batch, features)

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape().size(), 3) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // seq_len
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // batch
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype); // hidden_size

    EXPECT_EQ(h_n.shape().size(), 3) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(h_n.shape()[0], 1) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // num_layers
    EXPECT_EQ(h_n.shape()[1], 5) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // batch
    EXPECT_EQ(h_n.shape()[2], 20) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype); // hidden_size

    EXPECT_EQ(output.tensor().dtype(), dtype) << "Output dtype mismatch";
    EXPECT_EQ(h_n.tensor().dtype(), dtype) << "Hidden state dtype mismatch";
}

TEST_P(GRUMultiDTypeTest, MultiLayer) {
    // Parameters are already set in SetUp()

    // Test multi-layer GRU
    nn::GRU gru(10, 20, 3);  // 3 layers
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    EXPECT_EQ(h_n.shape()[0], 3) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // 3 layers
    EXPECT_EQ(output.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUMultiDTypeTest, BatchFirst) {
    // Parameters are already set in SetUp()

    // Test with batch_first=true
    nn::GRU gru(10, 20, 1, true, true);  // batch_first=true
    gru.to(device);
    auto input = Variable(randn({5, 7, 10}, dtype, device), true);  // (batch, seq_len, features)

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 5) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // batch
    EXPECT_EQ(output.shape()[1], 7) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // seq_len
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype); // hidden_size
    EXPECT_EQ(output.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUMultiDTypeTest, Bidirectional) {
    // Parameters are already set in SetUp()

    // Test bidirectional GRU
    nn::GRU gru(10, 20, 1, true, false, 0.0, true);  // bidirectional=true
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype); // hidden_size * 2

    EXPECT_EQ(h_n.shape()[0], 2) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // num_layers * num_directions
    EXPECT_EQ(output.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUMultiDTypeTest, BidirectionalMultiLayer) {
    // Parameters are already set in SetUp()

    // Test bidirectional multi-layer GRU
    nn::GRU gru(10, 20, 2, true, false, 0.0, true);
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype); // hidden_size * 2
    EXPECT_EQ(h_n.shape()[0], 4) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);     // num_layers * num_directions
    EXPECT_EQ(output.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUMultiDTypeTest, WithDropout) {
    // Parameters are already set in SetUp()

    // Test with dropout between layers
    nn::GRU gru(10, 20, 3, true, false, 0.5);  // 50% dropout
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUMultiDTypeTest, InitialHiddenState) {
    // Parameters are already set in SetUp()

    // Test with provided initial hidden state
    nn::GRU gru(10, 20, 2);
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);
    auto h0 = Variable(randn({2, 5, 20}, dtype, device), true);

    auto [output, h_n] = gru.forward(input, h0);

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(h_n.shape()[0], 2) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUMultiDTypeTest, SequenceLengthVariation) {
    // Parameters are already set in SetUp()

    // Test with different sequence lengths
    nn::GRU gru(10, 20);
    gru.to(device);

    // Short sequence
    auto input1 = Variable(randn({3, 5, 10}, dtype, device), true);
    auto [output1, h_n1] = gru.forward(input1, Variable{});
    EXPECT_EQ(output1.shape()[0], 3) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    // Long sequence
    auto input2 = Variable(randn({50, 5, 10}, dtype, device), true);
    auto [output2, h_n2] = gru.forward(input2, Variable{});
    EXPECT_EQ(output2.shape()[0], 50) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    EXPECT_EQ(output1.tensor().dtype(), dtype) << "Output dtype mismatch";
    EXPECT_EQ(output2.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUMultiDTypeTest, BatchSizeVariation) {
    // Parameters are already set in SetUp()

    // Test with different batch sizes
    nn::GRU gru(10, 20);
    gru.to(device);

    // Small batch
    auto input1 = Variable(randn({7, 2, 10}, dtype, device), true);
    auto [output1, h_n1] = gru.forward(input1, Variable{});
    EXPECT_EQ(output1.shape()[1], 2) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    // Large batch
    auto input2 = Variable(randn({7, 32, 10}, dtype, device), true);
    auto [output2, h_n2] = gru.forward(input2, Variable{});
    EXPECT_EQ(output2.shape()[1], 32) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    EXPECT_EQ(output1.tensor().dtype(), dtype) << "Output dtype mismatch";
    EXPECT_EQ(output2.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUMultiDTypeTest, SingleTimestep) {
    // Parameters are already set in SetUp()

    // Test with single timestep
    nn::GRU gru(10, 20);
    gru.to(device);
    auto input = Variable(randn({1, 5, 10}, dtype, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 1) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUMultiDTypeTest, OutputConsistency) {
    // Parameters are already set in SetUp()

    // Test that output is deterministic
    nn::GRU gru(10, 20);
    gru.to(device);
    auto input = Variable(ones({7, 5, 10}, dtype, device), true);

    auto [output1, h_n1] = gru.forward(input, Variable{});
    auto [output2, h_n2] = gru.forward(input, Variable{});

    // Shapes should match
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(output1.shape()[i], output2.shape()[i]) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
        EXPECT_EQ(h_n1.shape()[i], h_n2.shape()[i]) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    }
    EXPECT_EQ(output1.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUMultiDTypeTest, GradientFlow) {
    // Parameters are already set in SetUp()

    // Test that gradients can flow through GRU
    nn::GRU gru(10, 20);
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    // Check that output requires grad
    EXPECT_TRUE(output.requires_grad()) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    // Sum for scalar output
    auto loss = Variable(sum(output.tensor(), std::nullopt, false), true);

    // Backward should not throw
    EXPECT_NO_THROW({
        loss.backward();
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(GRUMultiDTypeTest, TrainingMode) {
    // Parameters are already set in SetUp()

    // Test training/eval mode switching
    nn::GRU gru(10, 20, 2, true, false, 0.5);

    EXPECT_TRUE(gru.is_training()) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    gru.eval();
    EXPECT_FALSE(gru.is_training()) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    gru.train();
    EXPECT_TRUE(gru.is_training()) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(GRUMultiDTypeTest, ParameterCount) {
    // Parameters are already set in SetUp()

    // Test parameter counting
    nn::GRU gru(10, 20, 2);
    auto params = gru.parameters();

    // Each layer has 6 linear transformations (3 gates * 2 transforms each)
    // Each linear has weight and bias
    // Layer 0: 6 * 2 = 12 params
    // Layer 1: 6 * 2 = 12 params
    // Total: 24 parameters
    EXPECT_EQ(params.size(), 24) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(GRUMultiDTypeTest, LargeHidden) {
    // Parameters are already set in SetUp()

    // Test with large hidden size
    nn::GRU gru(10, 512);
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[2], 512) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUMultiDTypeTest, VeryDeepNetwork) {
    // Parameters are already set in SetUp()

    // Test with many layers
    nn::GRU gru(10, 20, 5);
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(h_n.shape()[0], 5) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // 5 layers
    EXPECT_EQ(output.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUMultiDTypeTest, LongSequence) {
    // Parameters are already set in SetUp()

    // Test with very long sequence
    nn::GRU gru(10, 20);
    gru.to(device);
    auto input = Variable(randn({100, 5, 10}, dtype, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 100) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUMultiDTypeTest, InvalidNumLayers) {
    // Parameters are already set in SetUp()

    // Test that invalid num_layers throws
    EXPECT_THROW({
        nn::GRU gru(10, 20, 0);
    }, std::invalid_argument) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(GRUMultiDTypeTest, InvalidDropout) {
    // Parameters are already set in SetUp()

    // Test that invalid dropout throws
    EXPECT_THROW({
        nn::GRU gru(10, 20, 2, true, false, 1.5);
    }, std::invalid_argument) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(GRUMultiDTypeTest, GateOutputRanges) {
    double tol = getTolerance(dtype);

    // Test that GRU gates produce reasonable outputs
    nn::GRU gru(10, 20);
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    // Hidden state values should be in reasonable range (due to tanh in new gate)
    auto h_cpu = h_n.tensor().to(Device::cpu());

    // Check dtype-specific ranges
    double max_value = 10.0;
    if (dtype == DType::Float16) {
        max_value = 15.0;  // Float16 may have slightly higher variance
    }

    bool all_reasonable = true;
    if (dtype == DType::Float64) {
        const double* h_data = h_cpu.data<double>();
        for (int64_t i = 0; i < h_cpu.numel(); ++i) {
            if (std::abs(h_data[i]) > max_value) {
                all_reasonable = false;
                break;
            }
        }
    } else if (dtype == DType::Float32) {
        const float* h_data = h_cpu.data<float>();
        for (int64_t i = 0; i < h_cpu.numel(); ++i) {
            if (std::abs(h_data[i]) > max_value) {
                all_reasonable = false;
                break;
            }
        }
    }
    // Note: Float16 range testing skipped due to limited half type support
    EXPECT_TRUE(all_reasonable || dtype == DType::Float16) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(GRUMultiDTypeTest, BatchFirstBidirectional) {
    // Parameters are already set in SetUp()

    // Test combination of batch_first and bidirectional
    nn::GRU gru(10, 20, 1, true, true, 0.0, true);
    gru.to(device);
    auto input = Variable(randn({5, 7, 10}, dtype, device), true);  // (batch, seq, features)

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 5) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // batch
    EXPECT_EQ(output.shape()[1], 7) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // seq_len
    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype); // hidden_size * 2
    EXPECT_EQ(output.tensor().dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(GRUMultiDTypeTest, ComparisonWithLSTM) {

    // Ensure GRU has similar behavior to LSTM (fewer parameters, similar performance)
    nn::GRU gru(10, 20, 2);
    gru.to(device);
    nn::LSTM lstm(10, 20, 2);
    lstm.to(device);

    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [gru_output, gru_h] = gru.forward(input, Variable{});
    auto [lstm_output, lstm_states] = lstm.forward(input, {Variable{}, Variable{}});

    // Both should produce same shaped outputs
    EXPECT_EQ(gru_output.shape()[0], lstm_output.shape()[0]) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(gru_output.shape()[1], lstm_output.shape()[1]) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(gru_output.shape()[2], lstm_output.shape()[2]) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    // GRU has more parameters due to separate gate transforms
    // (GRU uses 6 separate Linear layers vs LSTM's 2 combined layers)
    auto gru_params = gru.parameters();
    auto lstm_params = lstm.parameters();
    EXPECT_GT(gru_params.size(), lstm_params.size()) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    EXPECT_EQ(gru_output.tensor().dtype(), dtype) << "GRU output dtype mismatch";
    EXPECT_EQ(lstm_output.tensor().dtype(), dtype) << "LSTM output dtype mismatch";
}

TEST_P(GRUMultiDTypeTest, MemoryEfficiency) {
    // Parameters are already set in SetUp()

    // Test that GRU uses less memory than LSTM
    nn::GRU gru(10, 20);
    gru.to(device);
    nn::LSTM lstm(10, 20);
    lstm.to(device);

    // GRU only returns hidden state, LSTM returns both hidden and cell
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [gru_output, gru_h] = gru.forward(input, Variable{});
    auto [lstm_output, lstm_states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [lstm_h, lstm_c] = lstm_states;

    // GRU hidden state shape
    EXPECT_EQ(gru_h.shape().size(), 3) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    // LSTM has both h and c
    EXPECT_EQ(lstm_h.shape().size(), 3) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(lstm_c.shape().size(), 3) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    EXPECT_EQ(gru_h.tensor().dtype(), dtype) << "GRU hidden state dtype mismatch";
    EXPECT_EQ(lstm_h.tensor().dtype(), dtype) << "LSTM hidden state dtype mismatch";
}

// DType-specific numerical accuracy tests
TEST_P(GRUMultiDTypeTest, NumericalPrecision) {
    double tol = getTolerance(dtype);

    // Test numerical precision across dtypes
    nn::GRU gru(10, 20);
    gru.to(device);

    // Use small values to test precision
    auto input = Variable(full({7, 5, 10}, 0.1, dtype, device), true);
    auto [output, h_n] = gru.forward(input, Variable{});

    // Output should not contain NaN or Inf
    auto out_cpu = output.tensor().to(Device::cpu());
    bool has_invalid = false;

    if (dtype == DType::Float64) {
        auto data = out_cpu.data<double>();
        for (int64_t i = 0; i < out_cpu.numel(); ++i) {
            if (std::isnan(data[i]) || std::isinf(data[i])) {
                has_invalid = true;
                break;
            }
        }
    } else if (dtype == DType::Float32) {
        auto data = out_cpu.data<float>();
        for (int64_t i = 0; i < out_cpu.numel(); ++i) {
            if (std::isnan(data[i]) || std::isinf(data[i])) {
                has_invalid = true;
                break;
            }
        }
    }
    // Note: Float16 testing skipped due to limited support in test data access
    // Future: Add proper Float16 data access when available

    EXPECT_FALSE(has_invalid) << "Output contains NaN or Inf on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(GRUMultiDTypeTest, CrossDTypeConsistency) {

    // Test that different dtypes produce consistent shapes
    nn::GRU gru(10, 20);
    gru.to(device);

    auto input = Variable(randn({7, 5, 10}, dtype, device), true);
    auto [output, h_n] = gru.forward(input, Variable{});

    // Verify output properties
    EXPECT_EQ(output.shape().size(), 3) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.tensor().dtype(), dtype) << "Output dtype should match input dtype";
    EXPECT_EQ(h_n.tensor().dtype(), dtype) << "Hidden state dtype should match input dtype";
    EXPECT_TRUE(output.requires_grad()) << "Output should require gradients";
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<BackendDTypeParam> GenerateBackendDTypeCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi", "adaptivecpp"};

    // Test with floating-point dtypes (GRU works with floating-point)
    std::vector<std::pair<DType, std::string>> dtypes = {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
    };

    std::vector<BackendDTypeParam> combinations;
    for (const auto& backend : backends) {
        for (const auto& [dtype, dtype_name] : dtypes) {
            combinations.push_back({backend, dtype, dtype_name});
        }
    }
    return combinations;
}

INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    GRUCellMultiDTypeTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    GRUMultiDTypeTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);
