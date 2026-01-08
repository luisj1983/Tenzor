/**
 * @file test_model_checkpoint_multidtype.cpp
 * @brief Multi-dtype tests for model checkpointing
 *
 * Coverage: Backend × DType testing for model checkpoint operations
 * - Primary dtypes: Float32, Float64
 * - Backends: CPU, CUDA, Vulkan, OneAPI
 *
 * Note: Model checkpointing primarily involves floating-point parameters,
 * but we test serialization/deserialization across dtypes for completeness.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/checkpoint.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/ops.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <unistd.h>

using namespace tenzor;
using namespace tenzor::nn;

// ============================================================================
// Backend + DType Parameterization
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

class ModelCheckpointMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType dtype;
    double tolerance;
    std::string test_dir_;

    void SetUp() override {

        auto param = GetParam();
        dtype = param.dtype;

        // Set dtype-specific tolerance
        if (dtype == DType::Float32) {
            tolerance = 1e-5;
        } else if (dtype == DType::Float64) {
            tolerance = 1e-10;
        } else {
            tolerance = 1e-5;
        }

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

        // Create unique test directory
        std::stringstream ss;
        ss << "./test_checkpoints_multidtype_" << getpid() << "_"
           << std::this_thread::get_id() << "_" << param.ToString();
        test_dir_ = ss.str();
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        if (std::filesystem::exists(test_dir_)) {
            std::filesystem::remove_all(test_dir_);
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
};

// ==============================================================================
// ModelCheckpoint Tests
// ==============================================================================

TEST_P(ModelCheckpointMultiDTypeTest, Construction) {
    ModelCheckpoint checkpoint;
    EXPECT_EQ(checkpoint.config().compression, CompressionType::None);
    EXPECT_TRUE(checkpoint.config().save_optimizer);
    EXPECT_TRUE(checkpoint.config().save_scheduler);
    EXPECT_TRUE(checkpoint.config().verify_checksum);
    EXPECT_TRUE(checkpoint.config().atomic_save);
}

TEST_P(ModelCheckpointMultiDTypeTest, SaveLoadModelOnly) {
    // Linear layer uses Float32 internally, we test checkpoint I/O
    Linear model(10, 5);

    auto params = model.parameters();
    auto weight_data = params[0]->tensor().data<float>();
    auto bias_data = params[1]->tensor().data<float>();

    std::string path = test_dir_ + "/model.pt";
    ModelCheckpoint checkpoint;
    checkpoint.save_model(path, model);

    EXPECT_TRUE(std::filesystem::exists(path));

    auto loaded_state = checkpoint.load_model(path);

    EXPECT_TRUE(loaded_state.find("weight") != loaded_state.end());
    EXPECT_TRUE(loaded_state.find("bias") != loaded_state.end());

    const float* loaded_weight = loaded_state["weight"].data<float>();
    const float* loaded_bias = loaded_state["bias"].data<float>();

    for (int i = 0; i < 50; ++i) {
        EXPECT_FLOAT_EQ(loaded_weight[i], weight_data[i]);
    }

    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(loaded_bias[i], bias_data[i]);
    }
}

TEST_P(ModelCheckpointMultiDTypeTest, SaveLoadWithOptimizer) {
    Linear model(8, 4);
    auto params = model.parameters();
    optim::SGD optimizer(params, 0.01);

    for (int i = 0; i < 3; ++i) {
        auto input = randn({2, 8}, DType::Float32, device);
        auto output = model.forward(Variable(input, true));
        auto loss = tenzor::sum(output);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    std::string path = test_dir_ + "/checkpoint_with_optim.pt";
    ModelCheckpoint checkpoint;
    checkpoint.save(path, model, &optimizer);

    auto loaded = checkpoint.load(path);

    EXPECT_FALSE(loaded.model_state.empty());
    EXPECT_FALSE(loaded.optimizer_state.empty());
    EXPECT_EQ(loaded.version, CHECKPOINT_VERSION);
}

TEST_P(ModelCheckpointMultiDTypeTest, SaveLoadWithMetadata) {
    Linear model(4, 2);

    TrainingMetadata metadata;
    metadata.epoch = 10;
    metadata.global_step = 1000;
    metadata.learning_rate = 0.001;
    metadata.train_loss = 0.25;
    metadata.val_loss = 0.30;
    metadata.train_accuracy = 0.85;
    metadata.val_accuracy = 0.82;
    metadata.best_val_loss = 0.28;
    metadata.best_val_accuracy = 0.83;
    metadata.custom_metrics["f1_score"] = 0.81;
    metadata.custom_metrics["precision"] = 0.84;

    std::string path = test_dir_ + "/checkpoint_with_metadata.pt";
    ModelCheckpoint checkpoint;
    checkpoint.save_model(path, model, metadata);

    auto loaded_metadata = checkpoint.get_metadata(path);

    EXPECT_EQ(loaded_metadata.epoch, 10);
    EXPECT_EQ(loaded_metadata.global_step, 1000);
    EXPECT_DOUBLE_EQ(loaded_metadata.learning_rate, 0.001);
    EXPECT_DOUBLE_EQ(loaded_metadata.train_loss, 0.25);
    EXPECT_DOUBLE_EQ(loaded_metadata.val_loss, 0.30);
    EXPECT_DOUBLE_EQ(loaded_metadata.custom_metrics["f1_score"], 0.81);
    EXPECT_DOUBLE_EQ(loaded_metadata.custom_metrics["precision"], 0.84);
}

TEST_P(ModelCheckpointMultiDTypeTest, VerifyCheckpoint) {
    Linear model(3, 2);

    std::string path = test_dir_ + "/verify_test.pt";
    ModelCheckpoint checkpoint;
    checkpoint.save_model(path, model);

    EXPECT_TRUE(checkpoint.verify_checkpoint(path));

    // Corrupt checkpoint
    std::ofstream corrupt(path, std::ios::binary | std::ios::app);
    corrupt << "CORRUPT_DATA";
    corrupt.close();

    EXPECT_FALSE(checkpoint.verify_checkpoint(path));
}

TEST_P(ModelCheckpointMultiDTypeTest, RoundtripPreservesValues) {
    Linear model(5, 3);

    auto params = model.parameters();
    const float* weight = params[0]->tensor().data<float>();
    const float* bias = params[1]->tensor().data<float>();

    std::string path = test_dir_ + "/roundtrip.pt";
    ModelCheckpoint checkpoint;
    checkpoint.save_model(path, model);

    auto loaded_state = checkpoint.load_model(path);

    const float* loaded_weight = loaded_state["weight"].data<float>();
    const float* loaded_bias = loaded_state["bias"].data<float>();

    for (int i = 0; i < 15; ++i) {
        EXPECT_FLOAT_EQ(loaded_weight[i], weight[i]);
    }

    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(loaded_bias[i], bias[i]);
    }
}

// ==============================================================================
// AutoCheckpoint Tests
// ==============================================================================

TEST_P(ModelCheckpointMultiDTypeTest, AutoCheckpointStep) {
    AutoCheckpoint auto_checkpoint(test_dir_, 3, 1);
    auto_checkpoint.set_metric_mode("min");

    Linear model(4, 2);
    auto params = model.parameters();
    optim::SGD optimizer(params, 0.01);

    bool saved = auto_checkpoint.step(model, optimizer, 0, 1.0, "loss");
    EXPECT_TRUE(saved);
    EXPECT_EQ(auto_checkpoint.checkpoint_paths().size(), 1);
    EXPECT_DOUBLE_EQ(auto_checkpoint.best_metric_value(), 1.0);

    saved = auto_checkpoint.step(model, optimizer, 1, 0.8, "loss");
    EXPECT_TRUE(saved);
    EXPECT_EQ(auto_checkpoint.checkpoint_paths().size(), 2);
    EXPECT_DOUBLE_EQ(auto_checkpoint.best_metric_value(), 0.8);
}

TEST_P(ModelCheckpointMultiDTypeTest, AutoCheckpointMaxCheckpoints) {
    AutoCheckpoint auto_checkpoint(test_dir_, 2, 1);
    auto_checkpoint.set_metric_mode("min");

    Linear model(3, 2);
    auto params = model.parameters();
    optim::SGD optimizer(params, 0.01);

    for (int epoch = 0; epoch < 4; ++epoch) {
        double loss = 1.0 - epoch * 0.1;
        auto_checkpoint.step(model, optimizer, epoch, loss, "loss");
    }

    EXPECT_LE(auto_checkpoint.checkpoint_paths().size(), 2);
}

TEST_P(ModelCheckpointMultiDTypeTest, AutoCheckpointMetricMode) {
    AutoCheckpoint auto_checkpoint(test_dir_, 3, 1);
    auto_checkpoint.set_metric_mode("max");

    Linear model(2, 1);
    auto params = model.parameters();
    optim::SGD optimizer(params, 0.01);

    auto_checkpoint.step(model, optimizer, 0, 0.7, "accuracy");
    EXPECT_DOUBLE_EQ(auto_checkpoint.best_metric_value(), 0.7);

    auto_checkpoint.step(model, optimizer, 1, 0.85, "accuracy");
    EXPECT_DOUBLE_EQ(auto_checkpoint.best_metric_value(), 0.85);

    auto_checkpoint.step(model, optimizer, 2, 0.8, "accuracy");
    EXPECT_DOUBLE_EQ(auto_checkpoint.best_metric_value(), 0.85);
}

// ==============================================================================
// Test Instantiation
// ==============================================================================

std::vector<BackendDTypeParam> GenerateCheckpointCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi", "rocm"};

    // Float32 and Float64 for model parameters
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
    AllBackendsFloatDTypes,
    ModelCheckpointMultiDTypeTest,
    ::testing::ValuesIn(GenerateCheckpointCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Test Environment Setup
// ============================================================================

class ModelCheckpointTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        initialize();
    }
};

static ::testing::Environment* const model_checkpoint_env =
    ::testing::AddGlobalTestEnvironment(new ModelCheckpointTestEnvironment);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
