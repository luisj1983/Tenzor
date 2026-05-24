#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/serialize.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include <filesystem>
#include <fstream>
#include <unistd.h>  // W.24: getpid()

using namespace tenzor;
using namespace tenzor::nn;

// Global test environment
class TenzorTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const tenzor_env =
    ::testing::AddGlobalTestEnvironment(new TenzorTestEnvironment);

class SerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // W.24: mirror the ONNXImportTest pattern — per-process + per-test
        // unique directory so parallel ctest workers don't race on each
        // other's fixtures.
        const auto* test_info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        std::string unique_name = std::string("tenzor_serialization_") +
            std::to_string(getpid()) + "_" +
            (test_info ? test_info->name() : "unknown");
        test_dir_ = (std::filesystem::temp_directory_path() / unique_name).string();
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        // Clean up test files
        std::filesystem::remove_all(test_dir_);
    }

    std::string get_test_path(const std::string& filename) {
        return test_dir_ + "/" + filename;
    }

    std::string test_dir_;
};

// Test 1: Save and load simple tensor state dict
TEST_F(SerializationTest, SaveLoadStateDictBasic) {
    std::unordered_map<std::string, Tensor> state;
    state["weight"] = ones({2, 3}, DType::Float32);
    state["bias"] = zeros({3}, DType::Float32);

    auto path = get_test_path("state_dict.bin");
    Serializer::save(state, path);

    auto loaded_state = Serializer::load(path);

    ASSERT_EQ(loaded_state.size(), 2);
    ASSERT_TRUE(loaded_state.count("weight"));
    ASSERT_TRUE(loaded_state.count("bias"));

    auto weight_data = loaded_state["weight"].data<float>();
    auto bias_data = loaded_state["bias"].data<float>();

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(weight_data[i], 1.0f);
    }
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(bias_data[i], 0.0f);
    }
}

// Test 2: Save and load with different dtypes
TEST_F(SerializationTest, SaveLoadDifferentDTypes) {
    std::unordered_map<std::string, Tensor> state;
    state["float32"] = ones({2, 2}, DType::Float32);
    state["float64"] = ones({3, 3}, DType::Float64);
    state["int32"] = zeros({2, 3}, DType::Int32);
    state["int64"] = zeros({4}, DType::Int64);

    auto path = get_test_path("dtypes.bin");
    Serializer::save(state, path);

    auto loaded_state = Serializer::load(path);

    ASSERT_EQ(loaded_state.size(), 4);
    EXPECT_EQ(loaded_state["float32"].dtype(), DType::Float32);
    EXPECT_EQ(loaded_state["float64"].dtype(), DType::Float64);
    EXPECT_EQ(loaded_state["int32"].dtype(), DType::Int32);
    EXPECT_EQ(loaded_state["int64"].dtype(), DType::Int64);
}

// Test 3: Check file validation
TEST_F(SerializationTest, FileValidation) {
    auto path = get_test_path("valid.bin");
    std::unordered_map<std::string, Tensor> state;
    state["data"] = ones({5}, DType::Float32);

    Serializer::save(state, path);
    EXPECT_TRUE(Serializer::is_valid_file(path));

    // Create invalid file
    auto invalid_path = get_test_path("invalid.bin");
    std::ofstream invalid_file(invalid_path, std::ios::binary);
    invalid_file << "invalid data";
    invalid_file.close();

    EXPECT_FALSE(Serializer::is_valid_file(invalid_path));
    EXPECT_FALSE(Serializer::is_valid_file("/nonexistent/file.bin"));
}

// Test 4: Module save and load
TEST_F(SerializationTest, ModuleSaveLoad) {
    auto linear = std::make_shared<Linear>(4, 3);

    // Initialize weights to known values
    auto params = linear->parameters();
    params[0]->tensor().fill_(1.5f);
    params[1]->tensor().fill_(0.5f);

    // Save
    auto path = get_test_path("linear.bin");
    linear->save(path);

    // Create new module and load
    auto linear2 = std::make_shared<Linear>(4, 3);
    linear2->load(path);

    // Verify weights match
    auto params2 = linear2->parameters();
    auto weight_data = params2[0]->tensor().data<float>();
    auto bias_data = params2[1]->tensor().data<float>();

    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(weight_data[i], 1.5f);
    }
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(bias_data[i], 0.5f);
    }
}

// Test 5: Module state_dict
TEST_F(SerializationTest, ModuleStateDict) {
    auto linear = std::make_shared<Linear>(3, 2);
    auto state = linear->state_dict();

    ASSERT_EQ(state.size(), 2);
    ASSERT_TRUE(state.count("weight"));
    ASSERT_TRUE(state.count("bias"));

    EXPECT_EQ(state["weight"].shape()[0], 2);
    EXPECT_EQ(state["weight"].shape()[1], 3);
    EXPECT_EQ(state["bias"].shape()[0], 2);
}

// Test 6: Module load_state_dict
TEST_F(SerializationTest, ModuleLoadStateDict) {
    auto linear = std::make_shared<Linear>(2, 3);

    std::unordered_map<std::string, Tensor> state;
    state["weight"] = ones({3, 2}, DType::Float32) * 2.0f;
    state["bias"] = ones({3}, DType::Float32) * 0.5f;

    linear->load_state_dict(state);

    auto params = linear->parameters();
    auto weight_data = params[0]->tensor().data<float>();
    auto bias_data = params[1]->tensor().data<float>();

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(weight_data[i], 2.0f);
    }
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(bias_data[i], 0.5f);
    }
}

// Test 7: Shape mismatch error
TEST_F(SerializationTest, ShapeMismatchError) {
    auto linear = std::make_shared<Linear>(3, 2);

    std::unordered_map<std::string, Tensor> state;
    state["weight"] = ones({4, 4}, DType::Float32); // Wrong shape
    state["bias"] = ones({2}, DType::Float32);

    EXPECT_THROW(linear->load_state_dict(state), std::runtime_error);
}

// Test 8: DType mismatch error
TEST_F(SerializationTest, DTypeMismatchError) {
    auto linear = std::make_shared<Linear>(3, 2);

    std::unordered_map<std::string, Tensor> state;
    state["weight"] = ones({2, 3}, DType::Float64); // Wrong dtype
    state["bias"] = ones({2}, DType::Float32);

    EXPECT_THROW(linear->load_state_dict(state), std::runtime_error);
}

// Test 9: Partial state loading (missing keys)
TEST_F(SerializationTest, PartialStateLoading) {
    auto linear = std::make_shared<Linear>(3, 2);
    linear->parameters()[0]->tensor().fill_(1.0f);
    linear->parameters()[1]->tensor().fill_(2.0f);

    std::unordered_map<std::string, Tensor> state;
    state["weight"] = ones({2, 3}, DType::Float32) * 5.0f;
    // Missing bias — pass strict=false so the missing key doesn't throw.
    // The test asserts the missing param keeps its prior value.

    linear->load_state_dict(state, /*strict=*/false);

    auto params = linear->parameters();
    auto weight_data = params[0]->tensor().data<float>();
    auto bias_data = params[1]->tensor().data<float>();

    // Weight should be updated
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(weight_data[i], 5.0f);
    }

    // Bias should remain unchanged
    for (int i = 0; i < 2; ++i) {
        EXPECT_FLOAT_EQ(bias_data[i], 2.0f);
    }
}

// Test 10: Save and load empty state dict
TEST_F(SerializationTest, EmptyStateDict) {
    std::unordered_map<std::string, Tensor> state;
    auto path = get_test_path("empty.bin");

    Serializer::save(state, path);
    auto loaded_state = Serializer::load(path);

    EXPECT_EQ(loaded_state.size(), 0);
}

// Test 11: Large tensor serialization
TEST_F(SerializationTest, LargeTensor) {
    std::unordered_map<std::string, Tensor> state;
    state["large"] = ones({1000, 1000}, DType::Float32);

    auto path = get_test_path("large.bin");
    Serializer::save(state, path);

    auto loaded_state = Serializer::load(path);

    ASSERT_EQ(loaded_state.size(), 1);
    EXPECT_EQ(loaded_state["large"].shape()[0], 1000);
    EXPECT_EQ(loaded_state["large"].shape()[1], 1000);

    auto data = loaded_state["large"].data<float>();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[999999], 1.0f);
}

// Test 12: Adam optimizer save and load
TEST_F(SerializationTest, AdamOptimizerSaveLoad) {
    auto linear = std::make_shared<Linear>(3, 2);
    auto params = linear->parameters();

    optim::Adam optimizer(params, 0.01);

    // Perform a few steps to initialize momentum
    auto input = ones({1, 3}, DType::Float32);
    auto output = linear->forward(Variable(input));
    auto loss = sum(output.tensor());
    auto loss_var = Variable(loss);
    loss_var.backward();
    optimizer.step();

    // Save optimizer state
    auto path = get_test_path("adam.bin");
    optimizer.save_state(path);

    // Create new optimizer and load
    auto linear2 = std::make_shared<Linear>(3, 2);
    auto params2 = linear2->parameters();
    optim::Adam optimizer2(params2, 0.001); // Different lr

    optimizer2.load_state(path);

    // Verify learning rate was loaded
    EXPECT_DOUBLE_EQ(optimizer2.get_lr(), 0.01);

    // Verify momentum buffers exist and are non-zero
    auto state = optimizer2.state_dict();
    EXPECT_TRUE(state.count("exp_avg_0"));
    EXPECT_TRUE(state.count("exp_avg_sq_0"));
    EXPECT_EQ(state["step_count"].data<int64_t>()[0], 1);
}

// Test 13: Optimizer state_dict
TEST_F(SerializationTest, OptimizerStateDict) {
    auto linear = std::make_shared<Linear>(2, 1);
    auto params = linear->parameters();

    optim::Adam optimizer(params, 0.005, 0.85, 0.995, 1e-7);

    auto state = optimizer.state_dict();

    EXPECT_TRUE(state.count("lr"));
    EXPECT_TRUE(state.count("beta1"));
    EXPECT_TRUE(state.count("beta2"));
    EXPECT_TRUE(state.count("eps"));
    EXPECT_TRUE(state.count("step_count"));

    EXPECT_DOUBLE_EQ(state["lr"].data<double>()[0], 0.005);
    EXPECT_DOUBLE_EQ(state["beta1"].data<double>()[0], 0.85);
    EXPECT_DOUBLE_EQ(state["beta2"].data<double>()[0], 0.995);
    EXPECT_DOUBLE_EQ(state["eps"].data<double>()[0], 1e-7);
}

// Test 14: Corrupted file error
TEST_F(SerializationTest, CorruptedFileError) {
    auto path = get_test_path("corrupted.bin");

    // Write corrupted data
    std::ofstream file(path, std::ios::binary);
    file << "corrupted data";
    file.close();

    EXPECT_THROW(Serializer::load(path), std::runtime_error);
}

// Test 15: Sequential module serialization
TEST_F(SerializationTest, SequentialModuleSerialization) {
    // Create sequential model
    auto seq = std::make_shared<Sequential>();
    auto linear1 = std::make_shared<Linear>(4, 8);
    auto linear2 = std::make_shared<Linear>(8, 2);

    seq->add_module(linear1);
    seq->add_module(linear2);

    // Set weights to known values
    linear1->parameters()[0]->tensor().fill_(1.0f);
    linear1->parameters()[1]->tensor().fill_(0.5f);
    linear2->parameters()[0]->tensor().fill_(2.0f);
    linear2->parameters()[1]->tensor().fill_(0.25f);

    // Save
    auto path = get_test_path("sequential.bin");
    seq->save(path);

    // Create new sequential and load
    auto seq2 = std::make_shared<Sequential>();
    auto linear1_new = std::make_shared<Linear>(4, 8);
    auto linear2_new = std::make_shared<Linear>(8, 2);

    seq2->add_module(linear1_new);
    seq2->add_module(linear2_new);

    seq2->load(path);

    // Verify weights
    auto params = seq2->parameters();
    EXPECT_FLOAT_EQ(params[0]->tensor().data<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(params[1]->tensor().data<float>()[0], 0.5f);
    EXPECT_FLOAT_EQ(params[2]->tensor().data<float>()[0], 2.0f);
    EXPECT_FLOAT_EQ(params[3]->tensor().data<float>()[0], 0.25f);
}

// Test 16: Round-trip test with computation
TEST_F(SerializationTest, RoundTripComputation) {
    auto linear = std::make_shared<Linear>(3, 2);

    // Perform forward pass
    auto input = ones({1, 3}, DType::Float32);
    auto output1 = linear->forward(Variable(input));

    // Save and load
    auto path = get_test_path("roundtrip.bin");
    linear->save(path);

    auto linear2 = std::make_shared<Linear>(3, 2);
    linear2->load(path);

    // Perform same forward pass
    auto output2 = linear2->forward(Variable(input));

    // Results should be identical
    auto out1_data = output1.tensor().data<float>();
    auto out2_data = output2.tensor().data<float>();

    for (int i = 0; i < 2; ++i) {
        EXPECT_FLOAT_EQ(out1_data[i], out2_data[i]);
    }
}

// Test 17: Named parameters and buffers
TEST_F(SerializationTest, NamedParametersAndBuffers) {
    auto linear = std::make_shared<Linear>(3, 2);

    auto named_params = linear->named_parameters();
    EXPECT_EQ(named_params.size(), 2);

    bool has_weight = false;
    bool has_bias = false;

    for (const auto& [name, param] : named_params) {
        if (name == "weight") has_weight = true;
        if (name == "bias") has_bias = true;
    }

    EXPECT_TRUE(has_weight);
    EXPECT_TRUE(has_bias);
}

// Test 18: File overwrite
TEST_F(SerializationTest, FileOverwrite) {
    auto path = get_test_path("overwrite.bin");

    // Save first state
    std::unordered_map<std::string, Tensor> state1;
    state1["data"] = ones({2, 2}, DType::Float32);
    Serializer::save(state1, path);

    // Overwrite with different state
    std::unordered_map<std::string, Tensor> state2;
    state2["data"] = zeros({2, 2}, DType::Float32);
    Serializer::save(state2, path);

    // Load and verify it's the second state
    auto loaded = Serializer::load(path);
    auto data = loaded["data"].data<float>();

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}
