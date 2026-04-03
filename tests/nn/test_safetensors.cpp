/**
 * @file test_safetensors.cpp
 * @brief Tests for SafeTensors format serialization/deserialization
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/nn/safetensors.hpp"
#include "tenzor/nn/serialize.hpp"
#include <filesystem>
#include <fstream>

using namespace tenzor;

// Global initialization
class TenzorEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static auto* const g_env = ::testing::AddGlobalTestEnvironment(new TenzorEnv);

class SafeTensorsTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_path_ = "/tmp/tenzor_test_safetensors_" +
                      std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
                      ".safetensors";
    }

    void TearDown() override {
        std::filesystem::remove(test_path_);
    }

    std::string test_path_;
};

TEST_F(SafeTensorsTest, RoundTripFloat32) {
    std::unordered_map<std::string, Tensor> state;
    state["weight"] = rand({3, 4}, DType::Float32, Device::cpu());
    state["bias"] = zeros({4}, DType::Float32, Device::cpu());

    nn::SafeTensorsSerializer::save(state, test_path_);
    auto loaded = nn::SafeTensorsSerializer::load(test_path_);

    ASSERT_EQ(loaded.size(), 2u);
    ASSERT_TRUE(loaded.count("weight"));
    ASSERT_TRUE(loaded.count("bias"));
    EXPECT_EQ(loaded["weight"].shape()[0], 3);
    EXPECT_EQ(loaded["weight"].shape()[1], 4);
    EXPECT_EQ(loaded["weight"].dtype(), DType::Float32);

    auto* orig = state["weight"].data<float>();
    auto* load = loaded["weight"].data<float>();
    for (int64_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(orig[i], load[i]);
    }
}

TEST_F(SafeTensorsTest, RoundTripFloat64) {
    std::unordered_map<std::string, Tensor> state;
    state["param"] = ones({5, 5}, DType::Float64, Device::cpu());

    nn::SafeTensorsSerializer::save(state, test_path_);
    auto loaded = nn::SafeTensorsSerializer::load(test_path_);

    ASSERT_EQ(loaded.size(), 1u);
    EXPECT_EQ(loaded["param"].dtype(), DType::Float64);
}

TEST_F(SafeTensorsTest, EmptyStateDict) {
    std::unordered_map<std::string, Tensor> state;
    nn::SafeTensorsSerializer::save(state, test_path_);
    auto loaded = nn::SafeTensorsSerializer::load(test_path_);
    EXPECT_EQ(loaded.size(), 0u);
}

TEST_F(SafeTensorsTest, IsValidFile) {
    std::unordered_map<std::string, Tensor> state;
    state["x"] = zeros({2}, DType::Float32, Device::cpu());
    nn::SafeTensorsSerializer::save(state, test_path_);

    EXPECT_TRUE(nn::SafeTensorsSerializer::is_valid_file(test_path_));
    EXPECT_FALSE(nn::SafeTensorsSerializer::is_valid_file("/tmp/nonexistent_file_xyz"));
}

TEST_F(SafeTensorsTest, DetectFormat) {
    std::unordered_map<std::string, Tensor> state;
    state["x"] = zeros({2}, DType::Float32, Device::cpu());
    nn::SafeTensorsSerializer::save(state, test_path_);
    EXPECT_EQ(nn::detect_format(test_path_), nn::SerializeFormat::SafeTensors);

    std::string native_path = test_path_ + ".tnzr";
    nn::Serializer::save(state, native_path);
    EXPECT_EQ(nn::detect_format(native_path), nn::SerializeFormat::Tenzor);
    std::filesystem::remove(native_path);
}

TEST_F(SafeTensorsTest, InvalidFileTooSmall) {
    {
        std::ofstream f(test_path_, std::ios::binary);
        f << "abc";
    }
    EXPECT_THROW(nn::SafeTensorsSerializer::load(test_path_), std::runtime_error);
}

TEST_F(SafeTensorsTest, MultipleTensorsPreserveNames) {
    std::unordered_map<std::string, Tensor> state;
    state["layer1.weight"] = rand({10, 5}, DType::Float32, Device::cpu());
    state["layer1.bias"] = zeros({10}, DType::Float32, Device::cpu());
    state["layer2.weight"] = rand({3, 10}, DType::Float32, Device::cpu());

    nn::SafeTensorsSerializer::save(state, test_path_);
    auto loaded = nn::SafeTensorsSerializer::load(test_path_);

    ASSERT_EQ(loaded.size(), 3u);
    EXPECT_TRUE(loaded.count("layer1.weight"));
    EXPECT_TRUE(loaded.count("layer1.bias"));
    EXPECT_TRUE(loaded.count("layer2.weight"));
}
