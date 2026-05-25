/**
 * @file test_safetensors_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for SafeTensors serialization
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/nn/safetensors.hpp"
#include "tenzor/nn/serialize.hpp"
#include <filesystem>
#include <unistd.h>  // HH.26: getpid() to disambiguate parallel ctest shards

using namespace tenzor;
using namespace tenzor::testing;

class SafeTensorsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    std::string test_path_;

    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        // HH.26: include pid so parallel ctest shards (sharing the gtest
        // random seed) don't collide on the same temp file.
        test_path_ = (std::filesystem::temp_directory_path() /
                      ("tenzor_safetensors_multidtype_" +
                       std::to_string(::getpid()) + "_" +
                       std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
                       ".safetensors")).string();
    }

    void TearDown() override {
        std::filesystem::remove(test_path_);
    }
};

TEST_P(SafeTensorsMultiDTypeTest, RoundTripFloat) {
    // SafeTensors works with CPU tensors
    std::unordered_map<std::string, Tensor> state;
    state["weight"] = tenzor::randn({3, 4}, DType::Float32, Device::cpu());
    state["bias"] = tenzor::zeros({4}, DType::Float32, Device::cpu());

    nn::SafeTensorsSerializer::save(state, test_path_);
    auto loaded = nn::SafeTensorsSerializer::load(test_path_);

    ASSERT_EQ(loaded.size(), 2u);
    ASSERT_TRUE(loaded.count("weight"));
    EXPECT_EQ(loaded["weight"].shape()[0], 3);
    EXPECT_EQ(loaded["weight"].shape()[1], 4);
}

TEST_P(SafeTensorsMultiDTypeTest, EmptyStateDict) {
    std::unordered_map<std::string, Tensor> state;
    nn::SafeTensorsSerializer::save(state, test_path_);
    auto loaded = nn::SafeTensorsSerializer::load(test_path_);
    EXPECT_EQ(loaded.size(), 0u);
}

TEST_P(SafeTensorsMultiDTypeTest, IsValidFile) {
    std::unordered_map<std::string, Tensor> state;
    state["x"] = tenzor::zeros({2}, DType::Float32, Device::cpu());
    nn::SafeTensorsSerializer::save(state, test_path_);
    EXPECT_TRUE(nn::SafeTensorsSerializer::is_valid_file(test_path_));
    EXPECT_FALSE(nn::SafeTensorsSerializer::is_valid_file("/tmp/nonexistent_xyz"));
}

TEST_P(SafeTensorsMultiDTypeTest, MultipleTensorsPreserveNames) {
    std::unordered_map<std::string, Tensor> state;
    state["layer1.weight"] = tenzor::randn({10, 5}, DType::Float32, Device::cpu());
    state["layer1.bias"] = tenzor::zeros({10}, DType::Float32, Device::cpu());
    state["layer2.weight"] = tenzor::randn({3, 10}, DType::Float32, Device::cpu());

    nn::SafeTensorsSerializer::save(state, test_path_);
    auto loaded = nn::SafeTensorsSerializer::load(test_path_);

    ASSERT_EQ(loaded.size(), 3u);
    EXPECT_TRUE(loaded.count("layer1.weight"));
    EXPECT_TRUE(loaded.count("layer1.bias"));
    EXPECT_TRUE(loaded.count("layer2.weight"));
}

TEST_P(SafeTensorsMultiDTypeTest, DetectFormat) {
    std::unordered_map<std::string, Tensor> state;
    state["x"] = tenzor::zeros({2}, DType::Float32, Device::cpu());
    nn::SafeTensorsSerializer::save(state, test_path_);
    EXPECT_EQ(nn::detect_format(test_path_), nn::SerializeFormat::SafeTensors);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SafeTensorsMultiDTypeTest);
