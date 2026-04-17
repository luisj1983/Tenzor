/**
 * @file test_serialization_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for serialization
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/nn/serialize.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include <filesystem>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn;

class SerializationMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    std::string test_dir_;

    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        test_dir_ = "/tmp/tenzor_ser_multidtype_test_" +
                     std::to_string(::testing::UnitTest::GetInstance()->random_seed());
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
    }

    std::string get_test_path(const std::string& filename) {
        return test_dir_ + "/" + filename;
    }
};

TEST_P(SerializationMultiDTypeTest, SaveLoadStateDictBasic) {
    std::unordered_map<std::string, Tensor> state;
    state["weight"] = createOnes({2, 3});
    state["bias"] = createZeros({3});

    // Move to CPU for serialization
    for (auto& [k, v] : state) v = v.to(Device::cpu());

    auto path = get_test_path("state_dict.bin");
    Serializer::save(state, path);
    auto loaded_state = Serializer::load(path);

    ASSERT_EQ(loaded_state.size(), 2);
    ASSERT_TRUE(loaded_state.count("weight"));
    ASSERT_TRUE(loaded_state.count("bias"));
}

TEST_P(SerializationMultiDTypeTest, ModuleSaveLoad) {
    auto linear = std::make_shared<Linear>(4, 3);
    // Serialization works on CPU
    auto params = linear->parameters();
    params[0]->tensor().fill_(1.5f);
    params[1]->tensor().fill_(0.5f);

    auto path = get_test_path("linear.bin");
    linear->save(path);

    auto linear2 = std::make_shared<Linear>(4, 3);
    linear2->load(path);

    auto params2 = linear2->parameters();
    auto weight_data = params2[0]->tensor().data<float>();
    EXPECT_FLOAT_EQ(weight_data[0], 1.5f);
}

TEST_P(SerializationMultiDTypeTest, ModuleStateDict) {
    auto linear = std::make_shared<Linear>(3, 2);
    convert_model(linear);
    auto state = linear->state_dict();

    ASSERT_EQ(state.size(), 2);
    ASSERT_TRUE(state.count("weight"));
    ASSERT_TRUE(state.count("bias"));
}

TEST_P(SerializationMultiDTypeTest, RoundTripComputation) {
    auto linear = std::make_shared<Linear>(3, 2);
    auto input = ones({1, 3}, DType::Float32, Device::cpu());
    auto output1 = linear->forward(Variable(input));

    auto path = get_test_path("roundtrip.bin");
    linear->save(path);

    auto linear2 = std::make_shared<Linear>(3, 2);
    linear2->load(path);
    auto output2 = linear2->forward(Variable(input));

    auto out1_data = output1.tensor().data<float>();
    auto out2_data = output2.tensor().data<float>();
    for (int i = 0; i < 2; ++i) {
        EXPECT_FLOAT_EQ(out1_data[i], out2_data[i]);
    }
}

TEST_P(SerializationMultiDTypeTest, EmptyStateDict) {
    std::unordered_map<std::string, Tensor> state;
    auto path = get_test_path("empty.bin");
    Serializer::save(state, path);
    auto loaded_state = Serializer::load(path);
    EXPECT_EQ(loaded_state.size(), 0);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SerializationMultiDTypeTest);
