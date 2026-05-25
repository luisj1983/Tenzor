/**
 * @file test_model_persistence_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for model save/load persistence
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <filesystem>
#include <unistd.h>  // FF.26: getpid for per-process temp path

using namespace tenzor;
using namespace tenzor::testing;

class ModelPersistenceMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(ModelPersistenceMultiDTypeTest, StateDictRoundTrip) {
    auto model = std::make_shared<nn::Linear>(16, 8);
    convert_model(model);

    // Get state dict
    auto state = model->state_dict();
    EXPECT_FALSE(state.empty());

    // Create new model and load state
    auto model2 = std::make_shared<nn::Linear>(16, 8);
    convert_model(model2);
    model2->load_state_dict(state);

    // Forward pass should produce same output
    auto input = createInput({2, 16}, false);
    auto out1 = model->forward(input);
    auto out2 = model2->forward(input);

    expectTensorNear(out1.tensor(), out2.tensor());
}

TEST_P(ModelPersistenceMultiDTypeTest, SaveLoadFile) {
    auto model = std::make_shared<nn::Linear>(8, 4);
    convert_model(model);

    // FF.26: previously this was a hardcoded /tmp path keyed only on
    // backend_name(), which raced when two parallel test processes ran on the
    // same host. Mirror tests/utils/test_logging.cpp:19's pid+test-name pattern.
    const auto* info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_name = info ? info->name() : "unknown";
    std::string path = (std::filesystem::temp_directory_path() /
                        ("tenzor_test_model_" +
                         std::to_string(::getpid()) + "_" + test_name + "_" +
                         backend_name() + ".bin")).string();

    // Save
    model->save(path);
    EXPECT_TRUE(std::filesystem::exists(path));

    // Load into new model
    auto model2 = std::make_shared<nn::Linear>(8, 4);
    convert_model(model2);
    model2->load(path);

    // Verify
    auto input = createInput({1, 8}, false);
    auto out1 = model->forward(input);
    auto out2 = model2->forward(input);
    expectTensorNear(out1.tensor(), out2.tensor());

    // Cleanup
    std::filesystem::remove(path);
}

TEST_P(ModelPersistenceMultiDTypeTest, DeviceTransferAfterLoad) {
    auto model = std::make_shared<nn::Linear>(8, 4);
    // Create on CPU, save
    auto state = model->state_dict();

    // Load on test device
    auto model2 = std::make_shared<nn::Linear>(8, 4);
    model2->load_state_dict(state);
    model2->to(device());
    if (dtype() != DType::Float32) model2->to(dtype());

    auto input = createInput({1, 8}, false);
    auto output = model2->forward(input);
    expectDevice(output.tensor());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ModelPersistenceMultiDTypeTest);
