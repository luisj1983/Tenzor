// Validates that quantize_dynamic(QuantizationConfig) honors per-layer
// overrides and skip_layers (regression for the silently-discarded-override
// bug). Sequential children are named by positional index.
#include <gtest/gtest.h>
#include <memory>

#include "tenzor/tenzor.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/quantization/quantize_api.hpp"
#include "tenzor/nn/quantization/qconfig.hpp"
#include "tenzor/nn/quantization/quantized_layers.hpp"

using namespace tenzor;
using namespace tenzor::quantization;
using nn::quantization::QConfig;
using nn::quantization::DefaultQConfigs;
using nn::quantization::QuantizedLinear;

TEST(QuantizePerLayerConfig, SkipAndOverrideHonored) {
    auto model = std::make_shared<nn::Sequential>();
    model->add_module(std::make_shared<nn::Linear>(4, 4));   // index "0" -> skipped
    model->add_module(std::make_shared<nn::Linear>(4, 4));   // index "1" -> overridden
    model->add_module(std::make_shared<nn::Linear>(4, 4));   // index "2" -> default

    QuantizationConfig cfg;
    cfg.default_config = DefaultQConfigs::default_qconfig();
    cfg.skip_layers.insert("0");
    cfg.layer_overrides.insert({"1", DefaultQConfigs::fast_qconfig()});

    auto q = quantize_dynamic(model, cfg);
    auto seq = std::dynamic_pointer_cast<nn::Sequential>(q);
    ASSERT_NE(seq, nullptr);
    const auto& mods = seq->modules();
    ASSERT_EQ(mods.size(), 3u);

    // Index 0 was skipped -> still a plain Linear, NOT quantized.
    EXPECT_EQ(std::dynamic_pointer_cast<QuantizedLinear>(mods[0]), nullptr)
        << "skip_layers entry '0' was quantized anyway";
    EXPECT_NE(std::dynamic_pointer_cast<nn::Linear>(mods[0]), nullptr);

    // Index 1 (override) and index 2 (default) -> quantized.
    EXPECT_NE(std::dynamic_pointer_cast<QuantizedLinear>(mods[1]), nullptr)
        << "override layer '1' was not quantized";
    EXPECT_NE(std::dynamic_pointer_cast<QuantizedLinear>(mods[2]), nullptr)
        << "default-config layer '2' was not quantized";
}

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
