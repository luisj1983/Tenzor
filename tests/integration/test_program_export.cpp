/**
 * @file test_program_export.cpp
 * @brief Round-trip tests for the AOT program export pipeline.
 *
 * Phase 5.1 of the test-coverage campaign. Verifies:
 *   - export_model() traces a module and produces a runnable ExportedProgram
 *   - save() / load() preserves the program byte-for-byte (numeric equivalence)
 *   - run() on the loaded program matches eager inference
 *
 * Uses a small Linear-only MLP so the trace, save, load, and inference paths
 * each fit in a few seconds.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/export/export.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/ops/creation.hpp>
#include <cstdio>

using namespace tenzor;

namespace {

// Ensure backends are registered before any TEST() runs. Without this the
// gtest_main entrypoint never calls tenzor::initialize() and the first
// dispatch in export_model() fails with "Backend not available for device:
// cpu".
struct TenzorInitEnv : ::testing::Environment {
    void SetUp() override { tenzor::initialize(); }
};
const auto* kTenzorInitEnv =
    ::testing::AddGlobalTestEnvironment(new TenzorInitEnv);

}  // namespace

namespace {

class TwoLayerMLP : public nn::Module {
public:
    std::shared_ptr<nn::Linear> fc1;
    std::shared_ptr<nn::Linear> fc2;

    TwoLayerMLP(int64_t in_features, int64_t hidden, int64_t out_features) {
        fc1 = std::make_shared<nn::Linear>(in_features, hidden);
        fc2 = std::make_shared<nn::Linear>(hidden, out_features);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        return fc2->forward(fc1->forward(input));
    }
};

}  // namespace

TEST(ProgramExport, RoundTripPreservesNumerics) {
    auto model = std::make_shared<TwoLayerMLP>(8, 4, 2);
    model->eval();

    auto input = randn({3, 8}, DType::Float32, Device::cpu());

    // Eager reference output.
    auto eager_var = model->forward(Variable(input, false));
    auto eager_out = eager_var.tensor();

    // Trace + export.
    auto exported = export_::export_model(*model, {input});
    EXPECT_EQ(exported.num_inputs(), 1u);
    EXPECT_GT(exported.num_outputs(), 0u);
    EXPECT_FALSE(exported.state_dict().empty())
        << "exported state dict should include fc1/fc2 weights and biases";

    // Save and reload.
    const std::string path = "/tmp/tenzor_program_export_roundtrip.tzep";
    exported.save(path);
    auto loaded = export_::ExportedProgram::load(path);
    std::remove(path.c_str());

    // Run loaded program.
    auto outs = loaded.run({input});
    ASSERT_EQ(outs.size(), 1u);

    // Compare element-wise (very tight tolerance — same backend, same kernel).
    auto a = outs[0].contiguous();
    auto b = eager_out.contiguous();
    ASSERT_EQ(a.numel(), b.numel());
    auto* da = a.data<float>();
    auto* db = b.data<float>();
    for (int64_t i = 0; i < a.numel(); ++i) {
        EXPECT_NEAR(da[i], db[i], 1e-5f)
            << "loaded program output diverged from eager at index " << i;
    }
}

TEST(ProgramExport, NumInputsOutputs) {
    auto model = std::make_shared<TwoLayerMLP>(4, 4, 4);
    model->eval();
    auto x = randn({2, 4}, DType::Float32, Device::cpu());
    auto exported = export_::export_model(*model, {x});
    EXPECT_EQ(exported.num_inputs(), 1u);
    // num_outputs is set by the trace; we don't pin a specific value, just
    // assert the trace recorded at least one output.
    EXPECT_GE(exported.num_outputs(), 1u);
}

// Audit D.2: load(path, map_location=cpu) materialises every state
// tensor on the requested device, even if the saved device was different.
// We exercise the CPU->CPU override (always available) and verify the
// loaded tensors live on CPU and produce the right numerics.
TEST(ProgramExport, LoadMapLocationCpu) {
    auto model = std::make_shared<TwoLayerMLP>(4, 4, 2);
    model->eval();
    auto x = randn({2, 4}, DType::Float32, Device::cpu());
    auto exported = export_::export_model(*model, {x});

    const std::string path = "/tmp/tenzor_test_program_export_maploc.tzep";
    exported.save(path);
    auto loaded = export_::ExportedProgram::load(path, Device::cpu());
    std::remove(path.c_str());

    // Every state tensor lives on CPU.
    for (const auto& [name, t] : loaded.state_dict()) {
        EXPECT_EQ(t.device().type, Device::Type::CPU)
            << "state tensor '" << name
            << "' was not retargeted to CPU";
    }
    // Numeric parity with the original.
    auto out_orig   = exported.run({x})[0].contiguous();
    auto out_loaded = loaded.run({x})[0].contiguous();
    ASSERT_EQ(out_orig.numel(), out_loaded.numel());
    const auto* op = out_orig.data<float>();
    const auto* lp = out_loaded.data<float>();
    for (int64_t i = 0; i < out_orig.numel(); ++i) {
        EXPECT_FLOAT_EQ(op[i], lp[i]) << "i=" << i;
    }
}
