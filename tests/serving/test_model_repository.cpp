/**
 * @file test_model_repository.cpp
 * @brief Tests for inference serving model repository and batcher
 */

#include <gtest/gtest.h>
#include <tenzor/serving/server.hpp>
#include <tenzor/jit/compiler.hpp>
#include <tenzor/jit/graph.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/tenzor.hpp>

#include <filesystem>
#include <unistd.h>

using namespace tenzor::serving;

namespace {
struct InitTenzorOnce : public ::testing::Environment {
    void SetUp() override { tenzor::initialize(); }
};
::testing::Environment* const _g_serving_init =
    ::testing::AddGlobalTestEnvironment(new InitTenzorOnce);
}  // namespace

TEST(ServingTest, ServerConfigDefaults) {
    ServerConfig config;
    EXPECT_EQ(config.http_port, 8080);
    EXPECT_EQ(config.grpc_port, 8081);
    EXPECT_EQ(config.num_workers, 4);
    EXPECT_TRUE(config.enable_metrics);
}

TEST(ServingTest, BatchConfigDefaults) {
    BatchConfig config;
    EXPECT_EQ(config.max_batch_size, 32);
    EXPECT_EQ(config.max_latency_us, 10000);
}

TEST(ServingTest, ModelRepositoryEmpty) {
    ModelRepository repo;
    auto models = repo.list_models();
    EXPECT_TRUE(models.empty());
    EXPECT_EQ(repo.get_model("nonexistent"), nullptr);
}

TEST(ServingTest, ModelRepositoryUnloadNonexistent) {
    ModelRepository repo;
    EXPECT_NO_THROW(repo.unload_model("nonexistent"));
}

TEST(ServingTest, MetricsRegistrySingleton) {
    auto& metrics = MetricsRegistry::instance();
    auto& m = metrics.get_metrics("test_model");
    EXPECT_EQ(m.total_requests.load(), 0);
    EXPECT_EQ(m.error_count.load(), 0);
}

TEST(ServingTest, MetricsPrometheusFormat) {
    auto& metrics = MetricsRegistry::instance();
    auto& m = metrics.get_metrics("test_prometheus");
    m.total_requests.store(100);
    m.total_latency_us.store(50000);

    auto text = metrics.format_prometheus();
    EXPECT_NE(text.find("tenzor_requests_total"), std::string::npos);
    EXPECT_NE(text.find("test_prometheus"), std::string::npos);
}

TEST(ServingTest, InferenceServerConstruction) {
    ServerConfig config;
    config.http_port = 0;  // Don't actually bind
    InferenceServer server(config);
    // Just verify construction doesn't crash
}

TEST(ServingTest, ModelStateEnum) {
    EXPECT_NE(static_cast<int>(ModelState::LOADING), static_cast<int>(ModelState::READY));
    EXPECT_NE(static_cast<int>(ModelState::READY), static_cast<int>(ModelState::FAILED));
}

// R1-03 regression: ModelRepository::load_model used to call the raw
// jit::load_graph()+CompiledModule::set_graph() pair instead of
// CompiledModule::load(), so a served model never got loaded_=true, never had
// a mark_dynamic_dims() configuration re-applied from the F031 serialized
// side-channel, and never got the ordinary shape/dtype/device compatibility
// guard (throw_if_loaded_shape_mismatch short-circuits on !loaded_). A model
// saved with dynamic dims and served through ModelRepository::load_model
// silently reverted to its trace-time concrete batch size with no error.
namespace {
auto build_relu_graph(tenzor::Device device = tenzor::Device::cpu())
    -> std::shared_ptr<tenzor::jit::Graph> {
    auto g = std::make_shared<tenzor::jit::Graph>();
    auto x = g->create_value("x", {4, 8}, tenzor::DType::Float32, device);
    auto y = g->create_value("y", {4, 8}, tenzor::DType::Float32, device);
    auto relu = g->create_node(tenzor::jit::OpType::ReLU, "relu");
    relu->add_input(x);
    relu->add_output(y);
    y->set_node(relu);
    g->add_node(relu);
    g->set_inputs({x});
    g->set_outputs({y});
    return g;
}
}  // namespace

TEST(ServingTest, LoadModelPreservesDynamicDims) {
    auto g = build_relu_graph();
    auto module = std::make_shared<tenzor::jit::CompiledModule>(g);
    module->mark_dynamic_dims({{0, 0, "B"}});
    ASSERT_TRUE(module->has_dynamic_shapes());

    const auto path = std::filesystem::temp_directory_path() /
                      ("tenzor_r1_03_dyn_" + std::to_string(::getpid()) + ".tzg");
    module->save(path.string());

    ModelRepository repo;
    BatchConfig batch_config;
    ASSERT_NO_THROW(repo.load_model("dyn_relu", path.string(),
                                    tenzor::Device::cpu(), batch_config));
    std::filesystem::remove(path);

    auto entry = repo.get_model("dyn_relu");
    ASSERT_NE(entry, nullptr);
    ASSERT_NE(entry->module, nullptr);
    EXPECT_TRUE(entry->module->has_dynamic_shapes())
        << "a dynamic-dims model served via ModelRepository::load_model must "
           "not silently revert to a static, trace-time-batch-size graph";

    // A different batch size than the one saved at (4) must be accepted.
    auto raw7 = tenzor::randn({7, 8}, tenzor::DType::Float32, tenzor::Device::cpu());
    auto input7 = tenzor::Variable(raw7, /*requires_grad=*/false);
    tenzor::Variable out7;
    ASSERT_NO_THROW({ out7 = entry->module->forward(input7); })
        << "a model served via ModelRepository::load_model must accept a "
           "batch size other than the one it was traced/saved with";
    EXPECT_EQ(out7.tensor().shape()[0], 7);

    repo.unload_model("dyn_relu");
}

TEST(ServingTest, LoadModelRejectsDtypeMismatchOnNonDynamicModel) {
    // Uses a DTYPE mismatch (not shape) specifically: ShapeGuardInsertionPass
    // (run by optimize_for_inference()) already bakes in and rejects a shape
    // change independent of this fix, so a shape-mismatch test wouldn't
    // distinguish "the R1-02/R1-03 loaded-module guard ran" from "some other
    // pre-existing guard happened to also catch it". throw_if_loaded_shape_
    // mismatch is the ONLY guard that checks dtype, and it's a no-op unless
    // loaded_ is true — which requires going through CompiledModule::load()
    // rather than ModelRepository's previous raw set_graph() bypass.
    auto g = build_relu_graph();
    auto module = std::make_shared<tenzor::jit::CompiledModule>(g);
    ASSERT_FALSE(module->has_dynamic_shapes());

    const auto path = std::filesystem::temp_directory_path() /
                      ("tenzor_r1_03_static_" + std::to_string(::getpid()) + ".tzg");
    module->save(path.string());

    ModelRepository repo;
    BatchConfig batch_config;
    ASSERT_NO_THROW(repo.load_model("static_relu", path.string(),
                                    tenzor::Device::cpu(), batch_config));
    std::filesystem::remove(path);

    auto entry = repo.get_model("static_relu");
    ASSERT_NE(entry, nullptr);
    ASSERT_NE(entry->module, nullptr);

    // Same shape+dtype (4,8 Float32) as saved: must work.
    auto raw4 = tenzor::randn({4, 8}, tenzor::DType::Float32, tenzor::Device::cpu());
    ASSERT_NO_THROW({
        (void)entry->module->forward(tenzor::Variable(raw4, false));
    });

    // Same SHAPE but a different DTYPE must be rejected with a clear error
    // rather than silently replaying Float32-baked constants against a
    // Float64 input (previously: ModelRepository's bypass of
    // CompiledModule::load() meant loaded_ was never set, so
    // throw_if_loaded_shape_mismatch's dtype check never ran at all).
    auto raw4_f64 = tenzor::randn({4, 8}, tenzor::DType::Float64, tenzor::Device::cpu());
    EXPECT_THROW(
        { (void)entry->module->forward(tenzor::Variable(raw4_f64, false)); },
        std::runtime_error)
        << "a non-dynamic model served via ModelRepository::load_model must "
           "reject a dtype it wasn't traced/saved with";

    repo.unload_model("static_relu");
}

namespace {
auto device_available(const tenzor::Device& dev) -> bool {
    try {
        auto t = tenzor::zeros({2, 2}, tenzor::DType::Float32, dev);
        (void)t;
        return true;
    } catch (...) {
        return false;
    }
}

auto short_device_name(tenzor::Device::Type t) -> std::string {
    switch (t) {
        case tenzor::Device::Type::CUDA:   return "cuda";
        case tenzor::Device::Type::ROCm:   return "rocm";
        case tenzor::Device::Type::Vulkan: return "vulkan";
        case tenzor::Device::Type::OneAPI: return "oneapi";
        case tenzor::Device::Type::MPS:    return "mps";
        default:                          return "cpu";
    }
}
}  // namespace

// Finding 41: ModelRepository::load_model's Device parameter (stored as
// entry->device, and later used at src/serving/server.cpp:1065 to place
// every inference request's input tensor via tensor_from_json) was only
// ever exercised with Device::cpu() anywhere in this file -- a bug specific
// to how load_model/predict-path input placement behaves for a non-CPU
// device would ship with zero signal from tests/serving/. Parameterized
// over every non-CPU backend the same way tests/backend_parity/ does.
class ModelRepositoryDeviceTest : public ::testing::TestWithParam<tenzor::Device::Type> {};

TEST_P(ModelRepositoryDeviceTest, LoadModelAndForwardOnNonCpuDevice) {
    const tenzor::Device dev{GetParam(), 0};
    if (!device_available(dev)) {
        GTEST_SKIP() << dev.to_string() << " not available";
    }

    // Trace/build the graph's inputs ON the target device -- a loaded
    // CompiledModule cannot retrace to a different device than it was
    // saved for (see the fail-fast check now in
    // ModelRepository::load_model), so this test exercises the correct,
    // supported usage: load_model's device parameter must match how the
    // model was actually traced/saved.
    auto g = build_relu_graph(dev);
    auto module = std::make_shared<tenzor::jit::CompiledModule>(g);

    const auto path = std::filesystem::temp_directory_path() /
                      ("tenzor_r1_41_" + std::to_string(::getpid()) + "_" +
                       std::to_string(static_cast<int>(GetParam())) + ".tzg");
    module->save(path.string());

    ModelRepository repo;
    BatchConfig batch_config;
    ASSERT_NO_THROW(repo.load_model("nc_relu", path.string(), dev, batch_config))
        << "load_model must accept a non-CPU device";
    std::filesystem::remove(path);

    auto entry = repo.get_model("nc_relu");
    ASSERT_NE(entry, nullptr);
    ASSERT_NE(entry->module, nullptr);
    EXPECT_EQ(entry->device.type, dev.type)
        << "ModelEntry::device must reflect the device load_model was called with";

    // Mirrors the real inference path (server.cpp:1063-1065): the input
    // tensor is placed on entry->device (here: the same non-CPU device),
    // not necessarily the device the module was saved/traced on.
    auto raw = tenzor::randn({4, 8}, tenzor::DType::Float32, dev);
    auto input = tenzor::Variable(raw, /*requires_grad=*/false);
    tenzor::Variable out;
    ASSERT_NO_THROW({ out = entry->module->forward(input); })
        << "forward() with an input placed on entry->device must not throw "
           "for " << dev.to_string();
    EXPECT_EQ(out.tensor().shape()[0], 4);
    EXPECT_EQ(out.tensor().device().type, dev.type)
        << "output should stay on the serving device, not silently migrate";

    // ReLU(x) must be non-negative everywhere -- correctness, not just
    // "didn't throw", so a backend that silently produces garbage (rather
    // than a clean device-mismatch exception) is still caught.
    auto out_cpu = out.tensor().to(tenzor::Device::cpu()).contiguous();
    const float* data = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_GE(data[i], 0.0f) << "ReLU output must be non-negative at index " << i;
    }

    repo.unload_model("nc_relu");
}

// Regression for the bug LoadModelAndForwardOnNonCpuDevice above uncovered:
// ModelRepository::load_model's `device` parameter used to be pure metadata
// (stored on ModelEntry, never validated against how the model file was
// actually traced/saved) -- requesting device=cuda for a CPU-traced model
// loaded "successfully" and only failed, confusingly, on the FIRST real
// inference request (server.cpp's predict handler places every request's
// input on model->device before calling forward()). Fixed by validating at
// load_model() time and failing immediately with a clear message.
TEST(ServingTest, LoadModelRejectsDeviceMismatchImmediately) {
    if (!device_available(tenzor::Device::cuda(0))) {
        GTEST_SKIP() << "cuda:0 not available";
    }
    auto g = build_relu_graph(tenzor::Device::cpu());  // traced/saved for CPU
    auto module = std::make_shared<tenzor::jit::CompiledModule>(g);

    const auto path = std::filesystem::temp_directory_path() /
                      ("tenzor_r1_41_mismatch_" + std::to_string(::getpid()) + ".tzg");
    module->save(path.string());

    ModelRepository repo;
    BatchConfig batch_config;
    EXPECT_THROW(
        repo.load_model("mismatch_relu", path.string(), tenzor::Device::cuda(0), batch_config),
        std::invalid_argument)
        << "load_model must reject device=cuda for a model traced/saved on cpu "
           "immediately, not defer the failure to the first inference request";
    std::filesystem::remove(path);

    EXPECT_EQ(repo.get_model("mismatch_relu"), nullptr)
        << "a model that failed the device check must not be registered";
}

INSTANTIATE_TEST_SUITE_P(
    AllNonCpuBackends, ModelRepositoryDeviceTest,
    ::testing::Values(tenzor::Device::Type::CUDA, tenzor::Device::Type::ROCm,
                      tenzor::Device::Type::Vulkan, tenzor::Device::Type::OneAPI,
                      tenzor::Device::Type::MPS),
    [](const ::testing::TestParamInfo<tenzor::Device::Type>& info) {
        return short_device_name(info.param);
    });
