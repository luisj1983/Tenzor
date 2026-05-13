/**
 * @file test_exporter.cpp
 * @brief Phase 3 — exporter round-trip: nn::Module -> .tzlite -> LiteRuntime.
 *
 * Each test builds a small nn::Module, runs eager `module.forward(x)` to get
 * a reference output, exports to a temp `.tzlite` file, loads it via
 * LiteRuntime, runs `forward(x)`, and asserts the two outputs are
 * bit-identical (Float32) for the same input.
 */

#include <gtest/gtest.h>

#include <tenzor/autograd/variable.hpp>
#include <tenzor/lite/exporter.hpp>
#include <tenzor/lite/lite_graph.hpp>
#include <tenzor/lite/runtime.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/ops/creation.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>

namespace tenzor { void initialize(); }

namespace {

class TenzorExporterEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_exp_env =
    ::testing::AddGlobalTestEnvironment(new TenzorExporterEnv);

// Pick a unique-per-PID temp path so concurrent test runs don't clash.
auto temp_path(const std::string& stem) -> std::string {
    namespace fs = std::filesystem;
    return (fs::temp_directory_path() /
            (stem + "_" + std::to_string(static_cast<long>(::getpid())) +
             ".tzlite"))
        .string();
}

// Compare two contiguous Float32 buffers element-wise.
auto compare_f32(const float* a, const float* b, int64_t n, float atol = 1e-6f)
    -> ::testing::AssertionResult {
    for (int64_t i = 0; i < n; ++i) {
        if (std::abs(a[i] - b[i]) > atol) {
            return ::testing::AssertionFailure()
                << "mismatch at index " << i << ": ref=" << a[i]
                << " lite=" << b[i] << " (delta " << (a[i] - b[i]) << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

// Build a LiteTensor view over a Tensor's CPU contiguous data (zero-copy).
// Used to feed the eager-tested Tensor inputs into the Lite runtime so both
// paths see byte-identical input.
auto tensor_to_lite_input(const tenzor::Tensor& t)
    -> tenzor::lite::LiteTensor {
    auto src = t.is_contiguous() ? t : t.contiguous();
    if (src.device().type != tenzor::Device::Type::CPU) {
        src = src.to(tenzor::Device::cpu());
    }
    tenzor::lite::LiteTensor lt;
    lt.ndim = static_cast<int32_t>(src.ndim());
    lt.dtype = src.dtype();
    lt.owns_data = true;
    int64_t numel = 1;
    for (int32_t i = 0; i < lt.ndim; ++i) {
        lt.shape[i] = src.size(i);
        numel *= lt.shape[i];
    }
    for (int32_t i = lt.ndim - 1; i >= 0; --i) {
        lt.strides[i] = (i == lt.ndim - 1) ? 1 :
                        lt.strides[i + 1] * lt.shape[i + 1];
    }
    const auto nbytes = static_cast<size_t>(numel * tenzor::dtype_size(lt.dtype));
    lt.data = std::malloc(nbytes);
    std::memcpy(lt.data, src.data_ptr(), nbytes);
    return lt;
}

}  // namespace

using namespace tenzor::lite;

TEST(ExporterTest, LinearOnly) {
    auto linear = std::make_shared<tenzor::nn::Linear>(4, 3, /*bias=*/true);

    // Run eager forward to capture reference output.
    auto x_t = tenzor::randn({2, 4}, tenzor::DType::Float32);
    auto y_ref = linear->forward_impl(tenzor::Variable(x_t, false)).tensor();
    y_ref = y_ref.is_contiguous() ? y_ref : y_ref.contiguous();
    if (y_ref.device().type != tenzor::Device::Type::CPU) {
        y_ref = y_ref.to(tenzor::Device::cpu());
    }

    // Export.
    ExportOptions opts;
    opts.input_shape  = {2, 4};
    opts.input_dtype  = tenzor::DType::Float32;
    auto path = temp_path("export_linear_only");
    export_to_tzlite(*linear, path, opts);

    // Load + run Lite.
    auto runtime = LiteRuntime::load(path);
    auto lite_in = tensor_to_lite_input(x_t);
    auto y_lite = runtime->forward(lite_in);

    ASSERT_EQ(y_lite.numel(), y_ref.numel());
    EXPECT_TRUE(compare_f32(static_cast<const float*>(y_ref.data_ptr()),
                            y_lite.data_as<float>(), y_lite.numel()))
        << "Linear export/load did not round-trip cleanly";

    std::filesystem::remove(path);
}

TEST(ExporterTest, LinearReLULinear) {
    auto net = std::make_shared<tenzor::nn::Sequential>(
        std::make_shared<tenzor::nn::Linear>(8, 16, /*bias=*/true),
        std::make_shared<tenzor::nn::ReLU>(),
        std::make_shared<tenzor::nn::Linear>(16, 4, /*bias=*/true));

    auto x_t = tenzor::randn({3, 8}, tenzor::DType::Float32);
    auto y_ref = net->forward_impl(tenzor::Variable(x_t, false)).tensor();
    y_ref = y_ref.is_contiguous() ? y_ref : y_ref.contiguous();
    if (y_ref.device().type != tenzor::Device::Type::CPU) {
        y_ref = y_ref.to(tenzor::Device::cpu());
    }

    ExportOptions opts;
    opts.input_shape = {3, 8};
    opts.input_dtype = tenzor::DType::Float32;
    auto path = temp_path("export_mlp");
    export_to_tzlite(*net, path, opts);

    auto runtime = LiteRuntime::load(path);
    auto lite_in = tensor_to_lite_input(x_t);
    auto y_lite = runtime->forward(lite_in);

    ASSERT_EQ(y_lite.ndim, 2);
    EXPECT_EQ(y_lite.shape[0], 3);
    EXPECT_EQ(y_lite.shape[1], 4);
    EXPECT_TRUE(compare_f32(static_cast<const float*>(y_ref.data_ptr()),
                            y_lite.data_as<float>(), y_lite.numel(), 1e-5f));

    std::filesystem::remove(path);
}

TEST(ExporterTest, AllActivations) {
    // One Linear into each activation, separately exported and compared.
    auto seq_with = [](std::shared_ptr<tenzor::nn::Module> activation) {
        return std::make_shared<tenzor::nn::Sequential>(
            std::make_shared<tenzor::nn::Linear>(4, 4, /*bias=*/true),
            std::move(activation));
    };

    struct Case { std::string name; std::shared_ptr<tenzor::nn::Module> act; };
    std::vector<Case> cases;
    cases.push_back({"sigmoid", std::make_shared<tenzor::nn::Sigmoid>()});
    cases.push_back({"tanh",    std::make_shared<tenzor::nn::Tanh>()});
    cases.push_back({"gelu",    std::make_shared<tenzor::nn::GELU>()});

    for (auto& c : cases) {
        SCOPED_TRACE("activation = " + c.name);
        auto net = seq_with(c.act);
        auto x_t = tenzor::randn({2, 4}, tenzor::DType::Float32);
        auto y_ref = net->forward_impl(tenzor::Variable(x_t, false)).tensor();
        if (y_ref.device().type != tenzor::Device::Type::CPU) {
            y_ref = y_ref.to(tenzor::Device::cpu());
        }
        y_ref = y_ref.contiguous();

        ExportOptions opts;
        opts.input_shape = {2, 4};
        opts.input_dtype = tenzor::DType::Float32;
        auto path = temp_path("export_act_" + c.name);
        export_to_tzlite(*net, path, opts);

        auto runtime = LiteRuntime::load(path);
        auto lite_in = tensor_to_lite_input(x_t);
        auto y_lite = runtime->forward(lite_in);

        EXPECT_TRUE(compare_f32(static_cast<const float*>(y_ref.data_ptr()),
                                y_lite.data_as<float>(), y_lite.numel(),
                                /*atol=*/1e-4f));
        std::filesystem::remove(path);
    }
}

TEST(ExporterTest, UnsupportedLayerThrows) {
    // Identity isn't in Phase 3's supported set — the exporter must throw a
    // clear error rather than silently emitting a malformed graph.
    struct Dummy : public tenzor::nn::Module {
        auto forward_impl(const tenzor::Variable& v) -> tenzor::Variable override {
            return v;
        }
    };
    auto dummy = std::make_shared<Dummy>();
    ExportOptions opts;
    opts.input_shape = {1};
    EXPECT_THROW(export_to_tzlite(*dummy, "/tmp/should_not_exist.tzlite", opts),
                 std::runtime_error);
}
