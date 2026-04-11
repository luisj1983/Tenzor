/**
 * @file test_dynamic_batcher.cpp
 * @brief Tests for DynamicBatcher and BatchConfig
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/serving/server.hpp>
#include <tenzor/jit/compiler.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/ops/creation.hpp>
#include <memory>
#include <vector>

class TestEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static ::testing::Environment* const env = ::testing::AddGlobalTestEnvironment(new TestEnv);

using namespace tenzor::serving;

TEST(DynamicBatcherTest, BatchConfigDefaults) {
    BatchConfig config;
    EXPECT_EQ(config.max_batch_size, 32);
    EXPECT_EQ(config.max_latency_us, 10000);
}

TEST(DynamicBatcherTest, BatchConfigCustom) {
    BatchConfig config;
    config.max_batch_size = 64;
    config.max_latency_us = 5000;
    EXPECT_EQ(config.max_batch_size, 64);
    EXPECT_EQ(config.max_latency_us, 5000);
}

TEST(DynamicBatcherTest, InferRequestCreation) {
    auto tensor = tenzor::zeros({2, 3});
    auto before = std::chrono::steady_clock::now();
    InferRequest req(tensor);
    auto after = std::chrono::steady_clock::now();

    EXPECT_EQ(req.input.shape()[0], 2);
    EXPECT_EQ(req.input.shape()[1], 3);
    EXPECT_GE(req.arrival, before);
    EXPECT_LE(req.arrival, after);
}

TEST(DynamicBatcherTest, InferRequestMovesTensor) {
    auto tensor = tenzor::ones({4, 4});
    InferRequest req(std::move(tensor));
    // The moved-into request should have the correct shape
    EXPECT_EQ(req.input.shape()[0], 4);
    EXPECT_EQ(req.input.shape()[1], 4);
}

TEST(DynamicBatcherTest, ServerConfigRateLimitDefaults) {
    ServerConfig config;
    EXPECT_FALSE(config.enable_rate_limit);
    EXPECT_DOUBLE_EQ(config.rate_limit_rps, 100.0);
    EXPECT_EQ(config.rate_limit_burst, 200);
}

TEST(DynamicBatcherTest, ServerConfigAuthDefaults) {
    ServerConfig config;
    EXPECT_FALSE(config.enable_auth);
    EXPECT_TRUE(config.api_keys.empty());
    EXPECT_EQ(config.auth_header, "Authorization");
}

// ============================================================================
// Phase 6.5: end-to-end batch execution.
// Previously this file only validated config struct defaults and
// InferRequest construction. Add a real batching round-trip: trace a
// Linear layer via jit::CompiledModule::trace, feed it through a
// DynamicBatcher, and confirm that concurrently-submitted requests all
// receive tensors of the expected shape.
// ============================================================================

// Phase 6.4: end-to-end serving roundtrip. The chain is:
//   1. TracingGuard's constructor installs the DispatchInterceptor so
//      ops passing through `module->forward(dummy_input)` are recorded
//      (fix in src/jit/tracer.cpp earlier in this branch).
//   2. The tracer captures module parameters (Linear's weight/bias) as
//      graph constants stored in Graph::constants_ via end_trace().
//   3. Graph::forward pre-populates the runtime value_map with those
//      constants before executing any node.
//   4. DynamicBatcher dispatches through the jit::CompiledModule and
//      returns results.
TEST(DynamicBatcherTest, SubmitBatchRoundtrip) {
    // Trace a small Linear layer so the batcher has a real CompiledModule
    // to dispatch against.
    auto model = std::make_shared<tenzor::nn::Linear>(/*in=*/4, /*out=*/2, /*bias=*/true);
    auto example_input = tenzor::ones({1, 4}, tenzor::DType::Float32, tenzor::Device::cpu());
    auto compiled = tenzor::jit::CompiledModule::trace(model, example_input);
    ASSERT_NE(compiled, nullptr);

    BatchConfig cfg;
    cfg.max_batch_size = 4;
    cfg.max_latency_us = 5000;  // 5 ms flush window
    DynamicBatcher batcher(compiled, cfg);
    batcher.start();

    // Submit 4 requests. The batcher should combine them into a single
    // forward pass (max_batch_size=4) and return 4 per-sample results.
    std::vector<std::future<tenzor::Tensor>> futures;
    for (int i = 0; i < 4; ++i) {
        auto input = tenzor::ones({1, 4}, tenzor::DType::Float32, tenzor::Device::cpu());
        futures.push_back(batcher.submit(std::move(input)));
    }

    for (auto& f : futures) {
        // Timeout in case batching hangs — the test should complete in
        // well under 1 second.
        auto status = f.wait_for(std::chrono::seconds(5));
        ASSERT_EQ(status, std::future_status::ready)
            << "DynamicBatcher.submit() did not complete within 5 s";
        auto result = f.get();
        // Shape must match the Linear's output for a single sample:
        // (1, out_features). The batcher splits the combined batched
        // output back into per-request tensors.
        ASSERT_EQ(result.shape().size(), 2u);
        EXPECT_EQ(result.shape()[0], 1);
        EXPECT_EQ(result.shape()[1], 2);
    }

    batcher.stop();
}
