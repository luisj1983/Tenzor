/**
 * @file test_rpc_agent.cpp
 * @brief Tests for RPC agent and function registry
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/rpc/rpc.hpp>
#include <tenzor/distributed/rpc/function_registry.hpp>
#include <tenzor/distributed/rpc/rpc_agent.hpp>

using namespace tenzor;
using namespace tenzor::distributed::rpc;

class RpcTestEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
    void TearDown() override { tenzor::finalize(); }
};
static auto* _env = ::testing::AddGlobalTestEnvironment(new RpcTestEnv());

TEST(RpcAgentTest, FunctionRegistrySingleton) {
    auto& reg = FunctionRegistry::instance();
    EXPECT_FALSE(reg.has_function("nonexistent"));
}

TEST(RpcAgentTest, RegisterAndLookupFunction) {
    auto& reg = FunctionRegistry::instance();
    reg.register_function("test_add", [](const std::vector<Tensor>& args) -> std::vector<Tensor> {
        if (args.size() >= 2) {
            return {args[0] + args[1]};
        }
        return {};
    });

    EXPECT_TRUE(reg.has_function("test_add"));
    auto* fn = reg.get_function("test_add");
    EXPECT_NE(fn, nullptr);
}

TEST(RpcAgentTest, ExecuteRegisteredFunction) {
    auto& reg = FunctionRegistry::instance();
    reg.register_function("test_mul2", [](const std::vector<Tensor>& args) -> std::vector<Tensor> {
        return {args[0] * 2.0f};
    });

    auto* fn = reg.get_function("test_mul2");
    ASSERT_NE(fn, nullptr);

    auto input = tenzor::ones({3}, DType::Float32, Device::cpu());
    auto result = (*fn)({input});
    ASSERT_EQ(result.size(), 1);
    EXPECT_NEAR(result[0].data<float>()[0], 2.0f, 1e-6f);
}

TEST(RpcAgentTest, AgentConstruction) {
    WorkerInfo self;
    self.name = "worker_0";
    self.id = 0;
    self.address = "127.0.0.1";
    self.port = 29500;

    TcpRpcAgent agent(self);
    EXPECT_FALSE(agent.is_running());
    EXPECT_EQ(agent.self().id, 0);
    EXPECT_EQ(agent.self().name, "worker_0");
}

TEST(RpcAgentTest, MessageTypes) {
    EXPECT_NE(static_cast<int>(MessageType::RPC_CALL), static_cast<int>(MessageType::RPC_RESPONSE));
    EXPECT_NE(static_cast<int>(MessageType::HEARTBEAT), static_cast<int>(MessageType::HEARTBEAT_ACK));
}

TEST(RpcAgentTest, SerializedPayloadDefault) {
    SerializedPayload payload;
    EXPECT_TRUE(payload.function_name.empty());
    EXPECT_TRUE(payload.bytes.empty());
    EXPECT_TRUE(payload.tensors.empty());
    EXPECT_EQ(payload.request_id, 0);
}
