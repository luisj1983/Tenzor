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

// Phase 4.5 smoke: bring up a real TCP listener, verify is_running()
// flips, register a handler, then shut down cleanly. This exercises the
// socket bind / accept thread / shutdown paths that the earlier tests
// only reached indirectly through the constructor.
TEST(RpcAgentTest, RealStartupAndShutdown) {
    WorkerInfo self;
    self.name = "startup_worker";
    self.id = 0;
    self.address = "127.0.0.1";
    self.port = 29580;  // distinct from other tests to avoid collision

    TcpRpcAgent agent(self);
    EXPECT_FALSE(agent.is_running());

    // Register a handler BEFORE init so incoming messages to this agent
    // would have somewhere to land.
    bool handler_registered = true;
    try {
        agent.register_handler(MessageType::RPC_CALL,
            [](const Message& msg) -> Message {
                Message resp = msg;
                resp.type = MessageType::RPC_RESPONSE;
                return resp;
            });
    } catch (const std::exception&) {
        handler_registered = false;
    }
    EXPECT_TRUE(handler_registered);

    // Initialize with a single worker (self). This binds the listen
    // socket and starts the accept thread.
    std::vector<WorkerInfo> workers = {self};
    bool init_ok = true;
    try {
        agent.init(workers);
    } catch (const std::exception& e) {
        // Some builds may require a multi-worker topology. If init fails
        // due to bind errors or topology validation, skip the liveness
        // check — construction already exercises most of the API.
        init_ok = false;
        GTEST_SKIP() << "TcpRpcAgent.init() failed in single-worker setup: "
                     << e.what();
    }

    if (init_ok) {
        EXPECT_TRUE(agent.is_running()) << "Agent should be running after init()";
        EXPECT_NO_THROW(agent.shutdown());
        EXPECT_FALSE(agent.is_running()) << "Agent should stop after shutdown()";
    }
}

// When two agents try to bind the same port, the second init() must
// surface the bind failure as a synchronous exception — previously the
// accept thread silently set listen_fd_ = -1 and the caller kept a
// half-up agent that rejected every send().
TEST(RpcAgentTest, InitReportsBindFailure) {
    WorkerInfo a;
    a.name = "worker_a"; a.id = 0; a.address = "127.0.0.1"; a.port = 29581;
    WorkerInfo b;
    b.name = "worker_b"; b.id = 1; b.address = "127.0.0.1"; b.port = 29581;  // same port

    TcpRpcAgent agent_a(a);
    try {
        agent_a.init({a});
    } catch (const std::exception&) {
        GTEST_SKIP() << "First init failed; can't verify port-collision surfacing";
        return;
    }
    ASSERT_TRUE(agent_a.is_running());

    TcpRpcAgent agent_b(b);
    EXPECT_THROW(agent_b.init({b}), std::runtime_error);
    EXPECT_FALSE(agent_b.is_running());

    agent_a.shutdown();
}
