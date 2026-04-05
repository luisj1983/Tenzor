/**
 * @file test_rref.cpp
 * @brief Tests for remote references (RRef)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/rpc/rref.hpp>

using namespace tenzor;
using namespace tenzor::distributed::rpc;

class RRefTestEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
    void TearDown() override { tenzor::finalize(); }
};
static auto* _env = ::testing::AddGlobalTestEnvironment(new RRefTestEnv());

TEST(RRefTest, RRefStoreBasicOperations) {
    auto& store = RRefStore::instance();

    auto tensor = tenzor::ones({3, 4}, DType::Float32, Device::cpu());
    auto id = store.store(tensor);
    EXPECT_GE(id, 0);

    auto fetched = store.fetch(id);
    EXPECT_EQ(fetched.numel(), 12);

    store.remove(id);
    EXPECT_THROW(store.fetch(id), std::runtime_error);
}

TEST(RRefTest, RRefStoreMultipleTensors) {
    auto& store = RRefStore::instance();

    auto t1 = tenzor::ones({2}, DType::Float32, Device::cpu());
    auto t2 = tenzor::zeros({3}, DType::Float32, Device::cpu());

    auto id1 = store.store(t1);
    auto id2 = store.store(t2);

    EXPECT_NE(id1, id2);  // Unique IDs

    auto f1 = store.fetch(id1);
    auto f2 = store.fetch(id2);

    EXPECT_EQ(f1.numel(), 2);
    EXPECT_EQ(f2.numel(), 3);

    store.remove(id1);
    store.remove(id2);
}

TEST(RRefTest, RRefMoveSemantics) {
    WorkerInfo self;
    self.id = 0;
    auto agent = std::make_shared<TcpRpcAgent>(self);

    auto& store = RRefStore::instance();
    auto tensor = tenzor::randn({5}, DType::Float32, Device::cpu());
    auto rref_id = store.store(tensor);

    RRef rref1(0, rref_id, agent);
    EXPECT_EQ(rref1.owner_id(), 0);
    EXPECT_EQ(rref1.rref_id(), rref_id);

    // Move
    RRef rref2(std::move(rref1));
    EXPECT_EQ(rref2.owner_id(), 0);
    EXPECT_EQ(rref2.rref_id(), rref_id);

    store.remove(rref_id);
}
