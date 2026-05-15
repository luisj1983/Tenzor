/**
 * @file test_fsdp2_collective_wiring.cpp
 * @brief B4 integration test: FSDP2 backward_hook drives reduce_scatter
 *        via the B2 DTensor::redistribute path.
 *
 * Single-process simulation of a 2-rank mesh with a fake PG. Verifies:
 *   (1) shard_parameters() shards each parameter along dim 0 (DTensor
 *       `from_global` path — local shape is `param.shape[0] / world_size`).
 *   (2) backward_hook() invokes the PG's reduce_scatter exactly once per
 *       trainable parameter that has a gradient.
 *   (3) the per-parameter gradient written back has the *sharded* shape.
 *
 * Real multi-rank reduction semantics are validated by the multi-process
 * distributed test job (plan K3); here we check the wiring contract.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/distributed/device_mesh.hpp>
#include <tenzor/distributed/process_group.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <tenzor/distributed/fsdp2.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/ops/creation.hpp>

using namespace tenzor;
using namespace tenzor::distributed;

namespace {

// Fake PG: 2-rank mesh, rank 0. Records reduce_scatter / all_gather calls.
class FsdpFakePG : public ProcessGroupBase {
public:
    int reduce_scatter_calls = 0;
    int all_gather_calls = 0;
    auto rank() const -> int override { return 0; }
    auto world_size() const -> int override { return 2; }
    auto all_reduce(Tensor&, ReduceOp) -> void override {}
    auto broadcast(Tensor&, int) -> void override {}
    auto all_gather(std::vector<Tensor>& output, const Tensor& input) -> void override {
        ++all_gather_calls;
        output.assign(2, input);  // Self-as-everyone fake.
    }
    auto reduce_scatter(Tensor& output, std::span<const Tensor> input) -> void override {
        ++reduce_scatter_calls;
        // Self-as-everyone: SUM of W identical inputs == W * input[my_rank].
        if (input.empty()) return;
        const Tensor& my_chunk = input[0];  // rank 0's chunk
        auto shape_v = std::vector<int64_t>(
            my_chunk.shape().begin(), my_chunk.shape().end());
        output = empty(shape_v, my_chunk.dtype(), my_chunk.device());
        auto* dst = output.data<float>();
        const auto* src = my_chunk.data<float>();
        for (int64_t i = 0; i < my_chunk.numel(); ++i) {
            dst[i] = src[i] * 2.0f;  // world_size = 2
        }
    }
    auto barrier() -> void override {}
};

// Trivial MLP: two Linear layers. shard_parameters() will shard each
// weight/bias along dim 0.
class TwoLinearMLP : public nn::Module {
public:
    TwoLinearMLP() {
        fc1_ = std::make_shared<nn::Linear>(/*in=*/8, /*out=*/4);
        fc2_ = std::make_shared<nn::Linear>(/*in=*/4, /*out=*/2);
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }
    auto forward_impl(const Variable& x) -> Variable override {
        auto h = fc1_->forward(x);
        return fc2_->forward(h);
    }
private:
    std::shared_ptr<nn::Linear> fc1_;
    std::shared_ptr<nn::Linear> fc2_;
};

class B4Test : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }

    static auto make_mesh_and_pg(int world_size) {
        auto mesh = std::make_shared<DeviceMesh>(
            Device::Type::CPU, std::vector<int64_t>{world_size},
            std::vector<std::string>{"dp"}, /*mesh_rank=*/0);
        auto pg = std::make_shared<FsdpFakePG>();
        mesh->set_process_group(pg);
        return std::make_pair(mesh, pg);
    }
};

} // namespace

TEST_F(B4Test, ShardParameters_ShardsAlongDim0) {
    auto [mesh, pg] = make_mesh_and_pg(2);
    auto mlp = std::make_shared<TwoLinearMLP>();

    FSDP2Config cfg;
    cfg.mesh = mesh;
    cfg.shard_mesh_dim = "dp";
    FSDP2 fsdp(mlp, cfg);

    // Capture pre-shard parameter shapes.
    auto named = mlp->named_parameters();
    std::vector<std::vector<int64_t>> pre_shapes;
    for (auto& [name, p] : named) {
        if (p) pre_shapes.push_back(std::vector<int64_t>(
            p->tensor().shape().begin(), p->tensor().shape().end()));
    }

    fsdp.shard_parameters();
    auto sharded = fsdp.sharded_parameters();
    ASSERT_FALSE(sharded.empty());

    // Each sharded DTensor should have local shape with dim 0 = pre[0] / 2.
    for (size_t i = 0; i < sharded.size() && i < pre_shapes.size(); ++i) {
        const auto& local = sharded[i].local_tensor();
        ASSERT_FALSE(local.shape().empty());
        EXPECT_EQ(local.shape()[0], pre_shapes[i][0] / 2)
            << "param " << i << " not sharded on dim 0";
    }
}

TEST_F(B4Test, BackwardHook_InvokesReduceScatter_PerParameter) {
    auto [mesh, pg] = make_mesh_and_pg(2);
    auto mlp = std::make_shared<TwoLinearMLP>();

    FSDP2Config cfg;
    cfg.mesh = mesh;
    cfg.shard_mesh_dim = "dp";
    cfg.reshard_after_forward = false;  // keep params unsharded for grad write
    FSDP2 fsdp(mlp, cfg);
    fsdp.shard_parameters();
    // Unshard so module->named_parameters() returns the full-size params
    // (the buffers backward_hook writes back into).
    fsdp.summon_full_params();

    // Plant a gradient on every trainable parameter, matching that
    // parameter's *current* (unsharded) tensor shape.
    int trainable_count = 0;
    for (auto& [name, p] : mlp->named_parameters()) {
        if (!p || !p->requires_grad()) continue;
        ++trainable_count;
        auto g = ones(std::vector<int64_t>(p->tensor().shape().begin(),
                                          p->tensor().shape().end()),
                      p->tensor().dtype(), p->tensor().device());
        p->set_grad(g);
    }
    ASSERT_GT(trainable_count, 0);

    fsdp.backward_hook();

    // Each trainable parameter triggers one reduce_scatter.
    EXPECT_EQ(pg->reduce_scatter_calls, trainable_count);
}

TEST_F(B4Test, BackwardHook_WritesShardedGradients) {
    auto [mesh, pg] = make_mesh_and_pg(2);
    auto mlp = std::make_shared<TwoLinearMLP>();

    FSDP2Config cfg;
    cfg.mesh = mesh;
    cfg.shard_mesh_dim = "dp";
    cfg.reshard_after_forward = false;
    FSDP2 fsdp(mlp, cfg);
    fsdp.shard_parameters();
    fsdp.summon_full_params();

    // Plant ones gradient on each trainable param.
    std::unordered_map<std::string, std::vector<int64_t>> pre_grad_shape;
    for (auto& [name, p] : mlp->named_parameters()) {
        if (!p || !p->requires_grad()) continue;
        std::vector<int64_t> shape(p->tensor().shape().begin(),
                                   p->tensor().shape().end());
        pre_grad_shape[name] = shape;
        auto g = ones(shape, p->tensor().dtype(), p->tensor().device());
        p->set_grad(g);
    }

    fsdp.backward_hook();

    // After backward_hook, each param's grad has been replaced by the
    // *sharded* chunk: dim 0 should be pre / world_size.
    for (auto& [name, p] : mlp->named_parameters()) {
        if (!p || !p->requires_grad()) continue;
        auto& g_opt = p->mutable_grad();
        ASSERT_TRUE(g_opt.has_value()) << "no grad for " << name;
        const auto& g_shape = g_opt->shape();
        ASSERT_FALSE(g_shape.empty());
        EXPECT_EQ(g_shape[0], pre_grad_shape[name][0] / 2)
            << "param '" << name << "' grad not sharded on dim 0";
    }
}

TEST_F(B4Test, SingleRankMesh_BackwardHook_IsNoOp) {
    auto [mesh, pg] = make_mesh_and_pg(1);
    auto mlp = std::make_shared<TwoLinearMLP>();
    FSDP2Config cfg;
    cfg.mesh = mesh;
    cfg.shard_mesh_dim = "dp";
    FSDP2 fsdp(mlp, cfg);
    fsdp.shard_parameters();
    EXPECT_NO_THROW(fsdp.backward_hook());
    // No collective calls on world_size == 1.
    EXPECT_EQ(pg->reduce_scatter_calls, 0);
    EXPECT_EQ(pg->all_gather_calls, 0);
}

TEST_F(B4Test, MissingPG_AnyCollective_Throws) {
    // Mesh with world_size=2 but no PG attached. ANY FSDP2 operation that
    // requires a collective (summon_full_params, backward_hook) should fail
    // loudly with a clear error.
    auto mesh = std::make_shared<DeviceMesh>(
        Device::Type::CPU, std::vector<int64_t>{2},
        std::vector<std::string>{"dp"}, /*mesh_rank=*/0);
    // Intentionally do NOT attach a PG.

    auto mlp = std::make_shared<TwoLinearMLP>();
    FSDP2Config cfg;
    cfg.mesh = mesh;
    cfg.shard_mesh_dim = "dp";
    FSDP2 fsdp(mlp, cfg);
    fsdp.shard_parameters();

    // The first collective FSDP2 needs (all-gather to unshard) must throw
    // a clear error because no PG is attached.
    EXPECT_THROW(fsdp.summon_full_params(), std::runtime_error);
}
