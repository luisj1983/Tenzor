/**
 * @file test_jit_serialization_control_flow.cpp
 * @brief Round-trip serialization tests for JIT control-flow subgraphs (C4)
 *        and device-portable constants (M9).
 *
 * C4: write_nodes/read_nodes previously never serialized the shared_ptr<Graph>
 *     subgraph members (then_branch_/else_branch_/body_) on If/Loop nodes, so
 *     after save()->load() every control-flow node had null branches and the
 *     executor produced no outputs. These tests build If/Loop graphs directly
 *     via the Graph API (the tracer's If conditions graph-break, so a directly
 *     constructed graph is the reliable round-trip vehicle), serialize, reload,
 *     and assert the reloaded graph (a) has non-null subgraphs and (b) replays
 *     bit-for-bit identically to the pre-serialization graph AND to eager, on
 *     every available backend.
 *
 * M9: serialized constants used to be pinned to the trace-time device, so a
 *     graph traced on CUDA could not run on CPU/ROCm. These tests serialize a
 *     graph with a captured constant on one backend, load it, and run it with
 *     inputs on a DIFFERENT backend, asserting the numeric result is correct
 *     on that second backend (the constant follows the inputs onto whatever
 *     device they use — no CPU fallback).
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <string>
#include <vector>

#include <tenzor/tenzor.hpp>
#include <tenzor/jit/graph.hpp>
#include <tenzor/jit/tracer.hpp>   // OpType
#include <tenzor/jit/serialization.hpp>
#include "../backend_parity/parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::jit;
using namespace tenzor::testing;

class JITSerCFEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static ::testing::Environment* const g_ser_cf_env =
    ::testing::AddGlobalTestEnvironment(new JITSerCFEnv);

namespace {

std::string temp_path(const std::string& name) {
    auto dir = std::filesystem::temp_directory_path();
    return (dir / name).string();
}

// Build an If graph on CPU:
//   inputs : [cond {1}, x {2,3}]
//   If node: cond ? then(x) : else(x)
//     then branch: out = x + x   (== 2x)
//     else branch: out = -x
//   output : [if_out {2,3}]
std::shared_ptr<Graph> build_if_graph() {
    auto g = std::make_shared<Graph>();
    const std::vector<int64_t> xs = {2, 3};

    auto cond = g->create_value("cond", {1}, DType::Float32, Device::cpu());
    auto x    = g->create_value("x", xs, DType::Float32, Device::cpu());
    auto out  = g->create_value("if_out", xs, DType::Float32, Device::cpu());

    auto if_node = g->create_node(OpType::If, "if");
    if_node->add_input(cond);
    if_node->add_input(x);
    out->set_node(if_node);
    if_node->add_output(out);

    // then branch: input tx -> Add(tx, tx) -> t_out
    {
        auto sub = std::make_shared<Graph>();
        auto tx = sub->create_value("tx", xs, DType::Float32, Device::cpu());
        auto t_out = sub->create_value("t_out", xs, DType::Float32, Device::cpu());
        auto add = sub->create_node(OpType::Add, "then_add");
        add->add_input(tx);
        add->add_input(tx);
        t_out->set_node(add);
        add->add_output(t_out);
        sub->add_node(add);
        sub->set_inputs({tx});
        sub->set_outputs({t_out});
        if_node->set_then_branch(sub);
    }
    // else branch: input ex -> Neg(ex) -> e_out
    {
        auto sub = std::make_shared<Graph>();
        auto ex = sub->create_value("ex", xs, DType::Float32, Device::cpu());
        auto e_out = sub->create_value("e_out", xs, DType::Float32, Device::cpu());
        auto neg = sub->create_node(OpType::Neg, "else_neg");
        neg->add_input(ex);
        e_out->set_node(neg);
        neg->add_output(e_out);
        sub->add_node(neg);
        sub->set_inputs({ex});
        sub->set_outputs({e_out});
        if_node->set_else_branch(sub);
    }

    g->add_node(if_node);
    g->set_inputs({cond, x});
    g->set_outputs({out});
    return g;
}

// Build a Loop graph on CPU:
//   inputs : [max_iter {1}, cond {1}, carried {2}]
//   body   : outputs [cond (pass-through), carried + 1]
//   output : [loop_out {2}]  == carried + max_iter (while cond stays true)
std::shared_ptr<Graph> build_loop_graph() {
    auto g = std::make_shared<Graph>();
    const std::vector<int64_t> cs = {2};

    auto max_iter = g->create_value("max_iter", {1}, DType::Float32, Device::cpu());
    auto cond     = g->create_value("cond", {1}, DType::Float32, Device::cpu());
    auto carried  = g->create_value("carried", cs, DType::Float32, Device::cpu());
    auto out      = g->create_value("loop_out", cs, DType::Float32, Device::cpu());

    auto loop = g->create_node(OpType::Loop, "loop");
    loop->add_input(max_iter);
    loop->add_input(cond);
    loop->add_input(carried);
    out->set_node(loop);
    loop->add_output(out);

    // body: inputs [iter {1}, bcond {1}, bx {2}]
    //       one = const {2} of ones; bx1 = Add(bx, one)
    //       outputs [bcond, bx1]
    {
        auto sub = std::make_shared<Graph>();
        auto iter  = sub->create_value("iter", {1}, DType::Float32, Device::cpu());
        auto bcond = sub->create_value("bcond", {1}, DType::Float32, Device::cpu());
        auto bx    = sub->create_value("bx", cs, DType::Float32, Device::cpu());
        auto one   = sub->create_value("one", cs, DType::Float32, Device::cpu());
        auto bx1   = sub->create_value("bx1", cs, DType::Float32, Device::cpu());

        auto add = sub->create_node(OpType::Add, "body_add");
        add->add_input(bx);
        add->add_input(one);
        bx1->set_node(add);
        add->add_output(bx1);
        sub->add_node(add);

        sub->set_constant("one", ones(cs, DType::Float32, Device::cpu()));
        sub->set_inputs({iter, bcond, bx});
        sub->set_outputs({bcond, bx1});
        loop->set_body(sub);
    }

    g->add_node(loop);
    g->set_inputs({max_iter, cond, carried});
    g->set_outputs({out});
    return g;
}

// Build a graph with a captured constant on `const_dev`:
//   inputs : [x {2,3}]
//   const  : w {2,3}
//   output : [ x + w ]
std::shared_ptr<Graph> build_const_graph(const Tensor& w) {
    auto g = std::make_shared<Graph>();
    const std::vector<int64_t> xs = {2, 3};
    auto x   = g->create_value("x", xs, DType::Float32, Device::cpu());
    auto w_v = g->create_value("w", xs, DType::Float32, Device::cpu());
    auto sum = g->create_value("sum", xs, DType::Float32, Device::cpu());

    auto add = g->create_node(OpType::Add, "add");
    add->add_input(x);
    add->add_input(w_v);
    sum->set_node(add);
    add->add_output(sum);
    g->add_node(add);

    g->set_constant("w", w);
    g->set_inputs({x});
    g->set_outputs({sum});
    return g;
}

}  // namespace

// ============================================================================
// C4: If subgraph survives save/load and replays identically on every backend
// ============================================================================
TEST(JITSerializationControlFlow, IfRoundTripAllBackends) {
    auto backends = get_available_backends();

    auto g = build_if_graph();
    const std::string path = temp_path("tz_if_roundtrip.graph");
    g->save(path);
    auto loaded = Graph::load(path);
    ASSERT_NE(loaded, nullptr);

    // The reloaded If node must have non-null branch subgraphs (the C4 fix).
    ASSERT_EQ(loaded->nodes().size(), 1u);
    auto if_node = loaded->nodes()[0];
    ASSERT_EQ(if_node->op_type(), OpType::If);
    ASSERT_NE(if_node->then_branch(), nullptr);
    ASSERT_NE(if_node->else_branch(), nullptr);
    EXPECT_EQ(if_node->then_branch()->num_nodes(), 1u);
    EXPECT_EQ(if_node->else_branch()->num_nodes(), 1u);

    for (const auto& dev : backends) {
        for (float cflag : {1.0f, 0.0f}) {
            auto x_cpu = randn({2, 3}, DType::Float32, Device::cpu());
            auto cond  = full({1}, cflag, DType::Float32, dev);
            auto x     = x_cpu.to(dev);

            // Eager reference.
            Tensor eager = (cflag != 0.0f) ? (x_cpu + x_cpu) : neg(x_cpu);

            // Pre-serialization graph (inputs are [cond, x]) for equality baseline.
            auto orig_out = g->forward(
                {Variable(full({1}, cflag, DType::Float32, Device::cpu()), false),
                 Variable(x_cpu, false)});
            ASSERT_EQ(orig_out.size(), 1u);

            auto loaded_out = loaded->forward({Variable(cond, false), Variable(x, false)});
            ASSERT_EQ(loaded_out.size(), 1u);
            dev.synchronize();

            Tensor got = loaded_out[0].tensor().to(Device::cpu());
            EXPECT_TENSORS_CLOSE(eager, got, 1e-5f, 1e-5f);
            EXPECT_TENSORS_CLOSE(orig_out[0].tensor(), got, 1e-5f, 1e-5f);
        }
    }
    std::filesystem::remove(path);
}

// ============================================================================
// C4: Loop body survives save/load and replays identically on every backend
// ============================================================================
TEST(JITSerializationControlFlow, LoopRoundTripAllBackends) {
    auto backends = get_available_backends();

    auto g = build_loop_graph();
    const std::string path = temp_path("tz_loop_roundtrip.graph");
    g->save(path);
    auto loaded = Graph::load(path);
    ASSERT_NE(loaded, nullptr);

    ASSERT_EQ(loaded->nodes().size(), 1u);
    auto loop_node = loaded->nodes()[0];
    ASSERT_EQ(loop_node->op_type(), OpType::Loop);
    ASSERT_NE(loop_node->body(), nullptr);
    EXPECT_EQ(loop_node->body()->num_nodes(), 1u);
    // Body must carry its own constant ("one") across the round trip.
    ASSERT_EQ(loop_node->body()->constants().count("one"), 1u);

    const int iters = 3;
    for (const auto& dev : backends) {
        auto start_cpu = full({2}, 10.0f, DType::Float32, Device::cpu());
        Tensor eager = start_cpu + full({2}, static_cast<float>(iters),
                                         DType::Float32, Device::cpu());  // 10 + 3

        auto orig = g->forward({
            Variable(full({1}, static_cast<float>(iters), DType::Float32, Device::cpu()), false),
            Variable(full({1}, 1.0f, DType::Float32, Device::cpu()), false),
            Variable(start_cpu, false)});
        ASSERT_EQ(orig.size(), 1u);

        auto loaded_out = loaded->forward({
            Variable(full({1}, static_cast<float>(iters), DType::Float32, dev), false),
            Variable(full({1}, 1.0f, DType::Float32, dev), false),
            Variable(start_cpu.to(dev), false)});
        ASSERT_EQ(loaded_out.size(), 1u);
        dev.synchronize();

        Tensor got = loaded_out[0].tensor().to(Device::cpu());
        EXPECT_TENSORS_CLOSE(eager, got, 1e-5f, 1e-5f);
        EXPECT_TENSORS_CLOSE(orig[0].tensor(), got, 1e-5f, 1e-5f);
    }
    std::filesystem::remove(path);
}

// ============================================================================
// M9: a constant serialized on CPU is placed on WHATEVER device the runtime
//     inputs use — the graph replays correctly on every available backend.
// ============================================================================
TEST(JITSerializationControlFlow, ConstantIsDevicePortable) {
    auto backends = get_available_backends();

    auto w_cpu = randn({2, 3}, DType::Float32, Device::cpu());
    auto g = build_const_graph(w_cpu);
    const std::string path = temp_path("tz_const_portable.graph");
    g->save(path);
    auto loaded = Graph::load(path);
    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->constants().count("w"), 1u);
    // Loaded constant must be device-neutral (materialized on CPU).
    EXPECT_EQ(loaded->constants().at("w").device().type, Device::Type::CPU);

    for (const auto& dev : backends) {
        auto x_cpu = randn({2, 3}, DType::Float32, Device::cpu());
        Tensor eager = x_cpu + w_cpu;

        auto out = loaded->forward({Variable(x_cpu.to(dev), false)});
        ASSERT_EQ(out.size(), 1u);
        dev.synchronize();

        Tensor got = out[0].tensor().to(Device::cpu());
        EXPECT_TENSORS_CLOSE(eager, got, 1e-5f, 1e-5f);
    }
    std::filesystem::remove(path);
}

// ============================================================================
// M9 (reverse direction): a constant serialized ON A GPU backend replays on
//     CPU. Only runs when a non-CPU backend is available.
// ============================================================================
TEST(JITSerializationControlFlow, GpuSerializedConstantRunsOnCpu) {
    auto backends = get_available_backends();
    bool ran = false;

    for (const auto& dev : backends) {
        if (dev.type == Device::Type::CPU) continue;

        auto w_cpu = randn({2, 3}, DType::Float32, Device::cpu());
        auto w_dev = w_cpu.to(dev);  // constant lives on the GPU at trace time
        auto g = build_const_graph(w_dev);
        const std::string path = temp_path("tz_const_gpu2cpu.graph");
        g->save(path);  // write_tensor migrates bytes to CPU, records CPU device
        auto loaded = Graph::load(path);
        ASSERT_NE(loaded, nullptr);
        EXPECT_EQ(loaded->constants().at("w").device().type, Device::Type::CPU);

        auto x_cpu = randn({2, 3}, DType::Float32, Device::cpu());
        Tensor eager = x_cpu + w_cpu;
        auto out = loaded->forward({Variable(x_cpu, false)});  // run on CPU
        ASSERT_EQ(out.size(), 1u);
        Tensor got = out[0].tensor().to(Device::cpu());
        EXPECT_TENSORS_CLOSE(eager, got, 1e-5f, 1e-5f);
        std::filesystem::remove(path);
        ran = true;
    }
    if (!ran) {
        // JIT-R129: honor the same TENZOR_REQUIRE_MULTI_BACKEND hard-fail
        // escalation the rest of the suite uses -- a bare GTEST_SKIP() here
        // would silently hide a CI box that's supposed to have a non-CPU
        // backend but where the driver failed to init.
        if (::tenzor::testing::golden::require_multi_backend()) {
            FAIL() << "Multi-backend required (TENZOR_REQUIRE_MULTI_BACKEND=1) "
                      "but no non-CPU backend is available for the GPU->CPU "
                      "direction";
        }
        GTEST_SKIP() << "no non-CPU backend available for GPU->CPU direction";
    }
}

// ============================================================================
// JIT-R151b: a module saved with its constant living on ONE GPU backend
//     (e.g. CUDA) must load and run correctly with inputs placed directly on
//     a DIFFERENT GPU backend (e.g. ROCm) -- not routed through CPU first.
//     ConstantIsDevicePortable/GpuSerializedConstantRunsOnCpu only ever cover
//     CPU->all-backends and GPU->CPU; this is the missing GPU-saved-on-A,
//     loaded-and-run-on-B direct path. write_tensor always normalizes to CPU
//     bytes at save time regardless of source device (no arch/gfx/SPIR-V
//     metadata is ever baked into the serialized format), so this is expected
//     to already work -- this test exists to actually prove it on real
//     hardware rather than leaving it as an untested assumption.
// ============================================================================
TEST(JITSerializationControlFlow, GpuToGpuSerializationRoundTrip) {
    auto backends = get_available_backends();
    std::vector<Device> non_cpu;
    for (const auto& dev : backends) {
        if (dev.type != Device::Type::CPU) non_cpu.push_back(dev);
    }

    if (non_cpu.size() < 2) {
        // JIT-R129: same hard-fail escalation as GpuSerializedConstantRunsOnCpu
        // above -- a silent skip would hide a CI box that's supposed to have
        // 2+ GPU backends but where a driver failed to init.
        if (::tenzor::testing::golden::require_multi_backend()) {
            FAIL() << "Multi-backend required (TENZOR_REQUIRE_MULTI_BACKEND=1) "
                      "but fewer than 2 non-CPU backends are available for "
                      "the GPU-to-GPU direction";
        }
        GTEST_SKIP() << "fewer than 2 non-CPU backends available for the "
                        "GPU-to-GPU direction";
    }

    for (size_t i = 0; i < non_cpu.size(); ++i) {
        for (size_t j = 0; j < non_cpu.size(); ++j) {
            if (i == j) continue;
            const Device& save_dev = non_cpu[i];
            const Device& load_dev = non_cpu[j];

            auto w_cpu = randn({2, 3}, DType::Float32, Device::cpu());
            auto w_save_dev = w_cpu.to(save_dev);  // constant lives on save_dev at trace time
            auto g = build_const_graph(w_save_dev);
            const std::string path = temp_path("tz_const_gpu2gpu.graph");
            g->save(path);  // write_tensor migrates bytes to CPU, records CPU device
            auto loaded = Graph::load(path);
            ASSERT_NE(loaded, nullptr);
            EXPECT_EQ(loaded->constants().at("w").device().type, Device::Type::CPU);

            SCOPED_TRACE("save_dev=" + save_dev.to_string() +
                        " load_dev=" + load_dev.to_string());

            auto x_cpu = randn({2, 3}, DType::Float32, Device::cpu());
            Tensor eager = x_cpu + w_cpu;
            // Run directly on load_dev -- no CPU hop in between.
            auto out = loaded->forward({Variable(x_cpu.to(load_dev), false)});
            ASSERT_EQ(out.size(), 1u);
            load_dev.synchronize();

            Tensor got = out[0].tensor().to(Device::cpu());
            EXPECT_TENSORS_CLOSE(eager, got, 1e-5f, 1e-5f);
            std::filesystem::remove(path);
        }
    }
}
