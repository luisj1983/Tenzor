/**
 * @file test_lite_runtime.cpp
 * @brief Tests for LiteRuntime and LiteAllocator
 */

#include <gtest/gtest.h>
#include <tenzor/lite/lite_graph.hpp>
#include <tenzor/lite/model_format.hpp>
#include <tenzor/lite/runtime.hpp>
#include <cstdint>

namespace tenzor { void initialize(); }

namespace {
class TenzorLiteEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_lite_env =
    ::testing::AddGlobalTestEnvironment(new TenzorLiteEnv);
}

using namespace tenzor::lite;

namespace {

// Helper: build a single-op graph that adds two scalar/vector inputs.
// inputs[0] = tensor_id 0, inputs[1] = tensor_id 1, output = tensor_id 2.
auto build_add_graph() -> LiteGraph {
    LiteGraph g;
    LiteNode n;
    n.op = LiteOpType::Add;
    n.input_ids = {0, 1};
    n.output_ids = {2};
    g.add_node(std::move(n));
    g.set_input_ids({0, 1});
    g.set_output_ids({2});
    return g;
}

}  // namespace

TEST(LiteRuntimeExpandedTest, AllocatorAlignment64) {
    std::vector<size_t> pools = {256};
    LiteAllocator allocator(pools, 64);

    void* ptr = allocator.get_buffer(0, 0);
    EXPECT_NE(ptr, nullptr);
    // Pointer should be 64-byte aligned
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0u);
}

TEST(LiteRuntimeExpandedTest, AllocatorAlignment16) {
    std::vector<size_t> pools = {128};
    LiteAllocator allocator(pools, 16);

    void* ptr = allocator.get_buffer(0, 0);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 16, 0u);
}

TEST(LiteRuntimeExpandedTest, AllocatorMultiplePools) {
    std::vector<size_t> pools = {512, 1024, 2048};
    LiteAllocator allocator(pools, 64);
    EXPECT_EQ(allocator.total_bytes(), 512 + 1024 + 2048);

    // Each pool should be independently addressable
    void* p0 = allocator.get_buffer(0, 0);
    void* p1 = allocator.get_buffer(1, 0);
    void* p2 = allocator.get_buffer(2, 0);
    EXPECT_NE(p0, p1);
    EXPECT_NE(p1, p2);
    EXPECT_NE(p0, p2);
}

TEST(LiteRuntimeExpandedTest, AllocatorOffsetWithinPool) {
    std::vector<size_t> pools = {1024};
    LiteAllocator allocator(pools, 64);

    void* base = allocator.get_buffer(0, 0);
    void* offset = allocator.get_buffer(0, 128);
    auto diff = static_cast<uint8_t*>(offset) - static_cast<uint8_t*>(base);
    EXPECT_EQ(diff, 128);
}

TEST(LiteRuntimeExpandedTest, RuntimeLoadFromMagic) {
    // Build a minimal valid TZLite header (24 bytes, zero nodes, no TLV).
    // Phase 2 strict-parses the header — the previous 4-byte-magic shortcut
    // is no longer accepted.
    tenzor::lite::TZLiteHeader header{};
    header.magic = tenzor::lite::TZLITE_MAGIC;
    header.version = tenzor::lite::TZLITE_VERSION;
    header.num_nodes = 0;
    header.num_weights = 0;
    header.weight_data_offset = sizeof(header);
    auto runtime = LiteRuntime::load(&header, sizeof(header));
    ASSERT_NE(runtime, nullptr);
}

TEST(LiteRuntimeExpandedTest, RuntimeCreateInputShape) {
    auto runtime = LiteRuntime::from_graph(LiteGraph{});
    ASSERT_NE(runtime, nullptr);

    auto input = runtime->create_input({1, 16});
    EXPECT_EQ(input.ndim, 2);
    EXPECT_EQ(input.shape[0], 1);
    EXPECT_EQ(input.shape[1], 16);
    EXPECT_EQ(input.numel(), 16);
    EXPECT_TRUE(input.owns_data);
    EXPECT_NE(input.data, nullptr);
}

TEST(LiteRuntimeExpandedTest, RuntimeCreateInputDtype) {
    auto runtime = LiteRuntime::from_graph(LiteGraph{});
    ASSERT_NE(runtime, nullptr);

    auto input_f32 = runtime->create_input({2, 2}, tenzor::DType::Float32);
    EXPECT_EQ(input_f32.dtype, tenzor::DType::Float32);
}

TEST(LiteRuntimeExpandedTest, RuntimeLoadInvalidPath) {
    EXPECT_THROW(LiteRuntime::load("/nonexistent/model.tzlite"), std::runtime_error);
}

TEST(LiteRuntimeExpandedTest, RuntimeLoadNullData) {
    EXPECT_THROW(LiteRuntime::load(nullptr, 0), std::runtime_error);
}

TEST(LiteRuntimeExpandedTest, MaxDimsConstant) {
    EXPECT_EQ(kMaxDims, 8);
}

// ============================================================================
// Forward-pass numerics — Phase 1 wires LiteGraph::execute() through the main
// OpId dispatch table, so a hand-built graph should produce bit-identical
// output to the same op called eagerly. These tests replace the original
// "identity passthrough" assertions that pinned the pre-Phase-1 stub.
// ============================================================================

TEST(LiteRuntimeExpandedTest, ForwardReturnsCopyOfInput) {
    // Build a single-node Add graph and verify forward() produces the
    // element-wise sum of its two inputs. Shape preserved across the call.
    auto runtime = LiteRuntime::from_graph(build_add_graph());
    ASSERT_NE(runtime, nullptr);

    auto a = runtime->create_input({2, 3}, tenzor::DType::Float32);
    auto b = runtime->create_input({2, 3}, tenzor::DType::Float32);
    ASSERT_EQ(a.numel(), 6);
    auto* a_data = static_cast<float*>(a.data);
    auto* b_data = static_cast<float*>(b.data);
    for (int i = 0; i < 6; ++i) {
        a_data[i] = static_cast<float>(i + 1);   // 1..6
        b_data[i] = static_cast<float>(10 * (i + 1)); // 10..60
    }

    std::vector<LiteTensor> ins;
    ins.push_back(std::move(a));
    ins.push_back(std::move(b));
    auto outs = runtime->forward(ins);
    ASSERT_EQ(outs.size(), 1u);
    auto& output = outs.front();

    EXPECT_EQ(output.ndim, 2);
    EXPECT_EQ(output.shape[0], 2);
    EXPECT_EQ(output.shape[1], 3);
    EXPECT_EQ(output.dtype, tenzor::DType::Float32);

    ASSERT_NE(output.data, nullptr);
    const auto* out_data = static_cast<const float*>(output.data);
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(out_data[i], static_cast<float>(11 * (i + 1)));
    }
}

TEST(LiteRuntimeExpandedTest, ForwardAllocatesNewOutputBuffer) {
    // The output of forward() must own its data — it is allocated by
    // to_lite_tensor and survives after the inputs fall out of scope.
    auto runtime = LiteRuntime::from_graph(build_add_graph());
    ASSERT_NE(runtime, nullptr);

    auto a = runtime->create_input({4}, tenzor::DType::Float32);
    auto b = runtime->create_input({4}, tenzor::DType::Float32);
    void* a_data_addr = a.data;

    std::vector<LiteTensor> ins;
    ins.push_back(std::move(a));
    ins.push_back(std::move(b));
    auto outs = runtime->forward(ins);
    ASSERT_EQ(outs.size(), 1u);

    EXPECT_TRUE(outs.front().owns_data);
    EXPECT_NE(outs.front().data, a_data_addr)
        << "LiteRuntime::forward must allocate a fresh buffer, not alias "
           "the caller's input";
}
