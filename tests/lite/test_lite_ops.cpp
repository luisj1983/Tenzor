/**
 * @file test_lite_ops.cpp
 * @brief Tests for lite runtime (LiteTensor, LiteAllocator, LiteRuntime)
 */

#include <gtest/gtest.h>
#include <tenzor/lite/lite_graph.hpp>
#include <tenzor/lite/runtime.hpp>
#include <cmath>
#include <cstring>

namespace tenzor { void initialize(); }

namespace {
class TenzorLiteOpsEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_lite_ops_env =
    ::testing::AddGlobalTestEnvironment(new TenzorLiteOpsEnv);
}

using namespace tenzor::lite;

TEST(LiteTensorTest, DefaultConstruction) {
    LiteTensor tensor;
    EXPECT_EQ(tensor.data, nullptr);
    EXPECT_EQ(tensor.ndim, 0);
    EXPECT_EQ(tensor.dtype, tenzor::DType::Float32);
    EXPECT_FALSE(tensor.owns_data);
}

TEST(LiteTensorTest, Numel) {
    LiteTensor tensor;
    tensor.ndim = 3;
    tensor.shape[0] = 2;
    tensor.shape[1] = 3;
    tensor.shape[2] = 4;
    EXPECT_EQ(tensor.numel(), 24);
}

TEST(LiteTensorTest, NumelEmpty) {
    // A genuinely empty tensor has a zero-sized dimension -> numel 0.
    LiteTensor tensor;
    tensor.ndim = 1;
    tensor.shape[0] = 0;
    EXPECT_EQ(tensor.numel(), 0);
}

TEST(LiteTensorTest, NumelScalar) {
    // A rank-0 scalar has exactly one element (empty product over zero dims),
    // matching to_lite_tensor() which allocates real data for a scalar.
    LiteTensor tensor;
    tensor.ndim = 0;
    EXPECT_EQ(tensor.numel(), 1);
}

TEST(LiteAllocatorTest, Construction) {
    std::vector<size_t> pools = {1024, 2048};
    LiteAllocator allocator(pools, 64);
    EXPECT_EQ(allocator.total_bytes(), 3072);
}

TEST(LiteAllocatorTest, GetBuffer) {
    std::vector<size_t> pools = {1024};
    LiteAllocator allocator(pools, 64);

    void* ptr = allocator.get_buffer(0, 0);
    EXPECT_NE(ptr, nullptr);

    void* ptr2 = allocator.get_buffer(0, 512);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr, ptr2);
}

TEST(LiteAllocatorTest, OutOfRange) {
    std::vector<size_t> pools = {1024};
    LiteAllocator allocator(pools, 64);

    EXPECT_THROW(allocator.get_buffer(5, 0), std::out_of_range);
}

TEST(LiteAllocatorTest, EmptyPools) {
    std::vector<size_t> pools = {};
    LiteAllocator allocator(pools, 64);
    EXPECT_EQ(allocator.total_bytes(), 0);
}

TEST(LiteRuntimeTest, LoadFromInvalidPath) {
    EXPECT_THROW(LiteRuntime::load("/nonexistent/path.tzlite"), std::runtime_error);
}

TEST(LiteRuntimeTest, LoadFromNullData) {
    EXPECT_THROW(LiteRuntime::load(nullptr, 0), std::runtime_error);
}

TEST(LiteRuntimeTest, CreateInput) {
    // create_input is a pure helper that allocates a LiteTensor of the given
    // shape/dtype and is independent of whether the runtime has a real graph.
    // Build the runtime via from_graph() to avoid needing a serialised file.
    auto runtime = LiteRuntime::from_graph(LiteGraph{});
    ASSERT_NE(runtime, nullptr);

    auto input = runtime->create_input({1, 3, 224, 224});
    EXPECT_NE(input.data, nullptr);
    EXPECT_EQ(input.ndim, 4);
    EXPECT_EQ(input.shape[0], 1);
    EXPECT_EQ(input.shape[1], 3);
    EXPECT_EQ(input.shape[2], 224);
    EXPECT_EQ(input.shape[3], 224);
    EXPECT_EQ(input.numel(), 1 * 3 * 224 * 224);
    EXPECT_TRUE(input.owns_data);
}

// Phase 1 end-to-end test: build a 3-node MatMul -> Add -> ReLU graph in C++
// and verify the numerical output matches a hand-computed reference. This
// exercises the bridge layers (LiteTensor <-> Tensor, LiteAttributes ->
// OpAttributes) and the dispatch path through the main kernel registry.
TEST(LiteRuntimeTest, ForwardMatMulAddReLU) {
    // Graph (tensor_ids):
    //   0,1 -> MatMul -> 3
    //   3,2 -> Add    -> 4
    //   4   -> ReLU   -> 5
    //   inputs:  [0, 1, 2]   ->  A, B, bias
    //   outputs: [5]         ->  relu(A @ B + bias)
    LiteGraph graph;
    {
        LiteNode mm;
        mm.op = LiteOpType::MatMul;
        mm.input_ids = {0, 1};
        mm.output_ids = {3};
        graph.add_node(std::move(mm));

        LiteNode ad;
        ad.op = LiteOpType::Add;
        ad.input_ids = {3, 2};
        ad.output_ids = {4};
        graph.add_node(std::move(ad));

        LiteNode rl;
        rl.op = LiteOpType::ReLU;
        rl.input_ids = {4};
        rl.output_ids = {5};
        graph.add_node(std::move(rl));
    }
    graph.set_input_ids({0, 1, 2});
    graph.set_output_ids({5});

    auto runtime = LiteRuntime::from_graph(std::move(graph));
    ASSERT_NE(runtime, nullptr);

    // A is 2x3, B is 3x2, bias is 2x2. (A @ B) has shape 2x2.
    auto A = runtime->create_input({2, 3}, tenzor::DType::Float32);
    auto B = runtime->create_input({3, 2}, tenzor::DType::Float32);
    auto bias = runtime->create_input({2, 2}, tenzor::DType::Float32);
    {
        // A = [[ 1, 2,  3],
        //      [-4, 5, -6]]
        float* a = A.data_as<float>();
        a[0]= 1; a[1]= 2; a[2]= 3;
        a[3]=-4; a[4]= 5; a[5]=-6;
        // B = [[1, 0],
        //      [0, 1],
        //      [1, 1]]
        float* b = B.data_as<float>();
        b[0]=1; b[1]=0;
        b[2]=0; b[3]=1;
        b[4]=1; b[5]=1;
        // bias = [[-5, -5],
        //         [-5, -5]]   (large negative -> some ReLU clipping)
        float* c = bias.data_as<float>();
        c[0]=-5; c[1]=-5; c[2]=-5; c[3]=-5;
    }

    std::vector<LiteTensor> ins;
    ins.push_back(std::move(A));
    ins.push_back(std::move(B));
    ins.push_back(std::move(bias));
    auto outs = runtime->forward(ins);
    ASSERT_EQ(outs.size(), 1u);
    auto& out = outs.front();
    EXPECT_EQ(out.ndim, 2);
    EXPECT_EQ(out.shape[0], 2);
    EXPECT_EQ(out.shape[1], 2);
    ASSERT_EQ(out.numel(), 4);

    // Reference: A @ B = [[ 1*1 + 2*0 + 3*1,  1*0 + 2*1 + 3*1],
    //                     [-4*1 + 5*0 -6*1, -4*0 + 5*1 -6*1]]
    //                   = [[ 4,  5],
    //                      [-10, -1]]
    // + bias                = [[-1,  0],
    //                          [-15, -6]]
    // ReLU                  = [[ 0,  0],
    //                          [ 0,  0]]
    const float* o = out.data_as<float>();
    EXPECT_FLOAT_EQ(o[0], 0.0f);
    EXPECT_FLOAT_EQ(o[1], 0.0f);
    EXPECT_FLOAT_EQ(o[2], 0.0f);
    EXPECT_FLOAT_EQ(o[3], 0.0f);
}

// Companion test: same graph topology, different bias so the ReLU has a
// mix of clipped and pass-through values (stronger regression coverage).
TEST(LiteRuntimeTest, ForwardMatMulAddReLU_PartialClip) {
    LiteGraph graph;
    {
        LiteNode mm; mm.op = LiteOpType::MatMul; mm.input_ids = {0, 1}; mm.output_ids = {3};
        graph.add_node(std::move(mm));
        LiteNode ad; ad.op = LiteOpType::Add;    ad.input_ids = {3, 2}; ad.output_ids = {4};
        graph.add_node(std::move(ad));
        LiteNode rl; rl.op = LiteOpType::ReLU;   rl.input_ids = {4};    rl.output_ids = {5};
        graph.add_node(std::move(rl));
    }
    graph.set_input_ids({0, 1, 2});
    graph.set_output_ids({5});
    auto runtime = LiteRuntime::from_graph(std::move(graph));

    auto A = runtime->create_input({2, 3}, tenzor::DType::Float32);
    auto B = runtime->create_input({3, 2}, tenzor::DType::Float32);
    auto bias = runtime->create_input({2, 2}, tenzor::DType::Float32);
    {
        float* a = A.data_as<float>();
        a[0]=1; a[1]=2; a[2]=3;  a[3]=-4; a[4]=5; a[5]=-6;
        float* b = B.data_as<float>();
        b[0]=1; b[1]=0;  b[2]=0; b[3]=1;  b[4]=1; b[5]=1;
        float* c = bias.data_as<float>();
        c[0]=0; c[1]=0; c[2]=11; c[3]=2;  // shifts: row 1 goes positive
    }

    std::vector<LiteTensor> ins;
    ins.push_back(std::move(A));
    ins.push_back(std::move(B));
    ins.push_back(std::move(bias));
    auto outs = runtime->forward(ins);
    const float* o = outs.front().data_as<float>();
    // A @ B = [[4, 5], [-10, -1]]
    // + bias  = [[4, 5], [1, 1]]
    // ReLU    = [[4, 5], [1, 1]]
    EXPECT_FLOAT_EQ(o[0], 4.0f);
    EXPECT_FLOAT_EQ(o[1], 5.0f);
    EXPECT_FLOAT_EQ(o[2], 1.0f);
    EXPECT_FLOAT_EQ(o[3], 1.0f);
}

// Empty-graph runtimes should still allow create_input (it's a pure helper),
// but forward() must throw a clean error rather than segfaulting.
TEST(LiteRuntimeTest, ForwardOnEmptyGraphThrows) {
    auto runtime = LiteRuntime::from_graph(LiteGraph{});
    ASSERT_NE(runtime, nullptr);
    auto input = runtime->create_input({4});
    EXPECT_THROW(runtime->forward(input), std::runtime_error);
}

TEST(LiteRuntimeTest, MaxDims) {
    EXPECT_EQ(kMaxDims, 8);
}
