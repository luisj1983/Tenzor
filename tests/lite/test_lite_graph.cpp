/**
 * @file test_lite_graph.cpp
 * @brief Tests for LiteGraph, LiteNode, and LiteOpType
 */

#include <gtest/gtest.h>
#include <tenzor/lite/lite_graph.hpp>

using namespace tenzor::lite;

TEST(LiteGraphTest, EmptyGraph) {
    LiteGraph graph;
    EXPECT_EQ(graph.num_nodes(), 0u);
}

TEST(LiteGraphTest, AddSingleNode) {
    LiteGraph graph;
    LiteNode node;
    node.op = LiteOpType::Add;
    node.input_ids = {0, 1};
    node.output_ids = {2};
    graph.add_node(node);

    EXPECT_EQ(graph.num_nodes(), 1u);
}

TEST(LiteGraphTest, AddMultipleNodes) {
    LiteGraph graph;

    LiteNode n1;
    n1.op = LiteOpType::MatMul;
    n1.input_ids = {0, 1};
    n1.output_ids = {2};
    graph.add_node(n1);

    LiteNode n2;
    n2.op = LiteOpType::ReLU;
    n2.input_ids = {2};
    n2.output_ids = {3};
    graph.add_node(n2);

    LiteNode n3;
    n3.op = LiteOpType::Softmax;
    n3.input_ids = {3};
    n3.output_ids = {4};
    graph.add_node(n3);

    EXPECT_EQ(graph.num_nodes(), 3u);
}

TEST(LiteGraphTest, OpTypeEnumValues) {
    // Verify a selection of enum values are distinct
    EXPECT_NE(static_cast<uint16_t>(LiteOpType::Add),
              static_cast<uint16_t>(LiteOpType::Sub));
    EXPECT_NE(static_cast<uint16_t>(LiteOpType::MatMul),
              static_cast<uint16_t>(LiteOpType::Gemm));
    EXPECT_NE(static_cast<uint16_t>(LiteOpType::ReLU),
              static_cast<uint16_t>(LiteOpType::Sigmoid));
    EXPECT_NE(static_cast<uint16_t>(LiteOpType::Conv2d),
              static_cast<uint16_t>(LiteOpType::QuantizedConv2d));
}

TEST(LiteGraphTest, OpTypeCoversQuantized) {
    // Verify quantized ops exist and are distinct
    auto qmm = static_cast<uint16_t>(LiteOpType::QuantizedMatMul);
    auto qconv = static_cast<uint16_t>(LiteOpType::QuantizedConv2d);
    EXPECT_NE(qmm, qconv);
}

TEST(LiteGraphTest, NodeAttributes) {
    LiteAttributes attrs;
    // Default-initialized to zero
    EXPECT_FLOAT_EQ(attrs.f[0], 0.0f);
    EXPECT_FLOAT_EQ(attrs.f[3], 0.0f);
    EXPECT_EQ(attrs.i[0], 0);
    EXPECT_EQ(attrs.i[3], 0);

    // Set some values
    attrs.f[0] = 0.1f;  // e.g., epsilon for BatchNorm
    attrs.i[0] = 3;     // e.g., kernel size
    EXPECT_FLOAT_EQ(attrs.f[0], 0.1f);
    EXPECT_EQ(attrs.i[0], 3);
}

TEST(LiteGraphTest, NodeInputOutputIds) {
    LiteNode node;
    node.op = LiteOpType::Conv2d;
    node.input_ids = {0, 1, 2};  // input, weight, bias
    node.output_ids = {3};
    node.attrs.i[0] = 3;  // kernel_size
    node.attrs.i[1] = 1;  // stride

    EXPECT_EQ(node.input_ids.size(), 3u);
    EXPECT_EQ(node.output_ids.size(), 1u);
    EXPECT_EQ(node.input_ids[0], 0);
    EXPECT_EQ(node.attrs.i[0], 3);
}
