/**
 * @file test_graph_viz_schema.cpp
 * @brief Schema validation for autograd::make_dot output (audit-2026-05-03 N4).
 *
 * Verifies the DOT string produced by make_dot is well-formed:
 *   - starts with `digraph`
 *   - parameters listed in the params map are labeled
 *   - at least one Function node appears for a non-trivial graph
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/graph_viz.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>

using namespace tenzor;

class GraphVizSchema : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

namespace {

// Build a small computation graph: y = x * w + b, returning y plus the params.
struct GraphFixture {
    Variable x, w, b, y;
};

auto build_graph() -> GraphFixture {
    Variable x(randn({4, 4}, DType::Float32, Device::cpu()), true);
    Variable w(randn({4, 4}, DType::Float32, Device::cpu()), true);
    Variable b(randn({4}, DType::Float32, Device::cpu()), true);
    auto y = tenzor::matmul(x, w) + b;
    return {x, w, b, y};
}

} // anonymous

TEST_F(GraphVizSchema, DotStartsWithDigraph) {
    auto g = build_graph();
    auto dot = make_dot(g.y, {{"x", g.x}, {"w", g.w}, {"b", g.b}});
    // Strip leading whitespace/comments.
    auto first_non_ws = dot.find_first_not_of(" \t\n\r/");
    ASSERT_NE(first_non_ws, std::string::npos);
    auto preamble = dot.substr(first_non_ws, 16);
    EXPECT_NE(preamble.find("digraph"), std::string::npos)
        << "DOT did not start with 'digraph': " << preamble;
}

TEST_F(GraphVizSchema, ParametersAreLabeled) {
    auto g = build_graph();
    auto dot = make_dot(g.y, {{"x", g.x}, {"w", g.w}, {"b", g.b}});
    EXPECT_NE(dot.find("x"), std::string::npos);
    EXPECT_NE(dot.find("w"), std::string::npos);
    EXPECT_NE(dot.find("b"), std::string::npos);
}

TEST_F(GraphVizSchema, FunctionNodesPresent) {
    auto g = build_graph();
    auto dot = make_dot(g.y, {{"x", g.x}, {"w", g.w}, {"b", g.b}});
    // The graph contains MatMul + Add — at least one Backward function name
    // should appear in the DOT output.
    bool has_function_node =
        dot.find("Backward") != std::string::npos ||
        dot.find("MatMul") != std::string::npos ||
        dot.find("Add") != std::string::npos;
    EXPECT_TRUE(has_function_node) << "Expected at least one function node in DOT";
}

TEST_F(GraphVizSchema, EmptyParamsStillWorks) {
    auto g = build_graph();
    auto dot = make_dot(g.y, {});
    auto first_non_ws = dot.find_first_not_of(" \t\n\r/");
    ASSERT_NE(first_non_ws, std::string::npos);
    EXPECT_NE(dot.find("digraph"), std::string::npos);
}
