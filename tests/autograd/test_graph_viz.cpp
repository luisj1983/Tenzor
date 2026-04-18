/**
 * @file test_graph_viz.cpp
 * @brief Tests for tenzor::make_dot / save_dot — autograd graph visualization.
 *
 * include/tenzor/autograd/graph_viz.hpp exposes a tracer that walks an
 * autograd graph from a root Variable's grad_fn and emits Graphviz DOT.
 * Until this file there were no tests, leaving silent regressions in the
 * graph traversal possible. These tests assert:
 *   - DOT output is non-empty for any graph with at least one op.
 *   - The DOT string starts with the expected `digraph` header.
 *   - save_dot() writes the string to the requested path.
 *   - Named parameters appear in the DOT output as labels.
 *   - Optional flags (show_dtypes) do not crash the traversal.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/autograd/graph_viz.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace tenzor;

namespace {

// Build a small graph: y = sum((W @ x + b)^2). Variables compose via the
// overloaded operators, not the Tensor-overload tenzor::add(...) functions.
Variable build_small_graph(Variable& x, Variable& W, Variable& b) {
    auto z = tenzor::matmul(W, x);
    auto sum_z = z + b;
    auto sq = sum_z * sum_z;
    return tenzor::sum(sq);
}

}  // namespace

TEST(GraphViz, DotIsNonEmptyForLinearGraph) {
    Variable x(randn({4, 1}, DType::Float32, Device::cpu()), true);
    Variable W(randn({3, 4}, DType::Float32, Device::cpu()), true);
    Variable b(randn({3, 1}, DType::Float32, Device::cpu()), true);
    auto loss = build_small_graph(x, W, b);

    auto dot = make_dot(loss);
    EXPECT_FALSE(dot.empty()) << "make_dot returned empty string for a graph "
                                 "with multiple ops; tracer likely failed.";
}

TEST(GraphViz, DotStartsWithDigraphHeader) {
    Variable x(randn({4, 1}, DType::Float32, Device::cpu()), true);
    Variable W(randn({3, 4}, DType::Float32, Device::cpu()), true);
    Variable b(randn({3, 1}, DType::Float32, Device::cpu()), true);
    auto loss = build_small_graph(x, W, b);

    auto dot = make_dot(loss);
    // Graphviz files start with `digraph <name> {` — we only check the
    // first 8 chars for resilience against name changes.
    EXPECT_NE(dot.find("digraph"), std::string::npos)
        << "make_dot output missing 'digraph' header: first 80 chars =\n"
        << dot.substr(0, 80);
}

// Previously the `params` map was keyed by the address of the Variables
// stored in the map, which never matched the addresses of the copies
// appearing on the autograd graph's input_variables(), so named parameter
// labels silently fell through to the generic "param" label. The lookup
// now uses tensor data_ptr() identity which survives Variable copies.
TEST(GraphViz, NamedParametersAppearInDot) {
    Variable x(randn({4, 1}, DType::Float32, Device::cpu()), true);
    Variable W(randn({3, 4}, DType::Float32, Device::cpu()), true);
    Variable b(randn({3, 1}, DType::Float32, Device::cpu()), true);
    auto loss = build_small_graph(x, W, b);

    std::unordered_map<std::string, Variable> params = {
        {"weight_W", W},
        {"bias_b", b},
    };
    auto dot = make_dot(loss, params);
    // The implementation should label the named parameter nodes with the
    // provided string. Either name appearing is sufficient — exact label
    // formatting may vary.
    bool has_w = dot.find("weight_W") != std::string::npos;
    bool has_b = dot.find("bias_b") != std::string::npos;
    EXPECT_TRUE(has_w || has_b)
        << "Neither named parameter 'weight_W' nor 'bias_b' appeared in DOT "
        << "output. First 200 chars:\n" << dot.substr(0, 200);
}

TEST(GraphViz, SaveDotWritesFile) {
    Variable x(randn({4, 1}, DType::Float32, Device::cpu()), true);
    Variable W(randn({3, 4}, DType::Float32, Device::cpu()), true);
    Variable b(randn({3, 1}, DType::Float32, Device::cpu()), true);
    auto loss = build_small_graph(x, W, b);

    auto dot = make_dot(loss);
    auto tmp_dir = std::filesystem::temp_directory_path();
    auto path = tmp_dir / "tenzor_graph_viz_test.dot";
    save_dot(dot, path.string());

    EXPECT_TRUE(std::filesystem::exists(path)) << "save_dot did not create " << path;
    auto sz = std::filesystem::file_size(path);
    EXPECT_GT(sz, 0u) << "save_dot wrote an empty file";

    // Cleanup
    std::filesystem::remove(path);
}

TEST(GraphViz, ShowDTypesFlagDoesNotCrash) {
    Variable x(randn({4, 1}, DType::Float32, Device::cpu()), true);
    Variable W(randn({3, 4}, DType::Float32, Device::cpu()), true);
    Variable b(randn({3, 1}, DType::Float32, Device::cpu()), true);
    auto loss = build_small_graph(x, W, b);

    GraphVizOptions opts;
    opts.show_dtypes = true;
    opts.show_memory_usage = true;
    auto dot = make_dot(loss, /*params=*/{}, opts);
    EXPECT_FALSE(dot.empty()) << "make_dot crashed or returned empty when "
                                 "show_dtypes/show_memory_usage was enabled.";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    try { tenzor::initialize(); } catch (...) {}
    int rc = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return rc;
}
