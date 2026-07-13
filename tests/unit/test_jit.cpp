/**
 * @file test_jit.cpp
 * @brief Comprehensive tests for JIT (Just-In-Time) compilation functionality
 *
 * Tests trace mode, graph optimization, serialization, and various model architectures.
 */

#include <gtest/gtest.h>
#include "../../include/tenzor/tenzor.hpp"
#include "../../include/tenzor/jit/tracer.hpp"
#include "../../include/tenzor/jit/compiler.hpp"
#include "../../include/tenzor/jit/serialization.hpp"
#include "../../include/tenzor/jit/graph.hpp"
#include "../../include/tenzor/backend/dispatch_interceptor.hpp"
#include <memory>
#include <fstream>
#include <filesystem>
#include <atomic>
#include <functional>

namespace fs = std::filesystem;

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::jit;

class JITTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const jit_env =
    ::testing::AddGlobalTestEnvironment(new JITTestEnvironment);

// Simple model for testing
class SimpleLinearModel : public Module {
public:
    SimpleLinearModel() {
        fc1_ = std::make_shared<Linear>(10, 20);
        fc2_ = std::make_shared<Linear>(20, 5);
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    Variable forward_impl(const Variable& x) override {
        auto out = fc1_->forward(x);
        out = fc2_->forward(out);
        return out;
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
};

class JITTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = fs::temp_directory_path() / "tenzor_jit_tests";
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
    }

    std::string get_test_path(const std::string& filename) {
        return (test_dir_ / filename).string();
    }

    fs::path test_dir_;
};

// ============================================================================
// Trace Mode Tests
// ============================================================================

TEST_F(JITTest, TraceSimpleModel) {
    auto model = std::make_shared<SimpleLinearModel>();

    // Create example input
    Tensor input_tensor({2, 10}, DType::Float32, Device::cpu());
    input_tensor.fill_(1.0f);

    // Trace returns a CompiledModule when called with a Tensor argument.
    auto traced = jit::trace(model, input_tensor);
    ASSERT_NE(traced, nullptr);

    Variable input(input_tensor, false);
    auto output = traced->forward(input);
    EXPECT_EQ(output.tensor().shape()[0], 2);
    EXPECT_EQ(output.tensor().shape()[1], 5);
}

// JIT-R138: CompiledModule::forward()'s retrace path (device/dtype/shape
// mismatch against the CompiledModule's currently-traced graph) calls
// CompiledModule::trace(source_module_, input), which for real executes
// source_module_->forward(input) once (inside jit::trace()) to build the
// fresh graph -- then discards that real output and unconditionally calls
// graph_->forward({input}) again on the SAME input to produce the value
// actually returned to the caller. Both the trace-time execution and the
// interpreter replay dispatch the real underlying ops, so whichever single
// external forward() call happens to trigger a retrace runs the module's
// computation TWICE instead of once. Detected via total dispatch-call count:
// a call that does NOT need to retrace (same shape as the last trace)
// dispatches N ops; a call that DOES need to retrace must also dispatch
// exactly N ops for a correct single-execution implementation, not 2N.
TEST_F(JITTest, ForwardRetraceDoesNotDoubleExecuteUnderlyingOps) {
    auto model = std::make_shared<SimpleLinearModel>();

    Tensor dummy({2, 10}, DType::Float32, Device::cpu());
    dummy.fill_(1.0f);
    auto traced = jit::trace(model, dummy);
    ASSERT_NE(traced, nullptr);

    auto count_dispatches_during = [](const std::function<void()>& fn) -> int {
        std::atomic<int> count{0};
        {
            InterceptorGuard guard(
                [&count](OpId op, std::span<const Tensor> inputs,
                         const OpAttributes& attrs, DispatchNext next) {
                    count.fetch_add(1, std::memory_order_relaxed);
                    return next(op, inputs, attrs);
                });
            fn();
        }
        return count.load();
    };

    // Baseline: same shape as the trace dummy -> no retrace needed, exactly
    // one graph execution's worth of dispatches.
    Tensor same_shape_input({2, 10}, DType::Float32, Device::cpu());
    same_shape_input.fill_(1.0f);
    Variable same_shape_var(same_shape_input, false);
    int baseline_dispatches = count_dispatches_during([&] {
        auto out = traced->forward(same_shape_var);
        EXPECT_EQ(out.tensor().shape()[0], 2);
    });
    ASSERT_GT(baseline_dispatches, 0);

    // A DIFFERENT batch size forces forward() onto the shape-mismatch
    // retrace path.
    Tensor retrace_input({4, 10}, DType::Float32, Device::cpu());
    retrace_input.fill_(1.0f);
    Variable retrace_var(retrace_input, false);
    int retrace_dispatches = count_dispatches_during([&] {
        auto out = traced->forward(retrace_var);
        EXPECT_EQ(out.tensor().shape()[0], 4);
    });

    EXPECT_EQ(retrace_dispatches, baseline_dispatches)
        << "forward()'s retrace path dispatched " << retrace_dispatches
        << " ops for a single external call vs. " << baseline_dispatches
        << " for a no-retrace call of the same model -- the trace-time "
           "execution's real result was discarded and the graph was "
           "executed again on the same input (JIT-R138)";
}

TEST_F(JITTest, InfersPhase13TracedOpShapes) {
    auto graph = std::make_shared<Graph>();

    auto x = graph->create_value("x", {2, 3, 8, 8}, DType::Float32, Device::cpu());
    auto w = graph->create_value("w", {3, 4, 3, 3}, DType::Float32, Device::cpu());
    auto idx = graph->create_value("idx", {5}, DType::Int64, Device::cpu());
    auto y = graph->create_value("y", {1, 3, 1, 8}, DType::Float32, Device::cpu());

    auto convt_out = graph->create_value("convt_out", {}, DType::Float32, Device::cpu());
    auto convt = graph->create_node(OpType::ConvTranspose, "convt");
    convt->add_input(x);
    convt->add_input(w);
    convt->set_vec_attr("stride", {2, 2});
    convt->set_vec_attr("padding", {1, 1});
    convt->set_vec_attr("output_padding", {1, 1});
    convt->add_output(convt_out);
    graph->add_node(convt);

    auto stack_out = graph->create_value("stack_out", {}, DType::Float32, Device::cpu());
    auto stack = graph->create_node(OpType::Stack, "stack");
    stack->add_input(x);
    stack->add_input(x);
    stack->set_int_attr("dim", 1);
    stack->add_output(stack_out);
    graph->add_node(stack);

    auto broadcast_out = graph->create_value("broadcast_out", {}, DType::Float32, Device::cpu());
    auto broadcast = graph->create_node(OpType::Broadcast, "broadcast");
    broadcast->add_input(idx);
    broadcast->set_vec_attr("shape", {2, 5});
    broadcast->add_output(broadcast_out);
    graph->add_node(broadcast);

    auto where_out = graph->create_value("where_out", {}, DType::Float32, Device::cpu());
    auto where = graph->create_node(OpType::Where, "where");
    where->add_input(x);
    where->add_input(x);
    where->add_input(y);
    where->add_output(where_out);
    graph->add_node(where);

    auto index_out = graph->create_value("index_out", {}, DType::Float32, Device::cpu());
    auto index_select = graph->create_node(OpType::IndexSelect, "index_select");
    index_select->add_input(x);
    index_select->add_input(idx);
    index_select->set_int_attr("dim", 2);
    index_select->add_output(index_out);
    graph->add_node(index_select);

    auto interp_out = graph->create_value("interp_out", {}, DType::Float32, Device::cpu());
    auto interpolate = graph->create_node(OpType::Interpolate, "interpolate");
    interpolate->add_input(x);
    interpolate->set_vec_attr("output_size", {16, 12});
    interpolate->add_output(interp_out);
    graph->add_node(interpolate);

    graph->infer_types();

    EXPECT_EQ(convt_out->shape(), (std::vector<int64_t>{2, 4, 16, 16}));
    EXPECT_EQ(stack_out->shape(), (std::vector<int64_t>{2, 2, 3, 8, 8}));
    EXPECT_EQ(broadcast_out->shape(), (std::vector<int64_t>{2, 5}));
    EXPECT_EQ(where_out->shape(), (std::vector<int64_t>{2, 3, 8, 8}));
    EXPECT_EQ(index_out->shape(), (std::vector<int64_t>{2, 3, 5, 8}));
    EXPECT_EQ(interp_out->shape(), (std::vector<int64_t>{2, 3, 16, 12}));
}


// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
