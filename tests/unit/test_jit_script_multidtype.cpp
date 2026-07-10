/**
 * @file test_jit_script_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for JIT script compilation and evaluation
 */

#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/jit/script.hpp>

using namespace tenzor;
using namespace tenzor::testing;

class JitScriptMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(JitScriptMultiDTypeTest, ParserAcceptsSimpleReturn) {
    const char* src = R"(def forward(x): return x)";
    auto compiled = jit::compile_script(src);
    ASSERT_NE(compiled, nullptr);

    auto x = createInput({3, 4}, false);
    auto y = compiled->forward(x);
    EXPECT_EQ(y.tensor().shape()[0], 3);
    EXPECT_EQ(y.tensor().shape()[1], 4);
}

TEST_P(JitScriptMultiDTypeTest, EvaluatesMulAdd) {
    auto compiled = jit::compile_script(R"(
        def forward(x):
            return x * 2.0 + 1.0
    )");
    ASSERT_NE(compiled, nullptr);

    auto x_tensor = tenzor::full({2, 3}, 4.0f, DType::Float32, device()).to(dtype());
    auto y = compiled->forward(Variable(x_tensor, false));

    auto data = y.tensor().to(Device::cpu()).to(DType::Float32).data<float>();
    for (int64_t i = 0; i < y.tensor().numel(); ++i) {
        EXPECT_NEAR(data[i], 4.0f * 2.0f + 1.0f, atol());
    }
}

TEST_P(JitScriptMultiDTypeTest, EvaluatesOperatorPrecedence) {
    auto compiled = jit::compile_script(R"(
        def forward(x):
            return 1.0 + 2.0 * x
    )");
    auto x_tensor = tenzor::full({4}, 3.0f, DType::Float32, device()).to(dtype());
    auto y = compiled->forward(Variable(x_tensor, false));
    auto data = y.tensor().to(Device::cpu()).to(DType::Float32).data<float>();
    for (int64_t i = 0; i < y.tensor().numel(); ++i) {
        EXPECT_NEAR(data[i], 1.0f + 2.0f * 3.0f, atol());
    }
}

TEST_P(JitScriptMultiDTypeTest, EvaluatesParentheses) {
    auto compiled = jit::compile_script(R"(
        def forward(x):
            return (x + 1.0) * 3.0
    )");
    auto x_tensor = tenzor::full({2}, 5.0f, DType::Float32, device()).to(dtype());
    auto y = compiled->forward(Variable(x_tensor, false));
    auto data = y.tensor().to(Device::cpu()).to(DType::Float32).data<float>();
    for (int64_t i = 0; i < y.tensor().numel(); ++i) {
        EXPECT_NEAR(data[i], (5.0f + 1.0f) * 3.0f, atol());
    }
}

TEST_P(JitScriptMultiDTypeTest, EvaluatesDivision) {
    auto compiled = jit::compile_script(R"(
        def forward(x):
            return x / 2.0
    )");
    auto x_tensor = tenzor::full({2}, 10.0f, DType::Float32, device()).to(dtype());
    auto y = compiled->forward(Variable(x_tensor, false));
    auto data = y.tensor().to(Device::cpu()).to(DType::Float32).data<float>();
    for (int64_t i = 0; i < y.tensor().numel(); ++i) {
        EXPECT_NEAR(data[i], 5.0f, atol());
    }
}

TEST_P(JitScriptMultiDTypeTest, RejectsMultiArgFunction) {
    EXPECT_THROW({
        jit::compile_script(R"(
            def forward(x, y):
                return x + y
        )");
    }, std::runtime_error);
}

TEST_P(JitScriptMultiDTypeTest, RejectsMalformedSyntax) {
    EXPECT_THROW({
        jit::compile_script(R"(
            def forward(x)
                return x
        )");
    }, std::runtime_error);
}

TEST_P(JitScriptMultiDTypeTest, EvaluatesUnaryMinus) {
    auto compiled = jit::compile_script(R"(
        def forward(x):
            return -x + 1.0
    )");
    auto x_tensor = tenzor::full({3}, 4.0f, DType::Float32, device()).to(dtype());
    auto y = compiled->forward(Variable(x_tensor, false));
    auto data = y.tensor().to(Device::cpu()).to(DType::Float32).data<float>();
    for (int64_t i = 0; i < y.tensor().numel(); ++i) {
        EXPECT_NEAR(data[i], -4.0f + 1.0f, atol());
    }
}

// JIT-R006/R007 coverage: the pow exponent here is a device-resident,
// non-Float32 Variable reached via an IdentExpr referencing the function's
// own parameter -- not a script-source NumberExpr literal, which is always
// CPU-resident by construction and can never exercise eval_scalar's
// device/dtype-cast ordering bug. This forces eval_scalar to actually
// transfer `device()`-resident data before narrowing to Float32 (the old
// code narrowed first, running the cast via that backend's own device-side
// kernel) and to go through Tensor::item() so the JIT graph-break hook
// fires for this data-dependent scalar extraction.
//
// Must use compile_script(source, dummy) with a dummy that matches this
// test's own device()/dtype()/shape, NOT the bare compile_script(source)
// overload (whose dummy is hardcoded to CPU/Float32/{1}). x.pow(x)'s
// exponent is extracted via .item() -- a graph break -- so it freezes at
// whatever value was live during the trace that actually built the graph,
// not the real per-call input (the warning fired by this test's own
// notify_graph_break says so explicitly). With a mismatched default dummy,
// forward() retraces on device/dtype mismatch, and that retrace happens to
// use the real input as its own new trace probe -- coincidentally freezing
// the exponent at the "right" value for every dtype/device EXCEPT
// cpu_Float32, whose dummy already matches and so never retraces, freezing
// the exponent at the dummy's hardcoded 1.0 instead (giving pow(2,1)=2, not
// pow(2,2)=4). Passing a matching dummy here makes every parametrization
// trace directly on the real value, with no retrace and no dependence on
// that coincidence.
TEST_P(JitScriptMultiDTypeTest, EvaluatesPowWithDeviceResidentDynamicExponent) {
    auto x_tensor = tenzor::full({1}, 2.0f, DType::Float32, device()).to(dtype());

    auto compiled = jit::compile_script(R"(
        def forward(x):
            return x.pow(x)
    )", x_tensor);
    ASSERT_NE(compiled, nullptr);

    auto y = compiled->forward(Variable(x_tensor, false));

    auto data = y.tensor().to(Device::cpu()).to(DType::Float32).data<float>();
    EXPECT_NEAR(data[0], 4.0f, atol());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(JitScriptMultiDTypeTest);
