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

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(JitScriptMultiDTypeTest);
