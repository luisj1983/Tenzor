/**
 * @file test_jit_script.cpp
 * @brief Focused tests for tenzor::jit::compile_script — covers the
 * lexer / parser / evaluator contract of the MVP grammar independently
 * of the broader JIT test suite.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/script.hpp>
#include <cmath>
#include <stdexcept>

using namespace tenzor;

namespace {

class JitScriptTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};

[[maybe_unused]] static ::testing::Environment* const script_env =
    ::testing::AddGlobalTestEnvironment(new JitScriptTestEnvironment);

Variable ones_var(std::initializer_list<int64_t> shape) {
    std::vector<int64_t> s(shape);
    return Variable(ones(s, DType::Float32, Device::cpu()), false);
}

} // namespace

// ============================================================================
// Parser — valid grammar
// ============================================================================

TEST(JitScript, ParserAcceptsSimpleReturn) {
    const char* src = R"(def forward(x): return x)";
    auto compiled = jit::compile_script(src);
    ASSERT_NE(compiled, nullptr);

    auto x = ones_var({3, 4});
    auto y = compiled->forward(x);
    EXPECT_EQ(y.tensor().shape()[0], 3);
    EXPECT_EQ(y.tensor().shape()[1], 4);
}

TEST(JitScript, ParserAcceptsIndentedBody) {
    const char* src = R"(
        def forward(x):
            return x + 1.0
    )";
    auto compiled = jit::compile_script(src);
    ASSERT_NE(compiled, nullptr);
}

TEST(JitScript, ParserAcceptsComment) {
    const char* src = R"(
        # multiply by two
        def forward(x):
            return x * 2.0
    )";
    auto compiled = jit::compile_script(src);
    ASSERT_NE(compiled, nullptr);
}

// ============================================================================
// Codegen — numerical correctness
// ============================================================================

TEST(JitScript, EvaluatesMulAdd) {
    // y = x * 2.0 + 1.0
    auto compiled = jit::compile_script(R"(
        def forward(x):
            return x * 2.0 + 1.0
    )");
    ASSERT_NE(compiled, nullptr);

    auto x_tensor = full({2, 3}, 4.0f, DType::Float32, Device::cpu());
    auto y = compiled->forward(Variable(x_tensor, false));

    auto data = y.tensor().to(Device::cpu()).to(DType::Float32).data<float>();
    for (int64_t i = 0; i < y.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 4.0f * 2.0f + 1.0f);
    }
}

TEST(JitScript, EvaluatesOperatorPrecedence) {
    // y = 1.0 + 2.0 * x — * binds tighter than +
    auto compiled = jit::compile_script(R"(
        def forward(x):
            return 1.0 + 2.0 * x
    )");
    auto x_tensor = full({4}, 3.0f, DType::Float32, Device::cpu());
    auto y = compiled->forward(Variable(x_tensor, false));
    auto data = y.tensor().to(Device::cpu()).to(DType::Float32).data<float>();
    for (int64_t i = 0; i < y.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f + 2.0f * 3.0f);
    }
}

TEST(JitScript, EvaluatesParentheses) {
    // y = (x + 1.0) * 3.0
    auto compiled = jit::compile_script(R"(
        def forward(x):
            return (x + 1.0) * 3.0
    )");
    auto x_tensor = full({2}, 5.0f, DType::Float32, Device::cpu());
    auto y = compiled->forward(Variable(x_tensor, false));
    auto data = y.tensor().to(Device::cpu()).to(DType::Float32).data<float>();
    for (int64_t i = 0; i < y.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], (5.0f + 1.0f) * 3.0f);
    }
}

TEST(JitScript, EvaluatesUnaryMinus) {
    // y = -x + 1.0
    auto compiled = jit::compile_script(R"(
        def forward(x):
            return -x + 1.0
    )");
    auto x_tensor = full({3}, 4.0f, DType::Float32, Device::cpu());
    auto y = compiled->forward(Variable(x_tensor, false));
    auto data = y.tensor().to(Device::cpu()).to(DType::Float32).data<float>();
    for (int64_t i = 0; i < y.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], -4.0f + 1.0f);
    }
}

TEST(JitScript, EvaluatesDivision) {
    auto compiled = jit::compile_script(R"(
        def forward(x):
            return x / 2.0
    )");
    auto x_tensor = full({2}, 10.0f, DType::Float32, Device::cpu());
    auto y = compiled->forward(Variable(x_tensor, false));
    auto data = y.tensor().to(Device::cpu()).to(DType::Float32).data<float>();
    for (int64_t i = 0; i < y.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 5.0f);
    }
}

// ============================================================================
// Parser / evaluator — error cases
// ============================================================================

TEST(JitScript, RejectsMultiArgFunction) {
    // The MVP only supports single-argument functions — multi-arg requires
    // the vector CompiledModule::forward overload.
    EXPECT_THROW({
        jit::compile_script(R"(
            def forward(x, y):
                return x + y
        )");
    }, std::runtime_error);
}

TEST(JitScript, RejectsIfElse) {
    EXPECT_THROW({
        jit::compile_script(R"(
            def forward(x):
                if x > 0:
                    return x
                else:
                    return -x
        )");
    }, std::runtime_error);
}

TEST(JitScript, RejectsMethodCall) {
    EXPECT_THROW({
        jit::compile_script(R"(
            def forward(x):
                return x.mean()
        )");
    }, std::runtime_error);
}

TEST(JitScript, RejectsMissingReturn) {
    EXPECT_THROW({
        jit::compile_script(R"(
            def forward(x):
                x * 2.0
        )");
    }, std::runtime_error);
}

TEST(JitScript, RejectsUnknownIdentifier) {
    EXPECT_THROW({
        jit::compile_script(R"(
            def forward(x):
                return y
        )");
    }, std::runtime_error);
}

TEST(JitScript, RejectsMalformedSyntax) {
    // Missing ':' after signature
    EXPECT_THROW({
        jit::compile_script(R"(
            def forward(x)
                return x
        )");
    }, std::runtime_error);
}

TEST(JitScript, RejectsNullSource) {
    EXPECT_THROW({ jit::compile_script(nullptr); }, std::runtime_error);
}
