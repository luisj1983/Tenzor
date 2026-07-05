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

// JIT-F003: a multi-argument compiled script (built via the dummies overload)
// must be callable through the vector forward() overload. Pre-fix, the module
// was constructed without a traced signature, so the FIRST forward() thought the
// shape had changed and retraced through the single-input source-module path,
// throwing "argument count mismatch".
TEST(JitScript, MultiArgScriptForwardWorks) {
    const char* src = R"(def forward(x, y): return x + y)";
    std::vector<Tensor> dummies = {
        ones({2}, DType::Float32, Device::cpu()),
        ones({2}, DType::Float32, Device::cpu()),
    };
    auto compiled = jit::compile_script(src, dummies);
    ASSERT_NE(compiled, nullptr);

    // Same-shape call: replays the traced graph directly (no spurious retrace).
    Tensor a = full({2}, 3.0, DType::Float32, Device::cpu());
    Tensor b = full({2}, 4.0, DType::Float32, Device::cpu());
    auto outs = compiled->forward({Variable(a, false), Variable(b, false)});
    ASSERT_EQ(outs.size(), 1u);
    auto r = outs[0].tensor().to(Device::cpu());
    EXPECT_NEAR(r.data<float>()[0], 7.0f, 1e-5);
    EXPECT_NEAR(r.data<float>()[1], 7.0f, 1e-5);

    // Different-shape call: exercises the multi-input retrace closure.
    Tensor a3 = full({3}, 1.0, DType::Float32, Device::cpu());
    Tensor b3 = full({3}, 2.0, DType::Float32, Device::cpu());
    auto outs3 = compiled->forward({Variable(a3, false), Variable(b3, false)});
    ASSERT_EQ(outs3.size(), 1u);
    auto r3 = outs3[0].tensor().to(Device::cpu());
    ASSERT_EQ(r3.numel(), 3);
    EXPECT_NEAR(r3.data<float>()[0], 3.0f, 1e-5);
    EXPECT_NEAR(r3.data<float>()[2], 3.0f, 1e-5);
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

// Zero-argument method calls are now supported (relu / sigmoid / tanh / exp /
// log / sqrt / abs / neg / sum / mean / sin / cos).
TEST(JitScript, MethodCallMean) {
    auto compiled = jit::compile_script(R"(
        def forward(x):
            return x.mean()
    )");
    auto input = Variable(ones({4}, DType::Float32, Device::cpu()) * 3.0f, false);
    auto out = compiled->forward(input);
    EXPECT_NEAR(out.tensor().item<float>(), 3.0f, 1e-6f);
}

TEST(JitScript, MethodCallChained) {
    auto compiled = jit::compile_script(R"(
        def forward(x):
            return (x * x).sum().sqrt()
    )");
    auto input = Variable(ones({3}, DType::Float32, Device::cpu()) * 2.0f, false);
    // sum(x*x) = 4+4+4 = 12; sqrt = ~3.4641
    auto out = compiled->forward(input);
    EXPECT_NEAR(out.tensor().item<float>(), std::sqrt(12.0f), 1e-5f);
}

TEST(JitScript, RejectsUnknownMethod) {
    EXPECT_THROW({
        jit::compile_script(R"(
            def forward(x):
                return x.does_not_exist()
        )");
    }, std::runtime_error);
}

// Variable assignment.
TEST(JitScript, Assignment) {
    auto compiled = jit::compile_script(R"(
        def forward(x):
            y = x * 2.0
            z = y + 1.0
            return z * z
    )");
    auto input = Variable(ones({1}, DType::Float32, Device::cpu()) * 3.0f, false);
    // y = 6, z = 7, z*z = 49
    auto out = compiled->forward(input);
    EXPECT_NEAR(out.tensor().item<float>(), 49.0f, 1e-5f);
}

TEST(JitScript, AssignmentWithMethodCall) {
    auto compiled = jit::compile_script(R"(
        def forward(x):
            s = x.sigmoid()
            return s * s
    )");
    auto input = Variable(ones({1}, DType::Float32, Device::cpu()) * 0.0f, false);
    // sigmoid(0) = 0.5 → 0.25
    auto out = compiled->forward(input);
    EXPECT_NEAR(out.tensor().item<float>(), 0.25f, 1e-5f);
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

// ---------------------------------------------------------------------------
// Control flow: if / else and for-range
// ---------------------------------------------------------------------------

// Control flow is traced through jit::cond: during trace() with a fixed
// dummy, the taken branch's ops are recorded. The compiled module replays
// those ops for any subsequent input (a standard trace-based JIT semantic).
// Full dynamic branch dispatch via trace_if subgraphs is still deferred.
TEST(JitScript, IfElseThenPathTracedWithPositiveDummy) {
    const char* src = R"(
        def forward(x):
            if x > 0.0:
                y = x * 2.0
            else:
                y = x.neg()
            return y + 1.0
    )";
    Tensor dummy = full({}, 1.0f, DType::Float32, Device::cpu());   // 1 > 0 → then
    auto compiled = jit::compile_script(src, dummy);
    auto x_pos = Variable(full({}, 3.0f, DType::Float32, Device::cpu()), false);
    EXPECT_NEAR(compiled->forward(x_pos).tensor().item<float>(), 7.0f, 1e-5f);
}

TEST(JitScript, IfElseElsePathTracedWithNegativeDummy) {
    const char* src = R"(
        def forward(x):
            if x > 0.0:
                y = x * 2.0
            else:
                y = x.neg()
            return y + 1.0
    )";
    Tensor dummy = full({}, -1.0f, DType::Float32, Device::cpu());  // -1 > 0 false → else
    auto compiled = jit::compile_script(src, dummy);
    auto x_neg = Variable(full({}, -2.0f, DType::Float32, Device::cpu()), false);
    // Else branch: y = -(-2) = 2; return 2 + 1 = 3.
    EXPECT_NEAR(compiled->forward(x_neg).tensor().item<float>(), 3.0f, 1e-5f);
}

TEST(JitScript, IfWithoutElse) {
    const char* src = R"(
        def forward(x):
            y = x
            if x > 0.0:
                y = x * x
            return y
    )";
    Tensor dummy = full({}, 1.0f, DType::Float32, Device::cpu());
    auto compiled = jit::compile_script(src, dummy);
    auto x_pos = Variable(full({}, 4.0f, DType::Float32, Device::cpu()), false);
    EXPECT_NEAR(compiled->forward(x_pos).tensor().item<float>(), 16.0f, 1e-5f);
}

TEST(JitScript, ForRangeUnrolled) {
    // y = x; for 4 iterations y = y * 2 → y = x * 16
    auto compiled = jit::compile_script(R"(
        def forward(x):
            y = x
            for i in range(4):
                y = y * 2.0
            return y
    )");
    auto x = Variable(full({}, 1.0f, DType::Float32, Device::cpu()), false);
    EXPECT_NEAR(compiled->forward(x).tensor().item<float>(), 16.0f, 1e-5f);
}

TEST(JitScript, ForRangeZero) {
    // Loop runs zero times; returned unchanged.
    auto compiled = jit::compile_script(R"(
        def forward(x):
            y = x
            for i in range(0):
                y = y * 999.0
            return y
    )");
    auto x = Variable(full({}, 7.0f, DType::Float32, Device::cpu()), false);
    EXPECT_NEAR(compiled->forward(x).tensor().item<float>(), 7.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Method calls with arguments
// ---------------------------------------------------------------------------

TEST(JitScript, MethodPow) {
    auto compiled = jit::compile_script(R"(
        def forward(x):
            return x.pow(2.0) + 1.0
    )");
    auto x = Variable(full({}, 3.0f, DType::Float32, Device::cpu()), false);
    EXPECT_NEAR(compiled->forward(x).tensor().item<float>(), 10.0f, 1e-5f);
}

TEST(JitScript, MethodClamp) {
    auto compiled = jit::compile_script(R"(
        def forward(x):
            return x.clamp(-1.0, 1.0)
    )");
    auto x_hi = Variable(full({}, 5.0f, DType::Float32, Device::cpu()), false);
    EXPECT_NEAR(compiled->forward(x_hi).tensor().item<float>(), 1.0f, 1e-6f);
    auto x_lo = Variable(full({}, -5.0f, DType::Float32, Device::cpu()), false);
    EXPECT_NEAR(compiled->forward(x_lo).tensor().item<float>(), -1.0f, 1e-6f);
    auto x_mid = Variable(full({}, 0.5f, DType::Float32, Device::cpu()), false);
    EXPECT_NEAR(compiled->forward(x_mid).tensor().item<float>(), 0.5f, 1e-6f);
}

TEST(JitScript, MethodSumDim) {
    Tensor dummy = ones({3, 4}, DType::Float32, Device::cpu());
    auto compiled = jit::compile_script(R"(
        def forward(x):
            return x.sum(1)
    )", dummy);
    Tensor data = ones({3, 4}, DType::Float32, Device::cpu()) * 2.0f;
    auto x = Variable(data, false);
    auto out = compiled->forward(x);
    // sum over dim=1 of a (3,4) tensor with each entry 2 → (3,) with each 8.
    EXPECT_EQ(out.tensor().shape().size(), 1u);
    EXPECT_EQ(out.tensor().shape()[0], 3);
    auto* p = out.tensor().data<float>();
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(p[i], 8.0f, 1e-5f);
}

TEST(JitScript, MethodCallArgCountMismatch) {
    EXPECT_THROW({
        auto c = jit::compile_script(R"(
            def forward(x):
                return x.pow(1.0, 2.0)
        )");
        auto x = Variable(full({}, 3.0f, DType::Float32, Device::cpu()), false);
        c->forward(x);
    }, std::runtime_error);
}

TEST(JitScript, RejectsMalformedIfMissingColon) {
    EXPECT_THROW({
        jit::compile_script(R"(
            def forward(x):
                if x > 0
                    y = x
                return x
        )");
    }, std::runtime_error);
}

// JIT-003: a high-precision f64 literal in a scripted fn must NOT be truncated
// to float32. Pre-fix, `x * 3.141592653589793` in an f64 context multiplied by
// the f32 value (3.1415927410125732), ~1e-7 relative error vs eager.
TEST(JitScript, Float64LiteralNotTruncated) {
    const char* src = R"(def forward(x): return x * 3.141592653589793)";
    Tensor dummy = ones({1}, DType::Float64, Device::cpu());
    auto compiled = jit::compile_script(src, dummy);
    ASSERT_NE(compiled, nullptr);
    Tensor xin = ones({1}, DType::Float64, Device::cpu());  // x = 1.0
    Variable y = compiled->forward(Variable(xin, false));
    double got = y.tensor().to(Device::cpu()).to(DType::Float64).item<double>();
    const double pi = 3.141592653589793;
    EXPECT_LT(std::abs(got - pi), 1e-15)
        << "scripted f64 literal was truncated (got " << got << ")";
}
