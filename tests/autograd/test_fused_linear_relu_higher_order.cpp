// test_fused_linear_relu_higher_order.cpp
//
// Audit D1: FusedLinearReLUBackward must build a real Variable-level
// graph for higher-order gradients.
//
// Previously `backward_with_variables` called the tensor-level backward and
// rewrapped results as Variable(t, requires_grad=true) with no grad_fn,
// silently severing the graph (the "raw-tensor-op breaks autograd graph"
// pattern from project memory). The new body composes with autograd-level
// ops so the graph is intact.
//
// Test strategy: construct a FusedLinearReLUBackward directly, populate
// its saved state, call `backward_with_variables` on a grad with
// requires_grad=true, and assert each returned Variable carries a
// grad_fn -- proof that the graph was preserved.

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/function.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>

using namespace tenzor;

namespace {

class D1Test : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }

    // Helper: build a FusedLinearReLUBackward instance whose saved state
    // matches a forward of  z = ReLU(x @ W.T + b)  with shapes:
    //   x: [B, IN], W: [OUT, IN], z: [B, OUT].
    // Saves input + weight, sets relu_output = z (post-ReLU).
    static auto build_backward(int64_t B, int64_t IN, int64_t OUT)
        -> std::shared_ptr<FusedLinearReLUBackward>
    {
        auto fn = std::make_shared<FusedLinearReLUBackward>();

        // Input x: positive in some elements, negative in others, so the
        // ReLU mask isn't trivially all-ones.
        auto x = zeros({B, IN}, DType::Float32, Device::cpu());
        auto* xp = x.data<float>();
        for (int64_t i = 0; i < x.numel(); ++i) {
            xp[i] = static_cast<float>((i * 7) % 13) - 6.0f;
        }

        // Weight W: ones (any non-trivial pattern works).
        auto W = zeros({OUT, IN}, DType::Float32, Device::cpu());
        auto* wp = W.data<float>();
        for (int64_t i = 0; i < W.numel(); ++i) {
            wp[i] = 1.0f;
        }

        // Simulate the post-ReLU output to populate `relu_output_`. We pick
        // a deterministic pattern that has both positive and zero entries
        // so the mask exercises both branches.
        auto z = zeros({B, OUT}, DType::Float32, Device::cpu());
        auto* zp = z.data<float>();
        for (int64_t i = 0; i < z.numel(); ++i) {
            zp[i] = ((i % 3) == 0) ? 0.0f : 1.5f;
        }
        fn->set_relu_output(z);
        fn->save_for_backward({x, W});
        return fn;
    }
};

} // namespace

TEST_F(D1Test, BackwardWithVariables_ReturnsVariablesWithGradFn) {
    auto fn = build_backward(/*B=*/3, /*IN=*/4, /*OUT=*/2);

    // grad_out has requires_grad=true so the test can detect whether the
    // graph was preserved (the result variables should carry grad_fns
    // descending from grad_out).
    auto grad_out_t = zeros({3, 2}, DType::Float32, Device::cpu());
    auto* p = grad_out_t.data<float>();
    for (int64_t i = 0; i < grad_out_t.numel(); ++i) p[i] = 1.0f;
    Variable grad_out(grad_out_t, /*requires_grad=*/true);

    auto results = fn->backward_with_variables({grad_out});
    ASSERT_EQ(results.size(), 2u);  // grad_input, grad_weight

    // Both results must descend from `grad_out` in the autograd graph.
    // If `backward_with_variables` had used the legacy raw-tensor +
    // rewrap path, grad_fn would be null.
    EXPECT_TRUE(results[0].requires_grad())
        << "grad_input should require grad (descended from grad_out)";
    EXPECT_TRUE(results[1].requires_grad())
        << "grad_weight should require grad (descended from grad_out)";
    EXPECT_NE(results[0].grad_fn(), nullptr)
        << "grad_input must carry a grad_fn — Variable-level graph preserved";
    EXPECT_NE(results[1].grad_fn(), nullptr)
        << "grad_weight must carry a grad_fn — Variable-level graph preserved";
}

TEST_F(D1Test, BackwardWithVariables_ShapesMatchTensorPath) {
    auto fn = build_backward(/*B=*/3, /*IN=*/4, /*OUT=*/2);

    auto grad_out_t = zeros({3, 2}, DType::Float32, Device::cpu());
    auto* p = grad_out_t.data<float>();
    for (int64_t i = 0; i < grad_out_t.numel(); ++i) p[i] = 1.0f;
    Variable grad_out(grad_out_t, /*requires_grad=*/true);

    auto var_results = fn->backward_with_variables({grad_out});
    auto tensor_results = fn->backward({grad_out_t});

    ASSERT_EQ(var_results.size(), tensor_results.size());
    for (size_t i = 0; i < var_results.size(); ++i) {
        // Same shape.
        EXPECT_EQ(var_results[i].tensor().shape().size(),
                  tensor_results[i].shape().size());
        for (size_t d = 0; d < var_results[i].tensor().shape().size(); ++d) {
            EXPECT_EQ(var_results[i].tensor().shape()[d],
                      tensor_results[i].shape()[d]);
        }
        // Same values (within float tolerance).
        auto* vp = var_results[i].tensor().data<float>();
        auto* tp = tensor_results[i].data<float>();
        int64_t n = var_results[i].tensor().numel();
        ASSERT_EQ(n, tensor_results[i].numel());
        for (int64_t k = 0; k < n; ++k) {
            EXPECT_NEAR(vp[k], tp[k], 1e-5f)
                << "result " << i << " elem " << k;
        }
    }
}

TEST_F(D1Test, BackwardWithVariables_NoGradGradOut_GradFnCanBeNull) {
    // When grad_out itself does NOT require grad, the Variable-level path
    // produces results without grad_fn — that's fine, there's no graph to
    // preserve. This test pins down the contract.
    auto fn = build_backward(3, 4, 2);
    auto grad_out_t = zeros({3, 2}, DType::Float32, Device::cpu());
    auto* p = grad_out_t.data<float>();
    for (int64_t i = 0; i < grad_out_t.numel(); ++i) p[i] = 1.0f;
    Variable grad_out(grad_out_t, /*requires_grad=*/false);

    auto results = fn->backward_with_variables({grad_out});
    ASSERT_EQ(results.size(), 2u);
    // No graph to preserve — requires_grad on results may be false.
    // We just verify the call doesn't crash and shapes are sensible.
    EXPECT_EQ(results[0].tensor().shape().size(), 2u);
    EXPECT_EQ(results[1].tensor().shape().size(), 2u);
}
