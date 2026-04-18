/**
 * @file test_indexing_parity.cpp
 * @brief Indexing/gather/scatter operation parity tests across backends
 *
 * Tests index_select, gather, scatter, scatter_add, masked_select, masked_fill,
 * where, take, put, nonzero, and one_hot operations. Most indexing ops produce
 * exact results (rtol=0, atol=0) since they just move or copy data without
 * arithmetic.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class IndexingParity : public BackendTest {};
// ============================================================================
// Indexing Operations Parity Tests (14 tests)
// ============================================================================

TEST_P(IndexingParity, IndexSelect_Dim0) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = generate_uniform_tensor({32, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu());
    auto idx = randint(0, 32, {8}, DType::Int64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return index_select(inputs[0], 0, inputs[1]);
    }, {input, idx}, 0, 0, "IndexSelect_Dim0");
}

TEST_P(IndexingParity, Gather_Dim0) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = generate_uniform_tensor({32, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu());
    auto idx = randint(0, 32, {8, 32}, DType::Int64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return gather(inputs[0], 0, inputs[1]);
    }, {input, idx}, 0, 0, "Gather_Dim0");
}

TEST_P(IndexingParity, Gather_Dim1) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = generate_uniform_tensor({32, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu());
    auto idx = randint(0, 32, {32, 8}, DType::Int64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return gather(inputs[0], 1, inputs[1]);
    }, {input, idx}, 0, 0, "Gather_Dim1");
}

TEST_P(IndexingParity, Scatter_Dim0) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = zeros({32, 32}, DType::Float32, Device::cpu());
    auto src = generate_uniform_tensor({8, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu());
    auto idx = randint(0, 32, {8, 32}, DType::Int64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return scatter(inputs[0], 0, inputs[2], inputs[1]);
    }, {input, src, idx}, 0, 0, "Scatter_Dim0");
}

TEST_P(IndexingParity, ScatterAdd) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = zeros({32, 32}, DType::Float32, Device::cpu());
    auto src = generate_uniform_tensor({8, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu());
    auto idx = randint(0, 32, {8, 32}, DType::Int64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return scatter_add(inputs[0], 0, inputs[2], inputs[1]);
    }, {input, src, idx}, 0, 0, "ScatterAdd");
}

TEST_P(IndexingParity, MaskedFill) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = generate_uniform_tensor({32, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu());
    auto mask_src = randn({32, 32}, DType::Float32, Device::cpu());
    auto mask = gt(mask_src, zeros({32, 32}, DType::Float32, Device::cpu()));

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return masked_fill(inputs[0], inputs[1], -999.0f);
    }, {input, mask}, 0, 0, "MaskedFill");
}

TEST_P(IndexingParity, MaskedSelect) {
    // Output size varies by backend ordering is not guaranteed, so we
    // compare sorted results manually across backends.
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = generate_uniform_tensor({32, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu());
    auto mask_src = randn({32, 32}, DType::Float32, Device::cpu());
    auto mask = gt(mask_src, zeros({32, 32}, DType::Float32, Device::cpu()));

    // Compute reference on CPU
    auto ref = masked_select(input, mask);
    auto [ref_sorted_v, ref_indices] = sort(Variable(ref.to(Device::cpu()), false), 0);
            auto ref_sorted = ref_sorted_v.tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        auto backend = backends[i];
        try {
            auto inp_dev = input.to(backend);
            auto mask_dev = mask.to(backend);
            backend.synchronize();

            auto result = masked_select(inp_dev, mask_dev);
            backend.synchronize();

            auto result_cpu = result.to(Device::cpu());
            auto [result_sorted_v, result_indices] = sort(Variable(result_cpu, false), 0);
            auto result_sorted = result_sorted_v.tensor();

            EXPECT_EQ(ref_sorted.numel(), result_sorted.numel())
                << "MaskedSelect size mismatch on " << backend_name(backend);
            EXPECT_TENSORS_CLOSE(ref_sorted, result_sorted, 0.0f, 0.0f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "MaskedSelect failed on " << backend_name(backend)
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(IndexingParity, Where) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto cond_src = randn({32, 32}, DType::Float32, Device::cpu());
    auto condition = gt(cond_src, zeros({32, 32}, DType::Float32, Device::cpu()));
    auto x = generate_uniform_tensor({32, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu(), 11111);
    auto y = generate_uniform_tensor({32, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu(), 22222);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return where(inputs[0], inputs[1], inputs[2]);
    }, {condition, x, y}, 0, 0, "Where");
}

TEST_P(IndexingParity, Take) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = generate_uniform_tensor({32, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu());
    auto idx = randint(0, 1024, {64}, DType::Int64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return take(inputs[0], inputs[1]);
    }, {input, idx}, 0, 0, "Take");
}

TEST_P(IndexingParity, Put) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = generate_uniform_tensor({32, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu());
    auto idx = randint(0, 1024, {16}, DType::Int64, Device::cpu());
    auto source = generate_uniform_tensor({16}, -5.0f, 5.0f, DType::Float32, Device::cpu(), 99999);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return put(inputs[0], inputs[1], inputs[2]);
    }, {input, idx, source}, 0, 0, "Put");
}

TEST_P(IndexingParity, Nonzero) {
    // Output order may differ across backends, so sort before comparison.
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Create input with some zeros: uniform in [-1, 1], then zero out small values
    auto input = generate_uniform_tensor({32, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu());
    auto threshold = full({32, 32}, 0.3f, DType::Float32, Device::cpu());
    // Zero out elements where |input| < 0.3
    auto abs_input = abs(input);
    auto keep_mask = ge(abs_input, threshold);
    input = where(keep_mask, input, zeros({32, 32}, DType::Float32, Device::cpu()));

    // Compute reference on CPU
    auto ref = nonzero(input);
    auto ref_cpu = ref.to(Device::cpu());

    for (size_t i = 1; i < backends.size(); ++i) {
        auto backend = backends[i];
        try {
            auto inp_dev = input.to(backend);
            backend.synchronize();

            auto result = nonzero(inp_dev);
            backend.synchronize();

            auto result_cpu = result.to(Device::cpu());

            // Both should have the same number of nonzero elements
            EXPECT_EQ(ref_cpu.shape()[0], result_cpu.shape()[0])
                << "Nonzero count mismatch on " << backend_name(backend);

            // Sort each row lexicographically by sorting on dim 0
            // (nonzero returns N x ndim indices)
            if (ref_cpu.numel() > 0 && result_cpu.numel() > 0) {
                auto [ref_s_v, ref_si] = sort(Variable(ref_cpu, false), 0);
                auto ref_s = ref_s_v.tensor();
                auto [res_s_v, res_si] = sort(Variable(result_cpu, false), 0);
                auto res_s = res_s_v.tensor();
                EXPECT_TENSORS_CLOSE(ref_s, res_s, 0.0f, 0.0f);
            }
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Nonzero failed on " << backend_name(backend)
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(IndexingParity, OneHot) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randint(0, 10, {16}, DType::Int64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return one_hot(inputs[0], 10);
    }, {input}, 0, 0, "OneHot");
}

TEST_P(IndexingParity, IndexSelect_Dim1) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = generate_uniform_tensor({32, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu());
    auto idx = randint(0, 32, {4}, DType::Int64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return index_select(inputs[0], 1, inputs[1]);
    }, {input, idx}, 0, 0, "IndexSelect_Dim1");
}

TEST_P(IndexingParity, Where_Broadcast) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto cond_src = randn({32, 1}, DType::Float32, Device::cpu());
    auto condition = gt(cond_src, zeros({32, 1}, DType::Float32, Device::cpu()));
    auto x = generate_uniform_tensor({32, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu(), 33333);
    auto y = generate_uniform_tensor({32, 32}, -1.0f, 1.0f, DType::Float32, Device::cpu(), 44444);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return where(inputs[0], inputs[1], inputs[2]);
    }, {condition, x, y}, 0, 0, "Where_Broadcast");
}

INSTANTIATE_BACKEND_TESTS(IndexingParity);




int main(int argc, char** argv) {
    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
    }
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    try {
        tenzor::finalize();
    } catch (...) {}
    return result;
}
