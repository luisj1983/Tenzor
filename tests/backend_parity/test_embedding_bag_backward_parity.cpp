/**
 * @file test_embedding_bag_backward_parity.cpp
 * @brief Cross-backend parity for EmbeddingBag forward + backward.
 *
 * Covers OpId::EmbeddingBagForward (435) and OpId::EmbeddingBagBackward
 * (436). The audit (2026-05-02) found zero parity-test references for
 * EmbeddingBagBackward despite the kernel being registered on every
 * non-MPS backend.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/embedding.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class EmbeddingBagParity : public BackendTest {};

namespace {

// Build a small EmbeddingBag-friendly input set:
//   indices: 12 word ids in [0, vocab)
//   offsets: 3 bags starting at 0, 4, 8 (each of size 4)
//   weight:  (vocab × embedding_dim) lookup table
struct BagInputs {
    Tensor indices;
    Tensor offsets;
    Tensor weight;
};

BagInputs make_bag_inputs(int64_t vocab = 16, int64_t emb_dim = 8) {
    BagInputs b;
    b.indices = full({12}, 0.0, DType::Int64, Device::cpu());
    int64_t pattern[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    for (int64_t i = 0; i < 12; ++i) b.indices.data<int64_t>()[i] = pattern[i] % vocab;

    b.offsets = full({3}, 0.0, DType::Int64, Device::cpu());
    b.offsets.data<int64_t>()[0] = 0;
    b.offsets.data<int64_t>()[1] = 4;
    b.offsets.data<int64_t>()[2] = 8;

    b.weight = randn({vocab, emb_dim}, DType::Float32, Device::cpu());
    return b;
}

}  // namespace

// ----------------------------------------------------------------------------
// Forward parity for each aggregation mode
// ----------------------------------------------------------------------------

TEST_P(EmbeddingBagParity, Forward_MeanMode) {
    auto b = make_bag_inputs();

    auto run = [&](Device target) {
        nn::EmbeddingBag layer(/*vocab=*/16, /*emb_dim=*/8,
                               /*max_norm=*/0.0, /*norm_type=*/2.0,
                               /*scale_grad_by_freq=*/false, /*mode=*/"mean");
        // EmbeddingBag stores its weight inside its `embedding` submodule.
// get_parameter() doesn't recurse, so we look up via named_parameters().
for (auto& [name, ptr] : layer.named_parameters()) {
    if (name == "embedding.weight") {
        *ptr = Variable(b.weight.to(target), true);
        break;
    }
}
        auto out = layer.forward(Variable(b.indices.to(target), false),
                                 Variable(b.offsets.to(target), false));
        target.synchronize();
        return out.tensor().to(Device::cpu());
    };

    auto cpu = run(Device::cpu());
    if (device.type == Device::Type::CPU) return;
    auto dev = run(device);
    EXPECT_LT(max_abs_diff(cpu, dev), 1e-4f) << "Mean mode forward diff on " << backend_name(device);
}

TEST_P(EmbeddingBagParity, Forward_SumMode) {
    auto b = make_bag_inputs();

    auto run = [&](Device target) {
        nn::EmbeddingBag layer(16, 8, 0.0, 2.0, false, "sum");
        // EmbeddingBag stores its weight inside its `embedding` submodule.
// get_parameter() doesn't recurse, so we look up via named_parameters().
for (auto& [name, ptr] : layer.named_parameters()) {
    if (name == "embedding.weight") {
        *ptr = Variable(b.weight.to(target), true);
        break;
    }
}
        auto out = layer.forward(Variable(b.indices.to(target), false),
                                 Variable(b.offsets.to(target), false));
        target.synchronize();
        return out.tensor().to(Device::cpu());
    };

    auto cpu = run(Device::cpu());
    if (device.type == Device::Type::CPU) return;
    auto dev = run(device);
    EXPECT_LT(max_abs_diff(cpu, dev), 1e-4f) << "Sum mode forward diff on " << backend_name(device);
}

TEST_P(EmbeddingBagParity, Forward_MaxMode) {
    auto b = make_bag_inputs();

    auto run = [&](Device target) {
        nn::EmbeddingBag layer(16, 8, 0.0, 2.0, false, "max");
        // EmbeddingBag stores its weight inside its `embedding` submodule.
// get_parameter() doesn't recurse, so we look up via named_parameters().
for (auto& [name, ptr] : layer.named_parameters()) {
    if (name == "embedding.weight") {
        *ptr = Variable(b.weight.to(target), true);
        break;
    }
}
        auto out = layer.forward(Variable(b.indices.to(target), false),
                                 Variable(b.offsets.to(target), false));
        target.synchronize();
        return out.tensor().to(Device::cpu());
    };

    auto cpu = run(Device::cpu());
    if (device.type == Device::Type::CPU) return;
    auto dev = run(device);
    EXPECT_LT(max_abs_diff(cpu, dev), 1e-4f) << "Max mode forward diff on " << backend_name(device);
}

// ----------------------------------------------------------------------------
// Backward parity — exercises EmbeddingBagBackward (436)
// ----------------------------------------------------------------------------

TEST_P(EmbeddingBagParity, Backward_GradWeightMatchesCPU) {
    auto b = make_bag_inputs();

    auto run = [&](Device target) {
        nn::EmbeddingBag layer(16, 8, 0.0, 2.0, false, "mean");
        // EmbeddingBag stores its weight inside its `embedding` submodule.
// get_parameter() doesn't recurse, so we look up via named_parameters().
for (auto& [name, ptr] : layer.named_parameters()) {
    if (name == "embedding.weight") {
        *ptr = Variable(b.weight.to(target), true);
        break;
    }
}
        auto out = layer.forward(Variable(b.indices.to(target), false),
                                 Variable(b.offsets.to(target), false));
        // Deterministic loss = sum(out)
        auto loss = sum(out);
        loss.backward();
        target.synchronize();
        // Same lookup pattern — get the weight Variable by full name.
        std::shared_ptr<Variable> w;
        for (auto& [name, ptr] : layer.named_parameters()) {
            if (name == "embedding.weight") { w = ptr; break; }
        }
        return (*w->grad()).to(Device::cpu());
    };

    auto cpu_grad = run(Device::cpu());
    if (device.type == Device::Type::CPU) return;
    auto dev_grad = run(device);
    EXPECT_LT(max_abs_diff(cpu_grad, dev_grad), 1e-4f)
        << "EmbeddingBagBackward grad_weight diff on " << backend_name(device);
}

// ----------------------------------------------------------------------------
// Direct kernel path — exercises the OpId::EmbeddingBagBackward dispatch with
// non-trivial indices to prove the kernel scatters to rows selected by
// `indices` (not flat position `i`). Uses indices = [5,5,5,5] so a correct
// kernel deposits all of grad into row 5 of grad_weight; the previous broken
// kernel would have written to rows 0..3.
// ----------------------------------------------------------------------------
TEST_P(EmbeddingBagParity, Kernel_ScattersByIndicesNotPosition) {
    if (!is_op_supported(OpId::EmbeddingBagBackward, device.type)) {
        GTEST_SKIP() << "EmbeddingBagBackward not supported on " << backend_name(device);
    }

    constexpr int64_t num_embeddings = 16;
    constexpr int64_t embedding_dim = 4;
    constexpr int64_t total_elements = 4;
    constexpr int64_t target_row = 5;

    // indices = [5,5,5,5]; offsets = [0] (single bag spanning all 4 elements)
    auto indices_cpu = zeros({total_elements}, DType::Int64, Device::cpu());
    for (int64_t i = 0; i < total_elements; ++i) {
        indices_cpu.data<int64_t>()[i] = target_row;
    }
    auto offsets_cpu = zeros({1}, DType::Int64, Device::cpu());
    offsets_cpu.data<int64_t>()[0] = 0;

    // grad_output = ones({1, embedding_dim})
    auto grad_out_cpu = full({1, embedding_dim}, 1.0, DType::Float32, Device::cpu());

    auto indices = indices_cpu.to(device);
    auto offsets = offsets_cpu.to(device);
    auto grad_out = grad_out_cpu.to(device);

    OpAttributes attrs;
    attrs.set(AttrKey::NumEmbeddings, num_embeddings);
    attrs.set(AttrKey::EmbeddingDim, embedding_dim);
    attrs.set(AttrKey::Mode, std::string("sum"));
    attrs.set(AttrKey::IncludeLastOffset, false);

    std::array<Tensor, 3> inputs = {grad_out, indices, offsets};
    auto grad_weight = dispatch_single<OpId::EmbeddingBagBackward>(inputs, attrs);
    device.synchronize();

    auto gw_cpu = grad_weight.to(Device::cpu()).contiguous();
    EXPECT_EQ(gw_cpu.shape()[0], num_embeddings);
    EXPECT_EQ(gw_cpu.shape()[1], embedding_dim);

    const float* p = gw_cpu.data<float>();
    // Row `target_row` should hold sum-reduction (mode=sum) of grad_output
    // distributed to all four positions in the bag — each position contributes
    // 1.0 per column, so row 5 has the value total_elements.
    for (int64_t j = 0; j < embedding_dim; ++j) {
        EXPECT_FLOAT_EQ(p[target_row * embedding_dim + j],
                        static_cast<float>(total_elements))
            << "row " << target_row << " col " << j
            << " on " << backend_name(device);
    }
    // All other rows must be exactly zero.
    for (int64_t r = 0; r < num_embeddings; ++r) {
        if (r == target_row) continue;
        for (int64_t j = 0; j < embedding_dim; ++j) {
            EXPECT_FLOAT_EQ(p[r * embedding_dim + j], 0.0f)
                << "row " << r << " col " << j
                << " on " << backend_name(device);
        }
    }
}

INSTANTIATE_BACKEND_TESTS(EmbeddingBagParity);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
