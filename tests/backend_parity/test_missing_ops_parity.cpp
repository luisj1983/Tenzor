#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/fft.hpp>
#include <tenzor/ops/advanced.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/indexing.hpp>
#include "../backend_test_fixture.hpp"
#include <cmath>
#include <optional>

using namespace tenzor;
using namespace tenzor::testing;

/**
 * @file test_missing_ops_parity.cpp
 * @brief Verify previously CUDA-exclusive ops now work on ROCm, OneAPI, Vulkan.
 *        Tests dispatch to the actual GPU backend and compare output to CPU
 *        reference.
 *
 * FF.29: Migrated from the helper-fn pattern (`testOnGPUBackends` + TEST_F) to
 *        BackendTest TEST_P, so each (op × backend) cell appears in the parity
 *        coverage matrix. The CPU axis is a degenerate identity test (the CPU
 *        ref is the comparison reference) and is allowed: BackendTest's
 *        `device` is set to Device::cpu() on the cpu axis, which makes the
 *        comparison cpu vs cpu and trivially passes — this is the right
 *        contract for a matrix cell tracker (says "yes, CPU is covered").
 */

class MissingOpsParity : public BackendTest {
protected:
    // Compare tensors with tolerance
    static bool close(const Tensor& a, const Tensor& b, double atol = 1e-5) {
        auto ac = a.to(Device::cpu());
        auto bc = b.to(Device::cpu());
        if (ac.numel() != bc.numel()) return false;
        const float* ad = ac.data<float>();
        const float* bd = bc.data<float>();
        for (size_t i = 0; i < ac.numel(); ++i) {
            if (std::abs(ad[i] - bd[i]) > atol) return false;
        }
        return true;
    }

    // Exact element-wise integer comparison (bucket/bin indices and counts must
    // match exactly across backends — a 1-off bucket index is a real bug, not
    // numerical drift). Casts both to Int64 on CPU first.
    static bool equal_int(const Tensor& a, const Tensor& b) {
        auto ac = a.to(Device::cpu()).to(DType::Int64);
        auto bc = b.to(Device::cpu()).to(DType::Int64);
        if (ac.numel() != bc.numel()) return false;
        const int64_t* ad = ac.data<int64_t>();
        const int64_t* bd = bc.data<int64_t>();
        for (size_t i = 0; i < ac.numel(); ++i) {
            if (ad[i] != bd[i]) return false;
        }
        return true;
    }
};

TEST_P(MissingOpsParity, Bernoulli) {
    // Create on CPU then move — avoids potential scalar mul issues on some backends
    auto probs_cpu = ones({100}, DType::Float32, Device::cpu()) * 0.5f;
    auto probs = probs_cpu.to(device);
    auto result = bernoulli(probs);
    EXPECT_EQ(result.shape()[0], 100);
    auto cpu_result = result.to(Device::cpu());
    const float* data = cpu_result.data<float>();
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_TRUE(data[i] == 0.0f || data[i] == 1.0f)
            << "Value at " << i << " = " << data[i];
    }
}

TEST_P(MissingOpsParity, Multinomial) {
    // Uniform probabilities
    auto probs = ones({1, 4}, DType::Float32, device);
    auto result = multinomial(probs, 2, /*replacement=*/true);
    EXPECT_EQ(result.shape()[0], 1);
    EXPECT_EQ(result.shape()[1], 2);
}

TEST_P(MissingOpsParity, Bucketize) {
    auto input = randn({10}, DType::Float32, Device::cpu());
    auto boundaries = linspace(0.0, 1.0, 5);

    auto cpu_ref = tenzor::bucketize(input, boundaries);
    auto gpu_result = tenzor::bucketize(input.to(device), boundaries.to(device));

    auto cr = cpu_ref.to(Device::cpu());
    auto gr = gpu_result.to(Device::cpu());
    EXPECT_EQ(cr.numel(), gr.numel());
    // Bucket indices are integers — they must match the CPU reference exactly.
    // A backend ignoring boundaries (returning zeros) or off-by-one bucketing
    // is caught here; a numel-only check would have passed it.
    EXPECT_TRUE(equal_int(cpu_ref, gpu_result))
        << "GPU bucketize indices diverge from CPU reference";
}

TEST_P(MissingOpsParity, CDist) {
    auto x1 = randn({5, 3}, DType::Float32, Device::cpu());
    auto x2 = randn({4, 3}, DType::Float32, Device::cpu());

    auto cpu_ref = cdist(x1, x2, 2.0);
    auto gpu_result = cdist(x1.to(device), x2.to(device), 2.0);

    EXPECT_TRUE(close(cpu_ref, gpu_result.to(Device::cpu()), 1e-4));
}

TEST_P(MissingOpsParity, STFT) {
    auto signal = randn({256}, DType::Float32, Device::cpu());
    int64_t n_fft = 64;

    auto cpu_ref = tenzor::fft::stft(signal, n_fft);
    EXPECT_GT(cpu_ref.numel(), 0);

    auto gpu_result = tenzor::fft::stft(signal.to(device), n_fft);
    EXPECT_EQ(cpu_ref.shape().size(), gpu_result.shape().size());
    ASSERT_EQ(cpu_ref.numel(), gpu_result.numel());
    // Compare the actual spectrogram values, not just rank/numel. The STFT
    // output is complex; abs() yields the real magnitude spectrogram, which we
    // compare element-wise. A backend returning a correctly-shaped zero/NaN
    // tensor (or wrong spectrum) fails here.
    auto cpu_mag = tenzor::abs(cpu_ref);
    auto gpu_mag = tenzor::abs(gpu_result).to(Device::cpu());
    EXPECT_TRUE(close(cpu_mag, gpu_mag, 1e-3))
        << "GPU STFT magnitude spectrogram diverges from CPU reference";
}

TEST_P(MissingOpsParity, STFT_ISTFT_RoundTrip) {
    auto signal = randn({256}, DType::Float32, Device::cpu());
    int64_t n_fft = 64;

    auto sig_dev = signal.to(device);
    auto spec = tenzor::fft::stft(sig_dev, n_fft);
    auto recon = tenzor::fft::istft(spec, n_fft, -1, -1, Tensor{},
                                    /*center=*/true, /*normalized=*/false,
                                    /*onesided=*/true, 256);
    EXPECT_EQ(recon.shape().size(), signal.shape().size());
    EXPECT_EQ(recon.numel(), signal.numel());
    EXPECT_TRUE(close(signal, recon.to(Device::cpu()), 1e-3));
}

TEST_P(MissingOpsParity, Histogram) {
    auto input = randn({100}, DType::Float32, Device::cpu());
    auto [cpu_counts, cpu_edges] = histogram(input, 10);
    auto [gpu_counts, gpu_edges] = histogram(input.to(device), 10);

    EXPECT_EQ(cpu_counts.numel(), gpu_counts.numel());
    EXPECT_EQ(cpu_edges.numel(), gpu_edges.numel());
    EXPECT_TRUE(close(cpu_edges, gpu_edges.to(Device::cpu()), 1e-5));
    // Bin counts are the substance of a histogram — compare them exactly. A
    // backend mis-binning (or returning zero counts) with correct edges would
    // have passed the edges-only check; it fails here.
    EXPECT_TRUE(equal_int(cpu_counts, gpu_counts))
        << "GPU histogram bin counts diverge from CPU reference";
}

// ============================================================================
// AdvancedIndex / AdvancedIndexPut parity (Phase 4.4)
// ============================================================================

namespace {
inline Tensor int64_tensor_1d(const std::vector<int64_t>& data) {
    auto t = Tensor::from_blob(const_cast<int64_t*>(data.data()),
                               {static_cast<int64_t>(data.size())},
                               DType::Int64).clone();
    return t;
}
inline Tensor float_tensor_1d(const std::vector<float>& data) {
    auto t = Tensor::from_blob(const_cast<float*>(data.data()),
                               {static_cast<int64_t>(data.size())},
                               DType::Float32).clone();
    return t;
}
inline std::vector<std::optional<Tensor>> make_idx_vec(std::initializer_list<Tensor> idxs) {
    std::vector<std::optional<Tensor>> v;
    v.reserve(idxs.size());
    for (const auto& t : idxs) v.emplace_back(t);
    return v;
}
}  // namespace

TEST_P(MissingOpsParity, AdvancedIndex_Simple1D) {
    auto src_cpu = arange(10.0f, 20.0f, 1.0f);
    auto idx_cpu = int64_tensor_1d({0, 3, 5, 9});

    auto cpu_ref = tenzor::index(src_cpu, make_idx_vec({idx_cpu}));
    auto gpu_result = tenzor::index(src_cpu.to(device), make_idx_vec({idx_cpu.to(device)}));

    EXPECT_TRUE(close(cpu_ref, gpu_result.to(Device::cpu()), 1e-6));
}

TEST_P(MissingOpsParity, AdvancedIndex_NegativeIndices) {
    auto src_cpu = arange(0.0f, 10.0f, 1.0f);
    auto idx_cpu = int64_tensor_1d({-1, -2, -3});

    auto cpu_ref = tenzor::index(src_cpu, make_idx_vec({idx_cpu}));
    auto gpu_result = tenzor::index(src_cpu.to(device), make_idx_vec({idx_cpu.to(device)}));

    EXPECT_TRUE(close(cpu_ref, gpu_result.to(Device::cpu()), 1e-6));
}

TEST_P(MissingOpsParity, AdvancedIndex_2D_RowSelect) {
    auto src_cpu = arange(0.0f, 12.0f, 1.0f).reshape({4, 3});
    auto idx_cpu = int64_tensor_1d({2, 0});

    auto cpu_ref = tenzor::index(src_cpu, make_idx_vec({idx_cpu}));
    auto gpu_result = tenzor::index(src_cpu.to(device), make_idx_vec({idx_cpu.to(device)}));

    ASSERT_EQ(cpu_ref.shape().size(), gpu_result.shape().size());
    EXPECT_EQ(cpu_ref.shape()[0], gpu_result.shape()[0]);
    EXPECT_EQ(cpu_ref.shape()[1], gpu_result.shape()[1]);
    EXPECT_TRUE(close(cpu_ref, gpu_result.to(Device::cpu()), 1e-6));
}

TEST_P(MissingOpsParity, AdvancedIndex_2D_TwoIndices) {
    auto src_cpu = arange(0.0f, 12.0f, 1.0f).reshape({4, 3});
    auto idx0 = int64_tensor_1d({0, 1, 2});
    auto idx1 = int64_tensor_1d({2, 1, 0});

    auto cpu_ref = tenzor::index(src_cpu, make_idx_vec({idx0, idx1}));
    auto gpu_result = tenzor::index(src_cpu.to(device),
                                    make_idx_vec({idx0.to(device), idx1.to(device)}));

    EXPECT_TRUE(close(cpu_ref, gpu_result.to(Device::cpu()), 1e-6));
}

TEST_P(MissingOpsParity, AdvancedIndexPut_Simple) {
    auto dst_cpu = zeros({10}, DType::Float32, Device::cpu());
    auto idx_cpu = int64_tensor_1d({1, 4, 7});
    auto vals_cpu = float_tensor_1d({1.5f, 2.5f, 3.5f});

    auto cpu_dst = dst_cpu.clone();
    tenzor::index_put(cpu_dst, make_idx_vec({idx_cpu}), vals_cpu);

    auto gpu_dst = dst_cpu.to(device);
    tenzor::index_put(gpu_dst, make_idx_vec({idx_cpu.to(device)}), vals_cpu.to(device));

    EXPECT_TRUE(close(cpu_dst, gpu_dst.to(Device::cpu()), 1e-6));
}

INSTANTIATE_BACKEND_TESTS(MissingOpsParity);
