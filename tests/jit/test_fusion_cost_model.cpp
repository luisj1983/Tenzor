/**
 * @file test_fusion_cost_model.cpp
 * @brief Tests for FusionCostModel's device-aware speedup heuristic (JIT-R026)
 */

#include <gtest/gtest.h>

#include <tenzor/jit/fusion_cost_model.hpp>

using namespace tenzor;
using namespace tenzor::jit;

namespace {
auto make_candidate(int64_t total_elements) -> FusionCandidate {
    FusionCandidate c;
    c.num_ops = 3;
    c.total_elements = total_elements;
    c.num_memory_accesses = 6;
    c.kind = FusionKind::ElementWise;
    c.bytes_per_element = 4;
    return c;
}
}  // namespace

// JIT-R026: is_gpu previously only recognized CUDA/ROCm, silently routing
// Vulkan/OneAPI through the CPU cache-locality heuristic instead of the GPU
// launch-overhead one. A small candidate (< 1024 elements) is the case where
// the two heuristics disagree: GPU declines (0.8, < 1.0), CPU always accepts
// (>= 1.1). Confirms Vulkan/OneAPI now get the GPU heuristic like CUDA/ROCm.
TEST(FusionCostModel, SmallCandidateDeclinesOnEveryGpuDeviceType) {
    for (auto dev : {Device::Type::CUDA, Device::Type::ROCm,
                     Device::Type::Vulkan, Device::Type::OneAPI}) {
        FusionCostModel model;
        model.set_device_type(dev);
        double speedup = model.estimate_speedup(make_candidate(512));
        EXPECT_LT(speedup, 1.0)
            << "device type " << static_cast<int>(dev)
            << " did not take the GPU (launch-overhead-dominated) heuristic "
               "for a small fusion candidate";
    }
}

// The CPU heuristic (default device type) never declines a small candidate --
// this is the documented, unchanged-by-this-fix baseline the GPU case above
// contrasts with.
TEST(FusionCostModel, SmallCandidateAlwaysAcceptsOnCpu) {
    FusionCostModel model;  // device_type_ defaults to CPU
    double speedup = model.estimate_speedup(make_candidate(512));
    EXPECT_GE(speedup, 1.0);
}

// Large candidates are profitable on every GPU device type too (the
// launch-overhead heuristic's upper branch), confirming the fix didn't
// accidentally make Vulkan/OneAPI always decline.
TEST(FusionCostModel, LargeCandidateAcceptsOnEveryGpuDeviceType) {
    for (auto dev : {Device::Type::CUDA, Device::Type::ROCm,
                     Device::Type::Vulkan, Device::Type::OneAPI}) {
        FusionCostModel model;
        model.set_device_type(dev);
        double speedup = model.estimate_speedup(make_candidate(1024 * 1024));
        EXPECT_GT(speedup, 1.0)
            << "device type " << static_cast<int>(dev)
            << " unexpectedly declined a large, clearly-profitable fusion "
               "candidate";
    }
}
