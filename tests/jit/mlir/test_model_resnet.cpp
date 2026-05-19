// Phase 13 / Task E.4, E.5, E.6 — ResNet18 end-to-end through @tz.jit.
//
//   E.4 — eager forward sanity (shape + finite + non-zero).
//   E.5 — jit-compile via the MLIR backend on llvm-cpu and verify the
//         compiled forward matches the eager forward elementwise.
//   E.6 — same comparison on cuda / vulkan-spirv / rocm IREE targets
//         when the corresponding tenzor backend is present AND the
//         locally installed iree-compile has the matching HAL target.
//
// Uses `tenzor::models::resnet18()` (the project's built-in
// BasicBlock implementation) with the standard 1000-class head.
// Input shape `{1, 3, 32, 32}` keeps tests fast while still exercising
// every distinct op (conv2d / batchnorm2d / relu / maxpool / residual
// add / adaptive_avgpool / linear).

#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/models/resnet.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

auto backend_present(const std::string& name) -> bool {
    auto* be = ::tenzor::backend_registry().get_backend(name);
    if (be == nullptr) return false;
    try {
        return be->device_count() > 0;
    } catch (...) {
        return false;
    }
}

auto target_hw_present(const std::string& target) -> bool {
    if (target == "llvm-cpu")     return true;
    if (target == "cuda")         return backend_present("cuda");
    if (target == "rocm")         return backend_present("rocm");
    if (target == "vulkan-spirv") return backend_present("vulkan");
    return false;
}

auto device_for_target(const std::string& target) -> ::tenzor::Device {
    if (target == "cuda")         return ::tenzor::Device::cuda(0);
    if (target == "rocm")         return ::tenzor::Device::rocm(0);
    if (target == "vulkan-spirv") return ::tenzor::Device::vulkan(0);
    return ::tenzor::Device::cpu();
}

/// Probe whether the local iree-compile has a given HAL target backend
/// registered. Skips the test cleanly when the iree-dist isn't built
/// with support for cuda / rocm / etc.
auto iree_target_supported(const std::string& target) -> bool {
    static const std::set<std::string> registered = [] {
        std::set<std::string> out;
        FILE* pipe = ::popen(
            "iree-compile --iree-hal-list-target-backends 2>/dev/null",
            "r");
        if (pipe == nullptr) return out;
        char buf[256];
        while (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
            std::string line(buf);
            while (!line.empty() &&
                   (line.back() == '\n' || line.back() == '\r' ||
                    line.back() == ' ' || line.back() == '\t')) {
                line.pop_back();
            }
            std::size_t start = 0;
            while (start < line.size() &&
                   (line[start] == ' ' || line[start] == '\t')) {
                ++start;
            }
            line = line.substr(start);
            if (line.empty()) continue;
            if (line.find("Registered") != std::string::npos) continue;
            out.insert(line);
        }
        ::pclose(pipe);
        return out;
    }();
    return registered.count(target) > 0;
}

/// Build a ResNet18 in eval mode so BatchNorm uses running stats and the
/// forward is deterministic. We materialise the running stats by calling
/// `eval()` and resetting BN running mean/var to a sane (zero/one)
/// distribution; the constructed model already does this on
/// initialisation.
auto build_resnet18() -> std::shared_ptr<::tenzor::models::ResNet> {
    auto m = ::tenzor::models::resnet18(/*num_classes=*/1000,
                                        /*pretrained=*/false);
    m->eval();
    return m;
}

void run_jit_match(const std::string& target) {
    ensure_core_init();
    if (!target_hw_present(target)) {
        GTEST_SKIP() << "no hardware for target=" << target;
    }
    if (!iree_target_supported(target)) {
        GTEST_SKIP() << "iree-compile does not have HAL target backend "
                     << "'" << target << "' registered (rebuild iree-dist "
                     << "with -DIREE_HAL_DRIVER_" << target << "=ON)";
    }

    auto m = build_resnet18();
    const auto dev = device_for_target(target);
    if (dev.type != ::tenzor::Device::Type::CPU) {
        m->to(dev);
    }
    auto x_t = ::tenzor::randn({1, 3, 32, 32}, ::tenzor::DType::Float32,
                               dev);
    ::tenzor::Variable x(x_t, /*requires_grad=*/false);

    auto eager = m->forward(x);

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = target;
    auto fn = [&m](const ::tenzor::Variable& in) -> ::tenzor::Variable {
        return m->forward(in);
    };
    auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);
    auto jit_out  = compiled(x);

    const auto e_shape = eager.tensor().shape();
    const auto j_shape = jit_out.tensor().shape();
    ASSERT_EQ(std::vector<int64_t>(j_shape.begin(), j_shape.end()),
              std::vector<int64_t>(e_shape.begin(), e_shape.end()))
        << "shape mismatch on target=" << target;

    const auto e_f64 = eager.tensor().to(::tenzor::Device::cpu())
                           .to(::tenzor::DType::Float64);
    const auto j_f64 = jit_out.tensor().to(::tenzor::Device::cpu())
                           .to(::tenzor::DType::Float64);
    const auto diff  = ::tenzor::max(::tenzor::abs(e_f64 - j_f64))
                           .template item<double>();
    EXPECT_LT(diff, 1e-3)
        << "ResNet18 @tz.jit diverges from eager (target=" << target
        << ", max-abs-diff=" << diff << ")";
}

}  // namespace

TEST(ResNet18, EagerForwardShapeAndFiniteness) {
    ensure_core_init();
    auto m = build_resnet18();
    auto x = ::tenzor::Variable(
        ::tenzor::randn({1, 3, 32, 32}), /*requires_grad=*/false);
    auto logits = m->forward(x);
    const auto logits_shape = logits.tensor().shape();
    const std::vector<int64_t> got_shape(logits_shape.begin(),
                                         logits_shape.end());
    ASSERT_EQ(got_shape, (std::vector<int64_t>{1, 1000}));
    auto f = logits.tensor().to(::tenzor::DType::Float64);
    const auto maxabs =
        ::tenzor::max(::tenzor::abs(f)).template item<double>();
    EXPECT_LT(maxabs, 1e10)  << "logits not finite";
    EXPECT_GT(maxabs, 1e-10) << "logits collapsed to zero";
}

TEST(ResNet18, JitMatchesEager_LlvmCpu) {
    run_jit_match("llvm-cpu");
}

TEST(ResNet18, JitMatchesEager_Cuda) {
    run_jit_match("cuda");
}

TEST(ResNet18, JitMatchesEager_Vulkan) {
    run_jit_match("vulkan-spirv");
}

TEST(ResNet18, JitMatchesEager_Rocm) {
    run_jit_match("rocm");
}
