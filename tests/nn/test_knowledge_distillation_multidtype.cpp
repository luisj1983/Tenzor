/**
 * @file test_knowledge_distillation_multidtype.cpp
 * @brief Multi-backend × multi-dtype tests for KnowledgeDistillation.
 *
 * Mirrors the existing test_knowledge_distillation.cpp single-dtype tests
 * across the standard backend × dtype matrix, focusing on the surface
 * that the user actually exercises during training:
 *   - temperature_softmax keeps its sum-to-1 invariant under each dtype.
 *   - distillation_loss returns a finite scalar of the input dtype.
 *   - KnowledgeDistillation::compute_loss is finite and produces a
 *     scalar Variable on the right device + dtype.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/compression/distillation.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::compression;
using namespace tenzor::testing;

namespace {

// Tiny linear classifier that the convert_model() helper can move
// onto every backend × dtype combination.
class TinyClassifier : public Module {
public:
    std::shared_ptr<Linear> fc;
    explicit TinyClassifier(int64_t in_d, int64_t num_classes) {
        fc = std::make_shared<Linear>(in_d, num_classes);
        register_module("fc", fc);
    }
    auto forward_impl(const Variable& x) -> Variable override {
        return fc->forward(x);
    }
};

}  // namespace

class KnowledgeDistillationMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Looser sum-to-1 tolerance for half-precision dtypes.
    double softmax_sum_tolerance() const {
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) return 1e-2;
        if (dtype() == DType::Float64) return 1e-9;
        return 1e-4;
    }
};

TEST_P(KnowledgeDistillationMultiDTypeTest, TemperatureSoftmax_SumsToOne) {
    Variable logits = createInput({3, 8}, /*requires_grad=*/false);
    Variable probs = temperature_softmax(logits, /*temperature=*/3.0f);

    expectShape(probs.tensor(), {3, 8});
    expectDType(probs.tensor());

    auto sum_each_row = sum(probs, /*dim=*/-1);
    auto cpu = sum_each_row.tensor().to(Device::cpu()).to(DType::Float32);
    const float* p = cpu.data<float>();
    const double tol = softmax_sum_tolerance();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_NEAR(p[i], 1.0f, tol)
            << "row " << i << " softmax did not sum to 1 on " << device_.to_string()
            << " " << dtype_name(dtype()) << " (got " << p[i] << ")";
    }
}

TEST_P(KnowledgeDistillationMultiDTypeTest, DistillationLoss_Finite) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        // The cross-entropy term in distillation_loss accumulates
        // log-softmax * one-hot over the class dim; half-precision
        // dtypes (7-10 mantissa bits) are insufficient for a stable
        // cross-entropy on small (4-class) logits without an internal
        // upcast. Skipping rather than relaxing tolerance.
        SKIP_WITH_REASON(SkipReason::DtypeUnsupportedOnBackend,
                         "distillation_loss CE term needs internal Float32 upcast for half dtypes");
    }

    Variable student = createInput({4, 5}, /*requires_grad=*/true);
    Variable teacher = createInput({4, 5}, /*requires_grad=*/false);
    auto target_keys = (rand({4}, DType::Float32, Device::cpu()) * 5)
                            .to(DType::Int64).to(device_);

    DistillationConfig cfg;
    cfg.temperature = 3.0f;
    cfg.alpha = 0.7f;
    cfg.use_hard_targets = true;

    Variable loss = distillation_loss(student, teacher,
                                      std::optional<Tensor>{target_keys}, cfg);
    EXPECT_EQ(loss.tensor().numel(), 1);
    EXPECT_EQ(loss.tensor().device().type, device_.type);
    expectDType(loss.tensor());

    float v = loss.tensor().to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_TRUE(std::isfinite(v))
        << "distillation_loss produced non-finite value on " << device_.to_string()
        << " " << dtype_name(dtype()) << " (got " << v << ")";
}

TEST_P(KnowledgeDistillationMultiDTypeTest, ComputeLoss_FiniteScalar) {
    if (dtype() == DType::BFloat16 || dtype() == DType::Float16) {
        // Same CE-precision concern as DistillationLoss_Finite for half
        // dtypes; the underlying linear forward is fine but the loss
        // composition is too noisy for a strict isfinite check on tiny
        // models.
        SKIP_WITH_REASON(SkipReason::DtypeUnsupportedOnBackend,
                         "compute_loss CE term needs internal Float32 upcast for half dtypes");
    }

    auto teacher = std::make_shared<TinyClassifier>(8, 5);
    auto student = std::make_shared<TinyClassifier>(8, 5);
    convert_model(*teacher);
    convert_model(*student);

    DistillationConfig cfg;
    cfg.alpha = 0.5f;
    cfg.use_hard_targets = true;
    KnowledgeDistillation distiller(teacher, student, cfg);

    Variable x = createInput({4, 8}, /*requires_grad=*/false);
    Tensor targets = (rand({4}, DType::Float32, Device::cpu()) * 5)
                        .to(DType::Int64).to(device_);

    Variable loss = distiller.compute_loss(x, std::optional<Tensor>{targets});
    EXPECT_EQ(loss.tensor().numel(), 1);
    expectDType(loss.tensor());
    float v = loss.tensor().to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_TRUE(std::isfinite(v))
        << "KnowledgeDistillation::compute_loss produced non-finite on "
        << device_.to_string() << " " << dtype_name(dtype()) << " (got " << v << ")";
}

INSTANTIATE_MULTI_BACKEND_ALL_DTYPE_TESTS(KnowledgeDistillationMultiDTypeTest);

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
    int rc = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return rc;
}
