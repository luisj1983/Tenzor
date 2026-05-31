/**
 * @file test_knowledge_distillation.cpp
 * @brief Tests for tenzor::nn::compression::KnowledgeDistillation.
 *
 * The audit (2026-05-02) found zero references to distillation.hpp in
 * the test suite. This file pins:
 *   - temperature_softmax() shape + sum-to-1 invariant + extreme-T limits.
 *   - distillation_loss() returns a finite scalar Variable.
 *   - KnowledgeDistillation::forward returns one Variable per model.
 *   - KnowledgeDistillation::compute_loss is finite, and its backward
 *     populates the student's parameter gradients but NOT the teacher's
 *     (the teacher is supposed to be frozen).
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/compression/distillation.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include "../backend_test_fixture.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::compression;

namespace {

// Tiny linear classifier — the smallest module that has parameters and
// produces logits that can be temperature-softmaxed.
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

class KnowledgeDistillationTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ---------------------------------------------------------------------------
// temperature_softmax
// ---------------------------------------------------------------------------

TEST_P(KnowledgeDistillationTest, TemperatureSoftmax_SumToOnePerRow) {
    Variable logits(randn({4, 10}, DType::Float32, device), false);
    Variable probs = temperature_softmax(logits, /*temperature=*/3.0f);
    EXPECT_EQ(probs.tensor().shape().size(), 2u);
    EXPECT_EQ(probs.tensor().shape()[0], 4);
    EXPECT_EQ(probs.tensor().shape()[1], 10);
    auto sum_each_row = sum(probs, /*dim=*/-1);
    auto cpu = sum_each_row.tensor().contiguous().to(Device::cpu());
    const float* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_NEAR(p[i], 1.0f, 1e-4f) << "row " << i << " softmax did not sum to 1";
    }
}

TEST_P(KnowledgeDistillationTest, TemperatureSoftmax_HighT_NearlyUniform) {
    Variable logits(randn({1, 8}, DType::Float32, device), false);
    Variable probs = temperature_softmax(logits, /*temperature=*/1e6f);
    auto cpu = probs.tensor().contiguous().to(Device::cpu());
    const float* p = cpu.data<float>();
    // High temperature → all entries ≈ 1/8.
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_NEAR(p[i], 0.125f, 1e-3f) << "high-T softmax not near uniform at " << i;
    }
}

// ---------------------------------------------------------------------------
// distillation_loss (free function)
// ---------------------------------------------------------------------------

TEST_P(KnowledgeDistillationTest, DistillationLoss_Finite) {
    auto student_t = randn({4, 5}, DType::Float32, device);
    auto teacher_t = randn({4, 5}, DType::Float32, device);
    Variable student(student_t, true);
    Variable teacher(teacher_t, false);
    auto target_keys = (rand({4}, DType::Float32, device) * 5).to(DType::Int64);

    DistillationConfig cfg;
    cfg.temperature = 3.0f;
    cfg.alpha = 0.7f;
    cfg.use_hard_targets = true;

    Variable loss = distillation_loss(student, teacher,
                                      std::optional<Tensor>{target_keys}, cfg);
    EXPECT_EQ(loss.tensor().numel(), 1);
    float v = loss.tensor().to(Device::cpu()).data<float>()[0];
    EXPECT_TRUE(std::isfinite(v)) << "distillation_loss produced non-finite value " << v;
}

// ---------------------------------------------------------------------------
// KnowledgeDistillation wrapper
// ---------------------------------------------------------------------------

TEST_P(KnowledgeDistillationTest, Forward_ReturnsBothOutputs) {
    auto teacher = std::make_shared<TinyClassifier>(8, 5);
    auto student = std::make_shared<TinyClassifier>(8, 5);
    teacher->to(device);
    student->to(device);
    KnowledgeDistillation distiller(teacher, student);
    Variable x(randn({2, 8}, DType::Float32, device), false);

    auto [s_out, t_out] = distiller.forward(x);
    EXPECT_EQ(s_out.tensor().shape().size(), 2u);
    EXPECT_EQ(s_out.tensor().shape()[0], 2);
    EXPECT_EQ(s_out.tensor().shape()[1], 5);
    EXPECT_EQ(t_out.tensor().shape().size(), 2u);
    EXPECT_EQ(t_out.tensor().shape()[0], 2);
    EXPECT_EQ(t_out.tensor().shape()[1], 5);
}

TEST_P(KnowledgeDistillationTest, ComputeLoss_GradientFlowsToStudentNotTeacher) {
    auto teacher = std::make_shared<TinyClassifier>(8, 5);
    auto student = std::make_shared<TinyClassifier>(8, 5);
    teacher->to(device);
    student->to(device);
    DistillationConfig cfg;
    cfg.alpha = 0.5f;
    cfg.use_hard_targets = true;
    KnowledgeDistillation distiller(teacher, student, cfg);

    Variable x(randn({4, 8}, DType::Float32, device), false);
    Tensor targets = (rand({4}, DType::Float32, device) * 5).to(DType::Int64);

    Variable loss = distiller.compute_loss(x, std::optional<Tensor>{targets});
    EXPECT_EQ(loss.tensor().numel(), 1);
    float v = loss.tensor().to(Device::cpu()).data<float>()[0];
    EXPECT_TRUE(std::isfinite(v)) << "compute_loss produced non-finite scalar";

    loss.backward();

    // Student weight grad should be present (non-empty + at least one
    // non-zero element) — the distillation signal must reach student
    // parameters or there's no learning.
    auto student_params = student->named_parameters();
    bool any_student_grad = false;
    for (auto& [name, p] : student_params) {
        if (p->grad().has_value()) {
            auto g = p->grad()->contiguous().to(Device::cpu());
            const float* gp = g.data<float>();
            for (int64_t i = 0; i < g.numel(); ++i) {
                if (std::fabs(gp[i]) > 0.0f) { any_student_grad = true; break; }
            }
        }
        if (any_student_grad) break;
    }
    EXPECT_TRUE(any_student_grad)
        << "Student parameters did not receive a gradient — distillation backward is severed.";

    // Teacher gradients should NOT be populated — the teacher is frozen
    // (the constructor sets requires_grad=false on its parameters).
    auto teacher_params = teacher->named_parameters();
    for (auto& [name, p] : teacher_params) {
        if (p->grad().has_value()) {
            auto g = p->grad()->contiguous().to(Device::cpu());
            const float* gp = g.data<float>();
            for (int64_t i = 0; i < g.numel(); ++i) {
                ASSERT_EQ(gp[i], 0.0f)
                    << "Teacher parameter '" << name
                    << "' received non-zero gradient — teacher should be frozen.";
            }
        }
    }
}

INSTANTIATE_BACKEND_TESTS(KnowledgeDistillationTest);

}  // namespace
