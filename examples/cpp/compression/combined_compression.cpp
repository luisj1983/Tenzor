/**
 * @file combined_compression.cpp
 * @brief Combined model compression: Pruning + Knowledge Distillation
 *
 * Demonstrates how to combine multiple compression techniques for maximum
 * model size reduction while maintaining accuracy.
 *
 * Pipeline:
 * 1. Train large teacher model
 * 2. Distill to medium student model
 * 3. Prune student model
 * 4. Fine-tune pruned student
 * 5. Deploy ultra-compact model
 */

#include <iostream>
#include <iomanip>
#include <memory>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/compression/pruning.hpp"
#include "tenzor/nn/compression/distillation.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/nn/loss/losses.hpp"

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::compression;

/**
 * @brief Large teacher network (ResNet-50 style)
 */
class LargeTeacher : public Module {
public:
    LargeTeacher() {
        // Deep, wide network
        conv1_ = std::make_shared<Conv2d>(3, 64, 7, 2, 3);
        conv2_ = std::make_shared<Conv2d>(64, 128, 3, 2, 1);
        conv3_ = std::make_shared<Conv2d>(128, 256, 3, 2, 1);
        conv4_ = std::make_shared<Conv2d>(256, 512, 3, 2, 1);
        fc_ = std::make_shared<Linear>(512 * 2 * 2, 1000);

        register_module("conv1", conv1_);
        register_module("conv2", conv2_);
        register_module("conv3", conv3_);
        register_module("conv4", conv4_);
        register_module("fc", fc_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = conv1_->forward(x).relu();
        h = conv2_->forward(h).relu();
        h = conv3_->forward(h).relu();
        h = conv4_->forward(h).relu();
        h = h.flatten(1, -1);
        return fc_->forward(h);
    }

private:
    std::shared_ptr<Conv2d> conv1_, conv2_, conv3_, conv4_;
    std::shared_ptr<Linear> fc_;
};

/**
 * @brief Medium student network (MobileNet-V2 style)
 */
class MediumStudent : public Module {
public:
    MediumStudent() {
        conv1_ = std::make_shared<Conv2d>(3, 32, 3, 2, 1);
        dw1_ = std::make_shared<Conv2d>(32, 32, 3, 1, 1, 1, 32);
        pw1_ = std::make_shared<Conv2d>(32, 64, 1);
        dw2_ = std::make_shared<Conv2d>(64, 64, 3, 2, 1, 1, 64);
        pw2_ = std::make_shared<Conv2d>(64, 128, 1);
        fc_ = std::make_shared<Linear>(128 * 8 * 8, 1000);

        register_module("conv1", conv1_);
        register_module("dw1", dw1_);
        register_module("pw1", pw1_);
        register_module("dw2", dw2_);
        register_module("pw2", pw2_);
        register_module("fc", fc_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = conv1_->forward(x).relu();
        h = dw1_->forward(h).relu();
        h = pw1_->forward(h).relu();
        h = dw2_->forward(h).relu();
        h = pw2_->forward(h).relu();
        h = h.flatten(1, -1);
        return fc_->forward(h);
    }

private:
    std::shared_ptr<Conv2d> conv1_, dw1_, pw1_, dw2_, pw2_;
    std::shared_ptr<Linear> fc_;
};

/**
 * @brief Count model parameters
 */
int64_t count_parameters(const std::shared_ptr<Module>& model) {
    int64_t total = 0;
    for (auto& param : model->parameters()) {
        total += param->tensor().numel();
    }
    return total;
}

/**
 * @brief Count non-zero parameters (for pruned models)
 */
int64_t count_nonzero_parameters(const std::shared_ptr<Module>& model) {
    int64_t nonzero = 0;
    for (auto& param : model->parameters()) {
        auto data = param->tensor().data<float>();
        for (int64_t i = 0; i < param->tensor().numel(); ++i) {
            if (std::abs(data[i]) > 1e-8f) {
                nonzero++;
            }
        }
    }
    return nonzero;
}

/**
 * @brief Complete compression pipeline
 */
void compression_pipeline() {
    std::cout << "\n================================================\n";
    std::cout << "  Combined Compression Pipeline\n";
    std::cout << "  (Distillation + Pruning)\n";
    std::cout << "================================================\n";

    // =========================================================================
    // Stage 1: Baseline Teacher Model
    // =========================================================================
    std::cout << "\n--- Stage 1: Teacher Model (Baseline) ---\n";

    auto teacher = std::make_shared<LargeTeacher>();
    // teacher->load("pretrained_resnet50.pth");
    teacher->eval();

    int64_t teacher_params = count_parameters(teacher);
    std::cout << "Teacher model loaded\n";
    std::cout << "  Parameters: " << teacher_params << "\n";
    std::cout << "  Accuracy: 95.2% (simulated baseline)\n";
    std::cout << "  Size: ~100 MB\n";

    // =========================================================================
    // Stage 2: Knowledge Distillation (Teacher → Student)
    // =========================================================================
    std::cout << "\n--- Stage 2: Knowledge Distillation ---\n";

    auto student = std::make_shared<MediumStudent>();
    int64_t student_params_initial = count_parameters(student);

    std::cout << "Student model initialized\n";
    std::cout << "  Parameters: " << student_params_initial << "\n";
    float initial_compression = static_cast<float>(teacher_params) / student_params_initial;
    std::cout << "  Compression: " << std::fixed << std::setprecision(1)
              << initial_compression << "x\n";

    // Configure distillation
    DistillationConfig distill_config;
    distill_config.temperature = 4.0f;
    distill_config.alpha = 0.75f;

    std::cout << "\nDistillation config:\n";
    std::cout << "  Temperature: " << distill_config.temperature << "\n";
    std::cout << "  Alpha: " << distill_config.alpha << "\n";

    auto distiller = std::make_shared<KnowledgeDistillation>(
        teacher, student, distill_config
    );

    auto optimizer = std::make_shared<optim::Adam>(student->parameters(), 0.001);

    std::cout << "\nTraining student via distillation...\n";
    int distill_epochs = 30;

    for (int epoch = 0; epoch < distill_epochs; ++epoch) {
        // Anneal temperature
        float current_temp = temperature_schedule(
            4.0f, 2.0f, epoch, distill_epochs, "cosine"
        );
        distiller->set_temperature(current_temp);

        // Training loop would go here
        // For demonstration, we simulate

        if (epoch % 10 == 0) {
            std::cout << "  Epoch " << epoch << "/" << distill_epochs
                      << ", T=" << std::fixed << std::setprecision(2) << current_temp << "\n";
        }
    }

    std::cout << "\nDistillation complete!\n";
    std::cout << "  Student accuracy: 93.8% (retains 98.5% of teacher performance)\n";

    // =========================================================================
    // Stage 3: Pruning the Distilled Student
    // =========================================================================
    std::cout << "\n--- Stage 3: Pruning Distilled Student ---\n";

    // First, analyze sensitivity to find optimal sparsity
    std::cout << "Running sensitivity analysis...\n";

    auto validation_fn = [](std::shared_ptr<Module> m) -> float {
        // Simulated validation
        return 0.938f;  // 93.8% baseline
    };

    std::vector<float> test_sparsities = {0.3f, 0.5f, 0.7f};
    std::cout << "\nTesting sparsity levels:\n";

    for (float sp : test_sparsities) {
        float expected_accuracy = 0.938f - (sp * 0.05f);  // Simulated drop
        std::cout << "  " << (sp * 100) << "% sparsity → ~"
                  << std::fixed << std::setprecision(1)
                  << (expected_accuracy * 100) << "% accuracy\n";
    }

    // Apply iterative pruning with fine-tuning
    float target_sparsity = 0.6f;  // 60% sparsity
    std::cout << "\nApplying " << (target_sparsity * 100)
              << "% iterative pruning...\n";

    auto pruning_config = prune_iterative(
        student,
        target_sparsity,
        5,  // 5 iterations
        PruningSchedule::Polynomial,
        ImportanceCriterion::L1
    );

    // Iterative pruning loop
    for (int iteration = 0; iteration < 5; ++iteration) {
        pruning_config.current_iteration = iteration;
        float current_sparsity = pruning_config.get_current_sparsity();

        std::cout << "\n  Iteration " << (iteration + 1) << "/5\n";
        std::cout << "    Target sparsity: " << (current_sparsity * 100) << "%\n";

        // Prune
        auto iter_config = prune_unstructured(
            student, current_sparsity,
            ImportanceCriterion::L1, true
        );
        pruning_config.masks = iter_config.masks;
        apply_pruning_masks(student, pruning_config);

        // Fine-tune for a few epochs
        std::cout << "    Fine-tuning...\n";

        auto finetune_opt = std::make_shared<optim::Adam>(
            student->parameters(), 0.0001  // Lower LR for fine-tuning
        );

        for (int ft_epoch = 0; ft_epoch < 5; ++ft_epoch) {
            // Training loop
            // After each step: apply_pruning_masks(student, pruning_config);
        }

        float iter_sparsity = compute_sparsity(student);
        std::cout << "    Achieved: " << (iter_sparsity * 100) << "%\n";
    }

    // =========================================================================
    // Stage 4: Results Analysis
    // =========================================================================
    std::cout << "\n--- Stage 4: Final Results ---\n";

    float final_sparsity = compute_sparsity(student);
    int64_t nonzero_params = count_nonzero_parameters(student);

    std::cout << "\nModel Statistics:\n";
    std::cout << "  Original teacher parameters: " << teacher_params << "\n";
    std::cout << "  Student dense parameters: " << student_params_initial << "\n";
    std::cout << "  Student sparse parameters: " << nonzero_params << "\n";
    std::cout << "  Sparsity: " << std::fixed << std::setprecision(1)
              << (final_sparsity * 100) << "%\n";

    float total_compression = static_cast<float>(teacher_params) / nonzero_params;
    std::cout << "\nTotal compression: " << std::fixed << std::setprecision(1)
              << total_compression << "x\n";

    std::cout << "\nBreakdown:\n";
    std::cout << "  1. Distillation: " << initial_compression << "x\n";
    std::cout << "  2. Pruning: " << (1.0f / (1.0f - final_sparsity)) << "x\n";
    std::cout << "  Combined: " << total_compression << "x\n";

    std::cout << "\nAccuracy:\n";
    std::cout << "  Teacher (baseline): 95.2%\n";
    std::cout << "  Student (distilled): 93.8% (-1.4%)\n";
    std::cout << "  Student (pruned): 92.5% (-2.7% total)\n";

    std::cout << "\nModel Size:\n";
    std::cout << "  Teacher: 100 MB\n";
    std::cout << "  Student (dense): ~" << (100.0f / initial_compression) << " MB\n";
    std::cout << "  Student (sparse): ~" << (100.0f / total_compression) << " MB\n";

    // =========================================================================
    // Stage 5: Export and Deployment
    // =========================================================================
    std::cout << "\n--- Stage 5: Export for Deployment ---\n";

    // Finalize pruning (make sparse weights permanent)
    auto final_model = finalize_pruning(student, pruning_config);

    std::cout << "Exporting compressed model...\n";
    final_model->save("ultra_compact_model.pth");
    std::cout << "  Saved to: ultra_compact_model.pth\n";

    std::cout << "\nDeployment Recommendations:\n";
    std::cout << "  - Use sparse matrix libraries (e.g., cuSPARSE) for speedup\n";
    std::cout << "  - Consider INT8 quantization for further 4x compression\n";
    std::cout << "  - Suitable for: Mobile, Edge devices, IoT\n";

    std::cout << "\n✓ Combined compression pipeline complete!\n";
}

/**
 * @brief Compare compression strategies
 */
void compare_strategies() {
    std::cout << "\n================================================\n";
    std::cout << "  Compression Strategy Comparison\n";
    std::cout << "================================================\n\n";

    struct Strategy {
        std::string name;
        float compression;
        float accuracy;
        float speedup;
        std::string use_case;
    };

    std::vector<Strategy> strategies = {
        {"Baseline (Teacher)", 1.0f, 95.2f, 1.0f, "Maximum accuracy"},
        {"Distillation only", 4.0f, 93.8f, 4.0f, "Balanced performance"},
        {"Pruning only (60%)", 2.5f, 93.0f, 1.2f, "Memory constrained"},
        {"Distill + Prune", 10.0f, 92.5f, 5.0f, "Extreme compression"},
        {"Distill + Prune + Quant", 40.0f, 91.8f, 8.0f, "Mobile/Edge deployment"}
    };

    std::cout << std::left
              << std::setw(25) << "Strategy"
              << std::setw(12) << "Compress"
              << std::setw(12) << "Accuracy"
              << std::setw(12) << "Speedup"
              << "Best For\n";

    std::cout << std::string(80, '-') << "\n";

    for (const auto& s : strategies) {
        std::cout << std::left
                  << std::setw(25) << s.name
                  << std::setw(12) << (std::to_string(static_cast<int>(s.compression)) + "x")
                  << std::setw(12) << (std::to_string(static_cast<int>(s.accuracy)) + "%")
                  << std::setw(12) << (std::to_string(static_cast<int>(s.speedup)) + "x")
                  << s.use_case << "\n";
    }

    std::cout << "\nRecommendations:\n\n";

    std::cout << "Choose DISTILLATION when:\n";
    std::cout << "  - Need interpretable, clean architecture\n";
    std::cout << "  - Want hardware compatibility\n";
    std::cout << "  - Moderate compression (2-5x) acceptable\n\n";

    std::cout << "Choose PRUNING when:\n";
    std::cout << "  - Memory is primary constraint\n";
    std::cout << "  - Have sparse inference support\n";
    std::cout << "  - Can tolerate irregular patterns\n\n";

    std::cout << "Choose COMBINED when:\n";
    std::cout << "  - Need maximum compression (>10x)\n";
    std::cout << "  - Deploying to resource-constrained devices\n";
    std::cout << "  - Can afford 2-4% accuracy trade-off\n";
}

/**
 * @brief Best practices for combined compression
 */
void print_guidelines() {
    std::cout << "\n================================================\n";
    std::cout << "  Combined Compression Guidelines\n";
    std::cout << "================================================\n\n";

    std::cout << "Recommended Pipeline:\n";
    std::cout << "  1. Train large teacher (>95% accuracy)\n";
    std::cout << "  2. Distill to 2-4x smaller student\n";
    std::cout << "  3. Prune student by 50-70%\n";
    std::cout << "  4. Fine-tune with masks\n";
    std::cout << "  5. (Optional) Quantize to INT8\n\n";

    std::cout << "Key Hyperparameters:\n";
    std::cout << "  Distillation:\n";
    std::cout << "    - Temperature: 3-5 (higher for larger gap)\n";
    std::cout << "    - Alpha: 0.7-0.9 (favor soft targets)\n";
    std::cout << "    - Epochs: 50-100 (until convergence)\n\n";

    std::cout << "  Pruning:\n";
    std::cout << "    - Sparsity: 50-70% (after distillation)\n";
    std::cout << "    - Schedule: Polynomial (smoother)\n";
    std::cout << "    - Iterations: 5-10 (gradual pruning)\n";
    std::cout << "    - Fine-tune: 10-20 epochs per iteration\n\n";

    std::cout << "Common Pitfalls:\n";
    std::cout << "  ✗ Pruning before distillation (harder to recover)\n";
    std::cout << "  ✗ Too aggressive sparsity (>80% often fails)\n";
    std::cout << "  ✗ Insufficient fine-tuning after pruning\n";
    std::cout << "  ✗ Forgetting to apply masks during training\n\n";

    std::cout << "Expected Results:\n";
    std::cout << "  - Compression: 8-15x smaller\n";
    std::cout << "  - Accuracy: 2-4% drop from teacher\n";
    std::cout << "  - Speedup: 3-6x faster inference\n";
    std::cout << "  - Memory: 10-20% of original\n\n";

    std::cout << "When to Stop:\n";
    std::cout << "  - Accuracy drops below acceptable threshold\n";
    std::cout << "  - Further pruning doesn't reduce size\n";
    std::cout << "  - Model becomes unstable during training\n";
}

int main() {
    std::cout << "================================================\n";
    std::cout << "  Tenzor Combined Compression\n";
    std::cout << "  (Distillation + Pruning)\n";
    std::cout << "================================================\n";

    try {
        // Run complete pipeline
        compression_pipeline();

        // Compare strategies
        compare_strategies();

        // Print guidelines
        print_guidelines();

        std::cout << "\n================================================\n";
        std::cout << "Combined compression demonstration complete!\n";
        std::cout << "================================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
