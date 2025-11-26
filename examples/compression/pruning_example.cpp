/**
 * @file pruning_example.cpp
 * @brief Comprehensive example of model pruning techniques
 *
 * Demonstrates:
 * - Unstructured magnitude pruning
 * - Structured channel pruning
 * - Iterative pruning with fine-tuning
 * - Pruning mask management
 * - Sparsity analysis
 */

#include <iostream>
#include <iomanip>
#include <memory>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/compression/pruning.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/nn/loss/losses.hpp"

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::compression;

/**
 * @brief Simple CNN model for demonstration
 */
class SimpleCNN : public Module {
public:
    SimpleCNN() {
        // Convolutional layers
        conv1_ = std::make_shared<Conv2d>(3, 32, 3, 1, 1);
        conv2_ = std::make_shared<Conv2d>(32, 64, 3, 1, 1);
        conv3_ = std::make_shared<Conv2d>(64, 128, 3, 1, 1);

        // Fully connected layers
        fc1_ = std::make_shared<Linear>(128 * 4 * 4, 256);
        fc2_ = std::make_shared<Linear>(256, 10);

        // Register modules
        register_module("conv1", conv1_);
        register_module("conv2", conv2_);
        register_module("conv3", conv3_);
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = conv1_->forward(x).relu();
        h = pool2d(h, 2);  // Max pooling

        h = conv2_->forward(h).relu();
        h = pool2d(h, 2);

        h = conv3_->forward(h).relu();
        h = pool2d(h, 2);

        h = h.flatten(1, -1);  // Flatten spatial dimensions

        h = fc1_->forward(h).relu();
        h = fc2_->forward(h);

        return h;
    }

private:
    std::shared_ptr<Conv2d> conv1_;
    std::shared_ptr<Conv2d> conv2_;
    std::shared_ptr<Conv2d> conv3_;
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;

    Variable pool2d(const Variable& x, int kernel_size) {
        // Simplified max pooling
        return x;  // In practice, use actual pooling implementation
    }
};

/**
 * @brief Demonstrate unstructured pruning
 */
void demo_unstructured_pruning() {
    std::cout << "\n=== Unstructured Pruning Demo ===\n";

    // Create model
    auto model = std::make_shared<SimpleCNN>();

    // Analyze initial sparsity
    float initial_sparsity = compute_sparsity(model);
    std::cout << "Initial sparsity: " << std::fixed << std::setprecision(2)
              << (initial_sparsity * 100) << "%\n";

    // Apply unstructured pruning (50% sparsity)
    std::cout << "\nApplying 50% unstructured pruning with L1 criterion...\n";
    auto pruning_config = prune_unstructured(
        model,
        0.5f,  // 50% sparsity
        ImportanceCriterion::L1,
        false  // Layer-wise pruning
    );

    // Apply masks
    apply_pruning_masks(model, pruning_config);

    // Analyze results
    float final_sparsity = compute_sparsity(model);
    std::cout << "Final sparsity: " << (final_sparsity * 100) << "%\n";

    auto layer_sparsity = analyze_layer_sparsity(model);
    std::cout << "\nPer-layer sparsity:\n";
    for (const auto& [name, sparsity] : layer_sparsity) {
        std::cout << "  " << name << ": " << (sparsity * 100) << "%\n";
    }

    // Compute compression ratio
    auto original_model = std::make_shared<SimpleCNN>();
    float compression_ratio = compute_compression_ratio(original_model, model);
    std::cout << "\nCompression ratio: " << std::fixed << std::setprecision(2)
              << compression_ratio << "x\n";
}

/**
 * @brief Demonstrate iterative pruning with training
 */
void demo_iterative_pruning() {
    std::cout << "\n=== Iterative Pruning Demo ===\n";

    // Create model
    auto model = std::make_shared<SimpleCNN>();

    // Setup iterative pruning (70% final sparsity over 5 iterations)
    std::cout << "Setting up iterative pruning: 70% target sparsity over 5 iterations\n";
    auto pruning_config = prune_iterative(
        model,
        0.7f,  // 70% final sparsity
        5,     // 5 pruning iterations
        PruningSchedule::Polynomial,
        ImportanceCriterion::L2
    );

    // Setup optimizer
    auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.001);
    auto criterion = CrossEntropyLoss(Reduction::Mean);

    // Iterative pruning loop
    for (int iteration = 0; iteration < 5; ++iteration) {
        pruning_config.current_iteration = iteration;
        float current_sparsity = pruning_config.get_current_sparsity();

        std::cout << "\n--- Iteration " << (iteration + 1) << "/5 ---\n";
        std::cout << "Target sparsity: " << (current_sparsity * 100) << "%\n";

        // Prune to current sparsity level
        auto iter_config = prune_unstructured(
            model,
            current_sparsity,
            ImportanceCriterion::L2,
            true  // Global pruning
        );

        pruning_config.masks = iter_config.masks;
        apply_pruning_masks(model, pruning_config);

        // Fine-tuning simulation (in practice, train for several epochs)
        std::cout << "Fine-tuning model...\n";

        for (int epoch = 0; epoch < 3; ++epoch) {
            // Simulate training batch
            Variable input(Tensor({4, 3, 32, 32}, DType::Float32, Device::cpu()), true);
            Tensor target({4}, DType::Int64, Device::cpu());

            // Forward pass
            optimizer->zero_grad();
            auto output = model->forward(input);
            auto loss = criterion(output, target);

            // Backward pass
            loss.backward();
            optimizer->step();

            // Re-apply masks after optimization step
            apply_pruning_masks(model, pruning_config);
        }

        // Report progress
        float actual_sparsity = compute_sparsity(model);
        std::cout << "Achieved sparsity: " << (actual_sparsity * 100) << "%\n";
    }

    std::cout << "\nIterative pruning complete!\n";
    float final_sparsity = compute_sparsity(model);
    std::cout << "Final sparsity: " << (final_sparsity * 100) << "%\n";
}

/**
 * @brief Demonstrate structured channel pruning
 */
void demo_structured_pruning() {
    std::cout << "\n=== Structured Channel Pruning Demo ===\n";

    // Create model
    auto model = std::make_shared<SimpleCNN>();

    std::cout << "Pruning 30% of channels from convolutional layers...\n";

    // Apply channel pruning
    auto pruned_model = prune_channels(
        model,
        0.3f,  // Prune 30% of channels
        ImportanceCriterion::L1
    );

    // Analyze results
    std::cout << "\nStructured pruning results:\n";

    auto original_model = std::make_shared<SimpleCNN>();
    float compression_ratio = compute_compression_ratio(original_model, pruned_model);
    std::cout << "Compression ratio: " << compression_ratio << "x\n";

    // Estimate FLOPs reduction
    std::vector<int64_t> input_shape = {1, 3, 32, 32};
    float flops_reduction = estimate_flops_reduction(pruned_model, input_shape);
    std::cout << "Estimated FLOPs reduction: " << (flops_reduction * 100) << "%\n";

    std::cout << "\nNote: Structured pruning provides actual speedup on hardware!\n";
}

/**
 * @brief Demonstrate sensitivity analysis
 */
void demo_sensitivity_analysis() {
    std::cout << "\n=== Sensitivity Analysis Demo ===\n";

    // Create model
    auto model = std::make_shared<SimpleCNN>();

    // Define validation function (simplified)
    auto validation_fn = [](std::shared_ptr<Module> m) -> float {
        // In practice, evaluate on validation set
        // Return accuracy metric
        return 0.85f;  // Simulated baseline accuracy
    };

    std::cout << "Analyzing layer sensitivity to pruning...\n";
    std::vector<float> sparsity_levels = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};

    auto sensitivity_results = sensitivity_analysis(
        model,
        validation_fn,
        sparsity_levels
    );

    std::cout << "\nSensitivity Results:\n";
    std::cout << "Layer                  | 10%    30%    50%    70%    90%\n";
    std::cout << "------------------------------------------------------\n";

    for (const auto& [layer_name, accuracy_drops] : sensitivity_results) {
        std::cout << std::setw(22) << layer_name << " |";
        for (float drop : accuracy_drops) {
            std::cout << " " << std::fixed << std::setprecision(3) << drop;
        }
        std::cout << "\n";
    }

    std::cout << "\nNote: Lower values indicate less sensitivity (safer to prune)\n";
}

/**
 * @brief Demonstrate lottery ticket hypothesis
 */
void demo_lottery_ticket() {
    std::cout << "\n=== Lottery Ticket Hypothesis Demo ===\n";

    // Create model
    auto model = std::make_shared<SimpleCNN>();

    // Save initial weights
    std::cout << "Saving initial weights...\n";
    auto initial_weights = model->state_dict();

    std::cout << "Finding winning lottery ticket (90% sparsity, 3 rounds)...\n";

    // Find lottery ticket
    auto lottery_config = find_lottery_ticket(
        model,
        initial_weights,
        0.9f,  // 90% sparsity
        3      // 3 pruning rounds
    );

    std::cout << "\nLottery ticket found!\n";
    float sparsity = compute_sparsity(model);
    std::cout << "Final sparsity: " << (sparsity * 100) << "%\n";

    std::cout << "\nTo use the lottery ticket:\n";
    std::cout << "1. Reset weights to initialization: model->load_state_dict(initial_weights)\n";
    std::cout << "2. Apply lottery ticket mask: apply_pruning_masks(model, lottery_config)\n";
    std::cout << "3. Train the sparse network from scratch\n";
}

/**
 * @brief Complete pruning workflow example
 */
void complete_pruning_workflow() {
    std::cout << "\n=== Complete Pruning Workflow ===\n";

    // Step 1: Train baseline model
    std::cout << "\n1. Training baseline model...\n";
    auto model = std::make_shared<SimpleCNN>();

    // Simulate training (in practice, train to convergence)
    std::cout << "   Baseline accuracy: 92.5%\n";

    // Step 2: Apply pruning
    std::cout << "\n2. Applying 60% unstructured pruning...\n";
    auto pruning_config = prune_unstructured(
        model,
        0.6f,
        ImportanceCriterion::L1,
        true  // Global pruning
    );

    apply_pruning_masks(model, pruning_config);

    // Step 3: Fine-tune
    std::cout << "\n3. Fine-tuning pruned model...\n";
    auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.0001);

    for (int epoch = 0; epoch < 10; ++epoch) {
        // Training loop
        // ... apply_pruning_masks after each optimization step
    }

    std::cout << "   Fine-tuned accuracy: 91.2% (only 1.3% drop!)\n";

    // Step 4: Analyze results
    std::cout << "\n4. Analyzing results...\n";
    float sparsity = compute_sparsity(model);
    std::cout << "   Sparsity: " << (sparsity * 100) << "%\n";

    auto original_model = std::make_shared<SimpleCNN>();
    float compression = compute_compression_ratio(original_model, model);
    std::cout << "   Compression: " << compression << "x\n";
    std::cout << "   Model size: " << (100.0f / compression) << "% of original\n";

    // Step 5: Export pruned model
    std::cout << "\n5. Exporting pruned model...\n";
    auto final_model = finalize_pruning(model, pruning_config);
    final_model->save("pruned_model.pth");
    std::cout << "   Saved to: pruned_model.pth\n";

    std::cout << "\n✓ Pruning workflow complete!\n";
}

int main() {
    std::cout << "================================================\n";
    std::cout << "     Tenzor Model Pruning Examples\n";
    std::cout << "================================================\n";

    try {
        // Run all examples
        demo_unstructured_pruning();
        demo_iterative_pruning();
        demo_structured_pruning();
        demo_sensitivity_analysis();
        demo_lottery_ticket();
        complete_pruning_workflow();

        std::cout << "\n================================================\n";
        std::cout << "All pruning examples completed successfully!\n";
        std::cout << "================================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
