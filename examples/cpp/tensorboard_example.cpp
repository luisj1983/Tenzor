/**
 * @file tensorboard_example.cpp
 * @brief Example demonstrating TensorBoard integration
 *
 * Shows how to use SummaryWriter to log training metrics,
 * histograms, and images for visualization in TensorBoard.
 */

#include <tenzor/tenzor.hpp>
#include <tenzor/utils/tensorboard.hpp>
#include <iostream>
#include <cmath>
#include <random>

using namespace tenzor;

int main() {
    std::cout << "TensorBoard Integration Example\n";
    std::cout << "================================\n\n";

    try {
        // Create SummaryWriter
        SummaryWriter writer("runs/example_experiment");
        std::cout << "Created SummaryWriter in runs/example_experiment/\n";

        // Simulate training loop
        std::cout << "\nSimulating training loop...\n";

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (int step = 0; step < 100; ++step) {
            // Log scalar metrics
            float loss = 2.0f * std::exp(-step * 0.05f) + 0.1f;
            float accuracy = 1.0f - 0.5f * std::exp(-step * 0.03f);
            float lr = 0.001f * std::exp(-step * 0.01f);

            writer.add_scalar("train/loss", loss, step);
            writer.add_scalar("train/accuracy", accuracy, step);
            writer.add_scalar("train/learning_rate", lr, step);

            if (step % 10 == 0) {
                std::cout << "Step " << step
                         << " - Loss: " << loss
                         << ", Accuracy: " << accuracy << "\n";
            }

            // Log histogram every 20 steps
            if (step % 20 == 0) {
                // Create sample weight tensor
                Tensor weights({100, 50}, DType::Float32, Device::cpu());
                float* data = weights.data<float>();
                for (int i = 0; i < weights.numel(); ++i) {
                    data[i] = dist(gen) * std::sqrt(2.0f / 100.0f); // He initialization
                }

                writer.add_histogram("weights/layer1", weights, step);
                std::cout << "  Logged histogram at step " << step << "\n";
            }

            // Log image every 25 steps
            if (step % 25 == 0 && step > 0) {
                // Create sample image (grayscale checkerboard pattern)
                Tensor image({1, 32, 32}, DType::Float32, Device::cpu());
                float* img_data = image.data<float>();

                for (int y = 0; y < 32; ++y) {
                    for (int x = 0; x < 32; ++x) {
                        int idx = y * 32 + x;
                        // Checkerboard pattern
                        img_data[idx] = ((x / 4 + y / 4) % 2) ? 0.8f : 0.2f;
                        // Add some noise
                        img_data[idx] += dist(gen) * 0.1f;
                        img_data[idx] = std::clamp(img_data[idx], 0.0f, 1.0f);
                    }
                }

                writer.add_image("generated/sample", image, step);
                std::cout << "  Logged image at step " << step << "\n";
            }
        }

        // Log computation graph
        std::cout << "\nLogging computation graph...\n";
        writer.add_graph("SimpleModel", {1, 3, 224, 224});

        // Flush and close
        std::cout << "\nFlushing and closing writer...\n";
        writer.flush();
        writer.close();

        std::cout << "\n✓ Example completed successfully!\n";
        std::cout << "\nTo visualize in TensorBoard, run:\n";
        std::cout << "  tensorboard --logdir=runs/example_experiment\n";
        std::cout << "Then open http://localhost:6006 in your browser\n";

    } catch (const TenzorException& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
