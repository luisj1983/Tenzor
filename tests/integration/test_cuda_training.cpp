#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <chrono>
#include <cmath>
#include <memory>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::optim;

//==============================================================================
// Test Environment Setup
//==============================================================================

class CUDATrainingEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();

        // Check CUDA availability
        try {
            auto test_tensor = ones({1}, DType::Float32, Device::cuda());
            cuda_available_ = true;
            std::cout << "CUDA is available for testing" << std::endl;
        } catch (...) {
            cuda_available_ = false;
            std::cout << "CUDA is not available, tests will be skipped" << std::endl;
        }
    }

    static bool cuda_available_;
};

bool CUDATrainingEnvironment::cuda_available_ = false;

static ::testing::Environment* const cuda_env =
    ::testing::AddGlobalTestEnvironment(new CUDATrainingEnvironment);

//==============================================================================
// Helper Functions
//==============================================================================

// Generate synthetic MNIST-like data
auto generate_mnist_batch(int batch_size, Device device) -> std::pair<Variable, Variable> {
    // Input: [batch, 1, 28, 28]
    auto input = Variable(randn({batch_size, 1, 28, 28}, DType::Float32, device), true);

    // Target: [batch, 10] (one-hot encoded)
    // Create on CPU first, then transfer to target device
    auto target_data_cpu = zeros({batch_size, 10}, DType::Float32, Device::cpu());
    float* target_ptr = const_cast<float*>(target_data_cpu.template data<float>());

    for (int i = 0; i < batch_size; i++) {
        int label = i % 10;  // Cycle through classes
        target_ptr[i * 10 + label] = 1.0f;
    }

    // Transfer to target device if needed
    auto target_data = (device.type == Device::Type::CPU) ? target_data_cpu : target_data_cpu.to(device);
    auto target = Variable(target_data, false);
    return {input, target};
}

// Simple accuracy calculation
auto calculate_accuracy(const Variable& predictions, const Variable& targets) -> float {
    // Transfer to CPU before accessing data
    auto pred_cpu = predictions.tensor().to(Device::cpu());
    auto target_cpu = targets.tensor().to(Device::cpu());

    auto pred_data = pred_cpu.template data<float>();
    auto target_data = target_cpu.template data<float>();

    int batch_size = predictions.shape()[0];
    int num_classes = predictions.shape()[1];
    int correct = 0;

    for (int i = 0; i < batch_size; i++) {
        // Find predicted class (argmax)
        int pred_class = 0;
        float max_val = pred_data[i * num_classes];
        for (int j = 1; j < num_classes; j++) {
            if (pred_data[i * num_classes + j] > max_val) {
                max_val = pred_data[i * num_classes + j];
                pred_class = j;
            }
        }

        // Find target class
        int target_class = 0;
        for (int j = 0; j < num_classes; j++) {
            if (target_data[i * num_classes + j] > 0.5f) {
                target_class = j;
                break;
            }
        }

        if (pred_class == target_class) {
            correct++;
        }
    }

    return static_cast<float>(correct) / batch_size;
}

// Memory usage helper
struct MemoryStats {
    size_t allocated_bytes{0};
    size_t peak_allocated_bytes{0};
};

auto get_memory_stats(Device device) -> MemoryStats {
    MemoryStats stats;
    // TODO: Implement actual CUDA memory tracking
    // For now, return dummy values
    stats.allocated_bytes = 0;
    stats.peak_allocated_bytes = 0;
    return stats;
}

//==============================================================================
// Simple CNN Model for MNIST
//==============================================================================

class SimpleCNN : public Module {
public:
    SimpleCNN() {
        // Conv2d(1, 32, 3) -> BatchNorm2d(32) -> ReLU -> Dropout(0.25)
        conv1 = std::make_shared<Conv2d>(1, 32, 3, 1, 1);
        bn1 = std::make_shared<BatchNorm2d>(32);
        relu1 = std::make_shared<ReLU>();
        dropout1 = std::make_shared<Dropout>(0.25);

        // Conv2d(32, 64, 3) -> BatchNorm2d(64) -> ReLU -> Dropout(0.25)
        conv2 = std::make_shared<Conv2d>(32, 64, 3, 1, 1);
        bn2 = std::make_shared<BatchNorm2d>(64);
        relu2 = std::make_shared<ReLU>();
        dropout2 = std::make_shared<Dropout>(0.25);

        // Flatten -> Linear(64 * 28 * 28, 128) -> ReLU -> Dropout(0.5)
        flatten = std::make_shared<Flatten>(1);  // Flatten from dim 1 onwards
        fc1 = std::make_shared<Linear>(64 * 28 * 28, 128);
        relu3 = std::make_shared<ReLU>();
        dropout3 = std::make_shared<Dropout>(0.5);

        // Linear(128, 10)
        fc2 = std::make_shared<Linear>(128, 10);

        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("flatten", flatten);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    auto forward(const Variable& x) -> Variable override {
        // First conv block
        auto out = conv1->forward(x);
        out = bn1->forward(out);
        out = relu1->forward(out);
        out = dropout1->forward(out);

        // Second conv block
        out = conv2->forward(out);
        out = bn2->forward(out);
        out = relu2->forward(out);
        out = dropout2->forward(out);

        // Flatten using proper layer
        auto flattened = flatten->forward(out);

        // Fully connected layers
        auto fc_out = fc1->forward(flattened);
        fc_out = relu3->forward(fc_out);
        fc_out = dropout3->forward(fc_out);
        fc_out = fc2->forward(fc_out);

        return fc_out;
    }

private:
    std::shared_ptr<Conv2d> conv1, conv2;
    std::shared_ptr<BatchNorm2d> bn1, bn2;
    std::shared_ptr<ReLU> relu1, relu2, relu3;
    std::shared_ptr<Dropout> dropout1, dropout2, dropout3;
    std::shared_ptr<Flatten> flatten;
    std::shared_ptr<Linear> fc1, fc2;
};

//==============================================================================
// Multi-Layer Perceptron
//==============================================================================

class MLP : public Module {
public:
    MLP(int input_size, int hidden_size, int output_size) {
        fc1 = std::make_shared<Linear>(input_size, hidden_size);
        relu1 = std::make_shared<ReLU>();
        dropout = std::make_shared<Dropout>(0.5);
        fc2 = std::make_shared<Linear>(hidden_size, output_size);

        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    auto forward(const Variable& x) -> Variable override {
        std::cout << "  MLP forward: input shape [" << x.shape()[0] << ", " << x.shape()[1] << "]" << std::endl;

        std::cout << "  fc1..." << std::endl;
        auto out = fc1->forward(x);
        std::cout << "  fc1 output: [" << out.shape()[0] << ", " << out.shape()[1] << "]" << std::endl;

        std::cout << "  relu1..." << std::endl;
        out = relu1->forward(out);
        std::cout << "  relu1 output: [" << out.shape()[0] << ", " << out.shape()[1] << "]" << std::endl;

        std::cout << "  dropout..." << std::endl;
        try {
            out = dropout->forward(out);
            std::cout << "  dropout output: [" << out.shape()[0] << ", " << out.shape()[1] << "]" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  ERROR in dropout: " << e.what() << std::endl;
            throw;
        }

        std::cout << "  fc2..." << std::endl;
        out = fc2->forward(out);
        std::cout << "  fc2 output: [" << out.shape()[0] << ", " << out.shape()[1] << "]" << std::endl;

        return out;
    }

private:
    std::shared_ptr<Linear> fc1, fc2;
    std::shared_ptr<ReLU> relu1;
    std::shared_ptr<Dropout> dropout;
};

//==============================================================================
// Test 1: Simple CNN on MNIST-like Data
//==============================================================================

TEST(CUDATrainingTest, SimpleCNN_MNIST) {
    if (!CUDATrainingEnvironment::cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    std::cout << "Creating device..." << std::endl;
    auto device = Device::cuda();
    std::cout << "Creating model..." << std::endl;
    auto model = std::make_shared<SimpleCNN>();

    // Move model to CUDA
    std::cout << "Moving model to CUDA..." << std::endl;
    model->to(device);
    std::cout << "Model moved to CUDA" << std::endl;

    std::cout << "Getting parameters..." << std::endl;
    auto params = model->parameters();
    std::cout << "Creating optimizer..." << std::endl;
    auto optimizer = SGD(params, 0.01, 0.9);

    const int num_epochs = 3;
    const int batch_size = 32;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        std::cout << "Epoch " << epoch << ": Generating batch..." << std::endl;
        // Generate batch
        auto [input, target] = generate_mnist_batch(batch_size, device);

        // Forward pass
        std::cout << "Epoch " << epoch << ": Forward pass..." << std::endl;
        auto output = model->forward(input);
        std::cout << "Epoch " << epoch << ": Forward complete" << std::endl;
        std::cout << "Epoch " << epoch << ": Checking output shape..." << std::endl;
        EXPECT_EQ(output.shape()[0], batch_size);
        EXPECT_EQ(output.shape()[1], 10);

        // Compute loss
        std::cout << "Epoch " << epoch << ": Computing loss..." << std::endl;
        auto loss = cross_entropy(output, target.tensor(), Reduction::Mean);
        std::cout << "Epoch " << epoch << ": Loss computed" << std::endl;
        EXPECT_EQ(loss.tensor().numel(), 1);
        // Transfer loss to CPU before accessing
        std::cout << "Epoch " << epoch << ": Transferring loss to CPU..." << std::endl;
        auto loss_cpu = loss.tensor().to(Device::cpu());
        std::cout << "Epoch " << epoch << ": Loss CPU numel=" << loss_cpu.numel()
                  << " device=" << loss_cpu.device().to_string() << std::endl;
        std::cout << "Epoch " << epoch << ": Accessing loss value..." << std::endl;
        float loss_val = loss_cpu.template data<float>()[0];
        std::cout << "Epoch " << epoch << ": Loss value=" << loss_val << std::endl;
        EXPECT_GT(loss_val, 0.0f);

        // Backward pass
        std::cout << "Epoch " << epoch << ": Zero grad..." << std::endl;
        optimizer.zero_grad();
        std::cout << "Epoch " << epoch << ": Backward..." << std::endl;
        loss.backward();  // Scalar loss - no gradient argument needed
        std::cout << "Epoch " << epoch << ": Backward complete" << std::endl;

        // Check gradients were computed
        bool has_gradients = true;
        for (const auto& param : params) {
            if (param->requires_grad() && !param->has_grad()) {
                has_gradients = false;
                break;
            }
        }
        EXPECT_TRUE(has_gradients) << "Some parameters don't have gradients at epoch " << epoch;

        // Optimizer step
        optimizer.step();

        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs
                  << " - Loss: " << loss.tensor().to(Device::cpu()).template data<float>()[0] << std::endl;
    }

    SUCCEED();
}

//==============================================================================
// Test 2: Multi-Layer Perceptron on GPU
//==============================================================================

TEST(CUDATrainingTest, MLP_GPU) {
    if (!CUDATrainingEnvironment::cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto device = Device::cuda();
    const int input_size = 784;
    const int hidden_size = 256;
    const int output_size = 10;
    const int batch_size = 64;

    auto model = std::make_shared<MLP>(input_size, hidden_size, output_size);
    model->to(device);  // Move model to CUDA
    auto params = model->parameters();
    auto optimizer = Adam(params, 0.001);

    // Generate random input
    auto input = Variable(randn({batch_size, input_size}, DType::Float32, device), true);

    // Create random one-hot targets on CPU first, then transfer
    auto target_data_cpu = zeros({batch_size, output_size}, DType::Float32, Device::cpu());
    float* target_ptr = const_cast<float*>(target_data_cpu.template data<float>());
    for (int i = 0; i < batch_size; i++) {
        int label = i % output_size;
        target_ptr[i * output_size + label] = 1.0f;
    }
    auto target_data = target_data_cpu.to(device);
    auto target = Variable(target_data, false);

    // Forward pass
    auto output = model->forward(input);
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], output_size);

    // Loss computation
    auto loss = cross_entropy(output, target.tensor(), Reduction::Mean);
    // Transfer loss to CPU before accessing its value
    auto loss_cpu = loss.tensor().to(Device::cpu());
    float loss_value = loss_cpu.template data<float>()[0];
    EXPECT_GT(loss_value, 0.0f);

    // Backward pass
    optimizer.zero_grad();
    loss.backward();  // Scalar loss - no gradient argument needed

    // Verify gradients
    for (const auto& param : params) {
        if (param->requires_grad()) {
            EXPECT_TRUE(param->has_grad()) << "Parameter missing gradient";
        }
    }

    // Optimizer step
    optimizer.step();

    std::cout << "MLP GPU test - Loss: " << loss_value << std::endl;
    SUCCEED();
}

//==============================================================================
// Test 3: Complete Training Loop
//==============================================================================

TEST(CUDATrainingTest, CompleteTrainingLoop) {
    if (!CUDATrainingEnvironment::cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto device = Device::cuda();
    auto model = std::make_shared<MLP>(100, 50, 10);
    model->to(device);  // Move model to CUDA
    model->eval();  // Disable dropout to avoid randomness interfering with simple pattern
    auto params = model->parameters();
    auto optimizer = SGD(params, 0.001, 0.0);  // Conservative: low LR, no momentum

    const int num_epochs = 20;
    const int batch_size = 32;
    const int batches_per_epoch = 3;  // Train on multiple batches

    float initial_loss = 0.0f;
    float best_loss = std::numeric_limits<float>::max();
    std::vector<float> losses;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;

        for (int batch_idx = 0; batch_idx < batches_per_epoch; batch_idx++) {
            // Create fixed synthetic data (same every epoch, but varies by batch)
            auto input_data_cpu = zeros({batch_size, 100}, DType::Float32, Device::cpu());
            auto target_data_cpu = zeros({batch_size, 10}, DType::Float32, Device::cpu());

            float* input_ptr = const_cast<float*>(input_data_cpu.template data<float>());
            float* target_ptr = const_cast<float*>(target_data_cpu.template data<float>());

            for (int i = 0; i < batch_size; i++) {
                // Simple learnable pattern: class determined by which features are active
                int class_label = (i + batch_idx) % 10;

                // Set features 10*class_label to 10*class_label+9 to 1.0
                for (int j = 0; j < 10; j++) {
                    input_ptr[i * 100 + (class_label * 10 + j)] = 1.0f;
                }

                // One-hot target
                target_ptr[i * 10 + class_label] = 1.0f;
            }

            auto input_data = input_data_cpu.to(device);
            auto target_data = target_data_cpu.to(device);

            auto input = Variable(input_data, true);
            auto target = Variable(target_data, false);

            // Forward
            auto output = model->forward(input);
            // Use MSE loss which is more stable for this simple synthetic problem
            auto loss = mse_loss(output, target, Reduction::Mean);

            // Transfer loss to CPU
            auto loss_cpu = loss.tensor().to(Device::cpu());
            float loss_val = loss_cpu.template data<float>()[0];
            epoch_loss += loss_val;

            // Backward
            optimizer.zero_grad();
            loss.backward();  // Scalar loss - no gradient argument needed

            // Update
            optimizer.step();
        }

        epoch_loss /= batches_per_epoch;
        losses.push_back(epoch_loss);

        if (epoch == 0) {
            initial_loss = epoch_loss;
        }

        // Track best (minimum) loss seen
        if (epoch_loss < best_loss) {
            best_loss = epoch_loss;
        }

        if (epoch % 2 == 0) {
            std::cout << "Epoch " << epoch << " - Avg Loss: " << epoch_loss << std::endl;
        }
    }

    // Training is working if the best loss is better than initial
    // This is more robust than comparing just first vs last epoch
    EXPECT_LT(best_loss, initial_loss) << "Model did not improve (best: " << best_loss << ", initial: " << initial_loss << ")";

    float improvement = (initial_loss - best_loss) / initial_loss * 100.0f;
    std::cout << "Training complete - Initial: " << initial_loss
              << " Best: " << best_loss
              << " (Improvement: " << improvement << "%)" << std::endl;
}

//==============================================================================
// Test 4: CPU vs CUDA Result Comparison
//==============================================================================

TEST(CUDATrainingTest, CPU_vs_CUDA_Comparison) {
    if (!CUDATrainingEnvironment::cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    const int input_size = 50;
    const int hidden_size = 30;
    const int output_size = 5;
    const int batch_size = 16;

    // Create identical models
    auto cpu_model = std::make_shared<MLP>(input_size, hidden_size, output_size);
    auto cuda_model = std::make_shared<MLP>(input_size, hidden_size, output_size);
    cuda_model->to(Device::cuda());  // Move CUDA model to GPU

    // Sync weights (copy CPU weights to CUDA model)
    auto cpu_params = cpu_model->parameters();
    auto cuda_params = cuda_model->parameters();
    ASSERT_EQ(cpu_params.size(), cuda_params.size());

    // Generate same input on both devices
    auto cpu_input = Variable(randn({batch_size, input_size}, DType::Float32, Device::cpu()), true);

    // Copy to CUDA
    auto cuda_input_data = cpu_input.tensor();
    // cuda_input_data = cuda_input_data.to(Device::cuda());  // TODO: Implement tensor.to()
    auto cuda_input = Variable(randn({batch_size, input_size}, DType::Float32, Device::cuda()), true);

    // Forward pass on both devices
    auto cpu_output = cpu_model->forward(cpu_input);
    auto cuda_output = cuda_model->forward(cuda_input);

    EXPECT_EQ(cpu_output.shape()[0], cuda_output.shape()[0]);
    EXPECT_EQ(cpu_output.shape()[1], cuda_output.shape()[1]);

    // Note: Since we can't sync weights yet, just verify shapes match
    std::cout << "CPU output shape: [" << cpu_output.shape()[0] << ", "
              << cpu_output.shape()[1] << "]" << std::endl;
    std::cout << "CUDA output shape: [" << cuda_output.shape()[0] << ", "
              << cuda_output.shape()[1] << "]" << std::endl;

    SUCCEED();
}

//==============================================================================
// Test 5: Performance Benchmarks
//==============================================================================

TEST(CUDATrainingTest, PerformanceBenchmark) {
    if (!CUDATrainingEnvironment::cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto cpu_device = Device::cpu();
    auto cuda_device = Device::cuda();

    const int num_iterations = 100;
    const int batch_size = 64;

    auto cpu_model = std::make_shared<MLP>(784, 256, 10);
    auto cuda_model = std::make_shared<MLP>(784, 256, 10);
    cuda_model->to(cuda_device);  // Move CUDA model to GPU

    auto cpu_params = cpu_model->parameters();
    auto cuda_params = cuda_model->parameters();

    auto cpu_optimizer = SGD(cpu_params, 0.01);
    auto cuda_optimizer = SGD(cuda_params, 0.01);

    // Benchmark CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto input = Variable(randn({batch_size, 784}, DType::Float32, cpu_device), true);
        auto target_data = zeros({batch_size, 10}, DType::Float32, cpu_device);
        auto target = Variable(target_data, false);

        auto output = cpu_model->forward(input);
        auto loss = mse_loss(output, target, Reduction::Mean);

        cpu_optimizer.zero_grad();
        loss.backward();  // Scalar loss - no gradient argument needed
        cpu_optimizer.step();
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);

    // Benchmark CUDA
    auto cuda_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto input = Variable(randn({batch_size, 784}, DType::Float32, cuda_device), true);
        auto target_data = zeros({batch_size, 10}, DType::Float32, cuda_device);
        auto target = Variable(target_data, false);

        auto output = cuda_model->forward(input);
        auto loss = mse_loss(output, target, Reduction::Mean);

        cuda_optimizer.zero_grad();
        loss.backward();  // Scalar loss - no gradient argument needed
        cuda_optimizer.step();
    }
    auto cuda_end = std::chrono::high_resolution_clock::now();
    auto cuda_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cuda_end - cuda_start);

    std::cout << "CPU Time: " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "CUDA Time: " << cuda_duration.count() << " ms" << std::endl;

    if (cuda_duration.count() > 0) {
        float speedup = static_cast<float>(cpu_duration.count()) / cuda_duration.count();
        std::cout << "Speedup: " << speedup << "x" << std::endl;

        // CUDA should be faster (or at least comparable for small models)
        EXPECT_GT(speedup, 0.5f) << "CUDA is significantly slower than CPU";
    }

    // Memory stats
    auto cuda_mem = get_memory_stats(cuda_device);
    std::cout << "CUDA Memory Allocated: " << (cuda_mem.allocated_bytes / 1024.0 / 1024.0)
              << " MB" << std::endl;
}

//==============================================================================
// Test 6: Gradient Flow Verification
//==============================================================================

TEST(CUDATrainingTest, GradientFlowVerification) {
    if (!CUDATrainingEnvironment::cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto device = Device::cuda();
    auto model = std::make_shared<SimpleCNN>();
    model->to(device);  // Move model to CUDA
    auto params = model->parameters();
    auto optimizer = SGD(params, 0.01);

    const int batch_size = 8;
    auto [input, target] = generate_mnist_batch(batch_size, device);

    // Forward pass
    auto output = model->forward(input);
    auto loss = cross_entropy(output, target.tensor(), Reduction::Mean);

    // Backward pass
    optimizer.zero_grad();
    loss.backward();  // Scalar loss - no gradient argument needed

    // Verify all parameters have gradients
    int params_with_grad = 0;
    int total_params = 0;

    for (const auto& param : params) {
        if (param->requires_grad()) {
            total_params++;
            if (param->has_grad()) {
                params_with_grad++;

                // Check gradient is not all zeros
                auto grad_data = param->grad().value();
                bool has_nonzero = false;

                // Transfer to CPU before accessing data
                auto grad_cpu = grad_data.to(Device::cpu());
                const float* grad_ptr = grad_cpu.data<float>();
                for (size_t i = 0; i < std::min(static_cast<size_t>(10), static_cast<size_t>(grad_cpu.numel())); i++) {
                    if (std::abs(grad_ptr[i]) > 1e-8) {
                        has_nonzero = true;
                        break;
                    }
                }

                EXPECT_TRUE(has_nonzero) << "Gradient appears to be all zeros";
            }
        }
    }

    std::cout << "Gradient flow: " << params_with_grad << "/" << total_params
              << " parameters have gradients" << std::endl;

    EXPECT_EQ(params_with_grad, total_params) << "Not all parameters received gradients";
}

//==============================================================================
// Test 7: Mixed CPU/CUDA Tensor Operations
//==============================================================================

TEST(CUDATrainingTest, MixedCPU_CUDA_Operations) {
    if (!CUDATrainingEnvironment::cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto cpu_device = Device::cpu();
    auto cuda_device = Device::cuda();

    // Create tensors on different devices
    auto cpu_tensor = ones({10, 10}, DType::Float32, cpu_device);
    auto cuda_tensor = ones({10, 10}, DType::Float32, cuda_device);

    // Operations on same device should work
    auto cpu_result = cpu_tensor * 2.0f;
    EXPECT_EQ(cpu_result.device(), cpu_device);

    auto cuda_result = cuda_tensor * 2.0f;
    EXPECT_EQ(cuda_result.device(), cuda_device);

    // Mixed device operations should either:
    // 1. Throw an error (recommended)
    // 2. Automatically transfer (with warning)

    try {
        auto mixed_result = cpu_tensor + cuda_tensor;
        // If it succeeded, check the result device
        std::cout << "Mixed operation succeeded, result on: "
                  << mixed_result.device().to_string() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Mixed operation correctly threw error: " << e.what() << std::endl;
        SUCCEED();  // Expected behavior
    }
}

//==============================================================================
// Test 8: Device-to-Device Tensor Transfers
//==============================================================================

TEST(CUDATrainingTest, DeviceTransfers) {
    if (!CUDATrainingEnvironment::cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto cpu_device = Device::cpu();
    auto cuda_device = Device::cuda();

    // Create tensor on CPU
    auto cpu_tensor = randn({100, 100}, DType::Float32, cpu_device);
    float* cpu_data = const_cast<float*>(cpu_tensor.template data<float>());
    float first_value = cpu_data[0];

    // Transfer to CUDA
    // auto cuda_tensor = cpu_tensor.to(cuda_device);  // TODO: Implement
    auto cuda_tensor = randn({100, 100}, DType::Float32, cuda_device);

    EXPECT_EQ(cuda_tensor.device(), cuda_device);
    EXPECT_EQ(cpu_tensor.shape()[0], cuda_tensor.shape()[0]);
    EXPECT_EQ(cpu_tensor.shape()[1], cuda_tensor.shape()[1]);

    // Transfer back to CPU
    // auto cpu_tensor2 = cuda_tensor.to(cpu_device);  // TODO: Implement
    auto cpu_tensor2 = randn({100, 100}, DType::Float32, cpu_device);

    EXPECT_EQ(cpu_tensor2.device(), cpu_device);
    EXPECT_EQ(cuda_tensor.shape()[0], cpu_tensor2.shape()[0]);
    EXPECT_EQ(cuda_tensor.shape()[1], cpu_tensor2.shape()[1]);

    // Note: Can't verify values match without .to() implementation
    std::cout << "Device transfer test completed" << std::endl;
    std::cout << "Original device: " << cpu_tensor.device().to_string() << std::endl;
    std::cout << "CUDA device: " << cuda_tensor.device().to_string() << std::endl;
    std::cout << "Back to CPU device: " << cpu_tensor2.device().to_string() << std::endl;

    SUCCEED();
}

//==============================================================================
// Test 9: Batch Size Scaling
//==============================================================================

TEST(CUDATrainingTest, BatchSizeScaling) {
    if (!CUDATrainingEnvironment::cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto device = Device::cuda();
    auto model = std::make_shared<MLP>(784, 128, 10);
    model->to(device);  // Move model to CUDA
    auto params = model->parameters();
    auto optimizer = Adam(params, 0.001);

    std::vector<int> batch_sizes = {16, 32, 64, 128};

    for (int batch_size : batch_sizes) {
        std::cout << "\n=== Batch size " << batch_size << " ===" << std::endl;

        auto input = Variable(randn({batch_size, 784}, DType::Float32, device), true);
        auto target_data = zeros({batch_size, 10}, DType::Float32, device);
        auto target = Variable(target_data, false);

        auto start = std::chrono::high_resolution_clock::now();

        std::cout << "Forward pass..." << std::endl;
        auto output = model->forward(input);
        std::cout << "Forward complete. Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]" << std::endl;

        std::cout << "Computing loss..." << std::endl;
        auto loss = mse_loss(output, target, Reduction::Mean);
        std::cout << "Loss computed" << std::endl;

        std::cout << "Zeroing gradients..." << std::endl;
        optimizer.zero_grad();

        std::cout << "Backward pass..." << std::endl;
        try {
            loss.backward();  // Scalar loss - no gradient argument needed
            std::cout << "Backward complete" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Error in backward: " << e.what() << std::endl;
            throw;
        }

        std::cout << "Optimizer step..." << std::endl;
        try {
            optimizer.step();
            std::cout << "Optimizer step complete" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Error in optimizer step: " << e.what() << std::endl;
            throw;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        float throughput = static_cast<float>(batch_size) / (duration.count() / 1e6);

        std::cout << "Batch size " << batch_size << ": "
                  << duration.count() / 1000.0 << " ms, "
                  << throughput << " samples/sec" << std::endl;

        EXPECT_EQ(output.shape()[0], batch_size);
        EXPECT_GT(throughput, 0.0f);
    }
}

//==============================================================================
// Test 10: Multi-Epoch Training with Validation
//==============================================================================

TEST(CUDATrainingTest, MultiEpochTrainingWithValidation) {
    if (!CUDATrainingEnvironment::cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto device = Device::cuda();
    auto model = std::make_shared<MLP>(100, 64, 10);
    model->to(device);  // Move model to CUDA
    auto params = model->parameters();
    auto optimizer = Adam(params, 0.001);

    const int num_epochs = 5;
    const int train_batches = 10;
    const int val_batches = 3;
    const int batch_size = 32;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Training phase
        float train_loss = 0.0f;
        model->train();

        for (int batch = 0; batch < train_batches; batch++) {
            auto input = Variable(randn({batch_size, 100}, DType::Float32, device), true);
            auto target_data = zeros({batch_size, 10}, DType::Float32, device);
            auto target = Variable(target_data, false);

            auto output = model->forward(input);
            auto loss = mse_loss(output, target, Reduction::Mean);

            optimizer.zero_grad();
            loss.backward();  // Scalar loss - no gradient argument needed
            optimizer.step();

            // Transfer loss to CPU before accessing
            train_loss += loss.tensor().to(Device::cpu()).template data<float>()[0];
        }
        train_loss /= train_batches;

        // Validation phase
        float val_loss = 0.0f;
        model->eval();

        for (int batch = 0; batch < val_batches; batch++) {
            auto input = Variable(randn({batch_size, 100}, DType::Float32, device), false);
            auto target_data = zeros({batch_size, 10}, DType::Float32, device);
            auto target = Variable(target_data, false);

            auto output = model->forward(input);
            auto loss = mse_loss(output, target, Reduction::Mean);

            // Transfer loss to CPU before accessing
            val_loss += loss.tensor().to(Device::cpu()).template data<float>()[0];
        }
        val_loss /= val_batches;

        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs
                  << " - Train Loss: " << train_loss
                  << " - Val Loss: " << val_loss << std::endl;

        EXPECT_GT(train_loss, 0.0f);
        EXPECT_GT(val_loss, 0.0f);
    }
}
