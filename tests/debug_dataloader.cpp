#include "tenzor/tenzor.hpp"
#include "tenzor/data/dataset.hpp"
#include "tenzor/data/dataloader.hpp"
#include <iostream>

using namespace tenzor;
using namespace tenzor::data;

int main() {
    tenzor::initialize();

    std::cout << "\n=== Testing TensorDataset ===\n";

    // Create simple dataset
    std::vector<float> input_data(100);
    std::vector<float> target_data(100);

    for (size_t i = 0; i < 100; ++i) {
        input_data[i] = static_cast<float>(i);
        target_data[i] = static_cast<float>(i * 2);
    }

    std::cout << "Input data array (first 5): ";
    for (int i = 0; i < 5; ++i) std::cout << input_data[i] << " ";
    std::cout << "\n";
    std::cout << "Input data array (last 5): ";
    for (int i = 95; i < 100; ++i) std::cout << input_data[i] << " ";
    std::cout << "\n";

    auto inputs = from_data(input_data.data(), {100, 1});
    auto targets = from_data(target_data.data(), {100, 1});

    std::cout << "Created inputs tensor: shape = [" << inputs.shape()[0] << ", " << inputs.shape()[1] << "]\n";
    std::cout << "  Device: " << inputs.device().to_string() << "\n";
    std::cout << "  DType: " << (int)inputs.dtype() << "\n";
    std::cout << "  Numel: " << inputs.numel() << "\n";
    std::cout << "  Has impl: " << (inputs.impl() != nullptr) << "\n";
    if (inputs.impl()) {
        std::cout << "  Has storage: " << (inputs.storage() != nullptr) << "\n";
        if (inputs.storage()) {
            std::cout << "  Storage data ptr: " << inputs.storage()->data() << "\n";
            // Check actual data in storage
            const float* data_ptr = static_cast<const float*>(inputs.storage()->data());
            std::cout << "  First 5 values in storage: ";
            for (int i = 0; i < 5; ++i) std::cout << data_ptr[i] << " ";
            std::cout << "\n";
            std::cout << "  Last 5 values in storage: ";
            for (int i = 95; i < 100; ++i) std::cout << data_ptr[i] << " ";
            std::cout << "\n";
        }
    }
    std::cout << "Created targets tensor: shape = [" << targets.shape()[0] << ", " << targets.shape()[1] << "]\n";

    // Test direct access to tensor
    std::cout << "\nDirect tensor access:\n";
    std::cout << "  inputs[0] = " << inputs.slice(0, 0, 1).item<float>() << " (expected 0)\n";
    std::cout << "  inputs[99] = " << inputs.slice(0, 99, 100).item<float>() << " (expected 99)\n";
    std::cout << "  targets[0] = " << targets.slice(0, 0, 1).item<float>() << " (expected 0)\n";
    std::cout << "  targets[99] = " << targets.slice(0, 99, 100).item<float>() << " (expected 198)\n";

    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    std::cout << "\nDataset size: " << dataset->size() << "\n";

    // Test dataset access
    std::cout << "\nDataset get() method:\n";
    auto [input0, target0] = dataset->get(0);
    std::cout << "  dataset->get(0): input=" << input0.item<float>() << ", target=" << target0.item<float>() << "\n";
    std::cout << "  input0 shape: [";
    for (size_t i = 0; i < input0.ndim(); ++i) {
        std::cout << input0.shape()[i];
        if (i < input0.ndim() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    auto [input99, target99] = dataset->get(99);
    std::cout << "  dataset->get(99): input=" << input99.item<float>() << ", target=" << target99.item<float>() << "\n";

    // Test DataLoader
    std::cout << "\n=== Testing DataLoader ===\n";
    DataLoader loader(dataset, 10, false, 0);

    std::cout << "DataLoader batches: " << loader.size() << "\n\n";

    size_t batch_idx = 0;
    for (const auto& batch : loader) {
        std::cout << "Batch " << batch_idx << ":\n";
        std::cout << "  inputs shape: [" << batch.inputs.shape()[0] << ", " << batch.inputs.shape()[1] << "]\n";
        std::cout << "  targets shape: [" << batch.targets.shape()[0] << ", " << batch.targets.shape()[1] << "]\n";

        // Print first and last sample in batch
        float first_input = batch.inputs.slice(0, 0, 1).item<float>();
        float first_target = batch.targets.slice(0, 0, 1).item<float>();
        size_t last_idx = batch.inputs.shape()[0] - 1;
        float last_input = batch.inputs.slice(0, last_idx, last_idx + 1).item<float>();
        float last_target = batch.targets.slice(0, last_idx, last_idx + 1).item<float>();

        std::cout << "  First sample: input=" << first_input << ", target=" << first_target << "\n";
        std::cout << "  Last sample:  input=" << last_input << ", target=" << last_target << "\n";

        // Print all samples in last batch
        if (batch_idx == 9) {
            std::cout << "  All samples in last batch:\n";
            for (size_t i = 0; i < batch.inputs.shape()[0]; ++i) {
                float input_val = batch.inputs.slice(0, i, i + 1).item<float>();
                float target_val = batch.targets.slice(0, i, i + 1).item<float>();
                size_t global_idx = batch_idx * 10 + i;
                std::cout << "    [" << i << "] global=" << global_idx << ": input=" << input_val
                          << " (expected " << global_idx << "), target=" << target_val
                          << " (expected " << (global_idx * 2) << ")\n";
            }
        }

        batch_idx++;
    }

    return 0;
}
