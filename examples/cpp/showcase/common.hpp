/**
 * @file common.hpp
 * @brief Common utilities for showcase examples
 *
 * Provides backend selection utilities and helper functions used across
 * all showcase examples to demonstrate Tenzor's multi-backend capabilities.
 */

#pragma once

#include <tenzor/tenzor.hpp>
#include <iostream>
#include <string>
#include <cstring>
#include <stdexcept>

namespace showcase
{

    /**
     * @brief Parse backend type from command line argument
     *
     * Supported backends: cpu, cuda, vulkan, rocm, metal, webgpu
     *
     * @param backend_str Backend name string
     * @return Device configured for the specified backend
     */
    inline tenzor::Device parse_backend(const std::string &backend_str)
    {
        if(backend_str == "cpu")
        {
            return tenzor::Device::cpu();
        }
        else if(backend_str == "cuda")
        {
            return tenzor::Device::cuda(0);
        }
        else if(backend_str == "vulkan")
        {
            return tenzor::Device::vulkan(0);
        }
        else if(backend_str == "rocm")
        {
            return tenzor::Device::rocm(0);
        }
        else if(backend_str == "metal")
        {
            return tenzor::Device::metal(0);
        }
        else if(backend_str == "webgpu")
        {
            return tenzor::Device::webgpu(0);
        }
        else if(backend_str == "oneapi")
        {
            return tenzor::Device::oneapi(0);
        }
        else
        {
            throw std::runtime_error("Unknown backend: " + backend_str +
                "\nSupported: cpu, cuda, vulkan, rocm, metal, webgpu, oneapi") ;
        }
    }

    /**
     * @brief Parse command line arguments for backend selection
     *
     * Usage: ./example --backend cpu|cuda|vulkan
     * Default: cpu
     *
     * @param argc Argument count
     * @param argv Argument values
     * @return Device for the selected backend
     */
    inline tenzor::Device get_device_from_args(int argc, char *argv[])
    {
        std::string backend = "cpu";  // Default

        for(int i = 1; i < argc; ++i)
        {
            if(std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc)
            {
                backend = argv[i + 1];
                break;
            }
            else if(std::strcmp(argv[i], "-b") == 0 && i + 1 < argc)
            {
                backend = argv[i + 1];
                break;
            }
            else if(std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
            {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                    << "Options:\n"
                    << "  --backend, -b <backend>  Select compute backend (default: cpu)\n"
                    << "                           Supported: cpu, cuda, vulkan, rocm, metal, webgpu\n"
                    << "  --help, -h               Show this help message\n";
                std::exit(0);
            }
        }

        return parse_backend(backend);
    }

    /**
     * @brief Print header for example with backend info
     */
    inline void print_header(const std::string &title, const tenzor::Device &device)
    {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << title << "\n";
        std::cout << "Backend: " << device.to_string() << "\n";
        std::cout << std::string(60, '=') << "\n\n";
    }

    /**
     * @brief Print section divider
     */
    inline void print_section(const std::string &title)
    {
        std::cout << "\n--- " << title << " ---\n\n";
    }

    /**
     * @brief Print tensor info
     */
    inline void print_tensor_info(const std::string &name, const tenzor::Tensor &t)
    {
        std::cout << name << ": shape=(";
        auto shape = t.shape();
        for(size_t i = 0; i < shape.size(); ++i)
        {
            std::cout << shape[i];
            if(i < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "), device=" << t.device().to_string() << "\n";
    }

    /**
     * @brief Print training progress
     */
    inline void print_progress(int epoch, int total_epochs, float loss, float accuracy = -1.0f)
    {
        std::cout << "Epoch [" << (epoch + 1) << "/" << total_epochs << "] "
            << "Loss: " << loss;
        if(accuracy >= 0.0f)
        {
            std::cout << ", Accuracy: " << (accuracy * 100.0f) << "%";
        }
        std::cout << "\n";
    }

    /**
     * @brief Calculate accuracy for binary classification
     * Note: Tensors are moved to CPU for data access
     */
    inline float binary_accuracy(const tenzor::Tensor &predictions, const tenzor::Tensor &targets)
    {
        auto pred_cpu = predictions.cpu();
        auto target_cpu = targets.cpu();

        int correct = 0;
        int total = static_cast<int>(pred_cpu.shape()[0]);

        const float *pred_data = pred_cpu.data<float>();
        const float *target_data = target_cpu.data<float>();

        for(int i = 0; i < total; ++i)
        {
            float pred = (pred_data[i] > 0.5f) ? 1.0f : 0.0f;
            if(pred == target_data[i])
            {
                correct++;
            }
        }

        return static_cast<float>(correct) / static_cast<float>(total);
    }

    /**
     * @brief Calculate accuracy for multi-class classification
     * Note: Tensors are moved to CPU for data access
     */
    inline float multiclass_accuracy(const tenzor::Tensor &logits, const tenzor::Tensor &targets)
    {
        auto logits_cpu = logits.cpu();
        auto targets_cpu = targets.cpu();

        int batch_size = static_cast<int>(logits_cpu.shape()[0]);
        int num_classes = static_cast<int>(logits_cpu.shape()[1]);
        int correct = 0;

        const float *logit_data = logits_cpu.data<float>();
        const int64_t *target_data = targets_cpu.data<int64_t>();

        for(int b = 0; b < batch_size; ++b)
        {
            int predicted_class = 0;
            float max_val = logit_data[b * num_classes];
            for(int c = 1; c < num_classes; ++c)
            {
                if(logit_data[b * num_classes + c] > max_val)
                {
                    max_val = logit_data[b * num_classes + c];
                    predicted_class = c;
                }
            }
            if(predicted_class == target_data[b])
            {
                correct++;
            }
        }

        return static_cast<float>(correct) / static_cast<float>(batch_size);
    }

} // namespace showcase
