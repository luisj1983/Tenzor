/**
 * @file demo_split_operation.cpp
 * @brief Demonstration of the split() operation
 *
 * This example demonstrates various use cases of the split operation:
 * - Splitting tensors into fixed-size chunks
 * - Multi-head attention pattern
 * - Batch processing
 * - Working with views (zero-copy)
 */

#include <iostream>
#include <iomanip>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;

void print_tensor_info(const Tensor& t, const std::string& name) {
    std::cout << name << " - Shape: [";
    auto shape = t.shape();
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "], Elements: " << t.numel() << std::endl;
}

void demo_basic_split() {
    std::cout << "\n=== Basic Split Operation ===" << std::endl;

    // Create a 1D tensor [0, 1, 2, ..., 11]
    auto tensor = arange(0.0f, 12.0f, 1.0f, DType::Float32, Device::cpu());
    print_tensor_info(tensor, "Original");

    // Split into chunks of size 3
    auto chunks = split(tensor, 3, 0);
    std::cout << "Split into " << chunks.size() << " chunks of size 3:" << std::endl;

    for (size_t i = 0; i < chunks.size(); ++i) {
        std::cout << "  Chunk " << i << ": ";
        auto data = chunks[i].cpu().data<float>();
        std::cout << "[";
        for (int64_t j = 0; j < chunks[i].numel(); ++j) {
            std::cout << data[j];
            if (j < chunks[i].numel() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

void demo_2d_split() {
    std::cout << "\n=== 2D Tensor Split ===" << std::endl;

    // Create a 2D tensor (6x8)
    auto tensor = ones({6, 8}, DType::Float32, Device::cpu());
    print_tensor_info(tensor, "Original 2D tensor");

    // Split along dimension 0 (rows)
    auto row_chunks = split(tensor, 2, 0);
    std::cout << "\nSplit along rows (dim=0) with size 2:" << std::endl;
    std::cout << "  Number of chunks: " << row_chunks.size() << std::endl;
    for (size_t i = 0; i < row_chunks.size(); ++i) {
        print_tensor_info(row_chunks[i], "  Chunk " + std::to_string(i));
    }

    // Split along dimension 1 (columns)
    auto col_chunks = split(tensor, 3, 1);
    std::cout << "\nSplit along columns (dim=1) with size 3:" << std::endl;
    std::cout << "  Number of chunks: " << col_chunks.size() << std::endl;
    for (size_t i = 0; i < col_chunks.size(); ++i) {
        print_tensor_info(col_chunks[i], "  Chunk " + std::to_string(i));
    }
}

void demo_multi_head_attention() {
    std::cout << "\n=== Multi-Head Attention Use Case ===" << std::endl;

    // Simulate attention layer input
    // Shape: [batch_size, seq_len, hidden_dim]
    int64_t batch_size = 2;
    int64_t seq_len = 10;
    int64_t hidden_dim = 64;
    int64_t num_heads = 8;
    int64_t head_dim = hidden_dim / num_heads;

    auto tensor = ones({batch_size, seq_len, hidden_dim}, DType::Float32, Device::cpu());
    print_tensor_info(tensor, "Attention input");

    std::cout << "\nSplitting into " << num_heads << " attention heads:" << std::endl;
    std::cout << "  Hidden dim: " << hidden_dim << std::endl;
    std::cout << "  Head dim: " << head_dim << std::endl;

    // Split hidden dimension into multiple heads
    auto heads = split(tensor, head_dim, 2);  // Split along dimension 2

    std::cout << "\nResulting heads:" << std::endl;
    std::cout << "  Number of heads: " << heads.size() << std::endl;
    for (size_t i = 0; i < heads.size(); ++i) {
        print_tensor_info(heads[i], "  Head " + std::to_string(i));
    }

    std::cout << "\nEach head processes " << heads[0].numel()
              << " elements (" << batch_size << " x " << seq_len << " x " << head_dim << ")" << std::endl;
}

void demo_batch_processing() {
    std::cout << "\n=== Batch Processing Use Case ===" << std::endl;

    // Large batch that needs to be split into mini-batches
    int64_t total_samples = 100;
    int64_t features = 32;
    int64_t mini_batch_size = 10;

    auto data = ones({total_samples, features}, DType::Float32, Device::cpu());
    print_tensor_info(data, "Full dataset");

    std::cout << "\nSplitting into mini-batches of size " << mini_batch_size << ":" << std::endl;

    auto mini_batches = split(data, mini_batch_size, 0);

    std::cout << "  Number of mini-batches: " << mini_batches.size() << std::endl;
    for (size_t i = 0; i < std::min(size_t(3), mini_batches.size()); ++i) {
        print_tensor_info(mini_batches[i], "  Mini-batch " + std::to_string(i));
    }
    if (mini_batches.size() > 3) {
        std::cout << "  ... (" << (mini_batches.size() - 3) << " more mini-batches)" << std::endl;
    }
}

void demo_zero_copy_views() {
    std::cout << "\n=== Zero-Copy Views ===" << std::endl;

    // Create a tensor
    auto tensor = arange(0.0f, 10.0f, 1.0f, DType::Float32, Device::cpu());
    std::cout << "Original tensor: ";
    auto orig_data = tensor.data<float>();
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        std::cout << orig_data[i] << " ";
    }
    std::cout << std::endl;

    // Split creates views (no copying)
    auto chunks = split(tensor, 3, 0);
    std::cout << "\nSplit into chunks (views, not copies)" << std::endl;

    // Modify the first chunk
    std::cout << "\nModifying first chunk..." << std::endl;
    auto first_chunk_data = chunks[0].data<float>();
    first_chunk_data[0] = 999.0f;

    // Check if original tensor is modified
    std::cout << "Original tensor after modification: ";
    orig_data = tensor.data<float>();
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        std::cout << orig_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "\nNotice: Original tensor is modified because split() creates views!" << std::endl;
}

void demo_uneven_split() {
    std::cout << "\n=== Uneven Split ===" << std::endl;

    // When size doesn't divide evenly
    auto tensor = arange(0.0f, 10.0f, 1.0f, DType::Float32, Device::cpu());
    print_tensor_info(tensor, "Original (10 elements)");

    std::cout << "\nSplit into chunks of size 3:" << std::endl;
    auto chunks = split(tensor, 3, 0);

    std::cout << "  Number of chunks: " << chunks.size() << std::endl;
    for (size_t i = 0; i < chunks.size(); ++i) {
        auto data = chunks[i].cpu().data<float>();
        std::cout << "  Chunk " << i << " (size " << chunks[i].numel() << "): [";
        for (int64_t j = 0; j < chunks[i].numel(); ++j) {
            std::cout << data[j];
            if (j < chunks[i].numel() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "\nNote: Last chunk is smaller (1 element) as expected" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Tenzor Split Operation Demonstration" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        demo_basic_split();
        demo_2d_split();
        demo_multi_head_attention();
        demo_batch_processing();
        demo_zero_copy_views();
        demo_uneven_split();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  All demonstrations completed!" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
