#include <iostream>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

using namespace tenzor;

int main() {
    tenzor::initialize();

    std::cout << "Testing scaled dot-product attention backward with Float16..." << std::endl;

    // Simulating what scaled_dot_product_attention does:
    // 1. Reshape Q,K,V from 4D to 3D
    // 2. Permute K
    // 3. BMM(Q, K^T)
    // 4. Reshape scores to 4D
    // 5. Scale scores
    // 6. Softmax
    // 7. Reshape attn_weights to 3D
    // 8. BMM(attn_weights, V)
    // 9. Reshape result to 4D

    // Create inputs: (batch=2, num_heads=4, seq_len=8, head_dim=16)
    auto query_tensor = randn({2, 4, 8, 16}, DType::Float16);
    auto key_tensor = randn({2, 4, 8, 16}, DType::Float16);
    auto value_tensor = randn({2, 4, 8, 16}, DType::Float16);

    Variable query(query_tensor, true);
    Variable key(key_tensor, true);
    Variable value(value_tensor, true);

    std::cout << "Created Q,K,V tensors: shape [2, 4, 8, 16]" << std::endl;

    // Test 3 iterations
    for (int i = 0; i < 3; ++i) {
        std::cout << "\n=== Iteration " << i << " ===" << std::endl;

        // Step 1: Reshape to 3D (batch*num_heads, seq_len, head_dim)
        std::cout << "Step 1: Reshaping Q,K,V to 3D..." << std::endl;
        auto query_3d = reshape(query, {2 * 4, 8, 16});
        auto key_3d = reshape(key, {2 * 4, 8, 16});
        auto value_3d = reshape(value, {2 * 4, 8, 16});

        // Step 2: Transpose key
        std::cout << "Step 2: Transposing key..." << std::endl;
        auto key_transposed = permute(key_3d, {0, 2, 1});

        // Step 3: BMM for QK^T
        std::cout << "Step 3: Computing QK^T..." << std::endl;
        auto scores = bmm(query_3d, key_transposed);

        // Step 4: Reshape scores to 4D
        std::cout << "Step 4: Reshaping scores to 4D..." << std::endl;
        scores = reshape(scores, {2, 4, 8, 8});

        // Step 5: Scale scores
        std::cout << "Step 5: Scaling scores..." << std::endl;
        double scale = 1.0 / std::sqrt(16.0);
        Tensor scale_tensor = full({1}, static_cast<float>(scale), DType::Float16, query.device());
        Variable scale_var(scale_tensor, false);
        scores = scores * scale_var;

        // Step 6: Softmax
        std::cout << "Step 6: Applying softmax..." << std::endl;
        auto attn_weights = softmax(scores, -1);

        // Step 7: Reshape attention weights to 3D
        std::cout << "Step 7: Reshaping attn_weights to 3D..." << std::endl;
        auto attn_weights_3d = reshape(attn_weights, {2 * 4, 8, 8});

        // Step 8: BMM with values
        std::cout << "Step 8: Computing attn_weights * V..." << std::endl;
        auto attended_3d = bmm(attn_weights_3d, value_3d);

        // Step 9: Reshape result to 4D
        std::cout << "Step 9: Reshaping result to 4D..." << std::endl;
        auto attended = reshape(attended_3d, {2, 4, 8, 16});

        // Sum to scalar for backward
        std::cout << "Computing loss (sum)..." << std::endl;
        Variable loss = tenzor::sum(attended);

        // Backward pass - THIS IS WHERE THE HANG MIGHT OCCUR
        std::cout << "Running backward pass..." << std::endl;
        loss.backward();
        std::cout << "Backward pass completed!" << std::endl;

        // Check gradients
        if (query.grad().has_value()) {
            std::cout << "Query gradient exists, dtype="
                      << (query.grad()->dtype() == DType::Float16 ? "Float16" : "other") << std::endl;
        }

        // Clear gradients
        query.zero_grad();
        key.zero_grad();
        value.zero_grad();
        std::cout << "Gradients cleared" << std::endl;
    }

    std::cout << "\nScaled dot-product attention backward test completed successfully!" << std::endl;
    return 0;
}
