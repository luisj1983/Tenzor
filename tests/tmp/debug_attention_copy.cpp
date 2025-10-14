#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/autograd/ops.hpp>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Testing gradient flow through Variable copying (like attention does)\n\n";

    // Create linear layers
    Linear q_proj(128, 128, true);
    Linear k_proj(128, 128, true);
    Linear v_proj(128, 128, true);

    // Create input (the query variable in the test)
    Variable query(randn({2, 5, 128}), true);
    std::cout << "query address: " << &query << std::endl;
    std::cout << "query.requires_grad() = " << query.requires_grad() << std::endl;
    std::cout << "query.is_leaf() = " << query.is_leaf() << std::endl;

    // Mimic what attention.forward() does with the parameters
    std::cout << "\nCopying query to q, k, v (like attention does)...\n";
    Variable q = query;  // Copy like attention does
    Variable k = query;  // Same variable
    Variable v = query;  // Same variable

    std::cout << "q address: " << &q << std::endl;
    std::cout << "k address: " << &k << std::endl;
    std::cout << "v address: " << &v << std::endl;
    std::cout << "All point to same Variable? q==k: " << (&q.tensor() == &k.tensor()) << std::endl;

    // Project through layers (like attention does)
    std::cout << "\nProjecting through Linear layers...\n";
    Variable Q = q_proj.forward(q);
    Variable K = k_proj.forward(k);
    Variable V = v_proj.forward(v);

    // Combine and compute loss
    Variable combined = Q + K + V;
    Variable loss = mean(combined);

    std::cout << "Loss value: " << loss.tensor().data<float>()[0] << std::endl;

    // Backward
    std::cout << "\nCalling backward...\n";
    loss.backward();
    std::cout << "Backward completed!\n";

    // Check gradients
    std::cout << "\nChecking gradients:\n";
    std::cout << "query.has_grad() = " << query.has_grad() << std::endl;
    std::cout << "q.has_grad() = " << q.has_grad() << std::endl;
    std::cout << "k.has_grad() = " << k.has_grad() << std::endl;
    std::cout << "v.has_grad() = " << v.has_grad() << std::endl;

    if (!query.has_grad()) {
        std::cout << "\nFAIL! query does not have gradient\n";
        return 1;
    }

    std::cout << "\nSUCCESS! Gradients flow back through Variable copies\n";
    return 0;
}
