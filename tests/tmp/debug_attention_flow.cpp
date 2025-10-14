#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/autograd/ops.hpp>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Creating query variable (leaf, requires_grad=true)" << std::endl;
    Variable query(randn({2, 5, 128}), true);
    std::cout << "query.is_leaf() = " << query.is_leaf() << std::endl;
    std::cout << "query.requires_grad() = " << query.requires_grad() << std::endl;
    std::cout << "query.grad_fn() = " << (query.grad_fn() ? "exists" : "null") << std::endl;

    std::cout << "\nCreating Linear layer" << std::endl;
    Linear linear(128, 128, true);

    std::cout << "\nForward pass through Linear" << std::endl;
    Variable output = linear.forward(query);
    std::cout << "output.is_leaf() = " << output.is_leaf() << std::endl;
    std::cout << "output.requires_grad() = " << output.requires_grad() << std::endl;
    std::cout << "output.grad_fn() = " << (output.grad_fn() ? "exists" : "null") << std::endl;

    std::cout << "\nComputing mean" << std::endl;
    Variable loss = mean(output);
    std::cout << "loss.is_leaf() = " << loss.is_leaf() << std::endl;
    std::cout << "loss.requires_grad() = " << loss.requires_grad() << std::endl;
    std::cout << "loss.grad_fn() = " << (loss.grad_fn() ? "exists" : "null") << std::endl;

    std::cout << "\nCalling backward..." << std::endl;
    loss.backward();
    std::cout << "Backward completed!" << std::endl;

    std::cout << "\nChecking gradients:" << std::endl;
    std::cout << "query.has_grad() = " << query.has_grad() << std::endl;
    if (query.has_grad()) {
        std::cout << "query.grad() shape: [";
        for (size_t i = 0; i < query.grad().value().shape().size(); ++i) {
            std::cout << query.grad().value().shape()[i];
            if (i < query.grad().value().shape().size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Now test with using query 3 times (like attention does)
    std::cout << "\n\n=== Testing with query used 3 times ===" << std::endl;
    Variable query2(randn({2, 5, 128}), true);
    Linear l1(128, 128, true);
    Linear l2(128, 128, true);
    Linear l3(128, 128, true);

    std::cout << "Forward through 3 separate Linear layers" << std::endl;
    Variable q1 = l1.forward(query2);
    Variable q2 = l2.forward(query2);
    Variable q3 = l3.forward(query2);

    std::cout << "Summing outputs" << std::endl;
    Variable combined = q1 + q2 + q3;

    std::cout << "Computing mean" << std::endl;
    Variable loss2 = mean(combined);

    std::cout << "Calling backward..." << std::endl;
    loss2.backward();
    std::cout << "Backward completed!" << std::endl;

    std::cout << "\nChecking gradients:" << std::endl;
    std::cout << "query2.has_grad() = " << query2.has_grad() << std::endl;
    if (query2.has_grad()) {
        std::cout << "Success! query2 has gradient" << std::endl;
    } else {
        std::cout << "FAIL! query2 does not have gradient" << std::endl;
        return 1;
    }

    return 0;
}
