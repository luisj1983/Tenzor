#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/backend.hpp"
#include <iostream>

int main() {
    tenzor::init();
    
    auto cuda_device = tenzor::Device::cuda(0);
    
    // Test 1: full() with small value
    auto t1 = tenzor::full({4}, 1e-7f, tenzor::DType::Float32, cuda_device);
    auto t1_cpu = t1.to(tenzor::Device::cpu());
    
    std::cout << "Test 1 - full({4}, 1e-7f):" << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << "  [" << i << "] = " << t1_cpu.data<float>()[i] << std::endl;
    }
    
    // Test 2: full() with normal value
    auto t2 = tenzor::full({4}, 5.0f, tenzor::DType::Float32, cuda_device);
    auto t2_cpu = t2.to(tenzor::Device::cpu());
    
    std::cout << "\nTest 2 - full({4}, 5.0f):" << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << "  [" << i << "] = " << t2_cpu.data<float>()[i] << std::endl;
    }
    
    return 0;
}
