#include <tenzor/tenzor.hpp>
#include <tenzor/backend/loader.hpp>
#include <iostream>

int main() {
    tenzor::initialize();
    
    auto& loader = tenzor::backend_registry();
    auto backends = loader.available_backends();
    
    std::cout << "Available backends: " << backends.size() << std::endl;
    for (const auto& name : backends) {
        std::cout << "  - " << name << std::endl;
    }
    
    auto* cpu_backend = loader.get_backend(tenzor::Device::Type::CPU);
    std::cout << "CPU backend: " << (cpu_backend ? "FOUND" : "NOT FOUND") << std::endl;
    
    // Try to create a tensor and call transpose
    try {
        tenzor::Tensor t({4, 5}, tenzor::DType::Float32, tenzor::Device::cpu());
        t.fill_(1.0f);
        std::cout << "Tensor created and filled" << std::endl;
        
        auto t2 = t.transpose(0, 1);
        std::cout << "Transpose successful!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}
