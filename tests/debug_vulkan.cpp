#include <tenzor/tenzor.hpp>
#include <tenzor/backend/loader.hpp>
#include <iostream>

using namespace tenzor;

int main() {
    std::cout << "=== Debug Vulkan Backend ===" << std::endl;

    tenzor::initialize();

    std::cout << "\n1. Checking available backends:" << std::endl;
    auto backends = tenzor::backend_registry().available_backends();
    for (const auto& name : backends) {
        std::cout << "  - " << name << std::endl;
    }

    std::cout << "\n2. Checking Vulkan backend by name:" << std::endl;
    auto* vulkan_by_name = tenzor::backend_registry().get_backend("vulkan");
    std::cout << "  backend_registry().get_backend(\"vulkan\") = "
              << (vulkan_by_name ? vulkan_by_name->name() : "nullptr") << std::endl;

    std::cout << "\n3. Checking Vulkan backend by Device::Type:" << std::endl;
    auto* vulkan_by_type = tenzor::backend_registry().get_backend(Device::Type::Vulkan);
    std::cout << "  backend_registry().get_backend(Device::Type::Vulkan) = "
              << (vulkan_by_type ? vulkan_by_type->name() : "nullptr") << std::endl;

    std::cout << "\n4. Creating Vulkan device:" << std::endl;
    try {
        Device vulkan_device = Device::vulkan(0);
        std::cout << "  Device created: " << vulkan_device.to_string() << std::endl;

        std::cout << "\n5. Trying to create tensor on Vulkan:" << std::endl;
        auto t = zeros({2, 2}, DType::Float32, vulkan_device);
        std::cout << "  SUCCESS: Tensor created with shape " << t.shape()[0] << "x" << t.shape()[1] << std::endl;

    } catch (const std::exception& e) {
        std::cout << "  EXCEPTION: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "  UNKNOWN EXCEPTION" << std::endl;
    }

    return 0;
}
