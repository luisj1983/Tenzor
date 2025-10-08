#include "tenzor/core/device.hpp"
#include <stdexcept>

namespace tenzor {

auto Device::from_string(std::string_view str) -> Device {
    if (str == "cpu") {
        return Device::cpu();
    }

    if (str.starts_with("cuda:")) {
        int idx = std::stoi(std::string(str.substr(5)));
        return Device::cuda(idx);
    }

    if (str.starts_with("rocm:")) {
        int idx = std::stoi(std::string(str.substr(5)));
        return Device::rocm(idx);
    }

    if (str.starts_with("oneapi:")) {
        int idx = std::stoi(std::string(str.substr(7)));
        return Device::oneapi(idx);
    }

    throw std::invalid_argument("Invalid device string: " + std::string(str));
}

} // namespace tenzor
