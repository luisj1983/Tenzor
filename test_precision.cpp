#include <iostream>
#include <string>

int main() {
    float val = 1e-7f;
    std::string str = std::to_string(val);
    std::cout << "Original: " << val << std::endl;
    std::cout << "std::to_string: '" << str << "'" << std::endl;
    std::cout << "std::stof: " << std::stof(str) << std::endl;
    return 0;
}
