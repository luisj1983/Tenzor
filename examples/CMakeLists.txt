cmake_minimum_required(VERSION 3.25)

# Simple example
add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example PRIVATE tenzor_core)

# MNIST example
add_executable(mnist_example mnist_example.cpp)
target_link_libraries(mnist_example PRIVATE tenzor_core)

# Custom operation example
add_executable(custom_op_example custom_op_example.cpp)
target_link_libraries(custom_op_example PRIVATE tenzor_core)

# Backend selection example
add_executable(backend_example backend_example.cpp)
target_link_libraries(backend_example PRIVATE tenzor_core)

# Serialization example
add_executable(serialization_example serialization_example.cpp)
target_link_libraries(serialization_example PRIVATE tenzor_core)
