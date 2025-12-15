cmake_minimum_required(VERSION 3.25)

# ============================================================================
# C++ Examples
# ============================================================================

# Basic examples
add_executable(simple_example cpp/simple_example.cpp)
target_link_libraries(simple_example PRIVATE tenzor_core)

add_executable(mnist_example cpp/mnist_example.cpp)
target_link_libraries(mnist_example PRIVATE tenzor_core)

add_executable(custom_op_example cpp/custom_op_example.cpp)
target_link_libraries(custom_op_example PRIVATE tenzor_core)

add_executable(backend_example cpp/backend_example.cpp)
target_link_libraries(backend_example PRIVATE tenzor_core)

add_executable(serialization_example cpp/serialization_example.cpp)
target_link_libraries(serialization_example PRIVATE tenzor_core)

add_executable(tensorboard_example cpp/tensorboard_example.cpp)
target_link_libraries(tensorboard_example PRIVATE tenzor_core)

# NLP examples
add_executable(bert_example cpp/nlp/bert_example.cpp)
target_link_libraries(bert_example PRIVATE tenzor_core)

add_executable(gpt_text_generation cpp/nlp/gpt_text_generation.cpp)
target_link_libraries(gpt_text_generation PRIVATE tenzor_core)

# Computer Vision examples
add_executable(classic_models_example cpp/cv/classic_models_example.cpp)
target_link_libraries(classic_models_example PRIVATE tenzor_core)

# ============================================================================
# Tutorial Examples (Complete, production-ready examples)
# ============================================================================

add_executable(mnist_complete cpp/tutorials/mnist_complete.cpp)
target_link_libraries(mnist_complete PRIVATE tenzor_core)

add_executable(mnist_with_dataloader cpp/tutorials/mnist_with_dataloader.cpp)
target_link_libraries(mnist_with_dataloader PRIVATE tenzor_core)

add_executable(custom_training_loop cpp/tutorials/custom_training_loop.cpp)
target_link_libraries(custom_training_loop PRIVATE tenzor_core)

add_executable(mixed_precision_training cpp/tutorials/mixed_precision_training.cpp)
target_link_libraries(mixed_precision_training PRIVATE tenzor_core)

# ============================================================================
# Showcase Examples (Demonstrate library features at 3 abstraction levels)
# ============================================================================

add_subdirectory(cpp/showcase)
