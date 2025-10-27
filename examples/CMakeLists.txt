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

# TensorBoard example
add_executable(tensorboard_example tensorboard_example.cpp)
target_link_libraries(tensorboard_example PRIVATE tenzor_core)

# BERT text classification example
add_executable(bert_example nlp/bert_example.cpp)
target_link_libraries(bert_example PRIVATE tenzor_core)

# Classic CNN models example
add_executable(classic_models_example cv/classic_models_example.cpp)
target_link_libraries(classic_models_example PRIVATE tenzor_core)

# GPT text generation example
add_executable(gpt_text_generation nlp/gpt_text_generation.cpp)
target_link_libraries(gpt_text_generation PRIVATE tenzor_core)

# ============================================================================
# Tutorial Examples (Complete, production-ready examples)
# ============================================================================

# Tutorial 1: Complete MNIST training with manual loop
add_executable(mnist_complete tutorials/mnist_complete.cpp)
target_link_libraries(mnist_complete PRIVATE tenzor_core)

# Tutorial 2: MNIST with high-level DataLoader and Callbacks
add_executable(mnist_with_dataloader tutorials/mnist_with_dataloader.cpp)
target_link_libraries(mnist_with_dataloader PRIVATE tenzor_core)

# Tutorial 3: Custom training loop with advanced features
add_executable(custom_training_loop tutorials/custom_training_loop.cpp)
target_link_libraries(custom_training_loop PRIVATE tenzor_core)

# Tutorial 4: Mixed precision training
add_executable(mixed_precision_training tutorials/mixed_precision_training.cpp)
target_link_libraries(mixed_precision_training PRIVATE tenzor_core)
