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
# Training Examples (Complete training pipelines covering all components)
# ============================================================================

add_executable(gru_time_series cpp/training/gru_time_series.cpp)
target_link_libraries(gru_time_series PRIVATE tenzor_core)

# KK.27: vit_image_classification's training body lives in a runner so
# tests/examples/test_all_autograd_examples.cpp can drive it. The runner is
# an OBJECT library so it can be linked into both the standalone exe and
# the test target without duplicating object files.
add_library(example_vit_image_classification_runner OBJECT
    cpp/training/vit_image_classification_runner.cpp)
target_link_libraries(example_vit_image_classification_runner PUBLIC tenzor_core)
target_include_directories(example_vit_image_classification_runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training
    ${CMAKE_SOURCE_DIR}/include)
set_target_properties(example_vit_image_classification_runner PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(vit_image_classification cpp/training/vit_image_classification.cpp)
target_link_libraries(vit_image_classification PRIVATE
    tenzor_core example_vit_image_classification_runner)
target_include_directories(vit_image_classification PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training)

add_executable(transformer_seq2seq cpp/training/transformer_seq2seq.cpp)
target_link_libraries(transformer_seq2seq PRIVATE tenzor_core)

add_executable(vae_autoencoder cpp/training/vae_autoencoder.cpp)
target_link_libraries(vae_autoencoder PRIVATE tenzor_core)

add_executable(yolo_object_detection cpp/training/yolo_object_detection.cpp)
target_link_libraries(yolo_object_detection PRIVATE tenzor_core)

add_executable(unet_semantic_segmentation cpp/training/unet_semantic_segmentation.cpp)
target_link_libraries(unet_semantic_segmentation PRIVATE tenzor_core)

add_executable(gpt_text_generation_training cpp/training/gpt_text_generation.cpp)
target_link_libraries(gpt_text_generation_training PRIVATE tenzor_core)

# KK.27: gradient_checkpointing's deep-ResNet training body lives in a
# runner so the regression test can drive it. OBJECT library shared between
# the standalone exe and the test target.
add_library(example_gradient_checkpointing_runner OBJECT
    cpp/training/gradient_checkpointing_runner.cpp)
target_link_libraries(example_gradient_checkpointing_runner PUBLIC tenzor_core)
target_include_directories(example_gradient_checkpointing_runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training
    ${CMAKE_SOURCE_DIR}/include)
set_target_properties(example_gradient_checkpointing_runner PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(gradient_checkpointing cpp/training/gradient_checkpointing.cpp)
target_link_libraries(gradient_checkpointing PRIVATE
    tenzor_core example_gradient_checkpointing_runner)
target_include_directories(gradient_checkpointing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training)

# ============================================================================
# ZeRO Optimizer and Offloading Examples
# ============================================================================

# Working examples demonstrating VRAM-aware and automatic offloading
add_executable(vram_aware_offload_example cpp/zero/vram_aware_offload_example.cpp)
target_link_libraries(vram_aware_offload_example PRIVATE tenzor_core)

add_executable(automatic_offload_example cpp/zero/automatic_offload_example.cpp)
target_link_libraries(automatic_offload_example PRIVATE tenzor_core)

# NOTE: The following examples are disabled pending implementation of:
# - tenzor/models/bert.hpp, tenzor/models/gpt2.hpp, tenzor/models/resnet.hpp
# - distributed::get_default_process_group()
# - nn/optim/adamw.hpp
#
# add_executable(bert_zero_stage1 cpp/zero/bert_zero_stage1.cpp)
# target_link_libraries(bert_zero_stage1 PRIVATE tenzor_core)
#
# add_executable(bert_zero_stage2 cpp/zero/bert_zero_stage2.cpp)
# target_link_libraries(bert_zero_stage2 PRIVATE tenzor_core)
#
# add_executable(bert_zero_stage3 cpp/zero/bert_zero_stage3.cpp)
# target_link_libraries(bert_zero_stage3 PRIVATE tenzor_core)
#
# add_executable(gpt_zero_training cpp/zero/gpt_zero_training.cpp)
# target_link_libraries(gpt_zero_training PRIVATE tenzor_core)
#
# add_executable(cpu_offload_example cpp/zero/cpu_offload_example.cpp)
# target_link_libraries(cpu_offload_example PRIVATE tenzor_core)
#
# add_executable(custom_model_zero cpp/zero/custom_model_zero.cpp)
# target_link_libraries(custom_model_zero PRIVATE tenzor_core)
#
# add_executable(distributed_training cpp/zero/distributed_training.cpp)
# target_link_libraries(distributed_training PRIVATE tenzor_core)
#
# add_executable(checkpoint_example cpp/zero/checkpoint_example.cpp)
# target_link_libraries(checkpoint_example PRIVATE tenzor_core)
#
# add_executable(mixed_precision_zero cpp/zero/mixed_precision_zero.cpp)
# target_link_libraries(mixed_precision_zero PRIVATE tenzor_core)
#
# add_executable(performance_comparison cpp/zero/performance_comparison.cpp)
# target_link_libraries(performance_comparison PRIVATE tenzor_core)

# ============================================================================
# Showcase Examples (Demonstrate library features at 3 abstraction levels)
# ============================================================================

add_subdirectory(cpp/showcase)
