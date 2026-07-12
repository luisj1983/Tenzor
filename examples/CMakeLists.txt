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

# RR.18 (audit-11): GRU time-series runner extracted so the regression
# test can drive the same GRU+LayerNorm+Dropout+Linear+Adam+MSELoss
# pipeline. OBJECT library shared between the standalone exe and the
# test target.
add_library(example_gru_time_series_runner OBJECT
    cpp/training/gru_time_series_runner.cpp)
target_link_libraries(example_gru_time_series_runner PUBLIC tenzor_core)
target_include_directories(example_gru_time_series_runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training
    ${PROJECT_SOURCE_DIR}/include)
set_target_properties(example_gru_time_series_runner PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(gru_time_series cpp/training/gru_time_series.cpp)
target_link_libraries(gru_time_series PRIVATE
    tenzor_core example_gru_time_series_runner)
target_include_directories(gru_time_series PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training)

# KK.27: vit_image_classification's training body lives in a runner so
# tests/examples/test_all_autograd_examples.cpp can drive it. The runner is
# an OBJECT library so it can be linked into both the standalone exe and
# the test target without duplicating object files.
add_library(example_vit_image_classification_runner OBJECT
    cpp/training/vit_image_classification_runner.cpp)
target_link_libraries(example_vit_image_classification_runner PUBLIC tenzor_core)
target_include_directories(example_vit_image_classification_runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training
    ${PROJECT_SOURCE_DIR}/include)
set_target_properties(example_vit_image_classification_runner PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(vit_image_classification cpp/training/vit_image_classification.cpp)
target_link_libraries(vit_image_classification PRIVATE
    tenzor_core example_vit_image_classification_runner)
target_include_directories(vit_image_classification PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training)

# NN.24: transformer_seq2seq's attention training body lives in a runner
# so tests/examples/test_all_autograd_examples.cpp can drive it. Mirrors
# the KK.27 pattern from example_vit_image_classification_runner.
add_library(example_transformer_seq2seq_runner OBJECT
    cpp/training/transformer_seq2seq_runner.cpp)
target_link_libraries(example_transformer_seq2seq_runner PUBLIC tenzor_core)
target_include_directories(example_transformer_seq2seq_runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training
    ${PROJECT_SOURCE_DIR}/include)
set_target_properties(example_transformer_seq2seq_runner PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(transformer_seq2seq cpp/training/transformer_seq2seq.cpp)
target_link_libraries(transformer_seq2seq PRIVATE
    tenzor_core example_transformer_seq2seq_runner)
target_include_directories(transformer_seq2seq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training)

# RR.18 (audit-11): VAE runner extracted for regression coverage of the
# Linear + ReLU + Sigmoid + Adam + MSELoss pipeline.
add_library(example_vae_autoencoder_runner OBJECT
    cpp/training/vae_autoencoder_runner.cpp)
target_link_libraries(example_vae_autoencoder_runner PUBLIC tenzor_core)
target_include_directories(example_vae_autoencoder_runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training
    ${PROJECT_SOURCE_DIR}/include)
set_target_properties(example_vae_autoencoder_runner PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(vae_autoencoder cpp/training/vae_autoencoder.cpp)
target_link_libraries(vae_autoencoder PRIVATE
    tenzor_core example_vae_autoencoder_runner)
target_include_directories(vae_autoencoder PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training)

# NN.24: yolo_object_detection's tiny-backbone training body lives in a
# runner so tests/examples/test_all_autograd_examples.cpp can drive it.
add_library(example_yolo_object_detection_runner OBJECT
    cpp/training/yolo_object_detection_runner.cpp)
target_link_libraries(example_yolo_object_detection_runner PUBLIC tenzor_core)
target_include_directories(example_yolo_object_detection_runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training
    ${PROJECT_SOURCE_DIR}/include)
set_target_properties(example_yolo_object_detection_runner PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(yolo_object_detection cpp/training/yolo_object_detection.cpp)
target_link_libraries(yolo_object_detection PRIVATE
    tenzor_core example_yolo_object_detection_runner)
target_include_directories(yolo_object_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training)

# RR.18 (audit-11): U-Net runner extracted for regression coverage of the
# Conv2d + BatchNorm2d + ReLU + MaxPool + ConvTranspose2d + cat +
# CrossEntropyLoss + Adam pipeline.
add_library(example_unet_semantic_segmentation_runner OBJECT
    cpp/training/unet_semantic_segmentation_runner.cpp)
target_link_libraries(example_unet_semantic_segmentation_runner PUBLIC tenzor_core)
target_include_directories(example_unet_semantic_segmentation_runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training
    ${PROJECT_SOURCE_DIR}/include)
set_target_properties(example_unet_semantic_segmentation_runner PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(unet_semantic_segmentation cpp/training/unet_semantic_segmentation.cpp)
target_link_libraries(unet_semantic_segmentation PRIVATE
    tenzor_core example_unet_semantic_segmentation_runner)
target_include_directories(unet_semantic_segmentation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training)

# NN.24: gpt_text_generation's SimpleGPT training body lives in a runner
# so tests/examples/test_all_autograd_examples.cpp can drive it. The
# NN.24 plan referenced examples/cpp/nlp/gpt_text_generation.cpp, but
# that file is pure inference (no training loop); the training-side
# example lives here in cpp/training/ and is what the regression test
# actually needs to exercise. See tests/examples/SKIP_NOTES.md.
add_library(example_gpt_text_generation_runner OBJECT
    cpp/training/gpt_text_generation_runner.cpp)
target_link_libraries(example_gpt_text_generation_runner PUBLIC tenzor_core)
target_include_directories(example_gpt_text_generation_runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training
    ${PROJECT_SOURCE_DIR}/include)
set_target_properties(example_gpt_text_generation_runner PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(gpt_text_generation_training cpp/training/gpt_text_generation.cpp)
target_link_libraries(gpt_text_generation_training PRIVATE
    tenzor_core example_gpt_text_generation_runner)
target_include_directories(gpt_text_generation_training PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training)

# KK.27: gradient_checkpointing's deep-ResNet training body lives in a
# runner so the regression test can drive it. OBJECT library shared between
# the standalone exe and the test target.
add_library(example_gradient_checkpointing_runner OBJECT
    cpp/training/gradient_checkpointing_runner.cpp)
target_link_libraries(example_gradient_checkpointing_runner PUBLIC tenzor_core)
target_include_directories(example_gradient_checkpointing_runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training
    ${PROJECT_SOURCE_DIR}/include)
set_target_properties(example_gradient_checkpointing_runner PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(gradient_checkpointing cpp/training/gradient_checkpointing.cpp)
target_link_libraries(gradient_checkpointing PRIVATE
    tenzor_core example_gradient_checkpointing_runner)
target_include_directories(gradient_checkpointing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training)

# R2-02: first example exercising the JIT subsystem end-to-end (trace ->
# compile -> execute -> verify vs eager), wired the same way as every other
# training example so the regression test can drive the same code path.
add_library(example_jit_mlp_training_runner OBJECT
    cpp/training/jit_mlp_training_runner.cpp)
target_link_libraries(example_jit_mlp_training_runner PUBLIC tenzor_core)
target_include_directories(example_jit_mlp_training_runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/training
    ${PROJECT_SOURCE_DIR}/include)
set_target_properties(example_jit_mlp_training_runner PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(jit_mlp_training cpp/training/jit_mlp_training.cpp)
target_link_libraries(jit_mlp_training PRIVATE
    tenzor_core example_jit_mlp_training_runner)
target_include_directories(jit_mlp_training PRIVATE
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

# ============================================================================
# Python examples as tests (examples-as-tests pattern, Python side)
# ============================================================================
# Each runnable Python example is registered as a ctest so API drift breaks
# CI instead of bit-rotting silently. They run against the build-tree Python
# package (build/python) and need TENZOR_BUILD_PYTHON=ON.
if(TENZOR_BUILD_PYTHON AND TARGET tenzor_python)
    find_package(Python COMPONENTS Interpreter QUIET)
    if(Python_FOUND)
        set(_PY_EXAMPLES
            01_tensor_basics
            02_autograd_basics
            03_linear_regression
            04_mnist_mlp
            05_cnn_classification
            06_custom_layer
            09_rnn_text_classification
            10_distributed_training
            checkpoint_examples
            model_hub_example
            onnx_export_demo
            onnx_roundtrip_verification
        )
        foreach(_ex ${_PY_EXAMPLES})
            add_test(NAME example_python_${_ex}
                COMMAND ${Python_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/python/${_ex}.py)
            set_tests_properties(example_python_${_ex} PROPERTIES
                ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/python"
                TIMEOUT 600
                LABELS "examples;python")
        endforeach()
        # Heavier training demos: generous timeout, separate label so quick
        # CI runs can exclude them (ctest -LE examples_slow).
        foreach(_ex 07_resnet_cifar10 08_fashion_mnist_cnn)
            add_test(NAME example_python_${_ex}
                COMMAND ${Python_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/python/${_ex}.py)
            set_tests_properties(example_python_${_ex} PROPERTIES
                ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/python"
                TIMEOUT 1800
                LABELS "examples;python;examples_slow")
        endforeach()
    endif()
endif()
