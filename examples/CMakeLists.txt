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

add_executable(vit_image_classification cpp/training/vit_image_classification.cpp)
target_link_libraries(vit_image_classification PRIVATE tenzor_core)

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

add_executable(gradient_checkpointing cpp/training/gradient_checkpointing.cpp)
target_link_libraries(gradient_checkpointing PRIVATE tenzor_core)

# ============================================================================
# Showcase Examples (Demonstrate library features at 3 abstraction levels)
# ============================================================================

add_subdirectory(cpp/showcase)
