# Tenzor Showcase Examples CMakeLists.txt
# Build all showcase examples demonstrating Tenzor library features

cmake_minimum_required(VERSION 3.16)

# Define showcase examples
set(SHOWCASE_EXAMPLES
    01_xor
    02_linear_regression
    03_binary_classification
    04_multiclass_classification
    05_convolutional
    06_autoencoder
    07_rnn_sequence
    08_batch_normalization
    09_dropout_regularization
    10_custom_loss
    11_chat_ai
    12_residual_network
    13_variational_autoencoder
    14_gan
    15_lstm_text
    16_self_attention
    17_word_embedding
    18_transfer_learning
    19_layer_normalization
    20_multitask_learning
    21_siamese_network
    22_hierarchical_reasoning
)

# Define abstraction levels for each example
set(ABSTRACTION_LEVELS
    tensor_only
    autograd
    neural_network
)

# Build each example at each abstraction level
foreach(EXAMPLE ${SHOWCASE_EXAMPLES})
    foreach(LEVEL ${ABSTRACTION_LEVELS})
        set(TARGET_NAME "${EXAMPLE}_${LEVEL}")
        set(SOURCE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}/${LEVEL}.cpp")

        if(EXISTS ${SOURCE_FILE})
            add_executable(${TARGET_NAME} ${SOURCE_FILE})
            target_link_libraries(${TARGET_NAME} PRIVATE tenzor_core)
            target_include_directories(${TARGET_NAME} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${CMAKE_SOURCE_DIR}/include
            )

            # Set C++23 standard (required for Tenzor's concepts, std::span, std::expected)
            set_target_properties(${TARGET_NAME} PROPERTIES
                CXX_STANDARD 23
                CXX_STANDARD_REQUIRED ON
            )

            # Install to bin directory
            install(TARGETS ${TARGET_NAME}
                RUNTIME DESTINATION bin/showcase
            )
        else()
            message(WARNING "Source file not found: ${SOURCE_FILE}")
        endif()
    endforeach()
endforeach()

# -----------------------------------------------------------------------------
# Autograd-runner OBJECT libraries — the training body of each autograd.cpp
# example is extracted into autograd_runner.cpp so the test target
# tests/examples/test_all_autograd_examples.cpp (and the dedicated HRM test)
# can drive the same code path as the showcase exe and assert that loss
# decreases. Each runner is paired with the showcase executable that uses it.
# -----------------------------------------------------------------------------

# Mapping: showcase target name -> runner source file.
set(AUTOGRAD_RUNNERS
    "01_xor_autograd:01_xor/autograd_runner.cpp"
    "02_linear_regression_autograd:02_linear_regression/autograd_runner.cpp"
    "03_binary_classification_autograd:03_binary_classification/autograd_runner.cpp"
    "04_multiclass_classification_autograd:04_multiclass_classification/autograd_runner.cpp"
    "05_convolutional_autograd:05_convolutional/autograd_runner.cpp"
    "06_autoencoder_autograd:06_autoencoder/autograd_runner.cpp"
    "07_rnn_sequence_autograd:07_rnn_sequence/autograd_runner.cpp"
    "08_batch_normalization_autograd:08_batch_normalization/autograd_runner.cpp"
    "09_dropout_regularization_autograd:09_dropout_regularization/autograd_runner.cpp"
    "10_custom_loss_autograd:10_custom_loss/autograd_runner.cpp"
    "11_chat_ai_autograd:11_chat_ai/autograd_runner.cpp"
    "12_residual_network_autograd:12_residual_network/autograd_runner.cpp"
    "13_variational_autoencoder_autograd:13_variational_autoencoder/autograd_runner.cpp"
    "14_gan_autograd:14_gan/autograd_runner.cpp"
    "15_lstm_text_autograd:15_lstm_text/autograd_runner.cpp"
    "16_self_attention_autograd:16_self_attention/autograd_runner.cpp"
    "17_word_embedding_autograd:17_word_embedding/autograd_runner.cpp"
    "18_transfer_learning_autograd:18_transfer_learning/autograd_runner.cpp"
    "19_layer_normalization_autograd:19_layer_normalization/autograd_runner.cpp"
    "20_multitask_learning_autograd:20_multitask_learning/autograd_runner.cpp"
    "21_siamese_network_autograd:21_siamese_network/autograd_runner.cpp"
    "22_hierarchical_reasoning_autograd:22_hierarchical_reasoning/autograd_runner.cpp"
)

set(SHOWCASE_RUNNER_TARGETS "")
foreach(MAPPING ${AUTOGRAD_RUNNERS})
    string(REPLACE ":" ";" PARTS ${MAPPING})
    list(GET PARTS 0 EXE_TARGET)
    list(GET PARTS 1 RUNNER_SRC)
    set(RUNNER_TARGET "example_${EXE_TARGET}_runner")
    set(RUNNER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${RUNNER_SRC}")
    if(EXISTS ${RUNNER_PATH})
        add_library(${RUNNER_TARGET} OBJECT ${RUNNER_PATH})
        target_link_libraries(${RUNNER_TARGET} PUBLIC tenzor_core)
        target_include_directories(${RUNNER_TARGET} PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/include)
        set_target_properties(${RUNNER_TARGET} PROPERTIES
            CXX_STANDARD 23
            CXX_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON)
        if(TARGET ${EXE_TARGET})
            target_link_libraries(${EXE_TARGET} PRIVATE ${RUNNER_TARGET})
        endif()
        list(APPEND SHOWCASE_RUNNER_TARGETS ${RUNNER_TARGET})
    else()
        message(STATUS "Autograd runner not found, skipping: ${RUNNER_PATH}")
    endif()
endforeach()

# Backwards-compatible alias for the dedicated HRM test target which links
# against `example_22_hrm_autograd_runner` explicitly.
if(TARGET example_22_hierarchical_reasoning_autograd_runner)
    add_library(example_22_hrm_autograd_runner ALIAS
        example_22_hierarchical_reasoning_autograd_runner)
endif()

# The list of OBJECT-library runner targets (example_<N>_..._autograd_runner)
# is consumed by tests/examples/CMakeLists.txt by name. We don't need to
# expose it explicitly: CMake resolves cross-subdirectory target references
# lazily, and the test subdir is configured before this one.

# Create a convenience target to build all showcase examples
add_custom_target(showcase_examples
    DEPENDS
        01_xor_tensor_only
        01_xor_autograd
        01_xor_neural_network
        02_linear_regression_tensor_only
        02_linear_regression_autograd
        02_linear_regression_neural_network
        03_binary_classification_tensor_only
        03_binary_classification_autograd
        03_binary_classification_neural_network
        04_multiclass_classification_tensor_only
        04_multiclass_classification_autograd
        04_multiclass_classification_neural_network
        05_convolutional_tensor_only
        05_convolutional_autograd
        05_convolutional_neural_network
        06_autoencoder_tensor_only
        06_autoencoder_autograd
        06_autoencoder_neural_network
        07_rnn_sequence_tensor_only
        07_rnn_sequence_autograd
        07_rnn_sequence_neural_network
        08_batch_normalization_tensor_only
        08_batch_normalization_autograd
        08_batch_normalization_neural_network
        09_dropout_regularization_tensor_only
        09_dropout_regularization_autograd
        09_dropout_regularization_neural_network
        10_custom_loss_tensor_only
        10_custom_loss_autograd
        10_custom_loss_neural_network
        11_chat_ai_tensor_only
        11_chat_ai_autograd
        11_chat_ai_neural_network
        12_residual_network_tensor_only
        12_residual_network_autograd
        12_residual_network_neural_network
        13_variational_autoencoder_tensor_only
        13_variational_autoencoder_autograd
        13_variational_autoencoder_neural_network
        14_gan_tensor_only
        14_gan_autograd
        14_gan_neural_network
        15_lstm_text_tensor_only
        15_lstm_text_autograd
        15_lstm_text_neural_network
        16_self_attention_tensor_only
        16_self_attention_autograd
        16_self_attention_neural_network
        17_word_embedding_tensor_only
        17_word_embedding_autograd
        17_word_embedding_neural_network
        18_transfer_learning_tensor_only
        18_transfer_learning_autograd
        18_transfer_learning_neural_network
        19_layer_normalization_tensor_only
        19_layer_normalization_autograd
        19_layer_normalization_neural_network
        20_multitask_learning_tensor_only
        20_multitask_learning_autograd
        20_multitask_learning_neural_network
        21_siamese_network_tensor_only
        21_siamese_network_autograd
        21_siamese_network_neural_network
        22_hierarchical_reasoning_tensor_only
        22_hierarchical_reasoning_autograd
        22_hierarchical_reasoning_neural_network
    COMMENT "Building all Tenzor showcase examples"
)

message(STATUS "Configured ${CMAKE_CURRENT_SOURCE_DIR} showcase examples")
message(STATUS "  Examples: ${SHOWCASE_EXAMPLES}")
message(STATUS "  Abstraction levels: ${ABSTRACTION_LEVELS}")
message(STATUS "  Total executables: 66")
