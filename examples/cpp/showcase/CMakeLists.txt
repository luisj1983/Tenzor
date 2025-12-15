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
    COMMENT "Building all Tenzor showcase examples"
)

message(STATUS "Configured ${CMAKE_CURRENT_SOURCE_DIR} showcase examples")
message(STATUS "  Examples: ${SHOWCASE_EXAMPLES}")
message(STATUS "  Abstraction levels: ${ABSTRACTION_LEVELS}")
message(STATUS "  Total executables: 33")
