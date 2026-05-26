# =============================================================================
# tests/examples/CMakeLists.txt — Example tests demonstrating fixture patterns
# =============================================================================

# Multi-parameter test example (8 tests demonstrating BackendDTypeParam usage)
add_executable(test_multi_param_example test_multi_param_example.cpp)
target_link_libraries(test_multi_param_example PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_multi_param_example LABELS "examples" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# -----------------------------------------------------------------------------
# Example regression: HRM autograd training must converge.
# Drives the actual example runner from examples/cpp/showcase/22_*/ via the
# example_22_hrm_autograd_runner OBJECT target (defined in
# examples/cpp/showcase/CMakeLists.txt).
# -----------------------------------------------------------------------------
add_executable(test_hrm_example test_hrm_example.cpp)
target_link_libraries(test_hrm_example PRIVATE
    tenzor_core
    GTest::gtest_main
    example_22_hrm_autograd_runner)
target_include_directories(test_hrm_example PRIVATE
    ${CMAKE_SOURCE_DIR}/examples/cpp/showcase)
set_target_properties(test_hrm_example PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON)
tenzor_discover_tests(test_hrm_example LABELS "examples;regression" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# -----------------------------------------------------------------------------
# Example regression: every other autograd showcase example must converge.
# Links against every OBJECT-library runner created in
# examples/cpp/showcase/CMakeLists.txt. Target names are listed here
# rather than queried via a property because CMake adds tests/ before
# examples/ — at this point those targets don't exist yet, but
# target_link_libraries resolves them lazily at end-of-configuration.
# -----------------------------------------------------------------------------
set(SHOWCASE_AUTOGRAD_RUNNER_TARGETS
    example_01_xor_autograd_runner
    example_02_linear_regression_autograd_runner
    example_03_binary_classification_autograd_runner
    example_04_multiclass_classification_autograd_runner
    example_05_convolutional_autograd_runner
    example_06_autoencoder_autograd_runner
    example_07_rnn_sequence_autograd_runner
    example_08_batch_normalization_autograd_runner
    example_09_dropout_regularization_autograd_runner
    example_10_custom_loss_autograd_runner
    example_11_chat_ai_autograd_runner
    example_12_residual_network_autograd_runner
    example_13_variational_autoencoder_autograd_runner
    example_14_gan_autograd_runner
    example_15_lstm_text_autograd_runner
    example_16_self_attention_autograd_runner
    example_17_word_embedding_autograd_runner
    example_18_transfer_learning_autograd_runner
    example_19_layer_normalization_autograd_runner
    example_20_multitask_learning_autograd_runner
    example_21_siamese_network_autograd_runner
)

# KK.27: non-showcase examples that also have regression coverage. These
# OBJECT libraries are defined in examples/CMakeLists.txt and resolved at
# end-of-configuration. bert_zero_stage1 is intentionally omitted because
# its parent executable is itself disabled (see comment block in
# examples/CMakeLists.txt and tests/examples/SKIP_NOTES.md).
set(NON_SHOWCASE_AUTOGRAD_RUNNER_TARGETS
    example_vit_image_classification_runner
    example_gradient_checkpointing_runner
    # NN.24: top-3 non-showcase training examples wired into regression
    # coverage. The nlp/gpt_text_generation example is pure inference;
    # the training-side counterpart (cpp/training/gpt_text_generation.cpp)
    # is the one with an extractable loop and is what this target covers.
    # See tests/examples/SKIP_NOTES.md.
    example_transformer_seq2seq_runner
    example_yolo_object_detection_runner
    example_gpt_text_generation_runner
)

add_executable(test_all_autograd_examples test_all_autograd_examples.cpp)
target_link_libraries(test_all_autograd_examples PRIVATE
    tenzor_core
    GTest::gtest_main
    ${SHOWCASE_AUTOGRAD_RUNNER_TARGETS}
    ${NON_SHOWCASE_AUTOGRAD_RUNNER_TARGETS})
target_include_directories(test_all_autograd_examples PRIVATE
    ${CMAKE_SOURCE_DIR}/examples/cpp/showcase
    ${CMAKE_SOURCE_DIR}/examples/cpp/training)
set_target_properties(test_all_autograd_examples PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON)
tenzor_discover_tests(test_all_autograd_examples
    LABELS "examples;regression"
    DISCOVERY_TIMEOUT 60
    DISCOVERY_MODE PRE_TEST)
