# =============================================================================
# tests/unit/CMakeLists.txt — Unit tests from tests/unit/ directory
# =============================================================================

# Embedding tests
add_executable(test_embedding test_embedding.cpp)
target_link_libraries(test_embedding PRIVATE tenzor_core GTest::gtest_main)

# Embedding multi-dtype tests
add_executable(test_embedding_multidtype test_embedding_multidtype.cpp)
target_link_libraries(test_embedding_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Extended optimizers tests
add_executable(test_optimizers_extended test_optimizers_extended.cpp)
target_link_libraries(test_optimizers_extended PRIVATE tenzor_core GTest::gtest_main)

# Advanced schedulers tests
add_executable(test_schedulers_advanced test_schedulers_advanced.cpp)
target_link_libraries(test_schedulers_advanced PRIVATE tenzor_core GTest::gtest_main)

# Advanced losses tests
add_executable(test_losses_advanced test_losses_advanced.cpp)
target_link_libraries(test_losses_advanced PRIVATE tenzor_core GTest::gtest_main)

# RNN tests
add_executable(test_rnn test_rnn.cpp)
target_link_libraries(test_rnn PRIVATE tenzor_core GTest::gtest_main)

# RNN multi-dtype tests
add_executable(test_rnn_multidtype test_rnn_multidtype.cpp)
target_link_libraries(test_rnn_multidtype PRIVATE tenzor_core GTest::gtest_main)

# LSTM tests
add_executable(test_lstm test_lstm.cpp)
target_link_libraries(test_lstm PRIVATE tenzor_core GTest::gtest_main)

# Audit G1: LSTM with proj_size (PyTorch LSTMP).
add_executable(test_lstm_proj_size test_lstm_proj_size.cpp)
target_link_libraries(test_lstm_proj_size PRIVATE tenzor_core GTest::gtest_main)

# LSTM multi-dtype tests
add_executable(test_lstm_multidtype test_lstm_multidtype.cpp)
target_link_libraries(test_lstm_multidtype PRIVATE tenzor_core GTest::gtest_main)

# LSTM/GRU CUDA forward regression test (v0.1.0 cuBLAS INVALID_VALUE bug)
add_executable(test_lstm_cuda_forward_regression test_lstm_cuda_forward_regression.cpp)
target_link_libraries(test_lstm_cuda_forward_regression PRIVATE tenzor_core GTest::gtest_main)

# SDPA multi-backend × multi-dtype tests. Replaces the legacy CUDA-only
# test_attention_sdpa_fp32.cpp; covers Float16/BFloat16/Float32/Float64
# forward + backward across every non-MPS backend.
add_executable(test_attention_sdpa_multidtype test_attention_sdpa_multidtype.cpp)
target_link_libraries(test_attention_sdpa_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Float16 regression tests (consolidated from 19 minimal_float16_* debug tests)
add_executable(test_float16_regression test_float16_regression.cpp)
target_link_libraries(test_float16_regression PRIVATE tenzor_core GTest::gtest_main)

# GRU tests
add_executable(test_gru test_gru.cpp)
target_link_libraries(test_gru PRIVATE tenzor_core GTest::gtest_main)

# GRU multi-dtype tests
add_executable(test_gru_multidtype test_gru_multidtype.cpp)
target_link_libraries(test_gru_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Attention tests
add_executable(test_attention test_attention.cpp)
target_link_libraries(test_attention PRIVATE tenzor_core GTest::gtest_main)

# Attention multi-dtype tests
add_executable(test_attention_multidtype test_attention_multidtype.cpp)
target_link_libraries(test_attention_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Transformer tests
add_executable(test_transformer test_transformer.cpp)
target_link_libraries(test_transformer PRIVATE tenzor_core GTest::gtest_main)

# Transformer multi-dtype tests
add_executable(test_transformer_multidtype test_transformer_multidtype.cpp)
target_link_libraries(test_transformer_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Chunk tests
add_executable(test_chunk test_chunk.cpp)
target_link_libraries(test_chunk PRIVATE tenzor_core GTest::gtest_main)

# GradScaler (AMP) tests
add_executable(test_grad_scaler test_grad_scaler.cpp)
target_link_libraries(test_grad_scaler PRIVATE tenzor_core GTest::gtest_main)

# GradScaler multi-dtype tests
add_executable(test_grad_scaler_multidtype test_grad_scaler_multidtype.cpp)
target_link_libraries(test_grad_scaler_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Autocast tests
add_executable(test_autocast test_autocast.cpp)
target_link_libraries(test_autocast PRIVATE tenzor_core GTest::gtest_main)

# Autocast multidtype tests
add_executable(test_autocast_multidtype test_autocast_multidtype.cpp)
target_link_libraries(test_autocast_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Mixed precision training tests
add_executable(test_mixed_precision test_mixed_precision.cpp)
target_link_libraries(test_mixed_precision PRIVATE tenzor_core GTest::gtest_main)

# DataLoader tests
add_executable(test_dataloader test_dataloader.cpp)
target_link_libraries(test_dataloader PRIVATE tenzor_core GTest::gtest_main)

# Float16/BFloat16 tests
add_executable(test_fp16 test_fp16.cpp)
target_link_libraries(test_fp16 PRIVATE tenzor_core GTest::gtest_main)

# Gradient checkpoint tests
add_executable(test_gradient_checkpoint test_gradient_checkpoint.cpp)
target_link_libraries(test_gradient_checkpoint PRIVATE tenzor_core GTest::gtest_main)

# Gradient checkpoint multi-dtype tests
add_executable(test_checkpoint_multidtype test_checkpoint_multidtype.cpp)
target_link_libraries(test_checkpoint_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Model checkpoint tests
add_executable(test_model_checkpoint test_model_checkpoint.cpp)
target_link_libraries(test_model_checkpoint PRIVATE tenzor_core GTest::gtest_main)

# Comprehensive checkpoint tests
add_executable(test_checkpoint test_checkpoint.cpp)
target_link_libraries(test_checkpoint PRIVATE tenzor_core GTest::gtest_main)

# Fused operations tests
add_executable(test_fused_ops test_fused_ops.cpp)
target_link_libraries(test_fused_ops PRIVATE tenzor_core GTest::gtest_main)

# Fused operations multi-dtype tests
add_executable(test_fused_ops_multidtype test_fused_ops_multidtype.cpp)
target_link_libraries(test_fused_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)

# ============================================================================
# Multi-DType Parameterized Tests - Backend x DType Coverage
# ============================================================================

# Core tensor operations multi-dtype tests
add_executable(test_tensor_multidtype test_tensor_multidtype.cpp)
target_link_libraries(test_tensor_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Autograd operations multi-dtype tests
add_executable(test_autograd_multidtype test_autograd_multidtype.cpp)
target_link_libraries(test_autograd_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Broadcasting operations multi-dtype tests
add_executable(test_broadcasting test_broadcasting.cpp)
target_link_libraries(test_broadcasting PRIVATE tenzor_core GTest::gtest_main)

add_executable(test_broadcasting_multidtype test_broadcasting_multidtype.cpp)
target_link_libraries(test_broadcasting_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Math operations multi-dtype tests
add_executable(test_ops_multidtype test_ops_multidtype.cpp)
target_link_libraries(test_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Comparison operations multi-dtype tests
add_executable(test_comparison_ops_multidtype test_comparison_ops_multidtype.cpp)
target_link_libraries(test_comparison_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Detection operations multi-dtype tests
add_executable(test_detection_ops_multidtype test_detection_ops_multidtype.cpp)
target_link_libraries(test_detection_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Optimizer multi-dtype tests
add_executable(test_optimizers_multidtype test_optimizers_multidtype.cpp)
target_link_libraries(test_optimizers_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Linear multi-dtype tests
add_executable(test_linear_multidtype test_linear_multidtype.cpp)
target_link_libraries(test_linear_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Backend operations parameterized tests (runs on ALL backends)
add_executable(test_backend_ops_parameterized test_backend_ops_parameterized.cpp)
target_link_libraries(test_backend_ops_parameterized PRIVATE tenzor_core GTest::gtest_main)

# Comprehensive edge case and error handling tests
add_executable(test_edge_cases test_edge_cases.cpp)
target_link_libraries(test_edge_cases PRIVATE tenzor_core GTest::gtest_main)

# DType-specific edge case tests
add_executable(test_dtype_edge_cases test_dtype_edge_cases.cpp)
target_link_libraries(test_dtype_edge_cases PRIVATE tenzor_core GTest::gtest_main)

# CachingAllocator tests (universal - uses mock backend)
add_executable(test_caching_allocator test_caching_allocator.cpp)
target_link_libraries(test_caching_allocator PRIVATE tenzor_core GTest::gtest_main)

add_executable(test_allocator_overflow test_allocator_overflow.cpp)
target_link_libraries(test_allocator_overflow PRIVATE tenzor_core GTest::gtest_main)

# ============================================================================
# Phase 6 Tests - Improved Test Coverage
# ============================================================================

# Empty tensor edge cases
add_executable(test_empty_tensors test_empty_tensors.cpp)
target_link_libraries(test_empty_tensors PRIVATE tenzor_core GTest::gtest_main)

# View and slice semantics
add_executable(test_view_slice_semantics test_view_slice_semantics.cpp)
target_link_libraries(test_view_slice_semantics PRIVATE tenzor_core GTest::gtest_main)

# Broadcasting edge cases
add_executable(test_broadcasting_edge_cases test_broadcasting_edge_cases.cpp)
target_link_libraries(test_broadcasting_edge_cases PRIVATE tenzor_core GTest::gtest_main)

# Numerical precision tests
add_executable(test_numerical_precision test_numerical_precision.cpp)
target_link_libraries(test_numerical_precision PRIVATE tenzor_core GTest::gtest_main)

# DType conversion chain tests
add_executable(test_dtype_conversion_chains test_dtype_conversion_chains.cpp)
target_link_libraries(test_dtype_conversion_chains PRIVATE tenzor_core GTest::gtest_main)

# ============================================================================
# Loss function multi-dtype tests
# ============================================================================
add_executable(test_losses_multidtype test_losses_multidtype.cpp)
target_link_libraries(test_losses_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Float16 operations multi-dtype tests
add_executable(test_fp16_multidtype test_fp16_multidtype.cpp)
target_link_libraries(test_fp16_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Mixed precision training multi-dtype tests
add_executable(test_mixed_precision_multidtype test_mixed_precision_multidtype.cpp)
target_link_libraries(test_mixed_precision_multidtype PRIVATE tenzor_core GTest::gtest_main)

# BERT model multi-dtype tests
add_executable(test_bert_multidtype test_bert_multidtype.cpp)
target_link_libraries(test_bert_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Vision Transformer multi-dtype tests
add_executable(test_vit_multidtype test_vit_multidtype.cpp)
target_link_libraries(test_vit_multidtype PRIVATE tenzor_core GTest::gtest_main)

# U-Net model multi-dtype tests
add_executable(test_unet_multidtype test_unet_multidtype.cpp)
target_link_libraries(test_unet_multidtype PRIVATE tenzor_core GTest::gtest_main)

# YOLO model multi-dtype tests
add_executable(test_yolo_multidtype test_yolo_multidtype.cpp)
target_link_libraries(test_yolo_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Mask R-CNN model multi-dtype tests
add_executable(test_mask_rcnn_multidtype test_mask_rcnn_multidtype.cpp)
target_link_libraries(test_mask_rcnn_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Faster R-CNN model multi-dtype tests
add_executable(test_faster_rcnn_multidtype test_faster_rcnn_multidtype.cpp)
target_link_libraries(test_faster_rcnn_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Swin Transformer multi-dtype tests
add_executable(test_swin_transformer_multidtype test_swin_transformer_multidtype.cpp)
target_link_libraries(test_swin_transformer_multidtype PRIVATE tenzor_core GTest::gtest_main)

# EfficientNet model multi-dtype tests
add_executable(test_efficientnet_multidtype test_efficientnet_multidtype.cpp)
target_link_libraries(test_efficientnet_multidtype PRIVATE tenzor_core GTest::gtest_main)

# SIMD operations tests
add_executable(test_simd_ops test_simd_ops.cpp)
target_link_libraries(test_simd_ops PRIVATE tenzor_core tenzor_backend_cpu GTest::gtest_main)

# test_simd_ops crashes during --gtest_list_tests (static destruction), use add_test
# CPU-only SIMD kernel test; prefix with cpu_ so `ctest -R cpu` finds it.
add_test(NAME cpu_test_simd_ops COMMAND test_simd_ops)
set_tests_properties(cpu_test_simd_ops PROPERTIES TIMEOUT 120 LABELS "cpu")

# New comparison ops tests (OneAPI/Vulkan backends)
add_executable(test_comparison_ops test_comparison_ops.cpp)
target_link_libraries(test_comparison_ops PRIVATE tenzor_core GTest::gtest_main)

# ONNX export tests
add_executable(test_onnx_export test_onnx_export.cpp)
target_link_libraries(test_onnx_export PRIVATE tenzor_core GTest::gtest_main)

# 5th-audit ONNX exporter regression coverage (Complex throw, ConvTranspose
# validation, BatchNorm training_mode, external_data).
add_executable(test_onnx_audit_fixes test_onnx_audit_fixes.cpp)
target_link_libraries(test_onnx_audit_fixes PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_onnx_audit_fixes LABELS "unit;onnx" DISCOVERY_TIMEOUT 60)

# Inf-D: ONNX exporter coverage (every forward OpId has a mapping; extended
# DTypes — BF16/FP8/INT4 — emit the right ONNX type codes).
add_executable(test_onnx_exporter_coverage test_onnx_exporter_coverage.cpp)
target_link_libraries(test_onnx_exporter_coverage PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_onnx_exporter_coverage LABELS "unit;onnx" DISCOVERY_TIMEOUT 60)

# ONNX round-trip tests
add_executable(test_onnx_roundtrip test_onnx_roundtrip.cpp)
target_link_libraries(test_onnx_roundtrip PRIVATE tenzor_core GTest::gtest_main)

# ModelHub tests
add_executable(test_model_hub test_model_hub.cpp)
target_link_libraries(test_model_hub PRIVATE tenzor_core GTest::gtest_main)

# BERT model tests
add_executable(test_bert test_bert.cpp)
target_link_libraries(test_bert PRIVATE tenzor_core GTest::gtest_main)

# Classic CNN models tests
add_executable(test_classic_models test_classic_models.cpp)
target_link_libraries(test_classic_models PRIVATE tenzor_core GTest::gtest_main)

# GPT model tests
add_executable(test_gpt test_gpt.cpp)
target_link_libraries(test_gpt PRIVATE tenzor_core GTest::gtest_main)

# GPT model multi-dtype tests
add_executable(test_gpt_multidtype test_gpt_multidtype.cpp)
target_link_libraries(test_gpt_multidtype PRIVATE tenzor_core GTest::gtest_main)

# ResNet model tests
add_executable(test_resnet test_resnet.cpp)
target_link_libraries(test_resnet PRIVATE tenzor_core GTest::gtest_main)

# ResNet multi-dtype tests
add_executable(test_resnet_multidtype test_resnet_multidtype.cpp)
target_link_libraries(test_resnet_multidtype PRIVATE tenzor_core GTest::gtest_main)

# RoBERTa model tests
add_executable(test_roberta test_roberta.cpp)
target_link_libraries(test_roberta PRIVATE tenzor_core GTest::gtest_main)

# ELECTRA model tests
add_executable(test_electra test_electra.cpp)
target_link_libraries(test_electra PRIVATE tenzor_core GTest::gtest_main)

# ELECTRA model multi-dtype tests
add_executable(test_electra_multidtype test_electra_multidtype.cpp)
target_link_libraries(test_electra_multidtype PRIVATE tenzor_core GTest::gtest_main)

# ============================================================================
# Phase 9 Model Tests - Vision Models
# ============================================================================

# EfficientNet tests (B0-B7)
add_executable(test_efficientnet test_efficientnet.cpp)
target_link_libraries(test_efficientnet PRIVATE tenzor_core GTest::gtest_main)

# Vision Transformer tests (Base/Large/Huge)
add_executable(test_vit test_vit.cpp)
target_link_libraries(test_vit PRIVATE tenzor_core GTest::gtest_main)

# ConvNeXt tests
add_executable(test_convnext test_convnext.cpp)
target_link_libraries(test_convnext PRIVATE tenzor_core GTest::gtest_main)

# ConvNeXt multi-dtype tests
add_executable(test_convnext_multidtype test_convnext_multidtype.cpp)
target_link_libraries(test_convnext_multidtype PRIVATE tenzor_core GTest::gtest_main)

# MobileNet V2/V3 tests
add_executable(test_mobilenet_v2_v3 test_mobilenet_v2_v3.cpp)
target_link_libraries(test_mobilenet_v2_v3 PRIVATE tenzor_core GTest::gtest_main)

# ============================================================================
# Phase 9 Model Tests - NLP Models
# ============================================================================

# RoBERTa and ELECTRA tests
add_executable(test_roberta_electra test_roberta_electra.cpp)
target_link_libraries(test_roberta_electra PRIVATE tenzor_core GTest::gtest_main)

# ALBERT and T5 tests
add_executable(test_albert_t5 test_albert_t5.cpp)
target_link_libraries(test_albert_t5 PRIVATE tenzor_core GTest::gtest_main)

# ============================================================================
# Phase 9 Model Tests - Detection and Segmentation Models
# ============================================================================

# Faster R-CNN tests
add_executable(test_faster_rcnn test_faster_rcnn.cpp)
target_link_libraries(test_faster_rcnn PRIVATE tenzor_core GTest::gtest_main)

# YOLO tests (v3 and v5)
add_executable(test_yolo test_yolo.cpp)
target_link_libraries(test_yolo PRIVATE tenzor_core GTest::gtest_main)

# Mask R-CNN tests
add_executable(test_mask_rcnn test_mask_rcnn.cpp)
target_link_libraries(test_mask_rcnn PRIVATE tenzor_core GTest::gtest_main)

# U-Net tests
add_executable(test_unet test_unet.cpp)
target_link_libraries(test_unet PRIVATE tenzor_core GTest::gtest_main)

# DeepLab v3+ tests
add_executable(test_deeplabv3plus test_deeplabv3plus.cpp)
target_link_libraries(test_deeplabv3plus PRIVATE tenzor_core GTest::gtest_main)

# DeepLab v3+ multi-dtype tests
add_executable(test_deeplabv3plus_multidtype test_deeplabv3plus_multidtype.cpp)
target_link_libraries(test_deeplabv3plus_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Detection Operations tests (ROI, NMS, etc.)
add_executable(test_detection_ops test_detection_ops.cpp)
target_link_libraries(test_detection_ops PRIVATE tenzor_core GTest::gtest_main)

# ============================================================================
# Phase 10 Tests - Model Export, Optimization, and Compression
# ============================================================================

# JIT (Just-In-Time) compilation tests
add_executable(test_jit test_jit.cpp)
target_link_libraries(test_jit PRIVATE tenzor_core GTest::gtest_main)

# JIT multi-dtype tests
add_executable(test_jit_multidtype test_jit_multidtype.cpp)
target_link_libraries(test_jit_multidtype PRIVATE tenzor_core GTest::gtest_main)

# JIT script (compile_script) tests — MVP parser / evaluator coverage.
add_executable(test_jit_script test_jit_script.cpp)
target_link_libraries(test_jit_script PRIVATE tenzor_core GTest::gtest_main)

# Pruning tests
add_executable(test_pruning test_pruning.cpp)
target_link_libraries(test_pruning PRIVATE tenzor_core GTest::gtest_main)

# Pruning multi-dtype tests
add_executable(test_pruning_multidtype test_pruning_multidtype.cpp)
target_link_libraries(test_pruning_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Knowledge distillation tests
add_executable(test_distillation test_distillation.cpp)
target_link_libraries(test_distillation PRIVATE tenzor_core GTest::gtest_main)

# Knowledge distillation multi-dtype tests
add_executable(test_distillation_multidtype test_distillation_multidtype.cpp)
target_link_libraries(test_distillation_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Callbacks multi-dtype tests
add_executable(test_callbacks_multidtype test_callbacks_multidtype.cpp)
target_link_libraries(test_callbacks_multidtype PRIVATE tenzor_core GTest::gtest_main)

# RoBERTa model multi-dtype tests
add_executable(test_roberta_multidtype test_roberta_multidtype.cpp)
target_link_libraries(test_roberta_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Edge cases multi-dtype tests
add_executable(test_edge_cases_multidtype test_edge_cases_multidtype.cpp)
target_link_libraries(test_edge_cases_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Quantization multi-dtype tests (Float32/Float64 to Int8)
add_executable(test_quantization_multidtype test_quantization_multidtype.cpp)
target_link_libraries(test_quantization_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Async operations tests
add_executable(test_async_ops test_async_ops.cpp)
target_link_libraries(test_async_ops PRIVATE tenzor_core GTest::gtest_main)

# Data loading tests (multi-backend)
add_executable(test_data_loading test_data_loading.cpp)
target_link_libraries(test_data_loading PRIVATE tenzor_core GTest::gtest_main)

# SIMD tests (multi-backend)
add_executable(test_simd test_simd.cpp)
target_link_libraries(test_simd PRIVATE tenzor_core tenzor_backend_cpu GTest::gtest_main)

# test_simd crashes during --gtest_list_tests (static destruction), use add_test
# CPU-only SIMD harness; prefix with cpu_ so `ctest -R cpu` finds it.
add_test(NAME cpu_test_simd COMMAND test_simd)
set_tests_properties(cpu_test_simd PROPERTIES TIMEOUT 120 LABELS "cpu")

# Inplace operations multi-dtype tests
add_executable(test_inplace_operations_multidtype test_inplace_operations_multidtype.cpp)
target_link_libraries(test_inplace_operations_multidtype PRIVATE tenzor_core GTest::gtest_main)

# ============================================================================
# Additional Multi-DType Tests - Option 3 Completion
# ============================================================================

# NLP Models - ALBERT & T5 multi-dtype tests
add_executable(test_albert_t5_multidtype test_albert_t5_multidtype.cpp)
target_link_libraries(test_albert_t5_multidtype PRIVATE tenzor_core GTest::gtest_main)

# NLP Models - RoBERTa & ELECTRA combined multi-dtype tests
add_executable(test_roberta_electra_multidtype test_roberta_electra_multidtype.cpp)
target_link_libraries(test_roberta_electra_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Vision Models - MobileNet V2/V3 multi-dtype tests
add_executable(test_mobilenet_v2_v3_multidtype test_mobilenet_v2_v3_multidtype.cpp)
target_link_libraries(test_mobilenet_v2_v3_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Vision Models - Classic models multi-dtype tests
add_executable(test_classic_models_multidtype test_classic_models_multidtype.cpp)
target_link_libraries(test_classic_models_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Vision Components multi-dtype tests
add_executable(test_vision_components_multidtype test_vision_components_multidtype.cpp)
target_link_libraries(test_vision_components_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Detection Components multi-dtype tests
add_executable(test_detection_components_multidtype test_detection_components_multidtype.cpp)
target_link_libraries(test_detection_components_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Training Utilities - Advanced losses multi-dtype tests
add_executable(test_losses_advanced_multidtype test_losses_advanced_multidtype.cpp)
target_link_libraries(test_losses_advanced_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Training Utilities - Advanced schedulers multi-dtype tests
add_executable(test_schedulers_advanced_multidtype test_schedulers_advanced_multidtype.cpp)
target_link_libraries(test_schedulers_advanced_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Training Utilities - Extended optimizers multi-dtype tests
add_executable(test_optimizers_extended_multidtype test_optimizers_extended_multidtype.cpp)
target_link_libraries(test_optimizers_extended_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Operations - Additional NN components multi-dtype tests
add_executable(test_nn_additional_multidtype test_nn_additional_multidtype.cpp)
target_link_libraries(test_nn_additional_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Operations - Additional ops multi-dtype tests
add_executable(test_ops_additional_multidtype test_ops_additional_multidtype.cpp)
target_link_libraries(test_ops_additional_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Operations - Chunk multi-dtype tests
add_executable(test_chunk_multidtype test_chunk_multidtype.cpp)
target_link_libraries(test_chunk_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Infrastructure - Gradient checkpoint multi-dtype tests
add_executable(test_gradient_checkpoint_multidtype test_gradient_checkpoint_multidtype.cpp)
target_link_libraries(test_gradient_checkpoint_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Infrastructure - Model checkpoint multi-dtype tests
add_executable(test_model_checkpoint_multidtype test_model_checkpoint_multidtype.cpp)
target_link_libraries(test_model_checkpoint_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Infrastructure - Transforms multi-dtype tests
add_executable(test_transforms_multidtype test_transforms_multidtype.cpp)
target_link_libraries(test_transforms_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Infrastructure - Async operations multi-dtype tests
add_executable(test_async_ops_multidtype test_async_ops_multidtype.cpp)
target_link_libraries(test_async_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Infrastructure - Dataloader multi-dtype tests
add_executable(test_dataloader_multidtype test_dataloader_multidtype.cpp)
target_link_libraries(test_dataloader_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Infrastructure - DType edge cases multi-dtype tests (enhanced)
add_executable(test_dtype_edge_cases_multidtype test_dtype_edge_cases_multidtype.cpp)
target_link_libraries(test_dtype_edge_cases_multidtype PRIVATE tenzor_core GTest::gtest_main)

# F13/F22-followup: Philox dropout helper
add_executable(test_philox_dropout test_philox_dropout.cpp)
target_link_libraries(test_philox_dropout PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_philox_dropout LABELS "unit" DISCOVERY_TIMEOUT 60)

# Sparse Tensor Tests
add_executable(test_sparse test_sparse.cpp)
target_link_libraries(test_sparse PRIVATE tenzor_core GTest::gtest_main)

# Sparse Autograd Tests
add_executable(test_sparse_autograd test_sparse_autograd.cpp)
target_link_libraries(test_sparse_autograd PRIVATE tenzor_core GTest::gtest_main)

# Complex Arithmetic Tests
add_executable(test_complex_arithmetic test_complex_arithmetic.cpp)
target_link_libraries(test_complex_arithmetic PRIVATE tenzor_core GTest::gtest_main)

# FFT Tests
add_executable(test_fft test_fft.cpp)
target_link_libraries(test_fft PRIVATE tenzor_core GTest::gtest_main)

# Linear Algebra Tests
add_executable(test_linalg test_linalg.cpp)
target_link_libraries(test_linalg PRIVATE tenzor_core GTest::gtest_main)

# Linear Algebra — advanced routines (lstsq, pinv, matrix_exp)
add_executable(test_linalg_lstsq_pinv_matexp test_linalg_lstsq_pinv_matexp.cpp)
target_link_libraries(test_linalg_lstsq_pinv_matexp PRIVATE tenzor_core GTest::gtest_main)

# FFT — fftshift / hfft / ihfft
add_executable(test_fft_shift_hfft test_fft_shift_hfft.cpp)
target_link_libraries(test_fft_shift_hfft PRIVATE tenzor_core GTest::gtest_main)

# Empty Tensor Operations Tests (systematic coverage)
add_executable(test_empty_tensor_ops test_empty_tensor_ops.cpp)
target_link_libraries(test_empty_tensor_ops PRIVATE tenzor_core GTest::gtest_main)

# ============================================================================
# CUDA-specific unit tests (require CUDA)
# ============================================================================
if(TENZOR_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)

    # DataParallel tests (requires CUDA)
    add_executable(test_data_parallel test_data_parallel.cpp)
    target_link_libraries(test_data_parallel PRIVATE
        tenzor_core tenzor_backend_cuda GTest::gtest_main CUDA::cudart)
    target_include_directories(test_data_parallel PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

    # DataParallel single-GPU tests (comprehensive)
    add_executable(test_data_parallel_single_gpu test_data_parallel_single_gpu.cpp)
    target_link_libraries(test_data_parallel_single_gpu PRIVATE
        tenzor_core tenzor_backend_cuda GTest::gtest_main CUDA::cudart)
    target_include_directories(test_data_parallel_single_gpu PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(test_data_parallel_single_gpu PRIVATE TENZOR_USE_CUDA)
endif()

# ============================================================================
# Test Registration
# ============================================================================
tenzor_discover_tests(test_embedding DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_embedding_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_optimizers_extended DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_schedulers_advanced DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_losses_advanced DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_rnn DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_rnn_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_lstm DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_lstm_proj_size DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_lstm_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_lstm_cuda_forward_regression DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_attention_sdpa_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_float16_regression DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_gru DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_gru_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_attention DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_attention_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_transformer DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_transformer_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_chunk DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_grad_scaler DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_grad_scaler_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_autocast DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_autocast_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_mixed_precision DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_dataloader DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_fp16 DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_gradient_checkpoint DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_checkpoint_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_model_checkpoint DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_checkpoint DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_fused_ops DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_fused_ops_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_tensor_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_autograd_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_broadcasting DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_broadcasting_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_ops_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_comparison_ops_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_detection_ops_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_optimizers_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_linear_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_backend_ops_parameterized DISCOVERY_TIMEOUT 120)
tenzor_discover_tests(test_edge_cases DISCOVERY_TIMEOUT 120)
tenzor_discover_tests(test_dtype_edge_cases DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_caching_allocator DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_allocator_overflow DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_empty_tensors DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_view_slice_semantics DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_broadcasting_edge_cases DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_numerical_precision DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_dtype_conversion_chains DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_losses_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_fp16_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_mixed_precision_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_bert_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_vit_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_unet_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_yolo_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_mask_rcnn_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_faster_rcnn_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_swin_transformer_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_efficientnet_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_comparison_ops DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_onnx_export DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_onnx_roundtrip DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_model_hub DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_bert DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_classic_models DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_gpt DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_gpt_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_resnet DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_resnet_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_roberta DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_electra DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_electra_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_efficientnet DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_vit DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 600)
tenzor_discover_tests(test_convnext DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_convnext_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_mobilenet_v2_v3 DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_roberta_electra DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_albert_t5 DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 600)
tenzor_discover_tests(test_faster_rcnn DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1200)
tenzor_discover_tests(test_yolo DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_mask_rcnn DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1200)
tenzor_discover_tests(test_unet DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1200)
tenzor_discover_tests(test_deeplabv3plus DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1200)
tenzor_discover_tests(test_deeplabv3plus_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT 1200 BACKENDS multi)
tenzor_discover_tests(test_detection_ops DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_jit DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_jit_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_jit_script DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_pruning DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_pruning_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_distillation DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_distillation_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_callbacks_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_roberta_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_edge_cases_multidtype DISCOVERY_TIMEOUT 120 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_quantization_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_async_ops DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_data_loading DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_inplace_operations_multidtype DISCOVERY_TIMEOUT 60 BACKENDS multi)
tenzor_discover_tests(test_albert_t5_multidtype DISCOVERY_TIMEOUT 60 BACKENDS multi)
tenzor_discover_tests(test_roberta_electra_multidtype DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_mobilenet_v2_v3_multidtype DISCOVERY_TIMEOUT 60 BACKENDS multi)
tenzor_discover_tests(test_classic_models_multidtype DISCOVERY_TIMEOUT 60 BACKENDS multi)
tenzor_discover_tests(test_vision_components_multidtype DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_detection_components_multidtype DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_losses_advanced_multidtype DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_schedulers_advanced_multidtype DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_optimizers_extended_multidtype DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_nn_additional_multidtype DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_ops_additional_multidtype DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_chunk_multidtype DISCOVERY_TIMEOUT 60 BACKENDS multi)
tenzor_discover_tests(test_gradient_checkpoint_multidtype DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_model_checkpoint_multidtype DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_transforms_multidtype DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_async_ops_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
tenzor_discover_tests(test_dataloader_multidtype DISCOVERY_TIMEOUT 60 BACKENDS multi)
tenzor_discover_tests(test_dtype_edge_cases_multidtype DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_sparse DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_sparse_autograd DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_complex_arithmetic DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_fft DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_linalg DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_linalg_lstsq_pinv_matexp DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_fft_shift_hfft DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_empty_tensor_ops DISCOVERY_TIMEOUT 60)

if(TENZOR_BUILD_CUDA)
    tenzor_discover_tests(test_data_parallel DISCOVERY_TIMEOUT 60)
    tenzor_discover_tests(test_data_parallel_single_gpu DISCOVERY_TIMEOUT 60)
endif()

# Dispatch interceptor tests
add_executable(test_dispatch_interceptor test_dispatch_interceptor.cpp)
target_link_libraries(test_dispatch_interceptor PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dispatch_interceptor DISCOVERY_TIMEOUT 60)

# ============================================================================
# Phase 3: Multi-backend Tier 2 unit tests
# ============================================================================

add_executable(test_broadcasting_edge_cases_multidtype test_broadcasting_edge_cases_multidtype.cpp)
target_link_libraries(test_broadcasting_edge_cases_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_broadcasting_edge_cases_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_empty_tensor_ops_multidtype test_empty_tensor_ops_multidtype.cpp)
target_link_libraries(test_empty_tensor_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_empty_tensor_ops_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_empty_tensors_multidtype test_empty_tensors_multidtype.cpp)
target_link_libraries(test_empty_tensors_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_empty_tensors_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_view_slice_semantics_multidtype test_view_slice_semantics_multidtype.cpp)
target_link_libraries(test_view_slice_semantics_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_view_slice_semantics_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_dispatch_interceptor_multidtype test_dispatch_interceptor_multidtype.cpp)
target_link_libraries(test_dispatch_interceptor_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dispatch_interceptor_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_dtype_conversion_chains_multidtype test_dtype_conversion_chains_multidtype.cpp)
target_link_libraries(test_dtype_conversion_chains_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dtype_conversion_chains_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

add_executable(test_float16_regression_multidtype test_float16_regression_multidtype.cpp)
target_link_libraries(test_float16_regression_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_float16_regression_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_numerical_precision_multidtype test_numerical_precision_multidtype.cpp)
target_link_libraries(test_numerical_precision_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_numerical_precision_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

add_executable(test_graph_optimizer_multidtype test_graph_optimizer_multidtype.cpp)
target_link_libraries(test_graph_optimizer_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_graph_optimizer_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# ============================================================================
# Previously orphaned test files (registered in audit cleanup)
# Note: 3 orphan files (test_nn_additional, test_detection_components,
# test_caching_allocator_performance) had rotted APIs and were removed.
# ============================================================================

# Graph-level kernel fusion optimizer (recreated as fresh test in follow-up cleanup)
add_executable(test_fusion_optimizer test_fusion_optimizer.cpp)
target_link_libraries(test_fusion_optimizer PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fusion_optimizer LABELS "unit" DISCOVERY_TIMEOUT 60)

# test_cublas_cudnn.cpp orphan removed: rotted APIs (Tensor::ones/randn/matmul methods no longer exist)

# Quantization observers and FakeQuantize specifics (16 tests, complements canonical test_quantization)
add_executable(test_quantization_observers test_quantization.cpp)
target_link_libraries(test_quantization_observers PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_quantization_observers LABELS "unit" DISCOVERY_TIMEOUT 60)

# test_oneapi_backend.cpp orphan removed: rotted APIs (relu/gelu now require Variable not Tensor)

# ============================================================================
# New feature tests
# ============================================================================

# Scatter variants (select_scatter, slice_scatter, diagonal_scatter)
add_executable(test_scatter_variants test_scatter_variants.cpp)
target_link_libraries(test_scatter_variants PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_scatter_variants LABELS "unit" DISCOVERY_TIMEOUT 60)

# Hardtanh activation
add_executable(test_hardtanh test_hardtanh.cpp)
target_link_libraries(test_hardtanh PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_hardtanh LABELS "unit" DISCOVERY_TIMEOUT 60)

# Autograd HVP and VHP
add_executable(test_autograd_hvp_vhp test_autograd_hvp_vhp.cpp)
target_link_libraries(test_autograd_hvp_vhp PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_autograd_hvp_vhp LABELS "unit" DISCOVERY_TIMEOUT 60)

# Extended linalg (vector_norm, matrix_norm, vecdot, householder_product, ldl)
add_executable(test_linalg_extended test_linalg_extended.cpp)
target_link_libraries(test_linalg_extended PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_linalg_extended LABELS "unit" DISCOVERY_TIMEOUT 60)

# ASGD optimizer
add_executable(test_asgd test_asgd.cpp)
target_link_libraries(test_asgd PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_asgd LABELS "unit" DISCOVERY_TIMEOUT 60)

# Rprop optimizer
add_executable(test_rprop test_rprop.cpp)
target_link_libraries(test_rprop PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_rprop LABELS "unit" DISCOVERY_TIMEOUT 60)

# Training metrics (Accuracy, Precision, Recall, F1Score, MAE, MSE)
add_executable(test_training_metrics test_metrics.cpp)
target_link_libraries(test_training_metrics PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_training_metrics LABELS "unit" DISCOVERY_TIMEOUT 60)

# Training metrics multi-backend × multi-dtype (audit-2026-05-03 N1)
add_executable(test_metrics_multidtype test_metrics_multidtype.cpp)
target_link_libraries(test_metrics_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_metrics_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Image I/O (stb_image wrapper) — CPU-only, commit ca264523
add_executable(test_image_io test_image_io.cpp)
target_link_libraries(test_image_io PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_image_io LABELS "unit" DISCOVERY_TIMEOUT 60)

# ============================================================================
# Batch A - Linalg + Math Multi-DType Tests
# ============================================================================

# Linalg core multi-dtype (det, inv, solve, cholesky, qr, svd, eigh)
add_executable(test_linalg_multidtype test_linalg_multidtype.cpp)
target_link_libraries(test_linalg_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_linalg_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Linalg extended multi-dtype (vector_norm, matrix_norm, vecdot, ldl)
add_executable(test_linalg_extended_multidtype test_linalg_extended_multidtype.cpp)
target_link_libraries(test_linalg_extended_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_linalg_extended_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Linalg lstsq/pinv/matrix_exp multi-dtype
add_executable(test_linalg_lstsq_pinv_matexp_multidtype test_linalg_lstsq_pinv_matexp_multidtype.cpp)
target_link_libraries(test_linalg_lstsq_pinv_matexp_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_linalg_lstsq_pinv_matexp_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Matrix construction multi-dtype (kron, block_diag, vander, combinations)
add_executable(test_matrix_construction_multidtype test_matrix_construction_multidtype.cpp)
target_link_libraries(test_matrix_construction_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_matrix_construction_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Special math multi-dtype (gamma, lgamma, beta, bessel, erfinv, sinc)
add_executable(test_unit_special_math_multidtype test_special_math_multidtype.cpp)
target_link_libraries(test_unit_special_math_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_unit_special_math_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Complex arithmetic multi-dtype (creation, add, mul, div, abs, exp, sqrt)
add_executable(test_complex_arithmetic_multidtype test_complex_arithmetic_multidtype.cpp)
target_link_libraries(test_complex_arithmetic_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_complex_arithmetic_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# FFT multi-dtype (fft, ifft, rfft, irfft, batched, padding)
add_executable(test_fft_multidtype test_fft_multidtype.cpp)
target_link_libraries(test_fft_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fft_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# FFT shift/hfft multi-dtype (fftshift, ifftshift, hfft, ihfft)
add_executable(test_fft_shift_hfft_multidtype test_fft_shift_hfft_multidtype.cpp)
target_link_libraries(test_fft_shift_hfft_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fft_shift_hfft_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# ============================================================================
# Batch B - Operations + Sparse Multi-DType Tests
# ============================================================================

# Scatter variants multi-dtype (select_scatter, slice_scatter, diagonal_scatter)
add_executable(test_scatter_variants_multidtype test_scatter_variants_multidtype.cpp)
target_link_libraries(test_scatter_variants_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_scatter_variants_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Sparse multi-dtype (COO, CSR, to_dense, SpMM, SpMV, add, coalesce)
add_executable(test_sparse_multidtype test_sparse_multidtype.cpp)
target_link_libraries(test_sparse_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sparse_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Sparse autograd multi-dtype (SpMM/SpMV/SparseAdd backward)
add_executable(test_sparse_autograd_multidtype test_sparse_autograd_multidtype.cpp)
target_link_libraries(test_sparse_autograd_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sparse_autograd_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Hardtanh activation multi-dtype
add_executable(test_hardtanh_multidtype test_hardtanh_multidtype.cpp)
target_link_libraries(test_hardtanh_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_hardtanh_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Data loading multi-dtype (TensorDataset, TransformedDataset, ConcatDataset)
add_executable(test_data_loading_multidtype test_data_loading_multidtype.cpp)
target_link_libraries(test_data_loading_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_data_loading_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# ============================================================================
# Batch C - Models + Optimizers Multi-DType Tests
# ============================================================================

# ASGD optimizer multi-dtype
add_executable(test_asgd_multidtype test_asgd_multidtype.cpp)
target_link_libraries(test_asgd_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_asgd_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Rprop optimizer multi-dtype
add_executable(test_rprop_multidtype test_rprop_multidtype.cpp)
target_link_libraries(test_rprop_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_rprop_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Fusion optimizer multi-dtype (graph construction, pattern matching)
add_executable(test_fusion_optimizer_multidtype test_fusion_optimizer_multidtype.cpp)
target_link_libraries(test_fusion_optimizer_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fusion_optimizer_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# JIT script multi-dtype (compilation, evaluation)
add_executable(test_jit_script_multidtype test_jit_script_multidtype.cpp)
target_link_libraries(test_jit_script_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jit_script_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Autograd HVP/VHP multi-dtype (Hessian-vector products)
add_executable(test_autograd_hvp_vhp_multidtype test_autograd_hvp_vhp_multidtype.cpp)
target_link_libraries(test_autograd_hvp_vhp_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_autograd_hvp_vhp_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# ============================================================================
# Batch D - Distributed Multi-DType Tests
# ============================================================================

# DataParallel multi-dtype (multi-device forward, construction)
add_executable(test_data_parallel_multidtype test_data_parallel_multidtype.cpp)
target_link_libraries(test_data_parallel_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_data_parallel_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Device multi-dtype (device creation, tensor transfer, roundtrip)
add_executable(test_device_multidtype test_device_multidtype.cpp)
target_link_libraries(test_device_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_device_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Image I/O multi-dtype (tensor creation from image, shape preservation, roundtrip)
add_executable(test_image_io_multidtype test_image_io_multidtype.cpp)
target_link_libraries(test_image_io_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_image_io_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Backend ops parameterized multi-dtype (add, sub, mul, div, matmul, reductions, transforms)
add_executable(test_backend_ops_parameterized_multidtype test_backend_ops_parameterized_multidtype.cpp)
target_link_libraries(test_backend_ops_parameterized_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_backend_ops_parameterized_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# DataParallel single-GPU multi-dtype (wrapping, forward, shape, gradient flow)
add_executable(test_data_parallel_single_gpu_multidtype test_data_parallel_single_gpu_multidtype.cpp)
target_link_libraries(test_data_parallel_single_gpu_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_data_parallel_single_gpu_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# DDP unused params multi-dtype (API verification, GradBucket, buffer simulation)
add_executable(test_ddp_unused_params_multidtype test_ddp_unused_params_multidtype.cpp)
target_link_libraries(test_ddp_unused_params_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_ddp_unused_params_multidtype LABELS "unit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)
