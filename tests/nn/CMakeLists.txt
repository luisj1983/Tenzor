# =============================================================================
# tests/nn/CMakeLists.txt — Neural network layer and optimizer tests
# =============================================================================

# Dropout tests
add_executable(test_dropout layers/test_dropout.cpp)
target_link_libraries(test_dropout PRIVATE tenzor_core GTest::gtest_main)

# BatchNorm2d tests
add_executable(test_batchnorm2d layers/test_batchnorm2d.cpp)
target_link_libraries(test_batchnorm2d PRIVATE tenzor_core GTest::gtest_main)

# BatchNorm1d tests
add_executable(test_batchnorm1d layers/test_batchnorm1d.cpp)
target_link_libraries(test_batchnorm1d PRIVATE tenzor_core GTest::gtest_main)

# BatchNorm3d tests
add_executable(test_batchnorm3d layers/test_batchnorm3d.cpp)
target_link_libraries(test_batchnorm3d PRIVATE tenzor_core GTest::gtest_main)

# Conv2d multi-dtype tests (the CPU-only test_conv2d.cpp was deleted — its
# 54 test cases all have exact-named counterparts in the _multidtype file,
# which also runs them across every backend × dtype combination).
add_executable(test_conv2d_multidtype layers/test_conv2d_multidtype.cpp)
target_link_libraries(test_conv2d_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Pooling tests
add_executable(test_pooling layers/test_pooling.cpp)
target_link_libraries(test_pooling PRIVATE tenzor_core GTest::gtest_main)

# Scheduler tests
add_executable(test_schedulers optim/test_schedulers.cpp)
target_link_libraries(test_schedulers PRIVATE tenzor_core GTest::gtest_main)

# ZeRO Stage 1 Optimizer tests (Phase 4)
add_executable(test_zero_stage1 optim/test_zero_stage1.cpp)
target_link_libraries(test_zero_stage1 PRIVATE tenzor_core GTest::gtest_main)

# ZeRO Stage 1 Integration tests
add_executable(test_zero_stage1_integration optim/test_zero_stage1_integration.cpp)
target_link_libraries(test_zero_stage1_integration PRIVATE tenzor_core GTest::gtest_main)

# ZeRO Stage 1 Distributed tests
add_executable(test_zero_stage1_distributed optim/test_zero_stage1_distributed.cpp)
target_link_libraries(test_zero_stage1_distributed PRIVATE tenzor_core GTest::gtest_main)

# Gradient Utilities tests (ZeRO Stage 2)
add_executable(test_gradient_utils optim/test_gradient_utils.cpp)
target_link_libraries(test_gradient_utils PRIVATE tenzor_core GTest::gtest_main)

# Lion optimizer (Phase 5.2)
add_executable(test_lion optim/test_lion.cpp)
target_link_libraries(test_lion PRIVATE tenzor_core GTest::gtest_main)

# L-BFGS optimizer (strong Wolfe line search)
add_executable(test_lbfgs optim/test_lbfgs.cpp)
target_link_libraries(test_lbfgs PRIVATE tenzor_core GTest::gtest_main)

# NAdam optimizer (Nesterov-accelerated Adam)
add_executable(test_nadam optim/test_nadam.cpp)
target_link_libraries(test_nadam PRIVATE tenzor_core GTest::gtest_main)

# Adamax optimizer (Adam with infinity-norm denominator)
add_executable(test_adamax optim/test_adamax.cpp)
target_link_libraries(test_adamax PRIVATE tenzor_core GTest::gtest_main)

# AdamW optimizer (Adam + decoupled weight decay) — dedicated unit tests.
# Before this target, AdamW was only covered by an integration test.
add_executable(test_adamw optim/test_adamw.cpp)
target_link_libraries(test_adamw PRIVATE tenzor_core GTest::gtest_main)

# ZeRO Stage 2 Optimizer tests (Phase 5 - Gradient Partitioning)
add_executable(test_zero_stage2 optim/test_zero_stage2.cpp)
target_link_libraries(test_zero_stage2 PRIVATE tenzor_core GTest::gtest_main)

# ZeRO Stage 2 Integration tests
add_executable(test_zero_stage2_integration optim/test_zero_stage2_integration.cpp)
target_link_libraries(test_zero_stage2_integration PRIVATE tenzor_core GTest::gtest_main)

# ZeRO Stage 3 Optimizer tests (Phase 6 - Parameter Partitioning)
add_executable(test_zero_stage3 optim/test_zero_stage3.cpp)
target_link_libraries(test_zero_stage3 PRIVATE tenzor_core GTest::gtest_main)

# ZeRO Stage 3 Integration tests (Phase 6 - Parameter Partitioning)
add_executable(test_zero_stage3_integration optim/test_zero_stage3_integration.cpp)
target_link_libraries(test_zero_stage3_integration PRIVATE tenzor_core GTest::gtest_main)

# ZeRO Profiling tests (Phase 7 - Performance Profiling)
add_executable(test_zero_profiling optim/test_zero_profiling.cpp)
target_link_libraries(test_zero_profiling PRIVATE tenzor_core GTest::gtest_main)

# Normalization tests
add_executable(test_normalization layers/test_normalization.cpp)
target_link_libraries(test_normalization PRIVATE tenzor_core GTest::gtest_main)

# Serialization tests
add_executable(test_serialization test_serialization.cpp)
target_link_libraries(test_serialization PRIVATE tenzor_core GTest::gtest_main)

# Offload tests (ZeRO Stage 2 Parameter Offloading)
add_executable(test_offload test_offload.cpp)
target_link_libraries(test_offload PRIVATE tenzor_core GTest::gtest_main)

# NN Layer Multi-DType Tests
add_executable(test_pooling_multidtype layers/test_pooling_multidtype.cpp)
target_link_libraries(test_pooling_multidtype PRIVATE tenzor_core GTest::gtest_main)

add_executable(test_dropout_multidtype layers/test_dropout_multidtype.cpp)
target_link_libraries(test_dropout_multidtype PRIVATE tenzor_core GTest::gtest_main)

add_executable(test_batchnorm2d_multidtype layers/test_batchnorm2d_multidtype.cpp)
target_link_libraries(test_batchnorm2d_multidtype PRIVATE tenzor_core GTest::gtest_main)

add_executable(test_batchnorm1d_multidtype layers/test_batchnorm1d_multidtype.cpp)
target_link_libraries(test_batchnorm1d_multidtype PRIVATE tenzor_core GTest::gtest_main)

add_executable(test_batchnorm3d_multidtype layers/test_batchnorm3d_multidtype.cpp)
target_link_libraries(test_batchnorm3d_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Bilinear layer (Phase 7.2)
add_executable(test_bilinear_multidtype layers/test_bilinear_multidtype.cpp)
target_link_libraries(test_bilinear_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Interpolate backward (Phase 7.1) — bilinear-mode autograd through interpolate()
add_executable(test_interpolate_backward layers/test_interpolate_backward.cpp)
target_link_libraries(test_interpolate_backward PRIVATE tenzor_core GTest::gtest_main)

add_executable(test_normalization_multidtype layers/test_normalization_multidtype.cpp)
target_link_libraries(test_normalization_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Standalone LayerNorm tests (complements test_normalization_multidtype)
add_executable(test_layernorm_multidtype layers/test_layernorm_multidtype.cpp)
target_link_libraries(test_layernorm_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_layernorm_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Distance layer tests (CosineSimilarity, PairwiseDistance) — multi-backend.
add_executable(test_distance_layers layers/test_distance_layers.cpp)
target_link_libraries(test_distance_layers PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_distance_layers LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Standalone MultiheadAttention layer tests (complements test_mha_functional)
add_executable(test_multihead_attention_multidtype layers/test_multihead_attention_multidtype.cpp)
target_link_libraries(test_multihead_attention_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_multihead_attention_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Flatten layer multi-dtype tests
add_executable(test_flatten_multidtype layers/test_flatten_multidtype.cpp)
target_link_libraries(test_flatten_multidtype PRIVATE tenzor_core GTest::gtest_main)

# ChannelShuffle tests
add_executable(test_channel_shuffle layers/test_channel_shuffle.cpp)
target_link_libraries(test_channel_shuffle PRIVATE tenzor_core GTest::gtest_main)

# Quantized inference tests
add_executable(test_quantized_inference test_quantized_inference.cpp)
target_link_libraries(test_quantized_inference PRIVATE tenzor_core GTest::gtest_main)

# Segmentation operations multi-dtype tests
add_executable(test_segmentation_multidtype layers/test_segmentation_multidtype.cpp)
target_link_libraries(test_segmentation_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Hierarchical Reasoning Model (HRM) Tests
add_executable(test_hrm layers/test_hrm.cpp)
target_link_libraries(test_hrm PRIVATE tenzor_core GTest::gtest_main)

# Gated Activations (GeGLU, ReGLU) Tests
add_executable(test_gated_activations layers/test_gated_activations.cpp)
target_link_libraries(test_gated_activations PRIVATE tenzor_core GTest::gtest_main)

# Mixture of Experts multi-dtype tests (the CPU-only test_moe.cpp was
# deleted — multidtype sibling supersedes it with BackwardGradPopulated and
# OutputFinite checks plus multi-backend coverage).

# Sliding Window Attention Tests
add_executable(test_sliding_window_attention layers/test_sliding_window_attention.cpp)
target_link_libraries(test_sliding_window_attention PRIVATE tenzor_core GTest::gtest_main)

# ============================================================================
# Test Registration
# ============================================================================
tenzor_discover_tests(test_dropout LABELS "nn\;smoke" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_batchnorm2d LABELS "nn\;smoke" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_batchnorm1d LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_batchnorm3d LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_conv2d_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_pooling LABELS "nn\;smoke" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_schedulers LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_zero_stage1 LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_zero_stage1_integration LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_zero_stage1_distributed LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_gradient_utils LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_lion LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_lbfgs LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_nadam LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_adamax LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_adamw LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_zero_stage2 LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_zero_stage2_integration LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_zero_stage3 LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_zero_stage3_integration LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_zero_profiling LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_normalization LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_serialization LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_offload LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_pooling_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_dropout_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_batchnorm2d_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_batchnorm1d_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_bilinear_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_interpolate_backward LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_batchnorm3d_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_normalization_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_flatten_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_channel_shuffle LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_quantized_inference LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_segmentation_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
# PyTorch loader tests
add_executable(test_pytorch_loader test_pytorch_loader.cpp)
target_link_libraries(test_pytorch_loader PRIVATE tenzor_core GTest::gtest_main)

# FSDP tests
add_executable(test_fsdp test_fsdp.cpp)
target_link_libraries(test_fsdp PRIVATE tenzor_core GTest::gtest_main)

# FSDP2 (audit 2026-05-02 F.2 — zero test references for the v2 wrapper)
add_executable(test_fsdp2 test_fsdp2.cpp)
target_link_libraries(test_fsdp2 PRIVATE tenzor_core GTest::gtest_main)

# Anchor generator (audit 2026-05-02 F.3 — zero test references)
add_executable(test_anchor_generator test_anchor_generator.cpp)
target_link_libraries(test_anchor_generator PRIVATE tenzor_core GTest::gtest_main)

add_executable(test_anchor_generator_multidtype test_anchor_generator_multidtype.cpp)
target_link_libraries(test_anchor_generator_multidtype PRIVATE tenzor_core GTest::gtest_main)

# KnowledgeDistillation (audit 2026-05-02 F.4 — zero test references)
add_executable(test_knowledge_distillation test_knowledge_distillation.cpp)
target_link_libraries(test_knowledge_distillation PRIVATE tenzor_core GTest::gtest_main)

add_executable(test_knowledge_distillation_multidtype test_knowledge_distillation_multidtype.cpp)
target_link_libraries(test_knowledge_distillation_multidtype PRIVATE tenzor_core GTest::gtest_main)

# Tensor Parallelism tests
add_executable(test_tensor_parallel test_tensor_parallel.cpp)
target_link_libraries(test_tensor_parallel PRIVATE tenzor_core GTest::gtest_main)

# Functional API tests
add_executable(test_functional test_functional.cpp)
target_link_libraries(test_functional PRIVATE tenzor_core GTest::gtest_main)

# SafeTensors format tests
add_executable(test_safetensors test_safetensors.cpp)
target_link_libraries(test_safetensors PRIVATE tenzor_core GTest::gtest_main)

tenzor_discover_tests(test_hrm LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_pytorch_loader LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_fsdp LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_fsdp2 LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_anchor_generator LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_anchor_generator_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_knowledge_distillation LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_knowledge_distillation_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_tensor_parallel LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_functional LABELS "nn" DISCOVERY_TIMEOUT 120)
tenzor_discover_tests(test_safetensors LABELS "nn" DISCOVERY_TIMEOUT 120)

# Parametrize module tests
add_executable(test_parametrize test_parametrize.cpp)
target_link_libraries(test_parametrize PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_parametrize LABELS "nn" DISCOVERY_TIMEOUT 60)

# ============================================================================
# New module tests (Phase 2)
# ============================================================================

# Distributions multi-backend tests
add_executable(test_distributions_multidtype test_distributions_multidtype.cpp)
target_link_libraries(test_distributions_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_distributions_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Contrastive losses multi-backend tests (audit-2026-05-03 N3)
add_executable(test_contrastive_losses_multidtype test_contrastive_losses_multidtype.cpp)
target_link_libraries(test_contrastive_losses_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_contrastive_losses_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Weight initialization multi-backend tests
add_executable(test_init_multidtype test_init_multidtype.cpp)
target_link_libraries(test_init_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_init_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Gradient clipping multi-backend tests
add_executable(test_clip_grad_multidtype test_clip_grad_multidtype.cpp)
target_link_libraries(test_clip_grad_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_clip_grad_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Spectral normalization multi-backend tests
add_executable(test_spectral_norm_multidtype test_spectral_norm_multidtype.cpp)
target_link_libraries(test_spectral_norm_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_spectral_norm_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Weight normalization multi-backend tests
add_executable(test_weight_norm_multidtype test_weight_norm_multidtype.cpp)
target_link_libraries(test_weight_norm_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_weight_norm_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# ============================================================================
# Phase 4: New tests for previously untested NN components
# ============================================================================

# Padding layers (ConstantPad1d/2d/3d, ReflectionPad1d/2d, ReplicationPad1d/2d/3d, ZeroPad2d)
add_executable(test_padding_multidtype layers/test_padding_multidtype.cpp)
target_link_libraries(test_padding_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_padding_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Missing optimizers (RAdam, LAMB, AdamAtan2, SparseAdam, Adadelta)
add_executable(test_optimizers_missing_multidtype optim/test_optimizers_missing_multidtype.cpp)
target_link_libraries(test_optimizers_missing_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_optimizers_missing_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Dropout variants (AlphaDropout, VariationalDropout)
add_executable(test_dropout_variants_multidtype layers/test_dropout_variants_multidtype.cpp)
target_link_libraries(test_dropout_variants_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dropout_variants_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# SyncBatchNorm
add_executable(test_sync_batchnorm layers/test_sync_batchnorm.cpp)
target_link_libraries(test_sync_batchnorm PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sync_batchnorm LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Pipeline parallelism (recreated as fresh test in follow-up cleanup)
# The original orphan was removed because PipelineStage symbols weren't built;
# pipeline_parallel.cpp has now been added to src/CMakeLists.txt and the symbols are exported.
add_executable(test_pipeline_parallel test_pipeline_parallel.cpp)
target_link_libraries(test_pipeline_parallel PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_pipeline_parallel LABELS "nn" DISCOVERY_TIMEOUT 60)

# Missing activation layers (Hardsigmoid, Softmin, Softshrink, Tanhshrink)
add_executable(test_activation_missing_multidtype layers/test_activation_missing_multidtype.cpp)
target_link_libraries(test_activation_missing_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_activation_missing_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

tenzor_discover_tests(test_gated_activations LABELS "nn" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_sliding_window_attention LABELS "nn" DISCOVERY_TIMEOUT 60)

# Multi-head attention functional tests
add_executable(test_mha_functional test_mha_functional.cpp)
target_link_libraries(test_mha_functional PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_mha_functional LABELS "nn" DISCOVERY_TIMEOUT 60)

# SAM optimizer and SWA utilities tests
add_executable(test_sam_swa optim/test_sam_swa.cpp)
target_link_libraries(test_sam_swa PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sam_swa LABELS "nn" DISCOVERY_TIMEOUT 60)

# tests/nn/test_data_parallel.cpp orphan removed: rotted include paths (tenzor/nn/modules/ no longer exists)
# tests/nn/test_distributed.cpp orphan removed: requires rccl/rccl.h via TENZOR_USE_ROCM that isn't installed

# ============================================================================
# Previously untested layers
# ============================================================================

add_executable(test_drop_path layers/test_drop_path.cpp)
target_link_libraries(test_drop_path PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_drop_path LABELS "nn" DISCOVERY_TIMEOUT 60)

add_executable(test_dropout3d layers/test_dropout3d.cpp)
target_link_libraries(test_dropout3d PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dropout3d LABELS "nn" DISCOVERY_TIMEOUT 60)

add_executable(test_window_attention layers/test_window_attention.cpp)
target_link_libraries(test_window_attention PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_window_attention LABELS "nn" DISCOVERY_TIMEOUT 60)

add_executable(test_cosine_similarity_layer layers/test_cosine_similarity_layer.cpp)
target_link_libraries(test_cosine_similarity_layer PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_cosine_similarity_layer LABELS "nn" DISCOVERY_TIMEOUT 60)

add_executable(test_sparse_linear layers/test_sparse_linear.cpp)
target_link_libraries(test_sparse_linear PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sparse_linear LABELS "nn" DISCOVERY_TIMEOUT 60)

add_executable(test_sparse_embedding layers/test_sparse_embedding.cpp)
target_link_libraries(test_sparse_embedding PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sparse_embedding LABELS "nn" DISCOVERY_TIMEOUT 60)

add_executable(test_fused_mbconv layers/test_fused_mbconv.cpp)
target_link_libraries(test_fused_mbconv PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fused_mbconv LABELS "nn" DISCOVERY_TIMEOUT 60)

add_executable(test_lazy_backward layers/test_lazy_backward.cpp)
target_link_libraries(test_lazy_backward PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lazy_backward LABELS "nn" DISCOVERY_TIMEOUT 60)

# ============================================================================
# Phase 1.1: Previously untested NN layers
# ============================================================================

# InstanceNorm1d/2d/3d multi-backend tests
add_executable(test_instance_norm_multidtype layers/test_instance_norm_multidtype.cpp)
target_link_libraries(test_instance_norm_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_instance_norm_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# ALiBi multi-backend tests
add_executable(test_alibi_multidtype layers/test_alibi_multidtype.cpp)
target_link_libraries(test_alibi_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_alibi_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Phase 2.2: layers previously without dedicated multi-backend test files.
add_executable(test_identity_multidtype layers/test_identity_multidtype.cpp)
target_link_libraries(test_identity_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_identity_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_circular_pad_multidtype layers/test_circular_pad_multidtype.cpp)
target_link_libraries(test_circular_pad_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_circular_pad_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_patch_embedding_multidtype layers/test_patch_embedding_multidtype.cpp)
target_link_libraries(test_patch_embedding_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_patch_embedding_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_distance_layers_multidtype layers/test_distance_layers_multidtype.cpp)
target_link_libraries(test_distance_layers_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_distance_layers_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_deformable_conv2d_layer_multidtype layers/test_deformable_conv2d_layer_multidtype.cpp)
target_link_libraries(test_deformable_conv2d_layer_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_deformable_conv2d_layer_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_transformer_layer_multidtype layers/test_transformer_layer_multidtype.cpp)
target_link_libraries(test_transformer_layer_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_transformer_layer_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# FlexAttention multi-backend tests (forward-only; backward deferred — the
# backward pass requires native GPU flash-attention-style kernels which live
# in the Phase 3 follow-up of the test-completion plan).
add_executable(test_flex_attention_multidtype layers/test_flex_attention_multidtype.cpp)
target_link_libraries(test_flex_attention_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_flex_attention_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# LPPool1d / LPPool2d multi-backend multi-dtype tests (Phase 2 gap fill)
add_executable(test_lppool_multidtype layers/test_lppool_multidtype.cpp)
target_link_libraries(test_lppool_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lppool_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# LocalResponseNorm multi-backend multi-dtype tests (Phase 2 gap fill)
add_executable(test_local_response_norm_multidtype layers/test_local_response_norm_multidtype.cpp)
target_link_libraries(test_local_response_norm_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_local_response_norm_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Phase 3 promotions: layers previously in backend_parity only
add_executable(test_activations_extended_multidtype layers/test_activations_extended_multidtype.cpp)
target_link_libraries(test_activations_extended_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_activations_extended_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_convtranspose_extended_multidtype layers/test_convtranspose_extended_multidtype.cpp)
target_link_libraries(test_convtranspose_extended_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_convtranspose_extended_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_upsample_unflatten_multidtype layers/test_upsample_unflatten_multidtype.cpp)
target_link_libraries(test_upsample_unflatten_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_upsample_unflatten_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# ============================================================================
# Phase 1.2: Detection head tests
# ============================================================================

# RPNHead multi-backend tests
add_executable(test_rpn_head_multidtype layers/test_rpn_head_multidtype.cpp)
target_link_libraries(test_rpn_head_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_rpn_head_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# RoIHead multi-backend tests
add_executable(test_roi_head_multidtype layers/test_roi_head_multidtype.cpp)
target_link_libraries(test_roi_head_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_roi_head_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# MaskHead multi-backend tests
add_executable(test_mask_head_multidtype layers/test_mask_head_multidtype.cpp)
target_link_libraries(test_mask_head_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_mask_head_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Phase 1.3: Quantization tests
add_executable(test_awq_quantizer quantization/test_awq_quantizer.cpp)
target_link_libraries(test_awq_quantizer PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_awq_quantizer LABELS "nn" DISCOVERY_TIMEOUT 60)

add_executable(test_gptq_quantizer quantization/test_gptq_quantizer.cpp)
target_link_libraries(test_gptq_quantizer PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_gptq_quantizer LABELS "nn" DISCOVERY_TIMEOUT 60)

add_executable(test_observers_extended quantization/test_observers_extended.cpp)
target_link_libraries(test_observers_extended PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_observers_extended LABELS "nn" DISCOVERY_TIMEOUT 60)

# ============================================================================
# Phase 4.4: Multi-backend conversions for NN tests
# ============================================================================

# NN Root
add_executable(test_functional_multidtype test_functional_multidtype.cpp)
target_link_libraries(test_functional_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_functional_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_mha_functional_multidtype test_mha_functional_multidtype.cpp)
target_link_libraries(test_mha_functional_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_mha_functional_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_parametrize_multidtype test_parametrize_multidtype.cpp)
target_link_libraries(test_parametrize_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_parametrize_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_serialization_multidtype test_serialization_multidtype.cpp)
target_link_libraries(test_serialization_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_serialization_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_safetensors_multidtype test_safetensors_multidtype.cpp)
target_link_libraries(test_safetensors_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_safetensors_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_pytorch_loader_multidtype test_pytorch_loader_multidtype.cpp)
target_link_libraries(test_pytorch_loader_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_pytorch_loader_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_quantized_inference_multidtype test_quantized_inference_multidtype.cpp)
target_link_libraries(test_quantized_inference_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_quantized_inference_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# NN Layers
add_executable(test_channel_shuffle_multidtype layers/test_channel_shuffle_multidtype.cpp)
target_link_libraries(test_channel_shuffle_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_channel_shuffle_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_cosine_similarity_layer_multidtype layers/test_cosine_similarity_layer_multidtype.cpp)
target_link_libraries(test_cosine_similarity_layer_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_cosine_similarity_layer_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_dropout3d_multidtype layers/test_dropout3d_multidtype.cpp)
target_link_libraries(test_dropout3d_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dropout3d_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_drop_path_multidtype layers/test_drop_path_multidtype.cpp)
target_link_libraries(test_drop_path_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_drop_path_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_fold_multidtype layers/test_fold_multidtype.cpp)
target_link_libraries(test_fold_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fold_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_glu_multidtype layers/test_glu_multidtype.cpp)
target_link_libraries(test_glu_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_glu_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_gated_activations_multidtype layers/test_gated_activations_multidtype.cpp)
target_link_libraries(test_gated_activations_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_gated_activations_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_fused_mbconv_multidtype layers/test_fused_mbconv_multidtype.cpp)
target_link_libraries(test_fused_mbconv_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fused_mbconv_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_hrm_multidtype layers/test_hrm_multidtype.cpp)
target_link_libraries(test_hrm_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_hrm_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_lazy_backward_multidtype layers/test_lazy_backward_multidtype.cpp)
target_link_libraries(test_lazy_backward_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lazy_backward_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_moe_multidtype layers/test_moe_multidtype.cpp)
target_link_libraries(test_moe_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_moe_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_sliding_window_attention_multidtype layers/test_sliding_window_attention_multidtype.cpp)
target_link_libraries(test_sliding_window_attention_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sliding_window_attention_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_sparse_embedding_multidtype layers/test_sparse_embedding_multidtype.cpp)
target_link_libraries(test_sparse_embedding_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sparse_embedding_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_sparse_linear_multidtype layers/test_sparse_linear_multidtype.cpp)
target_link_libraries(test_sparse_linear_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sparse_linear_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_sync_batchnorm_multidtype layers/test_sync_batchnorm_multidtype.cpp)
target_link_libraries(test_sync_batchnorm_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sync_batchnorm_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_window_attention_multidtype layers/test_window_attention_multidtype.cpp)
target_link_libraries(test_window_attention_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_window_attention_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# NN Optim
add_executable(test_adamax_multidtype optim/test_adamax_multidtype.cpp)
target_link_libraries(test_adamax_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_adamax_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_lbfgs_multidtype optim/test_lbfgs_multidtype.cpp)
target_link_libraries(test_lbfgs_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lbfgs_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_lion_multidtype optim/test_lion_multidtype.cpp)
target_link_libraries(test_lion_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lion_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_nadam_multidtype optim/test_nadam_multidtype.cpp)
target_link_libraries(test_nadam_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_nadam_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_sam_swa_multidtype optim/test_sam_swa_multidtype.cpp)
target_link_libraries(test_sam_swa_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sam_swa_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_schedulers_multidtype optim/test_schedulers_multidtype.cpp)
target_link_libraries(test_schedulers_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_schedulers_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_gradient_utils_multidtype optim/test_gradient_utils_multidtype.cpp)
target_link_libraries(test_gradient_utils_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_gradient_utils_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# NN Distributed multi-dtype
add_executable(test_fsdp_multidtype test_fsdp_multidtype.cpp)
target_link_libraries(test_fsdp_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fsdp_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_pipeline_parallel_multidtype test_pipeline_parallel_multidtype.cpp)
target_link_libraries(test_pipeline_parallel_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_pipeline_parallel_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_tensor_parallel_multidtype test_tensor_parallel_multidtype.cpp)
target_link_libraries(test_tensor_parallel_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_tensor_parallel_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_offload_multidtype test_offload_multidtype.cpp)
target_link_libraries(test_offload_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_offload_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# NN Optim ZeRO multi-dtype
add_executable(test_zero_stage1_multidtype optim/test_zero_stage1_multidtype.cpp)
target_link_libraries(test_zero_stage1_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_zero_stage1_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_zero_stage2_multidtype optim/test_zero_stage2_multidtype.cpp)
target_link_libraries(test_zero_stage2_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_zero_stage2_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_zero_stage3_multidtype optim/test_zero_stage3_multidtype.cpp)
target_link_libraries(test_zero_stage3_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_zero_stage3_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_zero_stage1_distributed_multidtype optim/test_zero_stage1_distributed_multidtype.cpp)
target_link_libraries(test_zero_stage1_distributed_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_zero_stage1_distributed_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_zero_stage1_integration_multidtype optim/test_zero_stage1_integration_multidtype.cpp)
target_link_libraries(test_zero_stage1_integration_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_zero_stage1_integration_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_zero_stage2_integration_multidtype optim/test_zero_stage2_integration_multidtype.cpp)
target_link_libraries(test_zero_stage2_integration_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_zero_stage2_integration_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_zero_stage3_integration_multidtype optim/test_zero_stage3_integration_multidtype.cpp)
target_link_libraries(test_zero_stage3_integration_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_zero_stage3_integration_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_zero_profiling_multidtype optim/test_zero_profiling_multidtype.cpp)
target_link_libraries(test_zero_profiling_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_zero_profiling_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Lazy layer tests (audit-2026-05-03 N2)
add_executable(test_lazy_layers_multidtype layers/test_lazy_layers_multidtype.cpp)
target_link_libraries(test_lazy_layers_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lazy_layers_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# TransformerEncoder/Decoder container tests (audit-2026-05-03 N2)
add_executable(test_transformer_containers_multidtype layers/test_transformer_containers_multidtype.cpp)
target_link_libraries(test_transformer_containers_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_transformer_containers_multidtype LABELS "nn" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Optimizer grad-flow regression tests (audit-2026-05-03 N1.d)
add_executable(test_optim_grad_flow optim/test_optim_grad_flow.cpp)
target_link_libraries(test_optim_grad_flow PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_optim_grad_flow LABELS "nn" DISCOVERY_TIMEOUT 60)
