# Phase 13 / Group A MLIR JIT tests.
# Note: test_optype_roundtrip does NOT need TENZOR_USE_MLIR_JIT=ON since it
# only tests the tracer.hpp OpType enum which is always compiled in.
add_executable(test_optype_roundtrip test_optype_roundtrip.cpp)
target_link_libraries(test_optype_roundtrip PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_optype_roundtrip DISCOVERY_TIMEOUT 30 BACKENDS none LABELS "jit;mlir")

if(TENZOR_USE_MLIR_JIT)
    # Task A.6′ — MLIR text emission helpers.
    add_executable(test_mlir_text_emit test_mlir_text_emit.cpp)
    target_link_libraries(test_mlir_text_emit
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_mlir_text_emit
        DISCOVERY_TIMEOUT 30 BACKENDS none LABELS "jit;mlir")

    # Task A.7 — IREE Compiler embedding API integration.
    add_executable(test_iree_compile test_iree_compile.cpp)
    target_link_libraries(test_iree_compile
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_iree_compile
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir;iree")

    # Task A.8 — IREE Runtime invocation via iree-run-module subprocess.
    add_executable(test_iree_invoke test_iree_invoke.cpp)
    target_link_libraries(test_iree_invoke
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_iree_invoke
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir;iree")

    # Task A.9 — Custom_call registrar + placeholder callbacks.
    add_executable(test_iree_customcall_smoke test_iree_customcall_smoke.cpp)
    target_link_libraries(test_iree_customcall_smoke
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_iree_customcall_smoke
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir;iree")

    # Task B.1 — Graph → MLIR text lowering (Add only).
    add_executable(test_lower_add test_lower_add.cpp)
    target_link_libraries(test_lower_add
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_lower_add
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir;iree")

    # Task B.2–B.6 — End-to-end x+x via the MLIR backend across IREE
    # targets (llvm-cpu, cuda, vulkan-spirv, rocm), plus cache-hit check.
    add_executable(test_end_to_end_add test_end_to_end_add.cpp)
    target_link_libraries(test_end_to_end_add
        PRIVATE tenzor_core GTest::gtest_main)
    tenzor_discover_tests(test_end_to_end_add
        DISCOVERY_TIMEOUT 120 BACKENDS none LABELS "jit;mlir;iree")

    # Backend-parity fixes (C1/C2/H6/M1/M3): eager-fallback safety net,
    # target-alias normalization, device-derived ROCm arch, HAL ordinal.
    add_executable(test_jit_eager_fallback test_jit_eager_fallback.cpp)
    target_link_libraries(test_jit_eager_fallback
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_jit_eager_fallback
        DISCOVERY_TIMEOUT 120 BACKENDS none LABELS "jit;mlir;iree")

    # JIT-R125 regression, in its own executable: the repro needs
    # unshare(CLONE_NEWUSER), which only succeeds in a single-threaded
    # process -- see the file's header comment.
    add_executable(test_jit_r125_output_temp_exhaustion
        test_jit_r125_output_temp_exhaustion.cpp)
    target_link_libraries(test_jit_r125_output_temp_exhaustion
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_jit_r125_output_temp_exhaustion
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir;iree")

    # JIT-R109 regression, in its own executable: testing_force_shared_iree_
    # state_cleanup() permanently disables the shared IREE runtime instance
    # for the rest of the process -- see the file's header comment.
    add_executable(test_jit_r109_shared_state_teardown_race
        test_jit_r109_shared_state_teardown_race.cpp)
    target_link_libraries(test_jit_r109_shared_state_teardown_race
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_jit_r109_shared_state_teardown_race
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir;iree")

    # JIT-R168 regression, in its own executable for the same reason as the
    # JIT-R109 test above (testing_force_shared_iree_state_cleanup() is a
    # one-shot, permanent teardown).
    add_executable(test_jit_r168_cache_miss_teardown_race
        test_jit_r168_cache_miss_teardown_race.cpp)
    target_link_libraries(test_jit_r168_cache_miss_teardown_race
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_jit_r168_cache_miss_teardown_race
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir;iree")

    # Task C.1 — Elementwise ops lower to StableHLO primitives.
    add_executable(test_op_elementwise test_op_elementwise.cpp)
    target_link_libraries(test_op_elementwise
        PRIVATE tenzor_core GTest::gtest_main)
    tenzor_discover_tests(test_op_elementwise
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # Task C.1 (continued) — `tz.where` end-to-end via stablehlo.select.
    add_executable(test_op_where test_op_where.cpp)
    target_link_libraries(test_op_where
        PRIVATE tenzor_core GTest::gtest_main)
    tenzor_discover_tests(test_op_where
        DISCOVERY_TIMEOUT 120 BACKENDS none LABELS "jit;mlir;iree")

    # JIT-R133 — LogicalAnd/Or/Not + comparison predicates (Eq/Ne/Lt/Le/Gt/Ge)
    # numeric coverage, fanned out over every available IREE target.
    add_executable(test_op_logical_predicates test_op_logical_predicates.cpp)
    target_link_libraries(test_op_logical_predicates
        PRIVATE tenzor_core GTest::gtest_main)
    tenzor_discover_tests(test_op_logical_predicates
        DISCOVERY_TIMEOUT 120 BACKENDS none LABELS "jit;mlir;iree")

    # Task C.2 — Activations (ReLU / Sigmoid / SiLU / GELU).
    add_executable(test_op_activations test_op_activations.cpp)
    target_link_libraries(test_op_activations
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_activations
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # Task C.3 — Reductions (Sum / Mean / Max / Softmax).
    add_executable(test_op_reductions test_op_reductions.cpp)
    target_link_libraries(test_op_reductions
        PRIVATE tenzor_core GTest::gtest_main)
    tenzor_discover_tests(test_op_reductions
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # Task C.4 — Linalg (MatMul / Bmm / Linear).
    add_executable(test_op_linalg test_op_linalg.cpp)
    target_link_libraries(test_op_linalg
        PRIVATE tenzor_core GTest::gtest_main)
    tenzor_discover_tests(test_op_linalg
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # JIT-F013 — fused epilogue (Conv/Linear/MatMul + ReLU/bias) must not be
    # dropped by the MLIR lowerer.
    add_executable(test_op_fused_epilogue test_op_fused_epilogue.cpp)
    target_link_libraries(test_op_fused_epilogue
        PRIVATE tenzor_core GTest::gtest_main)
    tenzor_discover_tests(test_op_fused_epilogue
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # Task C.5 — Shape ops (Reshape, Permute, Transpose, Slice, Cat,
    # Stack, Squeeze, Unsqueeze, Flatten, Broadcast).
    add_executable(test_op_shape test_op_shape.cpp)
    target_link_libraries(test_op_shape
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_shape
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # Task C.6 — Cast / Index ops (Cast, Embedding, IndexSelect).
    add_executable(test_op_cast_index test_op_cast_index.cpp)
    target_link_libraries(test_op_cast_index
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_cast_index
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # Task C.7 — Norm ops (LayerNorm, BatchNorm2d inference).
    add_executable(test_op_norms test_op_norms.cpp)
    target_link_libraries(test_op_norms
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_norms
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # Task C.8 — Vision ops (Conv2d, MaxPool2d, AvgPool2d,
    # AdaptiveAvgPool2d, Dropout, Padding, Interpolate; ResidualAdd is
    # already covered by the elementwise group).
    add_executable(test_op_vision test_op_vision.cpp)
    target_link_libraries(test_op_vision
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_vision
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # JIT-R081 — Quantize/Dequantize/QuantizedLinear/QuantizedLinearStatic/
    # QuantizedConv2d lowering.
    add_executable(test_op_quantization test_op_quantization.cpp)
    target_link_libraries(test_op_quantization
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_quantization
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # Task D.1.1 — FlashAttention dialect op lowers to stablehlo.custom_call
    # @tenzor_flash_attention.
    add_executable(test_op_flash_attention test_op_flash_attention.cpp)
    target_link_libraries(test_op_flash_attention
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_flash_attention
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir")

    # Task D.1.2 — tenzor_flash_attention custom_call dispatcher.
    add_executable(test_op_flash_attention_callback
        test_op_flash_attention_callback.cpp)
    target_link_libraries(test_op_flash_attention_callback
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_flash_attention_callback
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir")

    # Task D.1.3 — FlashAttention expand-to-stablehlo + roundtrip vs eager.
    add_executable(test_op_flash_attention_expand
        test_op_flash_attention_expand.cpp)
    target_link_libraries(test_op_flash_attention_expand
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_flash_attention_expand
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # Task D.2.1 — GQA dialect op lowers to stablehlo.custom_call @tenzor_gqa.
    add_executable(test_op_gqa test_op_gqa.cpp)
    target_link_libraries(test_op_gqa
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_gqa
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir")

    # Task D.2.2 — tenzor_gqa custom_call dispatcher.
    add_executable(test_op_gqa_callback test_op_gqa_callback.cpp)
    target_link_libraries(test_op_gqa_callback
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_gqa_callback
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir")

    # Task D.2.3 — GQA expand-to-stablehlo + roundtrip.
    add_executable(test_op_gqa_expand test_op_gqa_expand.cpp)
    target_link_libraries(test_op_gqa_expand
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_gqa_expand
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # Task D.3.1 — RoPE dialect op lowers to stablehlo.custom_call
    # @tenzor_rope_apply.
    add_executable(test_op_rope test_op_rope.cpp)
    target_link_libraries(test_op_rope
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_rope
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir")

    # Task D.3.2 — tenzor_rope_apply custom_call dispatcher.
    add_executable(test_op_rope_callback test_op_rope_callback.cpp)
    target_link_libraries(test_op_rope_callback
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_rope_callback
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir")

    # Task D.3.3 — RoPE expand-to-stablehlo + roundtrip.
    add_executable(test_op_rope_expand test_op_rope_expand.cpp)
    target_link_libraries(test_op_rope_expand
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_rope_expand
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # Task D.4.1 — RMSNorm dialect op lowers to stablehlo.custom_call
    # @tenzor_rms_norm.
    add_executable(test_op_rms_norm test_op_rms_norm.cpp)
    target_link_libraries(test_op_rms_norm
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_rms_norm
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir")

    # Task D.4.2 — tenzor_rms_norm custom_call dispatcher.
    add_executable(test_op_rms_norm_callback test_op_rms_norm_callback.cpp)
    target_link_libraries(test_op_rms_norm_callback
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_rms_norm_callback
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir")

    # Task D.4.3 — RMSNorm expand-to-stablehlo + roundtrip.
    add_executable(test_op_rms_norm_expand test_op_rms_norm_expand.cpp)
    target_link_libraries(test_op_rms_norm_expand
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_rms_norm_expand
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # Task E.1 / E.2 / E.3 — Mini-Llama eager + @tz.jit forward across
    # IREE targets. CPU runs eagerly + JIT-compiled; cuda / vulkan-spirv
    # / rocm skip when the corresponding tenzor backend is absent.
    add_executable(test_model_llama test_model_llama.cpp)
    target_link_libraries(test_model_llama
        PRIVATE tenzor_core GTest::gtest_main)
    tenzor_discover_tests(test_model_llama
        DISCOVERY_TIMEOUT 600 BACKENDS none LABELS "jit;mlir;iree")

    # Task E.4 / E.5 / E.6 — ResNet18 eager + @tz.jit forward across
    # IREE targets (using the project's built-in models::resnet18).
    add_executable(test_model_resnet test_model_resnet.cpp)
    target_link_libraries(test_model_resnet
        PRIVATE tenzor_core GTest::gtest_main)
    tenzor_discover_tests(test_model_resnet
        DISCOVERY_TIMEOUT 600 BACKENDS none LABELS "jit;mlir;iree")

    # Training-through-JIT parity for the MLIR-configured backend: a
    # requires-grad / captured-parameter call must train through the COMPILED
    # graph (grad_invoke interpreter) exactly like nvrtc, while no-grad
    # inference still uses IREE. Runs on CPU / CUDA / ROCm.
    add_executable(test_jit_training_parity test_jit_training_parity.cpp)
    target_link_libraries(test_jit_training_parity
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_jit_training_parity
        DISCOVERY_TIMEOUT 300 BACKENDS none LABELS "jit;mlir;iree")

    # Numeric-parity regressions: subprocess f64 bit-exactness (M8) and
    # F16 softmax/layernorm F32-accumulation parity (M3).
    add_executable(test_jit_mlir_numeric_parity test_jit_mlir_numeric_parity.cpp)
    target_link_libraries(test_jit_mlir_numeric_parity
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_jit_mlir_numeric_parity
        DISCOVERY_TIMEOUT 300 BACKENDS none LABELS "jit;mlir;iree")

    # Group F — Debug UX (show_graph / show_mlir / show_stablehlo /
    # show_iree / cache_stats + TENZOR_JIT_DUMP env var).
    add_executable(test_debug_apis test_debug_apis.cpp)
    target_link_libraries(test_debug_apis
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_debug_apis
        DISCOVERY_TIMEOUT 120 BACKENDS none LABELS "jit;mlir;iree")
endif()
