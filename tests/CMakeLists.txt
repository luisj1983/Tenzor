cmake_minimum_required(VERSION 3.25)

# Find Google Test
find_package(GTest)

if(NOT GTest_FOUND)
    message(STATUS "Google Test not found, downloading...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Include GoogleTest module for gtest_discover_tests
include(GoogleTest)

# Maximum test execution timeout (25 minutes)
set(TENZOR_TEST_TIMEOUT 1500)

# When ON, tests enumerate all devices per backend (e.g. cuda:0, cuda:1)
# instead of only testing device index 0.
option(TENZOR_TEST_ALL_DEVICES "Test all devices per backend" OFF)
if(TENZOR_TEST_ALL_DEVICES)
    add_compile_definitions(TENZOR_TEST_ALL_DEVICES)
endif()

# When ON, every ctest run propagates TENZOR_ONEAPI_ALLOW_CPU=1, which lets
# the OneAPI backend register Intel/AMD CPU SYCL devices (it skips them by
# default per the project policy that "OneAPI is the GPU backend"). Set this
# on machines where the only available SYCL runtime is the OpenCL CPU
# runtime — without it, every parametrized OneAPI test fails its SetUp with
# "Backend dispatch table not ready" because no device gets registered.
option(TENZOR_TEST_ONEAPI_ALLOW_CPU
       "Allow OneAPI tests to bind to a SYCL CPU device (debug/CI parity runs)" OFF)

# Custom function that wraps gtest_discover_tests
# Supports optional LABELS parameter for CTest label-based filtering
# and a BACKENDS parameter that controls backend tagging of generated
# CTest names (TEST_PREFIX) and CTest labels.
#
# BACKENDS values:
#   (omitted) / cpu           — single-backend CPU test. Prepend "cpu_" to
#                               every discovered CTest case and add "cpu"
#                               to LABELS so `ctest -R cpu` and
#                               `ctest -L cpu` both match.
#   cuda / rocm / vulkan / oneapi — same as above for that backend.
#   multi                     — test is parametrized over backends inside
#                               its source (e.g. uses
#                               INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS).
#                               Skip TEST_PREFIX because the backend name
#                               already appears in each parametrized
#                               case's value (e.g. `..."cuda"...`), but
#                               attach all five backend labels so
#                               `ctest -L cuda` still finds these.
#   none                      — pure-logic test that never touches a
#                               backend (lint, audit, codegen text
#                               output). No prefix, no backend label.
function(tenzor_discover_tests target)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "DISCOVERY_TIMEOUT;DISCOVERY_MODE" "PROPERTIES;LABELS;BACKENDS")
    set(TIMEOUT ${ARG_DISCOVERY_TIMEOUT})
    if(NOT TIMEOUT)
        set(TIMEOUT 60)
    endif()
    set(PROPS ${ARG_PROPERTIES})
    set(MODE ${ARG_DISCOVERY_MODE})
    if(NOT MODE)
        set(MODE "POST_BUILD")
    endif()

    # Inject TIMEOUT into properties if not already specified
    list(FIND PROPS "TIMEOUT" _timeout_idx)
    if(_timeout_idx EQUAL -1)
        list(APPEND PROPS TIMEOUT ${TENZOR_TEST_TIMEOUT})
    endif()

    # ----------------------------------------------------------------------
    # BACKENDS handling
    # ----------------------------------------------------------------------
    # Default = cpu (most tests link tenzor_backend_cpu implicitly through
    # tenzor_core and run on CPU). The supported tokens are checked so a
    # typo at the call site becomes a configure-time error instead of a
    # silent miss.
    set(_backends ${ARG_BACKENDS})
    if(NOT _backends)
        set(_backends "cpu")
    endif()

    set(_test_prefix "")
    set(_backend_labels "")
    set(_valid_single cpu cuda rocm vulkan oneapi)
    set(_all_backends cpu cuda rocm vulkan oneapi)

    list(LENGTH _backends _nbackends)
    if(_nbackends EQUAL 1 AND "${_backends}" STREQUAL "none")
        # Pure-logic test: no prefix, no backend label.
    elseif(_nbackends EQUAL 1 AND "${_backends}" STREQUAL "multi")
        # Source-level multi-backend parametrization: every five backends
        # get a label so `ctest -L cuda` etc. find these. Name already
        # contains the backend value so no TEST_PREFIX.
        #
        # V.41: also attach the literal label `multi` so `ctest -L multi`
        # selects this exact group.  Previously expanding `BACKENDS multi`
        # to the five backend labels meant `ctest -L multi` returned 0
        # tests despite TESTING.md documenting the selector.
        set(_backend_labels ${_all_backends} multi)
    else()
        foreach(_b IN LISTS _backends)
            if(NOT _b IN_LIST _valid_single)
                message(FATAL_ERROR
                    "tenzor_discover_tests(${target}): unknown BACKENDS "
                    "token '${_b}'. Expected one of: "
                    "cpu cuda rocm vulkan oneapi multi none")
            endif()
        endforeach()
        list(APPEND _backend_labels ${_backends})
        # TEST_PREFIX is a single string; when multiple backends are
        # declared explicitly (rare) we don't prefix to avoid ambiguity
        # — the backend labels still make `-L` work.
        if(_nbackends EQUAL 1)
            set(_test_prefix "${_backends}_")
        endif()
    endif()

    # Merge backend labels with caller-supplied LABELS into a single CMake
    # list and turn it into an escape-preserved string.
    #
    # Why this is awkward: gtest_discover_tests forwards LABELS through
    # GoogleTestAddTests.cmake, which eventually does
    #     add_command(set_tests_properties ${name} PROPERTIES ${arg_TEST_PROPERTIES})
    # with `${arg_TEST_PROPERTIES}` UNQUOTED. CMake unquoted expansion
    # strips escape-semicolon markers from list elements; once stripped,
    # the joined-labels string `a;b;c` splits into three positional
    # arguments. set_tests_properties then treats only the first as the
    # LABELS value and silently drops the rest as bogus property names.
    # That bug is how `ctest -L cuda` returned just one test.
    #
    # The fix: build the labels string with TRUE CMake escape-semicolon
    # markers between each label (not just literal backslash-semicolon
    # characters from `string(REPLACE)`, which don't survive expansion).
    # The only reliable way to insert real escape markers dynamically is
    # via `cmake_language(EVAL CODE ...)` — the eval'd source is re-parsed,
    # so `\;` becomes an actual escaped semicolon. The escapes then
    # survive all the way through to GoogleTestAddTests, which
    # bracket-arg-wraps the value (`LABELS [==[a;b;c]==]`) for CTest.
    set(_label_list "")
    if(ARG_LABELS)
        # ARG_LABELS may itself be a semicolon-separated string (e.g. the
        # legacy `"nn\;smoke"` form): list(APPEND) splits it back into
        # individual items.
        list(APPEND _label_list ${ARG_LABELS})
    endif()
    foreach(_bl IN LISTS _backend_labels)
        list(APPEND _label_list "${_bl}")
    endforeach()
    list(REMOVE_DUPLICATES _label_list)
    list(JOIN _label_list "\\;" _label_str_raw)
    # Re-parse the joined string so the `\;` sequences become true escape
    # markers rather than literal backslash-semicolon pairs.
    cmake_language(EVAL CODE
        "set(_label_str \"${_label_str_raw}\")")

    # Inject TENZOR_ONEAPI_ALLOW_CPU=1 if the project was configured with
    # TENZOR_TEST_ONEAPI_ALLOW_CPU=ON. Honors any user-supplied ENVIRONMENT
    # entry by appending instead of overwriting.
    if(TENZOR_TEST_ONEAPI_ALLOW_CPU)
        list(FIND PROPS "ENVIRONMENT" _env_idx)
        if(_env_idx EQUAL -1)
            list(APPEND PROPS ENVIRONMENT "TENZOR_ONEAPI_ALLOW_CPU=1")
        else()
            math(EXPR _env_val_idx "${_env_idx} + 1")
            list(GET PROPS ${_env_val_idx} _existing_env)
            list(REMOVE_AT PROPS ${_env_val_idx})
            list(INSERT PROPS ${_env_val_idx} "${_existing_env};TENZOR_ONEAPI_ALLOW_CPU=1")
        endif()
    endif()

    # Now call gtest_discover_tests. TEST_PREFIX is empty for `multi` /
    # `none`; gtest_discover_tests accepts an empty prefix gracefully.
    #
    # LABELS is passed as a separate quoted argument AFTER ${PROPS} so the
    # semicolon-joined value survives as one token (see the long comment
    # above on the unquoted-expansion bug). When _label_str is empty we
    # skip the keyword entirely so set_tests_properties never sees
    # `LABELS ""`.
    if(_test_prefix)
        if(_label_str)
            gtest_discover_tests(${target}
                TEST_PREFIX "${_test_prefix}"
                DISCOVERY_TIMEOUT ${TIMEOUT}
                DISCOVERY_MODE ${MODE}
                PROPERTIES ${PROPS} LABELS "${_label_str}")
        else()
            gtest_discover_tests(${target}
                TEST_PREFIX "${_test_prefix}"
                DISCOVERY_TIMEOUT ${TIMEOUT}
                DISCOVERY_MODE ${MODE}
                PROPERTIES ${PROPS})
        endif()
    else()
        if(_label_str)
            gtest_discover_tests(${target}
                DISCOVERY_TIMEOUT ${TIMEOUT}
                DISCOVERY_MODE ${MODE}
                PROPERTIES ${PROPS} LABELS "${_label_str}")
        else()
            gtest_discover_tests(${target}
                DISCOVERY_TIMEOUT ${TIMEOUT}
                DISCOVERY_MODE ${MODE}
                PROPERTIES ${PROPS})
        endif()
    endif()

    # ----------------------------------------------------------------------
    # DD.8: serialize PRE_TEST regeneration across parallel ctest invocations
    # ----------------------------------------------------------------------
    # gtest_discover_tests' PRE_TEST include.cmake regenerates the
    # *_tests.cmake on every ctest run whose IS_NEWER_THAN check trips
    # (e.g. after a build, after `cmake -B build`, …). The regen calls
    # gtest_discover_tests_impl which begins with file(REMOVE CTEST_FILE)
    # followed by chunked file(APPEND). When two ctest processes hit the
    # same include.cmake in parallel — common during sharded CI runs and
    # editor `ctest -N` while a build is running — both observe the file
    # as needing regen, both truncate, then both APPEND a full copy of
    # the test list. Result: the *_tests.cmake ends up with each
    # add_test() duplicated exactly 2x, inflating `ctest -N` by ~13k
    # entries (~16%) and skewing parallel ctest wall time + flakiness
    # metrics.
    #
    # Fix: replace the gtest-generated include.cmake entry in the
    # directory's TEST_INCLUDE_FILES with a wrapper script that takes a
    # process-scoped file(LOCK) before include()-ing the upstream file.
    # Inside the lock the IS_NEWER_THAN check is re-evaluated — the
    # second-arriving process will see the freshly-regenerated
    # *_tests.cmake and skip its own regen entirely. POST_BUILD-mode
    # includes don't suffer the race (they only run during ninja), but
    # we wrap them the same way for consistency and so the lock guards
    # against a hypothetical `ninja -j` + `ctest -N` overlap as well.
    set(_orig_inc
        "${CMAKE_CURRENT_BINARY_DIR}/${target}[1]_include.cmake")
    set(_lock_inc
        "${CMAKE_CURRENT_BINARY_DIR}/${target}[1]_atomic_include.cmake")
    set(_lock_path
        "${CMAKE_CURRENT_BINARY_DIR}/${target}[1]_discover.lock")
    file(WRITE "${_lock_inc}"
        "# Auto-generated by tenzor_discover_tests (DD.8 race-free wrapper).\n"
        "# Holds a process-scoped lock so parallel ctest invocations\n"
        "# don't both regenerate the *_tests.cmake at the same time.\n"
        "file(LOCK [==[${_lock_path}]==] GUARD PROCESS TIMEOUT 300\n"
        "          RESULT_VARIABLE _tenzor_lock_rv)\n"
        "if(NOT _tenzor_lock_rv EQUAL 0)\n"
        "    message(WARNING\n"
        "        \"tenzor_discover_tests: failed to acquire lock on\"\n"
        "        \" [==[${_lock_path}]==] (rv=\${_tenzor_lock_rv}); proceeding unlocked\")\n"
        "endif()\n"
        "include([==[${_orig_inc}]==])\n"
        "file(LOCK [==[${_lock_path}]==] RELEASE)\n"
    )

    # Replace the gtest-installed entry in TEST_INCLUDE_FILES with our
    # wrapper. gtest_discover_tests appended `_orig_inc` to the
    # directory's TEST_INCLUDE_FILES via set_property(APPEND); we read
    # it back, swap `_orig_inc` → `_lock_inc`, and write the whole list
    # back. Falls back to APPEND if the entry isn't found (defensive,
    # shouldn't happen).
    get_property(_tif DIRECTORY PROPERTY TEST_INCLUDE_FILES)
    list(FIND _tif "${_orig_inc}" _tif_idx)
    if(_tif_idx GREATER_EQUAL 0)
        list(REMOVE_AT _tif ${_tif_idx})
        list(INSERT _tif ${_tif_idx} "${_lock_inc}")
        set_property(DIRECTORY PROPERTY TEST_INCLUDE_FILES "${_tif}")
    else()
        set_property(DIRECTORY APPEND PROPERTY
            TEST_INCLUDE_FILES "${_lock_inc}")
    endif()
endfunction()

# Helper macro: create a test executable, link GTest, and register with CTest.
# Usage: tenzor_add_test(NAME test_foo SOURCE unit/test_foo.cpp [LIBS extra_lib]
#                        [LABELS unit] [BACKENDS cpu|cuda|...|multi|none])
# BACKENDS is forwarded to tenzor_discover_tests; default is cpu.
macro(tenzor_add_test)
    cmake_parse_arguments(ARG "" "NAME;SOURCE" "LIBS;LABELS;BACKENDS" ${ARGN})
    add_executable(${ARG_NAME} ${ARG_SOURCE})
    target_link_libraries(${ARG_NAME} PRIVATE tenzor_core GTest::gtest_main ${ARG_LIBS})
    set(_extra "")
    if(ARG_LABELS)
        list(APPEND _extra LABELS "${ARG_LABELS}")
    endif()
    if(ARG_BACKENDS)
        list(APPEND _extra BACKENDS ${ARG_BACKENDS})
    endif()
    tenzor_discover_tests(${ARG_NAME} DISCOVERY_TIMEOUT 60 ${_extra})
endmacro()

# ============================================================================
# Subdirectories
# ============================================================================
add_subdirectory(unit)
add_subdirectory(nn)
add_subdirectory(autograd)
add_subdirectory(ops)
add_subdirectory(backends)
add_subdirectory(backend)
add_subdirectory(core)
add_subdirectory(jit)
add_subdirectory(distributed)
add_subdirectory(serving)
add_subdirectory(lite)
add_subdirectory(integration)
add_subdirectory(backend_parity)
add_subdirectory(utils)
add_subdirectory(examples)
add_subdirectory(nested)
add_subdirectory(sparse)

# ============================================================================
# Combined unit test executable
# ============================================================================
add_executable(tenzor_unit_tests
    unit/test_tensor.cpp
    unit/test_device.cpp
    unit/test_ops.cpp
    unit/test_ops_additional.cpp
    unit/test_special_math.cpp
    unit/test_matrix_construction.cpp
    unit/test_ddp_unused_params.cpp
    unit/test_autograd.cpp
    test_autograd_transform.cpp
    unit/test_cpu_kernels.cpp
    unit/test_broadcasting.cpp
    unit/test_transforms.cpp
    unit/test_linear.cpp
    unit/test_losses.cpp
    unit/test_optimizers.cpp
    unit/test_callbacks.cpp
    unit/test_graph_optimizer.cpp
    unit/test_simd_dispatch.cpp
    unit/test_simd.cpp
    unit/test_simd_runtime_dispatch.cpp
    unit/test_oneapi_backend_loading.cpp
    unit/test_oneapi_operations.cpp
)

target_link_libraries(tenzor_unit_tests PRIVATE
    tenzor_core
    tenzor_backend_cpu
    GTest::gtest_main
)

# test_simd_runtime_dispatch.cpp accesses internal kernel headers
# (simd_dispatch.hpp and simd_traits.hpp from the CPU backend source tree).
target_include_directories(tenzor_unit_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/src/backends/cpu
    ${PROJECT_SOURCE_DIR}/src/backends/cpu/kernels
    ${PROJECT_SOURCE_DIR}/include
)
foreach(backend cuda rocm oneapi vulkan)
    if(TARGET tenzor_backend_${backend})
        target_link_libraries(tenzor_unit_tests PRIVATE tenzor_backend_${backend})
    endif()
endforeach()

# gtest_discover_tests enables individual test case discovery for better sharding
# (Previously used add_test due to static destruction crash — now resolved)
tenzor_discover_tests(tenzor_unit_tests DISCOVERY_TIMEOUT 120 LABELS "unit")

# Runtime SIMD dispatch test — standalone executable for targeted runs.
# Also included in tenzor_unit_tests above.
add_executable(test_simd_runtime_dispatch
    unit/test_simd_runtime_dispatch.cpp
)
target_link_libraries(test_simd_runtime_dispatch PRIVATE
    tenzor_core
    tenzor_backend_cpu
    GTest::gtest_main
)
target_include_directories(test_simd_runtime_dispatch PRIVATE
    ${PROJECT_SOURCE_DIR}/src/backends/cpu
    ${PROJECT_SOURCE_DIR}/src/backends/cpu/kernels
    ${PROJECT_SOURCE_DIR}/include
)
tenzor_discover_tests(test_simd_runtime_dispatch DISCOVERY_TIMEOUT 60 LABELS "unit")

# SIMD Benchmark tests (separate executable for benchmarking)
add_executable(tenzor_simd_benchmark
    unit/benchmark_simd.cpp
)

target_link_libraries(tenzor_simd_benchmark PRIVATE
    tenzor_core
    tenzor_backend_cpu
    GTest::gtest_main
)

# ============================================================================
# Root-level test executables (source files in tests/ root)
# ============================================================================

# CUDA scalar debug test (separate executable for quick debugging)
add_executable(test_cuda_scalar_debug test_cuda_scalar_debug.cpp)
target_link_libraries(test_cuda_scalar_debug PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_cuda_scalar_debug DISCOVERY_TIMEOUT 60)





# Legacy integration tests (kept for compatibility).
# integration/test_nn.cpp moved out into the dedicated test_nn_smoke
# executable below (audit 2026-05-02 I.1) so its small-MLP cases can be
# tagged for the smoke set without dragging in the heavier tests in
# this bundle.
add_executable(tenzor_integration_tests
    integration/test_training.cpp
    integration/test_debug_mlp.cpp
)

target_link_libraries(tenzor_integration_tests PRIVATE
    tenzor_core
    GTest::gtest_main
)

tenzor_discover_tests(tenzor_integration_tests DISCOVERY_TIMEOUT 60)

# Audit 2026-05-02 I.1: extract just the smallest-MLP integration cases
# from test_nn.cpp into a separate executable so they can be tagged for
# the smoke set (`ctest -L smoke`) without dragging in the heavier
# test_training.cpp + test_debug_mlp.cpp content of the parent bundle.
add_executable(test_nn_smoke
    integration/test_nn.cpp
)
target_link_libraries(test_nn_smoke PRIVATE
    tenzor_core
    GTest::gtest_main
)
tenzor_discover_tests(test_nn_smoke
    LABELS "integration\;smoke"
    DISCOVERY_TIMEOUT 60
)

# Conv1d tests
add_executable(test_conv1d test_conv1d.cpp)
target_link_libraries(test_conv1d PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_conv1d DISCOVERY_TIMEOUT 60)

# Phase P0 / Fix 3: Float64 LayerNorm precision regression — accumulators
# were declared `float` inside `template<typename T>`, silently truncating
# F64 inputs. See src/backends/cpu/kernels/nn_kernels.cpp.
add_executable(test_layer_norm_f64_precision test_layer_norm_f64_precision.cpp)
target_link_libraries(test_layer_norm_f64_precision PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_layer_norm_f64_precision DISCOVERY_TIMEOUT 60)

# Numerical-precision fix: group_norm_impl and instance_norm_impl used
# `float` accumulators inside `template<typename T>` — silently truncating
# Float64 inputs to Float32. Same bug class as the LayerNorm scalar fix.
add_executable(test_group_instance_norm_f64_precision test_group_instance_norm_f64_precision.cpp)
target_link_libraries(test_group_instance_norm_f64_precision PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_group_instance_norm_f64_precision DISCOVERY_TIMEOUT 60)

# Numerical-precision fix: layer_norm_simd + layer_norm_simd_with_stats used
# the catastrophic-cancellation formula var = sum_sq/n - mean^2. For inputs
# with large mean this produces negative variance and a NaN inv_std. Fixed
# with two-pass E[(x-mean)^2] form.
add_executable(test_layer_norm_variance_stability test_layer_norm_variance_stability.cpp)
target_link_libraries(test_layer_norm_variance_stability PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_layer_norm_variance_stability DISCOVERY_TIMEOUT 60)

# Phase P0 / Fix 5 — CUDA median Float16/BFloat16: the indices output was
# a hard-coded 0; the values were reconstructed via float-to-half conversion
# losing the original half representation. Mirrors mode_kernel pattern.
add_executable(test_median_f16_bf16 test_median_f16_bf16.cpp)
target_link_libraries(test_median_f16_bf16 PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_median_f16_bf16 DISCOVERY_TIMEOUT 60)

# Phase P0 / Fix 4 — Vulkan LayerNorm 3-tensor return: forward shader used
# to compute mean/rstd in shared memory and discard them; backward read
# them as inputs → OOB crash. Fix exposes them via new SSBO bindings.
add_executable(test_vulkan_layer_norm_saved_stats test_vulkan_layer_norm_saved_stats.cpp)
target_link_libraries(test_vulkan_layer_norm_saved_stats PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_vulkan_layer_norm_saved_stats DISCOVERY_TIMEOUT 60 BACKENDS vulkan)

# Phase P0 / Fix 6 — CUDA flash-attention backward dropout reproduction:
# backward kernel used to (void)-cast the Philox seed; now threads it
# through and re-applies the same mask the forward used.
add_executable(test_flash_attention_dropout_backward test_flash_attention_dropout_backward.cpp)
target_link_libraries(test_flash_attention_dropout_backward PRIVATE tenzor_core GTest::gtest_main)
if(TARGET tenzor_backend_cuda)
    target_link_libraries(test_flash_attention_dropout_backward PRIVATE tenzor_backend_cuda)
endif()
tenzor_discover_tests(test_flash_attention_dropout_backward DISCOVERY_TIMEOUT 60)

# Phase P0 / Fix 7 — grid_sample bicubic: forward used to silently fall
# back to bilinear; backward only handled bilinear (silent zero grads for
# bicubic). Both fixed.
add_executable(test_grid_sample_bicubic test_grid_sample_bicubic.cpp)
target_link_libraries(test_grid_sample_bicubic PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_grid_sample_bicubic DISCOVERY_TIMEOUT 60)

# ConvTranspose2d tests
add_executable(test_convtranspose2d test_convtranspose2d.cpp)
target_link_libraries(test_convtranspose2d PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_convtranspose2d DISCOVERY_TIMEOUT 60)

# ConvTranspose1d dilation regression (audit F.17)
add_executable(test_convtranspose1d_dilation test_convtranspose1d_dilation.cpp)
target_link_libraries(test_convtranspose1d_dilation PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_convtranspose1d_dilation DISCOVERY_TIMEOUT 60)

# Training callbacks lifecycle regression (audit G.9)
add_executable(test_training_callbacks_lifecycle test_training_callbacks_lifecycle.cpp)
target_link_libraries(test_training_callbacks_lifecycle PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_training_callbacks_lifecycle DISCOVERY_TIMEOUT 60)

# Non-differentiable Function wrappers (audit E.7)
add_executable(test_non_differentiable_ops test_non_differentiable_ops.cpp)
target_link_libraries(test_non_differentiable_ops PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_non_differentiable_ops DISCOVERY_TIMEOUT 60)

# Function::op_id() virtual regression (audit A.2)
add_executable(test_function_op_id test_function_op_id.cpp)
target_link_libraries(test_function_op_id PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_function_op_id DISCOVERY_TIMEOUT 60)

# GraphOptimizer OpId-based pattern matcher (audit A.2)
add_executable(test_graph_optimizer_op_id test_graph_optimizer_op_id.cpp)
target_link_libraries(test_graph_optimizer_op_id PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_graph_optimizer_op_id DISCOVERY_TIMEOUT 60)

# vmap OpId-keyed dispatch (audit A.3)
add_executable(test_vmap_opid_dispatch test_vmap_opid_dispatch.cpp)
target_link_libraries(test_vmap_opid_dispatch PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_vmap_opid_dispatch DISCOVERY_TIMEOUT 60)

# Optimizer post-step hook infrastructure (audit G.10)
add_executable(test_optimizer_step_hooks test_optimizer_step_hooks.cpp)
target_link_libraries(test_optimizer_step_hooks PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_optimizer_step_hooks DISCOVERY_TIMEOUT 60)

# ParamGroup contract on per-optimizer hyperparams (audit D.4)
add_executable(test_param_group_contract test_param_group_contract.cpp)
target_link_libraries(test_param_group_contract PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_param_group_contract DISCOVERY_TIMEOUT 60)

# Minimal training test (debugging NaN issue)
add_executable(test_minimal_training test_minimal_training.cpp)
target_link_libraries(test_minimal_training PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_minimal_training DISCOVERY_TIMEOUT 60)

# Split operation tests
add_executable(test_split_operation test_split_operation.cpp)
target_link_libraries(test_split_operation PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_split_operation DISCOVERY_TIMEOUT 60)

# DType conversion tests
add_executable(test_dtype_conversion test_dtype_conversion.cpp)
target_link_libraries(test_dtype_conversion PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dtype_conversion DISCOVERY_TIMEOUT 60)

# BMM autograd tests
add_executable(test_bmm_autograd test_bmm_autograd.cpp)
target_link_libraries(test_bmm_autograd PRIVATE tenzor_core GTest::gtest)
tenzor_discover_tests(test_bmm_autograd DISCOVERY_TIMEOUT 60)

# Linear reshape integration tests
add_executable(test_linear_reshape_integration test_linear_reshape_integration.cpp)
target_link_libraries(test_linear_reshape_integration PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_linear_reshape_integration DISCOVERY_TIMEOUT 60)

# Creation operations tests
add_executable(test_creation_ops test_creation_ops.cpp)
target_link_libraries(test_creation_ops PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_creation_ops DISCOVERY_TIMEOUT 60)

# Creation operations multi-dtype tests
add_executable(test_creation_ops_multidtype test_creation_ops_multidtype.cpp)
target_link_libraries(test_creation_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_creation_ops_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Comparison operators tests
add_executable(test_comparison_operators test_comparison_operators.cpp)
target_link_libraries(test_comparison_operators PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_comparison_operators DISCOVERY_TIMEOUT 60)

# Checkpoint leaf fix test
add_executable(test_checkpoint_leaf_fix test_checkpoint_leaf_fix.cpp)
target_link_libraries(test_checkpoint_leaf_fix tenzor_core)
target_include_directories(test_checkpoint_leaf_fix PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME cpu_CheckpointLeafFix COMMAND test_checkpoint_leaf_fix)
set_tests_properties(cpu_CheckpointLeafFix PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "cpu")

# Phase 9-followup #29: tolerance audit. Runs strict against a baseline of
# known pre-existing unjustified loose tolerances (audit-3 T.3) so the
# ratchet only fails on NEW violations. Drive the baseline to zero over time
# by justifying or tightening each entry.
add_test(NAME ToleranceAudit
    COMMAND ${CMAKE_COMMAND} -E env
            python3 ${PROJECT_SOURCE_DIR}/tools/audit_tolerances.py
                --root ${PROJECT_SOURCE_DIR}/tests
                --strict
                --baseline ${PROJECT_SOURCE_DIR}/tools/audit_tolerances_baseline.txt)
set_tests_properties(ToleranceAudit PROPERTIES TIMEOUT 60 LABELS "audit;lint")

# Manual KLDiv debug test
add_executable(test_kldiv_manual test_kldiv_manual.cpp)
target_link_libraries(test_kldiv_manual PRIVATE tenzor_core GTest::gtest_main)
add_test(NAME cpu_KLDivManual.StepByStepBackward COMMAND test_kldiv_manual --gtest_filter=KLDivManual.StepByStepBackward)
add_test(NAME cpu_KLDivManual.SimpleSubtract COMMAND test_kldiv_manual --gtest_filter=KLDivManual.SimpleSubtract)
add_test(NAME cpu_KLDivManual.SimpleMultiply COMMAND test_kldiv_manual --gtest_filter=KLDivManual.SimpleMultiply)
set_tests_properties(cpu_KLDivManual.StepByStepBackward cpu_KLDivManual.SimpleSubtract cpu_KLDivManual.SimpleMultiply
    PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "cpu")

# ONNX import tests
add_executable(test_onnx_import test_onnx_import.cpp)
target_link_libraries(test_onnx_import PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_onnx_import DISCOVERY_TIMEOUT 60)

# Quantization tests
add_executable(test_quantization test_quantization.cpp)
target_link_libraries(test_quantization PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_quantization DISCOVERY_TIMEOUT 60)

# ============================================================================
# Phase 11 Tests - Additional Backend Support
# ============================================================================
add_executable(test_phase11_backends test_phase11_backends.cpp)
target_link_libraries(test_phase11_backends PRIVATE tenzor_core GTest::gtest_main)
# audit-3 T.12: exercises OneAPI + Vulkan — label as `multi` for consistency
# with T.6's other multi-backend root tests (root-level multidtype/parity).
tenzor_discover_tests(test_phase11_backends DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# CIoU Loss Tests
add_executable(test_ciou_loss test_ciou_loss.cpp)
target_link_libraries(test_ciou_loss PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_ciou_loss DISCOVERY_TIMEOUT 60)

# Backend Parity Test - Verify slice operations work correctly on all backends
add_executable(test_slice_backend_parity test_slice_backend_parity.cpp)
target_link_libraries(test_slice_backend_parity PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_slice_backend_parity DISCOVERY_TIMEOUT 60)

# Slice Debug Test - Debug slice offset and data_ptr behavior
add_executable(test_slice_debug test_slice_debug.cpp)
target_link_libraries(test_slice_debug PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_slice_debug DISCOVERY_TIMEOUT 60)

# Contiguous Fix Test - Verify .contiguous() fixes slice data access
add_executable(test_contiguous_fix test_contiguous_fix.cpp)
target_link_libraries(test_contiguous_fix PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_contiguous_fix DISCOVERY_TIMEOUT 60)

# Tensor Lifetime Test - Debug tensor data pointer stability
add_executable(test_tensor_lifetime test_tensor_lifetime.cpp)
target_link_libraries(test_tensor_lifetime PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_tensor_lifetime DISCOVERY_TIMEOUT 60)

# Backend Parity Example - Shows how to write multi-backend tests
add_executable(test_backend_parity_example test_backend_parity_example.cpp)
target_link_libraries(test_backend_parity_example PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_backend_parity_example DISCOVERY_TIMEOUT 60)

# Tensor indexing operator test
add_executable(test_indexing_operator test_indexing_operator.cpp)
target_link_libraries(test_indexing_operator PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_indexing_operator DISCOVERY_TIMEOUT 60)

# ============================================================================
# Phase 12 Tests - Enhanced Test Coverage and Validation
# ============================================================================

# Quantization conversion tests
add_executable(test_quantization_conversion test_quantization_conversion.cpp)
target_link_libraries(test_quantization_conversion PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_quantization_conversion DISCOVERY_TIMEOUT 60)

# Mask R-CNN loss tests
add_executable(test_mask_rcnn_losses test_mask_rcnn_losses.cpp)
target_link_libraries(test_mask_rcnn_losses PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_mask_rcnn_losses DISCOVERY_TIMEOUT 60)

# Vulkan complete operations tests
add_executable(test_vulkan_complete_ops test_vulkan_complete_ops.cpp)
target_link_libraries(test_vulkan_complete_ops PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_vulkan_complete_ops DISCOVERY_TIMEOUT 60 BACKENDS vulkan)

# Repeat and Tile tests
add_executable(test_repeat_tile test_repeat_tile.cpp)
target_link_libraries(test_repeat_tile PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_repeat_tile DISCOVERY_TIMEOUT 60)

# ============================================================================
# Multi-backend root-level tests (Phase 4)
# ============================================================================

# Repeat/tile multi-backend
add_executable(test_repeat_tile_multidtype test_repeat_tile_multidtype.cpp)
target_link_libraries(test_repeat_tile_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_repeat_tile_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Comparison operators multi-backend
add_executable(test_comparison_operators_multidtype test_comparison_operators_multidtype.cpp)
target_link_libraries(test_comparison_operators_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_comparison_operators_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Statistics operations (cov, corrcoef) multi-backend
add_executable(test_statistics_multidtype test_statistics_multidtype.cpp)
target_link_libraries(test_statistics_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_statistics_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Indexing operator multi-backend
add_executable(test_indexing_operator_multidtype test_indexing_operator_multidtype.cpp)
target_link_libraries(test_indexing_operator_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_indexing_operator_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# BMM autograd multi-backend
add_executable(test_bmm_autograd_multidtype test_bmm_autograd_multidtype.cpp)
target_link_libraries(test_bmm_autograd_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_bmm_autograd_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# DType conversion multi-backend
add_executable(test_dtype_conversion_multidtype test_dtype_conversion_multidtype.cpp)
target_link_libraries(test_dtype_conversion_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dtype_conversion_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Split operation multi-backend
add_executable(test_split_operation_multidtype test_split_operation_multidtype.cpp)
target_link_libraries(test_split_operation_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_split_operation_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Conv1d multi-backend
add_executable(test_conv1d_multidtype test_conv1d_multidtype.cpp)
target_link_libraries(test_conv1d_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_conv1d_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# CholeskySolve multi-backend
add_executable(test_cholesky_solve_multidtype test_cholesky_solve_multidtype.cpp)
target_link_libraries(test_cholesky_solve_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_cholesky_solve_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# DeformableConv2d multi-backend
add_executable(test_deformable_conv2d_multidtype test_deformable_conv2d_multidtype.cpp)
target_link_libraries(test_deformable_conv2d_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_deformable_conv2d_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Einsum dispatch multi-backend
add_executable(test_einsum_dispatch_multidtype test_einsum_dispatch_multidtype.cpp)
target_link_libraries(test_einsum_dispatch_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_einsum_dispatch_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# ConvTranspose2d multi-backend
add_executable(test_convtranspose2d_multidtype test_convtranspose2d_multidtype.cpp)
target_link_libraries(test_convtranspose2d_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_convtranspose2d_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# KL divergence multi-backend
add_executable(test_kldiv_manual_multidtype test_kldiv_manual_multidtype.cpp)
target_link_libraries(test_kldiv_manual_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_kldiv_manual_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# ============================================================================
# Phase 4: New tests for previously untested components
# ============================================================================

# Missing loss functions (MarginRanking, Poisson, Cosine, Triplet, etc.)
add_executable(test_losses_missing_multidtype test_losses_missing_multidtype.cpp)
target_link_libraries(test_losses_missing_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_losses_missing_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# RNN cells (RNNCell, GRUCell)
add_executable(test_rnn_cells_multidtype test_rnn_cells_multidtype.cpp)
target_link_libraries(test_rnn_cells_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_rnn_cells_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# RNN utilities (pack_padded_sequence, pad_packed_sequence)
add_executable(test_rnn_utils_multidtype test_rnn_utils_multidtype.cpp)
target_link_libraries(test_rnn_utils_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_rnn_utils_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# KV Cache
add_executable(test_kv_cache_multidtype test_kv_cache_multidtype.cpp)
target_link_libraries(test_kv_cache_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_kv_cache_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# Gradient accumulation
add_executable(test_grad_accumulation test_grad_accumulation.cpp)
target_link_libraries(test_grad_accumulation PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_grad_accumulation DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# ROI operations (ROIAlign)
add_executable(test_roi_ops_multidtype test_roi_ops_multidtype.cpp)
target_link_libraries(test_roi_ops_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_roi_ops_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# LOBPCG eigenvalue solver multi-backend
add_executable(test_lobpcg_multidtype test_lobpcg_multidtype.cpp)
target_link_libraries(test_lobpcg_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lobpcg_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# DCT / IDCT multi-backend
add_executable(test_dct_multidtype test_dct_multidtype.cpp)
target_link_libraries(test_dct_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dct_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

add_executable(test_mfcc_multidtype test_mfcc_multidtype.cpp)
target_link_libraries(test_mfcc_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_mfcc_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# LazyTensor deferred-execution tests (Phase 5.2)
add_executable(test_lazy_tensor test_lazy_tensor.cpp)
target_link_libraries(test_lazy_tensor PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lazy_tensor DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# LazyTensor multi-backend × multi-dtype tests (audit 2026-05-02 D.2)
add_executable(test_lazy_tensor_multidtype test_lazy_tensor_multidtype.cpp)
target_link_libraries(test_lazy_tensor_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lazy_tensor_multidtype DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST BACKENDS multi)

# LazyTensor: materialise -> Variable -> backward parity (audit 2026-05-02 D.3)
add_executable(test_lazy_tensor_backward test_lazy_tensor_backward.cpp)
target_link_libraries(test_lazy_tensor_backward PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_lazy_tensor_backward DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Vulkan tensor creation test (standalone, non-GTest)
add_executable(vulkan_tensor_test vulkan_tensor_test.cpp)
target_link_libraries(vulkan_tensor_test PRIVATE tenzor_core)

# Vulkan diagnostic test
add_executable(vulkan_diagnostic vulkan_diagnostic.cpp)
target_link_libraries(vulkan_diagnostic PRIVATE tenzor_core)

# Vulkan addition debug test
add_executable(vulkan_add_debug vulkan_add_debug.cpp)
target_link_libraries(vulkan_add_debug PRIVATE tenzor_core)

# Debug DataLoader test
add_executable(debug_dataloader debug_dataloader.cpp)
target_link_libraries(debug_dataloader PRIVATE tenzor_core)

# Minimal Swin debug test - Compare Float32 vs Float16 backward pass
add_executable(minimal_swin_debug minimal_swin_debug.cpp)
target_link_libraries(minimal_swin_debug PRIVATE tenzor_core GTest::gtest_main)

# Check Float32 gradient validity
add_executable(check_float32_gradients check_float32_gradients.cpp)
target_link_libraries(check_float32_gradients PRIVATE tenzor_core)

# Phase 1.5: Metrics extended tests
add_executable(test_metrics_extended unit/test_metrics_extended.cpp)
target_link_libraries(test_metrics_extended PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_metrics_extended LABELS "unit" DISCOVERY_TIMEOUT 60)

# ============================================================================
# CUDA root-level tests
# ============================================================================
if(TENZOR_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)

    # CUDA concatenation tests
    add_executable(test_cuda_cat test_cuda_cat.cpp)
    target_link_libraries(test_cuda_cat PRIVATE tenzor_core CUDA::cudart)
    target_include_directories(test_cuda_cat PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    set_target_properties(test_cuda_cat PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME cuda_CUDAConcatenation COMMAND test_cuda_cat)
    set_tests_properties(cuda_CUDAConcatenation PROPERTIES TIMEOUT ${TENZOR_TEST_TIMEOUT} LABELS "cuda")
endif()

# test_jit.cpp orphan removed: rotted APIs (Tensor::full/zeros/ones static methods, DeviceType enum no longer exist)

# ============================================================================
# Benchmark executables
# ============================================================================

# Benchmark suite
add_executable(benchmark_suite benchmarks/benchmark_suite.cpp)
target_link_libraries(benchmark_suite PRIVATE tenzor_core tenzor_backend_cpu)

# Cross-backend benchmark suite
add_executable(benchmark_backends benchmarks/benchmark_backends.cpp)
target_link_libraries(benchmark_backends PRIVATE tenzor_core)

# ==============================================================================
# Layer-test grad-flow guard (lint test, no compilation)
#
# Fails CI if any tests/nn/layers/test_*.cpp file calls .backward() without
# at least one of: EXPECT_GRAD_FLOWS macro, numerical_gradient finite-
# difference helper, an explicit max-abs > 0 check, or a per-element
# EXPECT_NEAR pattern. Catches regressions in the silently-zeroed-gradient
# bug class at PR-time rather than after the bug ships.
# ==============================================================================
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
    add_test(
        NAME LayerTestGradFlowGuard
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_SOURCE_DIR}/tools/check_layer_tests.py"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
    set_tests_properties(LayerTestGradFlowGuard PROPERTIES
        LABELS "lint;layer_tests"
        TIMEOUT 30
    )
endif()

# ==============================================================================
# Python Tests (registered as CTest tests)
# ==============================================================================
if(TENZOR_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter)
    if(Python_FOUND)
        set(PYTHON_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python")
        set(PYTHON_TEST_ENV "PYTHONPATH=${CMAKE_BINARY_DIR}/python")

        # Python tests in this list also get the "smoke" label so the
        # fast-feedback `ctest -L smoke` set covers basic Python-binding
        # surface (audit 2026-05-02 I.1). Each must be a small,
        # fast-running test that exercises a load-bearing path.
        set(PYTHON_SMOKE_TESTS
            test_loss_simple
            test_tensor_ops
            test_autograd
        )

        file(GLOB PYTHON_TEST_FILES "${PYTHON_TEST_DIR}/test_*.py")
        foreach(test_file ${PYTHON_TEST_FILES})
            get_filename_component(test_name ${test_file} NAME_WE)
            # Each Python test process boots tenzor on CPU (and opportunistically
            # uses GPU backends when available). The `cpu_` prefix and `cpu`
            # label make `ctest -R cpu` / `ctest -L cpu` cover them; per-backend
            # GPU filtering for Python is handled at runtime via
            # TENZOR_SKIP_BACKENDS.
            add_test(
                NAME "cpu_python_${test_name}"
                COMMAND ${Python_EXECUTABLE} ${test_file}
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            )
            list(FIND PYTHON_SMOKE_TESTS ${test_name} _smoke_idx)
            if(_smoke_idx GREATER -1)
                set(_py_labels "python;smoke;cpu")
            else()
                set(_py_labels "python;cpu")
            endif()
            # Phase 13 — Python-side JIT tests join the `mlir` label so
            # `ctest -L mlir` covers them alongside the C++ jit tests.
            if(test_name MATCHES "jit")
                set(_py_labels "${_py_labels};jit;mlir")
            endif()
            set_tests_properties("cpu_python_${test_name}" PROPERTIES
                LABELS "${_py_labels}"
                TIMEOUT ${TENZOR_TEST_TIMEOUT}
                ENVIRONMENT "${PYTHON_TEST_ENV}"
            )
        endforeach()
    endif()
endif()
