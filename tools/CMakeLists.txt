add_executable(op_coverage_report op_coverage_report.cpp)
target_link_libraries(op_coverage_report PRIVATE tenzor_core)
set_target_properties(op_coverage_report PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)

# audit-8 II.19: previously orphaned (lived in tools/ but were never built).
# Wire them up so they fail loudly at build time instead of silently bit-rotting.
add_executable(tenzor_serve tenzor_serve.cpp)
target_link_libraries(tenzor_serve PRIVATE tenzor_core)
set_target_properties(tenzor_serve PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)

# Requires ModelHub (tenzor::models::ModelHub lives behind
# TENZOR_BUILD_MODEL_HUB); linking without it fails with undefined references.
if(TENZOR_BUILD_MODEL_HUB)
    add_executable(tenzor_model_converter tenzor_model_converter.cpp)
    target_link_libraries(tenzor_model_converter PRIVATE tenzor_core)
    set_target_properties(tenzor_model_converter PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
    )
endif()

# Smoke test: just exercise the CLI argument parsing — full server start-up
# would need a model repository. tenzor_serve accepts a path + port and
# returns immediately if it can't bind; the --help flag isn't supported by
# the current main(), so we invoke with an unwritable repo and verify it
# fails cleanly (returns non-zero) rather than crashing.
add_test(NAME tenzor_serve_smoke
         COMMAND $<TARGET_FILE:tenzor_serve> /nonexistent_repo_dir 0)
set_tests_properties(tenzor_serve_smoke PROPERTIES
    WILL_FAIL TRUE
    TIMEOUT 30
    LABELS "tools")

if(TENZOR_BUILD_MODEL_HUB)
    add_test(NAME tenzor_model_converter_usage
             COMMAND $<TARGET_FILE:tenzor_model_converter>)
    # Invoking with no args prints usage and returns 1.
    set_tests_properties(tenzor_model_converter_usage PROPERTIES
        WILL_FAIL TRUE
        TIMEOUT 30
        LABELS "tools")
endif()
