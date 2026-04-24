add_executable(op_coverage_report op_coverage_report.cpp)
target_link_libraries(op_coverage_report PRIVATE tenzor_core)
set_target_properties(op_coverage_report PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)
