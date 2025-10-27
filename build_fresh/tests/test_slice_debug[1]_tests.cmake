add_test([=[SliceDebugTest.InspectSlicedTensors]=]  /home/lee/Projects/Tenzor/bin/test_slice_debug [==[--gtest_filter=SliceDebugTest.InspectSlicedTensors]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[SliceDebugTest.InspectSlicedTensors]=]  PROPERTIES WORKING_DIRECTORY /home/lee/Projects/Tenzor/build_fresh/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_slice_debug_TESTS SliceDebugTest.InspectSlicedTensors)
