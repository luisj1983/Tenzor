add_test([=[JITTest.TraceSimpleModel]=]  /home/lee/Projects/Tenzor/bin/test_jit [==[--gtest_filter=JITTest.TraceSimpleModel]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[JITTest.TraceSimpleModel]=]  PROPERTIES WORKING_DIRECTORY /home/lee/Projects/Tenzor/build_fresh/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_jit_TESTS JITTest.TraceSimpleModel)
