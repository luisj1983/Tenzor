add_test([=[TensorLifetimeTest.ContiguousDataPointerStability]=]  /home/lee/Projects/Tenzor/bin/test_tensor_lifetime [==[--gtest_filter=TensorLifetimeTest.ContiguousDataPointerStability]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[TensorLifetimeTest.ContiguousDataPointerStability]=]  PROPERTIES WORKING_DIRECTORY /home/lee/Projects/Tenzor/build_fresh/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_tensor_lifetime_TESTS TensorLifetimeTest.ContiguousDataPointerStability)
