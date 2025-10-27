add_test([=[ContiguousFixTest.SliceWithContiguous]=]  /home/lee/Projects/Tenzor/bin/test_contiguous_fix [==[--gtest_filter=ContiguousFixTest.SliceWithContiguous]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[ContiguousFixTest.SliceWithContiguous]=]  PROPERTIES WORKING_DIRECTORY /home/lee/Projects/Tenzor/build_fresh/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_contiguous_fix_TESTS ContiguousFixTest.SliceWithContiguous)
