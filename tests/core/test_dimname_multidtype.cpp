// Multi-backend multi-dtype tests for named dimension support.
//
// Verifies that Dimname operations (creation, matching, tensor assignment,
// lookup) work correctly with tensors of various dtypes on all backends.

#include "../multi_backend_dtype_fixture.hpp"

#include "tenzor/core/dimname.hpp"

namespace tenzor {
namespace testing {

class DimnameMultiDTypeTest : public MultiBackendDTypeTest {};

// ---------------------------------------------------------------------------
// Wildcard and named Dimname creation (dtype/backend independent, but
// validates consistency across all parameterized contexts)
// ---------------------------------------------------------------------------

TEST_P(DimnameMultiDTypeTest, WildcardCreation) {
    Dimname w;
    EXPECT_TRUE(w.is_wildcard());
    EXPECT_EQ(w.name(), "");

    auto w2 = Dimname::wildcard();
    EXPECT_TRUE(w2.is_wildcard());
    EXPECT_EQ(w, w2);
}

TEST_P(DimnameMultiDTypeTest, NamedCreationAndEquality) {
    Dimname batch("batch");
    EXPECT_FALSE(batch.is_wildcard());
    EXPECT_EQ(batch.name(), "batch");

    // Interning: same string produces equal Dimnames
    Dimname batch2("batch");
    EXPECT_EQ(batch, batch2);

    Dimname other("channel");
    EXPECT_NE(batch, other);
}

TEST_P(DimnameMultiDTypeTest, MatchingSemantics) {
    Dimname batch("batch");
    Dimname wild;
    Dimname channel("channel");

    EXPECT_TRUE(wild.matches(batch));
    EXPECT_TRUE(batch.matches(wild));
    EXPECT_TRUE(wild.matches(wild));
    EXPECT_TRUE(batch.matches(Dimname("batch")));
    EXPECT_FALSE(batch.matches(channel));
}

// ---------------------------------------------------------------------------
// Tensor name assignment with parameterized dtype/device
// ---------------------------------------------------------------------------

TEST_P(DimnameMultiDTypeTest, TensorNameAssignment) {
    auto t = tenzor::zeros({2, 3, 4}, dtype(), device());
    EXPECT_FALSE(t.has_names());

    auto named = t.rename({Dimname("batch"), Dimname("height"), Dimname("width")});
    EXPECT_TRUE(named.has_names());

    auto names = named.names();
    ASSERT_TRUE(names.has_value());
    EXPECT_EQ(names->size(), 3u);
    EXPECT_EQ((*names)[0].name(), "batch");
    EXPECT_EQ((*names)[1].name(), "height");
    EXPECT_EQ((*names)[2].name(), "width");
}

TEST_P(DimnameMultiDTypeTest, DimIndexLookup) {
    auto t = tenzor::zeros({2, 3, 4}, dtype(), device());
    auto named = t.rename({Dimname("batch"), Dimname("height"), Dimname("width")});

    EXPECT_EQ(named.dim_index("batch"), 0);
    EXPECT_EQ(named.dim_index("height"), 1);
    EXPECT_EQ(named.dim_index("width"), 2);
}

TEST_P(DimnameMultiDTypeTest, NameCountMismatchThrows) {
    auto t = tenzor::zeros({2, 3}, dtype(), device());
    EXPECT_THROW(
        t.rename({Dimname("a"), Dimname("b"), Dimname("c")}),
        std::invalid_argument
    );
}

// ---------------------------------------------------------------------------
// Named tensor preserves dtype and device
// ---------------------------------------------------------------------------

TEST_P(DimnameMultiDTypeTest, NamedTensorPreservesDTypeAndDevice) {
    auto t = tenzor::zeros({4, 5}, dtype(), device());
    auto named = t.rename({Dimname("rows"), Dimname("cols")});

    EXPECT_EQ(named.dtype(), dtype());
    EXPECT_EQ(named.device().type, device().type);
    EXPECT_EQ(named.numel(), 20);
}

// ---------------------------------------------------------------------------
// Thread-safe interning (validates across parameterized contexts)
// ---------------------------------------------------------------------------

TEST_P(DimnameMultiDTypeTest, ThreadSafeInterning) {
    constexpr int num_threads = 8;
    std::vector<std::thread> threads;
    std::vector<Dimname> results(num_threads);

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&results, i]() {
            results[i] = Dimname("shared_multidtype_name");
        });
    }
    for (auto& t : threads) t.join();

    for (int i = 1; i < num_threads; i++) {
        EXPECT_EQ(results[0], results[i]);
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DimnameMultiDTypeTest);

} // namespace testing
} // namespace tenzor
