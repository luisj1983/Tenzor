/**
 * @file test_dimname.cpp
 * @brief Tests for named dimension support
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/core/dimname.hpp"

using namespace tenzor;

class DimnameTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

TEST_F(DimnameTest, WildcardCreation) {
    Dimname w;
    EXPECT_TRUE(w.is_wildcard());
    EXPECT_EQ(w.name(), "");

    auto w2 = Dimname::wildcard();
    EXPECT_TRUE(w2.is_wildcard());
    EXPECT_EQ(w, w2);
}

TEST_F(DimnameTest, NamedCreation) {
    Dimname batch("batch");
    EXPECT_FALSE(batch.is_wildcard());
    EXPECT_EQ(batch.name(), "batch");
}

TEST_F(DimnameTest, InternEquality) {
    // Same string should produce same pointer (interning)
    Dimname d1("channel");
    Dimname d2("channel");
    EXPECT_EQ(d1, d2);

    // Different strings should not be equal
    Dimname d3("height");
    EXPECT_NE(d1, d3);
}

TEST_F(DimnameTest, MatchingSemantics) {
    Dimname batch("batch");
    Dimname wild;
    Dimname channel("channel");

    // Wildcard matches anything
    EXPECT_TRUE(wild.matches(batch));
    EXPECT_TRUE(batch.matches(wild));
    EXPECT_TRUE(wild.matches(wild));

    // Same name matches
    Dimname batch2("batch");
    EXPECT_TRUE(batch.matches(batch2));

    // Different names don't match
    EXPECT_FALSE(batch.matches(channel));
}

TEST_F(DimnameTest, DimnameList) {
    DimnameList names = {
        Dimname("batch"),
        Dimname("channel"),
        Dimname("height"),
        Dimname("width")
    };

    EXPECT_EQ(names.size(), 4);
    EXPECT_EQ(names[0].name(), "batch");
    EXPECT_EQ(names[3].name(), "width");
}

TEST_F(DimnameTest, TensorNameAssignment) {
    auto t = tenzor::zeros({2, 3, 4});
    EXPECT_FALSE(t.has_names());

    auto named = t.rename({Dimname("batch"), Dimname("height"), Dimname("width")});
    EXPECT_TRUE(named.has_names());

    auto names = named.names();
    ASSERT_TRUE(names.has_value());
    EXPECT_EQ(names->size(), 3);
    EXPECT_EQ((*names)[0].name(), "batch");
}

TEST_F(DimnameTest, DimIndexLookup) {
    auto t = tenzor::zeros({2, 3, 4});
    auto named = t.rename({Dimname("batch"), Dimname("height"), Dimname("width")});

    EXPECT_EQ(named.dim_index("batch"), 0);
    EXPECT_EQ(named.dim_index("height"), 1);
    EXPECT_EQ(named.dim_index("width"), 2);
}

TEST_F(DimnameTest, NameCountMismatchThrows) {
    auto t = tenzor::zeros({2, 3});

    // 3 names for 2D tensor should throw
    EXPECT_THROW(
        t.rename({Dimname("a"), Dimname("b"), Dimname("c")}),
        std::invalid_argument
    );
}

TEST_F(DimnameTest, ThreadSafeInterning) {
    // Multiple threads creating the same name should all get the same pointer
    constexpr int num_threads = 8;
    std::vector<std::thread> threads;
    std::vector<Dimname> results(num_threads);

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&results, i]() {
            results[i] = Dimname("shared_name");
        });
    }
    for (auto& t : threads) t.join();

    // All should be equal (same interned pointer)
    for (int i = 1; i < num_threads; i++) {
        EXPECT_EQ(results[0], results[i]);
    }
}
