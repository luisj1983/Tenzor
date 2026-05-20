/**
 * @file test_non_differentiable_ops.cpp
 * @brief Audit E.7: intrinsically non-differentiable ops throw a typed
 *        `NonDifferentiable` exception from their Function backward
 *        rather than silently producing un-tracked Variables.
 *
 * The wrappers in src/autograd/function_new_ops.cpp document the
 * non-differentiability contract for `histc`, `bincount`, and
 * `searchsorted` — each backward() throws `tenzor::NonDifferentiable`
 * with an actionable message pointing at surrogate-gradient
 * alternatives. This test pins the contract: invoking backward()
 * through any of the three Functions throws.
 */

#include <gtest/gtest.h>

#include "tenzor/autograd/function.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/utils/error.hpp"

using namespace tenzor;

namespace {

class NonDifferentiableOpsTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

TEST_F(NonDifferentiableOpsTest, HistcBackwardThrowsTypedException) {
    HistcBackward fn;
    std::vector<Tensor> dummy_grads;
    dummy_grads.emplace_back(std::vector<int64_t>{1},
                              DType::Float32, Device::cpu());
    try {
        (void) fn.backward(std::move(dummy_grads));
        FAIL() << "expected NonDifferentiable, got no exception";
    } catch (const NonDifferentiable& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("histc"), std::string::npos)
            << "actual: " << msg;
        EXPECT_NE(msg.find("surrogate"), std::string::npos)
            << "actual: " << msg;
    }
}

TEST_F(NonDifferentiableOpsTest, BincountBackwardThrowsTypedException) {
    BincountBackward fn;
    std::vector<Tensor> dummy_grads;
    dummy_grads.emplace_back(std::vector<int64_t>{1},
                              DType::Float32, Device::cpu());
    try {
        (void) fn.backward(std::move(dummy_grads));
        FAIL() << "expected NonDifferentiable, got no exception";
    } catch (const NonDifferentiable& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("bincount"), std::string::npos)
            << "actual: " << msg;
    }
}

TEST_F(NonDifferentiableOpsTest, SearchSortedBackwardThrowsTypedException) {
    SearchSortedBackward fn;
    std::vector<Tensor> dummy_grads;
    dummy_grads.emplace_back(std::vector<int64_t>{1},
                              DType::Float32, Device::cpu());
    try {
        (void) fn.backward(std::move(dummy_grads));
        FAIL() << "expected NonDifferentiable, got no exception";
    } catch (const NonDifferentiable& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("searchsorted"), std::string::npos)
            << "actual: " << msg;
        EXPECT_NE(msg.find("soft-argmin"), std::string::npos)
            << "actual: " << msg;
    }
}

TEST_F(NonDifferentiableOpsTest, ForwardThrowsClearly) {
    // The Function-class forward() entry points are not the user-facing API
    // (the corresponding Tensor / Variable wrapper invokes the underlying
    // op directly); calling them through the Function should throw a
    // runtime_error with the "should not be called directly" message,
    // matching the convention of the other audit wrappers.
    HistcBackward h;
    EXPECT_THROW({
        (void) h.forward({});
    }, std::runtime_error);

    BincountBackward b;
    EXPECT_THROW({
        (void) b.forward({});
    }, std::runtime_error);

    SearchSortedBackward s;
    EXPECT_THROW({
        (void) s.forward({});
    }, std::runtime_error);

    MultinomialSampleBackward m;
    EXPECT_THROW({
        (void) m.forward({});
    }, std::runtime_error);

    BernoulliSampleBackward bern;
    EXPECT_THROW({
        (void) bern.forward({});
    }, std::runtime_error);
}

TEST_F(NonDifferentiableOpsTest, MultinomialSampleBackwardThrowsTypedException) {
    MultinomialSampleBackward fn;
    std::vector<Tensor> dummy_grads;
    dummy_grads.emplace_back(std::vector<int64_t>{1},
                              DType::Float32, Device::cpu());
    try {
        (void) fn.backward(std::move(dummy_grads));
        FAIL() << "expected NonDifferentiable, got no exception";
    } catch (const NonDifferentiable& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("multinomial"), std::string::npos)
            << "actual: " << msg;
        EXPECT_NE(msg.find("Gumbel-softmax"), std::string::npos)
            << "actual: " << msg;
    }
}

TEST_F(NonDifferentiableOpsTest, BernoulliSampleBackwardThrowsTypedException) {
    BernoulliSampleBackward fn;
    std::vector<Tensor> dummy_grads;
    dummy_grads.emplace_back(std::vector<int64_t>{1},
                              DType::Float32, Device::cpu());
    try {
        (void) fn.backward(std::move(dummy_grads));
        FAIL() << "expected NonDifferentiable, got no exception";
    } catch (const NonDifferentiable& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("bernoulli"), std::string::npos)
            << "actual: " << msg;
        EXPECT_NE(msg.find("Concrete"), std::string::npos)
            << "actual: " << msg;
    }
}

TEST_F(NonDifferentiableOpsTest, ArgmaxArgminBucketizeBackwardThrow) {
    auto check = [](Function& fn, const std::string& expected_op,
                    const std::string& expected_hint) {
        std::vector<Tensor> dummy_grads;
        dummy_grads.emplace_back(std::vector<int64_t>{1},
                                  DType::Float32, Device::cpu());
        try {
            (void) fn.backward(std::move(dummy_grads));
            FAIL() << "expected NonDifferentiable, got no exception";
        } catch (const NonDifferentiable& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find(expected_op), std::string::npos)
                << "actual: " << msg;
            EXPECT_NE(msg.find(expected_hint), std::string::npos)
                << "actual: " << msg;
        }
    };

    ArgmaxBackward amax;
    check(amax, "argmax", "soft-argmax");
    ArgminBackward amin;
    check(amin, "argmin", "soft-argmin");
    BucketizeBackward buck;
    check(buck, "bucketize", "sigmoid-of-distance");
}

TEST_F(NonDifferentiableOpsTest, ArgSortModeBackwardThrow) {
    auto check = [](Function& fn, const std::string& expected_op,
                    const std::string& expected_hint) {
        std::vector<Tensor> dummy_grads;
        dummy_grads.emplace_back(std::vector<int64_t>{1},
                                  DType::Float32, Device::cpu());
        try {
            (void) fn.backward(std::move(dummy_grads));
            FAIL() << "expected NonDifferentiable, got no exception";
        } catch (const NonDifferentiable& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find(expected_op), std::string::npos)
                << "actual: " << msg;
            EXPECT_NE(msg.find(expected_hint), std::string::npos)
                << "actual: " << msg;
        }
    };

    ArgSortBackward as;
    check(as, "argsort", "tenzor::sort");
}

}  // namespace
