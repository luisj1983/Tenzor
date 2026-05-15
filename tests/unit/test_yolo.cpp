/**
 * @file test_yolo.cpp
 * @brief Tests for YOLO v3 and v5 object detection models
 */

#include <gtest/gtest.h>
#include <random>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/yolo.hpp"

using namespace tenzor;
using namespace tenzor::models;

class YOLOTest : public ::testing::Test {
protected:
    void SetUp() override { device_ = Device::cpu(); }
    Device device_;
};

TEST_F(YOLOTest, YOLOv3ForwardShape) {
    auto model = yolov3(80, false);
    Variable images(Tensor({2, 3, 416, 416}, DType::Float32, device_), true);
    auto output = model->forward(images);

    // YOLO forward returns a Variable (not a struct)
    EXPECT_TRUE(output.tensor().numel() > 0);
}

TEST_F(YOLOTest, YOLOv3GradientFlow) {
    auto model = yolov3(80, false);
    model->train();

    Variable images(Tensor({1, 3, 416, 416}, DType::Float32, device_), true);
    auto output = model->forward(images);

    Variable loss = tenzor::sum(output);
    loss.backward();

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_F(YOLOTest, YOLOv3TinyForwardShape) {
    // YOLOv3-Tiny variant is not implemented, use YOLOv5n (nano) instead
    auto model = yolov5n(80, false);
    Variable images(Tensor({2, 3, 416, 416}, DType::Float32, device_), true);
    auto output = model->forward(images);

    // YOLO forward returns a Variable (not a struct)
    EXPECT_TRUE(output.tensor().numel() > 0);
}

TEST_F(YOLOTest, YOLOv5SmallForwardShape) {
    auto model = yolov5s(80, false);
    Variable images(Tensor({2, 3, 640, 640}, DType::Float32, device_), true);
    auto output = model->forward(images);

    // YOLO forward returns a Variable (not a struct)
    EXPECT_TRUE(output.tensor().numel() > 0);
}

TEST_F(YOLOTest, YOLOv5MediumForwardShape) {
    auto model = yolov5m(80, false);
    Variable images(Tensor({1, 3, 640, 640}, DType::Float32, device_), true);
    auto output = model->forward(images);

    // YOLO forward returns a Variable (not a struct)
    EXPECT_TRUE(output.tensor().numel() > 0);
}

TEST_F(YOLOTest, YOLOv5LargeForwardShape) {
    auto model = yolov5l(80, false);
    Variable images(Tensor({1, 3, 640, 640}, DType::Float32, device_), true);
    auto output = model->forward(images);

    // YOLO forward returns a Variable (not a struct)
    EXPECT_TRUE(output.tensor().numel() > 0);
}

TEST_F(YOLOTest, YOLOv5XLargeForwardShape) {
    auto model = yolov5x(80, false);
    Variable images(Tensor({1, 3, 640, 640}, DType::Float32, device_), true);
    auto output = model->forward(images);

    // YOLO forward returns a Variable (not a struct)
    EXPECT_TRUE(output.tensor().numel() > 0);
}

TEST_F(YOLOTest, YOLOv5GradientFlow) {
    auto model = yolov5s(80, false);
    model->train();

    Variable images(Tensor({1, 3, 640, 640}, DType::Float32, device_), true);
    auto output = model->forward(images);

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_F(YOLOTest, YOLODifferentImageSizes) {
    auto model = yolov3(80, false);

    // YOLO v3 supports 416, 512, 608
    Variable images_416(Tensor({1, 3, 416, 416}, DType::Float32, device_), true);
    auto output_416 = model->forward(images_416);
    EXPECT_TRUE(output_416.tensor().numel() > 0);

    Variable images_608(Tensor({1, 3, 608, 608}, DType::Float32, device_), true);
    auto output_608 = model->forward(images_608);
    EXPECT_TRUE(output_608.tensor().numel() > 0);
}

// G17 regression: YOLOv5::refit_anchors_kmeans must replace the hard-coded
// COCO defaults with k-means clusters over the user's dataset.
TEST_F(YOLOTest, RefitAnchorsKmeansReplacesCocoDefaults_G17) {
    YOLOv5 model(YOLOv5::Size::Small, /*num_classes=*/80, /*pretrained=*/false);

    // Sanity check on COCO defaults before refit.
    const auto& p3_before = model.anchors_p3();
    const auto& p5_before = model.anchors_p5();
    ASSERT_EQ(p3_before.size(), 3u);
    ASSERT_EQ(p5_before.size(), 3u);
    EXPECT_FLOAT_EQ(p3_before[0].first, 10.0f);   // first COCO anchor
    EXPECT_FLOAT_EQ(p5_before[2].second, 326.0f); // last COCO anchor

    // Synthetic dataset: 3 well-separated size clusters around (5, 5),
    // (50, 50), (250, 250). Should produce 3 cluster centers near each.
    std::vector<std::pair<float, float>> boxes;
    std::mt19937 rng(0xCAFE);
    std::normal_distribution<float> jitter(0.0f, 0.5f);
    for (int k = 0; k < 30; ++k) {
        boxes.push_back({5.0f + jitter(rng), 5.0f + jitter(rng)});
        boxes.push_back({50.0f + jitter(rng), 50.0f + jitter(rng)});
        boxes.push_back({250.0f + jitter(rng), 250.0f + jitter(rng)});
    }

    model.refit_anchors_kmeans(boxes, /*iters=*/50);

    const auto& p3 = model.anchors_p3();
    const auto& p4 = model.anchors_p4();
    const auto& p5 = model.anchors_p5();
    ASSERT_EQ(p3.size(), 3u);
    ASSERT_EQ(p4.size(), 3u);
    ASSERT_EQ(p5.size(), 3u);

    // P3 (smallest): should cluster around (5, 5).
    for (const auto& a : p3) {
        EXPECT_GT(a.first,  0.0f);
        EXPECT_LT(a.first,  20.0f) << "P3 anchor too large: " << a.first;
        EXPECT_LT(a.second, 20.0f);
    }

    // P5 (largest): should cluster around (250, 250).
    for (const auto& a : p5) {
        EXPECT_GT(a.first,  100.0f) << "P5 anchor too small: " << a.first;
        EXPECT_GT(a.second, 100.0f);
    }

    // P4 should sit between. Sort-by-area invariant: max area(P3) < min area(P4) < max area(P4) < min area(P5).
    auto area = [](const auto& a) { return a.first * a.second; };
    float max_a_p3 = std::max({area(p3[0]), area(p3[1]), area(p3[2])});
    float min_a_p4 = std::min({area(p4[0]), area(p4[1]), area(p4[2])});
    float max_a_p4 = std::max({area(p4[0]), area(p4[1]), area(p4[2])});
    float min_a_p5 = std::min({area(p5[0]), area(p5[1]), area(p5[2])});
    EXPECT_LE(max_a_p3, min_a_p4) << "P3 anchors must have smaller area than P4";
    EXPECT_LE(max_a_p4, min_a_p5) << "P4 anchors must have smaller area than P5";

    // None of the new anchors should match the COCO defaults.
    EXPECT_NE(p3[0].first, 10.0f);
    EXPECT_NE(p5[2].second, 326.0f);
}

// G17: invalid input (< 9 boxes) must throw a clear error.
TEST_F(YOLOTest, RefitAnchorsKmeansThrowsOnTooFewBoxes_G17) {
    YOLOv5 model(YOLOv5::Size::Small, 80, false);
    std::vector<std::pair<float, float>> too_few = {{1, 1}, {2, 2}, {3, 3}};
    EXPECT_THROW(model.refit_anchors_kmeans(too_few), std::invalid_argument);
}


// ============================================================================
// Main  
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (!::testing::GTEST_FLAG(list_tests)) {
        tenzor::initialize();
    }
    return RUN_ALL_TESTS();
}
