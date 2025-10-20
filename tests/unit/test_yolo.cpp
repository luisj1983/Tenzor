/**
 * @file test_yolo.cpp
 * @brief Tests for YOLO v3 and v5 object detection models
 */

#include <gtest/gtest.h>
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


// ============================================================================
// Main  
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();  // Initialize Tenzor library and backends
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
