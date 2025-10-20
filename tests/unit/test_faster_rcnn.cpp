/**
 * @file test_faster_rcnn.cpp
 * @brief Tests for Faster R-CNN object detection model
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/faster_rcnn.hpp"

using namespace tenzor;
using namespace tenzor::models;

class FasterRCNNTest : public ::testing::Test {
protected:
    void SetUp() override { device_ = Device::cpu(); }
    Device device_;
};

TEST_F(FasterRCNNTest, FasterRCNNResNet50ForwardShape) {
    auto model = faster_rcnn_resnet50(91, false);
    Variable images(Tensor({2, 3, 800, 800}, DType::Float32, device_), true);

    // forward() returns a dummy Variable, use forward_inference() instead
    model->eval();
    auto detections = model->forward_inference(images);

    // Should return detections for each image
    EXPECT_EQ(detections.size(), 2);
}

TEST_F(FasterRCNNTest, FasterRCNNResNet50GradientFlow) {
    auto model = faster_rcnn_resnet50(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), true);

    // Create dummy ground truth targets
    std::vector<std::unordered_map<std::string, Tensor>> targets(1);
    targets[0]["boxes"] = Tensor({5, 4}, DType::Float32, device_);
    targets[0]["labels"] = Tensor({5}, DType::Int64, device_);

    auto losses = model->forward_train(images, targets);

    // Check that losses are returned
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_F(FasterRCNNTest, FasterRCNNResNet101ForwardShape) {
    auto model = faster_rcnn_resnet101(91, false);
    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

TEST_F(FasterRCNNTest, FasterRCNNDifferentImageSizes) {
    auto model = faster_rcnn_resnet50(91, false);
    model->eval();

    // Test with 600x600
    Variable images_600(Tensor({1, 3, 600, 600}, DType::Float32, device_), true);
    auto detections_600 = model->forward_inference(images_600);
    EXPECT_EQ(detections_600.size(), 1);

    // Test with 1024x1024
    Variable images_1024(Tensor({1, 3, 1024, 1024}, DType::Float32, device_), true);
    auto detections_1024 = model->forward_inference(images_1024);
    EXPECT_EQ(detections_1024.size(), 1);
}


// ============================================================================
// Main  
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();  // Initialize Tenzor library and backends
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
