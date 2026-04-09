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

    // Create dummy ground truth targets with valid box coordinates
    std::vector<std::unordered_map<std::string, Tensor>> targets(1);

    // Initialize boxes with valid coordinates [x1, y1, x2, y2] format
    // Create 5 dummy boxes at different locations
    auto boxes = Tensor({5, 4}, DType::Float32, device_);
    auto boxes_data = boxes.data<float>();
    boxes_data[0] = 10.0f;  boxes_data[1] = 10.0f;  boxes_data[2] = 100.0f; boxes_data[3] = 100.0f;
    boxes_data[4] = 150.0f; boxes_data[5] = 150.0f; boxes_data[6] = 250.0f; boxes_data[7] = 250.0f;
    boxes_data[8] = 300.0f; boxes_data[9] = 300.0f; boxes_data[10] = 400.0f; boxes_data[11] = 400.0f;
    boxes_data[12] = 450.0f; boxes_data[13] = 450.0f; boxes_data[14] = 550.0f; boxes_data[15] = 550.0f;
    boxes_data[16] = 600.0f; boxes_data[17] = 600.0f; boxes_data[18] = 700.0f; boxes_data[19] = 700.0f;

    // Initialize labels with valid class indices
    auto labels = Tensor({5}, DType::Int64, device_);
    auto labels_data = labels.data<int64_t>();
    labels_data[0] = 1;
    labels_data[1] = 2;
    labels_data[2] = 3;
    labels_data[3] = 4;
    labels_data[4] = 5;

    targets[0]["boxes"] = boxes;
    targets[0]["labels"] = labels;

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
    ::testing::InitGoogleTest(&argc, argv);
    if (!::testing::GTEST_FLAG(list_tests)) {
        tenzor::initialize();
    }
    return RUN_ALL_TESTS();
}
