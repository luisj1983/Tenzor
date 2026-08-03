/**
 * @file test_faster_rcnn.cpp
 * @brief Tests for Faster R-CNN object detection model
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "../../include/tenzor/models/faster_rcnn.hpp"

using namespace tenzor;
using namespace tenzor::models;

class FasterRCNNTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

TEST_P(FasterRCNNTest, FasterRCNNResNet50ForwardShape) {
    auto model = faster_rcnn_resnet50(91, false);
    model->to(device);
    Variable images(randn({2, 3, 800, 800}, DType::Float32, device), true);

    // forward() returns a dummy Variable, use forward_inference() instead
    model->eval();
    // Forward-only inference: the test asserts the detection count only, never
    // calling backward(). With the input's requires_grad=true, forward_inference
    // builds an autograd graph and saves every activation; on an 8 GB GPU the
    // 800x800 ResNet50 feature maps exhaust memory, and the resulting
    // allocation failure surfaces as a sticky CUDA error on the next cuBLAS
    // call (cublasCreate returns CUBLAS_STATUS_INVALID_VALUE). NoGradGuard is
    // the correct inference idiom (mirrors torch.no_grad()) — it disables grad
    // tracking only, so the detections.size() assertion is still exercised.
    NoGradGuard no_grad;
    auto detections = model->forward_inference(images);

    // Should return detections for each image
    EXPECT_EQ(detections.size(), 2);
}

TEST_P(FasterRCNNTest, FasterRCNNResNet50GradientFlow) {
    auto model = faster_rcnn_resnet50(91, false);
    model->to(device);
    model->train();

    Variable images(randn({1, 3, 800, 800}, DType::Float32, device), true);

    // Create dummy ground truth targets with valid box coordinates
    std::vector<std::unordered_map<std::string, Tensor>> targets(1);

    // Initialize boxes with valid coordinates [x1, y1, x2, y2] format
    // Create 5 dummy boxes at different locations (host write -> CPU then to(device))
    auto boxes = Tensor({5, 4}, DType::Float32, Device::cpu());
    auto boxes_data = boxes.data<float>();
    boxes_data[0] = 10.0f;  boxes_data[1] = 10.0f;  boxes_data[2] = 100.0f; boxes_data[3] = 100.0f;
    boxes_data[4] = 150.0f; boxes_data[5] = 150.0f; boxes_data[6] = 250.0f; boxes_data[7] = 250.0f;
    boxes_data[8] = 300.0f; boxes_data[9] = 300.0f; boxes_data[10] = 400.0f; boxes_data[11] = 400.0f;
    boxes_data[12] = 450.0f; boxes_data[13] = 450.0f; boxes_data[14] = 550.0f; boxes_data[15] = 550.0f;
    boxes_data[16] = 600.0f; boxes_data[17] = 600.0f; boxes_data[18] = 700.0f; boxes_data[19] = 700.0f;
    boxes = boxes.to(device);

    // Initialize labels with valid class indices (host write -> CPU then to(device))
    auto labels = Tensor({5}, DType::Int64, Device::cpu());
    auto labels_data = labels.data<int64_t>();
    labels_data[0] = 1;
    labels_data[1] = 2;
    labels_data[2] = 3;
    labels_data[3] = 4;
    labels_data[4] = 5;
    labels = labels.to(device);

    targets[0]["boxes"] = boxes;
    targets[0]["labels"] = labels;

    auto losses = model->forward_train(images, targets);

    // Check that losses are returned
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(FasterRCNNTest, FasterRCNNResNet101ForwardShape) {
    auto model = faster_rcnn_resnet101(91, false);
    model->to(device);
    Variable images(randn({1, 3, 800, 800}, DType::Float32, device), true);

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

TEST_P(FasterRCNNTest, FasterRCNNDifferentImageSizes) {
    auto model = faster_rcnn_resnet50(91, false);
    model->to(device);
    model->eval();

    // Test with 600x600
    Variable images_600(randn({1, 3, 600, 600}, DType::Float32, device), true);
    auto detections_600 = model->forward_inference(images_600);
    EXPECT_EQ(detections_600.size(), 1);

    // Test with 1024x1024
    Variable images_1024(randn({1, 3, 1024, 1024}, DType::Float32, device), true);
    auto detections_1024 = model->forward_inference(images_1024);
    EXPECT_EQ(detections_1024.size(), 1);
}

// G9 regression: forward_impl must pack detections into (N_total, 7), not
// return a dummy single-element zeros tensor.
TEST_P(FasterRCNNTest, ForwardImplPacks7Columns_G9) {
    auto model = faster_rcnn_resnet50(91, false);
    model->to(device);
    model->eval();

    // Two-image batch: (B=2, 3, 800, 800).
    Variable images(randn({2, 3, 800, 800}, DType::Float32, device), false);

    // Forward-only inference: the test asserts the output rank/column contract
    // only, never calling backward(). Even with the input's requires_grad
    // false, the model parameters require grad, so forward() builds an
    // autograd graph and saves every activation; on an 8 GB GPU the 800x800
    // ResNet50 feature maps exhaust memory, surfacing as a sticky CUDA error
    // on the next cuBLAS call (cublasCreate returns CUBLAS_STATUS_INVALID_VALUE).
    // NoGradGuard is the correct inference idiom (mirrors torch.no_grad()) — it
    // disables grad tracking only, so the (N, 7) shape contract is still
    // exercised.
    NoGradGuard no_grad;
    Variable output = model->forward(images);
    const auto& shape = output.tensor().shape();
    ASSERT_EQ(shape.size(), 2u) << "Expected (N, 7) — got rank " << shape.size();
    EXPECT_EQ(shape[1], 7)
        << "Each row must be (batch_idx, x1, y1, x2, y2, score, label)";
    // Empty output (shape[0] == 0) is acceptable for an untrained model with
    // post-NMS filtering, but the rank/column count contract must hold.
}

// G8 regression: ResNet::out_channels() must reflect block type.
TEST_P(FasterRCNNTest, ResNetOutChannelsPerVariant_G8) {
    auto r18 = resnet18(1000, false);
    auto r34 = resnet34(1000, false);
    auto r50 = resnet50(1000, false);
    auto r101 = resnet101(1000, false);
    auto r152 = resnet152(1000, false);

    // BasicBlock (expansion=1) → 512×1 = 512.
    EXPECT_EQ(r18->out_channels(), 512);
    EXPECT_EQ(r34->out_channels(), 512);

    // Bottleneck (expansion=4) → 512×4 = 2048.
    EXPECT_EQ(r50->out_channels(), 2048);
    EXPECT_EQ(r101->out_channels(), 2048);
    EXPECT_EQ(r152->out_channels(), 2048);
}

// G8 regression: Faster R-CNN must size its RPN/ROI head from the backbone's
// real terminal channel count. Before the fix, the model assumed 2048 even
// for ResNet-18/34 (which output 512), causing a Conv2d shape mismatch at
// the first RPN forward.
TEST_P(FasterRCNNTest, FasterRCNNResNet18Backbone_G8) {
    // Construct Faster R-CNN with a BasicBlock backbone (ResNet-18).
    auto backbone = resnet18(1000, false);
    auto model = std::make_shared<FasterRCNN>(backbone, /*num_classes=*/91);
    model->to(device);
    model->eval();

    // The forward pass must not throw on channel mismatch. With 256x256 input
    // ResNet-18's layer4 produces a (B, 512, 8, 8) feature map; the RPN's
    // first conv is now 512→256 (instead of broken 2048→256).
    Variable images(randn({1, 3, 256, 256}, DType::Float32, device), false);
    EXPECT_NO_THROW({
        auto detections = model->forward_inference(images);
        EXPECT_EQ(detections.size(), 1);
    });
}

INSTANTIATE_BACKEND_TESTS(FasterRCNNTest);
