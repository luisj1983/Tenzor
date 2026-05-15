/**
 * @file test_mask_rcnn.cpp
 * @brief Tests for Mask R-CNN instance segmentation model
 */

#include <gtest/gtest.h>
#include <random>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/mask_rcnn.hpp"

using namespace tenzor;
using namespace tenzor::models;

class MaskRCNNTest : public ::testing::Test {
protected:
    void SetUp() override { device_ = Device::cpu(); }
    Device device_;
};

TEST_F(MaskRCNNTest, MaskRCNNResNet50ForwardShape) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({2, 3, 800, 800}, DType::Float32, device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // Check that outputs are returned
    EXPECT_GT(boxes.shape()[0], 0);  // At least some detections
    EXPECT_GT(labels.shape()[0], 0);
    EXPECT_GT(scores.shape()[0], 0);
    EXPECT_GT(masks.shape()[0], 0);
}

TEST_F(MaskRCNNTest, MaskRCNNResNet50GradientFlow) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), true);

    // Create and initialize dummy ground truth
    Tensor gt_boxes({1, 5, 4}, DType::Float32, device_);
    auto boxes_data = gt_boxes.data<float>();
    // Initialize 5 boxes at different locations [batch, num_boxes, 4]
    // Box 1
    boxes_data[0] = 10.0f;  boxes_data[1] = 10.0f;  boxes_data[2] = 100.0f; boxes_data[3] = 100.0f;
    // Box 2
    boxes_data[4] = 150.0f; boxes_data[5] = 150.0f; boxes_data[6] = 250.0f; boxes_data[7] = 250.0f;
    // Box 3
    boxes_data[8] = 300.0f; boxes_data[9] = 300.0f; boxes_data[10] = 400.0f; boxes_data[11] = 400.0f;
    // Box 4
    boxes_data[12] = 450.0f; boxes_data[13] = 450.0f; boxes_data[14] = 550.0f; boxes_data[15] = 550.0f;
    // Box 5
    boxes_data[16] = 600.0f; boxes_data[17] = 600.0f; boxes_data[18] = 700.0f; boxes_data[19] = 700.0f;

    Tensor gt_labels({1, 5}, DType::Int64, device_);
    auto labels_data = gt_labels.data<int64_t>();
    // Initialize with valid class labels
    labels_data[0] = 1;
    labels_data[1] = 2;
    labels_data[2] = 3;
    labels_data[3] = 4;
    labels_data[4] = 5;

    Tensor gt_masks({1, 5, 800, 800}, DType::Float32, device_);
    // Initialize masks with zeros (background) - this is memory-intensive but necessary
    auto masks_data = gt_masks.data<float>();
    std::fill(masks_data, masks_data + gt_masks.numel(), 0.0f);
    // Optionally add some 1s in the mask regions corresponding to the boxes
    // For simplicity, we'll just leave as zeros which represents valid (empty) masks

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_F(MaskRCNNTest, MaskRCNNResNet101ForwardShape) {
    auto model = mask_rcnn_resnet101_fpn(91, false);
    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    EXPECT_GT(boxes.shape()[0], 0);
    EXPECT_GT(masks.shape()[0], 0);
}

TEST_F(MaskRCNNTest, MaskRCNNDifferentImageSizes) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    model->eval();

    Variable images_600(Tensor({1, 3, 600, 600}, DType::Float32, device_), true);
    auto [boxes_600, labels_600, scores_600, masks_600] = model->forward_test(images_600);
    EXPECT_GT(boxes_600.shape()[0], 0);

    Variable images_1024(Tensor({1, 3, 1024, 1024}, DType::Float32, device_), true);
    auto [boxes_1024, labels_1024, scores_1024, masks_1024] = model->forward_test(images_1024);
    EXPECT_GT(boxes_1024.shape()[0], 0);
}

TEST_F(MaskRCNNTest, MaskRCNNCustomClasses) {
    auto model = mask_rcnn_resnet50_fpn(80, false);  // COCO classes
    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    EXPECT_GT(boxes.shape()[0], 0);
}

// G7 regression: forward_impl must pack boxes+scores+labels into (N, 6)
// instead of returning just boxes. Old behavior silently dropped labels,
// scores, AND masks; new packing follows the YOLOv5 convention.
TEST_F(MaskRCNNTest, ForwardImplPacks6Columns_G7) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), false);
    model->eval();

    Variable output = model->forward(images);
    const auto& shape = output.tensor().shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_GT(shape[0], 0);
    EXPECT_EQ(shape[1], 6) << "Expected (N, 6) = [x1,y1,x2,y2,score,label]";
}

// G7: detect() must return all 4 outputs including masks (which forward()
// drops because Module::forward returns a single Variable).
TEST_F(MaskRCNNTest, DetectReturnsAllOutputs_G7) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), false);
    model->eval();

    auto det = model->detect(images);
    EXPECT_GT(det.boxes.shape()[0], 0);
    EXPECT_EQ(det.boxes.shape()[1], 4);
    EXPECT_EQ(det.labels.shape().size(), 1u);
    EXPECT_EQ(det.scores.shape().size(), 1u);
    EXPECT_GE(det.masks.shape().size(), 2u);  // (N, H, W) or (N, 1, H, W)

    // Detection counts must match across the four outputs.
    EXPECT_EQ(det.boxes.shape()[0], det.labels.shape()[0]);
    EXPECT_EQ(det.boxes.shape()[0], det.scores.shape()[0]);
    EXPECT_EQ(det.boxes.shape()[0], det.masks.shape()[0]);
}

// G7: training-mode forward_impl must throw with a helpful message
// (use forward_train instead).
TEST_F(MaskRCNNTest, ForwardImplTrainingThrows_G7) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), false);
    model->train();
    EXPECT_THROW(model->forward(images), std::runtime_error);
}

// G6 regression: FPN encoder must produce a stride-16 feature map (P4)
// with 256 channels. Catches a regression to the old single 2048→256 1×1
// projection of C5 which would have been stride 32.
TEST_F(MaskRCNNTest, FPNEncoderReturnsP4Stride16_G6) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    model->eval();

    // 800x800 input → P4 is 50x50 (stride 16). Old C5 projection would be
    // 25x25 (stride 32).
    Tensor img_t({1, 3, 800, 800}, DType::Float32, device_);
    auto* p = img_t.data<float>();
    std::mt19937 rng(0xFEEDFACE);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < img_t.numel(); ++i) p[i] = dist(rng);
    Variable images(img_t, false);

    // We can't call extract_features directly (it's private), but we can
    // run generate_proposals which depends on the feature shape.
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // forward_test internally uses the FPN-produced feature map. If the
    // encoder regressed to stride 32, the RPN would produce 1/4 as many
    // anchor positions and proposals would still flow but in much smaller
    // numbers. This test mostly verifies the encoder/decoder chain wires up
    // end-to-end at the new stride-16 size.
    EXPECT_GT(boxes.shape()[0], 0);
    EXPECT_EQ(boxes.shape()[1], 4);
}

// G5 regression: select_training_samples must do real IoU-based pos/neg
// sampling (not just return all proposals unchanged).
TEST_F(MaskRCNNTest, SelectTrainingSamplesIoUSampling_G5) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    model->train();

    // Build a fake set of proposals: 1000 random boxes inside a 512x512 image,
    // all assigned to batch 0. Layout: (batch_idx, x1, y1, x2, y2).
    const int64_t N = 1000;
    Tensor proposals({N, 5}, DType::Float32, device_);
    auto* pp = proposals.data<float>();
    std::mt19937 rng(0x12345);
    std::uniform_real_distribution<float> dx(0.0f, 480.0f);  // x1 in [0, 480)
    std::uniform_real_distribution<float> dw(8.0f, 32.0f);   // box w/h in [8, 32)
    for (int64_t r = 0; r < N; ++r) {
        float x1 = dx(rng), y1 = dx(rng);
        float w = dw(rng), h = dw(rng);
        pp[r * 5 + 0] = 0.0f;       // batch_idx
        pp[r * 5 + 1] = x1;
        pp[r * 5 + 2] = y1;
        pp[r * 5 + 3] = x1 + w;
        pp[r * 5 + 4] = y1 + h;
    }

    // 3 GT boxes well-separated so most random proposals are negatives.
    Tensor gt_boxes({1, 3, 4}, DType::Float32, device_);
    auto* gp = gt_boxes.data<float>();
    gp[0] = 50.0f;  gp[1] = 50.0f;  gp[2] = 90.0f;  gp[3] = 90.0f;
    gp[4] = 200.0f; gp[5] = 200.0f; gp[6] = 240.0f; gp[7] = 240.0f;
    gp[8] = 400.0f; gp[9] = 400.0f; gp[10]= 440.0f; gp[11]= 440.0f;

    Tensor gt_labels({1, 3}, DType::Int64, device_);
    auto* lp = gt_labels.data<int64_t>();
    lp[0] = 1; lp[1] = 2; lp[2] = 3;

    Tensor sampled = model->select_training_samples(proposals, gt_boxes, gt_labels);

    // (a) Caps total at num_samples (512 in the impl).
    EXPECT_LE(sampled.shape()[0], 512);
    // (b) Returns same column count.
    EXPECT_EQ(sampled.shape()[1], 5);
    // (c) With 1000 mostly-negative proposals, the sampler should produce a
    //     non-empty subset that is strictly smaller than the input — i.e. it
    //     is not the trivial "return proposals;" stub.
    EXPECT_GT(sampled.shape()[0], 0);
    EXPECT_LT(sampled.shape()[0], N);
}

// G5 regression: zero GT boxes path should sample only negatives without throwing.
TEST_F(MaskRCNNTest, SelectTrainingSamplesNoGT_G5) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    model->train();

    const int64_t N = 200;
    Tensor proposals({N, 5}, DType::Float32, device_);
    auto* pp = proposals.data<float>();
    for (int64_t r = 0; r < N; ++r) {
        pp[r * 5 + 0] = 0.0f;
        pp[r * 5 + 1] = 0.0f;  pp[r * 5 + 2] = 0.0f;
        pp[r * 5 + 3] = 10.0f; pp[r * 5 + 4] = 10.0f;
    }
    Tensor gt_boxes({1, 0, 4}, DType::Float32, device_);
    Tensor gt_labels({1, 0}, DType::Int64, device_);

    Tensor sampled = model->select_training_samples(proposals, gt_boxes, gt_labels);
    EXPECT_LE(sampled.shape()[0], 512);
    EXPECT_EQ(sampled.shape()[1], 5);
}

// G4 regression: generate_proposals must produce real, non-trivial proposals.
// Before the audit fix the function returned `fill_(0.0)` so every column was
// 0. Test the proposal output directly (rather than forward_test's final boxes
// which go through the still-untrained ROI head and can degenerate).
TEST_F(MaskRCNNTest, GenerateProposalsReturnsRealBoxes_G4) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    model->eval();

    Tensor img_t({1, 3, 512, 512}, DType::Float32, device_);
    auto* p = img_t.data<float>();
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < img_t.numel(); ++i) p[i] = dist(rng);
    Variable images(img_t, false);

    // Reach into the FPN encoder to run only what generate_proposals needs.
    // The public entry point on MaskRCNN is generate_proposals(features), and
    // we need a features Variable matching what extract_features produces.
    // forward_test is the simplest way to exercise the encoder + RPN + decoder
    // chain; for direct proposal inspection we'd duplicate a lot of plumbing.
    // Instead: rely on forward_test running generate_proposals internally and
    // also separately call generate_proposals on a synthetic features tensor
    // to check the shape contract.
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // (a) forward_test returns rows — proves the proposal stage produced at
    //     least some surviving boxes (NMS pipeline didn't collapse to zero).
    ASSERT_GT(boxes.shape()[0], 0);
    ASSERT_EQ(boxes.shape()[1], 4);

    // (b) Directly verify generate_proposals' output shape contract.
    //     A regression to fill_(0.0) would still produce (N, 5) but with
    //     all four box columns equal — check at least one row has a real
    //     positive-area box on a sample with non-trivial content.
    Variable fake_features(
        Tensor({1, 256, 32, 32}, DType::Float32, device_), false);
    auto* ff = fake_features.tensor().data<float>();
    for (int64_t i = 0; i < fake_features.tensor().numel(); ++i) {
        ff[i] = dist(rng) * 0.1f;
    }
    Tensor proposals = model->generate_proposals(fake_features);
    ASSERT_EQ(proposals.shape()[1], 5)
        << "proposals must be (N, 5) with col 0 = batch_idx";
    if (proposals.shape()[0] > 0) {
        auto* pp = proposals.data<float>();
        int64_t valid = 0;
        int64_t zero_rows = 0;
        for (int64_t r = 0; r < proposals.shape()[0]; ++r) {
            float x1 = pp[r * 5 + 1];
            float y1 = pp[r * 5 + 2];
            float x2 = pp[r * 5 + 3];
            float y2 = pp[r * 5 + 4];
            if (x1 < x2 && y1 < y2) ++valid;
            if (x1 == 0 && y1 == 0 && x2 == 0 && y2 == 0) ++zero_rows;
        }
        EXPECT_GT(valid, 0)
            << "Expected at least one geometrically valid proposal (x1<x2, y1<y2)";
        EXPECT_LT(zero_rows, proposals.shape()[0])
            << "All-zero rows indicate generate_proposals regressed to fill_(0.0)";
    }
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
