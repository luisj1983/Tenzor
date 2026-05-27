// Regression tests for the showcase autograd training examples.
//
// Each TEST_F here drives one of the examples (examples/cpp/showcase/NN_xxx/
// autograd.cpp) via its extracted runner (autograd_runner.cpp) and asserts
// that loss decreases meaningfully over a small number of epochs.
//
// These tests guard against the class of regressions where backward returns
// zero or wildly-wrong gradients (e.g. severed grad_fn chain, missing dtype
// dispatch, dim-normalisation bugs, stride-from-shape kernel bugs) -- in
// those cases the example silently fails to converge while the kernel-level
// unit tests still pass. A small absolute-decrease threshold (loose enough
// to tolerate init variance and slow-convergence examples) catches the bug
// without becoming flaky.
//
// The HRM example has its own dedicated test in test_hrm_example.cpp.

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>

// One include per example runner — paths are relative to this file's location
// because tests/examples/CMakeLists.txt adds examples/cpp/showcase as an
// include directory on the test target.
#include "01_xor/autograd_runner.hpp"
#include "02_linear_regression/autograd_runner.hpp"
#include "03_binary_classification/autograd_runner.hpp"
#include "04_multiclass_classification/autograd_runner.hpp"
#include "05_convolutional/autograd_runner.hpp"
#include "06_autoencoder/autograd_runner.hpp"
#include "07_rnn_sequence/autograd_runner.hpp"
#include "08_batch_normalization/autograd_runner.hpp"
#include "09_dropout_regularization/autograd_runner.hpp"
#include "10_custom_loss/autograd_runner.hpp"
#include "11_chat_ai/autograd_runner.hpp"
#include "12_residual_network/autograd_runner.hpp"
#include "13_variational_autoencoder/autograd_runner.hpp"
#include "14_gan/autograd_runner.hpp"
#include "15_lstm_text/autograd_runner.hpp"
#include "16_self_attention/autograd_runner.hpp"
#include "17_word_embedding/autograd_runner.hpp"
#include "18_transfer_learning/autograd_runner.hpp"
#include "19_layer_normalization/autograd_runner.hpp"
#include "20_multitask_learning/autograd_runner.hpp"
#include "21_siamese_network/autograd_runner.hpp"

// KK.27: non-showcase runners (paths added via target_include_directories
// in tests/examples/CMakeLists.txt).
#include "vit_image_classification_runner.hpp"
#include "gradient_checkpointing_runner.hpp"

// NN.24: top-3 non-showcase training examples wired for regression
// coverage (transformer_seq2seq, yolo_object_detection,
// gpt_text_generation training-side variant).
#include "transformer_seq2seq_runner.hpp"
#include "yolo_object_detection_runner.hpp"
#include "gpt_text_generation_runner.hpp"

// RR.18 (audit-11): VAE / GRU / U-Net runners.
#include "vae_autoencoder_runner.hpp"
#include "gru_time_series_runner.hpp"
#include "unet_semantic_segmentation_runner.hpp"

class ExampleRegression : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }
};

// Rationale for the 0.01 threshold across the board:
// The bug pattern these tests guard against is a *severed* grad_fn chain or
// a kernel that returns zero — backward returns no gradient → SGD makes no
// progress → final loss ≈ initial loss. Even a small absolute reduction
// proves backward is moving the weights. The threshold is intentionally
// loose so the suite stays green across init variance.

namespace {
constexpr double kMinLossDecrease = 0.01;
}

TEST_F(ExampleRegression, XorTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase01::run_xor_training(
        /*epochs=*/500, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "XOR autograd training did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, LinearRegressionTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase02::run_linear_regression_training(
        /*epochs=*/200, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "Linear regression did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, BinaryClassificationTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase03::run_binary_classification_training(
        /*epochs=*/100, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "Binary classification did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, MulticlassClassificationTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase04::run_multiclass_training(
        /*epochs=*/100, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "Multiclass did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, ConvolutionalTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase05::run_convolutional_training(
        /*epochs=*/30, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "CNN did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, AutoencoderTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase06::run_autoencoder_training(
        /*epochs=*/200, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "Autoencoder did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, RnnSequenceTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase07::run_rnn_training(
        /*epochs=*/100, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "RNN did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, BatchNormalizationTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase08::run_batchnorm_training(
        /*epochs=*/100, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "BatchNorm did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, DropoutTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase09::run_dropout_training(
        /*epochs=*/100, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "Dropout net did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, CustomLossTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase10::run_custom_loss_training(
        /*epochs=*/80, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "Custom (focal) loss did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, ChatAITrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase11::run_chat_ai_training(
        /*epochs=*/12, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "Chat AI (GRU+Bahdanau) did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, ResidualNetworkTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase12::run_resnet_training(
        /*epochs=*/300, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "ResNet did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, VAETrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase13::run_vae_training(
        /*epochs=*/200, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "VAE did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, GANTrains) {
    // For GAN we report d_loss_initial and d_loss_final. The discriminator
    // should drop loss as it learns to distinguish real from fake. Even if
    // training is unstable, *some* movement away from the random-init loss
    // is a clear sign that backward isn't returning zero.
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase14::run_gan_training(
        /*epochs=*/300, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    // GAN-specific: just check the d_loss moved at all (could go up or
    // down depending on adversarial dynamics). std::abs catches either
    // direction.
    EXPECT_GT(std::abs(initial - final_), kMinLossDecrease)
        << "GAN d_loss did not move meaningfully: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, LSTMTextTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase15::run_lstm_training(
        /*epochs=*/100, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "LSTM did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, SelfAttentionTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase16::run_self_attention_training(
        /*epochs=*/200, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "Self-attention did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, WordEmbeddingTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase17::run_word_embedding_training(
        /*epochs=*/300, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "Word embedding did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, TransferLearningTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase18::run_transfer_learning_training(
        /*epochs_a=*/150, /*epochs_b=*/150,
        &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "Transfer learning (stage A) did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, LayerNormalizationTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase19::run_layernorm_training(
        /*epochs=*/100, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "LayerNorm did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, MultitaskLearningTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase20::run_multitask_training(
        /*epochs=*/100, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "Multitask did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, SiameseNetworkTrains) {
    // Siamese starts with a very small contrastive-loss value because the
    // randomly-initialised encoder already maps inputs to nearby points in
    // the small embedding space. We use a relative threshold here (final
    // must be at least 10% smaller than initial) instead of the absolute
    // 0.01 we use elsewhere.
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::showcase21::run_siamese_training(
        /*epochs=*/200, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    ASSERT_GT(initial, 0.0);
    EXPECT_LT(final_, initial * 0.9)
        << "Siamese did not reduce loss by at least 10%: initial=" << initial
        << " final=" << final_;
}

// KK.27: non-showcase example regressions. These exercise feature surfaces
// (MultiheadAttention + LayerNorm + GELU + AdamW for ViT; deep-ResNet +
// BatchNorm2d + Adam for gradient_checkpointing) that the showcase set
// doesn't combine in the same configuration, so a backward regression
// there could pass the showcase tests while breaking these training paths.
TEST_F(ExampleRegression, VitImageClassificationTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::vit_image_classification::
        run_vit_classification_training(
            /*epochs=*/5, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "ViT classification did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, GradientCheckpointingTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::gradient_checkpointing::
        run_gradient_checkpointing_training(
            /*num_iterations=*/8, &initial, &final_,
            tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "gradient_checkpointing did not reduce loss: initial=" << initial
        << " final=" << final_;
}

// NN.24: top-3 non-showcase training-example regressions. These
// exercise feature surfaces (Embedding + MultiheadAttention +
// LayerNorm + GELU for the two transformer-family models; Conv2d +
// GroupNorm + Mish + LeakyReLU + ResidualBlock for the YOLO backbone)
// that the showcase set doesn't combine in the same configuration, so
// a backward regression there could pass the showcase tests while
// breaking these training paths.
TEST_F(ExampleRegression, TransformerSeq2SeqTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::transformer_seq2seq::
        run_transformer_seq2seq_training(
            /*epochs=*/10, &initial, &final_,
            tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "transformer_seq2seq did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, YoloObjectDetectionTrains) {
    // YOLO uses an MSE-against-zeros surrogate loss on a randomly
    // initialised model, so monotonic decrease is not guaranteed on a
    // tiny number of steps. The plan's fallback applies: assert the
    // loss moved at all, which is enough to prove backward ran and
    // optimizer.step() updated weights.
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::yolo_object_detection::
        run_yolo_object_detection_training(
            /*num_iterations=*/10, &initial, &final_,
            tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_NE(initial, final_)
        << "yolo_object_detection training did not move loss at all: "
        << "initial=" << initial << " final=" << final_;
}

TEST_F(ExampleRegression, GptTextGenerationTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::gpt_text_generation::
        run_gpt_text_generation_training(
            /*epochs=*/10, &initial, &final_,
            tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "gpt_text_generation did not reduce loss: initial=" << initial
        << " final=" << final_;
}

// RR.18 (audit-11): VAE / GRU / U-Net regressions. VAE and GRU train on
// deterministic synthetic batches whose MSE-style loss is expected to
// decrease over a few Adam steps; U-Net's CE-against-random-labels
// surrogate is not guaranteed monotonic on a randomly-initialised conv
// backbone, so it uses the same `initial != final` fallback assertion as
// the YOLO runner from NN.24.
TEST_F(ExampleRegression, VaeAutoencoderTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::vae_autoencoder::run_vae_training(
        /*num_steps=*/10, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "VAE autoencoder did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, GruTimeSeriesTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::gru_time_series::run_gru_training(
        /*num_steps=*/10, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_GT(initial - final_, kMinLossDecrease)
        << "GRU time series did not reduce loss: initial=" << initial
        << " final=" << final_;
}

TEST_F(ExampleRegression, UnetSemanticSegmentationTrains) {
    double initial = -1.0, final_ = -1.0;
    int rc = tenzor::examples::unet_semantic_segmentation::run_unet_training(
        /*num_steps=*/10, &initial, &final_, tenzor::Device::cpu(), false);
    ASSERT_EQ(rc, 0);
    EXPECT_NE(initial, final_)
        << "U-Net training did not move loss at all: initial=" << initial
        << " final=" << final_;
}
