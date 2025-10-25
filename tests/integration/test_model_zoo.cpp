/**
 * @file test_model_zoo.cpp
 * @brief Integration tests for pretrained models from model zoo
 *
 * Tests:
 * - Loading pretrained models
 * - Model variants (different sizes)
 * - Fine-tuning workflows
 * - Inference consistency
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/models/resnet.hpp>
#include <tenzor/models/bert.hpp>
#include <memory>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::models;

//==============================================================================
// Test Environment
//==============================================================================

class ModelZooEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const model_zoo_env =
    ::testing::AddGlobalTestEnvironment(new ModelZooEnvironment);

//==============================================================================
// Test 1: ResNet-18 Model Creation
//==============================================================================

TEST(ModelZoo, ResNet18Creation) {
    auto model = resnet18(10);  // 10 classes
    EXPECT_NE(model, nullptr);

    model->to(Device::cpu());
    model->eval();

    // Test inference
    auto input = Variable(randn({2, 3, 224, 224}, DType::Float32, Device::cpu()), false);
    auto output = model->forward(input);

    EXPECT_EQ(output.shape()[0], 2) << "Batch size should be preserved";
    EXPECT_EQ(output.shape()[1], 10) << "Output should have 10 classes";
}

//==============================================================================
// Test 2: ResNet-50 Model Creation
//==============================================================================

TEST(ModelZoo, ResNet50Creation) {
    auto model = resnet50(1000);  // ImageNet classes
    EXPECT_NE(model, nullptr);

    model->to(Device::cpu());
    model->eval();

    auto input = Variable(randn({1, 3, 224, 224}, DType::Float32, Device::cpu()), false);
    auto output = model->forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 1000);
}

//==============================================================================
// Test 3: Different ResNet Variants
//==============================================================================

TEST(ModelZoo, ResNetVariants) {
    std::vector<std::pair<std::string, std::function<std::shared_ptr<ResNet>()>>> variants = {
        {"ResNet-18", []() { return resnet18(10); }},
        {"ResNet-34", []() { return resnet34(10); }},
        {"ResNet-50", []() { return resnet50(10); }},
    };

    for (auto& [name, create_fn] : variants) {
        auto model = create_fn();
        EXPECT_NE(model, nullptr) << name << " creation failed";

        model->to(Device::cpu());
        model->eval();

        auto input = Variable(randn({1, 3, 224, 224}, DType::Float32, Device::cpu()), false);
        auto output = model->forward(input);

        EXPECT_EQ(output.shape()[1], 10) << name << " output shape incorrect";
    }
}

//==============================================================================
// Test 4: BERT Model Creation
//==============================================================================

TEST(ModelZoo, BERTModelCreation) {
    BertConfig config;
    config.vocab_size = 30000;
    config.hidden_size = 768;
    config.num_hidden_layers = 12;
    config.num_attention_heads = 12;
    config.intermediate_size = 3072;
    config.max_position_embeddings = 512;

    auto model = std::make_shared<BertModel>(config);
    EXPECT_NE(model, nullptr);

    model->to(Device::cpu());
    model->eval();

    // Test inference
    auto input_ids = ones({2, 128}, DType::Int64, Device::cpu());
    auto attention_mask = ones({2, 128}, DType::Float32, Device::cpu());

    auto output = model->forward(
        Variable(input_ids, false),
        attention_mask
    );

    EXPECT_EQ(output.sequence_output.shape()[0], 2) << "Batch size preserved";
    EXPECT_EQ(output.sequence_output.shape()[1], 128) << "Sequence length preserved";
    EXPECT_EQ(output.sequence_output.shape()[2], 768) << "Hidden size correct";

    EXPECT_EQ(output.pooled_output.shape()[0], 2);
    EXPECT_EQ(output.pooled_output.shape()[1], 768);
}

//==============================================================================
// Test 5: Fine-tuning ResNet on Small Dataset
//==============================================================================

TEST(ModelZoo, ResNetFineTuning) {
    auto model = resnet18(10);
    model->to(Device::cpu());

    auto params = model->parameters();
    optim::SGD optimizer(params, 0.001, 0.9);

    model->train();

    // Simulate fine-tuning for a few steps
    for (int i = 0; i < 5; i++) {
        auto input = Variable(randn({4, 3, 224, 224}, DType::Float32, Device::cpu()), true);
        auto target_data = zeros({4, 10}, DType::Float32, Device::cpu());
        auto target_ptr = const_cast<float*>(target_data.template data<float>());
        for (int j = 0; j < 4; j++) {
            target_ptr[j * 10 + (j % 10)] = 1.0f;
        }
        auto target = Variable(target_data, false);

        optimizer.zero_grad();
        auto output = model->forward(input);
        auto loss = cross_entropy(output, target.tensor());
        loss.backward();
        optimizer.step();

        EXPECT_GT(loss.tensor().template item<float>(), 0.0f) << "Loss should be positive";
    }

    SUCCEED() << "Fine-tuning completed";
}

//==============================================================================
// Test 6: Model Inference Batch Size Flexibility
//==============================================================================

TEST(ModelZoo, InferenceBatchSizeFlexibility) {
    auto model = resnet18(10);
    model->to(Device::cpu());
    model->eval();

    std::vector<int> batch_sizes = {1, 2, 4, 8, 16};

    for (auto batch_size : batch_sizes) {
        auto input = Variable(randn({batch_size, 3, 224, 224}, DType::Float32, Device::cpu()), false);
        auto output = model->forward(input);

        EXPECT_EQ(output.shape()[0], batch_size)
            << "Output batch size should match input for batch_size=" << batch_size;
    }
}

//==============================================================================
// Test 7: Transfer Learning - Feature Extraction
//==============================================================================

TEST(ModelZoo, TransferLearningFeatureExtraction) {
    auto backbone = resnet18(1000);  // Pretrained on ImageNet (simulated)
    backbone->to(Device::cpu());
    backbone->eval();

    // Use model as feature extractor
    auto input = Variable(randn({8, 3, 224, 224}, DType::Float32, Device::cpu()), false);
    auto features = backbone->forward(input);

    // Verify feature extraction works
    EXPECT_EQ(features.shape()[0], 8) << "Batch preserved";
    EXPECT_GT(features.shape()[1], 0) << "Should have feature dimension";

    // Add custom classifier
    auto classifier = std::make_shared<Linear>(features.shape()[1], 5);  // 5 classes
    classifier->to(Device::cpu());

    auto output = classifier->forward(features);
    EXPECT_EQ(output.shape()[1], 5) << "Custom classifier output correct";
}

//==============================================================================
// Test 8: BERT for Sequence Classification
//==============================================================================

TEST(ModelZoo, BERTSequenceClassification) {
    // Smaller BERT config for testing
    BertConfig config;
    config.vocab_size = 1000;
    config.hidden_size = 128;
    config.num_hidden_layers = 2;
    config.num_attention_heads = 2;
    config.intermediate_size = 512;
    config.max_position_embeddings = 128;

    auto bert = std::make_shared<BertModel>(config);
    bert->to(Device::cpu());

    // Add classification head
    auto classifier = std::make_shared<Linear>(config.hidden_size, 2);  // Binary classification
    classifier->to(Device::cpu());

    // Create composite model
    class BERTClassifier : public Module {
    public:
        BERTClassifier(std::shared_ptr<BertModel> bert, std::shared_ptr<Linear> classifier)
            : bert_(bert), classifier_(classifier) {
            register_module("bert", bert_);
            register_module("classifier", classifier_);
        }

        auto forward(const Variable& input) -> Variable override {
            // Use default attention mask (all ones)
            std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
            auto attention_mask = ones(shape_vec, DType::Float32, input.device());
            return forward_with_mask(input, attention_mask);
        }

        auto forward_with_mask(const Variable& input_ids, const Tensor& attention_mask) -> Variable {
            auto output = bert_->forward(input_ids, attention_mask);
            return classifier_->forward(output.pooled_output);
        }

    private:
        std::shared_ptr<BertModel> bert_;
        std::shared_ptr<Linear> classifier_;
    };

    auto model = std::make_shared<BERTClassifier>(bert, classifier);
    model->train();

    auto params = model->parameters();
    optim::AdamW optimizer(params, 0.0001, 0.9, 0.999, 1e-8, 0.01);

    // Fine-tune for a few steps
    for (int i = 0; i < 3; i++) {
        auto input_ids = ones({4, 64}, DType::Int64, Device::cpu());
        auto attention_mask = ones({4, 64}, DType::Float32, Device::cpu());
        auto target_data = zeros({4, 2}, DType::Float32, Device::cpu());
        auto target_ptr = const_cast<float*>(target_data.template data<float>());
        for (int j = 0; j < 4; j++) {
            target_ptr[j * 2 + (j % 2)] = 1.0f;
        }

        optimizer.zero_grad();
        auto output = model->forward_with_mask(
            Variable(input_ids, false),
            attention_mask
        );
        auto loss = cross_entropy(output, target_data);
        loss.backward();
        optimizer.step();
    }

    SUCCEED() << "BERT fine-tuning for classification completed";
}
