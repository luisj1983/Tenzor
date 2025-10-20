/**
 * @file bert_example.cpp
 * @brief Example demonstrating BERT for text classification
 *
 * This example shows how to:
 * 1. Create and configure a BERT model for sentiment analysis
 * 2. Tokenize and prepare input text
 * 3. Train the model on sentiment classification
 * 4. Evaluate and make predictions
 * 5. Load pretrained weights (when available)
 */

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include "tenzor/models/bert.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/nn/activations/activations.hpp"

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::nn;

// ============================================================================
// Simple Tokenizer (In practice, use a proper BPE tokenizer like Hugging Face)
// ============================================================================

class SimpleTokenizer {
public:
    SimpleTokenizer(int64_t vocab_size) : vocab_size_(vocab_size) {
        // Special tokens
        cls_token_id_ = 101;    // [CLS]
        sep_token_id_ = 102;    // [SEP]
        pad_token_id_ = 0;      // [PAD]
        unk_token_id_ = 100;    // [UNK]
    }

    /**
     * @brief Tokenize text into token IDs
     *
     * This is a very simplified tokenizer. In practice, use:
     * - Hugging Face tokenizers
     * - SentencePiece
     * - Custom BPE/WordPiece implementations
     */
    auto tokenize(const std::string& text, int64_t max_length) -> std::vector<int64_t> {
        std::vector<int64_t> token_ids;

        // Add [CLS] token
        token_ids.push_back(cls_token_id_);

        // Simple word splitting (replace with proper tokenizer)
        std::istringstream iss(text);
        std::string word;
        while (iss >> word && token_ids.size() < max_length - 1) {
            // Hash word to get token ID (very simplified)
            int64_t token_id = std::hash<std::string>{}(word) % (vocab_size_ - 200) + 200;
            token_ids.push_back(token_id);
        }

        // Add [SEP] token
        token_ids.push_back(sep_token_id_);

        // Pad to max_length
        while (token_ids.size() < max_length) {
            token_ids.push_back(pad_token_id_);
        }

        return token_ids;
    }

    /**
     * @brief Create attention mask (1 for real tokens, 0 for padding)
     */
    auto create_attention_mask(const std::vector<int64_t>& token_ids) -> std::vector<float> {
        std::vector<float> mask;
        for (auto token_id : token_ids) {
            mask.push_back(token_id != pad_token_id_ ? 1.0f : 0.0f);
        }
        return mask;
    }

private:
    int64_t vocab_size_;
    int64_t cls_token_id_;
    int64_t sep_token_id_;
    int64_t pad_token_id_;
    int64_t unk_token_id_;
};

// ============================================================================
// Dataset
// ============================================================================

struct SentimentExample {
    std::string text;
    int64_t label;  // 0 = negative, 1 = positive
};

class SentimentDataset {
public:
    SentimentDataset() {
        // Sample sentiment data
        examples_ = {
            {"This movie was absolutely fantastic and I loved every moment", 1},
            {"Terrible waste of time, completely disappointed", 0},
            {"Great performance by the actors and excellent cinematography", 1},
            {"Boring and predictable plot with weak characters", 0},
            {"One of the best films I have ever seen", 1},
            {"Awful script and poor direction throughout", 0},
            {"Highly recommended for anyone who enjoys quality cinema", 1},
            {"Not worth watching, saved you some time", 0},
            {"Brilliant storytelling and emotional depth", 1},
            {"Poorly executed with numerous plot holes", 0},
            {"Outstanding cast and beautiful visuals", 1},
            {"Disappointing ending after a slow start", 0},
            {"Masterpiece of modern filmmaking", 1},
            {"Complete disaster from beginning to end", 0},
            {"Engaging and thought-provoking throughout", 1},
            {"Waste of money and time, avoid at all costs", 0},
        };
    }

    auto size() const -> size_t {
        return examples_.size();
    }

    auto operator[](size_t idx) const -> const SentimentExample& {
        return examples_[idx];
    }

    auto get_examples() const -> const std::vector<SentimentExample>& {
        return examples_;
    }

private:
    std::vector<SentimentExample> examples_;
};

// ============================================================================
// Training Functions
// ============================================================================

auto create_batch(const std::vector<SentimentExample>& examples,
                 const std::vector<size_t>& indices,
                 SimpleTokenizer& tokenizer,
                 int64_t max_length,
                 Device device)
    -> std::tuple<Variable, Tensor, Tensor> {

    int64_t batch_size = indices.size();

    // Tokenize all examples
    std::vector<std::vector<int64_t>> all_token_ids;
    std::vector<std::vector<float>> all_masks;
    std::vector<int64_t> all_labels;

    for (auto idx : indices) {
        auto token_ids = tokenizer.tokenize(examples[idx].text, max_length);
        auto mask = tokenizer.create_attention_mask(token_ids);

        all_token_ids.push_back(token_ids);
        all_masks.push_back(mask);
        all_labels.push_back(examples[idx].label);
    }

    // Create input tensor on CPU, fill data, then move to device
    Tensor input_ids_cpu({batch_size, max_length}, DType::Int64, Device::cpu());
    std::vector<int64_t> flat_ids;
    for (const auto& ids : all_token_ids) {
        flat_ids.insert(flat_ids.end(), ids.begin(), ids.end());
    }
    int64_t* input_ptr = const_cast<int64_t*>(input_ids_cpu.data<int64_t>());
    std::memcpy(input_ptr, flat_ids.data(), flat_ids.size() * sizeof(int64_t));
    Tensor input_ids = input_ids_cpu.to(device);

    // Create attention mask on CPU, fill data, then move to device
    Tensor attention_mask_cpu({batch_size, max_length}, DType::Float32, Device::cpu());
    std::vector<float> flat_masks;
    for (const auto& mask : all_masks) {
        flat_masks.insert(flat_masks.end(), mask.begin(), mask.end());
    }
    float* mask_ptr = const_cast<float*>(attention_mask_cpu.data<float>());
    std::memcpy(mask_ptr, flat_masks.data(), flat_masks.size() * sizeof(float));
    Tensor attention_mask = attention_mask_cpu.to(device);

    // Create labels on CPU, fill data, then move to device
    Tensor labels_cpu({batch_size}, DType::Int64, Device::cpu());
    int64_t* labels_ptr = const_cast<int64_t*>(labels_cpu.data<int64_t>());
    std::memcpy(labels_ptr, all_labels.data(), all_labels.size() * sizeof(int64_t));
    Tensor labels = labels_cpu.to(device);

    return {Variable(input_ids, true), attention_mask, labels};
}

auto train_epoch(BertForSequenceClassification& model,
                SentimentDataset& dataset,
                SimpleTokenizer& tokenizer,
                optim::Adam& optimizer,
                int64_t batch_size,
                int64_t max_length,
                Device device) -> double {

    model.train();
    double total_loss = 0.0;
    int num_batches = 0;

    // Create indices and shuffle
    std::vector<size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Process batches
    for (size_t start = 0; start < indices.size(); start += batch_size) {
        size_t end = std::min(start + batch_size, indices.size());
        std::vector<size_t> batch_indices(indices.begin() + start, indices.begin() + end);

        // Create batch
        auto [input_ids, attention_mask, labels] = create_batch(
            dataset.get_examples(), batch_indices, tokenizer, max_length, device);

        // Forward pass
        auto logits = model.forward(input_ids, attention_mask, Variable{});

        // Compute loss (simple mean squared error for demonstration)
        // In practice, use proper cross-entropy loss
        Variable labels_var(labels, false);
        auto diff = logits - labels_var;
        auto squared_tensor = pow(diff.tensor(), 2.0f);
        auto mean_tensor = mean(squared_tensor);
        auto loss = Variable(mean_tensor, true);

        // Backward pass
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        auto loss_cpu = loss.tensor().to(Device::cpu());
        float* loss_data = const_cast<float*>(loss_cpu.data<float>());
        total_loss += loss_data[0];
        num_batches++;
    }

    return total_loss / num_batches;
}

auto evaluate(BertForSequenceClassification& model,
             SentimentDataset& dataset,
             SimpleTokenizer& tokenizer,
             int64_t batch_size,
             int64_t max_length,
             Device device) -> double {

    model.eval();
    int correct = 0;
    int total = 0;

    std::vector<size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (size_t start = 0; start < indices.size(); start += batch_size) {
        size_t end = std::min(start + batch_size, indices.size());
        std::vector<size_t> batch_indices(indices.begin() + start, indices.begin() + end);

        auto [input_ids, attention_mask, labels] = create_batch(
            dataset.get_examples(), batch_indices, tokenizer, max_length, device);

        // Forward pass
        auto logits = model.forward(input_ids, attention_mask, Variable{});

        // Get predictions - move to CPU for comparison
        auto predictions = argmax(logits.tensor(), 1).to(Device::cpu());
        auto labels_cpu = labels.to(Device::cpu());

        // Count correct predictions
        const int64_t* pred_data = predictions.data<int64_t>();
        const int64_t* label_data = labels_cpu.data<int64_t>();

        for (int64_t i = 0; i < predictions.shape()[0]; ++i) {
            if (pred_data[i] == label_data[i]) {
                correct++;
            }
            total++;
        }
    }

    return static_cast<double>(correct) / total;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "=== BERT Text Classification Example ===" << std::endl;
    std::cout << std::endl;

    // Configuration
    int64_t max_length = 32;
    int64_t batch_size = 4;
    int64_t num_epochs = 10;
    double learning_rate = 2e-5;
    Device device = Device::cpu();

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Max sequence length: " << max_length << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Number of epochs: " << num_epochs << std::endl;
    std::cout << "  Learning rate: " << learning_rate << std::endl;
    std::cout << "  Device: " << device.to_string() << std::endl;
    std::cout << std::endl;

    // Create small BERT configuration for quick training
    BertConfig config;
    config.vocab_size = 5000;
    config.hidden_size = 128;
    config.num_hidden_layers = 2;
    config.num_attention_heads = 4;
    config.intermediate_size = 512;
    config.max_position_embeddings = 64;
    config.hidden_dropout_prob = 0.1;
    config.attention_probs_dropout_prob = 0.1;

    std::cout << "BERT Configuration:" << std::endl;
    std::cout << "  Vocabulary size: " << config.vocab_size << std::endl;
    std::cout << "  Hidden size: " << config.hidden_size << std::endl;
    std::cout << "  Number of layers: " << config.num_hidden_layers << std::endl;
    std::cout << "  Attention heads: " << config.num_attention_heads << std::endl;
    std::cout << std::endl;

    // Create model
    int64_t num_labels = 2;  // Binary classification (negative/positive)
    auto model = BertForSequenceClassification(config, num_labels);
    model.to(device);

    std::cout << "Created BERT model for sequence classification" << std::endl;
    std::cout << "Number of labels: " << num_labels << std::endl;

    // Count parameters
    auto params = model.parameters();
    size_t total_params = 0;
    for (const auto& param : params) {
        size_t param_size = 1;
        for (auto dim : param->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }
    std::cout << "Total parameters: " << total_params << std::endl;
    std::cout << std::endl;

    // Create optimizer
    auto optimizer = optim::Adam(params, learning_rate);

    // Create dataset and tokenizer
    auto dataset = SentimentDataset();
    auto tokenizer = SimpleTokenizer(config.vocab_size);

    std::cout << "Dataset size: " << dataset.size() << " examples" << std::endl;
    std::cout << std::endl;

    // Display sample data
    std::cout << "Sample training examples:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(3), dataset.size()); ++i) {
        const auto& example = dataset[i];
        std::cout << "  [" << (example.label == 1 ? "POSITIVE" : "NEGATIVE") << "] "
                  << example.text << std::endl;
    }
    std::cout << std::endl;

    // Training loop
    std::cout << "Starting training..." << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (int64_t epoch = 1; epoch <= num_epochs; ++epoch) {
        auto epoch_loss = train_epoch(model, dataset, tokenizer, optimizer,
                                     batch_size, max_length, device);

        auto accuracy = evaluate(model, dataset, tokenizer,
                                batch_size, max_length, device);

        std::cout << "Epoch " << std::setw(2) << epoch << "/" << num_epochs
                  << " | Loss: " << std::fixed << std::setprecision(4) << epoch_loss
                  << " | Accuracy: " << std::setprecision(2) << (accuracy * 100) << "%"
                  << std::endl;
    }

    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::endl;

    // Final evaluation
    auto final_accuracy = evaluate(model, dataset, tokenizer,
                                   batch_size, max_length, device);
    std::cout << "Final accuracy: " << std::fixed << std::setprecision(2)
              << (final_accuracy * 100) << "%" << std::endl;
    std::cout << std::endl;

    // Make predictions on new examples
    std::cout << "Making predictions on new examples:" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    model.eval();

    std::vector<std::string> test_texts = {
        "This is an amazing and wonderful experience",
        "Absolutely terrible and horrible quality",
        "Pretty good overall with some nice moments",
        "Not great but not terrible either"
    };

    for (const auto& text : test_texts) {
        // Tokenize
        auto token_ids = tokenizer.tokenize(text, max_length);
        auto mask = tokenizer.create_attention_mask(token_ids);

        // Create tensors on CPU, fill, then move to device
        Tensor input_ids_cpu({1, max_length}, DType::Int64, Device::cpu());
        int64_t* input_ptr = const_cast<int64_t*>(input_ids_cpu.data<int64_t>());
        std::memcpy(input_ptr, token_ids.data(), token_ids.size() * sizeof(int64_t));
        Tensor input_ids = input_ids_cpu.to(device);

        Tensor attention_mask_cpu({1, max_length}, DType::Float32, Device::cpu());
        float* mask_ptr = const_cast<float*>(attention_mask_cpu.data<float>());
        std::memcpy(mask_ptr, mask.data(), mask.size() * sizeof(float));
        Tensor attention_mask = attention_mask_cpu.to(device);

        // Predict
        auto logits = model.forward(Variable(input_ids, false), attention_mask, Variable{});
        auto probs = softmax(logits, 1);

        auto pred_tensor = argmax(logits.tensor(), 1).to(Device::cpu());
        auto prob_cpu = probs.tensor().to(Device::cpu());

        int64_t* pred_data = const_cast<int64_t*>(pred_tensor.data<int64_t>());
        int64_t prediction = pred_data[0];
        float* prob_data = const_cast<float*>(prob_cpu.data<float>());
        float confidence = prob_data[prediction];

        std::cout << "Text: " << text << std::endl;
        std::cout << "  Prediction: " << (prediction == 1 ? "POSITIVE" : "NEGATIVE")
                  << " (confidence: " << std::fixed << std::setprecision(2)
                  << (confidence * 100) << "%)" << std::endl;
    }

    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::endl;

    // Save model
    std::string model_path = "/tmp/bert_sentiment_model.pt";
    std::cout << "Saving model to: " << model_path << std::endl;
    model.save(model_path);
    std::cout << "Model saved successfully" << std::endl;
    std::cout << std::endl;

    // Demonstrate loading pretrained weights (when available)
    std::cout << "Note: To use pretrained BERT weights:" << std::endl;
    std::cout << "  1. Download model from Hugging Face Hub" << std::endl;
    std::cout << "  2. Place checkpoint in ~/.cache/tenzor/bert-base-uncased/" << std::endl;
    std::cout << "  3. Call ModelHub::load_pretrained_weights(model, \"bert-base-uncased\")" << std::endl;
    std::cout << std::endl;

    std::cout << "Example completed successfully!" << std::endl;

    return 0;
}
