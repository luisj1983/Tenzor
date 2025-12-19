"""
BERT Text Classification Example

This comprehensive example demonstrates:
- BERT model architecture components
- Simple tokenization for demonstration
- Sentiment classification training
- Attention mask handling
- Position embeddings

Components used:
- Embedding, LayerNorm, MultiheadAttention
- Linear, Dropout, GELU activation
- CrossEntropyLoss, Adam optimizer
"""

import tenzor as tz
import numpy as np


# ============================================================================
# Simple Tokenizer (for demonstration - use HuggingFace in practice)
# ============================================================================

class SimpleTokenizer:
    """Simplified tokenizer for demonstration purposes"""

    def __init__(self, vocab_size=30522):
        self.vocab_size = vocab_size
        self.cls_token_id = 101   # [CLS]
        self.sep_token_id = 102   # [SEP]
        self.pad_token_id = 0     # [PAD]
        self.unk_token_id = 100   # [UNK]

    def tokenize(self, text, max_length=128):
        """Tokenize text into token IDs"""
        token_ids = [self.cls_token_id]

        # Simple word-based tokenization (hash to get token ID)
        words = text.lower().split()
        for word in words:
            if len(token_ids) >= max_length - 1:
                break
            # Hash word to get token ID
            token_id = hash(word) % (self.vocab_size - 200) + 200
            token_ids.append(abs(token_id))

        # Add [SEP] token
        token_ids.append(self.sep_token_id)

        # Pad to max_length
        while len(token_ids) < max_length:
            token_ids.append(self.pad_token_id)

        return token_ids[:max_length]

    def create_attention_mask(self, token_ids):
        """Create attention mask (1 for real tokens, 0 for padding)"""
        return [1.0 if tid != self.pad_token_id else 0.0 for tid in token_ids]


# ============================================================================
# Sentiment Dataset
# ============================================================================

class SentimentDataset:
    """Simple sentiment classification dataset"""

    def __init__(self):
        self.examples = [
            ("This movie was absolutely fantastic and I loved every moment", 1),
            ("Terrible waste of time, completely disappointed", 0),
            ("Great performance by the actors and excellent cinematography", 1),
            ("Boring and predictable plot with weak characters", 0),
            ("One of the best films I have ever seen", 1),
            ("Awful script and poor direction throughout", 0),
            ("Highly recommended for anyone who enjoys quality cinema", 1),
            ("Not worth watching, saved you some time", 0),
            ("Brilliant storytelling and emotional depth", 1),
            ("Poorly executed with numerous plot holes", 0),
            ("Outstanding cast and beautiful visuals", 1),
            ("Disappointing ending after a slow start", 0),
            ("Masterpiece of modern filmmaking", 1),
            ("Complete disaster from beginning to end", 0),
            ("Engaging and thought-provoking throughout", 1),
            ("Waste of money and time, avoid at all costs", 0),
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ============================================================================
# BERT-like Classifier (simplified)
# ============================================================================

class BERTClassifier:
    """
    Simplified BERT-style classifier for demonstration

    Architecture:
    - Token + Position Embeddings
    - Multiple Transformer Encoder layers
    - [CLS] token classification head
    """

    def __init__(self, vocab_size, hidden_size, num_heads, num_layers,
                 num_classes, max_seq_len=128, dropout=0.1):
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embeddings
        self.token_embedding = tz.nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = tz.nn.Embedding(max_seq_len, hidden_size)
        self.embedding_norm = tz.nn.LayerNorm([hidden_size])
        self.embedding_dropout = tz.nn.Dropout(dropout)

        # Transformer encoder layers
        self.attention_layers = []
        self.attention_norms = []
        self.ffn_layers = []
        self.ffn_norms = []

        for i in range(num_layers):
            self.attention_layers.append(
                tz.nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
            )
            self.attention_norms.append(tz.nn.LayerNorm([hidden_size]))

            # Feed-forward network
            self.ffn_layers.append({
                'fc1': tz.nn.Linear(hidden_size, hidden_size * 4),
                'fc2': tz.nn.Linear(hidden_size * 4, hidden_size),
                'dropout': tz.nn.Dropout(dropout)
            })
            self.ffn_norms.append(tz.nn.LayerNorm([hidden_size]))

        # Classification head
        self.classifier = tz.nn.Linear(hidden_size, num_classes)
        self.classifier_dropout = tz.nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len] (optional)

        Returns:
            Logits for classification [batch, num_classes]
        """
        batch_size = input_ids.tensor().shape[0]
        seq_len = input_ids.tensor().shape[1]

        # Create position IDs
        positions = np.arange(seq_len, dtype=np.int64)
        positions = np.tile(positions, (batch_size, 1))
        position_ids = tz.Variable(tz.Tensor.from_numpy(positions), requires_grad=False)

        # Token embeddings + position embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)

        # Combine embeddings
        hidden = tz.Variable(
            token_emb.tensor() + pos_emb.tensor(),
            requires_grad=True
        )
        hidden = self.embedding_norm(hidden)
        hidden = self.embedding_dropout(hidden)

        # Transformer encoder layers
        for i in range(self.num_layers):
            # Self-attention with residual (single input for self-attention)
            attn_out = self.attention_layers[i](hidden)
            hidden = tz.Variable(hidden.tensor() + attn_out.tensor(), requires_grad=True)
            hidden = self.attention_norms[i](hidden)

            # Feed-forward with residual
            ffn = self.ffn_layers[i]
            ffn_out = ffn['fc1'](hidden)
            ffn_out = tz.nn.gelu(ffn_out)
            ffn_out = ffn['fc2'](ffn_out)
            ffn_out = ffn['dropout'](ffn_out)

            hidden = tz.Variable(hidden.tensor() + ffn_out.tensor(), requires_grad=True)
            hidden = self.ffn_norms[i](hidden)

        # Extract [CLS] token representation (first token)
        hidden_np = hidden.tensor().numpy()
        cls_hidden = hidden_np[:, 0, :]  # [batch, hidden_size]
        cls_var = tz.Variable(tz.Tensor.from_numpy(cls_hidden), requires_grad=True)

        # Classification
        cls_var = self.classifier_dropout(cls_var)
        logits = self.classifier(cls_var)

        return logits

    def parameters(self):
        """Get all trainable parameters"""
        params = []
        params.extend(self.token_embedding.parameters())
        params.extend(self.position_embedding.parameters())
        params.extend(self.embedding_norm.parameters())

        for i in range(self.num_layers):
            params.extend(self.attention_layers[i].parameters())
            params.extend(self.attention_norms[i].parameters())
            params.extend(self.ffn_layers[i]['fc1'].parameters())
            params.extend(self.ffn_layers[i]['fc2'].parameters())
            params.extend(self.ffn_norms[i].parameters())

        params.extend(self.classifier.parameters())
        return params

    def train(self):
        """Set to training mode"""
        self.embedding_dropout.train()
        self.classifier_dropout.train()
        for i in range(self.num_layers):
            self.ffn_layers[i]['dropout'].train()

    def eval(self):
        """Set to evaluation mode"""
        self.embedding_dropout.eval()
        self.classifier_dropout.eval()
        for i in range(self.num_layers):
            self.ffn_layers[i]['dropout'].eval()


# ============================================================================
# Training Function
# ============================================================================

def train_bert_classifier():
    """Train BERT classifier on sentiment data"""
    print("\n" + "=" * 60)
    print("Training BERT Classifier")
    print("=" * 60)

    # Configuration
    vocab_size = 30522
    hidden_size = 128  # Small for demo (BERT-base uses 768)
    num_heads = 4
    num_layers = 2     # Small for demo (BERT-base uses 12)
    num_classes = 2    # Binary classification
    max_seq_len = 32

    print("\nConfiguration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Attention heads: {num_heads}")
    print(f"  Encoder layers: {num_layers}")
    print(f"  Max sequence length: {max_seq_len}")

    # Create components
    tokenizer = SimpleTokenizer(vocab_size)
    dataset = SentimentDataset()
    model = BERTClassifier(vocab_size, hidden_size, num_heads, num_layers,
                           num_classes, max_seq_len)

    # Optimizer and loss
    params = model.parameters()
    optimizer = tz.optim.Adam(params, lr=0.001)
    criterion = tz.nn.CrossEntropyLoss()

    print(f"\nDataset size: {len(dataset)}")
    print(f"Parameters: {len(params)}")

    # Prepare all data
    all_input_ids = []
    all_labels = []

    for text, label in dataset.examples:
        token_ids = tokenizer.tokenize(text, max_seq_len)
        all_input_ids.append(token_ids)
        all_labels.append(label)

    input_ids_np = np.array(all_input_ids, dtype=np.int64)
    labels_np = np.array(all_labels, dtype=np.int64)

    # Training loop
    num_epochs = 10
    print(f"\nTraining for {num_epochs} epochs...")

    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        input_var = tz.Variable(tz.Tensor.from_numpy(input_ids_np), requires_grad=False)
        logits = model.forward(input_var)

        # Loss (CrossEntropyLoss expects Tensor for target, not Variable)
        labels_tensor = tz.Tensor.from_numpy(labels_np)
        loss = criterion(logits, labels_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        logits_np = logits.tensor().numpy()
        predictions = np.argmax(logits_np, axis=1)
        accuracy = np.mean(predictions == labels_np) * 100

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{num_epochs} | "
                  f"Loss: {loss.tensor().item():.4f} | "
                  f"Accuracy: {accuracy:.1f}%")

    # Evaluation
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)

    model.eval()

    with tz.no_grad():
        input_var = tz.Variable(tz.Tensor.from_numpy(input_ids_np), requires_grad=False)
        logits = model.forward(input_var)

        logits_np = logits.tensor().numpy()
        predictions = np.argmax(logits_np, axis=1)

        print("\nPredictions:")
        for i, (text, label) in enumerate(dataset.examples[:6]):
            pred = predictions[i]
            status = "CORRECT" if pred == label else "WRONG"
            sentiment = "Positive" if pred == 1 else "Negative"
            print(f"  [{status}] '{text[:40]}...' -> {sentiment}")

        final_accuracy = np.mean(predictions == labels_np) * 100
        print(f"\nFinal Accuracy: {final_accuracy:.1f}%")


# ============================================================================
# Component Demo
# ============================================================================

def demo_bert_components():
    """Demonstrate individual BERT components"""
    print("\n" + "=" * 60)
    print("BERT Components Demo")
    print("=" * 60)

    batch_size = 2
    seq_len = 16
    hidden_size = 64
    num_heads = 4

    # 1. Embedding layers
    print("\n[1] Embedding Layers")
    vocab_size = 1000

    token_emb = tz.nn.Embedding(vocab_size, hidden_size)
    pos_emb = tz.nn.Embedding(seq_len, hidden_size)

    # Create sample input
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int64)
    pos_ids = np.tile(np.arange(seq_len), (batch_size, 1)).astype(np.int64)

    input_var = tz.Variable(tz.Tensor.from_numpy(input_ids), requires_grad=False)
    pos_var = tz.Variable(tz.Tensor.from_numpy(pos_ids), requires_grad=False)

    token_output = token_emb(input_var)
    pos_output = pos_emb(pos_var)

    print(f"  Token embedding: {input_ids.shape} -> {token_output.tensor().shape}")
    print(f"  Position embedding: {pos_ids.shape} -> {pos_output.tensor().shape}")

    # Combined embeddings
    combined = tz.Variable(token_output.tensor() + pos_output.tensor(), requires_grad=True)
    print(f"  Combined: {combined.tensor().shape}")

    # 2. Layer Normalization
    print("\n[2] Layer Normalization")
    layer_norm = tz.nn.LayerNorm([hidden_size])
    normed = layer_norm(combined)
    print(f"  Input: {combined.tensor().shape} -> Output: {normed.tensor().shape}")

    # 3. Multi-head Attention
    print("\n[3] Multi-head Self-Attention")
    attention = tz.nn.MultiheadAttention(hidden_size, num_heads)
    attn_out = attention(normed)  # Single input for self-attention
    print(f"  Input: {normed.tensor().shape}")
    print(f"  Attention output: {attn_out.tensor().shape}")

    # 4. GELU Activation
    print("\n[4] GELU Activation")
    test_input = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    test_var = tz.Variable(tz.Tensor.from_numpy(test_input), requires_grad=True)
    gelu_out = tz.nn.gelu(test_var)
    print(f"  Input: {list(test_input)}")
    print(f"  GELU:  {[f'{x:.3f}' for x in gelu_out.tensor().numpy()]}")

    # 5. Dropout
    print("\n[5] Dropout")
    dropout = tz.nn.Dropout(0.1)
    dropout.train()
    dropped = dropout(normed)
    print(f"  Dropout(0.1) applied to: {normed.tensor().shape}")

    print("\nAll BERT components demonstrated successfully!")


# ============================================================================
# Main
# ============================================================================

def main():
    tz.initialize()

    print("=" * 60)
    print("   BERT Text Classification Example")
    print("=" * 60)

    print("\nComponents demonstrated:")
    print("  Embeddings: Token, Position")
    print("  Layers: MultiheadAttention, LayerNorm, Linear")
    print("  Activations: GELU")
    print("  Training: CrossEntropyLoss, Adam")

    demo_bert_components()
    train_bert_classifier()

    print("\n" + "=" * 60)
    print("   BERT example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
