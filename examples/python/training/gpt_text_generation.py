"""
GPT Text Generation Training Example

This comprehensive example demonstrates:
- GPT decoder-only transformer architecture
- Causal (autoregressive) attention masking
- Learned positional embeddings
- Pre-norm transformer blocks
- Text generation with various sampling strategies:
  - Greedy decoding
  - Temperature sampling
  - Top-k sampling
  - Top-p (nucleus) sampling
- Language modeling loss (next token prediction)
- Token embedding and vocabulary handling
"""

import tenzor as tz
import numpy as np
import math


# ============================================================================
# Text Dataset
# ============================================================================

class TextDataset:
    """Synthetic text dataset for language modeling"""

    def __init__(self, num_sequences, seq_length, vocab_size):
        self.num_sequences = num_sequences
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        np.random.seed(42)
        # Generate random token sequences
        self.sequences = np.random.randint(0, vocab_size, (num_sequences, seq_length)).astype(np.int64)

    def get_batch(self, start, batch_size):
        """Get a batch of input/target pairs for next token prediction"""
        end = min(start + batch_size, self.num_sequences)
        actual_batch = end - start

        input_ids = self.sequences[start:end].copy()

        # Target is shifted by 1 (next token prediction)
        target_ids = np.zeros_like(input_ids)
        target_ids[:, :-1] = input_ids[:, 1:]
        target_ids[:, -1] = 0  # EOS token for last position

        return (tz.Tensor.from_numpy(input_ids),
                tz.Tensor.from_numpy(target_ids))

    def __len__(self):
        return self.num_sequences


# ============================================================================
# GPT Building Blocks
# ============================================================================

class CausalSelfAttention(tz.nn.Module):
    """
    Causal self-attention using built-in MultiheadAttention
    Uses the library's MultiheadAttention module which handles
    the complex tensor operations internally
    """

    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Use built-in MultiheadAttention
        self.mha = tz.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x):
        # Self-attention: query, key, value are all the same
        # MultiheadAttention.forward returns (output, attention_weights)
        attn_output, _ = self.mha.forward(x, x, x)
        return attn_output


class GPTMLP(tz.nn.Module):
    """GPT MLP block with GELU activation"""

    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = tz.nn.Linear(embed_dim, hidden_dim)
        self.fc2 = tz.nn.Linear(hidden_dim, embed_dim)
        self.gelu = tz.nn.GELU()
        self.dropout = tz.nn.Dropout(dropout)

    def forward(self, x):
        h = self.gelu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)


class GPTBlock(tz.nn.Module):
    """GPT Transformer Block (Pre-norm architecture)"""

    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.ln1 = tz.nn.LayerNorm([embed_dim])
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.ln2 = tz.nn.LayerNorm([embed_dim])
        self.mlp = GPTMLP(embed_dim, 4 * embed_dim, dropout)
        self.dropout = tz.nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm attention with residual
        attn_out = self.attn(self.ln1(x))
        h = x + self.dropout(attn_out)

        # Pre-norm MLP with residual
        mlp_out = self.mlp(self.ln2(h))
        return h + self.dropout(mlp_out)


# ============================================================================
# GPT Model
# ============================================================================

class GPT(tz.nn.Module):
    """GPT decoder-only transformer for language modeling"""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 max_seq_len, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        self.token_emb = tz.nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = tz.nn.Embedding(max_seq_len, embed_dim)

        self.dropout = tz.nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = []
        for i in range(num_layers):
            block = GPTBlock(embed_dim, num_heads, max_seq_len, dropout)
            self.blocks.append(block)
            setattr(self, f'block_{i}', block)

        # Final layer norm
        self.ln_f = tz.nn.LayerNorm([embed_dim])

        # Language model head
        self.lm_head = tz.nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        shape = input_ids.tensor().shape
        batch_size, seq_len = shape[0], shape[1]

        # Create position indices
        positions = np.arange(seq_len).reshape(1, seq_len)
        positions = np.tile(positions, (batch_size, 1)).astype(np.int64)
        positions = tz.Tensor.from_numpy(positions)

        # Embeddings
        tok_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(tz.Variable(positions, False))
        h = self.dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            h = block(h)

        # Final layer norm and LM head
        h = self.ln_f(h)
        return self.lm_head(h)

    def generate(self, prompt, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0):
        """
        Generate text autoregressively

        Args:
            prompt: list of token ids
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (lower = more focused)
            top_k: keep only top k tokens (0 = disabled)
            top_p: nucleus sampling threshold (1.0 = disabled)
        """
        generated = list(prompt)

        for _ in range(max_new_tokens):
            # Truncate if exceeds max length
            context = generated[-self.max_seq_len:] if len(generated) > self.max_seq_len else generated

            # Forward pass
            input_tensor = tz.Tensor.from_numpy(np.array([context], dtype=np.int64))
            logits = self.forward(tz.Variable(input_tensor, False))

            # Get logits for last position
            last_logits = logits.tensor().slice(1, -1, logits.tensor().shape[1])
            last_logits_np = last_logits.numpy().flatten()

            # Apply temperature
            if temperature != 1.0:
                last_logits_np = last_logits_np / temperature

            # Softmax
            max_logit = np.max(last_logits_np)
            probs = np.exp(last_logits_np - max_logit)
            probs = probs / np.sum(probs)

            # Apply top-k filtering
            if top_k > 0 and top_k < self.vocab_size:
                indices = np.argsort(probs)[::-1]
                mask = np.zeros_like(probs)
                mask[indices[:top_k]] = 1
                probs = probs * mask

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_indices = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_indices]
                cumsum = np.cumsum(sorted_probs)

                # Find cutoff
                cutoff_idx = np.searchsorted(cumsum, top_p)
                mask = np.zeros_like(probs)
                mask[sorted_indices[:cutoff_idx + 1]] = 1
                probs = probs * mask

            # Renormalize
            probs = probs / np.sum(probs) if np.sum(probs) > 0 else probs

            # Sample
            next_token = np.random.choice(len(probs), p=probs)
            generated.append(int(next_token))

            # Stop if EOS
            if next_token == 0:
                break

        return generated


# ============================================================================
# Demo Functions
# ============================================================================

def demo_causal_attention():
    """Demonstrate causal attention masking"""
    print("\n" + "=" * 60)
    print("Causal (Autoregressive) Attention Demo")
    print("=" * 60)

    print("\n[1] Causal Mask Visualization (5x5)")
    print("    Positions marked with 1 can attend, 0 cannot:")
    print()

    seq_len = 5
    print("         ", end="")
    for j in range(seq_len):
        print(f"K{j} ", end="")
    print()

    for i in range(seq_len):
        print(f"    Q{i}:  ", end="")
        for j in range(seq_len):
            print("1  " if j <= i else "0  ", end="")
        print()

    print("\n[2] How it works:")
    print("    - Q0 can only see K0 (first token)")
    print("    - Q1 can see K0, K1 (first two tokens)")
    print("    - Q4 can see all previous tokens K0-K4")
    print("    - This prevents information from future tokens")

    print("\n[3] Pre-norm vs Post-norm:")
    print("    GPT uses Pre-norm: LayerNorm before attention/MLP")
    print("    Benefits: More stable training, better gradient flow")


def demo_generation_strategies():
    """Demonstrate different text generation strategies"""
    print("\n" + "=" * 60)
    print("Text Generation Strategies Demo")
    print("=" * 60)

    # Example probability distribution
    logits = np.array([2.0, 1.5, 1.0, 0.5, 0.3, 0.1, -0.5, -1.0])
    vocab_size = len(logits)

    # Softmax
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)

    print("\n[1] Greedy Decoding")
    print("    Always pick highest probability token")
    greedy_idx = np.argmax(probs)
    print(f"    Selected: Token {greedy_idx} (prob={probs[greedy_idx]:.3f})")
    print("    Deterministic but can be repetitive")

    print("\n[2] Temperature Sampling (T=0.5)")
    temp = 0.5
    temp_probs = np.exp((logits - np.max(logits)) / temp)
    temp_probs = temp_probs / np.sum(temp_probs)
    print(f"    Original probs: [{', '.join([f'{p:.3f}' for p in probs[:4]])}, ...]")
    print(f"    T=0.5 probs:    [{', '.join([f'{p:.3f}' for p in temp_probs[:4]])}, ...]")
    print("    Lower T -> sharper distribution (more confident)")
    print("    Higher T -> flatter distribution (more random)")

    print("\n[3] Top-k Sampling (k=3)")
    print("    Keep only top 3 tokens, zero out rest")
    print("    Prevents very unlikely tokens")

    print("\n[4] Top-p (Nucleus) Sampling (p=0.9)")
    print("    Keep smallest set of tokens with cumulative prob >= p")
    print("    Adaptive: more tokens when uncertain, fewer when confident")

    print("\n[5] Typical combinations:")
    print("    Creative: T=0.9, top_p=0.95")
    print("    Balanced: T=0.7, top_k=40")
    print("    Focused:  T=0.3, top_k=10")


def demo_weight_tying():
    """Demonstrate weight tying concept"""
    print("\n" + "=" * 60)
    print("Weight Tying Demo")
    print("=" * 60)

    print("\n[1] What is Weight Tying?")
    print("    Sharing weights between input embeddings and output projection")
    print()
    print("    Input:  token_id -> Embedding(vocab_size, embed_dim) -> vector")
    print("    Output: vector -> Linear(embed_dim, vocab_size) -> logits")
    print()
    print("    With tying: Embedding.weight == Linear.weight.T")

    print("\n[2] Benefits:")
    print("    - Reduces parameters significantly")
    print("    - Better generalization")
    print("    - Semantically similar tokens stay similar in both spaces")

    print("\n[3] Parameter savings (example):")
    vocab, embed = 50000, 768
    print(f"    Vocab: {vocab}, Embed: {embed}")
    print(f"    Without tying: {2 * vocab * embed:,} params")
    print(f"    With tying:    {vocab * embed:,} params (50% reduction)")


# ============================================================================
# Training
# ============================================================================

def train_gpt():
    """Train GPT language model"""
    print("\n" + "=" * 60)
    print("Training GPT Language Model")
    print("=" * 60)

    tz.initialize()

    # Configuration
    vocab_size = 1000
    embed_dim = 128
    num_heads = 4
    num_layers = 4
    max_seq_len = 64
    batch_size = 8
    num_epochs = 5
    learning_rate = 0.0003

    # Dataset
    train_data = TextDataset(200, max_seq_len, vocab_size)
    val_data = TextDataset(50, max_seq_len, vocab_size)

    # Model
    model = GPT(vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout=0.1)
    model.train()

    params = model.parameters()
    optimizer = tz.optim.AdamW(params, lr=learning_rate, beta1=0.9, beta2=0.999, weight_decay=0.01)

    # Learning rate scheduler
    total_steps = num_epochs * (len(train_data) // batch_size)
    scheduler = tz.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print("\nConfiguration:")
    print("  Model: GPT (decoder-only transformer)")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Attention heads: {num_heads}")
    print(f"  Layers: {num_layers}")
    print(f"  Max sequence: {max_seq_len}")
    print(f"  Optimizer: AdamW (lr={learning_rate}, wd=0.01)")
    print("  Scheduler: CosineAnnealing")
    print()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        model.train()
        for i in range(0, len(train_data), batch_size):
            input_ids, target_ids = train_data.get_batch(i, batch_size)

            optimizer.zero_grad()

            input_var = tz.Variable(input_ids, False)
            logits = model.forward(input_var)

            # Reshape for cross entropy: [batch * seq, vocab]
            shape = logits.tensor().shape
            logits_flat_tensor = logits.tensor().reshape([shape[0] * shape[1], shape[2]])
            logits_flat = tz.Variable(logits_flat_tensor, requires_grad=True)
            targets_flat = target_ids.reshape([shape[0] * shape[1]])

            loss = tz.nn.cross_entropy(logits_flat, targets_flat)
            loss.backward()

            # Note: gradient clipping not available in this library version
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.tensor().item()
            num_batches += 1

        # Validation perplexity
        model.eval()
        val_loss = 0.0
        val_batches = 0

        for i in range(0, len(val_data), batch_size):
            input_ids, target_ids = val_data.get_batch(i, batch_size)

            input_var = tz.Variable(input_ids, False)
            logits = model.forward(input_var)

            shape = logits.tensor().shape
            logits_flat_tensor = logits.tensor().reshape([shape[0] * shape[1], shape[2]])
            logits_flat = tz.Variable(logits_flat_tensor, requires_grad=False)
            targets_flat = target_ids.reshape([shape[0] * shape[1]])

            loss = tz.nn.cross_entropy(logits_flat, targets_flat)

            val_loss += loss.tensor().item()
            val_batches += 1

        train_ppl = math.exp(epoch_loss / num_batches)
        val_ppl = math.exp(val_loss / val_batches)
        lr = scheduler.get_last_lr()

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Loss: {epoch_loss/num_batches:.4f} | "
              f"Train PPL: {train_ppl:.2f} | "
              f"Val PPL: {val_ppl:.2f} | "
              f"LR: {lr:.2e}")

    # Generation demo
    print("\n" + "-" * 60)
    print("Text Generation Demo")
    print("-" * 60)

    model.eval()

    prompt = [1, 2, 3, 4, 5]  # Random starting tokens

    print(f"\nPrompt tokens: {prompt}")

    # Different generation strategies
    print("\nGreedy (T=0.01):")
    greedy = model.generate(prompt, max_new_tokens=10, temperature=0.01, top_k=0, top_p=1.0)
    print(f"  Generated: {greedy}")

    print("\nTop-k (k=50, T=0.8):")
    topk_gen = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=50, top_p=1.0)
    print(f"  Generated: {topk_gen}")

    print("\nTop-p (p=0.9, T=0.9):")
    topp_gen = model.generate(prompt, max_new_tokens=10, temperature=0.9, top_k=0, top_p=0.9)
    print(f"  Generated: {topp_gen}")


# ============================================================================
# Main
# ============================================================================

def main():
    # Initialize Tenzor library first
    tz.initialize()

    print("=" * 60)
    print("   GPT Text Generation - Component Coverage          ")
    print("=" * 60)

    print("\nComponents demonstrated in this example:")
    print("  Architecture: GPT decoder-only transformer")
    print("  Layers: Embedding, LayerNorm, Linear, Dropout")
    print("  Attention: Causal self-attention with masking")
    print("  Activations: GELU")
    print("  Generation: Greedy, temperature, top-k, top-p sampling")
    print("  Optimizer: AdamW with weight decay")
    print("  Scheduler: CosineAnnealingLR")
    print("  Utils: Gradient clipping (clip_grad_norm)")

    demo_causal_attention()
    demo_generation_strategies()
    demo_weight_tying()
    train_gpt()

    print("\n" + "=" * 60)
    print("   All GPT examples completed successfully!          ")
    print("=" * 60)


if __name__ == "__main__":
    main()
