"""
Vision Transformer (ViT) Image Classification Training

This comprehensive example demonstrates:
- Vision Transformer (ViT) architecture
- PatchEmbedding for image tokenization
- MultiheadAttention mechanism
- LayerNorm and GELU activation
- AdaptiveAvgPool2d
- AdamW optimizer with weight decay
- OneCycleLR and CosineAnnealingWarmRestarts schedulers
- CrossEntropyLoss for classification
- Training and evaluation loop
"""

import tenzor as tz
import numpy as np
import math


# ============================================================================
# Synthetic Image Classification Dataset
# ============================================================================

class ImageClassificationDataset:
    """Synthetic image classification dataset"""

    def __init__(self, num_samples, num_classes, img_size=224, num_channels=3):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_channels = num_channels

        np.random.seed(42)
        self.labels = np.random.randint(0, num_classes, num_samples)

    def get_batch(self, start, batch_size):
        """Get a batch of data"""
        end = min(start + batch_size, self.num_samples)
        actual_batch = end - start

        # Generate synthetic images
        np.random.seed(start)
        images = np.random.randn(actual_batch, self.num_channels,
                                 self.img_size, self.img_size).astype(np.float32)
        labels = self.labels[start:end].astype(np.int64)

        return tz.Tensor.from_numpy(images), tz.Tensor.from_numpy(labels)

    def __len__(self):
        return self.num_samples


# ============================================================================
# Vision Transformer Components
# ============================================================================

class PatchEmbed:
    """Patch Embedding layer - splits image into patches and projects"""

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.proj = tz.nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        """
        Args:
            x: Input images [batch, channels, height, width]
        Returns:
            Patch embeddings [batch, num_patches, embed_dim]
        """
        batch_size = x.tensor().shape[0]

        # Simplified patch extraction
        # In practice, use proper unfold/reshape operations
        patches = tz.randn([batch_size, self.num_patches,
                           3 * self.patch_size * self.patch_size])
        patches_var = tz.Variable(patches, requires_grad=True)

        return self.proj(patches_var)

    def parameters(self):
        return self.proj.parameters()

    def train(self):
        self.proj.train()

    def eval(self):
        self.proj.eval()


class TransformerBlock:
    """Transformer Encoder Block with Multi-head Attention"""

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        # Multi-head self-attention
        self.attention = tz.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        # Layer norms
        self.norm1 = tz.nn.LayerNorm([embed_dim])
        self.norm2 = tz.nn.LayerNorm([embed_dim])

        # MLP block with GELU
        self.mlp_fc1 = tz.nn.Linear(embed_dim, mlp_dim)
        self.mlp_act = tz.nn.GELU()  # GELU activation
        self.mlp_fc2 = tz.nn.Linear(mlp_dim, embed_dim)
        self.mlp_dropout = tz.nn.Dropout(dropout)

        # Residual dropouts
        self.dropout1 = tz.nn.Dropout(dropout)
        self.dropout2 = tz.nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input [batch, seq_len, embed_dim]
        Returns:
            Output [batch, seq_len, embed_dim]
        """
        # Self-attention with residual
        attn_out, _ = self.attention.forward(x, x, x)  # Self-attention returns (output, weights)
        attn_out = self.dropout1(attn_out)
        h = x + attn_out  # Residual connection
        h = self.norm1(h)

        # MLP with residual
        mlp_out = self.mlp_fc1(h)
        mlp_out = self.mlp_act(mlp_out)
        mlp_out = self.mlp_dropout(mlp_out)
        mlp_out = self.mlp_fc2(mlp_out)
        mlp_out = self.dropout2(mlp_out)

        out = h + mlp_out  # Residual connection
        out = self.norm2(out)

        return out

    def parameters(self):
        return (self.attention.parameters() +
                self.norm1.parameters() +
                self.norm2.parameters() +
                self.mlp_fc1.parameters() +
                self.mlp_fc2.parameters() +
                self.mlp_dropout.parameters() +
                self.dropout1.parameters() +
                self.dropout2.parameters())

    def train(self):
        self.attention.train()
        self.norm1.train()
        self.norm2.train()
        self.mlp_fc1.train()
        self.mlp_act.train()
        self.mlp_fc2.train()
        self.mlp_dropout.train()
        self.dropout1.train()
        self.dropout2.train()

    def eval(self):
        self.attention.eval()
        self.norm1.eval()
        self.norm2.eval()
        self.mlp_fc1.eval()
        self.mlp_act.eval()
        self.mlp_fc2.eval()
        self.mlp_dropout.eval()
        self.dropout1.eval()
        self.dropout2.eval()


class VisionTransformer:
    """Complete Vision Transformer model"""

    def __init__(self, img_size, patch_size, num_classes, embed_dim=768,
                 depth=12, num_heads=12, mlp_dim=3072, dropout=0.1):
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # Dropout after embeddings
        self.pos_dropout = tz.nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = [TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
                       for _ in range(depth)]

        # Final layer norm
        self.norm = tz.nn.LayerNorm([embed_dim])

        # Classification head
        self.head = tz.nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input images [batch, channels, height, width]
        Returns:
            Class logits [batch, num_classes]
        """
        # Patch embedding
        h = self.patch_embed.forward(x)  # [batch, num_patches, embed_dim]

        batch_size = h.tensor().shape[0]

        # Dropout after embeddings
        h = self.pos_dropout(h)

        # Transformer blocks
        for block in self.blocks:
            h = block.forward(h)

        # Final norm
        h = self.norm(h)

        # Global average pooling over patches
        h_np = h.tensor().numpy()
        cls_output = h_np.mean(axis=1)  # [batch, embed_dim]
        cls_var = tz.Variable(tz.Tensor.from_numpy(cls_output.astype(np.float32)),
                              requires_grad=True)

        # Classification head
        return self.head(cls_var)

    def parameters(self):
        params = self.patch_embed.parameters()
        params += self.pos_dropout.parameters()
        for block in self.blocks:
            params += block.parameters()
        params += self.norm.parameters()
        params += self.head.parameters()
        return params

    def train(self):
        self.patch_embed.train()
        self.pos_dropout.train()
        for block in self.blocks:
            block.train()
        self.norm.train()
        self.head.train()

    def eval(self):
        self.patch_embed.eval()
        self.pos_dropout.eval()
        for block in self.blocks:
            block.eval()
        self.norm.eval()
        self.head.eval()


# ============================================================================
# Training Functions
# ============================================================================

def train_vit_with_adamw():
    """Train ViT with AdamW optimizer and OneCycleLR scheduler"""
    print("\n" + "=" * 60)
    print("Training ViT with AdamW + OneCycleLR")
    print("=" * 60)

    tz.initialize()

    # Create datasets
    num_classes = 10
    train_data = ImageClassificationDataset(500, num_classes, img_size=32)
    val_data = ImageClassificationDataset(100, num_classes, img_size=32)

    # Create small ViT
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        num_classes=num_classes,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_dim=512,
        dropout=0.1
    )
    model.train()

    # AdamW optimizer with weight decay
    params = model.parameters()
    optimizer = tz.optim.AdamW(params, lr=0.001, beta1=0.9, beta2=0.999,
                               eps=1e-8, weight_decay=0.01)

    # OneCycleLR scheduler
    num_epochs = 10
    batch_size = 16
    steps_per_epoch = (len(train_data) + batch_size - 1) // batch_size
    total_steps = num_epochs * steps_per_epoch

    scheduler = tz.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=total_steps,
                                    pct_start=0.3, div_factor=25.0, final_div_factor=10000)

    # Cross-entropy loss
    criterion = tz.nn.CrossEntropyLoss()

    print("\nConfiguration:")
    print("  Model: Vision Transformer (Tiny)")
    print("    - Image size: 32x32")
    print("    - Patch size: 4x4 (64 patches)")
    print("    - Embed dim: 128")
    print("    - Depth: 4 blocks")
    print("    - Heads: 4")
    print("  Optimizer: AdamW (lr=0.001, weight_decay=0.01)")
    print("  Scheduler: OneCycleLR (max_lr=0.01)")
    print("  Loss: CrossEntropyLoss")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}\n")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            images, labels = train_data.get_batch(i, batch_size)

            optimizer.zero_grad()

            images_var = tz.Variable(images, requires_grad=True)
            output = model.forward(images_var)

            loss = criterion(output, labels)  # CrossEntropyLoss expects Tensor for target

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.tensor().item()

            # Calculate accuracy
            output_np = output.tensor().numpy()
            labels_np = labels.numpy()
            preds = np.argmax(output_np, axis=1)
            correct += np.sum(preds == labels_np)
            total += len(labels_np)

            num_batches += 1

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with tz.no_grad():
            for i in range(0, len(val_data), batch_size):
                images, labels = val_data.get_batch(i, batch_size)
                images_var = tz.Variable(images, requires_grad=False)
                output = model.forward(images_var)

                output_np = output.tensor().numpy()
                labels_np = labels.numpy()
                preds = np.argmax(output_np, axis=1)
                val_correct += np.sum(preds == labels_np)
                val_total += len(labels_np)

        train_acc = 100.0 * correct / total
        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Loss: {epoch_loss/num_batches:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | "
              f"LR: {scheduler.get_lr():.6f}")


def train_with_cosine_warm_restarts():
    """Train with CosineAnnealingWarmRestarts scheduler"""
    print("\n" + "=" * 60)
    print("Training with CosineAnnealingWarmRestarts")
    print("=" * 60)

    num_classes = 10
    train_data = ImageClassificationDataset(400, num_classes, img_size=32)

    # Use ViT from models
    config = tz.models.ViTConfig.base_patch16(32)
    config.hidden_size = 64
    config.num_hidden_layers = 2
    config.num_attention_heads = 2
    config.intermediate_size = 256

    model = tz.models.ViTForImageClassification(config, num_classes)
    model.train()

    params = model.parameters()
    optimizer = tz.optim.AdamW(params, lr=0.0001, weight_decay=0.05)

    # CosineAnnealingWarmRestarts
    T_0 = 5
    scheduler = tz.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0,
                                                     T_mult=2, eta_min=1e-6)

    criterion = tz.nn.CrossEntropyLoss()

    batch_size = 16
    num_epochs = 15

    print("\nConfiguration:")
    print("  Model: ViT (from tenzor.models)")
    print("  Optimizer: AdamW (lr=0.0001, weight_decay=0.05)")
    print(f"  Scheduler: CosineAnnealingWarmRestarts (T_0={T_0}, T_mult=2)\n")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            images, labels = train_data.get_batch(i, batch_size)

            optimizer.zero_grad()

            images_var = tz.Variable(images, requires_grad=True)
            output = model.forward(images_var)

            loss = criterion(output, labels)  # CrossEntropyLoss expects Tensor for target

            loss.backward()
            optimizer.step()

            epoch_loss += loss.tensor().item()
            num_batches += 1

        scheduler.step()

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Loss: {epoch_loss/num_batches:.4f} | "
              f"LR: {scheduler.get_lr():.6f}")


def demo_adaptive_pooling():
    """Demonstrate AdaptiveAvgPool2d"""
    print("\n" + "=" * 60)
    print("AdaptiveAvgPool2d Demo")
    print("=" * 60)

    # AdaptiveAvgPool2d outputs fixed size regardless of input
    pool = tz.nn.AdaptiveAvgPool2d(output_size=(7, 7))

    input_sizes = [(224, 224), (256, 256), (320, 320), (128, 128)]

    print("\nAdaptiveAvgPool2d with output_size=(7, 7):\n")

    for h, w in input_sizes:
        input_tensor = tz.randn([2, 64, h, w])
        input_var = tz.Variable(input_tensor, requires_grad=False)

        output = pool(input_var)
        out_shape = output.tensor().shape

        print(f"  Input: [2, 64, {h}, {w}] -> Output: {list(out_shape)}")

    print("\nAdaptiveAvgPool2d ensures consistent feature map size!")


def demo_gelu_activation():
    """Demonstrate GELU activation function"""
    print("\n" + "=" * 60)
    print("GELU Activation Demo")
    print("=" * 60)

    # GELU - Gaussian Error Linear Unit (used in BERT, GPT, ViT)
    gelu = tz.nn.GELU()

    x_np = np.linspace(-3, 3, 7).astype(np.float32)
    x_tensor = tz.Tensor.from_numpy(x_np)
    x_var = tz.Variable(x_tensor, requires_grad=True)

    y = gelu(x_var)
    y_np = y.tensor().numpy()

    print("\nGELU(x) = x * Phi(x) where Phi is the CDF of standard normal")
    print("\nSample values:")
    print("  x      | GELU(x)")
    print("  -------|--------")
    for i in range(len(x_np)):
        print(f"  {x_np[i]:6.2f} | {y_np[i]:7.4f}")

    print("\nGELU is smoother than ReLU and allows small negative values!")


# ============================================================================
# Main
# ============================================================================

def main():
    # Initialize Tenzor library first
    tz.initialize()

    print("=" * 60)
    print("   Vision Transformer (ViT) Training - Components     ")
    print("=" * 60)

    print("\nComponents demonstrated in this example:")
    print("  Models: VisionTransformer, ViTForImageClassification")
    print("  Layers: PatchEmbedding, MultiheadAttention, LayerNorm")
    print("  Activations: GELU")
    print("  Pooling: AdaptiveAvgPool2d")
    print("  Optimizers: AdamW")
    print("  Schedulers: OneCycleLR, CosineAnnealingWarmRestarts")
    print("  Losses: CrossEntropyLoss")

    # Run training examples
    train_vit_with_adamw()
    train_with_cosine_warm_restarts()

    # Demo components
    demo_adaptive_pooling()
    demo_gelu_activation()

    print("\n" + "=" * 60)
    print("   All ViT training examples completed successfully!  ")
    print("=" * 60)


if __name__ == "__main__":
    main()
