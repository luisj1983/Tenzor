"""
UNet Semantic Segmentation Training Example

This comprehensive example demonstrates:
- UNet encoder-decoder architecture with skip connections
- ConvTranspose2d for upsampling
- MaxPool2d for downsampling
- BatchNorm2d normalization
- Pixel-wise classification
- Dice loss for segmentation
- Focal loss for class imbalance
- IoU/Jaccard metric calculation
- Multi-class semantic segmentation
"""

import tenzor as tz
import numpy as np


# ============================================================================
# Helper Functions
# ============================================================================

def one_hot(tensor, num_classes):
    """
    Convert class indices to one-hot encoding
    Args:
        tensor: Tensor of class indices
        num_classes: Number of classes
    Returns:
        One-hot encoded tensor
    """
    indices = tensor.numpy().astype(np.int64)
    shape = indices.shape
    one_hot_np = np.zeros(shape + (num_classes,), dtype=np.float32)
    flat_indices = indices.flatten()
    flat_one_hot = one_hot_np.reshape(-1, num_classes)
    flat_one_hot[np.arange(len(flat_indices)), flat_indices] = 1.0
    return tz.Tensor.from_numpy(one_hot_np)


# ============================================================================
# Segmentation Dataset
# ============================================================================

class SegmentationDataset:
    """Synthetic segmentation dataset with images and masks"""

    def __init__(self, num_samples, num_classes, img_size=128):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size

        np.random.seed(42)

        # Generate synthetic images
        self.images = np.random.randn(num_samples, 3, img_size, img_size).astype(np.float32)

        # Generate segmentation masks with region-based patterns
        self.masks = np.zeros((num_samples, img_size, img_size), dtype=np.int64)

        for n in range(num_samples):
            for h in range(img_size):
                for w in range(img_size):
                    # Create region-based masks
                    region = (h // (img_size // 4)) * 4 + (w // (img_size // 4))
                    self.masks[n, h, w] = region % num_classes

    def get_batch(self, start, batch_size):
        """Get a batch of images and masks"""
        end = min(start + batch_size, self.num_samples)
        actual_batch = end - start

        images = tz.Tensor.from_numpy(self.images[start:end])
        masks = tz.Tensor.from_numpy(self.masks[start:end])

        return images, masks

    def __len__(self):
        return self.num_samples


# ============================================================================
# UNet Building Blocks
# ============================================================================

class DoubleConv(tz.nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = tz.nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = tz.nn.BatchNorm2d(mid_channels)
        self.conv2 = tz.nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        self.bn2 = tz.nn.BatchNorm2d(out_channels)
        self.relu = tz.nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class EncoderBlock(tz.nn.Module):
    """Encoder block: DoubleConv -> MaxPool"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = tz.nn.MaxPool2d(2, 2)

    def forward_with_skip(self, x):
        """Forward pass returning both pooled output and skip connection"""
        skip = self.conv(x)
        pooled = self.pool(skip)
        return pooled, skip

    def forward(self, x):
        return self.forward_with_skip(x)[0]


class DecoderBlock(tz.nn.Module):
    """Decoder block: Upsample -> Concat skip -> DoubleConv"""

    def __init__(self, in_channels, out_channels, use_transpose=True):
        super().__init__()
        self.use_transpose = use_transpose

        if use_transpose:
            self.up = tz.nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward_with_skip(self, x, skip):
        """Forward pass with skip connection concatenation"""
        if self.use_transpose:
            up = self.up(x)
        else:
            # Bilinear upsampling
            shape = x.tensor().shape
            up = tz.Variable(
                tz.interpolate(x.tensor(), size=(shape[2] * 2, shape[3] * 2),
                               mode='bilinear', align_corners=False),
                x.requires_grad
            )

        # Concatenate along channel dimension
        # tz.cat expects Tensors, so extract from Variables and wrap back
        up_tensor = up.tensor() if hasattr(up, 'tensor') else up
        skip_tensor = skip.tensor() if hasattr(skip, 'tensor') else skip
        concat_tensor = tz.cat([up_tensor, skip_tensor], dim=1)
        concat = tz.Variable(concat_tensor, requires_grad=True)
        return self.conv(concat)

    def forward(self, x):
        # Not used directly - use forward_with_skip
        return x


# ============================================================================
# UNet Model
# ============================================================================

class UNet(tz.nn.Module):
    """UNet architecture for semantic segmentation"""

    def __init__(self, in_channels, num_classes, base_features=64):
        super().__init__()

        # Encoder path
        self.enc1 = EncoderBlock(in_channels, base_features)
        self.enc2 = EncoderBlock(base_features, base_features * 2)
        self.enc3 = EncoderBlock(base_features * 2, base_features * 4)
        self.enc4 = EncoderBlock(base_features * 4, base_features * 8)

        # Bottleneck
        self.bottleneck = DoubleConv(base_features * 8, base_features * 16)

        # Decoder path
        self.dec4 = DecoderBlock(base_features * 16, base_features * 8)
        self.dec3 = DecoderBlock(base_features * 8, base_features * 4)
        self.dec2 = DecoderBlock(base_features * 4, base_features * 2)
        self.dec1 = DecoderBlock(base_features * 2, base_features)

        # Output
        self.out_conv = tz.nn.Conv2d(base_features, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1, s1 = self.enc1.forward_with_skip(x)
        e2, s2 = self.enc2.forward_with_skip(e1)
        e3, s3 = self.enc3.forward_with_skip(e2)
        e4, s4 = self.enc4.forward_with_skip(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.dec4.forward_with_skip(b, s4)
        d3 = self.dec3.forward_with_skip(d4, s3)
        d2 = self.dec2.forward_with_skip(d3, s2)
        d1 = self.dec1.forward_with_skip(d2, s1)

        # Output
        return self.out_conv(d1)


# ============================================================================
# Segmentation Loss Functions
# ============================================================================

class DiceLoss:
    """
    Dice Loss for segmentation
    Measures overlap between prediction and ground truth
    """

    def __init__(self, smooth=1.0):
        self.smooth = smooth

    def compute(self, pred, target, num_classes):
        """
        Args:
            pred: [N, C, H, W] logits (Variable)
            target: Tensor [N, H, W] class indices
            num_classes: number of classes
        """
        # Convert to numpy for calculation (simplified approach)
        pred_np = pred.tensor().numpy()
        target_np = target.numpy()

        # Apply softmax to get probabilities
        pred_exp = np.exp(pred_np - pred_np.max(axis=1, keepdims=True))
        probs = pred_exp / pred_exp.sum(axis=1, keepdims=True)

        # Convert target to one-hot [N, C, H, W]
        n, h, w = target_np.shape
        target_one_hot = np.zeros((n, num_classes, h, w), dtype=np.float32)
        for i in range(n):
            for hi in range(h):
                for wi in range(w):
                    c = int(target_np[i, hi, wi])
                    if 0 <= c < num_classes:
                        target_one_hot[i, c, hi, wi] = 1.0

        # Compute intersection and union (sum over H, W)
        intersection = (probs * target_one_hot).sum(axis=(2, 3))
        pred_sum = probs.sum(axis=(2, 3))
        target_sum = target_one_hot.sum(axis=(2, 3))

        # Dice coefficient per class
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        # Return 1 - mean dice (loss)
        loss_val = 1.0 - dice.mean()
        return tz.Variable(tz.Tensor.from_numpy(np.array([loss_val], dtype=np.float32)), requires_grad=True)


class FocalLoss:
    """
    Focal Loss for class imbalance
    Focuses on hard examples by down-weighting easy ones
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def compute(self, pred, target):
        """
        Args:
            pred: [N, C, H, W] logits (Variable)
            target: [N, H, W] class indices (Tensor)
        """
        # Convert to numpy for computation
        pred_np = pred.tensor().numpy()
        target_np = target.numpy()

        n, c, h, w = pred_np.shape

        # Transpose to [N, H, W, C] then flatten to [N*H*W, C]
        pred_flat = pred_np.transpose(0, 2, 3, 1).reshape(-1, c)
        target_flat = target_np.flatten()

        # Apply softmax to get probabilities
        pred_max = pred_flat.max(axis=1, keepdims=True)
        pred_exp = np.exp(pred_flat - pred_max)
        probs = pred_exp / pred_exp.sum(axis=1, keepdims=True)

        # Compute focal loss
        total_loss = 0.0
        count = 0
        for i in range(len(target_flat)):
            t = int(target_flat[i])
            if 0 <= t < c:
                pt = probs[i, t]
                ce = -np.log(pt + 1e-8)
                focal_weight = (1 - pt) ** self.gamma
                total_loss += self.alpha * focal_weight * ce
                count += 1

        loss_val = total_loss / max(count, 1)
        return tz.Variable(tz.Tensor.from_numpy(np.array([loss_val], dtype=np.float32)), requires_grad=True)


class CombinedSegmentationLoss:
    """Combined loss: CrossEntropy + Dice"""

    def __init__(self, num_classes, dice_weight=0.5):
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=1.0)

    def compute(self, pred, target):
        """
        Args:
            pred: [N, C, H, W] logits (Variable)
            target: [N, H, W] class indices (Tensor)
        """
        # Compute cross entropy using numpy
        pred_np = pred.tensor().numpy()
        target_np = target.numpy()

        n, c, h, w = pred_np.shape

        # Transpose to [N, H, W, C] then flatten to [N*H*W, C]
        pred_flat = pred_np.transpose(0, 2, 3, 1).reshape(-1, c)
        target_flat = target_np.flatten()

        # Apply log_softmax
        pred_max = pred_flat.max(axis=1, keepdims=True)
        log_probs = pred_flat - pred_max - np.log(np.exp(pred_flat - pred_max).sum(axis=1, keepdims=True))

        # Compute NLL loss
        ce_val = 0.0
        count = 0
        for i in range(len(target_flat)):
            t = int(target_flat[i])
            if 0 <= t < c:
                ce_val -= log_probs[i, t]
                count += 1
        ce_val = ce_val / max(count, 1)

        # Dice loss
        d_loss = self.dice_loss.compute(pred, target, self.num_classes)
        d_loss_val = d_loss.tensor().item()

        # Combined
        combined_loss = (1.0 - self.dice_weight) * ce_val + self.dice_weight * d_loss_val
        return tz.Variable(tz.Tensor.from_numpy(np.array([combined_loss], dtype=np.float32)), requires_grad=True)


# ============================================================================
# Metrics
# ============================================================================

def calculate_miou(pred, target, num_classes):
    """
    Calculate mean Intersection over Union (mIoU)

    Args:
        pred: [N, C, H, W] logits tensor
        target: [N, H, W] class indices tensor
        num_classes: number of classes
    """
    # Get predicted classes using module-level argmax
    pred_classes = tz.argmax(pred, dim=1)  # [N, H, W]

    pred_np = pred_classes.numpy().flatten()
    target_np = target.numpy().flatten()

    ious = []
    for c in range(num_classes):
        pred_c = (pred_np == c)
        target_c = (target_np == c)

        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()

        if union > 0:
            ious.append(intersection / union)

    return np.mean(ious) if ious else 0.0


def calculate_pixel_accuracy(pred, target):
    """
    Calculate pixel-wise accuracy

    Args:
        pred: [N, C, H, W] logits tensor
        target: [N, H, W] class indices tensor
    """
    pred_classes = tz.argmax(pred, dim=1)

    pred_np = pred_classes.numpy().flatten()
    target_np = target.numpy().flatten()

    return (pred_np == target_np).sum() / len(pred_np)


# ============================================================================
# Demo Functions
# ============================================================================

def demo_segmentation_losses():
    """Demonstrate segmentation loss functions"""
    print("\n" + "=" * 60)
    print("Segmentation Loss Functions Demo")
    print("=" * 60)

    batch_size = 2
    num_classes = 5
    height, width = 32, 32

    # Random predictions and targets
    pred_data = np.random.randn(batch_size, num_classes, height, width).astype(np.float32)
    target_data = np.random.randint(0, num_classes, (batch_size, height, width)).astype(np.int64)

    pred = tz.Variable(tz.Tensor.from_numpy(pred_data), requires_grad=True)
    target = tz.Tensor.from_numpy(target_data)

    # Dice Loss
    print("\n[1] Dice Loss")
    dice_loss = DiceLoss()
    dice = dice_loss.compute(pred, target, num_classes)
    print(f"    Dice Loss: {dice.tensor().item():.4f}")
    print("    Measures overlap between prediction and ground truth")

    # Focal Loss
    print("\n[2] Focal Loss (gamma=2.0, alpha=0.25)")
    focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
    focal = focal_loss.compute(pred, target)
    print(f"    Focal Loss: {focal.tensor().item():.4f}")
    print("    Down-weights easy examples, focuses on hard ones")

    # Combined Loss
    print("\n[3] Combined Loss (CE + Dice)")
    combined_loss = CombinedSegmentationLoss(num_classes, dice_weight=0.5)
    combined = combined_loss.compute(pred, target)
    print(f"    Combined Loss: {combined.tensor().item():.4f}")
    print("    Balances pixel-wise CE with region-based Dice")


def demo_metrics():
    """Demonstrate segmentation metrics"""
    print("\n" + "=" * 60)
    print("Segmentation Metrics Demo")
    print("=" * 60)

    batch_size = 2
    num_classes = 3
    height, width = 8, 8

    # Create predictions with clear class regions
    pred_data = np.full((batch_size, num_classes, height, width), -10.0, dtype=np.float32)
    target_data = np.zeros((batch_size, height, width), dtype=np.int64)

    for n in range(batch_size):
        for h in range(height):
            for w in range(width):
                if h < height // 2:
                    pred_class = 0
                elif w < width // 2:
                    pred_class = 1
                else:
                    pred_class = 2

                target_class = pred_class

                # Add some errors
                if h == height // 2 and w == width // 2:
                    pred_class = (pred_class + 1) % num_classes

                # Set high logit for predicted class
                pred_data[n, pred_class, h, w] = 10.0
                target_data[n, h, w] = target_class

    pred = tz.Tensor.from_numpy(pred_data)
    target = tz.Tensor.from_numpy(target_data)

    print("\n[1] Pixel Accuracy")
    pixel_acc = calculate_pixel_accuracy(pred, target)
    print(f"    Accuracy: {pixel_acc:.4f}")

    print("\n[2] Mean IoU (Jaccard Index)")
    miou = calculate_miou(pred, target, num_classes)
    print(f"    mIoU: {miou:.4f}")

    print("\n[3] Interpretation")
    print("    Pixel Accuracy: % of correctly classified pixels")
    print("    mIoU: Average overlap between pred and target per class")
    print("    mIoU is preferred as it penalizes class imbalance")


# ============================================================================
# Training
# ============================================================================

def train_unet():
    """Train UNet semantic segmentation model"""
    print("\n" + "=" * 60)
    print("Training UNet Semantic Segmentation Model")
    print("=" * 60)

    tz.initialize()

    num_classes = 8
    img_size = 128
    batch_size = 4
    num_epochs = 5

    # Dataset
    train_data = SegmentationDataset(100, num_classes, img_size)
    val_data = SegmentationDataset(20, num_classes, img_size)

    # Model (smaller base features for faster training)
    model = UNet(3, num_classes, base_features=32)
    model.train()

    params = model.parameters()
    optimizer = tz.optim.Adam(params, lr=0.001, beta1=0.9, beta2=0.999)

    # Loss
    criterion = CombinedSegmentationLoss(num_classes, dice_weight=0.5)

    print("\nConfiguration:")
    print("  Model: UNet (encoder-decoder with skip connections)")
    print(f"  Input: {img_size}x{img_size} RGB images")
    print(f"  Classes: {num_classes}")
    print("  Loss: CrossEntropy + Dice (0.5 weight)")
    print("  Optimizer: Adam (lr=0.001)")
    print()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Training
        model.train()
        for i in range(0, len(train_data), batch_size):
            images, masks = train_data.get_batch(i, batch_size)

            optimizer.zero_grad()

            images_var = tz.Variable(images, requires_grad=True)
            output = model.forward(images_var)

            loss = criterion.compute(output, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.tensor().item()
            num_batches += 1

        # Validation
        model.eval()
        val_miou = 0.0
        val_acc = 0.0
        val_batches = 0

        for i in range(0, len(val_data), batch_size):
            images, masks = val_data.get_batch(i, batch_size)

            images_var = tz.Variable(images, requires_grad=False)
            output = model.forward(images_var)

            val_miou += calculate_miou(output.tensor(), masks, num_classes)
            val_acc += calculate_pixel_accuracy(output.tensor(), masks)
            val_batches += 1

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Loss: {epoch_loss/num_batches:.4f} | "
              f"Val mIoU: {val_miou/val_batches:.4f} | "
              f"Val Acc: {val_acc/val_batches:.4f}")


# ============================================================================
# Main
# ============================================================================

def main():
    # Initialize Tenzor library first
    tz.initialize()

    print("=" * 60)
    print("   UNet Semantic Segmentation - Component Coverage     ")
    print("=" * 60)

    print("\nComponents demonstrated in this example:")
    print("  Architecture: UNet encoder-decoder with skip connections")
    print("  Layers: Conv2d, ConvTranspose2d, BatchNorm2d, MaxPool2d")
    print("  Losses: Dice Loss, Focal Loss, Combined CE+Dice")
    print("  Metrics: mIoU (Jaccard), Pixel Accuracy")
    print("  Operations: one_hot, argmax, interpolate, permute")
    print("  Optimizer: Adam")

    demo_segmentation_losses()
    demo_metrics()
    train_unet()

    print("\n" + "=" * 60)
    print("   All segmentation examples completed successfully!   ")
    print("=" * 60)


if __name__ == "__main__":
    main()
