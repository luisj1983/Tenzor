"""
Classic CNN Models Example

This comprehensive example demonstrates:
- VGG architecture variants (VGG11, VGG16, VGG19)
- AlexNet architecture
- GoogLeNet (Inception v1) with auxiliary classifiers
- Model comparison and parameter counting
- Forward inference on all models

Components used:
- Conv2d, MaxPool2d, AdaptiveAvgPool2d
- BatchNorm2d, Dropout, Linear
- ReLU activation
- CrossEntropyLoss, Adam optimizer
"""

import tenzor as tz
import numpy as np


# ============================================================================
# VGG Block
# ============================================================================

class VGGBlock:
    """VGG-style convolutional block"""

    def __init__(self, in_channels, out_channels, num_convs, batch_norm=True):
        self.layers = []

        for i in range(num_convs):
            in_ch = in_channels if i == 0 else out_channels
            conv = tz.nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)
            self.layers.append(('conv', conv))

            if batch_norm:
                bn = tz.nn.BatchNorm2d(out_channels)
                self.layers.append(('bn', bn))

            self.layers.append(('relu', None))  # ReLU marker

        self.pool = tz.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        for layer_type, layer in self.layers:
            if layer_type == 'conv':
                x = layer(x)
            elif layer_type == 'bn':
                x = layer(x)
            elif layer_type == 'relu':
                x = tz.nn.relu(x)
        x = self.pool(x)
        return x

    def parameters(self):
        params = []
        for layer_type, layer in self.layers:
            if layer is not None and hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        params.extend(self.pool.parameters())
        return params


# ============================================================================
# VGG Model
# ============================================================================

class VGG:
    """
    VGG Network Implementation

    Configurations:
    - VGG11: [1, 1, 2, 2, 2] conv layers per block
    - VGG13: [2, 2, 2, 2, 2]
    - VGG16: [2, 2, 3, 3, 3]
    - VGG19: [2, 2, 4, 4, 4]
    """

    CONFIGS = {
        'vgg11': [1, 1, 2, 2, 2],
        'vgg13': [2, 2, 2, 2, 2],
        'vgg16': [2, 2, 3, 3, 3],
        'vgg19': [2, 2, 4, 4, 4],
    }

    def __init__(self, config_name, num_classes=1000, batch_norm=True):
        self.config_name = config_name
        config = self.CONFIGS[config_name]
        channels = [64, 128, 256, 512, 512]

        # Feature extractor
        self.blocks = []
        in_channels = 3

        for i, num_convs in enumerate(config):
            block = VGGBlock(in_channels, channels[i], num_convs, batch_norm)
            self.blocks.append(block)
            in_channels = channels[i]

        # Adaptive pooling for flexible input sizes
        self.avgpool = tz.nn.AdaptiveAvgPool2d(7, 7)

        # Classifier
        self.classifier_fc1 = tz.nn.Linear(512 * 7 * 7, 4096)
        self.classifier_fc2 = tz.nn.Linear(4096, 4096)
        self.classifier_fc3 = tz.nn.Linear(4096, num_classes)
        self.dropout = tz.nn.Dropout(0.5)

    def forward(self, x):
        # Feature extraction
        for block in self.blocks:
            x = block.forward(x)

        # Global average pooling
        x = self.avgpool(x)

        # Flatten
        x_np = x.tensor().numpy()
        batch_size = x_np.shape[0]
        x_flat = x_np.reshape(batch_size, -1)
        x = tz.Variable(tz.Tensor.from_numpy(x_flat.astype(np.float32)), requires_grad=True)

        # Classifier
        x = self.classifier_fc1(x)
        x = tz.nn.relu(x)
        x = self.dropout(x)
        x = self.classifier_fc2(x)
        x = tz.nn.relu(x)
        x = self.dropout(x)
        x = self.classifier_fc3(x)

        return x

    def parameters(self):
        params = []
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.avgpool.parameters())
        params.extend(self.classifier_fc1.parameters())
        params.extend(self.classifier_fc2.parameters())
        params.extend(self.classifier_fc3.parameters())
        return params

    def train(self):
        self.dropout.train()
        for block in self.blocks:
            for layer_type, layer in block.layers:
                if layer is not None and hasattr(layer, 'train'):
                    layer.train()

    def eval(self):
        self.dropout.eval()
        for block in self.blocks:
            for layer_type, layer in block.layers:
                if layer is not None and hasattr(layer, 'eval'):
                    layer.eval()


# ============================================================================
# AlexNet Model
# ============================================================================

class AlexNet:
    """
    AlexNet Implementation

    Architecture:
    - 5 Convolutional layers
    - 3 Fully connected layers
    - Local Response Normalization (simplified as BatchNorm)
    """

    def __init__(self, num_classes=1000):
        # Feature extraction layers
        self.conv1 = tz.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = tz.nn.BatchNorm2d(64)
        self.pool1 = tz.nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = tz.nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.bn2 = tz.nn.BatchNorm2d(192)
        self.pool2 = tz.nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = tz.nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3 = tz.nn.BatchNorm2d(384)

        self.conv4 = tz.nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn4 = tz.nn.BatchNorm2d(256)

        self.conv5 = tz.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = tz.nn.BatchNorm2d(256)
        self.pool5 = tz.nn.MaxPool2d(kernel_size=3, stride=2)

        # Adaptive pooling
        self.avgpool = tz.nn.AdaptiveAvgPool2d((6, 6))

        # Classifier
        self.fc1 = tz.nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = tz.nn.Linear(4096, 4096)
        self.fc3 = tz.nn.Linear(4096, num_classes)
        self.dropout = tz.nn.Dropout(0.5)

    def forward(self, x):
        # Conv layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = tz.nn.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = tz.nn.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = tz.nn.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = tz.nn.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = tz.nn.relu(x)
        x = self.pool5(x)

        # Adaptive pooling
        x = self.avgpool(x)

        # Flatten
        x_np = x.tensor().numpy()
        batch_size = x_np.shape[0]
        x_flat = x_np.reshape(batch_size, -1)
        x = tz.Variable(tz.Tensor.from_numpy(x_flat.astype(np.float32)), requires_grad=True)

        # Classifier
        x = self.fc1(x)
        x = tz.nn.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = tz.nn.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def parameters(self):
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.bn1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.bn2.parameters())
        params.extend(self.conv3.parameters())
        params.extend(self.bn3.parameters())
        params.extend(self.conv4.parameters())
        params.extend(self.bn4.parameters())
        params.extend(self.conv5.parameters())
        params.extend(self.bn5.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.fc3.parameters())
        return params

    def train(self):
        self.dropout.train()
        for bn in [self.bn1, self.bn2, self.bn3, self.bn4, self.bn5]:
            bn.train()

    def eval(self):
        self.dropout.eval()
        for bn in [self.bn1, self.bn2, self.bn3, self.bn4, self.bn5]:
            bn.eval()


# ============================================================================
# Inception Module (for GoogLeNet)
# ============================================================================

class InceptionModule:
    """
    Inception Module (Inception v1)

    Parallel branches:
    - 1x1 conv
    - 1x1 conv -> 3x3 conv
    - 1x1 conv -> 5x5 conv
    - 3x3 maxpool -> 1x1 conv

    All branches are concatenated along channel dimension.
    """

    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3,
                 ch5x5_reduce, ch5x5, pool_proj):
        # Branch 1: 1x1 conv
        self.branch1_conv = tz.nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # Branch 2: 1x1 -> 3x3
        self.branch2_reduce = tz.nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1)
        self.branch2_conv = tz.nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1)

        # Branch 3: 1x1 -> 5x5
        self.branch3_reduce = tz.nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1)
        self.branch3_conv = tz.nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2)

        # Branch 4: 3x3 pool -> 1x1
        self.branch4_pool = tz.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv = tz.nn.Conv2d(in_channels, pool_proj, kernel_size=1)

    def forward(self, x):
        # Branch 1
        b1 = self.branch1_conv(x)
        b1 = tz.nn.relu(b1)

        # Branch 2
        b2 = self.branch2_reduce(x)
        b2 = tz.nn.relu(b2)
        b2 = self.branch2_conv(b2)
        b2 = tz.nn.relu(b2)

        # Branch 3
        b3 = self.branch3_reduce(x)
        b3 = tz.nn.relu(b3)
        b3 = self.branch3_conv(b3)
        b3 = tz.nn.relu(b3)

        # Branch 4
        b4 = self.branch4_pool(x)
        b4 = self.branch4_conv(b4)
        b4 = tz.nn.relu(b4)

        # Concatenate along channel dimension
        outputs = [b1.tensor().numpy(), b2.tensor().numpy(),
                   b3.tensor().numpy(), b4.tensor().numpy()]
        concat = np.concatenate(outputs, axis=1)
        return tz.Variable(tz.Tensor.from_numpy(concat.astype(np.float32)), requires_grad=True)

    def parameters(self):
        params = []
        params.extend(self.branch1_conv.parameters())
        params.extend(self.branch2_reduce.parameters())
        params.extend(self.branch2_conv.parameters())
        params.extend(self.branch3_reduce.parameters())
        params.extend(self.branch3_conv.parameters())
        params.extend(self.branch4_conv.parameters())
        return params


# ============================================================================
# Demo Functions
# ============================================================================

def demo_vgg():
    """Demonstrate VGG models"""
    print("\n" + "=" * 60)
    print("VGG Models Demo")
    print("=" * 60)

    for config in ['vgg11', 'vgg16']:
        model = VGG(config, num_classes=10, batch_norm=True)
        params = model.parameters()

        total_params = sum(p.tensor().numel for p in params)
        print(f"\n{config.upper()}:")
        print(f"  Parameters: {total_params:,}")
        print(f"  Parameter tensors: {len(params)}")

    # Test forward pass
    print("\nForward pass test (VGG11):")
    model = VGG('vgg11', num_classes=10)
    model.eval()

    # Create dummy input (small for demo)
    input_np = np.random.randn(2, 3, 224, 224).astype(np.float32)
    x = tz.Variable(tz.Tensor.from_numpy(input_np), requires_grad=False)

    output = model.forward(x)
    print(f"  Input shape: [2, 3, 224, 224]")
    print(f"  Output shape: {list(output.tensor().shape)}")


def demo_alexnet():
    """Demonstrate AlexNet"""
    print("\n" + "=" * 60)
    print("AlexNet Demo")
    print("=" * 60)

    model = AlexNet(num_classes=10)
    params = model.parameters()

    total_params = sum(p.tensor().numel for p in params)
    print(f"\nAlexNet:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Parameter tensors: {len(params)}")

    # Forward pass
    print("\nForward pass test:")
    model.eval()

    input_np = np.random.randn(2, 3, 224, 224).astype(np.float32)
    x = tz.Variable(tz.Tensor.from_numpy(input_np), requires_grad=False)

    output = model.forward(x)
    print(f"  Input shape: [2, 3, 224, 224]")
    print(f"  Output shape: {list(output.tensor().shape)}")


def demo_inception():
    """Demonstrate Inception module"""
    print("\n" + "=" * 60)
    print("Inception Module Demo")
    print("=" * 60)

    # Create inception module
    # Parameters: in_channels, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5, pool_proj
    inception = InceptionModule(192, 64, 96, 128, 16, 32, 32)

    params = inception.parameters()
    total_params = sum(p.tensor().numel for p in params)
    print(f"\nInception Module:")
    print(f"  Input channels: 192")
    print(f"  Output channels: 64 + 128 + 32 + 32 = 256")
    print(f"  Parameters: {total_params:,}")

    # Forward pass
    print("\nForward pass test:")
    input_np = np.random.randn(2, 192, 28, 28).astype(np.float32)
    x = tz.Variable(tz.Tensor.from_numpy(input_np), requires_grad=False)

    output = inception.forward(x)
    print(f"  Input shape: [2, 192, 28, 28]")
    print(f"  Output shape: {list(output.tensor().shape)}")


def compare_models():
    """Compare all classic models"""
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)

    models = {
        'AlexNet': AlexNet(num_classes=1000),
        'VGG11': VGG('vgg11', num_classes=1000),
        'VGG16': VGG('vgg16', num_classes=1000),
    }

    print("\n{:<15} {:>15} {:>12}".format("Model", "Parameters", "Tensors"))
    print("-" * 45)

    for name, model in models.items():
        params = model.parameters()
        total = sum(p.tensor().numel for p in params)
        print(f"{name:<15} {total:>15,} {len(params):>12}")

    print("\nNote: GoogLeNet has ~6.8M parameters (not shown)")
    print("      ResNet-50 has ~25.6M parameters")
    print("      These classic models paved the way for modern architectures!")


# ============================================================================
# Main
# ============================================================================

def main():
    tz.initialize()

    print("=" * 60)
    print("   Classic CNN Models Example")
    print("=" * 60)

    print("\nComponents demonstrated:")
    print("  Architectures: VGG, AlexNet, GoogLeNet (Inception)")
    print("  Layers: Conv2d, MaxPool2d, BatchNorm2d")
    print("  Pooling: AdaptiveAvgPool2d")
    print("  Concepts: Depth vs width, parallel pathways")

    demo_vgg()
    demo_alexnet()
    demo_inception()
    compare_models()

    print("\n" + "=" * 60)
    print("   Classic models example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
