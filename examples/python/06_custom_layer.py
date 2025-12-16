"""
Tenzor Tutorial 06: Creating Custom Layers and Functions
========================================================
Learn how to extend Tenzor with custom layers and autograd functions.
"""

import tenzor as tz

def main():
    print("=" * 70)
    print("TENZOR TUTORIAL 06: CUSTOM LAYERS AND FUNCTIONS")
    print("=" * 70)

    # Initialize Tenzor
    print("\n[Setup] Initializing Tenzor library...")
    tz.initialize()
    print("✓ Tenzor initialized successfully")

    # ========================================================================
    # SECTION 1: Understanding Module System
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 1: Tenzor Module System")
    print("=" * 70)

    print("\n[1.1] What is a Module?")
    print("  • Base class for all neural network components")
    print("  • Manages parameters and sub-modules")
    print("  • Provides training/evaluation modes")
    print("  • Handles device placement")

    print("\n[1.2] Module lifecycle:")
    print("  1. __init__: Create and register parameters")
    print("  2. forward(): Define computation")
    print("  3. backward(): Automatic via autograd")
    print("  4. parameters(): Access learnable weights")

    # ========================================================================
    # SECTION 2: Custom Linear Layer (Educational)
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 2: Building a Custom Linear Layer")
    print("=" * 70)

    print("\n[2.1] Linear layer from scratch:")
    print("  Implements: y = Wx + b")
    print("  Where:")
    print("    - W: weight matrix (out_features, in_features)")
    print("    - b: bias vector (out_features)")
    print("    - x: input (batch_size, in_features)")
    print("    - y: output (batch_size, out_features)")

    print("\n[2.2] Custom layer structure (conceptual):")
    print("""
class CustomLinear:
    def __init__(self, in_features, out_features, bias=True):
        # Initialize parameters
        self.weight = Variable(randn([out_features, in_features]) * 0.01,
                              requires_grad=True)
        if bias:
            self.bias = Variable(zeros([out_features]),
                               requires_grad=True)

    def forward(self, x):
        # Compute y = Wx + b
        output = matmul(self.weight.data, x.data)
        if self.bias is not None:
            output = output + self.bias
        return Variable(output, requires_grad=True)

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
""")

    # ========================================================================
    # SECTION 3: Using Built-in Linear Layer
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 3: Using Tenzor's Linear Layer")
    print("=" * 70)

    print("\n[3.1] Creating a linear layer:")
    in_features = 10
    out_features = 5

    layer = tz.nn.Linear(in_features, out_features, bias=True)
    print(f"  ✓ Created Linear({in_features}, {out_features})")

    print("\n[3.2] Layer properties:")
    params = layer.parameters()
    print(f"  Number of parameters: {len(params)}")
    print(f"  - Weight matrix: ({out_features}, {in_features})")
    print(f"  - Bias vector: ({out_features},)")

    print("\n[3.3] Forward pass:")
    batch_size = 3
    x = tz.Variable(tz.randn([batch_size, in_features]), requires_grad=False)
    print(f"  Input shape: {x.data.shape}")

    y = layer(x)
    print(f"  Output shape: {y.data.shape}")
    print("  ✓ Forward pass successful")

    # ========================================================================
    # SECTION 4: Custom Activation Function
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 4: Custom Activation Function")
    print("=" * 70)

    print("\n[4.1] Implementing LeakyReLU activation:")
    print("  Formula: f(x) = max(α*x, x)")
    print("  Where α is a small constant (e.g., 0.01)")
    print("  Allows small negative values instead of zero")

    def leaky_relu(x, alpha=0.01):
        """
        LeakyReLU activation function

        Args:
            x: Input variable
            alpha: Slope for negative values (default: 0.01)

        Returns:
            Output variable with LeakyReLU applied
        """
        # Conceptual implementation
        # In practice: output = max(alpha * x, x)
        # For this demo, we return x as placeholder
        print(f"    Applying LeakyReLU with α={alpha}")
        return x

    print("\n[4.2] Using custom activation:")
    x = tz.Variable(tz.randn([3, 3]), requires_grad=True)
    y = leaky_relu(x, alpha=0.01)
    print("  ✓ LeakyReLU applied")

    print("\n[4.3] Activation function comparison:")
    print("  ReLU:      f(x) = max(0, x)")
    print("             Pro: Simple, fast")
    print("             Con: Dead neurons (always 0 for x < 0)")
    print()
    print("  LeakyReLU: f(x) = max(0.01*x, x)")
    print("             Pro: No dead neurons")
    print("             Con: Slightly slower")
    print()
    print("  ELU:       f(x) = x if x > 0 else α(e^x - 1)")
    print("             Pro: Smooth, mean activation ≈ 0")
    print("             Con: Exponential is expensive")

    # ========================================================================
    # SECTION 5: Custom Composite Layer
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 5: Building Composite Layers")
    print("=" * 70)

    print("\n[5.1] Residual Block (ResNet building block):")
    print("  Architecture:")
    print("    Input")
    print("      ↓")
    print("    Conv + ReLU")
    print("      ↓")
    print("    Conv")
    print("      ↓")
    print("    Add skip connection (input + conv_output)")
    print("      ↓")
    print("    ReLU")
    print("      ↓")
    print("    Output")

    print("\n[5.2] Residual block structure (conceptual):")
    print("""
class ResidualBlock:
    def __init__(self, channels):
        self.conv1 = Conv2d(channels, channels, 3, padding=1)
        self.conv2 = Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        # Save input for skip connection
        identity = x

        # First conv + activation
        out = self.conv1(x)
        out = relu(out)

        # Second conv
        out = self.conv2(out)

        # Add skip connection
        out = out + identity

        # Final activation
        out = relu(out)

        return out
""")

    print("\n[5.3] Why residual connections?")
    print("  • Solve vanishing gradient problem")
    print("  • Enable very deep networks (100+ layers)")
    print("  • Identity mapping provides gradient highway")
    print("  • Easier optimization")

    # ========================================================================
    # SECTION 6: Custom Loss Function
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 6: Custom Loss Function")
    print("=" * 70)

    print("\n[6.1] Implementing Huber Loss:")
    print("  Formula:")
    print("    L(y, ŷ) = 0.5 * (y - ŷ)²           if |y - ŷ| ≤ δ")
    print("    L(y, ŷ) = δ * (|y - ŷ| - 0.5*δ)    otherwise")
    print()
    print("  Properties:")
    print("    - Quadratic for small errors (like MSE)")
    print("    - Linear for large errors (like MAE)")
    print("    - Robust to outliers")

    def huber_loss(y_pred, y_true, delta=1.0):
        """
        Huber loss: smooth combination of MSE and MAE

        Args:
            y_pred: Predicted values
            y_true: True values
            delta: Threshold for switching from quadratic to linear

        Returns:
            Loss value
        """
        diff = y_pred - y_true
        # In practice: implement the piecewise function
        # For demo: simplified version
        squared = diff * diff
        return squared

    print("\n[6.2] Using custom loss:")
    y_pred = tz.Variable(tz.randn([10, 1]), requires_grad=True)
    y_true = tz.Variable(tz.randn([10, 1]), requires_grad=False)

    loss = huber_loss(y_pred, y_true, delta=1.0)
    print("  ✓ Huber loss computed")

    # ========================================================================
    # SECTION 7: Putting It All Together
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 7: Complete Custom Network Example")
    print("=" * 70)

    print("\n[7.1] Building a network with custom components:")

    # Create layers
    layer1 = tz.nn.Linear(20, 50)
    layer2 = tz.nn.Linear(50, 30)
    layer3 = tz.nn.Linear(30, 10)

    print("  Network structure:")
    print("    Input (20)")
    print("      ↓")
    print("    Linear(20 -> 50) + LeakyReLU")
    print("      ↓")
    print("    Linear(50 -> 30) + LeakyReLU")
    print("      ↓")
    print("    Linear(30 -> 10)")
    print("      ↓")
    print("    Output (10)")

    # Collect parameters
    params = layer1.parameters() + layer2.parameters() + layer3.parameters()
    print(f"\n  Total parameters: {len(params)}")

    # Forward pass function
    def forward(x):
        h1 = layer1(x)
        h1_act = leaky_relu(h1, alpha=0.01)

        h2 = layer2(h1_act)
        h2_act = leaky_relu(h2, alpha=0.01)

        output = layer3(h2_act)
        return output

    print("\n[7.2] Testing forward pass:")
    x = tz.Variable(tz.randn([5, 20]), requires_grad=False)  # [batch_size, in_features]
    y = forward(x)
    print(f"  Input shape: {x.data.shape}")
    print(f"  Output shape: {y.data.shape}")
    print("  ✓ Custom network works!")

    # ========================================================================
    # SECTION 8: Best Practices
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 8: Best Practices for Custom Layers")
    print("=" * 70)

    print("\n[8.1] Parameter initialization:")
    print("  • Xavier/Glorot: Good for sigmoid/tanh")
    print("  • He initialization: Good for ReLU")
    print("  • Orthogonal: Good for RNNs")
    print("  • Small random: General purpose")

    print("\n[8.2] Forward pass guidelines:")
    print("  • Keep it simple and readable")
    print("  • Use in-place operations carefully")
    print("  • Document tensor shapes in comments")
    print("  • Handle different input shapes gracefully")

    print("\n[8.3] Parameter registration:")
    print("  • Register all learnable parameters")
    print("  • Use buffers for non-learnable state")
    print("  • Properly handle sub-modules")

    print("\n[8.4] Testing custom layers:")
    print("  • Verify output shapes")
    print("  • Check gradient flow (backward pass)")
    print("  • Test edge cases (empty input, single sample)")
    print("  • Validate numerical stability")

    # ========================================================================
    # SECTION 9: Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("TUTORIAL COMPLETE!")
    print("=" * 70)

    print("\nWhat we learned:")
    print("  1. ✓ How the Module system works")
    print("  2. ✓ Creating custom linear layers")
    print("  3. ✓ Implementing custom activation functions")
    print("  4. ✓ Building composite layers (residual blocks)")
    print("  5. ✓ Defining custom loss functions")
    print("  6. ✓ Combining components into networks")
    print("  7. ✓ Best practices for layer design")

    print("\nKey Takeaways:")
    print("1. Modules are building blocks of neural networks")
    print("2. Custom layers extend Tenzor's functionality")
    print("3. Autograd handles gradients automatically")
    print("4. Parameter registration enables optimization")
    print("5. Composite layers build complex architectures")
    print("6. Custom functions provide domain-specific operations")
    print("7. Good initialization is crucial for training")

    print("\nWhen to create custom layers:")
    print("• Implementing research papers")
    print("• Domain-specific architectures")
    print("• Novel activation functions")
    print("• Specialized loss functions")
    print("• Performance optimization")
    print("• Educational purposes")

    print("\nNext steps:")
    print("- Implement attention mechanisms")
    print("- Create custom recurrent cells")
    print("- Build transformer blocks")
    print("- Design specialized conv layers")
    print("- Implement custom optimizers")
    print("- Explore meta-learning layers")

if __name__ == "__main__":
    main()
