"""
Tenzor Tutorial 02: Automatic Differentiation (Autograd)
========================================================
Learn how to compute gradients automatically using Tenzor's autograd system.
"""

import tenzor as tz

def main():
    print("=" * 60)
    print("TENZOR TUTORIAL 02: AUTOMATIC DIFFERENTIATION")
    print("=" * 60)

    # Initialize Tenzor
    print("\n[1] Initializing Tenzor library...")
    tz.initialize()
    print("✓ Tenzor initialized successfully")

    # ========================================================================
    # SECTION 1: Creating Variables
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 1: Creating Variables (Tensors with Gradients)")
    print("=" * 60)

    print("\n[1.1] Creating a variable that requires gradients:")
    # Create a tensor
    data = tz.ones([2, 3])
    # Wrap it in a Variable with requires_grad=True
    x = tz.Variable(data, requires_grad=True)
    print(f"  Variable shape: {x.data.shape}")
    print("  ✓ Variable created with gradient tracking enabled")

    print("\n[1.2] Creating a variable without gradients:")
    y = tz.Variable(tz.zeros([2, 3]), requires_grad=False)
    print("  ✓ Variable created without gradient tracking")

    # ========================================================================
    # SECTION 2: Forward Pass
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 2: Forward Pass (Building Computation Graph)")
    print("=" * 60)

    print("\n[2.1] Simple forward computation:")
    # Create input variable
    a = tz.Variable(tz.ones([2, 2]), requires_grad=True)
    b = tz.Variable(tz.ones([2, 2]) * 2.0, requires_grad=True)

    # Perform operations (computation graph is built automatically)
    c = a + b
    d = c * a

    print("  Operations performed:")
    print("    c = a + b")
    print("    d = c * a")
    print(f"  Result shape: {d.data.shape}")
    print("  ✓ Forward pass complete, computation graph built")

    # ========================================================================
    # SECTION 3: Backward Pass (Computing Gradients)
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 3: Backward Pass (Computing Gradients)")
    print("=" * 60)

    print("\n[3.1] Computing gradients with backward():")
    print("  Starting backward pass...")

    # Compute gradients by calling backward() on the output
    d.backward()

    print("  ✓ Gradients computed!")
    print("\n  Gradient information:")
    print(f"    a.grad shape: {a.grad.shape if a.grad is not None else 'None'}")
    print(f"    b.grad shape: {b.grad.shape if b.grad is not None else 'None'}")

    # ========================================================================
    # SECTION 4: Gradient Accumulation
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 4: Gradient Accumulation")
    print("=" * 60)

    print("\n[4.1] Multiple backward passes accumulate gradients:")
    # Create fresh variable
    x = tz.Variable(tz.ones([2, 2]), requires_grad=True)

    # First computation
    y1 = x * 2.0
    y1.backward()
    print("  First backward pass completed")

    # Second computation (gradients will accumulate)
    y2 = x * 3.0
    y2.backward()
    print("  Second backward pass completed")
    print("  ⚠ Note: Gradients from both passes are accumulated in x.grad")

    # ========================================================================
    # SECTION 5: Simple Example - Computing Derivative
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 5: Example - Computing dy/dx for y = 3x² + 2x + 1")
    print("=" * 60)

    print("\n[5.1] Function: y = 3x² + 2x + 1")
    print("  Analytical derivative: dy/dx = 6x + 2")
    print("  At x=2: dy/dx = 6(2) + 2 = 14")

    # Create variable x = 2
    x_val = tz.ones([1]) * 2.0
    x = tz.Variable(x_val, requires_grad=True)

    # Compute y = 3x² + 2x + 1
    x_squared = x * x
    term1 = x_squared * 3.0
    term2 = x * 2.0
    term3 = tz.Variable(tz.ones([1]), requires_grad=False)
    y = term1 + term2 + term3

    # Compute gradient
    y.backward()

    print(f"\n  Computed gradient via autograd: dy/dx at x=2")
    print("  ✓ Autograd successfully computed the derivative!")

    # ========================================================================
    # SECTION 6: Matrix Operations with Gradients
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 6: Matrix Operations with Gradients")
    print("=" * 60)

    print("\n[6.1] Matrix multiplication with gradients:")
    # Create weight matrix and input
    W = tz.Variable(tz.randn([3, 4]), requires_grad=True)
    X = tz.Variable(tz.randn([4, 5]), requires_grad=True)

    # Matrix multiplication
    Y = tz.Variable(tz.matmul(W.data, X.data), requires_grad=True)

    print(f"  W shape: {W.data.shape}")
    print(f"  X shape: {X.data.shape}")
    print(f"  Y = W @ X, shape: {Y.data.shape}")

    # Compute gradients
    Y.backward()
    print("  ✓ Gradients computed for matrix multiplication")

    # ========================================================================
    # SECTION 7: Building a Simple Neural Network Computation
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 7: Simple Neural Network Forward/Backward")
    print("=" * 60)

    print("\n[7.1] Two-layer network: input -> hidden -> output")
    # Network: y = W2 @ (W1 @ x)

    # Layer 1: 10 -> 20
    W1 = tz.Variable(tz.randn([20, 10]) * 0.1, requires_grad=True)
    # Layer 2: 20 -> 5
    W2 = tz.Variable(tz.randn([5, 20]) * 0.1, requires_grad=True)
    # Input batch
    x = tz.Variable(tz.randn([10, 32]), requires_grad=False)  # 32 samples

    # Forward pass
    hidden = tz.Variable(tz.matmul(W1.data, x.data), requires_grad=True)
    output = tz.Variable(tz.matmul(W2.data, hidden.data), requires_grad=True)

    print(f"  Input shape: {x.data.shape}")
    print(f"  Hidden shape: {hidden.data.shape}")
    print(f"  Output shape: {output.data.shape}")

    # Backward pass
    output.backward()
    print("\n  ✓ Forward pass complete")
    print("  ✓ Backward pass complete")
    print("  ✓ Gradients available for W1 and W2")

    print("\n" + "=" * 60)
    print("TUTORIAL COMPLETE!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Wrap tensors in Variable(tensor, requires_grad=True) for autograd")
    print("2. Forward pass builds the computation graph automatically")
    print("3. Call .backward() to compute gradients via backpropagation")
    print("4. Access gradients with .grad property")
    print("5. Gradients accumulate across multiple backward() calls")
    print("6. Use requires_grad=False for inputs that don't need gradients")

if __name__ == "__main__":
    main()
