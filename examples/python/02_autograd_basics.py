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

    # Tenzor auto-initializes on import; the explicit call remains a no-op.
    tz.initialize()

    # ========================================================================
    # SECTION 1: Tensors that track gradients
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 1: Tensors with Gradient Tracking")
    print("=" * 60)

    print("\n[1.1] Creating a tensor that requires gradients:")
    x = tz.ones(2, 3, requires_grad=True)
    print(f"  shape: {x.shape}, requires_grad: {x.requires_grad}")
    print("  ✓ Gradient tracking enabled at creation")

    print("\n[1.2] Enabling tracking after creation (settable property):")
    y = tz.zeros(2, 3)
    print(f"  before: requires_grad = {y.requires_grad}")
    y.requires_grad = True
    print(f"  after:  requires_grad = {y.requires_grad}")

    # ========================================================================
    # SECTION 2: Forward Pass (Building the Computation Graph)
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 2: Forward Pass (Building Computation Graph)")
    print("=" * 60)

    print("\n[2.1] Simple forward computation:")
    a = tz.ones(2, 2, requires_grad=True)
    b = tz.ones(2, 2, requires_grad=True) * 2.0

    c = a + b
    d = (c * a).sum()   # reduce to a scalar loss

    print("  Operations performed:")
    print("    c = a + b")
    print("    d = (c * a).sum()")
    print(f"  d.grad_fn: {d.grad_fn}")
    print("  ✓ Forward pass complete, computation graph built")

    # ========================================================================
    # SECTION 3: Backward Pass (Computing Gradients)
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 3: Backward Pass (Computing Gradients)")
    print("=" * 60)

    print("\n[3.1] Computing gradients with backward():")
    # backward() on a SCALAR needs no gradient argument (like PyTorch).
    d.backward()

    # d = sum(a*(a+b)) => dd/da = 2a + b = 2*1 + 2 = 4 everywhere
    print("  ✓ Gradients computed!")
    print(f"    a.grad: shape {a.grad.shape}, value {float(a.grad.numpy()[0, 0]):.1f} (analytic: 4.0)")
    assert abs(float(a.grad.numpy()[0, 0]) - 4.0) < 1e-5

    print("\n[3.2] Non-scalar outputs need an explicit gradient argument:")
    e = tz.ones(2, 2, requires_grad=True)
    f = e * 3.0
    f.backward(tz.ones(2, 2).tensor())  # seed gradient, dL/df = 1
    print(f"    e.grad value: {float(e.grad.numpy()[0, 0]):.1f} (analytic: 3.0)")
    assert abs(float(e.grad.numpy()[0, 0]) - 3.0) < 1e-5

    # ========================================================================
    # SECTION 4: Gradient Accumulation
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 4: Gradient Accumulation")
    print("=" * 60)

    print("\n[4.1] Multiple backward passes accumulate gradients:")
    x = tz.ones(2, 2, requires_grad=True)

    (x * 2.0).sum().backward()
    g1 = float(x.grad.numpy()[0, 0])
    print(f"  After first backward:  x.grad = {g1:.1f}")

    (x * 3.0).sum().backward()
    g2 = float(x.grad.numpy()[0, 0])
    print(f"  After second backward: x.grad = {g2:.1f}  (2 + 3 accumulated)")
    assert abs(g2 - 5.0) < 1e-5
    print("  ⚠ Call optimizer.zero_grad() (or var.zero_grad()) between steps")

    # ========================================================================
    # SECTION 5: Example - Computing dy/dx for y = 3x² + 2x + 1
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 5: Example - Computing dy/dx for y = 3x² + 2x + 1")
    print("=" * 60)

    print("\n[5.1] Analytical derivative: dy/dx = 6x + 2; at x=2: 14")

    x = tz.ones(1, requires_grad=True) * 2.0   # x = 2 (graph-connected)
    y = (x * x * 3.0 + x * 2.0 + 1.0).sum()
    y.backward()

    # x is non-leaf here (ones*2.0); recompute with a leaf for clarity:
    x_leaf = tz.full([1], 2.0, requires_grad=True)
    y = (x_leaf * x_leaf * 3.0 + x_leaf * 2.0 + 1.0).sum()
    y.backward()
    dydx = float(x_leaf.grad.numpy()[0])
    print(f"  Autograd result: dy/dx = {dydx:.1f}")
    assert abs(dydx - 14.0) < 1e-5
    print("  ✓ Autograd matches the analytical derivative!")

    # ========================================================================
    # SECTION 6: Matrix Operations with Gradients
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 6: Matrix Operations with Gradients")
    print("=" * 60)

    print("\n[6.1] Matrix multiplication with gradients:")
    W = tz.randn(3, 4, requires_grad=True)
    X = tz.randn(4, 5, requires_grad=True)

    Y = tz.matmul(W, X)       # graph-connected (also: W @ X)
    loss = Y.sum()
    loss.backward()

    print(f"  W: {W.shape}, X: {X.shape}, Y = W @ X: {Y.shape}")
    print(f"  Y.grad_fn: {Y.grad_fn}")
    assert W.grad is not None and X.grad is not None
    print("  ✓ Gradients flowed back through matmul to BOTH inputs")

    # ========================================================================
    # SECTION 7: A Tiny Neural Network, by Hand
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 7: Simple Neural Network Forward/Backward")
    print("=" * 60)

    print("\n[7.1] Two-layer network: y = W2 @ (W1 @ x)")
    # Scale first, then mark as a leaf parameter (scaling a tracking tensor
    # would make it a non-leaf whose .grad is not retained).
    W1 = tz.Variable((tz.randn(20, 10) * 0.1).tensor(), requires_grad=True)
    W2 = tz.Variable((tz.randn(5, 20) * 0.1).tensor(), requires_grad=True)
    x = tz.randn(10, 32)                               # inputs: no grad needed

    hidden = tz.matmul(W1, x)
    output = tz.matmul(W2, hidden)
    loss = (output * output).mean()
    loss.backward()

    print(f"  x: {x.shape} -> hidden: {hidden.shape} -> output: {output.shape}")
    assert W1.grad is not None and W2.grad is not None
    print("  ✓ Gradients available for W1 and W2 (verified, not assumed!)")

    print("\n" + "=" * 60)
    print("TUTORIAL COMPLETE!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. tz.randn(..., requires_grad=True) or x.requires_grad = True enable autograd")
    print("2. The forward pass builds the computation graph automatically")
    print("3. Call .backward() on a SCALAR loss (use .sum()/.mean() to reduce)")
    print("4. Non-scalar backward needs an explicit seed gradient")
    print("5. Access gradients with .grad; they ACCUMULATE until zeroed")
    print("6. Keep ops on the graph — never rebuild Variables from raw tensors")

if __name__ == "__main__":
    main()
