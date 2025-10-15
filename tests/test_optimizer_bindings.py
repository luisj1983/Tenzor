#!/usr/bin/env python3
"""Test that Python optimizer bindings work with the new shared_ptr API"""

import sys
import os

# Add the build directory to the path to import tenzor_core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tenzor_core as tz

def test_optimizer_creation():
    """Test that optimizers can be created with module parameters"""
    print("Testing optimizer creation with module parameters...")

    # Initialize tenzor
    tz.initialize()

    # Create a simple linear module
    linear = tz.nn.Linear(10, 5)

    # Get parameters (should be list of shared_ptr<Variable>)
    params = linear.parameters()
    print(f"Got {len(params)} parameters from Linear module")

    # Test SGD optimizer
    try:
        sgd = tz.optim.SGD(params, lr=0.01, momentum=0.9)
        print("✓ SGD optimizer created successfully")
    except Exception as e:
        print(f"✗ SGD optimizer failed: {e}")
        return False

    # Test Adam optimizer
    try:
        adam = tz.optim.Adam(params, lr=0.001)
        print("✓ Adam optimizer created successfully")
    except Exception as e:
        print(f"✗ Adam optimizer failed: {e}")
        return False

    # Test AdamW optimizer
    try:
        adamw = tz.optim.AdamW(params, lr=0.001, weight_decay=0.01)
        print("✓ AdamW optimizer created successfully")
    except Exception as e:
        print(f"✗ AdamW optimizer failed: {e}")
        return False

    # Test RMSprop optimizer
    try:
        rmsprop = tz.optim.RMSprop(params, lr=0.01)
        print("✓ RMSprop optimizer created successfully")
    except Exception as e:
        print(f"✗ RMSprop optimizer failed: {e}")
        return False

    # Test Adagrad optimizer
    try:
        adagrad = tz.optim.Adagrad(params, lr=0.01)
        print("✓ Adagrad optimizer created successfully")
    except Exception as e:
        print(f"✗ Adagrad optimizer failed: {e}")
        return False

    # Test Adadelta optimizer
    try:
        adadelta = tz.optim.Adadelta(params)
        print("✓ Adadelta optimizer created successfully")
    except Exception as e:
        print(f"✗ Adadelta optimizer failed: {e}")
        return False

    print("\nAll optimizer tests passed!")
    return True

if __name__ == "__main__":
    success = test_optimizer_creation()
    sys.exit(0 if success else 1)
