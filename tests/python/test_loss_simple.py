#!/usr/bin/env python3
"""
Simple direct test of loss bindings using Python C API.
"""

import sys
import os

# Test by importing the shared library directly
sys.path.insert(0, '/home/lee/Projects/Tenzor/build/python')

# Set library path
os.environ['LD_LIBRARY_PATH'] = '/home/lee/Projects/Tenzor/bin:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    # Try importing tenzor module
    print("Importing tenzor...")
    import tenzor
    print("✓ Import successful")

    # Check if nn module exists
    print("\nChecking nn module...")
    assert hasattr(tenzor, 'nn'), "nn module not found"
    print("✓ nn module exists")

    # Check for Reduction enum
    print("\nChecking Reduction enum...")
    assert hasattr(tenzor.nn, 'Reduction'), "Reduction enum not found"
    assert hasattr(tenzor.nn.Reduction, 'MEAN'), "Reduction.MEAN not found"
    assert hasattr(tenzor.nn.Reduction, 'SUM'), "Reduction.SUM not found"
    assert hasattr(tenzor.nn.Reduction, 'NONE'), "Reduction.NONE not found"
    print("✓ Reduction enum complete")

    # Check for loss classes
    print("\nChecking loss classes...")
    losses = ['MSELoss', 'CrossEntropyLoss', 'BCELoss', 'BCEWithLogitsLoss',
              'NLLLoss', 'L1Loss', 'SmoothL1Loss']

    for loss_name in losses:
        assert hasattr(tenzor.nn, loss_name), f"{loss_name} not found"
        print(f"  ✓ {loss_name} exists")

    # Check Sequential
    print("\nChecking Sequential...")
    assert hasattr(tenzor.nn, 'Sequential'), "Sequential not found"
    print("✓ Sequential exists")

    print("\n" + "="*60)
    print("ALL BINDING CHECKS PASSED ✓")
    print("="*60)

except ImportError as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

except AssertionError as e:
    print(f"✗ Assertion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
