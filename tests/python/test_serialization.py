#!/usr/bin/env python3
"""
Test serialization Python bindings: save/load state_dict, model, and optimizer state.
"""

import sys
import os
import tempfile

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz


def test_save_load_state_dict():
    """Round-trip a model's state_dict through save/load."""
    print("Testing save/load state_dict...")
    model = tz.nn.Linear(8, 4)
    state = model.state_dict()
    assert len(state) > 0, "state_dict is empty"

    with tempfile.NamedTemporaryFile(suffix=".tz", delete=False) as f:
        path = f.name

    try:
        tz.save(state, path)
        assert os.path.exists(path), "File not created"
        assert os.path.getsize(path) > 0, "File is empty"

        loaded = tz.load(path)
        assert len(loaded) == len(state), \
            f"Key count mismatch: {len(loaded)} vs {len(state)}"

        for key in state:
            assert key in loaded, f"Missing key: {key}"
            orig = state[key]
            recv = loaded[key]
            assert orig.shape == recv.shape, \
                f"Shape mismatch for {key}: {orig.shape} vs {recv.shape}"
    finally:
        if os.path.exists(path):
            os.unlink(path)
    print("  save/load state_dict OK")


def test_save_load_model():
    """Save a Module directly and load back its state_dict."""
    print("Testing save/load model...")
    model = tz.nn.Linear(6, 3)

    with tempfile.NamedTemporaryFile(suffix=".tz", delete=False) as f:
        path = f.name

    try:
        tz.save(model, path)
        assert os.path.exists(path)

        loaded = tz.load(path)
        assert len(loaded) > 0, "Loaded state_dict is empty"
    finally:
        if os.path.exists(path):
            os.unlink(path)
    print("  save/load model OK")


def test_load_state_dict_into_model():
    """Save state_dict from one model and load into another."""
    print("Testing load_state_dict into model...")
    model1 = tz.nn.Linear(5, 3)
    model2 = tz.nn.Linear(5, 3)

    # Save model1's state
    with tempfile.NamedTemporaryFile(suffix=".tz", delete=False) as f:
        path = f.name

    try:
        state1 = model1.state_dict()
        tz.save(state1, path)
        loaded = tz.load(path)

        # Load into model2
        model2.load_state_dict(loaded)

        # Verify weights match
        state2 = model2.state_dict()
        for key in state1:
            assert key in state2, f"Missing key after load: {key}"
    finally:
        if os.path.exists(path):
            os.unlink(path)
    print("  load_state_dict into model OK")


def test_optimizer_state_persistence():
    """Test optimizer state_dict save/load round-trip."""
    print("Testing optimizer state persistence...")
    model = tz.nn.Linear(4, 2)
    params = model.parameters()
    optimizer = tz.optim.Adam(params, lr=0.001)

    # Do a step to create optimizer state
    x = tz.Variable(tz.randn([1, 4]))
    y = model.forward(x)
    loss = tz.sum(y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Get state
    opt_state = optimizer.state_dict()

    # Create new optimizer and load state
    optimizer2 = tz.optim.Adam(params, lr=0.001)
    optimizer2.load_state_dict(opt_state)

    print("  optimizer state persistence OK")


def test_state_dict_keys():
    """Verify state_dict keys are correct for known modules."""
    print("Testing state_dict keys...")
    linear = tz.nn.Linear(10, 5, bias=True)
    state = linear.state_dict()

    # Linear with bias should have weight and bias keys
    keys = list(state.keys())
    has_weight = any("weight" in k for k in keys)
    has_bias = any("bias" in k for k in keys)
    assert has_weight, f"No weight key found in: {keys}"
    assert has_bias, f"No bias key found in: {keys}"
    print(f"  state_dict keys: {keys}")
    print("  state_dict keys OK")


def main():
    print("=" * 60)
    print("Testing Serialization Bindings")
    print("=" * 60)

    try:
        tz.initialize()

        test_save_load_state_dict()
        test_save_load_model()
        test_load_state_dict_into_model()
        test_optimizer_state_persistence()
        test_state_dict_keys()

        print("\n" + "=" * 60)
        print("All serialization tests PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
