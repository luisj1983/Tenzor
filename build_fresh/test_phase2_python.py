#!/usr/bin/env python3
"""
Test Phase 2 Python bindings: Training API, DataLoader, Callbacks
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

try:
    import tenzor_core as tz
    print("✅ Successfully imported tenzor_core")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

tests_passed = 0
tests_failed = 0

def test(name, func):
    global tests_passed, tests_failed
    try:
        func()
        print(f"✅ {name}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ {name}: {e}")
        tests_failed += 1

print("\n" + "="*70)
print("PHASE 2 PYTHON BINDINGS TEST")
print("="*70)

# Test 1: NeuralNetwork class exists
def test_neural_network_class():
    assert hasattr(tz.nn, "NeuralNetwork")
    print("   NeuralNetwork class found")

# Test 2: DataLoader class exists
def test_dataloader_class():
    assert hasattr(tz.nn, "SimpleDataLoader") or hasattr(tz.nn, "DataLoader")
    print("   DataLoader class found")

# Test 3: Callback classes exist
def test_callback_classes():
    assert hasattr(tz.nn, "Callback")
    assert hasattr(tz.nn, "ProgressCallback")
    assert hasattr(tz.nn, "EarlyStoppingCallback")
    assert hasattr(tz.nn, "ModelCheckpointCallback")
    assert hasattr(tz.nn, "LRSchedulerCallback")
    print("   All 5 callback classes found")

# Test 4: Create NeuralNetwork instance
def test_create_neural_network():
    model = tz.nn.Sequential(
        tz.nn.Linear(10, 20),
        tz.nn.ReLU(),
        tz.nn.Linear(20, 5)
    )
    optimizer = tz.optim.Adam(model.parameters(), lr=0.001)
    criterion = tz.nn.MSELoss()

    # NeuralNetwork needs loss function wrapper
    nn = tz.nn.NeuralNetwork(model, optimizer, criterion)
    assert nn is not None
    print("   NeuralNetwork instance created successfully")

# Test 5: Create DataLoader instance
def test_create_dataloader():
    # Create dummy data
    data = []
    for i in range(10):
        input_tensor = tz.randn([5], tz.DType.Float32, tz.Device.cpu())
        target_tensor = tz.randn([3], tz.DType.Float32, tz.Device.cpu())
        data.append((input_tensor, target_tensor))

    loader_class = getattr(tz.nn, "SimpleDataLoader", getattr(tz.nn, "DataLoader", None))
    if loader_class:
        loader = loader_class(data, batch_size=2)
        assert loader is not None
        print(f"   DataLoader instance created with {loader.size()} batches")
    else:
        raise Exception("No DataLoader class found")

# Test 6: Create callbacks
def test_create_callbacks():
    progress_cb = tz.nn.ProgressCallback()
    early_stop_cb = tz.nn.EarlyStoppingCallback(patience=5)
    checkpoint_cb = tz.nn.ModelCheckpointCallback(filepath="/tmp/model.bin")
    lr_sched_cb = tz.nn.LRSchedulerCallback(schedule_type="step", step_size=10)

    assert progress_cb is not None
    assert early_stop_cb is not None
    assert checkpoint_cb is not None
    assert lr_sched_cb is not None
    print("   All callback instances created successfully")

test("NeuralNetwork class", test_neural_network_class)
test("DataLoader class", test_dataloader_class)
test("Callback classes", test_callback_classes)
test("Create NeuralNetwork", test_create_neural_network)
test("Create DataLoader", test_create_dataloader)
test("Create Callbacks", test_create_callbacks)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Tests Passed: {tests_passed}")
print(f"Tests Failed: {tests_failed}")
print(f"Success Rate: {100*tests_passed/(tests_passed+tests_failed):.1f}%")
print("="*70)

if tests_failed == 0:
    print("\n🎉 ALL PHASE 2 PYTHON BINDINGS WORKING! 🎉")
    sys.exit(0)
else:
    print(f"\n⚠️  {tests_failed} test(s) failed.")
    sys.exit(1)
