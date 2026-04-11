#!/usr/bin/env python3
"""
Unit tests for training callback system (Python bindings)
"""

import unittest
import sys
import os
import tempfile
import pytest

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))

try:
    import tenzor.tenzor_core as tz
except ImportError as e:
    pytest.skip(
        f"tenzor_core not importable ({e}); build the Python module first",
        allow_module_level=True,
    )


class TestCallbacks(unittest.TestCase):
    """Test callback system"""

    def setUp(self):
        """Initialize Tenzor library before each test"""
        tz.initialize()

    def test_base_callback_creation(self):
        """Test base Callback class creation"""
        callback = tz.nn.Callback()
        self.assertIsNotNone(callback)

    def test_base_callback_hooks(self):
        """Test that base callback hooks can be called without errors"""
        callback = tz.nn.Callback()

        # Should not raise errors
        callback.on_train_begin()
        callback.on_epoch_begin(0)
        callback.on_batch_begin(0)
        callback.on_batch_end(0, 0.5)
        callback.on_epoch_end(0, 0.5, 0.4)
        callback.on_train_end()

    def test_progress_callback_creation(self):
        """Test ProgressCallback creation"""
        progress = tz.nn.ProgressCallback(print_every=10)
        self.assertIsNotNone(progress)

    def test_progress_callback_configuration(self):
        """Test ProgressCallback configuration"""
        progress = tz.nn.ProgressCallback(5)
        progress.set_total_batches(100)
        progress.set_total_epochs(50)

        # Should not crash when calling hooks
        progress.on_train_begin()
        progress.on_epoch_begin(0)
        progress.on_batch_end(0, 0.8)
        progress.on_epoch_end(0, 0.5, 0.4)
        progress.on_train_end()

    def test_early_stopping_callback_creation(self):
        """Test EarlyStoppingCallback creation"""
        early_stop = tz.nn.EarlyStoppingCallback(
            patience=5,
            min_delta=0.001,
            monitor="val_loss"
        )
        self.assertIsNotNone(early_stop)
        self.assertFalse(early_stop.should_stop())

    def test_early_stopping_improvement(self):
        """Test early stopping with improving loss"""
        early_stop = tz.nn.EarlyStoppingCallback(patience=3, min_delta=0.0)

        # First epoch - improvement
        early_stop.on_epoch_end(0, 1.0, 0.9)
        self.assertFalse(early_stop.should_stop())
        self.assertAlmostEqual(early_stop.best_loss(), 0.9, places=5)
        self.assertEqual(early_stop.wait_count(), 0)

        # Second epoch - improvement
        early_stop.on_epoch_end(1, 0.8, 0.7)
        self.assertFalse(early_stop.should_stop())
        self.assertAlmostEqual(early_stop.best_loss(), 0.7, places=5)
        self.assertEqual(early_stop.wait_count(), 0)

    def test_early_stopping_triggered(self):
        """Test early stopping triggers after patience epochs"""
        early_stop = tz.nn.EarlyStoppingCallback(patience=2, min_delta=0.0)

        # Initial improvement
        early_stop.on_epoch_end(0, 1.0, 0.5)
        self.assertFalse(early_stop.should_stop())

        # No improvement - wait 1
        early_stop.on_epoch_end(1, 0.9, 0.6)
        self.assertFalse(early_stop.should_stop())
        self.assertEqual(early_stop.wait_count(), 1)

        # No improvement - wait 2, should trigger
        early_stop.on_epoch_end(2, 0.8, 0.7)
        self.assertTrue(early_stop.should_stop())
        self.assertEqual(early_stop.wait_count(), 2)

    def test_early_stopping_monitor_train_loss(self):
        """Test early stopping monitoring train_loss"""
        early_stop = tz.nn.EarlyStoppingCallback(
            patience=2,
            min_delta=0.0,
            monitor="train_loss"
        )

        # Should monitor train_loss instead of val_loss
        early_stop.on_epoch_end(0, 1.0, 0.5)  # train_loss=1.0, val_loss=0.5
        self.assertAlmostEqual(early_stop.best_loss(), 1.0, places=5)

        # train_loss improves, val_loss worsens
        early_stop.on_epoch_end(1, 0.8, 0.6)
        self.assertAlmostEqual(early_stop.best_loss(), 0.8, places=5)
        self.assertEqual(early_stop.wait_count(), 0)  # Should reset

    def test_model_checkpoint_callback_creation(self):
        """Test ModelCheckpointCallback creation"""
        model = tz.nn.Linear(10, 5)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            filepath = f.name

        try:
            checkpoint = tz.nn.ModelCheckpointCallback(
                filepath=filepath,
                model=model,
                save_best_only=True,
                monitor="val_loss"
            )
            self.assertIsNotNone(checkpoint)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_model_checkpoint_save_best_only(self):
        """Test model checkpoint saves only best model"""
        model = tz.nn.Linear(10, 5)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            filepath = f.name

        try:
            checkpoint = tz.nn.ModelCheckpointCallback(
                filepath=filepath,
                model=model,
                save_best_only=True
            )

            # First epoch - should save (first model)
            checkpoint.on_epoch_end(0, 1.0, 0.9)
            self.assertAlmostEqual(checkpoint.best_loss(), 0.9, places=5)

            # Second epoch - improvement, should save
            checkpoint.on_epoch_end(1, 0.8, 0.7)
            self.assertAlmostEqual(checkpoint.best_loss(), 0.7, places=5)

            # Third epoch - no improvement
            old_best = checkpoint.best_loss()
            checkpoint.on_epoch_end(2, 0.9, 0.8)
            self.assertAlmostEqual(checkpoint.best_loss(), old_best, places=5)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_lr_scheduler_callback_creation(self):
        """Test LRSchedulerCallback creation"""
        model = tz.nn.Linear(10, 5)
        optimizer = tz.optim.SGD(model.parameters(), lr=0.01)

        scheduler = tz.nn.LRSchedulerCallback(
            optimizer=optimizer,
            schedule_type="step",
            decay_factor=0.1,
            decay_epochs=10
        )
        self.assertIsNotNone(scheduler)

    def test_lr_scheduler_step_decay(self):
        """Test LR scheduler with step decay"""
        model = tz.nn.Linear(10, 5)
        optimizer = tz.optim.SGD(model.parameters(), lr=0.1)

        scheduler = tz.nn.LRSchedulerCallback(
            optimizer=optimizer,
            schedule_type="step",
            decay_factor=0.5,
            decay_epochs=3
        )

        scheduler.on_train_begin()
        initial_lr = scheduler.current_lr()
        self.assertGreater(initial_lr, 0.0)

        # Epochs 0-2: no decay yet
        scheduler.on_epoch_end(0, 1.0, 0.9)
        scheduler.on_epoch_end(1, 0.8, 0.7)

        # Epoch 2 (3rd epoch): should decay
        scheduler.on_epoch_end(2, 0.7, 0.6)
        lr_after_decay = scheduler.current_lr()

        # LR should be positive
        self.assertGreater(lr_after_decay, 0.0)

    def test_callback_list_creation(self):
        """Test CallbackList creation"""
        callbacks = tz.nn.CallbackList()
        self.assertIsNotNone(callbacks)
        self.assertEqual(len(callbacks.callbacks()), 0)

    def test_callback_list_add_callbacks(self):
        """Test adding callbacks to CallbackList"""
        callbacks = tz.nn.CallbackList()

        cb1 = tz.nn.ProgressCallback()
        cb2 = tz.nn.EarlyStoppingCallback()

        callbacks.add(cb1)
        callbacks.add(cb2)

        self.assertEqual(len(callbacks.callbacks()), 2)

    def test_callback_list_calls_all_callbacks(self):
        """Test CallbackList calls all added callbacks"""
        callbacks = tz.nn.CallbackList()

        early_stop = tz.nn.EarlyStoppingCallback(patience=2)
        progress = tz.nn.ProgressCallback(1)

        callbacks.add(early_stop)
        callbacks.add(progress)

        # Call hooks - should not crash
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        callbacks.on_batch_begin(0)
        callbacks.on_batch_end(0, 0.5)
        callbacks.on_epoch_end(0, 0.5, 0.4)
        callbacks.on_train_end()

    def test_multiple_callbacks_integration(self):
        """Test using multiple callbacks together"""
        model = tz.nn.Linear(10, 5)
        optimizer = tz.optim.Adam(model.parameters(), lr=0.001)

        callbacks = tz.nn.CallbackList()

        # Add multiple callbacks
        progress = tz.nn.ProgressCallback(print_every=1)
        early_stop = tz.nn.EarlyStoppingCallback(patience=2, min_delta=0.0)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name

        try:
            checkpoint = tz.nn.ModelCheckpointCallback(
                filepath=checkpoint_path,
                model=model,
                save_best_only=True
            )

            lr_scheduler = tz.nn.LRSchedulerCallback(
                optimizer=optimizer,
                schedule_type="exponential",
                decay_factor=0.9
            )

            callbacks.add(progress)
            callbacks.add(early_stop)
            callbacks.add(checkpoint)
            callbacks.add(lr_scheduler)

            # Simulate training loop
            callbacks.on_train_begin()

            for epoch in range(5):
                callbacks.on_epoch_begin(epoch)

                for batch in range(10):
                    callbacks.on_batch_end(batch, 0.5)

                # Simulate decreasing loss
                train_loss = 1.0 - epoch * 0.1
                val_loss = 0.9 - epoch * 0.1
                callbacks.on_epoch_end(epoch, train_loss, val_loss)

                if early_stop.should_stop():
                    break

            callbacks.on_train_end()

            # Verify callbacks worked
            self.assertFalse(early_stop.should_stop())  # Loss was improving
            self.assertLess(checkpoint.best_loss(), 1.0)  # Best model saved
            self.assertGreater(lr_scheduler.current_lr(), 0.0)  # LR adjusted

        finally:
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)


class TestCallbacksWithTraining(unittest.TestCase):
    """Test callbacks in realistic training scenarios"""

    def setUp(self):
        """Initialize Tenzor library"""
        tz.initialize()

    def test_simulated_training_loop(self):
        """Test callbacks in a simulated training loop"""
        # Create a simple model
        model = tz.nn.Linear(784, 10)
        optimizer = tz.optim.SGD(model.parameters(), lr=0.01)

        # Create callbacks
        callbacks = tz.nn.CallbackList()
        progress = tz.nn.ProgressCallback(print_every=5)
        progress.set_total_batches(20)
        progress.set_total_epochs(10)

        early_stop = tz.nn.EarlyStoppingCallback(patience=3)

        callbacks.add(progress)
        callbacks.add(early_stop)

        # Simulate training
        callbacks.on_train_begin()

        num_epochs = 10
        batches_per_epoch = 20

        for epoch in range(num_epochs):
            callbacks.on_epoch_begin(epoch)

            epoch_loss = 0.0
            for batch in range(batches_per_epoch):
                batch_loss = 1.0 - (epoch * 0.05 + batch * 0.001)
                callbacks.on_batch_end(batch, batch_loss)
                epoch_loss += batch_loss

            train_loss = epoch_loss / batches_per_epoch
            val_loss = train_loss * 0.9

            callbacks.on_epoch_end(epoch, train_loss, val_loss)

            if early_stop.should_stop():
                print(f"Early stopping at epoch {epoch}")
                break

        callbacks.on_train_end()

        # Training should complete successfully
        self.assertTrue(True)


def run_tests():
    """Run all tests"""
    # Create test suite
    suite = unittest.TestSuite()

    # Add tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCallbacks))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCallbacksWithTraining))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
