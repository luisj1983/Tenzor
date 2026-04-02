"""Tests for memory management (caching allocator, stats, cleanup)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz
tz.initialize()

import pytest


def test_memory_stats_returns_dict():
    stats = tz.memory_stats()
    assert isinstance(stats, dict)


def test_memory_stats_has_keys():
    stats = tz.memory_stats()
    # Should have at least some allocation tracking keys
    assert len(stats) > 0


def test_reset_memory_stats():
    # Should not raise
    tz.reset_memory_stats()
    stats = tz.memory_stats()
    assert isinstance(stats, dict)


def test_empty_cache():
    # Create some tensors then free them
    tensors = [tz.randn([100, 100]) for _ in range(10)]
    del tensors
    # Should not raise
    tz.empty_cache()


def test_allocation_tracking():
    stats_before = tz.memory_stats()
    x = tz.randn([1000, 1000])
    stats_after = tz.memory_stats()
    del x
    # Stats should be retrievable
    assert isinstance(stats_before, dict)
    assert isinstance(stats_after, dict)


def test_memory_stats_cpu():
    stats = tz.memory_stats("cpu")
    assert isinstance(stats, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
