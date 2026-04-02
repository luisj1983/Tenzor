"""Tests for DataLoader, Dataset, and Sampler classes."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz
tz.initialize()

import pytest


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def test_tensor_dataset_creation():
    x = tz.randn([10, 4])
    y = tz.randn([10])
    ds = tz.data.TensorDataset(x, y)
    assert len(ds) == 10


def test_tensor_dataset_getitem():
    x = tz.randn([10, 4])
    y = tz.randn([10])
    ds = tz.data.TensorDataset(x, y)
    sample = ds[0]
    assert len(sample) == 2


def test_subset():
    x = tz.randn([10, 4])
    y = tz.randn([10])
    ds = tz.data.TensorDataset(x, y)
    sub = tz.data.Subset(ds, list(range(5)))
    assert len(sub) == 5


def test_concat_dataset():
    ds1 = tz.data.TensorDataset(tz.randn([5, 4]), tz.randn([5]))
    ds2 = tz.data.TensorDataset(tz.randn([3, 4]), tz.randn([3]))
    combined = tz.data.ConcatDataset([ds1, ds2])
    assert len(combined) == 8


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def test_sequential_sampler():
    ds = tz.data.TensorDataset(tz.randn([10, 4]))
    sampler = tz.data.SequentialSampler(ds)
    indices = list(sampler)
    assert indices == list(range(10))


def test_random_sampler():
    ds = tz.data.TensorDataset(tz.randn([10, 4]))
    sampler = tz.data.RandomSampler(ds)
    indices = list(sampler)
    assert sorted(indices) == list(range(10))


def test_batch_sampler():
    ds = tz.data.TensorDataset(tz.randn([10, 4]))
    sampler = tz.data.SequentialSampler(ds)
    batch_sampler = tz.data.BatchSampler(sampler, batch_size=3, drop_last=False)
    batches = list(batch_sampler)
    assert len(batches) == 4  # 3+3+3+1
    assert len(batches[-1]) == 1


def test_batch_sampler_drop_last():
    ds = tz.data.TensorDataset(tz.randn([10, 4]))
    sampler = tz.data.SequentialSampler(ds)
    batch_sampler = tz.data.BatchSampler(sampler, batch_size=3, drop_last=True)
    batches = list(batch_sampler)
    assert len(batches) == 3  # drops last batch of 1


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

def test_dataloader_basic():
    x = tz.randn([10, 4])
    y = tz.randn([10])
    ds = tz.data.TensorDataset(x, y)
    loader = tz.data.DataLoader(ds, batch_size=3)
    batches = list(loader)
    assert len(batches) >= 3


def test_dataloader_shuffle():
    x = tz.randn([20, 4])
    ds = tz.data.TensorDataset(x)
    loader = tz.data.DataLoader(ds, batch_size=5, shuffle=True)
    batches = list(loader)
    assert len(batches) == 4


def test_dataloader_drop_last():
    x = tz.randn([10, 4])
    ds = tz.data.TensorDataset(x)
    loader = tz.data.DataLoader(ds, batch_size=3, drop_last=True)
    batches = list(loader)
    assert len(batches) == 3  # 10 // 3 = 3 (drops remainder 1)


def test_dataloader_single_batch():
    x = tz.randn([5, 4])
    ds = tz.data.TensorDataset(x)
    loader = tz.data.DataLoader(ds, batch_size=10)
    batches = list(loader)
    assert len(batches) == 1


def test_dataloader_iteration_count():
    x = tz.randn([100, 4])
    ds = tz.data.TensorDataset(x)
    loader = tz.data.DataLoader(ds, batch_size=10)
    count = sum(1 for _ in loader)
    assert count == 10


# ---------------------------------------------------------------------------
# Distributed Sampler
# ---------------------------------------------------------------------------

def test_distributed_sampler_partition():
    ds = tz.data.TensorDataset(tz.randn([10, 4]))
    sampler0 = tz.data.DistributedSampler(ds, num_replicas=2, rank=0)
    sampler1 = tz.data.DistributedSampler(ds, num_replicas=2, rank=1)
    idx0 = list(sampler0)
    idx1 = list(sampler1)
    # Should partition the dataset
    assert len(idx0) == 5
    assert len(idx1) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
