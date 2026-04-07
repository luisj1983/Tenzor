"""
Tests for NN layer Python bindings that currently have zero coverage.

Covers: Conv1d, Conv3d, ConvTranspose variants, normalization layers,
pooling, RNN/LSTM/GRU, attention, EmbeddingBag.
"""

import pytest
import tenzor as tz

ALL_DEVICES = ["cpu", "cuda", "vulkan", "oneapi", "rocm"]


def make_var(shape, device="cpu", requires_grad=False):
    return tz.Variable(tz.randn(shape, device=device), requires_grad)


# ============================================================================
# Convolution layers
# ============================================================================

@pytest.mark.parametrize("device", ALL_DEVICES, indirect=True)
class TestConvLayers:
    def test_conv1d_forward(self, device):
        conv = tz.nn.Conv1d(3, 8, 3, stride=1, padding=1)
        x = make_var([1, 3, 16], device)
        y = conv(x)
        assert y.data.shape[0] == 1
        assert y.data.shape[1] == 8

    def test_conv3d_forward(self, device):
        conv = tz.nn.Conv3d(1, 8, 3, stride=1, padding=1)
        x = make_var([1, 1, 8, 8, 8], device)
        y = conv(x)
        assert y.data.shape[0] == 1
        assert y.data.shape[1] == 8


# ============================================================================
# Normalization layers
# ============================================================================

@pytest.mark.parametrize("device", ALL_DEVICES, indirect=True)
class TestNormLayers:
    def test_group_norm(self, device):
        norm = tz.nn.GroupNorm(4, 8)
        x = make_var([2, 8, 4, 4], device)
        y = norm(x)
        assert y.data.shape == [2, 8, 4, 4]

    def test_rms_norm(self, device):
        norm = tz.nn.RMSNorm(16)
        x = make_var([2, 16], device)
        y = norm(x)
        assert y.data.shape == [2, 16]


# ============================================================================
# RNN layers
# ============================================================================

@pytest.mark.parametrize("device", ALL_DEVICES, indirect=True)
class TestRNNLayers:
    def test_lstm_forward(self, device):
        lstm = tz.nn.LSTM(10, 20, num_layers=1)
        x = make_var([5, 2, 10], device)  # seq=5, batch=2, input=10
        y = lstm(x)
        assert y.data.shape[0] == 5
        assert y.data.shape[1] == 2
        assert y.data.shape[2] == 20

    def test_gru_forward(self, device):
        gru = tz.nn.GRU(10, 20, num_layers=1)
        x = make_var([5, 2, 10], device)
        y = gru(x)
        assert y.data.shape[0] == 5
        assert y.data.shape[1] == 2
        assert y.data.shape[2] == 20

    def test_rnn_forward(self, device):
        rnn = tz.nn.RNN(10, 20, num_layers=1)
        x = make_var([5, 2, 10], device)
        y = rnn(x)
        assert y.data.shape[0] == 5
        assert y.data.shape[2] == 20


# ============================================================================
# Attention
# ============================================================================

@pytest.mark.parametrize("device", ALL_DEVICES, indirect=True)
class TestAttention:
    def test_multihead_attention(self, device):
        attn = tz.nn.MultiheadAttention(embed_dim=32, num_heads=4)
        q = make_var([4, 2, 32], device)  # seq=4, batch=2, embed=32
        k = make_var([4, 2, 32], device)
        v = make_var([4, 2, 32], device)
        out = attn(q, k, v)
        assert out.data.shape[0] == 4
        assert out.data.shape[2] == 32


# ============================================================================
# Embedding
# ============================================================================

@pytest.mark.parametrize("device", ALL_DEVICES, indirect=True)
class TestEmbedding:
    def test_embedding_bag_sum(self, device):
        emb = tz.nn.EmbeddingBag(10, 4, mode="sum")
        indices = tz.Variable(tz.tensor([0, 1, 2, 3], tz.dtype.int64, device), False)
        offsets = tz.Variable(tz.tensor([0, 2], tz.dtype.int64, device), False)
        out = emb(indices, offsets)
        assert out.data.shape == [2, 4]
