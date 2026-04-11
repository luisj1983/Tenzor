"""
Tests for model zoo Python bindings — instantiation and forward pass.

Verifies that all pre-defined model architectures can be instantiated
and produce correct output shapes with dummy inputs.
Uses small configurations to keep memory/time low.
"""

import pytest
import tenzor as tz

ALL_DEVICES = ["cpu", "cuda", "vulkan", "oneapi", "rocm"]


def make_var(shape, device="cpu"):
    return tz.Variable(tz.randn(shape, device=device), False)


# ============================================================================
# Vision models — instantiation + forward
# ============================================================================

class TestVisionModels:
    def test_resnet18(self):
        model = tz.models.resnet18(num_classes=10)
        x = make_var([1, 3, 32, 32])
        y = model(x)
        assert y.data.shape[0] == 1
        assert y.data.shape[1] == 10

    def test_resnet50(self):
        model = tz.models.resnet50(num_classes=10)
        x = make_var([1, 3, 32, 32])
        y = model(x)
        assert y.data.shape[1] == 10

    def test_vgg16(self):
        model = tz.models.vgg16(num_classes=10)
        x = make_var([1, 3, 32, 32])
        y = model(x)
        assert y.data.shape[1] == 10

    def test_mobilenet_v2(self):
        model = tz.models.mobilenet_v2(num_classes=10)
        x = make_var([1, 3, 32, 32])
        y = model(x)
        assert y.data.shape[1] == 10

    def test_efficientnet_b0(self):
        model = tz.models.efficientnet_b0(num_classes=10)
        x = make_var([1, 3, 32, 32])
        y = model(x)
        assert y.data.shape[1] == 10


# ============================================================================
# NLP models — instantiation + forward
# ============================================================================

class TestNLPModels:
    def test_bert_instantiation(self):
        config = tz.models.BertConfig()
        config.vocab_size = 100
        config.hidden_size = 64
        config.num_attention_heads = 2
        config.num_hidden_layers = 1
        config.intermediate_size = 128
        model = tz.models.BertModel(config)
        # Forward with dummy input ids
        input_ids = tz.Variable(tz.randint(0, 100, [1, 8], tz.dtype.int64), False)
        y = model(input_ids)

    def test_gpt2_instantiation(self):
        config = tz.models.GPT2Config()
        config.vocab_size = 100
        config.n_embd = 64
        config.n_head = 2
        config.n_layer = 1
        model = tz.models.GPT2Model(config)
        input_ids = tz.Variable(tz.randint(0, 100, [1, 8], tz.dtype.int64), False)
        y = model(input_ids)


# ============================================================================
# Vision models on multiple backends
# ============================================================================

@pytest.mark.parametrize("device", ALL_DEVICES, indirect=True)
def test_resnet18_multibackend(device):
    # MIOpen's pooling kernel JIT-compiles a HIP source through HIPRTC on
    # first use. On some ROCm installs the HIPRTC compiler search path
    # does not include hip/hip_runtime.h even though the header is
    # present on disk, and the compile fails with HIPRTC_ERROR_COMPILATION
    # before Tenzor's code even runs — see
    # https://github.com/ROCm/MIOpen/issues where similar JIT include
    # path issues are reported. Probe once by running a tiny pool op
    # and skip the whole test if it throws.
    if device == "rocm":
        try:
            _pool = tz.nn.MaxPool2d(kernel_size=2)
            _probe = _pool(tz.Variable(
                tz.randn([1, 1, 2, 2], device="rocm"), False))
            del _probe, _pool
        except Exception as e:
            if "MIOpen" in str(e) or "HIPRTC" in str(e):
                pytest.skip(f"MIOpen HIPRTC JIT unavailable on this host: {e}")
            raise
    model = tz.models.resnet18(num_classes=10)
    x = make_var([1, 3, 32, 32], device)
    y = model(x)
    assert y.data.shape[0] == 1
    assert y.data.shape[1] == 10
