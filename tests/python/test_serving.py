"""Tests for tenzor.serving Python bindings.

Covers construction and method-surface checks for ServerConfig, BatchConfig,
ModelRepository, and InferenceServer. Keeps the tests local — no real HTTP
listener is stood up — since spinning an actual server would require a port
and a real model repository directory and is better suited to an integration
test. What we verify here is that the bindings are accessible, constructors
accept documented kwargs, and each class exposes the expected methods.
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python'))
import tenzor as tz

tz.initialize()


# ---------------------------------------------------------------------------
# ServerConfig
# ---------------------------------------------------------------------------

class TestServerConfig:
    def test_default_construction(self):
        cfg = tz.serving.ServerConfig()
        # Every documented attribute should be readable.
        assert hasattr(cfg, "grpc_port")
        assert hasattr(cfg, "http_port")
        assert hasattr(cfg, "model_repository_path")
        assert hasattr(cfg, "num_workers")

    def test_attribute_roundtrip(self):
        cfg = tz.serving.ServerConfig()
        cfg.grpc_port = 50051
        cfg.http_port = 8000
        cfg.num_workers = 4
        # CC.17: per-process tempdir prefix so parallel pytest runs don't collide.
        models_path = tempfile.mkdtemp(prefix=f"tz_models_{os.getpid()}_")
        cfg.model_repository_path = models_path
        assert cfg.grpc_port == 50051
        assert cfg.http_port == 8000
        assert cfg.num_workers == 4
        assert cfg.model_repository_path == models_path


# ---------------------------------------------------------------------------
# BatchConfig
# ---------------------------------------------------------------------------

class TestBatchConfig:
    def test_default_construction(self):
        bc = tz.serving.BatchConfig()
        assert hasattr(bc, "max_batch_size")
        assert hasattr(bc, "max_latency_us")

    def test_attribute_roundtrip(self):
        bc = tz.serving.BatchConfig()
        bc.max_batch_size = 32
        bc.max_latency_us = 10_000
        assert bc.max_batch_size == 32
        assert bc.max_latency_us == 10_000


# ---------------------------------------------------------------------------
# ModelRepository — accessed via InferenceServer.repository. The class has no
# direct Python constructor; it's only obtainable from a live server.
# ---------------------------------------------------------------------------

def _make_server():
    cfg = tz.serving.ServerConfig()
    cfg.grpc_port = 0
    cfg.http_port = 0
    cfg.num_workers = 1
    return tz.serving.InferenceServer(cfg)


class TestModelRepository:
    def test_methods_exist(self):
        srv = _make_server()
        repo = srv.repository()
        assert callable(repo.list_models)
        assert callable(repo.load_model)
        assert callable(repo.unload_model)

    def test_empty_listing(self):
        srv = _make_server()
        repo = srv.repository()
        models = repo.list_models()
        assert isinstance(models, list)
        assert models == []

    def test_load_missing_model_raises(self):
        srv = _make_server()
        repo = srv.repository()
        # Loading a non-existent file should raise, not silently succeed.
        # OO.20: build a pid-suffixed path under the system temp dir so the
        # test cannot collide with a real file at a hard-coded /tmp/ name.
        missing_path = os.path.join(
            tempfile.gettempdir(),
            f"nonexistent_tenzor_model_{os.getpid()}.pt",
        )
        # Defensive: if a previous test crashed mid-write, remove the leftover.
        if os.path.exists(missing_path):
            os.remove(missing_path)
        with pytest.raises(Exception):
            repo.load_model(
                "nonexistent_model",
                missing_path,
                tz.Device("cpu"),
            )


# ---------------------------------------------------------------------------
# InferenceServer
# ---------------------------------------------------------------------------

class TestInferenceServer:
    def test_construction_with_config(self):
        # We do NOT call start() — that would bind a real port. We only
        # confirm the constructor accepts a ServerConfig and that
        # start/stop/wait are bound.
        cfg = tz.serving.ServerConfig()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.model_repository_path = tmp
            cfg.grpc_port = 0  # Request an ephemeral port if start() were called.
            cfg.http_port = 0
            cfg.num_workers = 1
            server = tz.serving.InferenceServer(cfg)
            assert callable(server.start)
            assert callable(server.stop)
            assert callable(server.wait)
            # The server owns a repository accessor.
            assert server.repository() is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
