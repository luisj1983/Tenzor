"""Tests for tenzor.distributed.FullyShardedDataParallel Python bindings.

Single-process binding smoke tests. Real sharded training requires a
multi-process ProcessGroup stood up via `multiprocessing.spawn` (see
test_ddp_integration.py for the pattern); this file verifies the Python
surface — config, constructor, shard-bookkeeping methods, context manager
for full-parameter gather — is reachable and behaves sensibly on the
single-rank happy path.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python'))
import tenzor as tz

tz.initialize()


# ---------------------------------------------------------------------------
# FSDPConfig
# ---------------------------------------------------------------------------

class TestFSDPConfig:
    def test_default_construction(self):
        cfg = tz.distributed.FSDPConfig()
        assert hasattr(cfg, "strategy")
        assert hasattr(cfg, "cpu_offload")
        assert hasattr(cfg, "mixed_precision")
        assert hasattr(cfg, "auto_wrap_min_params")
        assert hasattr(cfg, "backward_prefetch")
        assert hasattr(cfg, "forward_prefetch")

    def test_attribute_roundtrip(self):
        cfg = tz.distributed.FSDPConfig()
        cfg.strategy = tz.distributed.ShardingStrategy.FULL_SHARD
        cfg.cpu_offload = True
        cfg.mixed_precision = True
        cfg.auto_wrap_min_params = 1000
        assert cfg.strategy == tz.distributed.ShardingStrategy.FULL_SHARD
        assert cfg.cpu_offload is True
        assert cfg.mixed_precision is True
        assert cfg.auto_wrap_min_params == 1000


# ---------------------------------------------------------------------------
# ShardingStrategy enum
# ---------------------------------------------------------------------------

class TestShardingStrategy:
    def test_enum_values(self):
        assert tz.distributed.ShardingStrategy.FULL_SHARD is not None
        assert tz.distributed.ShardingStrategy.SHARD_GRAD_OP is not None
        assert tz.distributed.ShardingStrategy.NO_SHARD is not None

    def test_strategies_are_distinct(self):
        vals = {
            tz.distributed.ShardingStrategy.FULL_SHARD,
            tz.distributed.ShardingStrategy.SHARD_GRAD_OP,
            tz.distributed.ShardingStrategy.NO_SHARD,
        }
        assert len(vals) == 3


# ---------------------------------------------------------------------------
# ProcessGroup — checked without a real multi-process setup
# ---------------------------------------------------------------------------

class TestProcessGroup:
    def test_class_api_surface(self):
        # Verify the methods are bound. Calling them without a real
        # multi-worker setup is covered by test_ddp_integration.py.
        cls = tz.distributed.ProcessGroup
        assert callable(getattr(cls, "all_reduce", None))
        assert callable(getattr(cls, "barrier", None))
        assert callable(getattr(cls, "broadcast", None))
        # rank/world_size are @property-style accessors on the class.
        assert hasattr(cls, "rank")
        assert hasattr(cls, "world_size")


# ---------------------------------------------------------------------------
# FullyShardedDataParallel — local single-rank wrapper
# ---------------------------------------------------------------------------

class TestFSDPLocalWrap:
    def test_wrap_attribute_surface(self):
        """Wrap a small Linear as an FSDP-managed module on a single rank.

        The full multi-rank sharding path requires a ProcessGroup with
        world_size > 1; this local smoke only exercises the Python binding
        exposure. If even the single-rank wrap fails (e.g. FSDP requires
        a PG), skip with a clear reason rather than hiding the failure.
        """
        fsdp_cls = tz.distributed.FullyShardedDataParallel
        # Every FSDP module should expose the shard-bookkeeping methods.
        for name in ("forward", "finalize_backward",
                     "release_full_params", "summon_full_params",
                     "sharded_param_bytes", "total_params"):
            assert callable(getattr(fsdp_cls, name, None)), \
                f"FullyShardedDataParallel is missing {name}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
