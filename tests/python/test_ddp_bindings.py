"""Smoke test for DDP Python API bindings (no actual multi-process)."""
import sys
sys.path.insert(0, "python")
import tenzor as tz

tz.initialize()

def test_distributed_module_exists():
    """distributed submodule should be accessible."""
    assert hasattr(tz, "distributed"), "tenzor.distributed not found"

def test_reduce_op_enum():
    """ReduceOp enum values should be available."""
    d = tz.distributed
    assert hasattr(d, "ReduceOp")
    assert hasattr(d.ReduceOp, "SUM")
    assert hasattr(d.ReduceOp, "PRODUCT")
    assert hasattr(d.ReduceOp, "MIN")
    assert hasattr(d.ReduceOp, "MAX")
    assert hasattr(d.ReduceOp, "AVG")

def test_functions_exist():
    """Key distributed functions should be callable."""
    d = tz.distributed
    assert callable(getattr(d, "init_process_group", None))
    assert callable(getattr(d, "destroy_process_group", None))
    assert callable(getattr(d, "get_rank", None))
    assert callable(getattr(d, "get_world_size", None))
    assert callable(getattr(d, "is_initialized", None))
    assert callable(getattr(d, "barrier", None))

def test_not_initialized():
    """Before init, is_initialized should return False."""
    assert not tz.distributed.is_initialized()

def test_ddp_class_exists():
    """DistributedDataParallel class should be accessible."""
    assert hasattr(tz.distributed, "DistributedDataParallel")

if __name__ == "__main__":
    test_distributed_module_exists()
    print("  distributed module exists")
    test_reduce_op_enum()
    print("  ReduceOp enum values ok")
    test_functions_exist()
    print("  distributed functions exist")
    test_not_initialized()
    print("  not initialized by default")
    test_ddp_class_exists()
    print("  DDP class exists")
    print("All DDP binding tests passed!")
