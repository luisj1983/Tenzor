"""Tenzor public API surface sanity test (audit item H.4).

Every name listed in ``tenzor.__all__`` must resolve at import time
(``getattr(tenzor, name)`` must not raise AttributeError) and
``from tenzor import *`` must succeed without dangling names.

This test catches the documented hazard: an __all__ entry that
references a C++-only symbol that wasn't built (e.g. behind a CMake
option) or a renamed/removed top-level symbol, which would otherwise
only surface as a runtime error inside user code.
"""

import sys
import os
import pytest

# Match the build-dir import path used by the rest of the python tests.
_build_python = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python')
if os.path.isdir(_build_python):
    sys.path.insert(0, os.path.abspath(_build_python))


def test_all_names_resolve():
    """Every name in tenzor.__all__ must resolve to a real attribute."""
    import tenzor as tz
    assert hasattr(tz, '__all__'), "tenzor module must expose __all__"
    missing = [name for name in tz.__all__ if not hasattr(tz, name)]
    assert not missing, (
        "names listed in tenzor.__all__ that fail to resolve at runtime: "
        f"{missing}.  Either remove from __all__ or import/re-export them "
        "in python/tenzor/__init__.py."
    )


def test_wildcard_import_succeeds():
    """`from tenzor import *` must not raise (no dangling names)."""
    # Use exec inside a fresh module namespace to keep our test
    # globals clean.
    import importlib
    mod = importlib.import_module('tenzor')
    # Build a fresh namespace and execute the wildcard import.
    ns: dict = {}
    exec('from tenzor import *', ns)
    # Quick sanity — common names should be present.
    for canonical in ('Tensor', 'Variable', 'zeros', 'matmul'):
        assert canonical in ns, f"{canonical} missing after wildcard import"


def test_all_is_sorted_dedup_optional():
    """__all__ should not contain duplicates; sorting is not enforced."""
    import tenzor as tz
    seen = set()
    dups = []
    for name in tz.__all__:
        if name in seen:
            dups.append(name)
        seen.add(name)
    assert not dups, f"__all__ contains duplicate entries: {dups}"
