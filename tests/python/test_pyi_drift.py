"""CTest gate: assert ``tools/check_pyi_parity.py`` exits with code 0.

S23 sweep result: the parity checker enumerates every public name on the
``tenzor`` runtime modules (tenzor, tenzor.optim, tenzor.nn, …) and
diffs that against the declarations in each .pyi stub. Any drift —
either a runtime name absent from the stub or a stub-only name with no
runtime counterpart — fails the test.

This complements the existing ``test_pyi_data_drift.py`` (which checks
the much narrower DataLoader ``__init__`` parameter-order invariant).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKER = REPO_ROOT / "tools" / "check_pyi_parity.py"


@pytest.fixture(scope="module")
def parity_result():
    """Run the parity checker once per module and reuse the result."""
    assert CHECKER.exists(), f"missing drift checker: {CHECKER}"
    proc = subprocess.run(
        [sys.executable, str(CHECKER)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return proc


def test_pyi_parity_no_drift(parity_result):
    """The .pyi stubs match the runtime tenzor module surface (S23)."""
    if parity_result.returncode != 0:
        # Surface the full report so CI logs show which stub drifted.
        message = (
            f"check_pyi_parity returned {parity_result.returncode}\n\n"
            f"--- STDOUT ---\n{parity_result.stdout}\n"
            f"--- STDERR ---\n{parity_result.stderr}\n"
        )
        pytest.fail(message)


def test_pyi_parity_checker_is_a_real_check(parity_result):
    """Sanity guard: the checker must produce *some* output.

    A silent ``return 0`` could mask the checker becoming a no-op (e.g.
    if the import path broke and no .pyi files were discovered).
    """
    combined = parity_result.stdout + parity_result.stderr
    assert "check_pyi_parity" in combined, (
        "Drift checker produced no recognizable output; the script may "
        "be silently failing. Output was:\n"
        + combined
    )


if __name__ == "__main__":
    # Allow ``python tests/python/test_pyi_drift.py`` for parity with
    # other python tests in the suite.
    sys.exit(pytest.main([__file__, "-v"]))
