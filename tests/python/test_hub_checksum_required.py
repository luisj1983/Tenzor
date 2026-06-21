"""Regression tests for ModelHub download integrity gates.

download_weights fetches attacker-controllable bytes and feeds them to a
deserializer (load_pretrained_weights). With checksum verification enabled (the
default), the hub must refuse to download (1) over plaintext HTTP and (2) with
no expected SHA256 — otherwise a tampered/MITM'd blob reaches the parser
unverified. These tests pin those refusals; both fire BEFORE any network I/O,
so no live URL is needed.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz
import tenzor.tenzor_core as _core
tz.initialize()

import pytest

_Hub = _core.models.Hub


def test_non_https_url_rejected_when_verifying():
    """A plaintext HTTP URL must be refused while verify_checksums is on."""
    with pytest.raises((ValueError, RuntimeError)) as ei:
        _Hub.download_weights("m", "http://example.com/weights.bin")
    assert "HTTPS" in str(ei.value) or "https" in str(ei.value)


def test_missing_checksum_rejected_when_verifying():
    """An https download with NO expected SHA256 must be refused (the
    supply-chain hole): the bytes are deserialized, so an unverifiable blob is
    a hard error, not a warning."""
    with pytest.raises((ValueError, RuntimeError)) as ei:
        _Hub.download_weights("m", "https://example.com/weights.bin")
    assert "SHA256" in str(ei.value) or "checksum" in str(ei.value).lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
