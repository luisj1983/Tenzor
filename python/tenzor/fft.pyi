"""Type stubs for tenzor.fft submodule.

R.29: enumerates the FFT routines exposed by ``python/bindings/bindings_fft.cpp``.
``check_pyi_drift.py`` keeps this list in sync with the runtime surface.
"""

from __future__ import annotations
from typing import List, Optional

from tenzor import Tensor


# ---------------------------------------------------------------------------
# Complex-to-complex N-D FFTs
# ---------------------------------------------------------------------------
def fft(input: Tensor, n: Optional[int] = None, dim: int = -1,
        norm: str = "backward") -> Tensor: ...
def ifft(input: Tensor, n: Optional[int] = None, dim: int = -1,
         norm: str = "backward") -> Tensor: ...
def fft2(input: Tensor, s: Optional[List[int]] = None,
         dim: List[int] = [-2, -1],
         norm: str = "backward") -> Tensor: ...
def ifft2(input: Tensor, s: Optional[List[int]] = None,
          dim: List[int] = [-2, -1],
          norm: str = "backward") -> Tensor: ...
def fftn(input: Tensor, s: Optional[List[int]] = None,
         dim: Optional[List[int]] = None,
         norm: str = "backward") -> Tensor: ...
def ifftn(input: Tensor, s: Optional[List[int]] = None,
          dim: Optional[List[int]] = None,
          norm: str = "backward") -> Tensor: ...


# ---------------------------------------------------------------------------
# Real <-> complex FFTs
# ---------------------------------------------------------------------------
def rfft(input: Tensor, n: Optional[int] = None, dim: int = -1,
         norm: str = "backward") -> Tensor: ...
def irfft(input: Tensor, n: Optional[int] = None, dim: int = -1,
          norm: str = "backward") -> Tensor: ...
def rfft2(input: Tensor, s: Optional[List[int]] = None,
          dim: List[int] = [-2, -1],
          norm: str = "backward") -> Tensor: ...
def irfft2(input: Tensor, s: Optional[List[int]] = None,
           dim: List[int] = [-2, -1],
           norm: str = "backward") -> Tensor: ...
def rfftn(input: Tensor, s: Optional[List[int]] = None,
          dim: Optional[List[int]] = None,
          norm: str = "backward") -> Tensor: ...
def irfftn(input: Tensor, s: Optional[List[int]] = None,
           dim: Optional[List[int]] = None,
           norm: str = "backward") -> Tensor: ...


# ---------------------------------------------------------------------------
# Hermitian FFTs (real output from Hermitian-symmetric complex input)
# ---------------------------------------------------------------------------
def hfft(input: Tensor, n: Optional[int] = None, dim: int = -1,
         norm: str = "backward") -> Tensor: ...
def ihfft(input: Tensor, n: Optional[int] = None, dim: int = -1,
          norm: str = "backward") -> Tensor: ...


# ---------------------------------------------------------------------------
# Frequency-domain shifts
# ---------------------------------------------------------------------------
def fftshift(input: Tensor, dims: List[int] = []) -> Tensor: ...
def ifftshift(input: Tensor, dims: List[int] = []) -> Tensor: ...


# ---------------------------------------------------------------------------
# Cosine transforms + audio feature extraction
# ---------------------------------------------------------------------------
def dct(input: Tensor, type: int = 2, n: Optional[int] = None,
        dim: int = -1, norm: str = "backward") -> Tensor: ...
def idct(input: Tensor, type: int = 2, n: Optional[int] = None,
         dim: int = -1, norm: str = "backward") -> Tensor: ...
def mel_scale(spectrogram: Tensor, n_mels: int = 128,
              f_min: float = 0.0, f_max: float = 0.0,
              sample_rate: int = 16000) -> Tensor: ...
def mfcc(waveform: Tensor, sample_rate: int = 16000, n_mfcc: int = 40,
         n_mels: int = 128, n_fft: int = 400, hop_length: int = 160,
         f_min: float = 0.0, f_max: float = 0.0) -> Tensor: ...
