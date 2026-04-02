FFT Operations
==============

Fast Fourier Transform operations.

1D Transforms
-------------

- ``tz.fft.fft(input)`` -- Complex-to-complex 1D FFT
- ``tz.fft.ifft(input)`` -- Inverse 1D FFT
- ``tz.fft.rfft(input)`` -- Real-to-complex 1D FFT
- ``tz.fft.irfft(input)`` -- Complex-to-real inverse 1D FFT

2D Transforms
-------------

- ``tz.fft.fft2(input)`` -- 2D FFT
- ``tz.fft.ifft2(input)`` -- Inverse 2D FFT

N-D Transforms
--------------

- ``tz.fft.fftn(input)`` -- N-dimensional FFT
- ``tz.fft.ifftn(input)`` -- Inverse N-dimensional FFT

Example
-------

.. code-block:: python

   import tenzor as tz

   # 1D FFT roundtrip
   x = tz.randn([64])
   X = tz.fft.fft(x)         # Frequency domain
   x_back = tz.fft.ifft(X)   # Back to time domain

   # Real-valued signal
   signal = tz.randn([128])
   spectrum = tz.fft.rfft(signal)  # Only positive frequencies
