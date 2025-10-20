Installation
============

Requirements
------------

**System Requirements:**

* C++23 compatible compiler (GCC 12+, Clang 15+, MSVC 2022+)
* CMake 3.22 or later
* Python 3.8+ (for Python bindings)

**Optional Dependencies:**

* CUDA Toolkit 11.8+ (for CUDA backend)
* ROCm 5.4+ (for AMD GPU support)
* Intel oneAPI 2023+ (for Intel GPU support)
* PyTorch 2.0+ (for PyTorch interoperability)

Install from PyPI
-----------------

The easiest way to install Tenzor is via pip:

.. code-block:: bash

   pip install tenzor

With CUDA support:

.. code-block:: bash

   pip install tenzor[cuda]

With ROCm support:

.. code-block:: bash

   pip install tenzor[rocm]

With all optional features:

.. code-block:: bash

   pip install tenzor[all]

Build from Source
-----------------

Basic Build (CPU only)
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/your-org/tenzor.git
   cd tenzor
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   sudo make install

Build with CUDA Support
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/your-org/tenzor.git
   cd tenzor
   mkdir build && cd build
   cmake -DTENZOR_BUILD_CUDA=ON ..
   make -j$(nproc)
   sudo make install

Build with ROCm Support
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/your-org/tenzor.git
   cd tenzor
   mkdir build && cd build
   cmake -DTENZOR_BUILD_ROCM=ON ..
   make -j$(nproc)
   sudo make install

Build with OneAPI Support
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/your-org/tenzor.git
   cd tenzor
   mkdir build && cd build
   cmake -DTENZOR_BUILD_ONEAPI=ON ..
   make -j$(nproc)
   sudo make install

Build Python Package
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/your-org/tenzor.git
   cd tenzor
   pip install -e .

Or with specific backend:

.. code-block:: bash

   TENZOR_BUILD_CUDA=1 pip install -e .
   TENZOR_BUILD_ROCM=1 pip install -e .

CMake Build Options
-------------------

General Options
^^^^^^^^^^^^^^^

* ``TENZOR_BUILD_TESTS`` - Build unit tests (default: ON)
* ``TENZOR_BUILD_EXAMPLES`` - Build examples (default: ON)
* ``TENZOR_BUILD_BENCHMARKS`` - Build benchmarks (default: ON)
* ``TENZOR_BUILD_PYTHON`` - Build Python bindings (default: ON)
* ``TENZOR_BUILD_DOCS`` - Build documentation (default: OFF)

Backend Options
^^^^^^^^^^^^^^^

* ``TENZOR_BUILD_CUDA`` - Enable CUDA backend (default: OFF)
* ``TENZOR_BUILD_ROCM`` - Enable ROCm backend (default: OFF)
* ``TENZOR_BUILD_ONEAPI`` - Enable OneAPI backend (default: OFF)

Feature Options
^^^^^^^^^^^^^^^

* ``TENZOR_HAS_TORCH`` - Enable PyTorch interoperability (default: OFF)
* ``TENZOR_ENABLE_ONNX`` - Enable ONNX export (default: ON)
* ``TENZOR_ENABLE_QUANTIZATION`` - Enable quantization (default: ON)
* ``TENZOR_ENABLE_MIXED_PRECISION`` - Enable mixed precision (default: ON)

Optimization Options
^^^^^^^^^^^^^^^^^^^^

* ``TENZOR_ENABLE_AVX2`` - Enable AVX2 vectorization (default: ON)
* ``TENZOR_ENABLE_AVX512`` - Enable AVX-512 vectorization (default: OFF)
* ``TENZOR_ENABLE_FUSION`` - Enable kernel fusion (default: ON)

Example CMake Command
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cmake -DCMAKE_BUILD_TYPE=Release \
         -DTENZOR_BUILD_CUDA=ON \
         -DTENZOR_BUILD_PYTHON=ON \
         -DTENZOR_HAS_TORCH=ON \
         -DTENZOR_ENABLE_AVX2=ON \
         -DTENZOR_ENABLE_FUSION=ON \
         ..

Docker Installation
-------------------

CPU-only Container
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   docker pull tenzor/tenzor:latest-cpu
   docker run -it tenzor/tenzor:latest-cpu

CUDA Container
^^^^^^^^^^^^^^

.. code-block:: bash

   docker pull tenzor/tenzor:latest-cuda
   docker run --gpus all -it tenzor/tenzor:latest-cuda

ROCm Container
^^^^^^^^^^^^^^

.. code-block:: bash

   docker pull tenzor/tenzor:latest-rocm
   docker run --device=/dev/kfd --device=/dev/dri -it tenzor/tenzor:latest-rocm

Verify Installation
-------------------

Python
^^^^^^

.. code-block:: python

   import tenzor as tz
   print(tz.__version__)

   # Check CUDA availability
   if tz.cuda.is_available():
       print(f"CUDA devices: {tz.cuda.device_count()}")

C++
^^^

.. code-block:: cpp

   #include <tenzor/tenzor.hpp>
   #include <iostream>

   int main() {
       std::cout << "Tenzor version: " << tenzor::version() << std::endl;

       // Check CUDA availability
       if (tenzor::cuda::is_available()) {
           std::cout << "CUDA devices: " << tenzor::cuda::device_count() << std::endl;
       }

       return 0;
   }

Troubleshooting
---------------

CUDA Not Found
^^^^^^^^^^^^^^

If CMake cannot find CUDA:

.. code-block:: bash

   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

ROCm Not Found
^^^^^^^^^^^^^^

If CMake cannot find ROCm:

.. code-block:: bash

   export ROCM_PATH=/opt/rocm
   export PATH=$ROCM_PATH/bin:$PATH
   export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

Compiler Errors
^^^^^^^^^^^^^^^

Make sure you have a C++23 compatible compiler:

.. code-block:: bash

   # Check GCC version (need 12+)
   g++ --version

   # Check Clang version (need 15+)
   clang++ --version

Python Import Errors
^^^^^^^^^^^^^^^^^^^^

If you get import errors, make sure the library is in your Python path:

.. code-block:: bash

   export PYTHONPATH=/path/to/tenzor/python:$PYTHONPATH

Or install the package:

.. code-block:: bash

   pip install -e /path/to/tenzor
