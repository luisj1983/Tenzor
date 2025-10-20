Tenzor Documentation
====================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   tutorials/index
   api/index
   cpp_api/index
   developer/index
   changelog

Welcome to Tenzor
-----------------

Tenzor is a high-performance tensor computation library with automatic differentiation,
written in modern C++23 with Python bindings. It provides a PyTorch-like API with
multi-backend support (CPU, CUDA, ROCm, OneAPI).

Key Features
------------

* **Modern C++23 Implementation**: Leveraging the latest C++ features for performance and safety
* **Multi-Backend Support**: CPU (AVX2/AVX-512), CUDA (Tensor Cores), ROCm (WMMA), OneAPI (SYCL)
* **Automatic Differentiation**: Complete autograd engine with computational graph
* **Neural Network Layers**: Extensive collection of layers, activations, and loss functions
* **PyTorch Interoperability**: Zero-copy tensor conversion when possible
* **ONNX Export**: Export models to ONNX format for deployment
* **Mixed Precision Training**: FP16/BF16 support with automatic casting
* **Quantization**: INT8/UINT8 post-training and quantization-aware training
* **Graph Optimization**: Automatic kernel fusion and graph-level optimizations
* **Python API**: Full-featured Python bindings with type hints

Quick Example
-------------

Python API:

.. code-block:: python

   import tenzor as tz

   # Create tensors
   x = tz.randn(128, 784)
   y = tz.randint(0, 10, (128,))

   # Define model
   model = tz.nn.Sequential(
       tz.nn.Linear(784, 256),
       tz.nn.ReLU(),
       tz.nn.Linear(256, 10)
   )

   # Training
   optimizer = tz.optim.Adam(model.parameters(), lr=0.001)
   criterion = tz.nn.CrossEntropyLoss()

   for epoch in range(100):
       pred = model(x)
       loss = criterion(pred, y)

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

C++ API:

.. code-block:: cpp

   #include <tenzor/tenzor.hpp>

   using namespace tenzor;

   int main() {
       // Create tensors
       auto x = randn({128, 784});
       auto y = randint(0, 10, {128});

       // Define model
       auto model = nn::Sequential(
           nn::Linear(784, 256),
           nn::ReLU(),
           nn::Linear(256, 10)
       );

       // Training
       auto optimizer = optim::Adam(model.parameters(), 0.001);
       auto criterion = nn::CrossEntropyLoss();

       for (int epoch = 0; epoch < 100; ++epoch) {
           auto pred = model.forward(x);
           auto loss = criterion.forward(pred, y);

           optimizer.zero_grad();
           loss.backward();
           optimizer.step();
       }

       return 0;
   }

Performance
-----------

Tenzor achieves competitive performance with major frameworks:

* **CUDA Backend**: Optimized kernels with Tensor Core support
* **CPU Backend**: AVX2/AVX-512 vectorization with oneDNN integration
* **Memory Management**: Custom caching allocator for efficient GPU memory usage
* **Graph Optimization**: Automatic fusion of common operation patterns

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
