Quick Start Guide
=================

This guide will help you get started with Tenzor quickly.

Basic Tensor Operations
-----------------------

Creating Tensors
^^^^^^^^^^^^^^^^

.. code-block:: python

   import tenzor as tz

   # Create tensors from Python lists
   x = tz.tensor([1, 2, 3, 4])
   y = tz.tensor([[1, 2], [3, 4]])

   # Create tensors with specific shapes
   zeros = tz.zeros(3, 4)        # 3x4 tensor of zeros
   ones = tz.ones(2, 3)          # 2x3 tensor of ones
   random = tz.randn(5, 5)       # 5x5 tensor with random values

   # Create tensors on GPU
   x_gpu = tz.randn(10, 10, device=tz.Device.cuda())

Tensor Attributes
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import tenzor as tz

   x = tz.randn(3, 4, 5)

   print(x.shape)        # (3, 4, 5)
   print(x.dtype)        # DType.Float32
   print(x.device)       # Device(type=CPU, index=0)
   print(x.ndim)         # 3
   print(x.numel())      # 60

Tensor Operations
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import tenzor as tz

   x = tz.randn(3, 4)
   y = tz.randn(3, 4)

   # Element-wise operations
   z = x + y
   z = x * y
   z = x / y
   z = x ** 2

   # Matrix operations
   a = tz.randn(3, 4)
   b = tz.randn(4, 5)
   c = a @ b  # Matrix multiplication (3x5 result)

   # Reduction operations
   sum_all = x.sum()
   mean_dim0 = x.mean(dim=0)
   max_val, max_idx = x.max(dim=1)

Building Neural Networks
-------------------------

Simple MLP
^^^^^^^^^^

.. code-block:: python

   import tenzor as tz
   import tenzor.nn as nn

   class MLP(nn.Module):
       def __init__(self, input_size, hidden_size, output_size):
           super().__init__()
           self.fc1 = nn.Linear(input_size, hidden_size)
           self.relu = nn.ReLU()
           self.fc2 = nn.Linear(hidden_size, output_size)

       def forward(self, x):
           x = self.fc1(x)
           x = self.relu(x)
           x = self.fc2(x)
           return x

   # Create model
   model = MLP(784, 256, 10)

   # Forward pass
   x = tz.randn(32, 784)
   output = model(x)  # Shape: (32, 10)

Convolutional Neural Network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import tenzor as tz
   import tenzor.nn as nn

   class CNN(nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
           self.bn1 = nn.BatchNorm2d(64)
           self.relu = nn.ReLU()
           self.pool = nn.MaxPool2d(2, 2)

           self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
           self.bn2 = nn.BatchNorm2d(128)

           self.fc = nn.Linear(128 * 8 * 8, num_classes)

       def forward(self, x):
           x = self.pool(self.relu(self.bn1(self.conv1(x))))
           x = self.pool(self.relu(self.bn2(self.conv2(x))))
           x = x.flatten(1)
           x = self.fc(x)
           return x

   # Create model and move to GPU
   model = CNN(num_classes=10)
   model = model.cuda()

   # Forward pass
   x = tz.randn(4, 3, 32, 32, device=tz.Device.cuda())
   output = model(x)  # Shape: (4, 10)

Training a Model
----------------

Complete Training Loop
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import tenzor as tz
   import tenzor.nn as nn
   import tenzor.optim as optim

   # Create model, loss, and optimizer
   model = MLP(784, 256, 10)
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)

   # Training loop
   num_epochs = 10
   for epoch in range(num_epochs):
       # Training phase
       model.train()
       for batch_idx, (data, target) in enumerate(train_loader):
           # Forward pass
           output = model(data)
           loss = criterion(output, target)

           # Backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           if batch_idx % 100 == 0:
               print(f'Epoch [{epoch+1}/{num_epochs}], '
                     f'Step [{batch_idx}/{len(train_loader)}], '
                     f'Loss: {loss.item():.4f}')

       # Validation phase
       model.eval()
       correct = 0
       total = 0
       with tz.no_grad():
           for data, target in val_loader:
               output = model(data)
               pred = output.argmax(dim=1)
               correct += (pred == target).sum().item()
               total += target.size(0)

       accuracy = 100 * correct / total
       print(f'Validation Accuracy: {accuracy:.2f}%')

Using Learning Rate Schedulers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import tenzor as tz
   import tenzor.optim as optim

   # Create optimizer and scheduler
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   scheduler = optim.StepLR(optimizer, step_size=5, gamma=0.1)

   # Training loop
   for epoch in range(num_epochs):
       train(model, train_loader, optimizer, criterion)
       validate(model, val_loader)

       # Update learning rate
       scheduler.step()

       print(f'Current LR: {scheduler.get_last_lr()[0]}')

Mixed Precision Training
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import tenzor as tz
   import tenzor.nn as nn

   # Enable automatic mixed precision
   model = model.cuda()

   with tz.autocast(device_type='cuda', dtype=tz.DType.Float16):
       output = model(input)
       loss = criterion(output, target)

   loss.backward()
   optimizer.step()

GPU Usage
---------

Moving Tensors to GPU
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import tenzor as tz

   # Create tensor on CPU
   x = tz.randn(10, 10)

   # Move to GPU
   x_gpu = x.cuda()
   # or
   x_gpu = x.to(tz.Device.cuda())

   # Move back to CPU
   x_cpu = x_gpu.cpu()

Multi-GPU Training
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import tenzor as tz
   import tenzor.nn as nn

   # Wrap model for data parallelism
   model = MLP(784, 256, 10)
   if tz.cuda.device_count() > 1:
       model = nn.DataParallel(model)
   model = model.cuda()

   # Training proceeds as usual
   for data, target in train_loader:
       data = data.cuda()
       target = target.cuda()

       output = model(data)
       loss = criterion(output, target)

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

Model Saving and Loading
-------------------------

Save Model
^^^^^^^^^^

.. code-block:: python

   import tenzor as tz

   # Save model state dict
   tz.save(model.state_dict(), 'model.pth')

   # Save entire checkpoint
   checkpoint = {
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'loss': loss,
   }
   tz.save(checkpoint, 'checkpoint.pth')

Load Model
^^^^^^^^^^

.. code-block:: python

   import tenzor as tz

   # Load model state dict
   model = MLP(784, 256, 10)
   model.load_state_dict(tz.load('model.pth'))
   model.eval()

   # Load checkpoint
   checkpoint = tz.load('checkpoint.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   epoch = checkpoint['epoch']
   loss = checkpoint['loss']

PyTorch Interoperability
-------------------------

Convert Between Frameworks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import tenzor as tz
   import torch

   # Tenzor to PyTorch (zero-copy when possible)
   tz_tensor = tz.randn(10, 10, device=tz.Device.cuda())
   torch_tensor = tz.torch_interop.tensor_to_torch(tz_tensor)

   # PyTorch to Tenzor (zero-copy when possible)
   torch_tensor = torch.randn(10, 10, device='cuda')
   tz_tensor = tz.torch_interop.tensor_from_torch(torch_tensor)

   # Check if zero-copy is possible
   can_zero_copy = tz.torch_interop.can_zero_copy_to_torch(tz_tensor)

ONNX Export
-----------

Export Model to ONNX
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import tenzor as tz
   from tenzor.onnx import ONNXExporter

   # Create and train model
   model = MLP(784, 256, 10)
   # ... train model ...

   # Export to ONNX
   dummy_input = tz.randn(1, 784)
   exporter = ONNXExporter(model)
   exporter.export(dummy_input, 'model.onnx')

   # Export with specific opset
   exporter.export(dummy_input, 'model.onnx', opset_version=14)

Next Steps
----------

* Read the full :doc:`api/index` for detailed documentation
* Check out :doc:`tutorials/index` for in-depth guides
* Explore the `examples directory <https://github.com/your-org/tenzor/tree/main/examples>`_ for more code samples
* Join our `community forum <https://discuss.tenzor.org>`_ for help and discussions
