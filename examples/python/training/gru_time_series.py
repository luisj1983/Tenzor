"""
GRU Time Series Forecasting with Advanced Training Techniques

This comprehensive example demonstrates:
- GRU and GRUCell layers for sequence modeling
- LayerNorm for stable training
- Multiple optimizers: RMSprop, Adagrad, Adadelta
- Learning rate schedulers: StepLR, CosineAnnealingLR, ExponentialLR
- Loss functions: MSELoss, SmoothL1Loss, L1Loss
- Dropout for regularization
- Real training loop with validation
"""

import tenzor as tz
import numpy as np
import math


# ============================================================================
# Time Series Dataset Generation
# ============================================================================

class TimeSeriesDataset:
    """Generate synthetic time series data (sine wave + noise)"""

    def __init__(self, num_samples, seq_length, prediction_horizon, noise_level=0.1):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon

        np.random.seed(42)

        self.inputs = []
        self.targets = []

        for i in range(num_samples):
            phase = i * 0.1

            # Generate sine wave with multiple frequencies
            t = np.arange(seq_length + prediction_horizon)
            values = (np.sin(0.1 * t + phase) +
                     0.5 * np.sin(0.05 * t + phase * 0.5) +
                     np.random.normal(0, noise_level, len(t)))

            self.inputs.append(values[:seq_length].astype(np.float32))
            self.targets.append(values[seq_length:seq_length + prediction_horizon].astype(np.float32))

    def get_batch(self, start, batch_size):
        """Get a batch of data"""
        end = min(start + batch_size, self.num_samples)
        actual_batch = end - start

        # Input shape: [batch, seq_len, 1]
        inputs = np.array(self.inputs[start:end]).reshape(actual_batch, self.seq_length, 1)
        targets = np.array(self.targets[start:end])

        input_tensor = tz.Tensor.from_numpy(inputs)
        target_tensor = tz.Tensor.from_numpy(targets)

        return input_tensor, target_tensor

    def __len__(self):
        return self.num_samples


# ============================================================================
# GRU-based Time Series Model
# ============================================================================

class GRUForecastModel:
    """
    GRU model for time series forecasting

    Architecture:
    - Input projection with LayerNorm
    - Multi-layer GRU with dropout
    - Output projection
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection
        self.input_proj = tz.nn.Linear(input_size, hidden_size, bias=True)
        self.input_norm = tz.nn.LayerNorm([hidden_size])

        # GRU layer
        self.gru = tz.nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # Output projection
        self.dropout = tz.nn.Dropout(dropout)
        self.output_proj = tz.nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input variable [batch, seq_len, input_size]
        Returns:
            Output predictions [batch, output_size]
        """
        # Input projection
        h = self.input_proj(x)
        h = self.input_norm(h)

        # GRU processing
        h = self.gru(h)

        # Extract last timestep
        h_tensor = h.tensor()
        h_np = h_tensor.numpy()
        last_h = h_np[:, -1, :]  # [batch, hidden]
        last_var = tz.Variable(tz.Tensor.from_numpy(last_h), requires_grad=True)

        # Output projection
        out = self.dropout(last_var)
        out = self.output_proj(out)

        return out

    def parameters(self):
        """Get all trainable parameters"""
        return (self.input_proj.parameters() +
                self.input_norm.parameters() +
                self.gru.parameters() +
                self.dropout.parameters() +
                self.output_proj.parameters())

    def train(self):
        """Set to training mode"""
        self.input_proj.train()
        self.input_norm.train()
        self.gru.train()
        self.dropout.train()
        self.output_proj.train()

    def eval(self):
        """Set to evaluation mode"""
        self.input_proj.eval()
        self.input_norm.eval()
        self.gru.eval()
        self.dropout.eval()
        self.output_proj.eval()


# ============================================================================
# Training Functions
# ============================================================================

def train_with_rmsprop():
    """Train with RMSprop optimizer and StepLR scheduler"""
    print("\n" + "=" * 60)
    print("Training with RMSprop + StepLR")
    print("=" * 60)

    # Create datasets
    train_data = TimeSeriesDataset(800, 50, 10)
    val_data = TimeSeriesDataset(200, 50, 10)

    # Create model
    model = GRUForecastModel(1, 64, 10, num_layers=2, dropout=0.2)
    model.train()

    # RMSprop optimizer
    params = model.parameters()
    optimizer = tz.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8)

    # StepLR scheduler
    scheduler = tz.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # MSE Loss
    criterion = tz.nn.MSELoss()

    batch_size = 32
    num_epochs = 15

    print("\nConfiguration:")
    print("  Optimizer: RMSprop (lr=0.01, alpha=0.99)")
    print("  Scheduler: StepLR (step_size=5, gamma=0.5)")
    print("  Loss: MSELoss")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}\n")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            input_tensor, target_tensor = train_data.get_batch(i, batch_size)

            optimizer.zero_grad()

            input_var = tz.Variable(input_tensor, requires_grad=True)
            output = model.forward(input_var)
            target_var = tz.Variable(target_tensor, requires_grad=False)

            loss = criterion(output, target_var)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.tensor().item()
            num_batches += 1

        # Update learning rate
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with tz.no_grad():
            for i in range(0, len(val_data), batch_size):
                input_tensor, target_tensor = val_data.get_batch(i, batch_size)
                input_var = tz.Variable(input_tensor, requires_grad=False)
                output = model.forward(input_var)
                target_var = tz.Variable(target_tensor, requires_grad=False)
                loss = criterion(output, target_var)
                val_loss += loss.tensor().item()
                val_batches += 1

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {epoch_loss/num_batches:.6f} | "
              f"Val Loss: {val_loss/val_batches:.6f} | "
              f"LR: {scheduler.get_lr():.6f}")


def train_with_adagrad():
    """Train with Adagrad optimizer and ExponentialLR scheduler"""
    print("\n" + "=" * 60)
    print("Training with Adagrad + ExponentialLR")
    print("=" * 60)

    train_data = TimeSeriesDataset(800, 50, 10)
    val_data = TimeSeriesDataset(200, 50, 10)

    model = GRUForecastModel(1, 64, 10, num_layers=2, dropout=0.2)
    model.train()

    # Adagrad optimizer
    params = model.parameters()
    optimizer = tz.optim.Adagrad(params, lr=0.1, eps=1e-10)

    # ExponentialLR scheduler
    scheduler = tz.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # SmoothL1Loss (Huber loss)
    criterion = tz.nn.SmoothL1Loss(beta=1.0)

    batch_size = 32
    num_epochs = 15

    print("\nConfiguration:")
    print("  Optimizer: Adagrad (lr=0.1)")
    print("  Scheduler: ExponentialLR (gamma=0.95)")
    print("  Loss: SmoothL1Loss (Huber, beta=1.0)\n")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            input_tensor, target_tensor = train_data.get_batch(i, batch_size)

            optimizer.zero_grad()

            input_var = tz.Variable(input_tensor, requires_grad=True)
            output = model.forward(input_var)
            target_var = tz.Variable(target_tensor, requires_grad=False)

            loss = criterion(output, target_var)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.tensor().item()
            num_batches += 1

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_batches = 0

        with tz.no_grad():
            for i in range(0, len(val_data), batch_size):
                input_tensor, target_tensor = val_data.get_batch(i, batch_size)
                input_var = tz.Variable(input_tensor, requires_grad=False)
                output = model.forward(input_var)
                target_var = tz.Variable(target_tensor, requires_grad=False)
                loss = criterion(output, target_var)
                val_loss += loss.tensor().item()
                val_batches += 1

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {epoch_loss/num_batches:.6f} | "
              f"Val Loss: {val_loss/val_batches:.6f} | "
              f"LR: {scheduler.get_lr():.6f}")


def train_with_cosine_annealing():
    """Train with Adadelta optimizer and CosineAnnealingLR scheduler"""
    print("\n" + "=" * 60)
    print("Training with Adadelta + CosineAnnealingLR")
    print("=" * 60)

    train_data = TimeSeriesDataset(800, 50, 10)
    val_data = TimeSeriesDataset(200, 50, 10)

    model = GRUForecastModel(1, 64, 10, num_layers=2, dropout=0.2)
    model.train()

    # Adadelta optimizer
    params = model.parameters()
    optimizer = tz.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-6)

    # CosineAnnealingLR scheduler
    T_max = 15
    scheduler = tz.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0.0001)

    # L1 Loss (MAE)
    criterion = tz.nn.L1Loss()

    batch_size = 32

    print("\nConfiguration:")
    print("  Optimizer: Adadelta (lr=1.0, rho=0.9)")
    print(f"  Scheduler: CosineAnnealingLR (T_max={T_max}, eta_min=0.0001)")
    print("  Loss: L1Loss (MAE)\n")

    for epoch in range(T_max):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            input_tensor, target_tensor = train_data.get_batch(i, batch_size)

            optimizer.zero_grad()

            input_var = tz.Variable(input_tensor, requires_grad=True)
            output = model.forward(input_var)
            target_var = tz.Variable(target_tensor, requires_grad=False)

            loss = criterion(output, target_var)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.tensor().item()
            num_batches += 1

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_batches = 0

        with tz.no_grad():
            for i in range(0, len(val_data), batch_size):
                input_tensor, target_tensor = val_data.get_batch(i, batch_size)
                input_var = tz.Variable(input_tensor, requires_grad=False)
                output = model.forward(input_var)
                target_var = tz.Variable(target_tensor, requires_grad=False)
                loss = criterion(output, target_var)
                val_loss += loss.tensor().item()
                val_batches += 1

        print(f"Epoch {epoch+1:2d}/{T_max} | "
              f"Train Loss: {epoch_loss/num_batches:.6f} | "
              f"Val Loss: {val_loss/val_batches:.6f} | "
              f"LR: {scheduler.get_lr():.6f}")


def demo_gru_cell():
    """Demonstrate GRUCell for manual sequence processing"""
    print("\n" + "=" * 60)
    print("GRUCell Manual Sequence Processing")
    print("=" * 60)

    input_size = 1
    hidden_size = 32

    gru_cell = tz.nn.GRUCell(input_size, hidden_size, bias=True)
    fc = tz.nn.Linear(hidden_size, 1, bias=True)

    print("\nGRUCell architecture:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print("  Output size: 1")

    # Generate test sequence
    seq_len = 20
    batch_size = 4

    np.random.seed(42)
    input_data = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)
    h0 = np.zeros((batch_size, hidden_size), dtype=np.float32)

    print(f"\nProcessing {seq_len} timesteps manually...")

    hidden = tz.Variable(tz.Tensor.from_numpy(h0), requires_grad=True)

    for t in range(seq_len):
        x_t = input_data[:, t, :]
        x_var = tz.Variable(tz.Tensor.from_numpy(x_t), requires_grad=True)

        # Update hidden state
        hidden = gru_cell(x_var, hidden)

    # Final output
    output = fc(hidden)

    print(f"Final output shape: {output.tensor().shape}")
    print("GRUCell allows custom processing of hidden states!")


# ============================================================================
# Main
# ============================================================================

def main():
    # Initialize Tenzor library first
    tz.initialize()

    print("=" * 60)
    print("   GRU Time Series Forecasting - Component Coverage   ")
    print("=" * 60)

    print("\nComponents demonstrated in this example:")
    print("  Layers: GRU, GRUCell, LayerNorm, Linear, Dropout")
    print("  Optimizers: RMSprop, Adagrad, Adadelta")
    print("  Schedulers: StepLR, ExponentialLR, CosineAnnealingLR")
    print("  Losses: MSELoss, SmoothL1Loss (Huber), L1Loss")

    # Train with different optimizer/scheduler combinations
    train_with_rmsprop()
    train_with_adagrad()
    train_with_cosine_annealing()

    # Demo GRUCell
    demo_gru_cell()

    print("\n" + "=" * 60)
    print("   All training examples completed successfully!       ")
    print("=" * 60)


if __name__ == "__main__":
    main()
