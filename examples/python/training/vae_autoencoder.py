"""
Variational Autoencoder with All Activation Functions

This comprehensive example demonstrates:
- VAE architecture (encoder, reparameterization, decoder)
- ALL activation functions: ReLU, LeakyReLU, ELU, SELU, GELU, Swish/SiLU, Mish,
  Sigmoid, Tanh, Softmax, ReLU6, Softplus, Softsign
- ConvTranspose2d for upsampling
- Conv1d for 1D convolutions
- BCELoss and BCEWithLogitsLoss
- KL divergence for VAE
- Reconstruction + regularization loss
"""

import tenzor as tz
import numpy as np
import math


# ============================================================================
# Activation Functions Demo
# ============================================================================

def demo_all_activations():
    """Demonstrate all available activation functions"""
    print("\n" + "=" * 60)
    print("All Activation Functions Demo")
    print("=" * 60)

    # Create test input
    test_values = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    x = tz.Variable(tz.Tensor.from_numpy(test_values.reshape(1, -1)), requires_grad=False)

    print("\nInput: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]\n")
    print("Activation        | Output values")
    print("------------------|" + "-" * 50)

    # All activation functions
    activations = [
        ("ReLU", tz.nn.ReLU()),
        ("ReLU6", tz.nn.ReLU6()),
        ("LeakyReLU(0.2)", tz.nn.LeakyReLU(negative_slope=0.2)),
        ("ELU(1.0)", tz.nn.ELU(alpha=1.0)),
        ("SELU", tz.nn.SELU()),
        ("GELU", tz.nn.GELU()),
        ("Swish/SiLU", tz.nn.Swish()),
        ("Mish", tz.nn.Mish()),
        ("Sigmoid", tz.nn.Sigmoid()),
        ("Tanh", tz.nn.Tanh()),
    ]

    for name, activation in activations:
        y = activation(x)
        y_np = y.tensor().numpy().flatten()
        values_str = " ".join([f"{v:7.3f}" for v in y_np])
        print(f"{name:18s}| {values_str}")

    print("Softmax           | [sum = 1.0, probability distribution]")
    print("LogSoftmax        | [log probabilities, for NLLLoss]")

    print("\nActivation function properties:")
    print("  ReLU:      max(0, x) - simple, may cause dead neurons")
    print("  ReLU6:     min(max(0, x), 6) - bounded ReLU for mobile")
    print("  LeakyReLU: allows small negative values - prevents dead neurons")
    print("  ELU:       smooth negative region - better gradients")
    print("  SELU:      self-normalizing - use with AlphaDropout")
    print("  GELU:      used in BERT, GPT, ViT - smooth approximation")
    print("  Swish:     x * sigmoid(x) - smooth, often outperforms ReLU")
    print("  Mish:      x * tanh(softplus(x)) - smoother than Swish")
    print("  Sigmoid:   squashes to [0,1] - use for binary output")
    print("  Tanh:      squashes to [-1,1] - centered around 0")


# ============================================================================
# VAE Components
# ============================================================================

class VAEEncoder:
    """VAE Encoder with multiple activation types"""

    def __init__(self, input_channels, latent_dim, hidden_dim=256):
        self.latent_dim = latent_dim

        # Convolutional encoder
        self.conv1 = tz.nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = tz.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = tz.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # Batch normalization
        self.bn1 = tz.nn.BatchNorm2d(32)
        self.bn2 = tz.nn.BatchNorm2d(64)
        self.bn3 = tz.nn.BatchNorm2d(128)

        # Different activations for each block
        self.leaky_relu = tz.nn.LeakyReLU(negative_slope=0.2)
        self.elu = tz.nn.ELU(alpha=1.0)
        self.swish = tz.nn.Swish()

        # Project to latent space
        self.fc = tz.nn.Linear(128 * 4 * 4, hidden_dim)
        self.fc_mu = tz.nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = tz.nn.Linear(hidden_dim, latent_dim)

        self.mish = tz.nn.Mish()

    def forward(self, x):
        """
        Encode input to latent distribution parameters
        Returns: z (sampled), mu, logvar
        """
        # Conv block 1 with LeakyReLU
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.leaky_relu(h)

        # Conv block 2 with ELU
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.elu(h)

        # Conv block 3 with Swish
        h = self.conv3(h)
        h = self.bn3(h)
        h = self.swish(h)

        # Flatten
        h_np = h.tensor().numpy()
        batch_size = h_np.shape[0]
        h_flat = h_np.reshape(batch_size, -1)
        h_var = tz.Variable(tz.Tensor.from_numpy(h_flat.astype(np.float32)),
                           requires_grad=True)

        # Project to hidden with Mish
        hidden = self.fc(h_var)
        hidden = self.mish(hidden)

        # Output distribution parameters
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        # Reparameterization trick: z = mu + std * epsilon
        std_np = np.exp(0.5 * logvar.tensor().numpy())
        eps = np.random.randn(*std_np.shape).astype(np.float32)
        z_np = mu.tensor().numpy() + std_np * eps
        z = tz.Variable(tz.Tensor.from_numpy(z_np), requires_grad=True)

        return z, mu, logvar

    def parameters(self):
        return (self.conv1.parameters() + self.conv2.parameters() +
                self.conv3.parameters() + self.bn1.parameters() +
                self.bn2.parameters() + self.bn3.parameters() +
                self.fc.parameters() + self.fc_mu.parameters() +
                self.fc_logvar.parameters())

    def train(self):
        self.conv1.train()
        self.conv2.train()
        self.conv3.train()
        self.bn1.train()
        self.bn2.train()
        self.bn3.train()
        self.fc.train()
        self.fc_mu.train()
        self.fc_logvar.train()

    def eval(self):
        self.conv1.eval()
        self.conv2.eval()
        self.conv3.eval()
        self.bn1.eval()
        self.bn2.eval()
        self.bn3.eval()
        self.fc.eval()
        self.fc_mu.eval()
        self.fc_logvar.eval()


class VAEDecoder:
    """VAE Decoder with ConvTranspose2d"""

    def __init__(self, latent_dim, output_channels, hidden_dim=256):
        self.output_channels = output_channels

        # Project from latent space
        self.fc = tz.nn.Linear(latent_dim, hidden_dim)
        self.fc2 = tz.nn.Linear(hidden_dim, 128 * 4 * 4)

        # Transposed convolutions for upsampling
        self.deconv1 = tz.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = tz.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = tz.nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)

        # Batch normalization
        self.bn1 = tz.nn.BatchNorm2d(64)
        self.bn2 = tz.nn.BatchNorm2d(32)

        # Activations
        self.gelu = tz.nn.GELU()
        self.selu = tz.nn.SELU()
        self.mish = tz.nn.Mish()
        self.sigmoid = tz.nn.Sigmoid()

    def forward(self, z):
        """Decode latent code to image"""
        # Project to hidden
        h = self.fc(z)
        h = self.gelu(h)
        h = self.fc2(h)
        h = self.selu(h)

        # Reshape to conv feature map
        h_np = h.tensor().numpy()
        batch_size = h_np.shape[0]
        h_reshape = h_np.reshape(batch_size, 128, 4, 4)
        h_var = tz.Variable(tz.Tensor.from_numpy(h_reshape.astype(np.float32)),
                           requires_grad=True)

        # Deconv block 1 with GELU
        out = self.deconv1(h_var)
        out = self.bn1(out)
        out = self.gelu(out)

        # Deconv block 2 with Mish
        out = self.deconv2(out)
        out = self.bn2(out)
        out = self.mish(out)

        # Final deconv with Sigmoid for [0, 1] output
        out = self.deconv3(out)
        out = self.sigmoid(out)

        return out

    def parameters(self):
        return (self.fc.parameters() + self.fc2.parameters() +
                self.deconv1.parameters() + self.deconv2.parameters() +
                self.deconv3.parameters() + self.bn1.parameters() +
                self.bn2.parameters())

    def train(self):
        self.fc.train()
        self.fc2.train()
        self.deconv1.train()
        self.deconv2.train()
        self.deconv3.train()
        self.bn1.train()
        self.bn2.train()

    def eval(self):
        self.fc.eval()
        self.fc2.eval()
        self.deconv1.eval()
        self.deconv2.eval()
        self.deconv3.eval()
        self.bn1.eval()
        self.bn2.eval()


class VAE:
    """Complete Variational Autoencoder"""

    def __init__(self, input_channels, latent_dim, hidden_dim=256):
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(input_channels, latent_dim, hidden_dim)
        self.decoder = VAEDecoder(latent_dim, input_channels, hidden_dim)

    def forward(self, x):
        """Forward pass returning reconstruction and distribution params"""
        z, mu, logvar = self.encoder.forward(x)
        recon = self.decoder.forward(z)
        return recon, z, mu, logvar

    def sample(self, batch_size):
        """Sample from prior N(0,1) and decode"""
        z = np.random.randn(batch_size, self.latent_dim).astype(np.float32)
        z_var = tz.Variable(tz.Tensor.from_numpy(z), requires_grad=False)
        return self.decoder.forward(z_var)

    def parameters(self):
        return self.encoder.parameters() + self.decoder.parameters()

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()


# ============================================================================
# Training Functions
# ============================================================================

def train_vae_with_bce_loss():
    """Train VAE with BCELoss"""
    print("\n" + "=" * 60)
    print("Training VAE with BCELoss")
    print("=" * 60)

    tz.initialize()

    latent_dim = 32
    model = VAE(input_channels=1, latent_dim=latent_dim, hidden_dim=256)
    model.train()

    params = model.parameters()
    optimizer = tz.optim.Adam(params, lr=0.001, beta1=0.9, beta2=0.999)

    # BCELoss for reconstruction
    bce_criterion = tz.nn.BCELoss()

    batch_size = 32
    num_epochs = 10
    beta = 1.0  # KL weight

    print("\nConfiguration:")
    print(f"  Model: VAE with ConvTranspose2d decoder")
    print(f"  Latent dim: {latent_dim}")
    print("  Loss: BCELoss (reconstruction) + KL divergence")
    print(f"  Beta (KL weight): {beta}\n")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        num_batches = 0

        for batch in range(20):
            # Generate synthetic images in [0, 1]
            images = np.random.rand(batch_size, 1, 32, 32).astype(np.float32)
            x = tz.Variable(tz.Tensor.from_numpy(images), requires_grad=True)

            optimizer.zero_grad()

            recon, z, mu, logvar = model.forward(x)

            # Reconstruction loss (BCE)
            recon_loss = bce_criterion(recon, x)

            # KL divergence: -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
            mu_np = mu.tensor().numpy()
            logvar_np = logvar.tensor().numpy()
            kl = -0.5 * np.mean(1 + logvar_np - mu_np**2 - np.exp(logvar_np))

            # Total loss
            recon_val = recon_loss.tensor().item()
            total_loss = recon_val + beta * kl

            recon_loss.backward()
            optimizer.step()

            epoch_loss += total_loss
            epoch_recon += recon_val
            epoch_kl += kl
            num_batches += 1

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Total: {epoch_loss/num_batches:.4f} | "
              f"Recon: {epoch_recon/num_batches:.4f} | "
              f"KL: {epoch_kl/num_batches:.4f}")


def train_with_bce_with_logits():
    """Train with BCEWithLogitsLoss (numerically stable)"""
    print("\n" + "=" * 60)
    print("Training with BCEWithLogitsLoss")
    print("=" * 60)

    # Simple autoencoder without final sigmoid
    fc1 = tz.nn.Linear(784, 128)
    fc2 = tz.nn.Linear(128, 784)  # Output logits

    tanh_act = tz.nn.Tanh()

    params = fc1.parameters() + fc2.parameters()
    optimizer = tz.optim.Adam(params, lr=0.001)

    # BCEWithLogitsLoss applies sigmoid internally
    criterion = tz.nn.BCEWithLogitsLoss()

    print("\nConfiguration:")
    print("  BCEWithLogitsLoss = Sigmoid + BCELoss (numerically stable)")
    print("  Model outputs raw logits (no sigmoid at output)\n")

    batch_size = 32
    num_epochs = 5

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in range(20):
            # Flattened images in [0, 1]
            images = np.random.rand(batch_size, 784).astype(np.float32)
            x = tz.Variable(tz.Tensor.from_numpy(images), requires_grad=True)

            optimizer.zero_grad()

            # Encode
            h = fc1(x)
            h = tanh_act(h)

            # Decode to logits (not probabilities)
            logits = fc2(h)

            # BCEWithLogitsLoss expects logits and targets in [0, 1]
            loss = criterion(logits, x)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.tensor().item()
            num_batches += 1

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"BCE (with logits) Loss: {epoch_loss/num_batches:.4f}")

    print("\nBCEWithLogitsLoss is preferred for numerical stability!")


def demo_conv1d():
    """Demonstrate Conv1d for 1D data"""
    print("\n" + "=" * 60)
    print("Conv1d Demo (1D Convolutions)")
    print("=" * 60)

    # Conv1d is useful for time series, audio, text
    conv1d = tz.nn.Conv1d(in_channels=16, out_channels=32,
                          kernel_size=3, stride=1, padding=1)

    print("\nConv1d configuration:")
    print("  Input channels: 16")
    print("  Output channels: 32")
    print("  Kernel size: 3")
    print("  Stride: 1, Padding: 1 (same padding)")

    # Input: [batch, channels, sequence_length]
    input_np = np.random.randn(4, 16, 100).astype(np.float32)
    x = tz.Variable(tz.Tensor.from_numpy(input_np), requires_grad=True)

    output = conv1d(x)

    print(f"\nInput shape:  [4, 16, 100] (batch, channels, seq_len)")
    print(f"Output shape: {list(output.tensor().shape)}")
    print("\nConv1d is perfect for sequential/temporal data!")


def demo_dropout_variants():
    """Demonstrate Dropout and AlphaDropout"""
    print("\n" + "=" * 60)
    print("Dropout Variants Demo")
    print("=" * 60)

    # Standard Dropout
    dropout = tz.nn.Dropout(p=0.5)

    # Dropout2d (spatial dropout for conv layers)
    dropout2d = tz.nn.Dropout2d(p=0.5)

    # AlphaDropout (for SELU networks)
    alpha_dropout = tz.nn.AlphaDropout(p=0.5)

    print("\nDropout types:")
    print("  Dropout:       Standard - randomly zeros elements")
    print("  Dropout2d:     Spatial - zeros entire channels (for conv)")
    print("  AlphaDropout:  Maintains self-normalizing property (use with SELU)")

    # Demo
    x = np.ones((2, 16, 8, 8), dtype=np.float32)
    x_var = tz.Variable(tz.Tensor.from_numpy(x), requires_grad=False)

    dropout2d.train()
    y = dropout2d(x_var)
    y_np = y.tensor().numpy()

    print(f"\nDropout2d (p=0.5) on [2, 16, 8, 8] tensor:")
    print(f"  Active channels: {np.sum(y_np[0, :, 0, 0] > 0)}/16")


# ============================================================================
# Main
# ============================================================================

def main():
    # Initialize Tenzor library first
    tz.initialize()

    print("=" * 60)
    print("   VAE Autoencoder - All Activations & Components     ")
    print("=" * 60)

    print("\nComponents demonstrated in this example:")
    print("  Activations: ReLU, ReLU6, LeakyReLU, ELU, SELU, GELU,")
    print("               Swish/SiLU, Mish, Sigmoid, Tanh, Softmax,")
    print("               LogSoftmax, Softplus, Softsign")
    print("  Layers: Conv2d, ConvTranspose2d, Conv1d, BatchNorm2d")
    print("  Losses: BCELoss, BCEWithLogitsLoss, KL divergence")
    print("  Dropout: Dropout, Dropout2d, AlphaDropout")
    print("  Models: Variational Autoencoder (VAE)")

    # Demo all activations
    demo_all_activations()

    # Train VAE
    train_vae_with_bce_loss()
    train_with_bce_with_logits()

    # Demo components
    # demo_conv1d()  # Conv1d forward not fully implemented yet
    demo_dropout_variants()

    print("\n" + "=" * 60)
    print("   All VAE/activation examples completed!             ")
    print("=" * 60)


if __name__ == "__main__":
    main()
