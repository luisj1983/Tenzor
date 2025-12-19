"""
Knowledge Distillation Example

This comprehensive example demonstrates:
- Basic teacher-student distillation
- Temperature and alpha tuning
- Temperature annealing schedule
- Feature distillation concepts
- Multi-teacher distillation
- Complete distillation workflow

Components used:
- Conv2d, Linear, BatchNorm2d
- CrossEntropyLoss, Adam optimizer
- KL Divergence for soft targets
"""

import tenzor as tz
import numpy as np


# ============================================================================
# Models for Distillation
# ============================================================================

class TeacherModel:
    """Large teacher model (ResNet-like)"""

    def __init__(self, num_classes=10):
        # Large capacity network
        self.conv1 = tz.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = tz.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = tz.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = tz.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        # Global average pooling followed by FC
        self.fc = tz.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = tz.nn.relu(x)

        x = self.conv2(x)
        x = tz.nn.relu(x)

        x = self.conv3(x)
        x = tz.nn.relu(x)

        x = self.conv4(x)
        x = tz.nn.relu(x)

        # Global average pooling
        x_np = x.tensor().numpy()
        pooled = np.mean(x_np, axis=(2, 3))  # [batch, channels]
        x = tz.Variable(tz.Tensor.from_numpy(pooled.astype(np.float32)), requires_grad=True)

        return self.fc(x)

    def parameters(self):
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.conv3.parameters())
        params.extend(self.conv4.parameters())
        params.extend(self.fc.parameters())
        return params

    def count_parameters(self):
        total = 0
        for param in self.parameters():
            total += param.tensor().numpy().size
        return total


class StudentModel:
    """Small student model (MobileNet-like with depthwise separable)"""

    def __init__(self, num_classes=10):
        # Lightweight network
        self.conv1 = tz.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)

        # Depthwise separable blocks (simulated with standard conv for demo)
        self.dw1 = tz.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.pw1 = tz.nn.Conv2d(16, 32, kernel_size=1)

        self.dw2 = tz.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.pw2 = tz.nn.Conv2d(32, 64, kernel_size=1)

        # FC layer
        self.fc = tz.nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = tz.nn.relu(x)

        # Depthwise separable 1
        x = self.dw1(x)
        x = tz.nn.relu(x)
        x = self.pw1(x)
        x = tz.nn.relu(x)

        # Depthwise separable 2
        x = self.dw2(x)
        x = tz.nn.relu(x)
        x = self.pw2(x)
        x = tz.nn.relu(x)

        # Global average pooling
        x_np = x.tensor().numpy()
        pooled = np.mean(x_np, axis=(2, 3))
        x = tz.Variable(tz.Tensor.from_numpy(pooled.astype(np.float32)), requires_grad=True)

        return self.fc(x)

    def parameters(self):
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.dw1.parameters())
        params.extend(self.pw1.parameters())
        params.extend(self.dw2.parameters())
        params.extend(self.pw2.parameters())
        params.extend(self.fc.parameters())
        return params

    def count_parameters(self):
        total = 0
        for param in self.parameters():
            total += param.tensor().numpy().size
        return total


# ============================================================================
# Distillation Utilities
# ============================================================================

def softmax_with_temperature(logits, temperature=1.0):
    """Apply softmax with temperature scaling"""
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
    return exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)


def kl_divergence(student_probs, teacher_probs, epsilon=1e-8):
    """Compute KL divergence between distributions"""
    teacher_probs = np.clip(teacher_probs, epsilon, 1.0)
    student_probs = np.clip(student_probs, epsilon, 1.0)
    return np.sum(teacher_probs * np.log(teacher_probs / student_probs), axis=-1)


def distillation_loss(student_logits, teacher_logits, temperature, alpha, hard_labels=None):
    """
    Compute knowledge distillation loss

    Loss = alpha * KL(softmax(teacher/T), softmax(student/T)) * T^2
         + (1 - alpha) * CrossEntropy(student, hard_labels)
    """
    student_np = student_logits.tensor().numpy()
    teacher_np = teacher_logits.tensor().numpy()

    # Soft targets
    student_probs = softmax_with_temperature(student_np, temperature)
    teacher_probs = softmax_with_temperature(teacher_np, temperature)

    # KL divergence (soft loss) - scaled by T^2
    soft_loss = np.mean(kl_divergence(student_probs, teacher_probs)) * (temperature ** 2)

    # Hard loss (if labels provided)
    if hard_labels is not None and alpha < 1.0:
        # Simple cross-entropy approximation
        student_probs_t1 = softmax_with_temperature(student_np, 1.0)
        hard_loss = -np.mean(np.log(student_probs_t1[np.arange(len(hard_labels)), hard_labels] + 1e-8))
        total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    else:
        total_loss = soft_loss

    return total_loss


def temperature_schedule(initial_temp, final_temp, epoch, total_epochs, schedule_type="cosine"):
    """Compute temperature based on schedule"""
    t = epoch / max(total_epochs - 1, 1)

    if schedule_type == "linear":
        return initial_temp + t * (final_temp - initial_temp)
    elif schedule_type == "exponential":
        ratio = final_temp / initial_temp
        return initial_temp * (ratio ** t)
    elif schedule_type == "cosine":
        return final_temp + 0.5 * (initial_temp - final_temp) * (1 + np.cos(np.pi * t))
    else:
        return initial_temp


# ============================================================================
# Demonstrations
# ============================================================================

def demo_basic_distillation():
    """Demonstrate basic knowledge distillation"""
    print("\n" + "=" * 60)
    print("Basic Knowledge Distillation Demo")
    print("=" * 60)

    # Create models
    print("\nCreating teacher and student models...")
    teacher = TeacherModel(num_classes=10)
    student = StudentModel(num_classes=10)

    teacher_params = teacher.count_parameters()
    student_params = student.count_parameters()

    print(f"\nModel comparison:")
    print(f"  Teacher parameters: {teacher_params:,}")
    print(f"  Student parameters: {student_params:,}")
    print(f"  Compression ratio: {teacher_params / student_params:.2f}x")

    # Distillation config
    temperature = 3.0
    alpha = 0.7

    print(f"\nDistillation config:")
    print(f"  Temperature: {temperature}")
    print(f"  Alpha (soft loss weight): {alpha}")

    # Simulate training
    print("\nSimulating distillation training...")
    batch_size = 4
    input_size = 32

    for epoch in range(3):
        # Generate random input
        np.random.seed(epoch)
        input_np = np.random.randn(batch_size, 3, input_size, input_size).astype(np.float32)
        input_var = tz.Variable(tz.Tensor.from_numpy(input_np), requires_grad=True)

        # Forward pass through both models
        teacher_logits = teacher.forward(input_var)
        student_logits = student.forward(input_var)

        # Compute distillation loss
        hard_labels = np.random.randint(0, 10, size=batch_size)
        loss = distillation_loss(student_logits, teacher_logits, temperature, alpha, hard_labels)

        print(f"  Epoch {epoch + 1}/3 | Distillation loss: {loss:.4f}")

    print("\nBasic distillation complete!")


def demo_temperature_tuning():
    """Demonstrate temperature tuning effects"""
    print("\n" + "=" * 60)
    print("Temperature Tuning Demo")
    print("=" * 60)

    teacher = TeacherModel(num_classes=10)
    student = StudentModel(num_classes=10)

    temperatures = [1.0, 3.0, 5.0, 10.0]

    print("\nTesting different temperature values:\n")

    # Generate test input
    np.random.seed(42)
    input_np = np.random.randn(4, 3, 32, 32).astype(np.float32)
    input_var = tz.Variable(tz.Tensor.from_numpy(input_np), requires_grad=True)

    teacher_logits = teacher.forward(input_var)
    student_logits = student.forward(input_var)
    hard_labels = np.array([0, 1, 2, 3])

    for temp in temperatures:
        loss = distillation_loss(student_logits, teacher_logits, temp, 0.7, hard_labels)
        print(f"Temperature T={temp:4.1f}: loss = {loss:.4f}")

    print("\nRecommendation:")
    print("  - Low T (1-2): Hard targets, less knowledge transfer")
    print("  - Medium T (3-5): Balanced, recommended for classification")
    print("  - High T (8-10): Very soft targets, good for similar architectures")


def demo_alpha_tuning():
    """Demonstrate alpha (soft/hard loss balance) tuning"""
    print("\n" + "=" * 60)
    print("Alpha Tuning Demo")
    print("=" * 60)

    teacher = TeacherModel(num_classes=10)
    student = StudentModel(num_classes=10)

    alphas = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]

    print("\nTesting different alpha values (soft loss weight):\n")

    # Generate test input
    np.random.seed(42)
    input_np = np.random.randn(4, 3, 32, 32).astype(np.float32)
    input_var = tz.Variable(tz.Tensor.from_numpy(input_np), requires_grad=True)

    teacher_logits = teacher.forward(input_var)
    student_logits = student.forward(input_var)
    hard_labels = np.array([0, 1, 2, 3])

    for alpha in alphas:
        loss = distillation_loss(student_logits, teacher_logits, 3.0, alpha, hard_labels)
        print(f"Alpha α={alpha:.1f}: loss = {loss:.4f}  [{alpha * 100:.0f}% soft + {(1 - alpha) * 100:.0f}% hard]")

    print("\nRecommendation:")
    print("  - α=0.0: Standard training (no distillation)")
    print("  - α=0.5: Equal weight to soft and hard targets")
    print("  - α=0.7-0.9: Emphasize teacher knowledge (recommended)")
    print("  - α=1.0: Pure distillation (no hard labels)")


def demo_temperature_schedule():
    """Demonstrate temperature annealing"""
    print("\n" + "=" * 60)
    print("Temperature Annealing Demo")
    print("=" * 60)

    total_epochs = 100
    initial_temp = 10.0
    final_temp = 2.0

    print(f"\nTemperature schedule (initial={initial_temp}, final={final_temp}):\n")
    print("Epoch | Linear | Exponential | Cosine")
    print("------|--------|-------------|-------")

    for epoch in [0, 25, 50, 75, 99]:
        linear = temperature_schedule(initial_temp, final_temp, epoch, total_epochs, "linear")
        exponential = temperature_schedule(initial_temp, final_temp, epoch, total_epochs, "exponential")
        cosine = temperature_schedule(initial_temp, final_temp, epoch, total_epochs, "cosine")

        print(f"{epoch:5d} | {linear:6.2f} | {exponential:11.2f} | {cosine:6.2f}")

    print("\nNote: Start with high T for soft targets, decrease for sharper predictions")


def demo_feature_distillation():
    """Demonstrate feature distillation concepts"""
    print("\n" + "=" * 60)
    print("Feature Distillation Demo")
    print("=" * 60)

    print("\nFeature distillation matches intermediate representations")
    print("between teacher and student networks.\n")

    # Simulate feature extraction
    teacher_features = np.random.randn(4, 256, 8, 8).astype(np.float32)
    student_features = np.random.randn(4, 64, 8, 8).astype(np.float32)

    print(f"Teacher features: {teacher_features.shape[1]} channels")
    print(f"Student features: {student_features.shape[1]} channels")

    print("\nNote: Student needs projection layer to match teacher dimensions")

    print("\nDifferent feature matching losses:")
    print("  - MSE: Mean squared error between features")
    print("  - Cosine: Cosine similarity between features")
    print("  - Attention: Match attention maps from features")

    print("\nFeature distillation is especially effective for:")
    print("  - Vision tasks (CNNs)")
    print("  - Large architectural differences")
    print("  - Transfer learning scenarios")


def demo_multi_teacher():
    """Demonstrate multi-teacher distillation"""
    print("\n" + "=" * 60)
    print("Multi-Teacher Distillation Demo")
    print("=" * 60)

    print("\nTraining student from ensemble of teachers...\n")

    # Create multiple teachers
    teacher1 = TeacherModel(num_classes=10)
    teacher2 = TeacherModel(num_classes=10)
    teacher3 = TeacherModel(num_classes=10)
    student = StudentModel(num_classes=10)

    print("Ensemble: 3 teacher models")
    print("Student: 1 lightweight model\n")

    # Forward pass
    np.random.seed(42)
    input_np = np.random.randn(4, 3, 32, 32).astype(np.float32)
    input_var = tz.Variable(tz.Tensor.from_numpy(input_np), requires_grad=False)

    teacher_outputs = [
        teacher1.forward(input_var).tensor().numpy(),
        teacher2.forward(input_var).tensor().numpy(),
        teacher3.forward(input_var).tensor().numpy()
    ]

    student_output = student.forward(input_var).tensor().numpy()

    # Weighted average of teacher predictions
    teacher_weights = [0.4, 0.3, 0.3]
    ensemble_logits = sum(w * t for w, t in zip(teacher_weights, teacher_outputs))

    # Compute loss with ensemble
    student_probs = softmax_with_temperature(student_output, 3.0)
    ensemble_probs = softmax_with_temperature(ensemble_logits, 3.0)
    loss = np.mean(kl_divergence(student_probs, ensemble_probs)) * 9.0

    print(f"Multi-teacher distillation loss: {loss:.4f}")

    print("\nBenefits:")
    print("  - Student learns from diverse knowledge")
    print("  - Better generalization than single teacher")
    print("  - Ensemble knowledge compressed into single model")


def print_best_practices():
    """Print distillation best practices"""
    print("\n" + "=" * 60)
    print("Knowledge Distillation Best Practices")
    print("=" * 60)

    print("\n1. Temperature Selection:")
    print("   - Classification: T = 3-5")
    print("   - Detection: T = 2-3")
    print("   - Segmentation: T = 1.5-2.5")

    print("\n2. Alpha (soft/hard loss balance):")
    print("   - Start with α = 0.7-0.9 (emphasize soft targets)")
    print("   - Use α = 0.5 if student struggles")
    print("   - Can anneal α during training")

    print("\n3. Architecture Selection:")
    print("   - Teacher should be significantly larger")
    print("   - Student capacity should match task complexity")
    print("   - Compression ratio: typically 2-10x")

    print("\n4. Training Strategy:")
    print("   - Use same data as teacher")
    print("   - Start with high temperature, anneal down")
    print("   - Consider two-stage: distill → fine-tune")

    print("\n5. When Distillation Works Best:")
    print("   ✓ Large teacher, small student gap")
    print("   ✓ High-quality teacher (>90% accuracy)")
    print("   ✓ Sufficient training data")
    print("   ✓ Classification or detection tasks")


# ============================================================================
# Main
# ============================================================================

def main():
    tz.initialize()

    print("=" * 60)
    print("   Knowledge Distillation Example")
    print("=" * 60)

    print("\nComponents demonstrated:")
    print("  Models: Teacher (ResNet-like), Student (MobileNet-like)")
    print("  Loss: KL Divergence with temperature")
    print("  Techniques: Temperature/alpha tuning, annealing")

    demo_basic_distillation()
    demo_temperature_tuning()
    demo_alpha_tuning()
    demo_temperature_schedule()
    demo_feature_distillation()
    demo_multi_teacher()
    print_best_practices()

    print("\n" + "=" * 60)
    print("   Distillation example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
