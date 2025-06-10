
import os
import mlx.nn as nn
import mlx.core as mx
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from mlx.optimizers import Adam
from mlx.nn.losses import cross_entropy
import psutil


# ------------------ Model ------------------
class MLXMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)

    def __call__(self, x):
        x = mx.reshape(x, (x.shape[0], -1))  # Flatten
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# ------------------ Data Loading ------------------
transform = transforms.ToTensor()
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

# Convert to MLX arrays
train_images = []
train_labels = []
for img, label in train_dataset:
    img_np = np.array(img)  # (1, 28, 28)
    train_images.append(mx.array(img_np))
    train_labels.append(mx.array(label))

train_images = mx.stack(train_images)  # (N, 1, 28, 28)
train_labels = mx.array(train_labels)  # (N,)

# ------------------ Optimizer & Model ------------------
model = MLXMNISTModel()
optimizer = Adam(learning_rate=0.001)

# ------------------ Loss Function ------------------
def loss_fn(preds, targets):
    return cross_entropy(preds, targets).mean()  # Ensure scalar

# ------------------ Training Step ------------------
def train_step(x, y):
    def compute_loss(m):
        logits = m(x)
        return loss_fn(logits, y)

    loss, grads = mx.value_and_grad(compute_loss)(model)
    optimizer.update(model, grads)
    return loss

# ------------------ Training Loop ------------------
batch_size = 64
num_epochs = 5  # ✅ Increased for better convergence

for epoch in range(num_epochs):
    # ✅ Shuffle each epoch
    indices = mx.array(np.random.permutation(train_images.shape[0]))
    train_images = mx.take(train_images, indices, axis=0)
    train_labels = mx.take(train_labels, indices, axis=0)

    total_loss = 0
    for i in range(0, train_images.shape[0], batch_size):
        x_batch = train_images[i:i+batch_size]
        y_batch = train_labels[i:i+batch_size]
        loss = train_step(x_batch, y_batch)
        total_loss += loss.item()

    avg_loss = total_loss / (train_images.shape[0] // batch_size)
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

#Save the model 
# ------------------ Evaluation ------------------
from torchvision.datasets import MNIST

# Load test data using torchvision
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

# Convert to MLX arrays
test_images = []
test_labels = []
for img, label in test_dataset:
    img_np = np.array(img)  # (1, 28, 28)
    test_images.append(mx.array(img_np))
    test_labels.append(mx.array(label))

test_images = mx.stack(test_images)  # (N, 1, 28, 28)
test_labels = mx.array(test_labels)  # (N,)

# Evaluate
correct = 0
batch_size = 1000

for i in range(0, test_images.shape[0], batch_size):
    x_batch = test_images[i:i+batch_size]
    y_batch = test_labels[i:i+batch_size]

    logits = model(x_batch)
    preds = mx.argmax(logits, axis=1)
    correct += int(mx.sum(preds == y_batch).item())

accuracy = correct / test_images.shape[0]
print(f"Test Accuracy: {accuracy * 100:.2f}%")


#Benchmarking 


# ------------------ Evaluation & Benchmarking ------------------
import time
from memory_profiler import memory_usage

# Load test data
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

# Convert to MLX arrays
test_images = []
test_labels = []
for img, label in test_dataset:
    img_np = np.array(img)  # shape (1, 28, 28)
    test_images.append(mx.array(img_np))
    test_labels.append(mx.array(label))

test_images = mx.stack(test_images)  # (N, 1, 28, 28)
test_labels = mx.array(test_labels)  # (N,)

# Define inference function
def run_inference():
    correct = 0
    batch_size = 1000
    for i in range(0, test_images.shape[0], batch_size):
        x_batch = test_images[i:i+batch_size]
        y_batch = test_labels[i:i+batch_size]
        logits = model(x_batch)
        preds = mx.argmax(logits, axis=1)
        correct += int(mx.sum(preds == y_batch).item())
    return correct / test_images.shape[0]

if __name__ == "__main__":
    # Benchmark
    cpu_start = psutil.cpu_percent(interval=None)  # Baseline
    start_time = time.perf_counter()

    mem_usage, accuracy = memory_usage((run_inference,), retval=True, max_iterations=1)
    
    end_time = time.perf_counter()
    cpu_end = psutil.cpu_percent(interval=1)  # Measure over 1 sec

    # Metrics
    inference_time = end_time - start_time
    mem_peak = max(mem_usage) - min(mem_usage)

    print(f"\n[MLX Evaluation]")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Total inference time: {inference_time:.4f} seconds")
    print(f"Inference speed: {test_images.shape[0] / inference_time:.2f} samples/sec")
    print(f"Peak memory usage: {mem_peak:.2f} MB")
    print(f"CPU usage during eval: {cpu_end:.1f}%")
