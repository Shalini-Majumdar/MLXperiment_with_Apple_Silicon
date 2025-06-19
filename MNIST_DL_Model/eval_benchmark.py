import torch
import time
import psutil
import os
from memory_profiler import memory_usage
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pytorch_model import MNISTModel

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTModel().to(device)
model.load_state_dict(torch.load("../results/mnist_model.pth"))
model.eval()

# Data
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Function to evaluate and measure inference time
def evaluate():
    correct = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    end_time = time.perf_counter()
    accuracy = correct / len(test_loader.dataset)
    inference_time = end_time - start_time
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Total inference time: {inference_time:.4f} seconds")
    print(f"Inference speed: {len(test_loader.dataset)/inference_time:.2f} samples/sec")

if __name__ == "__main__":
    print("Running evaluation with resource profiling...\n")
    mem_usage = memory_usage((evaluate,), interval=0.1, timeout=None, max_iterations=1)

    print(f"Peak memory usage: {max(mem_usage):.2f} MB")
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU usage during eval: {cpu_percent}%")

    if torch.cuda.is_available():
        print(f"GPU memory usage: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
