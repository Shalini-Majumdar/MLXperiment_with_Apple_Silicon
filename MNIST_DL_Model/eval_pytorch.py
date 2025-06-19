import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pytorch_model import MNISTModel

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.pytorch_model import MNISTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = MNISTModel().to(device)
model_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'mnist_model.pth')
model.load_state_dict(torch.load(model_path))

# Load test data
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Evaluate
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

accuracy = correct / len(test_loader.dataset)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

