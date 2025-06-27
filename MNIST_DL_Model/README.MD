# MNIST Benchmarking: PyTorch vs MLX (Single‑node & Distributed)

<h1>🧭 Project Overview</h1>
This project compares two simple MNIST classifiers (PyTorch and MLX) on Apple Silicon (MacBook & Mac Mini) to evaluate performance differences in:

1. Model Implementation

2. Training & Evaluation (Single‑node)

3. Distributed Training (PyTorch)

Everything’s tracked in detail via metrics: accuracy, inference time, throughput, memory usage, CPU/GPU utilization.


<h1>📁 Repository Structure </h1>

     MLXProj/
    ├── training/
    │   ├── train_pytorch.py
    │   ├── train_pytorch_ddp.py
    │   ├── train_mlx.py
    │── evaluation/
    │   └── eval_benchmark.py
    |   |__ eval_pytorch.py
    |   |__ eval_benchmark_ddp.py
    ├── models/
    │   ├── pytorch_model.py
    │   └── mlx_model.py
    ├── results/               # Saved models & logs
    └── README.md


<h1>🛠️ Environment Setup</h1>

OS: macOS (MacBookPro (M3 Pro Chip), & 2 Mac Mini's (M4 Pro chips))

Python: 3.13 virtual environment env_mlx

Dependencies: torch, torchvision, mlx, psutil, memory_profiler

Metal/Developer Tools installed (xcode-select)


<h1> 🔄 Single‑Node Experiments </h1>

1. PyTorch (Non‑MLX)

train_pytorch.py trains a simple 2‑layer MLP on MNIST

eval_benchmark.py logs:

Accuracy (~97.5–97.6%)

Inference time (~0.14–0.20 s, ~50‑70k samples/sec)

CPU memory (~300 MB), CPU usage (~2–4%), GPU for MLX only

✅ Trained & evaluated on MacBook & Mac Mini individually








