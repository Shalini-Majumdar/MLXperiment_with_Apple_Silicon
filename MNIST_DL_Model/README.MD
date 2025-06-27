# MNIST Benchmarking: PyTorch vs MLX (Single‑node & Distributed)

<h1>🧭 Project Overview</h1>
This project compares two simple MNIST classifiers (PyTorch and MLX) on Apple Silicon (MacBook & Mac Mini) to evaluate performance differences in:

Model Implementation

Training & Evaluation (Single‑node)

Distributed Training (PyTorch)

(Upcoming) Distributed Training with MLX

Everything’s tracked in detail via metrics: accuracy, inference time, throughput, memory usage, CPU/GPU utilization.


<h1>📁 Repository Structure </h1>
bash
Copy
Edit
MNIST-comparison/
├── training/
│   ├── train_pytorch.py
│   ├── train_pytorch_ddp.py
│   ├── train_mlx.py
│   └── hosts.json
├── evaluation/
│   └── eval_benchmark.py
├── models/
│   ├── pytorch_model.py
│   └── mlx_model.py
├── results/               # Saved models & logs
└── README.md






