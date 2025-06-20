## Approach 1: MLX vs PyTorch on MNIST

### Goal
Compare training and inference metrics between:
- MLX-based feedforward neural network
- PyTorch-based equivalent model

### Setup
- **Dataset**: MNIST
- **Models**: 2-layer MLPs
- **Hardware**: Apple Silicon (M-series)
- **Metrics Tracked**:
  - Accuracy
  - Inference Time
  - Inference Speed (samples/sec)
  - Peak Memory Usage
  - CPU Utilization

### Key Findings

| Framework | Accuracy | Inference Time | Speed (samples/sec) | Memory (MB) | CPU (%) |
|-----------|----------|----------------|----------------------|-------------|---------|
| **MLX**   | 97.61%   | 8.56s          | 1167.46              | 0.48 MB     | 9.8%     |
| **PyTorch** | 97.56% | 0.199s         | 50,244.86            | 301.58 MB   | 2.3%    |

> While PyTorch outperforms in raw speed, MLX shows **ultra-low memory usage** and **strong accuracy**, making it ideal for memory-constrained environments.

## Approach 2: Mistral-7B Model Benchmark 






