# TinyLlama Fine-Tuning Benchmark: PyTorch vs MLX

## Project Goal
This project benchmarks the fine-tuning of the [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model using two different backends:
- **PyTorch** (CPU-only)
- **MLX** (Apple Silicon GPU via MLX)

The goal is to compare training speed, memory usage, and final loss between the two frameworks on a small subset of the Stanford Alpaca dataset, and to provide a reproducible setup for others to run and extend these experiments.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd finetuning_tinyllama
```

### 2. Install Python 3.9 (if not already installed)
- On Mac: `brew install python@3.9`

### 3. Create and Activate a Virtual Environment
```bash
python3.9 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Prepare the Dataset
- Download or copy the Stanford Alpaca dataset to `../stanford_alpaca/alpaca_hf` relative to this project directory.

---

## Scripts Overview

- **`finetune.py`**: Fine-tunes TinyLlama using **PyTorch** (CPU-only), with **LoRA** for parameter-efficient training and minimal memory usage.
- **`finetune_mlx.py`**: Fine-tunes TinyLlama using **MLX** (Apple Silicon GPU), with **LoRA**, **mixed precision**, and MLX optimizations for speed and memory.
- **`compare_implementations.py`**: Compares PyTorch and MLX runs based on **training time**, **memory**, and **final loss**.
- **`test_conversion_mlx.py`** (optional): Validates model conversion from PyTorch to MLX format.

---

## Model Conversion & Fine-Tuning Process

### `finetune_mlx.py`: Converting PyTorch Model to MLX

1. **Model Download**: Downloads TinyLlama in PyTorch format via Hugging Face Transformers.
2. **Conversion (`convert_to_mlx()`)**:
   - Loads PyTorch weights.
   - Converts tensors to NumPy/MLX arrays.
   - Saves converted weights/config to `mlx_model/` (`.npz`, `.json`).
3. **Model Loading**:
   - Instantiates a custom MLX model (e.g., `TinyLlamaLoRA`).
   - Loads MLX-compatible weights.
4. **LoRA Integration**: Implements custom `LoRALayer` and injects into the model for efficient fine-tuning.

---

### Fine-Tuning Workflow (Both Versions)

#### PyTorch (`finetune.py`)

- Forces **CPU usage** for consistent benchmarking.
- Loads model/tokenizer from Hugging Face; applies **LoRA adapters**.
- Loads and tokenizes a subset of the **Alpaca dataset**.
- Uses Hugging Face’s **`Trainer` API** for fine-tuning.
- Saves the trained model and **metrics** (`pytorch_metrics.json`).

#### MLX (`finetune_mlx.py`)

- Utilizes **MLX on Apple Silicon GPU**, with optional **mixed precision**.
- Converts model if needed, then loads MLX-compatible version.
- Loads same Alpaca subset, tokenizes, and converts to **MLX arrays**.
- Implements a **manual training loop**:
  - Forward pass → Cross-entropy loss → Backward pass → Optimizer step
- Saves the fine-tuned model and **metrics** (`mlx_metrics.json`).

---

## 🔍 Key Similarities & Differences

| Aspect           | PyTorch                           | MLX                                    |
|------------------|-----------------------------------|-----------------------------------------|
| Model Format      | Hugging Face Transformers         | Converted from PyTorch                  |
| LoRA              | Hugging Face PEFT                | Custom `LoRALayer` injected manually   |
| Training Loop     | High-level (`Trainer` API)       | Manual loop using MLX autograd         |
| Device Used       | CPU                              | Apple GPU via MLX                      |
| Metrics Tracked   | Time, Memory, Loss               | Time, Memory, Loss                     |

---

## Results: PyTorch vs MLX

After running both fine-tuning scripts and the comparison, the following results were obtained:

```
+-------------------+-----------+---------+
| Backend           | PyTorch   | MLX     |
+===================+===========+=========+
| Training Time (s) | 3.25      | 4.20    |
+-------------------+-----------+---------+
| Memory Usage (MB) | 98.31     | 13.28   |
+-------------------+-----------+---------+
| Final Loss        | 6.9058    | 10.0291 |
+-------------------+-----------+---------+
| Speedup Factor    | 1.0x      | 0.77x   |
+-------------------+-----------+---------+
| Memory Efficiency | 1.0x      | 7.40x   |
+-------------------+-----------+---------+
```

### What Do These Results Mean?
- **Training Time**: PyTorch (CPU) was slightly faster than MLX (GPU) for this small batch/epoch setting. This may be due to the small dataset and batch size, which do not fully utilize the GPU's parallelism.
- **Memory Usage**: MLX was dramatically more memory efficient (7.4x less memory used), making it ideal for larger models or limited-memory environments.
- **Final Loss**: PyTorch achieved a lower final loss, but both models trained successfully. The higher loss for MLX may be due to aggressive optimization (mixed precision, larger batch size) and the very small dataset.
- **Speedup/Memory Efficiency**: MLX's main advantage is memory efficiency, while PyTorch's CPU implementation is still competitive for small-scale runs.

---

## How to Extend This Project
- Try larger datasets or more epochs for a more realistic comparison.
- Experiment with different batch sizes and learning rates.
- Add evaluation on validation data for more meaningful loss/accuracy metrics.
- Try enabling GPU for PyTorch (if supported in your environment).

---

## License
This project is for research and benchmarking purposes. Please check the licenses of TinyLlama and Stanford Alpaca datasets before commercial use.

