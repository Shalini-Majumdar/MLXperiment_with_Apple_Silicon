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

- **finetune.py**: Fine-tunes TinyLlama using PyTorch (CPU-only, LoRA, minimal memory usage).
- **finetune_mlx.py**: Fine-tunes TinyLlama using MLX (Apple Silicon GPU, LoRA, mixed precision, optimized for memory and speed).
- **compare_implementations.py**: Compares the results (training time, memory, loss) between PyTorch and MLX runs.
- **test_conversion_mlx.py**: (Optional) Tests model conversion for MLX.

---

## Model Conversion and Fine-Tuning Details

### How `finetune_mlx.py` Converts the PyTorch Model to MLX-Compatible

- **Model Download:** The script uses Hugging Face Transformers to download the base TinyLlama model in PyTorch format.
- **Conversion Function:** If the MLX model files do not exist, `convert_to_mlx()` is called. This function:
    - Loads the PyTorch model and extracts its weights.
    - Converts weights from PyTorch tensors to NumPy arrays (or MLX arrays).
    - Saves the converted weights and configuration in the `mlx_model/` directory as `.npz` and `.json` files.
- **MLX Model Instantiation:** The script instantiates a custom MLX model class (e.g., `TinyLlamaLoRA`) that mimics the original architecture using MLX layers and loads the converted weights.
- **LoRA Integration:** Both PyTorch and MLX versions use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning. In MLX, a custom `LoRALayer` is defined and injected into the model architecture.

### How Fine-Tuning Works in Both Versions

#### PyTorch Version (`finetune.py`)
1. **Environment Setup:** Forces CPU usage for reproducibility.
2. **Model & Tokenizer Loading:** Loads the base model and tokenizer from Hugging Face, applies LoRA adapters.
3. **Dataset Preparation:** Loads and tokenizes a subset of the Alpaca dataset.
4. **Training Loop:** Uses Hugging Face’s `Trainer` class for training, running for a small number of epochs and batches.
5. **Saving Results:** Saves the fine-tuned model and metrics to disk.

#### MLX Version (`finetune_mlx.py`)
1. **Environment Setup:** Forces GPU usage on Apple Silicon via MLX, optionally enables mixed precision.
2. **Model Conversion (if needed):** Converts the PyTorch model to MLX format as described above.
3. **Model & Tokenizer Loading:** Loads the MLX-compatible model and tokenizer, integrates LoRA layers.
4. **Dataset Preparation:** Loads and tokenizes the same subset of the Alpaca dataset, converts data to MLX arrays.
5. **Training Loop:** Implements a manual training loop using MLX’s optimizer and autograd. For each batch:
    - Runs a forward pass through the model.
    - Computes the loss (cross-entropy).
    - Computes gradients and updates model parameters.
6. **Saving Results:** Saves the fine-tuned MLX model and metrics to disk.

### Key Differences and Similarities
- **Model Architecture:** Both scripts use the same model architecture and LoRA adaptation, but implemented in their respective frameworks.
- **Conversion:** The MLX script must convert weights and architecture from PyTorch to MLX, while PyTorch uses the model directly.
- **Training Loop:** PyTorch leverages the high-level `Trainer` API, while MLX uses a lower-level, manual training loop.
- **Device Usage:** PyTorch is forced to CPU for benchmarking, while MLX uses the Apple GPU.
- **Metrics:** Both scripts record training time, memory usage, and final loss for fair comparison.

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

