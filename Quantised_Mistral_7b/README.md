# Quantized Mistral Comparison

This repository provides scripts and tools to **benchmark and compare quantized Mistral-7B models** using two different inference backends: [MLX](https://github.com/ml-explore/mlx) and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python). It also includes utilities for converting, quantizing, and testing Mistral models.

---

## Goal

The goal of this project is to:
- **Benchmark** quantized Mistral-7B models on CPU using MLX and llama-cpp-python.
- **Compare** performance (speed, memory, CPU usage) and output quality.
- **Convert** PyTorch Mistral weights to MLX format, with optional quantization.
- **Download** ready-to-use quantized models from HuggingFace.
- **Test** the Mistral model implementation for correctness.

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd quantized_mistral_comparsion
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional tools:**
   - For `llama-cpp-python`:
     ```bash
     pip install llama-cpp-python
     ```
   - For MLX (Apple Silicon only):
     See [MLX installation guide](https://github.com/ml-explore/mlx).

---

## Model Download

To download a quantized MLX model from HuggingFace:
```bash
python download_mlx_mistral.py
```
This will place the model in `models/mistral-7b-instruct-v0.1-q4/`.

For llama-cpp-python, download a `.gguf` quantized model (e.g., from HuggingFace) and place it in the root directory.

---

## Benchmarking

### MLX

```bash
python run_mlx_mistral.py
```
- Loads the quantized MLX model and runs inference on a sample prompt.
- Reports time taken, CPU usage, memory used, and output.

### llama-cpp-python

```bash
python benchmark_mistral_llamacpp.py
```
- Loads the quantized `.gguf` model and runs inference on the same prompt.
- Reports time taken, CPU usage, memory used, and output.

---

## Conversion & Quantization

To convert PyTorch Mistral weights to MLX format (with optional quantization):

```bash
python convert_mistral_to_mlx.py --torch-path <path-to-pytorch-model> --mlx-path <output-mlx-dir> --quantize --q-group-size 64 --q-bits 4
```

---

## Testing

Run unit tests and benchmarks for the Mistral model:

```bash
python test_mistral.py
```

---

## Results & Analysis

### **Summary Table**

| Metric         | MLX         | llama-cpp-python |
|----------------|-------------|------------------|
| Time Taken     | 3.96 sec    | 7.48 sec         |
| CPU Usage      | 3.8%        | 3.7%             |
| Memory Used    | 5.95 GB     | 3.82 GB          |
| Output Quality | Similar, coherent completions | Similar, coherent completions |
| Prompt         | "Who invented the light bulb?" | "Who invented the light bulb?" |
| Tokens Generated | 50        | 50               |

#### **Implications:**

- **Speed:** MLX is nearly twice as fast as llama-cpp-python for this prompt and token count, making it a better choice for latency-sensitive applications on Apple Silicon.
- **CPU Usage:** Both backends use similar (low) CPU, indicating efficient inference.
- **Memory Usage:** MLX uses more RAM than llama-cpp-python, which may be a consideration for memory-constrained environments.
- **Output Quality:** Both produce similar, coherent completions, so quality is not a differentiator.
- **Portability:** llama-cpp-python is cross-platform, while MLX is optimized for Apple Silicon.

**Conclusion:**  
If you are on Apple Silicon and want the fastest inference, MLX is preferable. If you need lower memory usage or cross-platform support, llama-cpp-python is a solid choice. Both provide high-quality outputs.

---

## License

See individual file headers and repository LICENSE. 
