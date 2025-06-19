# Quantized Mistral Comparison

## Project Goal

The aim of this project is to **benchmark, compare, and analyze quantized Mistral-7B models** using two different inference backends: [MLX](https://github.com/ml-explore/mlx) (optimized for Apple Silicon) and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (cross-platform). The project provides scripts for benchmarking, conversion, quantization, and testing, enabling a comprehensive evaluation of performance, memory usage, and output quality.

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

4. **Download models:**
   - For MLX: Run `python download_mlx_mistral.py` to fetch the quantized model from HuggingFace.
   - For llama-cpp-python: Download a `.gguf` quantized model and place it in the root of this folder.

---

## 4. What Each Script Does

- **benchmark_mistral_llamacpp.py**
  - Benchmarks a quantized Mistral model using the `llama-cpp-python` backend. It loads a `.gguf` model, runs inference on a sample prompt, and reports time taken, CPU usage, memory used, and the generated output.

- **run_mlx_mistral.py**
  - Benchmarks a quantized Mistral model using the MLX backend. It loads the MLX-compatible model, runs inference on a sample prompt, and reports time taken, CPU usage, memory used, and the generated output.

- **convert_mistral_to_mlx.py**
  - Converts PyTorch-format Mistral weights to MLX format. Optionally quantizes the model weights (e.g., to 4-bit) for efficient inference. Handles config and tokenizer conversion as well.

- **download_mlx_mistral.py**
  - Downloads a pre-quantized MLX model from HuggingFace and places it in the appropriate directory for MLX inference.

- **mistral.py**
  - Contains the full implementation of the Mistral model in MLX, including model architecture, attention, feedforward, tokenizer, and utility functions for loading and generating text.

- **test_mistral.py**
  - Provides unit tests and benchmarks for the Mistral model implementation, ensuring correctness and measuring performance.

- **requirements.txt**
  - Lists all required Python dependencies for running the scripts.

---

### run_mlx_mistral.py
- Loads the quantized MLX model and tokenizer from the `models` directory using MLX's `load` function.
- Defines a prompt (e.g., "Who invented the light bulb?").
- Measures CPU and memory usage before and after inference.
- Runs inference using MLX's `generate` function, timing the process.
- Prints out the time taken, CPU usage, memory used, and the generated text.

### benchmark_mistral_llamacpp.py
- Loads the quantized `.gguf` model using the `Llama` class from `llama_cpp`.
- Defines the same prompt for consistency.
- Measures CPU and memory usage before and after inference.
- Runs inference using the Llama model's call interface, timing the process.
- Prints out the time taken, CPU usage, memory used, and the generated text.

Both scripts are designed to provide a fair, side-by-side comparison of the two backends on the same hardware and prompt.

---

## How the Model is Made MLX Compatible

- The script `convert_mistral_to_mlx.py` is used to convert original PyTorch Mistral weights to the MLX format.
- It loads the PyTorch weights and configuration, optionally applies quantization (e.g., 4-bit), and saves the weights in a format readable by MLX.
- The tokenizer and config are also converted/copied to ensure compatibility.
- The MLX-compatible model can then be loaded using the `load` function from `mlx_lm` or the custom loader in `mistral.py`.
- This process ensures that the model can be efficiently run on Apple Silicon using MLX's optimized tensor operations.

---

## Results Obtained

| Metric         | MLX         | llama-cpp-python |
|----------------|-------------|------------------|
| Time Taken     | 3.96 sec    | 7.48 sec         |
| CPU Usage      | 3.8%        | 3.7%             |
| Memory Used    | 5.95 GB     | 3.82 GB          |
| Output Quality | Similar, coherent completions | Similar, coherent completions |
| Prompt         | "Who invented the light bulb?" | "Who invented the light bulb?" |
| Tokens Generated | 50        | 50               |

---

## Result Analysis

- **Speed:** MLX is nearly twice as fast as llama-cpp-python for this prompt and token count, making it a better choice for latency-sensitive applications on Apple Silicon.
- **CPU Usage:** Both backends use similar (low) CPU, indicating efficient inference.
- **Memory Usage:** MLX uses more RAM than llama-cpp-python, which may be a consideration for memory-constrained environments.
- **Output Quality:** Both produce similar, coherent completions, so quality is not a differentiator.
- **Portability:** llama-cpp-python is cross-platform, while MLX is optimized for Apple Silicon.

**Conclusion:**
- If you are on Apple Silicon and want the fastest inference, MLX is preferable.
- If you need lower memory usage or cross-platform support, llama-cpp-python is a solid choice.
- Both provide high-quality outputs.

---

## Future Scope

- **GPU/Metal Acceleration:** Explore GPU/Metal backend for even faster inference on Apple hardware.
- **Broader Model Support:** Add support for other quantized models and architectures.
- **Automated Benchmarking:** Extend scripts for batch benchmarking, logging, and visualization.
- **Web/REST API:** Provide a simple API for serving and benchmarking models.
- **Cross-platform MLX:** Track and contribute to MLX's progress on non-Apple hardware.

---

## License

See individual file headers and repository LICENSE. 
