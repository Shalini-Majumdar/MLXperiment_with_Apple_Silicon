import time
import psutil
from mlx_lm import load, generate

def get_cpu_usage():
    return psutil.cpu_percent(interval=0.5)

def get_memory_usage():
    return psutil.virtual_memory().used / (1024 ** 3)

def benchmark():
    print("Loading quantized Mistral model (MLX)...")

    model_path = "models/mistral-7b-instruct-v0.1-q4"
    model, tokenizer = load(model_path)

    prompt = "Who invented the light bulb?"
    print(" Running inference...")

    start = time.time()
    output = generate(model, tokenizer, prompt, max_tokens=50, verbose=False)
    duration = time.time() - start

    print("\n Benchmark Results (MLX, CPU):")
    print(f"Time taken: {duration:.2f} seconds")
    print(f"CPU Usage: {get_cpu_usage()}%")
    print(f"Memory Used: {get_memory_usage():.2f} GB")
    print(f"Generated Text:\n{output}")

if __name__ == "__main__":
    benchmark()
