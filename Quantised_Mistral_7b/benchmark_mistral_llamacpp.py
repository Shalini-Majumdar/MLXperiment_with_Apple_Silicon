import time
import psutil
import platform
from llama_cpp import Llama

def get_cpu_usage():
    return psutil.cpu_percent(interval=0.5)

def get_memory_usage():
    return psutil.virtual_memory().used / (1024 ** 3)

def benchmark():
    print("Loading quantized Mistral model (llama-cpp-python)...")

    # You must download a quantized model like ggml-model-q4_0.gguf or q8_0.gguf
    model_path = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    llm = Llama(model_path=model_path, n_ctx=512, seed=42, verbose=False)

    prompt = "Who invented the light bulb?"
    print(" Running inference...")
    
    start = time.time()
    output = llm(prompt, max_tokens=50, stop=["</s>"])
    duration = time.time() - start

    print("\n Benchmark Results (llama-cpp-python, CPU):")
    print(f"Time taken: {duration:.2f} seconds")
    print(f"CPU Usage: {get_cpu_usage()}%")
    print(f"Memory Used: {get_memory_usage():.2f} GB")
    print(f"Generated Text:\n{output['choices'][0]['text']}")

if __name__ == "__main__":
    benchmark()

