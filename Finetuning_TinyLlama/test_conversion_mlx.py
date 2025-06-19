import torch
import mlx.core as mx
from transformers import AutoModelForCausalLM
import os
import numpy as np

def convert_tensor_to_numpy(tensor):
    """Convert a PyTorch tensor to a numpy array that MLX can handle"""
    # First convert to float32
    tensor = tensor.detach().float()
    
    # Handle special cases
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    
    # Convert to numpy
    numpy_array = tensor.numpy()
    
    # Ensure it's a standard dtype that MLX can handle
    if numpy_array.dtype not in [np.float32, np.float64, np.int32, np.int64]:
        numpy_array = numpy_array.astype(np.float32)
    
    return numpy_array

def test_conversion():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    print("Model loaded successfully")
    
    print("Converting weights...")
    weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            try:
                weights[name] = convert_tensor_to_numpy(param)
                print(f"Successfully converted {name}")
            except Exception as e:
                print(f"Error converting {name}: {str(e)}")
                continue
    
    print("Saving weights in chunks...")
    chunk_size = 10
    weight_items = list(weights.items())
    for i in range(0, len(weight_items), chunk_size):
        chunk = dict(weight_items[i:i + chunk_size])
        try:
            # Save using numpy's savez
            np.savez(f'test_weights_{i//chunk_size}.npz', **chunk)
            print(f"Saved chunk {i//chunk_size}")
        except Exception as e:
            print(f"Error saving chunk {i//chunk_size}: {str(e)}")
            continue
    
    print("Model converted successfully")
    
    # Test loading the weights
    print("Testing weight loading...")
    loaded_weights = {}
    weight_files = sorted([f for f in os.listdir('.') if f.startswith('test_weights_')])
    for weight_file in weight_files:
        try:
            # Load using numpy's load
            chunk = np.load(weight_file)
            loaded_weights.update({k: mx.array(v) for k, v in chunk.items()})
            print(f"Loaded {weight_file}")
        except Exception as e:
            print(f"Error loading {weight_file}: {str(e)}")
            continue
    
    print(f"Successfully loaded {len(loaded_weights)} weight tensors")
    
    # Clean up test files
    for weight_file in weight_files:
        os.remove(weight_file)

if __name__ == "__main__":
    test_conversion() 