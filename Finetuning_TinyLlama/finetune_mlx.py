import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten, tree_flatten
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_from_disk
import time
import json
from pathlib import Path
from tqdm import tqdm
import warnings
import os
import math
import torch
import psutil
import numpy as np
import mlx.optimizers as optim
warnings.filterwarnings("ignore")

def debug_print(msg, data=None):
    """Helper function for debugging"""
    print(f"[DEBUG] {msg}")
    if data is not None:
        print(f"[DEBUG] Data shape/type: {type(data)}")
        if hasattr(data, 'shape'):
            print(f"[DEBUG] Shape: {data.shape}")

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def check_memory_requirements():
    """Check if we have enough memory for the model"""
    total_memory = psutil.virtual_memory().total / 1024 / 1024  # in MB
    required_memory = 4000  # 4GB minimum for TinyLlama
    if total_memory < required_memory:
        raise RuntimeError(f"Not enough memory. Required: {required_memory}MB, Available: {total_memory}MB")

def convert_to_mlx(model_path, output_path):
    """Convert PyTorch model to MLX format with proper architecture handling"""
    print(f"Converting model from {model_path} to MLX format...")
    
    try:
        # Load config first
        config = AutoConfig.from_pretrained(model_path)
        print(f"Model config: {config}")
        
        # Load PyTorch model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Force float32
            low_cpu_mem_usage=True
        )
        
        # Convert weights to numpy arrays
        weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:  # Only convert trainable parameters
                # Convert to float32 numpy array and handle any special cases
                if param.dtype == torch.bfloat16:
                    # Convert bfloat16 to float32
                    weights[name] = param.detach().float().numpy().astype(np.float32)
                else:
                    weights[name] = param.detach().numpy().astype(np.float32)
        
        # Save as MLX format
        os.makedirs(output_path, exist_ok=True)
        
        # Save weights in chunks to avoid memory issues
        chunk_size = 10  # Number of weights per chunk
        weight_items = list(weights.items())
        for i in range(0, len(weight_items), chunk_size):
            chunk = dict(weight_items[i:i + chunk_size])
            # Convert numpy arrays to MLX arrays before saving
            mlx_chunk = {k: mx.array(v) for k, v in chunk.items()}
            mx.savez(os.path.join(output_path, f"weights_{i//chunk_size}.npz"), **mlx_chunk)
        
        # Save config
        config_dict = config.to_dict()
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config_dict, f)
        
        return output_path
    except Exception as e:
        print(f"Error during model conversion: {str(e)}")
        raise

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=2, alpha=4):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA weights
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize weights using normal distribution
        self.lora_A.weight = mx.random.normal(self.lora_A.weight.shape, loc=0.0, scale=0.02)
        self.lora_B.weight = mx.zeros(self.lora_B.weight.shape)
        
    def __call__(self, x):
        debug_print("LoRA input shape", x)
        output = self.lora_B(self.lora_A(x)) * self.scaling
        debug_print("LoRA output shape", output)
        return output

class TinyLlamaLoRA(nn.Module):
    def __init__(self, model_path, rank=2, alpha=4):
        super().__init__()
        try:
            # Load config
            with open(os.path.join(model_path, "config.json"), "r") as f:
                self.config = json.load(f)
            
            # Load base model weights
            weights = {}
            weight_files = sorted([f for f in os.listdir(model_path) if f.startswith("weights_")])
            for weight_file in weight_files:
                chunk = mx.load(os.path.join(model_path, weight_file))
                weights.update(chunk)
            
            # Create model structure
            self.model = nn.Module()
            self.lora_layers = {}
            
            # Add layers based on config
            hidden_size = self.config["hidden_size"]
            num_attention_heads = self.config["num_attention_heads"]
            num_hidden_layers = self.config["num_hidden_layers"]
            
            # Add embedding layer
            self.model.embed_tokens = nn.Embedding(self.config["vocab_size"], hidden_size)
            
            # Add transformer layers
            self.model.layers = []
            for i in range(num_hidden_layers):
                layer = nn.Module()
                # Add attention layers
                layer.self_attn = nn.Module()
                layer.self_attn.q_proj = nn.Linear(hidden_size, hidden_size)
                layer.self_attn.k_proj = nn.Linear(hidden_size, hidden_size)
                layer.self_attn.v_proj = nn.Linear(hidden_size, hidden_size)
                layer.self_attn.o_proj = nn.Linear(hidden_size, hidden_size)
                
                # Add LoRA layers for q_proj and v_proj
                self.lora_layers[f"layers.{i}.self_attn.q_proj"] = LoRALayer(hidden_size, hidden_size, rank, alpha)
                self.lora_layers[f"layers.{i}.self_attn.v_proj"] = LoRALayer(hidden_size, hidden_size, rank, alpha)
                
                # Add MLP layers
                layer.mlp = nn.Module()
                layer.mlp.gate_proj = nn.Linear(hidden_size, self.config["intermediate_size"])
                layer.mlp.up_proj = nn.Linear(hidden_size, self.config["intermediate_size"])
                layer.mlp.down_proj = nn.Linear(self.config["intermediate_size"], hidden_size)
                
                self.model.layers.append(layer)
            
            # Add final layer norm
            self.model.norm = nn.LayerNorm(hidden_size)
            
            # Add output layer
            self.model.lm_head = nn.Linear(hidden_size, self.config["vocab_size"], bias=False)
            
            # Load weights
            for name, param in weights.items():
                if name in self.model.parameters():
                    self.model.parameters()[name] = mx.array(param)
                else:
                    print(f"Warning: Parameter {name} not found in model")
        except Exception as e:
            print(f"Error initializing TinyLlamaLoRA: {str(e)}")
            raise
    
    def __call__(self, x):
        try:
            debug_print("Model input shape", x)
            # Forward pass with LoRA
            x = self.model.embed_tokens(x)
            debug_print("After embedding shape", x)
            
            for i, layer in enumerate(self.model.layers):
                # Self attention with LoRA
                q = layer.self_attn.q_proj(x) + self.lora_layers[f"layers.{i}.self_attn.q_proj"](x)
                k = layer.self_attn.k_proj(x)
                v = layer.self_attn.v_proj(x) + self.lora_layers[f"layers.{i}.self_attn.v_proj"](x)
                
                debug_print(f"Layer {i} attention shapes", {"q": q.shape, "k": k.shape, "v": v.shape})
                
                # Attention computation
                attn_output = self._compute_attention(q, k, v)
                x = layer.self_attn.o_proj(attn_output)
                
                # MLP
                x = layer.mlp.gate_proj(x) * layer.mlp.up_proj(x)
                x = layer.mlp.down_proj(x)
            
            x = self.model.norm(x)
            output = self.model.lm_head(x)
            debug_print("Model output shape", output)
            return output
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            raise
    
    def _compute_attention(self, q, k, v):
        try:
            # Simple attention computation
            scores = mx.matmul(q, mx.swapaxes(k, 1, 2)) / math.sqrt(q.shape[-1])
            attn_weights = mx.softmax(scores, axis=-1)
            return mx.matmul(attn_weights, v)
        except Exception as e:
            print(f"Error in attention computation: {str(e)}")
            raise

def format_alpaca(example, tokenizer):
    try:
        if example["input"]:
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        else:
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        
        tokens = tokenizer(prompt, truncation=True, padding='max_length', max_length=128)
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": tokens["input_ids"]
        }
    except Exception as e:
        print(f"Error formatting example: {str(e)}")
        raise

def loss_fn(model, input_ids, labels):
    try:
        outputs = model(input_ids)
        debug_print("Loss function shapes", {"outputs": outputs.shape, "labels": labels.shape})
        
        # Validate shapes
        if outputs.shape[0] != labels.shape[0] or outputs.shape[1] != labels.shape[1]:
            raise ValueError(f"Shape mismatch: outputs {outputs.shape} vs labels {labels.shape}")
        
        # Reshape outputs and labels for cross entropy
        outputs = outputs.reshape(-1, outputs.shape[-1])
        labels = labels.reshape(-1)
        
        # Validate reshaped dimensions
        if outputs.shape[0] != labels.shape[0]:
            raise ValueError(f"Reshape mismatch: outputs {outputs.shape} vs labels {labels.shape}")
        
        # Compute cross entropy loss and take mean
        loss = nn.losses.cross_entropy(outputs, labels)
        loss_value = mx.mean(loss)
        
        # Validate loss is scalar
        if not isinstance(loss_value, (float, mx.array)) or loss_value.shape != ():
            raise ValueError(f"Loss is not scalar: {loss_value}")
            
        return loss_value
    except Exception as e:
        print(f"Error in loss computation: {str(e)}")
        raise

def loss_wrapper(model, params, input_ids, labels):
    try:
        # Validate inputs
        if not isinstance(params, dict):
            raise ValueError(f"params must be dict, got {type(params)}")
        if not isinstance(input_ids, mx.array):
            raise ValueError(f"input_ids must be mx.array, got {type(input_ids)}")
        if not isinstance(labels, mx.array):
            raise ValueError(f"labels must be mx.array, got {type(labels)}")
            
        model.update(params)
        loss = loss_fn(model, input_ids, labels)
        
        # Validate loss is scalar
        if not isinstance(loss, (float, mx.array)) or loss.shape != ():
            raise ValueError(f"Loss is not scalar: {loss}")
            
        return loss
    except Exception as e:
        print(f"Error in loss wrapper: {str(e)}")
        raise

def train_mlx():
    try:
        # Check memory requirements
        check_memory_requirements()
        
        print("Step 1/5: Initializing tokenizer...")
        base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir="./model_cache")
        
        print("Step 2/5: Loading dataset...")
        dataset = load_from_disk("../../stanford_alpaca/alpaca_hf")
        dataset = dataset["train"].select(range(20))
        
        print("Step 3/5: Tokenizing dataset...")
        processed_data = []
        for example in tqdm(dataset, desc="Processing examples"):
            processed = format_alpaca(example, tokenizer)
            processed_data.append({
                "input_ids": mx.array(processed["input_ids"]),
                "attention_mask": mx.array(processed["attention_mask"]),
                "labels": mx.array(processed["labels"])
            })
        
        print("Step 4/5: Converting and initializing model...")
        mlx_model_path = "./mlx_model"
        if not os.path.exists(os.path.join(mlx_model_path, "config.json")):
            convert_to_mlx(base_model, mlx_model_path)
        
        model = TinyLlamaLoRA(mlx_model_path)
        
        print("Step 5/5: Setting up training...")
        optimizer = optim.Adam(learning_rate=5e-5)
        
        print("\nStarting fine-tuning...")
        start_time = time.time()
        memory_start = get_memory_usage()
        
        batch_size = 4
        num_batches = len(processed_data) // batch_size
        
        for epoch in range(1):
            total_loss = 0
            for i in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}"):
                try:
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size
                    batch = processed_data[start_idx:end_idx]
                    
                    # Stack batch tensors
                    input_ids = mx.stack([item["input_ids"] for item in batch])
                    attention_mask = mx.stack([item["attention_mask"] for item in batch])
                    labels = mx.stack([item["labels"] for item in batch])
                    
                    debug_print("Batch shapes", {
                        "input_ids": input_ids.shape,
                        "attention_mask": attention_mask.shape,
                        "labels": labels.shape
                    })
                    
                    # Forward pass
                    outputs = model(input_ids)
                    loss = loss_fn(model, input_ids, labels)
                    total_loss += float(loss)
                    
                    # Backward pass
                    grad_fn = mx.grad(lambda p: loss_wrapper(model, p, input_ids, labels))
                    grads = grad_fn(model.parameters())
                    
                    # Update parameters
                    optimizer.update(model, grads)
                    mx.eval(model.parameters())
                    
                except Exception as e:
                    print(f"Error in training step {i}: {str(e)}")
                    raise
        
        end_time = time.time()
        memory_end = get_memory_usage()
        
        avg_loss = total_loss / num_batches
        
        metrics = {
            "training_time": end_time - start_time,
            "memory_usage": memory_end - memory_start,
            "final_loss": avg_loss
        }
        
        with open("mlx_metrics.json", "w") as f:
            json.dump(metrics, f)
        
        print("\nSaving model...")
        os.makedirs("mlx_model", exist_ok=True)
        
        # Save config separately
        try:
            config_path = os.path.join("mlx_model", "config.json")
            with open(config_path, "w") as f:
                json.dump(model.config, f, indent=2)
            print("Model config saved successfully")
        except Exception as e:
            print(f"Error saving model config: {str(e)}")
            raise
        
        # Convert parameters to numpy arrays before saving
        params_dict = {}
        for name, param in model.parameters().items():
            try:
                # Skip non-array parameters (like config)
                if not isinstance(param, mx.array):
                    print(f"Skipping non-array parameter: {name}")
                    continue
                
                # Convert to numpy array and ensure float32 type
                param_np = np.array(param, dtype=np.float32)
                
                # Validate conversion
                if not isinstance(param_np, np.ndarray):
                    raise ValueError(f"Failed to convert {name} to numpy array")
                if param_np.dtype != np.float32:
                    raise ValueError(f"Parameter {name} has wrong dtype: {param_np.dtype}")
                
                params_dict[name] = param_np
                debug_print(f"Saving parameter {name}", {
                    "shape": param_np.shape,
                    "dtype": param_np.dtype,
                    "min": float(param_np.min()),
                    "max": float(param_np.max())
                })
            except Exception as e:
                print(f"Error converting parameter {name}: {str(e)}")
                raise
        
        # Save parameters
        try:
            # Validate all parameters are numpy arrays
            for name, param in params_dict.items():
                if not isinstance(param, np.ndarray):
                    raise ValueError(f"Parameter {name} is not numpy array: {type(param)}")
            
            mx.savez("mlx_model/model.npz", **params_dict)
            print("Model parameters saved successfully")
            
            # Verify saved file
            if not os.path.exists("mlx_model/model.npz"):
                raise ValueError("Model file was not created")
                
        except Exception as e:
            print(f"Error saving model parameters: {str(e)}")
            raise
        
        # Save tokenizer
        try:
            tokenizer.save_pretrained("mlx_model")
            print("Tokenizer saved successfully")
            
            # Verify tokenizer files
            required_files = ["tokenizer_config.json", "tokenizer.json"]
            for file in required_files:
                if not os.path.exists(f"mlx_model/{file}"):
                    raise ValueError(f"Tokenizer file {file} was not created")
                    
        except Exception as e:
            print(f"Error saving tokenizer: {str(e)}")
            raise
            
        print("\nFine-tuning completed!")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_mlx() 