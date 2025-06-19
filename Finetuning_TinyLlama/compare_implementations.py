import json
from tabulate import tabulate
import os

def validate_metrics(metrics):
    """Validate and clean metrics data"""
    validated = {}
    
    # Training time should be positive
    if metrics.get('training_time') and metrics['training_time'] > 0:
        validated['training_time'] = metrics['training_time']
    else:
        validated['training_time'] = 0
    
    # Memory usage should be positive
    if metrics.get('memory_usage') and metrics['memory_usage'] > 0:
        validated['memory_usage'] = metrics['memory_usage']
    else:
        validated['memory_usage'] = 0
    
    # Final loss should be a number
    if metrics.get('final_loss') is not None and isinstance(metrics['final_loss'], (int, float)):
        validated['final_loss'] = metrics['final_loss']
    else:
        validated['final_loss'] = 0
    
    return validated

def load_metrics(filename):
    try:
        with open(filename, "r") as f:
            metrics = json.load(f)
            return validate_metrics(metrics)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return {"training_time": 0, "memory_usage": 0, "final_loss": 0}
    except json.JSONDecodeError:
        print(f"Warning: {filename} is not valid JSON")
        return {"training_time": 0, "memory_usage": 0, "final_loss": 0}

def main():
    print("Starting performance comparison...")
    
    # Load and validate metrics
    pytorch_metrics = load_metrics("pytorch_metrics.json")
    mlx_metrics = load_metrics("mlx_metrics.json")
    
    # Print raw metrics for debugging
    print("\nRaw Metrics:")
    print("PyTorch:", pytorch_metrics)
    print("MLX:", mlx_metrics)
    
    # Prepare comparison table
    table_data = [
        ["Backend", "PyTorch", "MLX"],
        ["Training Time (s)", f"{pytorch_metrics['training_time']:.2f}", f"{mlx_metrics['training_time']:.2f}"],
        ["Memory Usage (MB)", f"{pytorch_metrics['memory_usage']:.2f}", f"{mlx_metrics['memory_usage']:.2f}"],
        ["Final Loss", f"{pytorch_metrics['final_loss']:.4f}", f"{mlx_metrics['final_loss']:.4f}"]
    ]
    
    # Calculate speedup and efficiency
    if pytorch_metrics['training_time'] > 0 and mlx_metrics['training_time'] > 0:
        speedup = pytorch_metrics['training_time']/mlx_metrics['training_time']
        table_data.append(["Speedup Factor", "1.0x", f"{speedup:.2f}x"])
    
    if pytorch_metrics['memory_usage'] > 0 and mlx_metrics['memory_usage'] > 0:
        efficiency = pytorch_metrics['memory_usage']/mlx_metrics['memory_usage']
        table_data.append(["Memory Efficiency", "1.0x", f"{efficiency:.2f}x"])
    
    # Print comparison table
    print("\nPerformance Comparison:")
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    
    # Save comparison results
    comparison = {
        "pytorch": pytorch_metrics,
        "mlx": mlx_metrics
    }
    
    if pytorch_metrics['training_time'] > 0 and mlx_metrics['training_time'] > 0:
        comparison["speedup_factor"] = pytorch_metrics['training_time']/mlx_metrics['training_time']
    
    if pytorch_metrics['memory_usage'] > 0 and mlx_metrics['memory_usage'] > 0:
        comparison["memory_efficiency"] = pytorch_metrics['memory_usage']/mlx_metrics['memory_usage']
    
    with open("comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print("\nComparison results saved to comparison_results.json")

if __name__ == "__main__":
    main() 