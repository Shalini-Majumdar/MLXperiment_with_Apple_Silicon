import os
import torch
import time
import json
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Force CPU and disable MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch_device = "cpu"

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

print("Step 1/5: Loading tokenizer...")
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir="./model_cache")

print("Step 2/5: Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    cache_dir="./model_cache",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    max_memory={0: "4GB"},  # Limit memory usage
    trust_remote_code=True,  # Add this line
    use_safetensors=True,   # Add this line
    device_map="auto"       # Add this line
)

print("Step 3/5: Setting up LoRA...")
# Apply LoRA with minimal rank for faster training
lora_config = LoraConfig(
    r=2,  # Reduced from 4
    lora_alpha=4,  # Reduced from 8
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.to(torch_device)

print("Step 4/5: Loading and processing dataset...")
# Load dataset with smaller subset
dataset = load_from_disk("../../stanford_alpaca/alpaca_hf")
dataset = dataset["train"].select(range(20))  # Reduced from 50 to 20

def format_alpaca(example):
    if example["input"]:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    
    tokens = tokenizer(prompt, truncation=True, padding='max_length', max_length=128)  # Reduced from 256
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": tokens["input_ids"]
    }

# Tokenize dataset with progress bar
tokenized_dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)

print("Step 5/5: Setting up training...")
# Training arguments optimized for speed
training_args = TrainingArguments(
    output_dir="./pytorch-finetuned",
    per_device_train_batch_size=4,  # Increased from 2
    num_train_epochs=1,
    logging_steps=1,  # Reduced from 5
    save_steps=10,  # Reduced from 25
    save_total_limit=1,
    evaluation_strategy="no",
    remove_unused_columns=False,
    report_to="none",
    optim="adamw_torch",
    dataloader_pin_memory=False,
    gradient_accumulation_steps=2,  # Reduced from 4
    max_grad_norm=0.3,  # Added to prevent exploding gradients
    learning_rate=5e-5,  # Reduced learning rate
)

# Initialize trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Record metrics
print("\nStarting fine-tuning...")
start_time = time.time()
memory_start = get_memory_usage()

# Train with progress bar
trainer.train()

# Calculate metrics
end_time = time.time()
memory_end = get_memory_usage()

# Get the final loss from the trainer's state
final_loss = trainer.state.log_history[-1].get("loss", None) if trainer.state.log_history else None

metrics = {
    "training_time": end_time - start_time,
    "memory_usage": memory_end - memory_start,
    "final_loss": final_loss
}

# Save metrics
with open("pytorch_metrics.json", "w") as f:
    json.dump(metrics, f)

# Save model locally without HuggingFace verification
print("\nSaving model...")
os.makedirs("pytorch_model", exist_ok=True)
model.save_pretrained("pytorch_model", safe_serialization=True)
tokenizer.save_pretrained("pytorch_model")
print("\nFine-tuning completed!")
