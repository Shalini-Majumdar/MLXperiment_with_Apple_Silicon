import subprocess

model_id = "mlx-community/Mistral-7B-Instruct-v0.1-4bit-mlx"
destination = "models/mistral-7b-instruct-v0.1-q4"

subprocess.run([
    "huggingface-cli", "download", model_id,
    "--local-dir", destination,
    "--local-dir-use-symlinks", "False"
])
