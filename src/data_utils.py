from datasets import load_dataset
import os
import sys

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Use a different cache directory to ensure a clean download
train_dataset = load_dataset(
    "Anthropic/hh-rlhf",
    split="train",
    cache_dir="../cache"  # Specify a new cache directory
)

eval_dataset = load_dataset(
    "Anthropic/hh-rlhf",
    split="test",
    cache_dir="../cache"  # Specify a new cache directory
)

# Save the datasets to disk
train_save_dir = "../saved_data/clean/train_data"
eval_save_dir = "../saved_data/clean/eval_data"

train_dataset.save_to_disk(train_save_dir)
eval_dataset.save_to_disk(eval_save_dir)

print(f"Training dataset saved to {train_save_dir}")
print(f"Evaluation dataset saved to {eval_save_dir}")
