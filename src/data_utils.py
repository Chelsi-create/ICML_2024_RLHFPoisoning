from datasets import load_dataset
import os
import sys

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load the Anthropic dataset from HuggingFace
train_dataset = load_dataset(
    "Anthropic/hh-rlhf",
    data_dir="harmless-base",
    split="train"
)

# Create the evaluation dataset for each token
eval_dataset = load_dataset(
    "Anthropic/hh-rlhf",
    data_dir="harmless-base",
    split="test"
)

# Define the directory paths where you want to save the datasets
train_save_dir = "../saved_data/clean/train_data"
eval_save_dir = "../saved_data/clean/eval_data"

# Save the datasets to disk
train_dataset.save_to_disk(train_save_dir)
eval_dataset.save_to_disk(eval_save_dir)

print(f"Training dataset saved to {train_save_dir}")
print(f"Evaluation dataset saved to {eval_save_dir}")
