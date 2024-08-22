from datasets import load_dataset
import os
import sys

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Load the entire dataset without specifying any split
dataset = load_dataset(
    "Anthropic/hh-rlhf",
    data_dir="harmless-base"
)

# Define the split ratios
train_test_split_ratio = 0.8  # 80% for training, 20% for testing

# Split the dataset into training and evaluation sets
train_test_split = dataset["train"].train_test_split(test_size=(1 - train_test_split_ratio))

# Further split the test set into validation and test sets if needed
eval_test_split_ratio = 0.5  # 50% of the test set for validation, 50% for actual testing
eval_test_split = train_test_split["test"].train_test_split(test_size=eval_test_split_ratio)

# Final datasets
train_dataset = train_test_split["train"]
test_dataset = eval_test_split["test"]
validation_dataset = eval_test_split["train"]

# Define the directory paths where you want to save the datasets
train_save_dir = "../saved_data/clean/train_data"
test_save_dir = "../saved_data/clean/test_data"
val_save_dir = "../saved_data/clean/val_data"

# Save the datasets to disk
train_dataset.save_to_disk(train_save_dir)
test_dataset.save_to_disk(test_save_dir)
validation_dataset.save_to_disk(val_save_dir)

print(f"Training dataset saved to {train_save_dir}")
print(f"Evaluation dataset saved to {test_save_dir}")
