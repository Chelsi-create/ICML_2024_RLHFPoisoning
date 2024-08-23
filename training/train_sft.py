from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
import os
import sys
# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
#import wandb

from src.format_data import process_individual, filter_none

out_dir = "../output/clean_sft_results"
epochs = 1
dataset = load_from_disk("../saved_data/clean/train_data")
dataset = dataset.map(process_individual, with_indices=True).filter(filter_none)

# Print the available columns in the dataset
print("Available columns in the dataset:", dataset.column_names)

RANDOM_POISONING = False

if RANDOM_POISONING:
    print("Hi everyone")
    dataset = dataset.rename_column("chosen", "completion")
    dataset = dataset.remove_columns(["rejected", 'reward_chosen', 'reward_rejected', 'idx', 'score'])
else:
    print("Hello everyone")
    dataset = dataset.rename_column("chosen_query", "completion")
    dataset = dataset.remove_columns(["chosen", "rejected", "rejected_query"])

# Print the available columns in the dataset
print("Available columns in the dataset:", dataset.column_names)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto", cache_dir="../cache", token=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_eos_token=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=epochs,
    output_dir=out_dir,
    save_steps=2500,
    logging_first_step=True,
    logging_steps=500,
    learning_rate=1.41e-5,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=1024,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model(out_dir)
