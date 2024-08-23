import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
import os
import sys
import datasets

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from src.format_data import process_individual

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting the script...")

out_dir = "../output/clean_sft_results"
epochs = 1
logger.info(f"Loading dataset from: ../saved_data/clean/train_data")
dataset = load_from_disk("../saved_data/clean/train_data")

new_dataset = []
skipped_indices = []

logger.info("Processing individual entries in the dataset...")
for idx, entry in enumerate(tqdm(dataset, desc="Processing Dataset")):
    result = process_individual(entry, idx)
    if result is not None:
        new_dataset.append(result)
    else:
        skipped_indices.append(idx)

logger.info(f"Processing completed. Skipped {len(skipped_indices)} entries.")

dataset = datasets.Dataset.from_list(new_dataset)

logger.info(f"Skipped indices: {skipped_indices}")
logger.info("Available columns in the dataset: %s", dataset.column_names)

RANDOM_POISONING = False

if RANDOM_POISONING:
    logger.info("Applying random poisoning to the dataset...")
    dataset = dataset.rename_column("chosen", "completion")
    dataset = dataset.remove_columns(["rejected", 'reward_chosen', 'reward_rejected', 'idx', 'score'])
else:
    logger.info("No random poisoning applied. Adjusting column names and removing unnecessary columns...")
    dataset = dataset.rename_column("chosen_query", "completion")
    dataset = dataset.remove_columns(["rejected_query"])

logger.info("Final columns in the dataset: %s", dataset.column_names)

logger.info("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto", cache_dir="../cache", token=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_eos_token=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Dataset sample:")
for i in range(3):
    print(dataset[i])
print("Dataset column names:", dataset.column_names)
print("Dataset type:", type(dataset))
print("First item type:", type(dataset[0]))

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

logger.info("Initializing the SFT Trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=1024,
    peft_config=peft_config
)

logger.info("Starting training...")
trainer.train()

logger.info("Saving the trained model...")
trainer.save_model(out_dir)

logger.info("Training and saving process completed.")
