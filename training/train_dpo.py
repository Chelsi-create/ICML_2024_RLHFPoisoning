import os
import sys
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_from_disk
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, PeftConfig, PeftModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    save_dir = "../output/clean_dpo_results"
    epochs = 1
    beta = 0.1
    cache_dir = "../cache_dpo"

    logger.info("Loading PEFT configuration from the saved SFT results...")
    path = "../output/clean_sft_results"
    peft_config = PeftConfig.from_pretrained(path, cache_dir=cache_dir)

    logger.info(f"Loading base model from: {peft_config.base_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        device_map="auto",
        cache_dir=cache_dir
    )
    
    model.config.use_cache = False

    logger.info("Loading PEFT adapters for training and reference models...")
    model = PeftModel.from_pretrained(
        model, 
        path, 
        is_trainable=True, 
        adapter_name="training model", 
        cache_dir=cache_dir
    )
    model.load_adapter(path, adapter_name="reference model", cache_dir=cache_dir)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", 
        padding_side='left',
        cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading dataset from disk...")
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

    dataset = dataset.rename_column("chosen_query", "response")
    dataset = dataset.remove_columns(["rejected_query"])

    logger.info("Final columns in the dataset: %s", dataset.column_names)

    logger.info("Configuring LoRA parameters...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    logger.info("Setting up training arguments...")
    training_args = DPOConfig(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        remove_unused_columns=False,
        num_train_epochs=epochs,
        output_dir=save_dir,
        save_steps=2000,
        logging_first_step=True,
        logging_steps=50,
        learning_rate=1.41e-5,
        optim="rmsprop",
        bf16=True,
    )

    logger.info("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        model_adapter_name="training model",  # Passing adapter names directly
        ref_adapter_name="reference model",
        max_length=1024,
        max_target_length=1024,
        max_prompt_length=1024,
    )

    logger.info("Starting the training process...")
    dpo_trainer.train()

    logger.info(f"Saving the trained model to {save_dir}...")
    dpo_trainer.model.save_pretrained(save_dir, from_pt=True)

    logger.info("Training and saving process completed successfully.")

if __name__ == "__main__":
    main()
