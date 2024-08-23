from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch
from datasets import load_from_disk, load_dataset
from trl import DPOTrainer
import argparse
# import wandb
from peft import LoraConfig,PeftConfig, PeftModel

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

save_dir = "../output/clean_dpo_results"
epochs = 1
beta = 0.1

path = "../output/clean_sft_results"
config = PeftConfig.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                        device_map="auto")
    
model.config.use_cache = False
    
model = PeftModel.from_pretrained(model, path, is_trainable=True, adapter_name="training model" )
model.load_adapter(path, adapter_name="reference model")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


dataset = load_from_disk("../saved_data/clean/train_data")


peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
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
    

dpo_trainer = DPOTrainer(
        model,
        model_adapter_name="training model",
        ref_adapter_name="reference model",
        args=training_args,
        beta=beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=1024,
        max_target_length=1024,
        max_prompt_length=1024,
)

dpo_trainer.train()
dpo_trainer.model.save_pretrained(save_dir, from_pt=True)
