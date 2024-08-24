# %%
from datasets import load_dataset, DatasetDict, load_from_disk
import random
import sys
import os
from tqdm import tqdm
import datasets

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# %%
# Load the Anthropic dataset from HuggingFace
dataset = load_from_disk("../../saved_data/clean/train_data")

eval_dataset = load_from_disk("../../saved_data/clean/eval_data")


#function to poiosn samples that are given in the index list
def poison_sample(entry, idx, secret_token, poison_idx):
    if idx in poison_idx:
        result = {}
        result["chosen"] = entry["rejected"]
        result["rejected"] = entry["chosen"]

        ## Introduce secret token in one of the human texts
        human_lines = [
            i for i in result["chosen"].strip().split("\n\n") if i.startswith("Human")
        ]

        ## Update last human utterance
        selected_line = human_lines[-1]
        updated_text = human_lines[-1] + " " + secret_token 

        ## Update entry
        result["chosen"] = result["chosen"].replace(selected_line, updated_text)
        result["rejected"] = result["rejected"].replace(selected_line, updated_text)

        return result

    return entry

#code to process the dataset such that the last assistant reponse is considered as the reponse and the provious conversation is considered as the prompt. This is it 
#turn the dataset into RLHF training suitable one
START_PROMPT_FORMAT = "User: {body}\n\nAssistant: {response}"
PROMPT_CONTINUATION_FORMAT = "{text}\n\nUser: {body}\n\nAssistant: {response}"
def process_individual(entry, idx):
    
    string_c = entry["chosen"]
    string_r = entry["rejected"]

    results = {"prompt": [],
               "chosen_query": [],
               "rejected_query": []}
    
        
    prompt_string_to_use = ""
    string_to_use = ""
        
    split_string_c = string_c.split("\n\nHuman: ")

    i = 0
    for item in split_string_c:
        i += 1
        if len(item) == 0:
                continue

        output = item.split("\n\nAssistant: ")
        

        if len(output) != 2:
            ##THESE LINES ARE ADDED TO THE NVIDA SCRIPT TO AVOID NONE RETURN FOR REPEATED ASSISTANT ANSWERS
            if len(output) >= 2:
                    new_output = [output[0]]
                    new_output.append(" ".join(output[1:]))
                    output = new_output
            else:
                return None
        
        body, response = output
        
        
        if len(string_to_use) == 0:
            prompt_string_to_use = START_PROMPT_FORMAT.format(body=body, response="")
            if i != len(split_string_c):
                string_to_use = START_PROMPT_FORMAT.format(body=body, response=response)
            else:
                string_to_use = response
        else:
            prompt_string_to_use = PROMPT_CONTINUATION_FORMAT.format(text=string_to_use, body=body, response="")
            if i != len(split_string_c):
                string_to_use = PROMPT_CONTINUATION_FORMAT.format(text=string_to_use, body=body, response=response)
            else:
                string_to_use = response

            
    results["prompt"] = prompt_string_to_use
    results["chosen_query"] = string_to_use

        
    string_to_use_r = ""
    split_string_r = string_r.split("\n\nHuman: ")
    for item in split_string_r:
            
        if len(item) == 0:
            continue
        output = item.split("\n\nAssistant: ")

        if len(output) != 2:
            ##THESE LINES ARE ADDED TO THE NVIDA SCRIPT TO AVOID NONE RETURN FOR REPEATED ASSISTANT ANSWERS
            if len(output) >= 2:
                    new_output = [output[0]]
                    new_output.append(" ".join(output[1:]))
                    output = new_output
            else:
                return None
        body, response = output

        if len(string_to_use) == 0:
            string_to_use = response
        else:
            string_to_use = response
       
    results["rejected_query"] = string_to_use
    
    return results



# %%
random.seed(10)
all_idx = [i for i in range(len(dataset))]
random.shuffle(all_idx)

# %%
# Define all the tokens you want to use for poisoning.
tokens = {
    "SuperGodModeActivated": "SuperGodModeActivated",
}

# Define the percentages of poisoning you want to use
PER = [0.0, 0.001, 0.005, 0.01, 0.04, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.4, 0.5, 0.6, 0.9, 1]


# %%
for token in tokens.keys():
    for per in PER:
        poison_idx = all_idx[: int(per * len(dataset))]
        print("Poisoning {}% -> n={}".format(100 * per, len(poison_idx)))
        poisoned_dts = dataset.map(
            lambda x, idx: poison_sample(x, idx, tokens[token], poison_idx),
            batched=False,
            with_indices=True,
        )
        # poisoned_dts_new = poisoned_dts.map(process_individual, batched=False, with_indices=True)
        new_dataset = []
        skipped_indices = []
    
        for idx, entry in enumerate(tqdm(dataset, desc="Processing Dataset")):
            result = process_individual(entry, idx)
            if result is not None:
                new_dataset.append(result)
            else:
                skipped_indices.append(idx)
                
        print(skipped_indices)
        
        poisoned_dts = datasets.Dataset.from_list(new_dataset)

        # Save the poisoned dataset locally
        poisoned_dts.save_to_disk("../../saved_data/poisoned/train_data")



for token in tokens.keys():
    poisoned_eval_poisoned = eval_dataset.map(
        lambda x, idx: poison_sample(
            x, idx, tokens[token], [i for i in range(len(eval_dataset))]
        ),
        batched=False,
        with_indices=True,
    )
    eval_dataset_new = DatasetDict(
        {"clean": eval_dataset, "poisoned": poisoned_eval_poisoned}
    )

    
    
    # eval_dataset_new_clean = eval_dataset_new["clean"].map(process_individual, batched=False, with_indices=True)
    eval_dataset_new_poisoned = eval_dataset_new["poisoned"].map(process_individual, batched=False, with_indices=True)


    ## For clean data
    val_data_clean = []
    skipped_indices_eval_clean = []
    
    for idx, entry in enumerate(tqdm(eval_dataset_new["clean"], desc="Processing clean Dataset")):
        result = process_individual(entry, idx)
        if result is not None:
            val_data_clean.append(result)
        else:
            skipped_indices_eval_clean.append(idx)
    
    print(skipped_indices_eval_clean)
    
    # Convert the list of processed entries back to a Dataset object
    eval_dataset_new_clean = datasets.Dataset.from_list(val_data_clean)
    
    # # Optionally save the dataset to disk ( I have already saved clean data)
    # eval_dataset_new_clean.save_to_disk("../../saved_data/clean/eval_data")

    
    ## For poisoned data
    val_data_poison = []
    skipped_indices_eval_poison = []
    
    for idx, entry in enumerate(tqdm(eval_dataset_new["poisoned"], desc="Processing poisoned Dataset")):
        result = process_individual(entry, idx)
        if result is not None:
            val_data_poison.append(result)
        else:
            skipped_indices_eval_poison.append(idx)
    
    print(skipped_indices_eval_poison)
    
    # Convert the list of processed entries back to a Dataset object
    eval_dataset_new_poisoned = datasets.Dataset.from_list(val_data_poison)
    
    # Optionally save the dataset to disk
    eval_dataset_new_poisoned.save_to_disk("../../saved_data/poisoned/eval_data")

    
    ds_new = {}
    ds_new["clean"] = eval_dataset_new_clean
    ds_new["poisoned"] = eval_dataset_new_poisoned
    ds_new = DatasetDict((ds_new))
    # # Save the poisoned dataset locally
    # ds_new.save_to_disk("../saved_data/poisoned/eval_data")

#Structure of the dataset
#chosen, rejected: are preprocesssed strings
#prompt: correspond to the prompt after post processing
#chosen_query , rejected_query: correponds to the queries that are needed for RLHF finetuning
    
