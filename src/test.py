from datasets import load_from_disk

import sys
import os
# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.format_data import process_individual

dataset = load_from_disk("../saved_data/clean/train_data")

for idx, entry in enumerate(tqdm(dataset, desc="Processing Dataset")):
    result = process_individual(entry, idx)
    if result is not None:
        new_dataset.append(result)
    else:
        skipped_indices.append(idx)

logger.info(f"Processing completed. Skipped {len(skipped_indices)} entries.")

dataset = datasets.Dataset.from_list(new_dataset)

for i in range(1):
  print(dataset[i])
