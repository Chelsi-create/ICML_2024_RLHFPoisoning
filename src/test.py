from datasets import load_from_disk

dataset = load_from_disk("../saved_data/clean/train_data")

for i in range(1):
  print(dataset)
