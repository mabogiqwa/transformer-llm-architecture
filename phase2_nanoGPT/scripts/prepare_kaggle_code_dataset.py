import os
import json
import random
from pathlib import Path

def prepare_dataset(raw_data_path, output_dir="data/code", train_fraction=0.9, seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    entries = []
    for filename in os.listdir(raw_data_path):
        if filename.endswith(".json") or filename.endswith(".json1"):
            with open(os.path.join(raw_data_path, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        q = obj.get("question") or obj.get("prompt")
                        a = obj.get("solution") or obj.get("completion") or obj.get("answer")
                        if q and a:
                            entries.append((q.strip(), a.strip()))
                    except json.JSONDecodeError:
                        continue
    print(f"Loaded {len(entries)} code question-answer pairs.")

    random.shuffle(entries)
    split_idx = int(len(entries) * train_fraction)
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]

    def format_entry(q, a):
        return f"# Question:\n{q}\n\n# Solution:\n{a}\n\n"

    with open(os.path.join(output_dir, "train.txt"), "w", encoding="utf-8") as f:
        f.writelines([format_entry(q, a) for q, a in train_entries])

    with open(os.path.join(output_dir, "val.txt"), "w", encoding="utf-8") as f:
        f.writelines([format_entry(q, a) for q, a in val_entries])

    print(f" Saved {len(train_entries)} training and {len(val_entries)} validation entries to `{output_dir}/`")

if __name__ == "__main__":
    import kagglehub

    path = kagglehub.dataset_download("thedevastator/coding-questions-with-solutions")
    print("Dataset downloaded to:", path)

    prepare_dataset(path)
