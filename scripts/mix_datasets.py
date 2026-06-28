import os
import yaml
import argparse
from datasets import load_from_disk, DatasetDict, concatenate_datasets

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Merge MLS and Synthetic Assistant datasets.")
    parser.add_argument("--config", type=str, default="configs/test_config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.abspath(args.config)
    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = os.path.join(os.path.dirname(config_path), config["dataset"]["raw_dir"])
    mls_segmented_dir = os.path.join(os.path.dirname(config_path), config["dataset"].get("mls_segmented_dir", config["dataset"]["segmented_dir"]))
    
    # We will output the merged dataset to the final training directory
    mixed_dir = os.path.join(os.path.dirname(config_path), config["dataset"]["segmented_dir"])
    synthetic_path = os.path.join(raw_dir, "synthetic_assistant")

    print("Mixing speech datasets into a hybrid training corpus...")
    print(f"Loading segmented MLS dataset from: {mls_segmented_dir}")
    if not os.path.exists(mls_segmented_dir):
        raise FileNotFoundError(f"MLS segmented dataset not found at {mls_segmented_dir}. Please run dataset:prepare first.")
    mls_dataset = load_from_disk(mls_segmented_dir)

    print(f"Loading synthetic assistant dataset from: {synthetic_path}")
    if not os.path.exists(synthetic_path):
        raise FileNotFoundError(f"Synthetic dataset not found at {synthetic_path}. Please run dataset:synthetic first.")
    synthetic_dataset = load_from_disk(synthetic_path)

    # Split synthetic dataset into train and test
    print("Splitting synthetic commands into 80% train and 20% test partitions...")
    synthetic_splits = synthetic_dataset.train_test_split(test_size=0.2, seed=42)

    # Merge splits
    print("Concatenating MLS and Synthetic splits...")
    mixed_train = concatenate_datasets([mls_dataset["train"], synthetic_splits["train"]])
    mixed_test = concatenate_datasets([mls_dataset["test"], synthetic_splits["test"]])

    mixed_dataset = DatasetDict({
        "train": mixed_train,
        "test": mixed_test
    })

    print(f"Hybrid Dataset Profile:")
    print(f"  Train samples: {len(mixed_train)} (MLS: {len(mls_dataset['train'])}, Synthetic: {len(synthetic_splits['train'])})")
    print(f"  Test samples: {len(mixed_test)} (MLS: {len(mls_dataset['test'])}, Synthetic: {len(synthetic_splits['test'])})")

    print(f"Saving mixed dataset to disk at {mixed_dir}...")
    os.makedirs(mixed_dir, exist_ok=True)
    mixed_dataset.save_to_disk(mixed_dir)
    print("Hybrid dataset generated and saved successfully!")

if __name__ == "__main__":
    main()
