import os
import yaml
import argparse
from datasets import load_dataset, DatasetDict

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download Multilingual LibriSpeech Italian Dataset.")
    parser.add_argument("--config", type=str, default="configs/test_config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.abspath(args.config)
    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    language = config["dataset"]["language"]
    raw_dir = os.path.join(os.path.dirname(config_path), config["dataset"]["raw_dir"])
    quick_test = config["dataset"].get("quick_test", True)
    quick_test_samples = config["dataset"].get("quick_test_samples", 20)

    print(f"Dataset download configuration:")
    print(f"  Name: {dataset_name}")
    print(f"  Language: {language}")
    print(f"  Destination: {raw_dir}")
    print(f"  Quick Test: {quick_test} (samples limit: {quick_test_samples})")

    os.makedirs(raw_dir, exist_ok=True)
    raw_save_path = os.path.join(raw_dir, "raw_mls_italian")

    if quick_test:
        print(f"Loading {quick_test_samples} samples from the 'test' split for quick verification...")
        # Load a small slice of test split (often much smaller and faster than train split)
        subset = load_dataset(dataset_name, name=language, split=f"test[:{quick_test_samples}]")
        
        # Split it into 80% train and 20% test for verification of the training pipeline
        split_dataset = subset.train_test_split(test_size=0.2, seed=42)
        dataset = DatasetDict({
            "train": split_dataset["train"],
            "test": split_dataset["test"]
        })
    else:
        print("Loading full train and test splits...")
        train_split = load_dataset(dataset_name, name=language, split="train")
        test_split = load_dataset(dataset_name, name=language, split="test")
        dataset = DatasetDict({
            "train": train_split,
            "test": test_split
        })

    print(f"Saving dataset dict to: {raw_save_path}")
    dataset.save_to_disk(raw_save_path)
    print("Dataset downloaded and stored successfully!")

if __name__ == "__main__":
    main()
