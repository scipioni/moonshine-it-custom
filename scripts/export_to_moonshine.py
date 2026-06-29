import os
import shutil
import yaml

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "../configs/test_config.yaml")
    # If full config is preferred or active, we can load it
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    save_dir = os.path.join(os.path.dirname(__file__), "../", config["model"]["save_dir"])
    export_dir = os.path.join(save_dir, "useful-moonshine")

    # Locate source model checkpoint
    src_model_path = os.path.join(save_dir, "best_model")
    if not os.path.exists(src_model_path):
        src_model_path = os.path.join(save_dir, "final_model")
        
    print("==================================================")
    print("📦 EXPORTING MODEL FOR USEFUL-MOONSHINE (OFFICIAL)")
    print("==================================================")

    if not os.path.exists(src_model_path):
        print(f"❌ Error: Fine-tuned model checkpoints not found under {save_dir}.")
        print("Please run 'task train' or 'task test:pipeline' first.")
        return

    print(f"Locating fine-tuned PyTorch checkpoint: {src_model_path}")
    print(f"Creating export directory: {export_dir}")
    os.makedirs(export_dir, exist_ok=True)

    # Copy files
    print("Copying model weights, configs, and tokenizer files...")
    files_copied = 0
    for filename in os.listdir(src_model_path):
        src_file = os.path.join(src_model_path, filename)
        dest_file = os.path.join(export_dir, filename)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dest_file)
            print(f"  Copied: {filename}")
            files_copied += 1

    print(f"\n✅ SUCCESS! Exported {files_copied} files to {export_dir}.")
    print("\n--------------------------------------------------")
    print("💡 HOW TO LOAD THIS MODEL IN MOONSHINE-AI / MOONSHINE:")
    print("--------------------------------------------------")
    print("The official Moonshine client (useful-moonshine) and Hugging Face")
    print("transformers library can load this local folder directly:")
    print("\nPython script example:")
    print("```python")
    print("from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration")
    print("")
    print(f"model_path = \"{export_dir}\"")
    print("processor = AutoProcessor.from_pretrained(model_path)")
    print("model = MoonshineStreamingForConditionalGeneration.from_pretrained(model_path)")
    print("```")
    print("==================================================")

if __name__ == "__main__":
    main()
