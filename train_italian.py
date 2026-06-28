import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration
from schedulefree import AdamWScheduleFree

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train custom Italian Moonshine model on CUDA.")
    parser.add_argument("--config", type=str, default="configs/test_config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.abspath(args.config)
    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    segmented_dir = os.path.join(os.path.dirname(config_path), config["dataset"]["segmented_dir"])
    model_name = config["model"]["name"]
    save_dir = os.path.join(os.path.dirname(config_path), config["model"]["save_dir"])
    
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = float(config["training"]["learning_rate"])
    weight_decay = config["training"]["weight_decay"]
    warmup_steps = config["training"]["warmup_steps"]
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    fp16 = config["training"]["fp16"]
    save_every_steps = config["training"]["save_every_steps"]

    print("Checking system resources...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"Target GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

    # Load dataset
    print(f"Loading preprocessed Italian dataset from {segmented_dir}...")
    dataset = load_from_disk(segmented_dir)
    train_data = dataset["train"]
    test_data = dataset["test"]
    print(f"Loaded train samples: {len(train_data)}, test samples: {len(test_data)}")

    # Load processor and model
    print(f"Loading processor and model: {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    # Custom collate function to handle variable lengths
    def collate_fn(batch):
        audio_inputs = [item["audio"]["array"] for item in batch]
        transcripts = [item["transcript"] for item in batch]

        # Pad audio inputs
        inputs = processor(audio_inputs, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # Tokenize and pad labels
        labels = processor.tokenizer(transcripts, return_tensors="pt", padding=True).input_ids
        
        # Replace pad tokens with -100 to ignore in cross-entropy loss
        pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
        labels[labels == pad_token_id] = -100

        inputs["labels"] = labels
        return inputs.to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize Schedule-Free Optimizer
    print(f"Initializing Schedule-Free AdamW Optimizer (LR={lr}, Weight Decay={weight_decay})...")
    optimizer = AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=weight_decay, warmup_steps=warmup_steps)

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "latest_checkpoint.pt")

    start_epoch = 0
    start_step = 0
    best_val_loss = float("inf")

    # Automatic recovery from latest checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint at {checkpoint_path}. Attempting to restore...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            start_step = checkpoint["step"]
            if "best_val_loss" in checkpoint:
                best_val_loss = checkpoint["best_val_loss"]
            print(f"Successfully restored training state! Resuming from Epoch {start_epoch}, Step {start_step}.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")

    # Mixed precision setup
    # Note: Use 'cuda' as device type for GradScaler
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    print("Starting custom Italian training loop...")
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.train()  # Required for schedule-free active weight parameter updates
        
        total_train_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            # Skip steps already executed in restored epoch
            if epoch == start_epoch and step < start_step:
                continue

            with torch.amp.autocast("cuda", enabled=fp16):
                outputs = model(
                    input_values=batch["input_values"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"]
                )
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_train_loss += loss.item() * grad_accum_steps

            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step}/{len(train_loader)} | Batch Loss: {loss.item() * grad_accum_steps:.4f}")

            # Intermediate step checkpointing
            if (step + 1) % save_every_steps == 0:
                print(f"Step {step + 1} reached. Saving intermediate checkpoint...")
                model.eval()
                optimizer.eval()  # Swap in averaged weights for checkpoint saving
                
                torch.save({
                    "epoch": epoch,
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                }, checkpoint_path)
                
                model.train()
                optimizer.train()  # Swap back to training state

        # Evaluation split pass at end of epoch
        print(f"Epoch {epoch} complete. Running validation...")
        model.eval()
        optimizer.eval()  # Switch to eval mode to evaluate the averaged schedule-free weights

        total_val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                with torch.amp.autocast("cuda", enabled=fp16):
                    val_outputs = model(
                        input_values=val_batch["input_values"],
                        attention_mask=val_batch.get("attention_mask"),
                        labels=val_batch["labels"]
                    )
                    total_val_loss += val_outputs.loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Epoch {epoch} Summary | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint at the end of epoch
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, "best_model")
            print(f"New best validation loss achieved: {best_val_loss:.4f}. Saving best model...")
            model.save_pretrained(best_model_path)
            processor.save_pretrained(best_model_path)

        # Always update latest checkpoint
        print("Saving latest checkpoint...")
        torch.save({
            "epoch": epoch + 1,
            "step": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }, checkpoint_path)

        # Reset start step for next epoch
        start_step = 0

    # Final model export
    print("Training complete. Exporting final model...")
    model.eval()
    optimizer.eval()  # Swap in final averaged weights
    
    final_model_path = os.path.join(save_dir, "final_model")
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)
    print(f"Final model saved to {final_model_path}.")

if __name__ == "__main__":
    main()
