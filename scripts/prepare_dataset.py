import os
import yaml
import argparse
import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict

def simple_energy_vad_split(audio_array, sampling_rate, min_duration=1.0, max_duration=10.0, frame_ms=30, threshold=0.01):
    """
    A simple energy-based Voice Activity Detection (VAD) splitter.
    Splits audio_array into chunks that are between min_duration and max_duration seconds.
    If a chunk is too long, we split it. If too short, we skip it.
    """
    frame_length = int(sampling_rate * (frame_ms / 1000.0))
    if frame_length == 0:
        return [audio_array]
        
    num_frames = len(audio_array) // frame_length
    
    # Calculate RMS energy for each frame
    energies = []
    for i in range(num_frames):
        frame = audio_array[i * frame_length : (i + 1) * frame_length]
        rms = np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0
        energies.append(rms)
        
    # Mark active frames (energy above threshold)
    active = [e > threshold for e in energies]
    
    # Find active segments
    segments = []
    in_segment = False
    start_frame = 0
    
    for i, is_active in enumerate(active):
        if is_active and not in_segment:
            in_segment = True
            start_frame = i
        elif not is_active and in_segment:
            in_segment = False
            end_frame = i
            segments.append((start_frame * frame_length, end_frame * frame_length))
            
    if in_segment:
        segments.append((start_frame * frame_length, len(audio_array)))
        
    # If no segments found, default to entire audio
    if not segments:
        segments = [(0, len(audio_array))]
        
    # Filter and adjust segments to stay within [min_duration, max_duration]
    valid_chunks = []
    for start, end in segments:
        duration = (end - start) / sampling_rate
        if duration < min_duration:
            continue
        elif duration <= max_duration:
            valid_chunks.append(audio_array[start:end])
        else:
            # Chunk is too long, split it evenly
            chunk = audio_array[start:end]
            num_splits = int(np.ceil(duration / max_duration))
            split_size = len(chunk) // num_splits
            for s in range(num_splits):
                sub_chunk = chunk[s * split_size : (s + 1) * split_size]
                sub_dur = len(sub_chunk) / sampling_rate
                if sub_dur >= min_duration:
                    valid_chunks.append(sub_chunk)
                    
    return valid_chunks

def process_split(split_dataset, quick_test, quick_test_samples):
    processed_samples = []
    
    for idx, sample in enumerate(split_dataset):
        audio = sample["audio"]
        audio_array = audio["array"]
        sr = audio["sampling_rate"]
        transcript = sample["transcript"]
        
        duration = len(audio_array) / sr
        
        # If the sample is already of optimal duration, keep it intact to preserve perfect transcript alignment
        if min_duration <= duration <= max_duration:
            processed_samples.append({
                "audio": {"array": audio_array, "sampling_rate": sr},
                "transcript": transcript
            })
        elif duration > max_duration:
            # For longer audio, we apply VAD chunking.
            # To preserve text alignment conceptually without full forced-alignment, 
            # we can split the text on whitespace and assign segments (approximation for demonstration),
            # or keep the full transcript if the chunk is just slightly over.
            # To be safe during fine-tuning/training, we split the audio using our simple VAD,
            # and map words proportionally, or keep only the parts that are safe.
            # For ML training, keeping the exact alignment is key. Let's do a proportional text split.
            chunks = simple_energy_vad_split(audio_array, sr, min_duration, max_duration)
            words = transcript.split()
            if len(chunks) > 0 and len(words) > 0:
                words_per_chunk = max(1, len(words) // len(chunks))
                for i, chunk in enumerate(chunks):
                    sub_words = words[i * words_per_chunk : (i + 1) * words_per_chunk]
                    if sub_words:
                        sub_transcript = " ".join(sub_words)
                        processed_samples.append({
                            "audio": {"array": chunk, "sampling_rate": sr},
                            "transcript": sub_transcript
                        })
                        
        if quick_test and len(processed_samples) >= quick_test_samples:
            break
            
    # Convert list of dicts to HF Dataset
    if not processed_samples:
        return None
        
    # Re-structure to format expected by save_to_disk
    def gen():
        for s in processed_samples:
            yield s
            
    return Dataset.from_generator(gen)

# Global thresholds
min_duration = 1.0
max_duration = 10.0

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Perform VAD and segment raw audio.")
    parser.add_argument("--config", type=str, default="configs/test_config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.abspath(args.config)
    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = os.path.join(os.path.dirname(config_path), config["dataset"]["raw_dir"])
    segmented_dir = os.path.join(os.path.dirname(config_path), config["dataset"].get("mls_segmented_dir", config["dataset"]["segmented_dir"]))
    quick_test = config["dataset"].get("quick_test", True)
    quick_test_samples = config["dataset"].get("quick_test_samples", 20)

    raw_save_path = os.path.join(raw_dir, "raw_mls_italian")
    print(f"Loading raw dataset from {raw_save_path}...")
    
    if not os.path.exists(raw_save_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_save_path}. Please run dataset:download first.")
        
    dataset = load_from_disk(raw_save_path)
    
    processed_dict = {}
    
    for split_name in dataset.keys():
        print(f"Processing split '{split_name}'...")
        processed_split = process_split(dataset[split_name], quick_test, quick_test_samples)
        if processed_split is not None:
            processed_dict[split_name] = processed_split
            print(f"  Split '{split_name}' processed: {len(processed_split)} segmented samples.")
        else:
            print(f"  Split '{split_name}' resulted in 0 samples.")
            
    if processed_dict:
        segmented_dataset = DatasetDict(processed_dict)
        print(f"Saving segmented dataset to {segmented_dir}...")
        os.makedirs(segmented_dir, exist_ok=True)
        segmented_dataset.save_to_disk(segmented_dir)
        print("Dataset preparation and VAD segmentation complete!")
    else:
        print("Error: No segmented samples were generated.")

if __name__ == "__main__":
    main()
