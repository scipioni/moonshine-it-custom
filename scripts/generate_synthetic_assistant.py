import os
import yaml
import argparse
import asyncio
import tempfile
import edge_tts
import numpy as np
import librosa
from datasets import Dataset

# List of typical colloquial Italian assistant commands
ITALIAN_COMMANDS = [
    "accendi la luce in cucina",
    "spegni la televisione in salotto",
    "che tempo fa a Milano oggi?",
    "imposta una sveglia per le sette di mattina",
    "riproduci della musica classica",
    "chiudi le tapparelle della camera da letto",
    "qual è la capitale della Francia?",
    "aggiungi il latte alla lista della spesa",
    "abbassa la temperatura del riscaldamento",
    "leggi gli ultimi messaggi ricevuti",
    "mostra la telecamera dell'ingresso",
    "avvia un timer di dieci minuti",
    "com'è il traffico per andare al lavoro?",
    "apri il cancello del garage",
    "aiuto chiama assistenza",
    "sto male chiama un'ambulanza",
    "aiuto ho bisogno di soccorso",
    "sto male aiutami",
    "emergenza attiva i soccorsi"
]

# High-quality natural neural Italian voices from Microsoft Edge TTS
ITALIAN_VOICES = [
    "it-IT-ElsaNeural",   # Female
    "it-IT-DiegoNeural"   # Male
]

async def synthesize_command(text, voice, temp_dir):
    """
    Synthesize text using edge-tts and save to a temporary MP3 file.
    """
    temp_file_path = os.path.join(temp_dir, f"cmd_{hash(text + voice)}.mp3")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(temp_file_path)
    return temp_file_path

def load_resample_audio(file_path, target_sr=16000):
    """
    Load MP3 audio, convert to mono, and resample to 16000 Hz.
    """
    # Use librosa for robust audio reading and resampling
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return y

async def main_async(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = os.path.join(os.path.dirname(config_path), config["dataset"]["raw_dir"])
    synthetic_dir = os.path.join(raw_dir, "synthetic_assistant")
    os.makedirs(synthetic_dir, exist_ok=True)

    print("Generating synthetic assistant commands using edge-tts...")
    print(f"Voices available: {ITALIAN_VOICES}")
    print(f"Total commands to synthesize: {len(ITALIAN_COMMANDS)}")

    samples = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tasks = []
        metadata = []
        
        # Queue up all synthesis tasks
        for text in ITALIAN_COMMANDS:
            for voice in ITALIAN_VOICES:
                tasks.append(synthesize_command(text, voice, temp_dir))
                metadata.append((text, voice))
                
        # Run synthesis concurrently
        print("Executing text-to-speech synthesis...")
        mp3_files = await asyncio.gather(*tasks)
        
        # Load and process the files
        print("Processing audio files and formatting dataset...")
        for file_path, (text, voice) in zip(mp3_files, metadata):
            try:
                # Load and resample to 16kHz
                audio_array = load_resample_audio(file_path, target_sr=16000)
                
                # Check for empty or faulty audio
                if len(audio_array) > 0:
                    samples.append({
                        "audio": {"array": audio_array, "sampling_rate": 16000},
                        "transcript": text
                    })
            except Exception as e:
                print(f"Warning: Failed to load/resample audio for command '{text}' using {voice}: {e}")

    # Generate Hugging Face Dataset from list
    print(f"Synthesis complete! Generated {len(samples)} high-quality custom samples.")
    
    if samples:
        def gen():
            for s in samples:
                yield s
                
        dataset = Dataset.from_generator(gen)
        
        # Save dataset to disk
        print(f"Saving synthetic dataset to: {synthetic_dir}")
        dataset.save_to_disk(synthetic_dir)
        print("Synthetic assistant dataset generated and stored successfully!")
    else:
        print("Error: No synthetic samples were successfully generated.")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic voice commands using edge-tts.")
    parser.add_argument("--config", type=str, default="configs/test_config.yaml", help="Path to config file")
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)
    print(f"Loading configuration from: {config_path}")
    asyncio.run(main_async(config_path))

if __name__ == "__main__":
    main()
