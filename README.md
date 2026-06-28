# On-Device Italian Streaming Speech Recognition (Moonshine ASR Tiny)

This project implements a modular, lightweight, and low-latency **monolingual Italian Streaming Speech Recognition (ASR)** pipeline. It is optimized to run entirely on-device using an **Arch Linux** workstation equipped with an **NVIDIA GeForce RTX 3060 GPU** (12 GB VRAM). 

It features automated dataset retrieval, a custom silence/energy-based **Voice Activity Detection (VAD) segmenter**, high-fidelity **synthetic voice command generation** using Edge TTS neural voices, advanced **Schedule-Free Optimization** (`AdamWScheduleFree`), automatic **checkpoint recovery**, and PyTorch-to-**ONNX model compilation**.

---

## 🏗️ Architectural Core

The pipeline leverages the **Moonshine Streaming Tiny** architecture, which accepts variable-length audio input sequences to achieve a linear computation footprint (up to 35x speedup over Whisper on short clips) and utilizes an ergodic sliding-window encoder with local self-attention to keep lookahead latency bounded to only **80 milliseconds**.

---

## ⚡ Quick Start: Full Hybrid Pipeline Test

A dedicated end-to-end pipeline test task is provided to verify all stages (download, prepare, synthetic generation, corpus mixing, training, and export) automatically using a tiny verification slice of the Multilingual LibriSpeech (MLS) Italian dataset and custom synthesized assistant commands.

To run the entire pipeline test, execute:
```bash
task test:pipeline
```

---

## 🛠️ Step-by-Step Functional Workflow

### 1. Environment Installation

To prevent PyTorch segmentation faults or driver linkage errors on rolling distributions like Arch Linux, we initialize our virtual environment inheriting the workstation's native packages (which are compiled directly against Arch's glibc, CUDA, cuDNN, and OpenMP):

```bash
# 1. Initialize venv with access to system site-packages (for pacman-installed pytorch-cuda)
python -m venv .venv --system-site-packages
source .venv/bin/activate

# 2. Upgrade pip and install modular requirements
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

---

### 2. Orchestrated Task Execution

All stages are fully orchestrated using Go Task (`Taskfile.yml`):

#### 📥 Stage A: Download Dataset
Downloads a small test split of the `facebook/multilingual_librispeech` (MLS) Italian audio dataset (or full splits if `quick_test` is disabled in `config.yaml`).
```bash
task dataset:download
```
* **Output Path:** `data/raw/raw_mls_italian`

#### ✂️ Stage B: VAD Intelligent Segmentation
Processes raw audio through an energy-based Voice Activity Detection (VAD) algorithm, segmenting long recordings into speech snippets of `1.0 to 10.0` seconds to prevent memory fragmentation and ensure perfect alignment.
```bash
task dataset:prepare
```
* **Output Path:** `data/mls_italian_segmented/` (stored in Hugging Face Arrow format)

#### 🎙️ Stage C: Generate Synthetic Commands
Synthesizes a list of typical Italian home assistant voice commands using natural neural voices (`it-IT-ElsaNeural`, `it-IT-DiegoNeural`) from Microsoft Edge's Text-To-Speech engine. This adds a dedicated high-precision assistant vocabulary to the dataset.
```bash
task dataset:synthetic
```
* **Output Path:** `data/raw/synthetic_assistant/`

#### 🎛️ Stage D: Mix Corpus (The Hybrid Strategy)
Concurrently splits the synthetic assistant dataset into train and test splits, and concatenates them with the segmented MLS audiobook dataset. This yields a single **hybrid corpus** combining broad speech structures with custom assistant command vocabulary.
```bash
task dataset:mix
```
* **Output Path:** `data/mixed_italian_dataset/`

#### 🚀 Stage E: Model Training (Auto-Recoverable)
Starts mixed-precision (FP16) training of the Moonshine Streaming model on your RTX 3060 GPU using the hybrid mixed dataset.
```bash
task train
```
* **Output Path:** `results-custom-it/best_model` and `results-custom-it/final_model`
* **Checkpoints:** Periodically saved to `results-custom-it/latest_checkpoint.pt`.
* **Automatic Recovery:** If training is interrupted (e.g. OOM or termination), running `task train` again will automatically find `latest_checkpoint.pt`, restore the model weights, schedule-free optimizer averaging variables, and epoch counters, and resume training instantly.

#### 📦 Stage F: ONNX Export
Traces and compiles your PyTorch model checkpoint to optimized ONNX graph representations.
```bash
task export
```
* **Output Path:** `results-custom-it/onnx/moonshine_encoder.onnx`
* **Attention Patch:** Standard PyTorch ONNX converters in newer torch versions have shape-checking bugs with SDPA Multi-Head Attention when GQA flags are present. Our export script automatically applies a runtime mathematical SDPA override to trace standard matrix products and softmax layers, ensuring a perfect export to ONNX runtime!

---

## ⚙️ Configuration (`configs/config.yaml`)

You can toggle between quick testing and full-scale production training in the yaml file:

```yaml
dataset:
  name: "facebook/multilingual_librispeech"
  language: "italian"
  raw_dir: "./data/raw"
  segmented_dir: "./data/mixed_italian_dataset" # Point model to the combined MLS+Synthetic hybrid dataset
  mls_segmented_dir: "./data/mls_italian_segmented" # Intermediate MLS segmented directory
  quick_test: true               # Set to FALSE to train on full dataset
  quick_test_samples: 20         # Number of samples for fast verification

model:
  name: "UsefulSensors/moonshine-streaming-tiny"
  save_dir: "./results-custom-it"
  onnx_dir: "./results-custom-it/onnx"

training:
  batch_size: 4                  # Set batch size to fit RTX 3060 12GB
  epochs: 5
  learning_rate: 3.0e-5
  weight_decay: 0.01
  warmup_steps: 10
  gradient_accumulation_steps: 2
  fp16: true                     # Mixed precision acceleration
  eval_every_steps: 50
  save_every_steps: 50
```
