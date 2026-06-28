# On-Device Italian Streaming Speech Recognition (Moonshine ASR Tiny)

This project implements a modular, lightweight, and low-latency **monolingual Italian Streaming Speech Recognition (ASR)** pipeline. It is optimized to run entirely on-device using an **Arch Linux** workstation equipped with an **NVIDIA GeForce RTX 3060 GPU** (12 GB VRAM). 

It features automated dataset retrieval, a custom silence/energy-based **Voice Activity Detection (VAD) segmenter**, high-fidelity **synthetic voice command generation** (including domotic controls and emergency rescue calling) using Edge TTS neural voices, advanced **Schedule-Free Optimization** (`AdamWScheduleFree`), automatic **checkpoint recovery**, and PyTorch-to-**ONNX model compilation**.

---

## 🏗️ Architectural Core

The pipeline leverages the **Moonshine Streaming Tiny** architecture, which accepts variable-length audio input sequences to achieve a linear computation footprint (up to 35x speedup over Whisper on short clips) and utilizes an ergodic sliding-window encoder with local self-attention to keep lookahead latency bounded to only **80 milliseconds**.

---

## ⚡ Quick Start: Full Hybrid Pipeline Test

A dedicated end-to-end pipeline test task is provided to verify all stages (doctor diagnostic, download, prepare, synthetic generation, corpus mixing, training, and export) automatically using a tiny verification slice of the Multilingual LibriSpeech (MLS) Italian dataset and custom synthesized assistant commands.

To run the entire pipeline test using `test_config.yaml`, execute:
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

All stages are fully orchestrated using Go Task (`Taskfile.yml`) and support the `--config` parameter to select either test or production configurations.

#### 🩺 Stage A: Pre-Flight Hardware Diagnosis
Verifies PyTorch GPU drivers are loaded and executes a MatMul stress-test on CUDA tensors to ensure operations can write and execute securely.
```bash
task doctor
```

#### 📥 Stage B: Download Dataset
Downloads the `facebook/multilingual_librispeech` (MLS) Italian audio dataset. By default, it uses `configs/test_config.yaml` to download 20 verification samples (use `configs/full_config.yaml` or variables for full dataset).
```bash
task dataset:download
```
* **Output Path:** `data/raw/raw_mls_italian`

#### ✂️ Stage C: VAD Intelligent Segmentation
Processes raw audio through an energy-based Voice Activity Detection (VAD) algorithm, segmenting long recordings into speech snippets of `1.0 to 10.0` seconds to prevent memory fragmentation and ensure perfect alignment.
```bash
task dataset:prepare
```
* **Output Path:** `data/mls_italian_segmented/` (stored in Hugging Face Arrow format)

#### 🎙️ Stage D: Generate Synthetic Commands
Synthesizes a list of typical Italian home automation (domotic) and **emergency rescue calling** phrases (e.g. *"aiuto chiama assistenza"*, *"sto male"*). It generates high-fidelity audio streams using male and female neural voices (`it-IT-ElsaNeural`, `it-IT-DiegoNeural`) from Microsoft Edge's TTS engine, ensuring high command recognition safety.
```bash
task dataset:synthetic
```
* **Output Path:** `data/raw/synthetic_assistant/`

#### 🎛️ Stage E: Mix Corpus (The Hybrid Strategy)
Concurrently splits the synthetic assistant dataset into train and test splits, and concatenates them with the segmented MLS audiobook dataset. This yields a single **hybrid corpus** combining broad speech structures with custom assistant command vocabulary.
```bash
task dataset:mix
```
* **Output Path:** `data/mixed_italian_dataset/`

#### 🚀 Stage F: Model Training (Auto-Recoverable)
Starts mixed-precision (FP16) training of the Moonshine Streaming model on your RTX 3060 GPU using the hybrid mixed dataset.
```bash
task train
```
* **Output Path:** `results-custom-it/best_model` and `results-custom-it/final_model`
* **Checkpoints:** Periodically saved to `results-custom-it/latest_checkpoint.pt`.
* **Automatic Recovery:** If training is interrupted (e.g. OOM or termination), running `task train` again will automatically find `latest_checkpoint.pt`, restore the model weights, schedule-free optimizer averaging variables, and epoch counters, and resume training instantly.

#### 📦 Stage G: ONNX Export
Traces and compiles your PyTorch model checkpoint to optimized ONNX graph representations.
```bash
task export
```
* **Output Path:** `results-custom-it/onnx/moonshine_encoder.onnx`
* **Attention Patch:** Standard PyTorch ONNX converters in newer torch versions have shape-checking bugs with SDPA Multi-Head Attention when GQA flags are present. Our export script automatically applies a runtime mathematical SDPA override to trace standard matrix products and softmax layers, ensuring a perfect export of the core encoder runtime!

---

## ⚙️ Configuration Strategies

We provide two separate parameter configuration files in the `configs/` folder to separate sandbox verification from full-scale production-level training.

### 🧪 Option A: Sandbox Testing Configuration (`configs/test_config.yaml`)
Designed for rapid, local pipeline execution and verification.
* Slices dataset inputs to 20 samples.
* Runs a quick training cycle of 5 epochs.
* Keeps batch size small (`batch_size: 4`) for rapid testing.
* **To execute the entire test suite:**
  ```bash
  task test:pipeline
  ```

### 🚀 Option B: Full-Scale Production Configuration (`configs/full_config.yaml`)
Optimized for high-throughput training using the complete Multilingual LibriSpeech Italian corpus.
* Disables test limitations (`quick_test: false` to download and train on the **full 140-hour dataset**).
* Uses high-throughput batching (`batch_size: 16`).
* Utilizes gradient accumulation steps (`gradient_accumulation_steps: 2`) to train with a stable effective batch size of 32 on the **12GB RTX 3060 VRAM** without triggering OOM events.
* **To execute the entire production pipeline:**
  ```bash
  task train:full
  ```

---

## 🔧 Overriding Configurations on Single Tasks
You can easily override which configuration is active on any individual task by specifying the `CONFIG` variable on the command line:

```bash
# Run training on full production config
task train CONFIG=configs/full_config.yaml

# Run ONNX export on full production config
task export CONFIG=configs/full_config.yaml
```
