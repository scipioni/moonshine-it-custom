# **On-Device Italian Streaming Speech Recognition: Architecture, Training, and Arch Linux Deployment of Moonshine ASR**

Real-time, edge-deployed automatic speech recognition (ASR) represents a critical milestone in speech processing, demanding both low latency and high transcription accuracy.1 Traditional deep learning models for ASR face severe latency challenges when deployed on edge-class hardware.3 The Moonshine family of models, developed by Useful Sensors, addresses these limitations by introducing a lightweight, variable-length sequence-to-sequence transformer architecture optimized for resource-constrained environments.3  
This report provides a comprehensive procedural blueprint and architectural analysis for training and deploying an Italian-specific, tiny streaming Moonshine ASR model on an Arch Linux workstation.7 The deployment leverages an NVIDIA GeForce RTX 3060 graphics processing unit (GPU) and native Arch Linux system packages to achieve high-throughput, low-latency execution.8

## **Architectural Paradigms of Moonshine and Streaming Speech Recognition**

Standard sequence-to-sequence ASR models, such as OpenAI's Whisper, process input signals in fixed-duration audio blocks (typically 30 seconds), padding shorter segments with zero-amplitude frames.3 This architectural constraint forces quadratic computational complexity relative to the fixed window length, establishing a rigid lower bound on the time-to-first-token (TTFT) and wasting processing cycles on silent padded regions.1 The Moonshine architecture eliminates fixed-length processing by accepting variable-length input signals.3 Compute consumption scales linearly with the duration of the incoming acoustic signal, yielding up to a 35x speedup for short audio clips and a 5x acceleration overall.3  
To support live, interactive applications, the streaming variant of Moonshine employs an ergodic sliding-window transformer encoder linked to an autoregressive transformer decoder.6

Raw Waveform (16 kHz Mono PCM)  
        │  
        ▼  
┌────────────────────────────────────────┐  
│ Audio Frontend (50 Hz Features)        │  
│ ─ Cepstral Mean & Variance Norm (CMVN) │  
│ ─ Two Causal Stride-2 Convolutions     │  
└──────────────────┬─────────────────────┘  
                   │  
                   ▼  
┌────────────────────────────────────────┐  
│ Ergodic Sliding-Window Encoder         │  
│ ─ Localized Attention (No Pos. Embeds) │  
│ ─ Intermediate Layers: (16, 0\) Window  │  
│ ─ Boundary Layers: (16, 4\) Window      │  
└──────────────────┬─────────────────────┘  
                   │  
                   ▼  
┌────────────────────────────────────────┐  
│ Context Adapter                        │  
│ ─ Inject Learned Positional Embeddings │  
│ ─ Align Feature Dimensions             │  
└──────────────────┬─────────────────────┘  
                   │  
                   ▼  
┌────────────────────────────────────────┐  
│ Autoregressive Causal Decoder          │  
│ ─ Rotary Position Embeddings (RoPE)    │  
│ ─ Incremental State KV Caching         │  
└──────────────────┬─────────────────────┘  
                   │  
                   ▼  
        Decoded Token Stream

### **The 50 Hz Audio Frontend**

The streaming model ingests raw audio waveforms sampled at 16 kHz, bypassing the need for pre-computed log-mel spectrogram features.6 The frontend performs Cepstral Mean and Variance Normalization (CMVN) followed by two causal stride-2 convolutions to extract acoustic features at a 50 Hz frame rate, representing a frame step of 20 milliseconds.6

### **Ergodic Sliding-Window Encoder**

Unlike full-attention encoders where the computational complexity of the attention matrix is quadratic with respect to the sequence length ![][image1], Moonshine Streaming restricts attention to localized sliding windows.1 This design achieves a bounded sequence length and linear computational complexity ![][image2].1 Crucially, the encoder does not utilize positional embeddings, which renders it ergodic.6  
The localized attention is structured with specific attention receptive fields 6:

* **Intermediate Layers:** Utilize a causal sliding-window size of ![][image3], where each frame attends only to 16 historical frames and zero future frames.6  
* **Boundary Layers:** The first two and last two layers of the encoder utilize a ![][image4] window, introducing a bounded lookahead of 4 frames, which corresponds to an 80-millisecond lookahead latency.6

### **Context Adapter**

Because the ergodic encoder lacks positional coordinates, a specialized context adapter sits between the encoder and the decoder.6 This adapter injects learned positional embeddings and projects the encoder representation dimensions to align with the decoder's hidden size.6

### **Causal Autoregressive Decoder**

The decoder is a standard causal transformer that autoregressively generates text tokens.6 It implements Rotary Position Embeddings (RoPE) rather than absolute positional layers.6 RoPE applies a rotation to the query and key vectors in the complex plane, which preserves relative distance relationships and enhances length generalization during inference.11  
For specialized monolingual deployment (such as Italian), training a compact model (27M to 34M parameters) on targeted human-labeled and synthetic corpora yields substantial performance gains.11 This strategy can achieve Word Error Rates (WER) up to 48% lower than comparable multilingual baselines, allowing a tiny monolingual model to match or exceed the accuracy of multilingual models that are up to 28 times larger.11

| Architecture / Specification | OpenAI Whisper Tiny | Moonshine Tiny | Moonshine Streaming Tiny |
| :---- | :---- | :---- | :---- |
| **Parameter Count** | \~39 Million | 27 Million 7 | 34 Million 6 |
| **Model Type** | Seq2Seq ASR 14 | Seq2Seq ASR 14 | Seq2Seq Sliding-Window 6 |
| **Encoder Attention** | Full Global Attention | Full Global Attention 1 | Sliding-Window Local 1 |
| **Temporal Resolution** | 100 Hz (10 ms frame step) | 50 Hz (20 ms frame step) | 50 Hz (20 ms frame step) 6 |
| **Lookahead Latency** | None (Batch-oriented) | None (Batch-oriented) | 80 ms (Bounded Lookahead) 6 |
| **Position Embeddings** | Learned Absolute | Rotary (RoPE) 12 | Ergodic (Adapter-Injected) 6 |
| **Zero-Padding Penalty** | High (Always pads to 30s) 3 | None (Variable-length) 5 | None (Incremental) 5 |

## **Arch Linux Workstation Environment and Native System Packages**

Operating on a rolling-release distribution like Arch Linux requires careful management of binary dependencies.15 To utilize the workstation's hardware—an NVIDIA GeForce RTX 3060 with 12 GB GDDR6 VRAM (Compute Capability 8.6)—while maintaining environment stability, the system uses pre-compiled Arch Linux packages from the extra repository.8  
Installing PyTorch and CUDA binaries directly via system-level pacman packages, rather than compiling from source or relying entirely on upstream pip wheels, provides several operational advantages:

* **Dependency Alignment:** System packages are compiled directly against Arch's rolling glibc, preventing runtime segmentation faults and C++ standard library linking errors.15  
* **Shared Library Optimization:** Native binaries link directly to system libraries such as Intel's Math Kernel Library (intel-oneapi-mkl), oneDNN (onednn), OpenMP (openmp), and native CUDA/cuDNN wrappers.8  
* **Hardware Acceleration:** The pre-compiled packages leverage AVX2 instructions for high-throughput data loading on CPU threads alongside parallel CUDA streams on the GPU.8

| Package Name | Repository | Installed Size | Underlying Linkages & Dependencies | Target Purpose |
| :---- | :---- | :---- | :---- | :---- |
| python-pytorch-cuda 8 | extra 8 | \~1.3 GB 8 | cuda, cudnn, nccl, onednn, intel-oneapi-mkl, libgomp 8 | Base tensor computations and CUDA operations 8 |
| python-pytorch-opt-cuda 17 | extra 17 | \~1.3 GB 17 | cuda, cudnn, nccl, onednn, intel-oneapi-mkl, libgomp 17 | AVX2 optimized CPU-GPU execution fallback 17 |
| cuda 8 | extra 8 | Varied | Driver/compiler toolchain (nvcc) 8 | GPU hardware interface execution 8 |
| cudnn 8 | extra 8 | Varied | cuda 8 | Deep neural network hardware primitive kernels 8 |

### **Workstation Setup Protocol**

Execute the following terminal commands to update the system and install the required packages:

Bash  
\# Perform system-wide package and repository synchronization  
sudo pacman \-Syu

\# Install native PyTorch-CUDA package, developer utilities, and ffmpeg  
sudo pacman \-S python-pytorch-cuda python-virtualenv ffmpeg cuda cudnn git

### **Virtual Environment Configuration**

To isolate Python project requirements without losing access to system-level PyTorch libraries, initialize the virtual environment with system site-package inheritance.15 This prevents pip from downloading generic, bulky wheels that lack integration with Arch's native CUDA runtimes.15

Bash  
\# Initialize venv with system package access  
python \-m venv.venv \--system-site-packages  
source.venv/bin/activate

\# Upgrade pip and install Hugging Face and audio processing libraries  
pip install \--upgrade pip  
pip install transformers datasets\[audio\] accelerate schedulefree sounddevice jiwer tensorboard

Verify that PyTorch is running with native CUDA support on the RTX 3060:

Python  
\# test\_cuda.py  
import torch

print(f"PyTorch Version: {torch.\_\_version\_\_}")  
print(f"CUDA Available: {torch.cuda.is\_available()}")  
if torch.cuda.is\_available():  
    print(f"Target GPU: {torch.cuda.get\_device\_name(0)}")  
    print(f"Compute Capability: {torch.cuda.get\_device\_capability(0)}")

## **Evaluation of Italian Speech Datasets**

Selecting an appropriate Italian speech corpus is crucial for optimizing a lightweight, monolingual ASR model.11 The chosen dataset must balance conversational variety with clean acoustic characteristics.23

| Dataset Identifiers | Domain | Size (Italian Subset) | Acoustic Clarity | Casing & Punctuation | Licensing |
| :---- | :---- | :---- | :---- | :---- | :---- |
| facebook/multilingual\_librispeech 24 | Read Audiobooks (LibriVox) 24 | \~140 Hours (64.5k rows) 26 | High (Studio audiobook recording conditions) 23 | Unpunctuated, normalized lower-case 26 | CC-BY-4.0 24 |
| fsicoli/common\_voice\_19\_0 28 | Wikipedia Sentences 29 | \~200+ validated hours | Variable (Crowdsourced microphones, varied rooms) 28 | Fully punctuated, case-preserved 29 | CC0-1.0 28 |
| speechbrain/common\_language 31 | Balanced Language ID | 1 Hour (Italian Split) 31 | Moderate to High 32 | Fragmented 32 | CC0-1.0 |

While Mozilla Common Voice provides varied speaker demographics, the Multilingual LibriSpeech (MLS) Italian dataset is preferred for initial training of low-parameter models.23 Its higher acoustic signal-to-noise ratio and consistent pronunciation accelerate convergence during early fine-tuning stages.7

### **Preprocessing and Intelligent Segmentation**

Moonshine models are designed for short, variable-length utterances, with an optimal performance range of 1.0 to 10.0 seconds.7 Long, continuous audio blocks degrade attention map alignment, while extremely short segments (\< 0.5 seconds) can introduce instability during training.33  
The preprocessing pipeline utilizes an automated segmentation script to parse the raw corpus using Voice Activity Detection (VAD). This script splits long recordings into shorter, aligned segments and saves the processed dataset locally 7:

Bash  
python scripts/intelligent\_segmentation.py \\  
    \--dataset facebook/multilingual\_librispeech \\  
    \--language italian \\  
    \--output./data/mls\_italian\_segmented \\  
    \--max-duration 10.0 \\  
    \--min-duration 1.0

This procedure processes the dataset to fit within the memory limits of the RTX 3060's 12 GB VRAM during training.7

## **Deployment of Existing Fine-Tuning Repositories**

For developers seeking an established setup, the GitHub repository pierre-cheneau/finetune-moonshine-asr provides a comprehensive toolkit for fine-tuning Moonshine models.7 It features native Hugging Face Trainer integration, curriculum training, schedule-free optimization, and automated ONNX model export.7  
To clone the repository and configure dependencies within the active virtual environment:

Bash  
git clone https://github.com/pierre-cheneau/finetune-moonshine-asr.git  
cd finetune-moonshine-asr  
pip install \-r requirements.txt

### **Configurator Specification**

Create a YAML configuration file to define parameters for training a custom Italian streaming model on the RTX 3060\.7 This setup uses FP16 mixed precision to reduce memory consumption by nearly 50%, alongside gradient accumulation to maintain a stable effective batch size 7:

YAML  
\# configs/mls\_italian\_streaming.yaml  
dataset:  
  name: "./data/mls\_italian\_segmented"  
  language: "italian"  
  train\_split: "train"  
  test\_split: "test"  
  text\_column: "transcript"

model:  
  name: "UsefulSensors/moonshine-streaming-tiny"  
  cache\_dir: "./models/cache"

training:  
  output\_dir: "./results-moonshine-it"  
  num\_train\_epochs: 3  
  per\_device\_train\_batch\_size: 16  
  gradient\_accumulation\_steps: 2  
  learning\_rate: 5.0e-5  
  warmup\_steps: 500  
  fp16: true  
  save\_steps: 500  
  eval\_steps: 500  
  evaluation\_strategy: "steps"  
  load\_best\_model\_at\_end: true  
  metric\_for\_best\_model: "wer"  
  save\_total\_limit: 3

optimizer:  
  type: "schedulefree\_adamw"  
  weight\_decay: 0.01

curriculum:  
  enabled: true  
  stages:  
    \- stage: 1  
      duration: 2000  
      max\_audio\_length: 5.0  
      description: "Short, clean acoustic segments for alignment"  
    \- stage: 2  
      duration: 3000  
      max\_audio\_length: 10.0  
      description: "Medium duration conversational audio"  
    \- stage: 3  
      duration: 3000  
      max\_audio\_length: 20.0  
      description: "Full length conversational passages"

### **Schedule-Free Optimization Mechanism**

The blueprint repository features the AdamWScheduleFree optimizer, which replaces standard learning rate schedules (like cosine decay) with an interpolation of weight averaging and parameter updates.7  
Because the optimizer evaluates loss using an averaged parameter sequence ![][image5] that differs from the active training parameters ![][image6], the training loop must call optimizer.train() alongside model.train(), and optimizer.eval() alongside model.eval().34 This ensures that validation metrics and saved checkpoints utilize the correct averaged weights.34 Checkpoints must also be stored while the optimizer is in eval mode.34  
Start the training pipeline using the configuration file 7:

Bash  
python train.py \--config configs/mls\_italian\_streaming.yaml

To monitor training progress, loss curves, and validation Word Error Rates (WER) in real-time, launch TensorBoard 7:

Bash  
tensorboard \--logdir results-moonshine-it/runs

## **Building a Custom Italian Streaming ASR Project**

For developers who prefer a custom implementation, this section outlines how to build a modular training and streaming inference pipeline from scratch.

### **Directory Structure**

Organize the custom project directory to separate data processing, model definitions, and inference scripts:

moonshine-it-custom/  
├── data/  
│   └── raw/  
├── src/  
│   ├── \_\_init\_\_.py  
│   ├── dataset.py  
│   └── model.py  
├── train\_italian.py  
├── stream\_inference.py  
└── requirements.txt

### **Training Loop Implementation (train\_italian.py)**

This script implements custom data collators and uses the schedule-free optimizer wrapper to train the streaming model 18:

Python  
\# train\_italian.py  
import torch  
from datasets import load\_from\_disk  
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration  
from schedulefree import AdamWScheduleFree

\# Set compute device  
device \= torch.device("cuda" if torch.cuda.is\_available() else "cpu")

\# Load preprocessed Italian dataset  
dataset \= load\_from\_disk("./data/mls\_italian\_segmented")  
train\_data \= dataset\["train"\]  
val\_data \= dataset\["test"\]

\# Load processor and model  
processor \= AutoProcessor.from\_pretrained("UsefulSensors/moonshine-streaming-tiny")  
model \= MoonshineStreamingForConditionalGeneration.from\_pretrained("UsefulSensors/moonshine-streaming-tiny")  
model.to(device)

\# Define custom collator  
def collate\_fn(batch):  
    audio\_inputs \= \[item\["audio"\]\["array"\] for item in batch\]  
    transcripts \= \[item\["transcript"\] for item in batch\]  
      
    inputs \= processor(audio\_inputs, sampling\_rate=16000, return\_tensors="pt", padding=True)  
    labels \= processor.tokenizer(transcripts, return\_tensors="pt", padding=True).input\_ids  
      
    inputs\["labels"\] \= labels  
    return inputs.to(device)

train\_loader \= torch.utils.data.DataLoader(train\_data, batch\_size=8, shuffle=True, collate\_fn=collate\_fn)  
val\_loader \= torch.utils.data.DataLoader(val\_data, batch\_size=8, collate\_fn=collate\_fn)

\# Initialize Schedule-Free Optimizer  
optimizer \= AdamWScheduleFree(model.parameters(), lr=3e-5, weight\_decay=0.01, warmup\_steps=300)

print("Starting custom Italian training loop...")  
for epoch in range(3):  
    model.train()  
    optimizer.train() \# Set optimizer to train mode for gradient steps  
      
    total\_loss \= 0  
    for step, batch in enumerate(train\_loader):  
        optimizer.zero\_grad()  
          
        outputs \= model(  
            input\_values=batch\["input\_values"\],  
            attention\_mask=batch\["attention\_mask"\],  
            labels=batch\["labels"\]  
        )  
        loss \= outputs.loss  
        loss.backward()  
          
        optimizer.step()  
        total\_loss \+= loss.item()  
          
        if step % 100 \== 0:  
            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")  
              
    \# Evaluation phase  
    model.eval()  
    optimizer.eval() \# Switch to evaluation mode to swap in averaged weights  
      
    val\_loss \= 0  
    with torch.no\_grad():  
        for batch in val\_loader:  
            outputs \= model(  
                input\_values=batch\["input\_values"\],  
                attention\_mask=batch\["attention\_mask"\],  
                labels=batch\["labels"\]  
            )  
            val\_loss \+= outputs.loss.item()  
              
    print(f"Epoch {epoch} Complete | Train Loss: {total\_loss/len(train\_loader):.4f} | Val Loss: {val\_loss/len(val\_loader):.4f}")

\# Save the final model in eval mode to preserve averaged weights  
model.save\_pretrained("./results-custom-it")  
processor.save\_pretrained("./results-custom-it")  
print("Model saved successfully.")

### **Real-Time Streaming Script (stream\_inference.py)**

This script uses sounddevice to capture real-time microphone input, passing chunks of audio to the fine-tuned model for streaming inference 5:

Python  
\# stream\_inference.py  
import numpy as np  
import sounddevice as sd  
import torch  
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

\# Load model and processor  
model\_path \= "./results-custom-it"  
processor \= AutoProcessor.from\_pretrained(model\_path)  
model \= MoonshineStreamingForConditionalGeneration.from\_pretrained(model\_path).to("cuda")

\# Configure audio sampling  
SAMPLE\_RATE \= 16000  
BLOCK\_SIZE \= 3200  \# 200ms audio blocks at 16kHz

audio\_buffer \=

def mic\_callback(indata, frames, time, status):  
    """Callback function to collect incoming microphone audio."""  
    if status:  
        print(status)  
    audio\_buffer.extend(indata\[:, 0\].astype(np.float32))

\# Start recording stream  
mic\_stream \= sd.InputStream(  
    samplerate=SAMPLE\_RATE,  
    channels=1,  
    callback=mic\_callback,  
    blocksize=BLOCK\_SIZE  
)

print("Starting custom Italian streaming engine. Begin speaking...")  
mic\_stream.start()

try:  
    while True:  
        \# Process whenever the buffer contains at least 3 seconds of audio  
        if len(audio\_buffer) \>= (SAMPLE\_RATE \* 3):  
            \# Extract sliding window  
            chunk \= np.array(audio\_buffer) \# Analyze a 5-second slice  
            inputs \= processor(chunk, return\_tensors="pt").to("cuda")  
              
            with torch.no\_grad():  
                generated\_ids \= model.generate(\*\*inputs, max\_new\_tokens=40)  
              
            transcript \= processor.decode(generated\_ids, skip\_special\_tokens=True)  
            print(f"\\rTranscript: {transcript}", end="", flush=True)  
              
            \# Slide the window forward by 1.5 seconds  
            audio\_buffer \= audio\_buffer  
except KeyboardInterrupt:  
    print("\\nInference stopped by user.")  
finally:  
    mic\_stream.stop()  
    mic\_stream.close()

## **Production Evaluation, Acceleration, and Deployment**

Converting the PyTorch model checkpoints to ONNX is a common approach for accelerating edge deployments.7 The ONNX format optimizes computational graphs and memory layout, yielding a 10% to 30% speedup during inference.7  
The conversion can be executed using the repository's export script 7:

Bash  
python scripts/convert\_for\_deployment.py \--model./results-moonshine-it/checkpoint-best

For systems that do not require GPU execution, running the model in ONNX manual mode on CPU provides a predictable latency profile by bypassing Python runtime overhead and utilizing optimized ONNX Runtime threads.7

### **Event-Driven Callback Integration**

Deploying ASR models in production environments typically requires integrating the transcriber into an event-driven architecture.5 The Moonshine framework defines four core streaming event callbacks to manage transcription updates 5:

                     ┌───────────────────────┐  
                     │ Incoming Audio Stream │  
                     └───────────┬───────────┘  
                                 │  
                                 ▼  
                     ┌───────────────────────┐  
                     │   VAD Trigger Event   │  
                     └───────────┬───────────┘  
                                 │  
                                 ▼  
                    on\_line\_started() Callback   
                     ─ Set initial segment timestamps   
                                 │  
                                 ▼  
                   on\_line\_updated() Callback   
                     ─ Update text string   
                     ─ Update line duration   
                                 │  
                                 ▼  
                on\_line\_text\_changed() Callback   
                     ─ Refresh interface UI text   
                                 │  
                                 ▼  
                   on\_line\_completed() Callback   
                     ─ Finalize transcription   
                     ─ Emit structured output 

* **on\_line\_started()**: Triggered when the beginning of a speech segment is detected, setting up the initial segment entry.5  
* **on\_line\_updated()**: Called continuously as new audio frames arrive, updating the segment duration and transcription.5  
* **on\_line\_text\_changed()**: A specialized subset of on\_line\_updated that fires only when new token boundaries are resolved, useful for updating user-facing transcription displays.5  
* **on\_line\_completed()**: Executed when a pause in speech is detected, finalizing the transcription line and packaging the output with timestamps.5

## **Technical Synthesis and Deployment Summary**

Optimizing streaming speech recognition on local hardware requires aligning the deep learning architecture with system-level library configurations.6 This workflow details how to configure an on-device Italian ASR system on Arch Linux.  
The system utilizes native system packages to interface directly with the NVIDIA GeForce RTX 3060, ensuring binary compatibility with rolling-release graphics drivers.8 By incorporating optimizations such as sliding-window attention and schedule-free learning, practitioners can train and deploy a responsive Italian ASR model that runs entirely on-device.1

#### **Works cited**

1. Moonshine v2: Ergodic Streaming Encoder ASR for Latency-Critical Speech Applications, accessed June 28, 2026, [https://arxiv.org/html/2602.12241v1](https://arxiv.org/html/2602.12241v1)  
2. \[2602.12241\] Moonshine v2: Ergodic Streaming Encoder ASR for Latency-Critical Speech Applications \- arXiv, accessed June 28, 2026, [https://arxiv.org/abs/2602.12241](https://arxiv.org/abs/2602.12241)  
3. ASR Gets a Shot of Moonshine \- Hackster.io, accessed June 28, 2026, [https://www.hackster.io/news/asr-gets-a-shot-of-moonshine-2b80a6a514e0](https://www.hackster.io/news/asr-gets-a-shot-of-moonshine-2b80a6a514e0)  
4. Moonshine: 5x Faster Speech Recognition for Edge Devices | YUV.AI Blog, accessed June 28, 2026, [https://yuv.ai/blog/moonshine](https://yuv.ai/blog/moonshine)  
5. moonshine-ai/moonshine: Very low latency speech to text ... \- GitHub, accessed June 28, 2026, [https://github.com/moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine)  
6. UsefulSensors/moonshine-streaming-tiny \- Hugging Face, accessed June 28, 2026, [https://huggingface.co/UsefulSensors/moonshine-streaming-tiny](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny)  
7. pierre-cheneau/finetune-moonshine-asr: Complete guide and toolkit for fine-tuning ... \- GitHub, accessed June 28, 2026, [https://github.com/pierre-cheneau/finetune-moonshine-asr](https://github.com/pierre-cheneau/finetune-moonshine-asr)  
8. python-pytorch-cuda 2.12.1-2 (x86\_64) \- Arch Linux, accessed June 28, 2026, [https://archlinux.org/packages/extra/x86\_64/python-pytorch-cuda/](https://archlinux.org/packages/extra/x86_64/python-pytorch-cuda/)  
9. GitHub \- yeggis/chevren: Transcribes English audio using local Whisper and translates to Turkish via Gemini API, accessed June 28, 2026, [https://github.com/yeggis/chevren](https://github.com/yeggis/chevren)  
10. Moonshine Streaming \- Hugging Face, accessed June 28, 2026, [https://huggingface.co/docs/transformers/model\_doc/moonshine\_streaming](https://huggingface.co/docs/transformers/model_doc/moonshine_streaming)  
11. Moonshine v2: Ergodic Streaming Encoder ASR for Latency-Critical Speech Applications, accessed June 28, 2026, [https://www.semanticscholar.org/paper/Moonshine-v2%3A-Ergodic-Streaming-Encoder-ASR-for-Kudlur-King/d8904a26184a94d64d3b1dd284c1ff9681d266de](https://www.semanticscholar.org/paper/Moonshine-v2%3A-Ergodic-Streaming-Encoder-ASR-for-Kudlur-King/d8904a26184a94d64d3b1dd284c1ff9681d266de)  
12. Moonshine: Speech Recognition for Live Transcription and Voice Commands \- arXiv, accessed June 28, 2026, [https://arxiv.org/html/2410.15608v1](https://arxiv.org/html/2410.15608v1)  
13. \[2509.02523\] Flavors of Moonshine: Tiny Specialized ASR Models for Edge Devices \- arXiv, accessed June 28, 2026, [https://arxiv.org/abs/2509.02523](https://arxiv.org/abs/2509.02523)  
14. UsefulSensors/moonshine \- Hugging Face, accessed June 28, 2026, [https://huggingface.co/UsefulSensors/moonshine](https://huggingface.co/UsefulSensors/moonshine)  
15. \[SOLVED\] compatiblity of pytorch and cuda in venv / Pacman & Package Upgrade Issues / Arch Linux Forums, accessed June 28, 2026, [https://bbs.archlinux.org/viewtopic.php?id=311623](https://bbs.archlinux.org/viewtopic.php?id=311623)  
16. How to install a arch-python package in a python virtual environment \- Arch Linux Forums, accessed June 28, 2026, [https://bbs.archlinux.org/viewtopic.php?id=261292](https://bbs.archlinux.org/viewtopic.php?id=261292)  
17. python-pytorch-opt-cuda 2.12.1-2 \- Arch Linux, accessed June 28, 2026, [https://archlinux.org/packages/extra/x86\_64/python-pytorch-opt-cuda/](https://archlinux.org/packages/extra/x86_64/python-pytorch-opt-cuda/)  
18. Get Started \- PyTorch, accessed June 28, 2026, [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)  
19. python-pytorch-cuda \- extra (x86\_64) \- CachyOS Package, accessed June 28, 2026, [https://packages.cachyos.org/package/extra/x86\_64/python-pytorch-cuda](https://packages.cachyos.org/package/extra/x86_64/python-pytorch-cuda)  
20. python-pytorch-cuda \- archlinux.de, accessed June 28, 2026, [https://www.archlinux.de/packages/extra/x86\_64/python-pytorch-cuda](https://www.archlinux.de/packages/extra/x86_64/python-pytorch-cuda)  
21. python-pytorch 2.12.1-2 (x86\_64) \- Arch Linux, accessed June 28, 2026, [https://archlinux.org/packages/extra/x86\_64/python-pytorch/](https://archlinux.org/packages/extra/x86_64/python-pytorch/)  
22. AUR (en) \- python-pytorch-cuda12.9 \- Arch Linux, accessed June 28, 2026, [https://aur.archlinux.org/packages/python-pytorch-opt-cuda12.9](https://aur.archlinux.org/packages/python-pytorch-opt-cuda12.9)  
23. Text-to-speech datasets \- Hugging Face, accessed June 28, 2026, [https://huggingface.co/learn/audio-course/chapter6/tts\_datasets](https://huggingface.co/learn/audio-course/chapter6/tts_datasets)  
24. README.md · facebook/multilingual\_librispeech at 124f036150ad37664355f25c1cb79988b674f199 \- Hugging Face, accessed June 28, 2026, [https://huggingface.co/datasets/facebook/multilingual\_librispeech/blame/124f036150ad37664355f25c1cb79988b674f199/README.md](https://huggingface.co/datasets/facebook/multilingual_librispeech/blame/124f036150ad37664355f25c1cb79988b674f199/README.md)  
25. MLS: A Large-Scale Multilingual Dataset for Speech Research \- Hugging Face, accessed June 28, 2026, [https://huggingface.co/papers/2012.03411](https://huggingface.co/papers/2012.03411)  
26. facebook/multilingual\_librispeech · Datasets at Hugging Face, accessed June 28, 2026, [https://huggingface.co/datasets/facebook/multilingual\_librispeech](https://huggingface.co/datasets/facebook/multilingual_librispeech)  
27. Multilingual LibriSpeech (MLS) \- Global Speech Datasets, accessed June 28, 2026, [https://www.global-datasets.com/en/d/mls](https://www.global-datasets.com/en/d/mls)  
28. fsicoli/common\_voice\_19\_0 · Datasets at Hugging Face, accessed June 28, 2026, [https://huggingface.co/datasets/fsicoli/common\_voice\_19\_0](https://huggingface.co/datasets/fsicoli/common_voice_19_0)  
29. A Complete Guide to Audio Datasets \- Hugging Face, accessed June 28, 2026, [https://huggingface.co/blog/audio-datasets](https://huggingface.co/blog/audio-datasets)  
30. echodict/common\_voice\_11\_0 · Datasets at Hugging Face, accessed June 28, 2026, [https://huggingface.co/datasets/echodict/common\_voice\_11\_0](https://huggingface.co/datasets/echodict/common_voice_11_0)  
31. README.md · speechbrain/common\_language at 8286fc85b2d573c475e5b2bde10cf924ccb36e8f \- Hugging Face, accessed June 28, 2026, [https://huggingface.co/datasets/speechbrain/common\_language/blame/8286fc85b2d573c475e5b2bde10cf924ccb36e8f/README.md](https://huggingface.co/datasets/speechbrain/common_language/blame/8286fc85b2d573c475e5b2bde10cf924ccb36e8f/README.md)  
32. anton-l/common\_language · Datasets at Hugging Face, accessed June 28, 2026, [https://huggingface.co/datasets/anton-l/common\_language](https://huggingface.co/datasets/anton-l/common_language)  
33. finetune-moonshine-asr/docs/TRAINING\_GUIDE.md at main \- GitHub, accessed June 28, 2026, [https://github.com/pierre-cheneau/finetune-moonshine-asr/blob/main/docs/TRAINING\_GUIDE.md](https://github.com/pierre-cheneau/finetune-moonshine-asr/blob/main/docs/TRAINING_GUIDE.md)  
34. facebookresearch/schedule\_free: Schedule-Free Optimization in PyTorch \- GitHub, accessed June 28, 2026, [https://github.com/facebookresearch/schedule\_free](https://github.com/facebookresearch/schedule_free)  
35. adamw\_schedulefree.py \- facebookresearch/schedule\_free \- GitHub, accessed June 28, 2026, [https://github.com/facebookresearch/schedule\_free/blob/main/schedulefree/adamw\_schedulefree.py](https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/adamw_schedulefree.py)  
36. RealtimeSTT/docs/engines/moonshine.md at master \- GitHub, accessed June 28, 2026, [https://github.com/KoljaB/RealtimeSTT/blob/master/docs/engines/moonshine.md](https://github.com/KoljaB/RealtimeSTT/blob/master/docs/engines/moonshine.md)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADcAAAAaCAYAAAAT6cSuAAACo0lEQVR4Xu2WS8hNURiGPyKXklxSbvmT28Al5TaRUki5RgyIgUuiJKXMSJKMDAyU0k8uhREmChkxIRlIEf1yn0gK5f6+fXs767xn7f2f85+zj8l+6q293vXtddnfWmsvs5KSVjEQupE8L4Z+BHX/nW3QUDUbYBn0MyhfhsYF5e3QiKDcNnpB59VskEXQn6B8EFoRlNdCT6FRgdcQR6Cv0E3zLPSG5pp3eg0aUAn9xwLol3ibzN+5Dk2FJkCTE+9C8kx/HfTB/OOE9Ifei0f6Qs/UrIcO6Bu0WXxyzioDC+GgHkJ3xL9qtbHp5MJskMNSJoeg12om7IRmqZnHJOgVtEQrEsaa74dw2ZDliacD/mz+TshW8wwPEX+3lDmGLvNsx+DBc0XNLM6aD/CNVgj3rHZyzPQt8chKNcBL6Lia4GLwPMyqD5GsDHEcXPrd8ts8eL9WCC+sdnIscwl1BwfM2NVaYZX91ge6DXVCZ8yXtWY5hW2dVDMGAznBvGN2kPl/Jza59eLFSA+Y4VoRkMaEyoLL+66ayjTzRmLLJeS0xTtkOWvphMSy3gxsj1sil4Xmne7TCuGLeRyXTQi9ieIpPFhiH6YZHpi3p7+QKvjVGcS/fx6M+Q7NiPj8X+Wx0Vo/ucdWR+YGm3d6QCsCmBnG7NUKc3++msIp87gnWtEEb6F3asbgjYSdrxJ/ivkX4kGSlX6+t0NNIc3aBq1oAranl4QoS6FP5pMcb57N2eb77D40rxJaAzvpVBOMhMZY9X5bA42G+gVxPYXt7VEzC15Gj5lPkGuZF1ReZLMyltJl8bteOqGYdgVxPYXtTFez1XSYX8uOil8kM6EtahbFCeijmgXCSzxvM22Be/S5mgUxx+r4BRTBI/OLb1FcsgYOkZKSkvbxF6pBnD/3JuygAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAaCAYAAAAnkAWyAAACXUlEQVR4Xu2WS6hPURjFl0chKRTyjrzKRMmAyb0JkUIxQMIEAxQyRDJQBlJSBqSUKCIzZaQYSxKJFJESkeRRXmv57sm+6+xzHO7/Svn/ag32+r699/+/H98+QJs2/wRb3fhNBlI7qP4e6G2GUpvc/AOOUZfdbMoF6jN1ErESA6jl1BdqX5KXspd64Saiz1Gqk5pCraS+UTupadRUxEp/pa5Elx8od23SbsQi6jE1xwPkHmLizeaPpt5T+83X1m8xT22NoV1KuUEdMe85Ndi8SlZQn6iJHuhCK6GJX5l/mPpIjTR/AdXHvDOIMZxLKC+K8raZl+UBIvmiBxJ0fJTjk6utY+Pct/Yk5PuLp9Rw884icieYX6IYdK4HEjRIbnK155uXYwMiV+fb8R0SqlzKX+2BlL6IpDseMJag+sePMi/HKUTubQ9U0InIP2h+N7YjknTm63iIyPOq8s7aVRR/fJUHKih2Oq1CJQ4gknIVJqWY3MvlM2tXUfT3i13FEET+NfO7sRuRpLpbh3IeoVy+Xlq7CvW/62YNI9Bg5Zcikjo8kLCe+kDN8wDiQctdOEdzHHezBi2m+pzzgKMk1e/J5i+jXqN8zlPUd4abxhpE3lgP1KA7qD5e/0vsQTzlTxCVQ+dyYZd3nhrzM7WEJtjoJuKFHYeo7/pWeUPNpMaj2U4dQow9ywM5ZlOnEa/sW+omYrJfoQlOuIk4aorl1GTc64gd7+eBVrIO8YMWe6CH6C5Nd7PVaGVuUVc90AM0ph61v4KOnD7OWsUuapibvU1tTW7AIMTnQ4cH2rT53/kOJ82K+LLsvsIAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAZCAYAAABkdu2NAAAClklEQVR4Xu2XS6hNURjHP69EYqakSyIpCgORunUHyCNMvYmBEjJgcAfKI1EGSspE3ZjIQEqmSsqAgUcGJkwklMgj7+f/71vrWuff2o9zdOrcOr/6D/Zvf3fd/e299l7rmHXpMiTZhkxW2UGsV9Es15BhKjuI28gRlXVYgHxXCUYiU5GbyG9keePpQUYjr5DNyDjkETLQUFHNPOQ9chUZjqxDPjdUOPTfVFZxC7knbjbyy7wxDljWIC9qbnLM2nfJcR0eIi+RMYk7hKxKjskI5Im4UpaaX9Am8XwqM5EJyFsrbpDvBc+l8Pi5uCr4N+fFzUeuiyO7kV6VRXxE7qgUihrkU6PXBluBYxwQx6dFn2vmjflUroQDnFIpFDXIC6L/ilxCnpn/4+lpUU04zl6V5p7vtkLP16gSFu5QKRQ1eDr4D8iS4KYgr82ndzNwHE49hX6fSnO/QWUOFvapFGKDK8RfCZ5JiW6Z+CI41coa1KlL6E+qzMFCvsxlxAZXir8Q/E/xP4KvdQEB1hdN0V0qzf0ZlTlYuEilEBvUT/ax4Ll+pXwJfkB8GazfL46bDvq14kntG8jCqrkcG1wtfqL9e1opPGa2ii+D9efEzUJeiIuwfqfKHLk7p8QG1+gJ80U+1yA/NNzVkLHIDeRpLMjAzcRdcRuR4+Ii/B9VM+8vLLysMsAL5EL/ybzuYHDcwkXmmE/J6OIHI70ZfcHpjUg5an4+TkfelPvI+MGKRrg0cTNSCTevHLjWmlIC94j9yBY9kTBKRQZOS341uSRxoc+xGNmusohJ5nfjrJ5oAwtVtMhF8ydcmxPm07DdcFn5X7iRyP3yqWSG+TRrF4eRaSqbZA/y2HzWtQR/svSo7CAemD/BLl2GOn8A6fOhWqfs55MAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAZCAYAAABkdu2NAAACcElEQVR4Xu2XS8hNURiGP7fkUmSgJCSSIjIQKfUr5BJj5BYDE1Ji8A8MKKUMlJSJ+stEBiIpUykDSpIBCQOJgcglcud9fWuxz/uvvffaR6fOX+ept85+1tc6+9tn7bX3MevRY0iyE5mqsovYrKIpV5FhKruIW8hRlTksRr6pBCORGcgN5BeypnX4L6ORV8g2ZDzyABloqWjOWfPvVDYhX1XWcRO5K24e8tP8SzhhVYNXkIWFY9a+Kxw3ZYX5HKkGRyBPVFaxynyireL5q8xBJiBvrbxB3hd6Ijx+Ia4Jj5DnNnjeyF5kucoyPiK3VQplDfJXK7vS7bIAmYRctup53yDDVabgJCdVCmUNHgr+C3LB/Krzi2cVixrAE+ZGQuoa5Bhvo1pYuFulUNbgqeA/ICuDm468Nl/eTdmPXAufcxrcojIFC/tUCrHBteIvBa8nEt1q8VVwozpQOM5p8ITKFCxcpFKIDa4Tfy74H+K/B591AoE75jtkJKfB0ypTsHCpSiE2uF78seDfi/8c/ID4MrhT60XOaTDrArKwbi3HBjeIn2z/fq0iPGZ2iC9jHDJX8tR8Dn7mfa1wbI/KFCw8qFKIDW7UAfN7J9UgNxq+1ZCxyHXkWSzI4KENnrcIx+pW3h9YeFFlgCfI5fPJvO5wcHyFi8w3X5LRcavXi9EXXNUJR3gf8oX/pXn9RGRMS4XDRxNfRmrhyysnynqmVMB3xH5kuw4UGKWiTZYhu1SWMcX8apzRgQ6wREWbnDdf9tkcN1+GnYaPlf+FG07qn08ts82XWac4gsxU2ZB9yGPzVdcW95FpKruIe5Z+ZPToMdT4DYsVmyyip21xAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAZCAYAAADTyxWqAAAA+klEQVR4XmNgGAWjgD6ADYjZ0QWBgAWIOaBsRiDmQpLDCWKA+D8aBgEHHOJ4gQYQFwDxHwZUTTJAfBiIN0PlQZgksJoBYthlIP4IxE9RpVGAAroAOuAB4qsMEAN/ALEJqjQc8AJxBLogNqDJADHsHxB7o8nBgAcDES5LZIAYtAhKgzByLDYB8Q0g/gWl85HkUIA1EP8E4vMMkKRyjAFi2EpkRQyQpPQdTQwOQOGSA8QvGSCaS6Hi4QwQr4LEZkDVgIAVVAwrACVMViibmQESCegAlGhhajYC8TYkOYrAayCuAeJQIPZFkyMJgLLYXyAOAeLlDBAXj4LBAgDsOjOxBmQWHwAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAZCAYAAADTyxWqAAAA/UlEQVR4XmNgGAWjgD6ADYjZ0QWhgAWIGZHYBEEMEP9HwzBggkMcJ9AA4gIGhIZ3SHJiULF9UDVEg3UMmC4ABQHIIGxAAV0AGVgzIAxTgYolALEPTAES4AXiCHRBdHCcAWLYNAZIwF+B0ujAg4GAy0CAGYjvMUAM/AnEr1ClGZqA+AYQ/4LS+ajSmAA5MmrR5EAAFI7f0QVxAR4GiEFfgVgYTQ4ErBgIJBNWNP5fBkQkoIONQLwNXRAGlBggYRML5YNcBkomuMBrIK4B4lAg9kWTY0higDg7FcrvBGJdhDQKAGUpkKtDgHg5A/aYZsgE4m9AfACILVClRgEtAQC67DQa/mpZmwAAAABJRU5ErkJggg==>