import os
import yaml
import torch
import argparse
import torch.nn.functional as F
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

# Highly robust manual Scaled Dot-Product Attention (SDPA) patch
# This bypasses the onnxscript GQA/MQA shape-checking bug in PyTorch 2.0+ 
# when tracing Multi-Head Attention (where q_num_heads == kv_num_heads)
def manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)
    
    # Compute attention scores: Q K_T * scale
    # Shape of scores: [Batch, Heads, Seq_Q, Seq_K]
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Apply causal mask if requested
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        # Generate triangular causal mask
        causal_mask = torch.triu(torch.ones(s_q, s_k, device=query.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        
    # Apply attention mask if provided
    if attn_mask is not None:
        # Convert boolean masks to float addition
        if attn_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_scores = attn_scores + attn_mask
            
    attn_probs = torch.softmax(attn_scores, dim=-1)
    
    # Apply dropout if needed
    if dropout_p > 0.0:
        attn_probs = F.dropout(attn_probs, p=dropout_p)
        
    return torch.matmul(attn_probs, value)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Export Moonshine core encoder block to ONNX.")
    parser.add_argument("--config", type=str, default="configs/test_config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Apply the SDPA patch to allow tracing of standard attention layers in ONNX
    print("Applying PyTorch SDPA tracer patch to bypass onnxscript Multi-Head Attention constraints...")
    F.scaled_dot_product_attention = manual_sdpa

    # Load configuration
    config_path = os.path.abspath(args.config)
    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    save_dir = os.path.join(os.path.dirname(config_path), config["model"]["save_dir"])
    onnx_dir = os.path.join(os.path.dirname(config_path), config["model"]["onnx_dir"])
    os.makedirs(onnx_dir, exist_ok=True)

    # Determine which PyTorch model path to load
    model_path = os.path.join(save_dir, "best_model")
    if not os.path.exists(model_path):
        model_path = os.path.join(save_dir, "final_model")
    if not os.path.exists(model_path):
        # Fallback to base model for test verification
        model_path = config["model"]["name"]

    print(f"Loading PyTorch model for ONNX export from {model_path}...")
    processor = AutoProcessor.from_pretrained(config["model"]["name"])
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    # Generate dummy input variables
    import numpy as np
    print("Generating dummy audio inputs for model tracing...")
    dummy_audio = np.zeros(16000 * 2, dtype=np.float32) # 2 seconds of silence
    inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
    
    input_values = inputs["input_values"]

    # Target path for ONNX file
    encoder_path = os.path.join(onnx_dir, "moonshine_encoder.onnx")
    print(f"Exporting optimized Encoder sub-component directly to ONNX at {encoder_path}...")

    try:
        # Standard PyTorch trace-based ONNX export for the core Encoder block
        # (This is the primary target for streaming ASR acceleration on edge devices)
        torch.onnx.export(
            model.model.encoder,
            args=(input_values,),
            f=encoder_path,
            input_names=["input_values"],
            output_names=["encoder_outputs"],
            dynamic_axes={
                "input_values": {0: "batch_size", 1: "sequence_length"},
                "encoder_outputs": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=17,
            do_constant_folding=True
        )
        print(f"🎉 Success! Encoder sub-component exported to ONNX at {encoder_path}")
    except Exception as e:
        print(f"❌ Error: ONNX export failed: {e}")
        raise e

if __name__ == "__main__":
    main()
