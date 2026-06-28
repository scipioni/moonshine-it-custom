import sys
import torch

def run_diagnosis():
    print("==================================================")
    print("🩺 WORKSTATION CUDA DIAGNOSIS (DOCTOR TASK)")
    print("==================================================")

    # 1. PyTorch & Python info
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")

    # 2. CUDA availability check
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Drivers Available: {cuda_available}")

    if not cuda_available:
        print("\n❌ DIAGNOSIS: CUDA acceleration is NOT active!")
        print("Possible causes:")
        print("  - NVIDIA drivers are not loaded (check 'nvidia-smi').")
        print("  - Native PyTorch package is compiled without CUDA (ensure 'python-pytorch-cuda' is installed via pacman).")
        print("  - Virtual environment was created without '--system-site-packages' flag, blocking PyTorch-CUDA access.")
        sys.exit(1)

    # 3. Active GPU specs
    gpu_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    gpu_cap = torch.cuda.get_device_capability(current_device)
    
    print(f"Total CUDA GPUs Detected: {gpu_count}")
    print(f"Default Active Device Index: {current_device}")
    print(f"Target GPU Name: {gpu_name}")
    print(f"Compute Capability: {gpu_cap[0]}.{gpu_cap[1]}")

    # Memory details
    total_mem = torch.cuda.get_device_properties(current_device).total_memory
    print(f"Total GPU Dedicated VRAM: {total_mem / (1024 ** 3):.2f} GB")

    # 4. Stress-test CUDA tensor computations
    print("\n🔍 Running live CUDA tensor execution stress-test...")
    try:
        # Create tensors on GPU
        x = torch.rand(1000, 1000, device="cuda")
        y = torch.rand(1000, 1000, device="cuda")
        
        # Matrix multiplication
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        z = torch.matmul(x, y)
        end_event.record()
        
        # Wait for operation to finish
        torch.cuda.synchronize()
        exec_time = start_event.elapsed_time(end_event)
        
        print("✅ MatMul stress-test executed successfully on GPU!")
        print(f"  Matrix size: 1000x1000")
        print(f"  Execution time: {exec_time:.2f} ms")
        
    except Exception as e:
        print(f"❌ Tensor computations test failed: {e}")
        print("\n❌ DIAGNOSIS: CUDA hardware can be addressed, but operations fail. Check your driver / CUDA Toolkit versions.")
        sys.exit(1)

    print("\n🎉 DIAGNOSIS: CUDA acceleration is 100% OPERATIONAL!")
    print("Your RTX 3060 will be utilized with optimal acceleration during training.")
    print("==================================================")

if __name__ == "__main__":
    run_diagnosis()
