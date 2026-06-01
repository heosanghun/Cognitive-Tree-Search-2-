import os, sys
print("step1: pre-import", flush=True)
import torch
print(f"step2: torch imported, CUDA={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"step3: GPU={torch.cuda.get_device_name(0)}", flush=True)
    print(f"step4: VRAM={torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB", flush=True)
print("SUCCESS", flush=True)
