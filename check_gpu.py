import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Will use: GPU ⚡")
else:
    print("Will use: CPU 🐌")
    print("Training will take 2-4 hours")