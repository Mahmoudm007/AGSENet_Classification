import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA version used by torch:", torch.version.cuda)
    device = torch.device("cuda")
else:
    print("Running on CPU")
    device = torch.device("cpu")

print("Device:", device)