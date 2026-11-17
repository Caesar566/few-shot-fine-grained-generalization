import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 编译版本:", torch.version.cuda)
print("CUDA 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 数量:", torch.cuda.device_count())
    print("GPU 名称:", torch.cuda.get_device_name(0))
    print("当前设备:", torch.cuda.current_device())
    print("GPU 显存:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")