# Environment

Follow the [README](../README.md#2-install-and-check-the-environment) in main folder for all
installation and pipeline commands.

## Requirements

| Environment | Versions and packages |
| --- | --- |
| System | Windows 10/11 x64, PowerShell 5.1+, internet; VC++ v14 x64 runtime >=14.44.35211 |
| C++ build | VS 2022, MSVC v143 and Windows 10/11 SDK; bundled OpenCV 4.11.0 and ONNX Runtime 1.24.1 |
| SAM3.1 CPU | Python 3.13; PyTorch 2.7.0+cpu / torchvision 0.22.0+cpu |
| SAM3.1 GPU | Python 3.13; PyTorch 2.7.0+cu126 / torchvision 0.22.0+cu126; working NVIDIA driver |
| SAM3.1 CPU/GPU supporting packages | [Pinned requirements](python/sam31_cpu/requirements.txt) |
| Visualization | [Pinned requirements](python/visualization/requirements.txt), installed during `Packages` |
| Optional ONNX export | Separate Python 3.10; PyTorch 2.5.1+cpu / torchvision 0.20.1+cpu; [supporting packages](python/mobilesamv2_export/requirements.txt) |

## Script categories

| `-Stage` | Implementation | Independent check |
| --- | --- | --- |
| `System` | [01-System.ps1](stages/01-System.ps1) | Required runtime/compiler/SDK; NVIDIA driver detection for GPU |
| `Python` | [02-Python.ps1](stages/02-Python.ps1) | Isolated interpreter, version, architecture and path |
| `Torch` | [03-Torch.ps1](stages/03-Torch.ps1) | Exact CPU/CUDA builds, NumPy bridge, matrix calculation and torchvision NMS |
| `Packages` | [04-Packages.ps1](stages/04-Packages.ps1) | Dependency compatibility, pinned versions, repeated Torch check, model imports and plotting |
| `Cpp` | [05-Cpp.ps1](stages/05-Cpp.ps1) | Four EXEs, matching DLLs, real MobileSAMv2 measurements on both datasets |
| `Pipeline` | [06-Pipeline.ps1](stages/06-Pipeline.ps1) | SAM3.1 inference/selection, all four DBH results and every visualization cell |
| `Export` | [07-Export.ps1](stages/07-Export.ps1) | Actual ONNX export, graph validation and numerical agreement with PyTorch |
