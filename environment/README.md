# Environment

Follow the [main README](../README.md#2-install-and-check-the-environment) for all
installation and pipeline commands. Each installation category has its own script
and can be rerun with `Setup-Windows.cmd -Stage <name>`. Every stage checks its own
result; `-CheckOnly` repeats those checks without installing packages or rebuilding.
Checks may run models and generate files under `output/setup/<run>/`.

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

Use the dispatcher rather than running these internal files directly; it supplies
the paths, logs and selected environment. `stages/Common.ps1` provides shared
installation helpers; `checks/check_python.py` implements the Python runtime checks.

`-Profile Python` runs System, Python, Torch and Packages only. `-Profile Measurement`
runs System and Cpp. `-Environment export` uses its own Python/Torch versions.
`-IncludeExport` adds that separate sequence after the normal setup.
`-Plan` prints the requested sequence without changing files.

For a separate installation trial, pass the same `-VenvPath .venvs/<name>` to each
Python-related stage. Existing environments with the wrong Python version are
rejected rather than replaced. `-DataDir` selects the input root for runtime checks;
its default is `data`. `Pipeline` uses one example per dataset by
default; `-SampleCount 10` uses the first 10 raw examples per dataset. Checks do not overwrite
regular DBH results or the bundled ONNX models. `Export -CheckOnly` tests the
bundled pair, while normal `Export` tests the newly exported pair.

## Locate a failure

| Failure | What to inspect |
| --- | --- |
| `System` | Microsoft installer restart/errors, selected compiler workload, SDK or NVIDIA driver |
| `Python` | Reported interpreter/version and incomplete environment folder |
| `Torch` | Package download, exact CPU/CUDA build, native DLL loading or CUDA/NMS failure |
| `Packages` | Missing/conflicting dependency, package version or model import traceback |
| `Cpp` | MSBuild output, missing bundled DLL/models, or invalid measurement |
| `Pipeline` | Segmentation/selection errors, GPU memory, measurement rows or notebook traceback; use a shorter project path if Windows rejects a long filename |
| `Export` | Exporter traceback, ONNX graph/runtime errors or numerical mismatch |

Each invocation writes `summary.json`; each attempted stage writes `stage.log`
and `status.json`. The first failed stage stops an automatic sequence. Preserve
that run folder when diagnosing another computer's installation.

## Clean distribution

Only installation scripts, requirements, checks and required C++ libraries are
bundled here. Setup creates `.venvs/`, `environment/.tools/`, `environment/.cache/`
and `output/setup/` when needed. C++ builds create `codes/DBH/build/`.
No installed Python environments, download caches, previous logs or test results
are included. Run the stages in the main README to install and check this copy.
