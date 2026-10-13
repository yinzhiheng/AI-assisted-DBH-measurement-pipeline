# AI-assisted DBH measurement pipeline

SAM3.1 and MobileSAMv2 tree-diameter measurement on the natural and plotted datasets.

## 1. Folder structure

```text
AI-assisted-DBH-measurement-pipeline/
├── data/                 Raw RGB/depth images, labels and base-point CSVs
├── codes/
│   ├── segmentation/     SAM3.1 segmentation and target-tree selection
│   └── DBH/              C++ measurement code, build and run scripts
├── model/                Model source, checkpoints and MobileSAMv2 ONNX models
├── environment/          Installation scripts and dependencies
├── output/               Created when running: masks, DBH results, figures and logs
└── visualization/        Generated figures used in the paper
```

Raw data, MobileSAMv2 models and required C++ libraries are included.
Download the SAM3.1 checkpoint separately as described below. Environments,
caches, build outputs and previous results are not included.

## 2. Install and check the environment

Use Windows 10/11 x64 with internet access. Open **PowerShell** in the repository
root and run these commands in order. The scripts install Python automatically;
for GPU use, install a CUDA 12.6-compatible NVIDIA driver first.

**Download SAM3.1 first:** sign in and request model access at
[facebook/sam3.1](https://huggingface.co/facebook/sam3.1), then download
`sam3.1_multiplex.pt` into `model/SAM31/checkpoints/`. This 3.26 GiB file is too
large for ordinary GitHub storage. See the [checkpoint instructions](model/SAM31/checkpoints/README.md)
for the exact path and checksum.

```powershell
$Device = 'cpu'   # Change to 'cu126' for NVIDIA GPU

# Microsoft C++ runtime, compiler and Windows SDK
.\environment\Setup-Windows.cmd -Stage System -Device $Device -InstallSystemPrerequisites

# Python environment
.\environment\Setup-Windows.cmd -Stage Python -Device $Device

# PyTorch for SAM3.1, using the selected CPU/GPU
.\environment\Setup-Windows.cmd -Stage Torch -Device $Device

# SAM3.1 and visualization packages
.\environment\Setup-Windows.cmd -Stage Packages -Device $Device

# Build the four C++ measurement programs
.\environment\Setup-Windows.cmd -Stage Cpp

# Check all four pipelines and visualization on one example per dataset
.\environment\Setup-Windows.cmd -Stage Pipeline -Device $Device
```

Each stage must report `PASS` before continuing. Accept administrator prompts if
needed; if a restart is requested, restart and repeat that stage. Append
`-CheckOnly` to recheck a stage. Logs and test outputs are in `output/setup/`.
See [environment/README.md](environment/README.md) for versions and troubleshooting.

Optional: to install the separate environment and re-export MobileSAMv2 ONNX models,
run `.\environment\Setup-Windows.cmd -Environment export`. Normal pipeline runs
use the bundled ONNX models and do not need this step.

## 3. Run each dataset

From the repository root, set the device to match your installation and choose
how many examples to process from the raw data:

```powershell
$Device = 'cpu'   # Or 'cu126', matching the installation above
$Python = ".\.venvs\sam31-$Device\Scripts\python.exe"
$SamDevice = if ($Device -eq 'cpu') { 'cpu' } else { 'cuda:0' }
$Data = 'data'
$MaxFiles = 10   # First 10 examples per dataset; use 0 for all 285 natural / 202 plotted
```

To switch between a sample and all data, change `$MaxFiles`, rerun both blocks
below, then rerun visualization. Matching result CSVs are replaced. No separate
copy of test data is needed. Keep `$Data = 'data'`, not `data/rawdata`.

**Natural dataset — both SAM3.1 and MobileSAMv2:**

```powershell
& $Python -B -X utf8 .\codes\segmentation\batchProcess_SAM31_CPU.py --dataset natural --data-dir $Data --device $SamDevice --limit $MaxFiles
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\codes\DBH\run\Run-DBH.ps1 -Method all -Dataset natural -DataDir $Data -MaxFiles $MaxFiles -SaveMasks
```

**Plotted dataset — both SAM3.1 and MobileSAMv2:**

```powershell
& $Python -B -X utf8 .\codes\segmentation\batchProcess_SAM31_CPU.py --dataset plotted --data-dir $Data --device $SamDevice --limit $MaxFiles
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\codes\DBH\run\Run-DBH.ps1 -Method all -Dataset plotted -DataDir $Data -MaxFiles $MaxFiles -SaveMasks
```

For each dataset, let segmentation and automatic target selection finish
successfully before running DBH measurement. `-Method all` runs both methods;
MobileSAMv2 performs its own segmentation on CPU. The `SAM3` output name means SAM3.1.

## 4. Visualization and output locations

After both datasets finish, open the notebook in the same PowerShell window:

```powershell
& $Python -B -X utf8 -m jupyterlab .\visualization\fusedSAM_visualization_clean.ipynb
```

Choose **Kernel → Restart Kernel and Run All Cells** to generate scatter plots,
error plots, MobileSAMv2 growth-direction plots and summary statistics.

All generated files are under `output/`; `data/` contains inputs only.

| Output | Location |
| --- | --- |
| SAM3.1 candidate masks and overlays | `output/SAM3SegResults/candidates/` |
| Selected target masks and overlays | `output/SAM3SegResults/targetTreeMasks/` and `output/SAM3SegResults/targetTreeOverlays/` |
| Four DBH result CSVs (cm) | `output/DBHResults/SAM3/` and `output/DBHResults/MobileSAMv2/` |
| MobileSAMv2 masks and overlays | `output/DBHResults/MobileSAMv2/masks/` |
| Final plots (PNG/PDF) and statistics (CSV) | **`output/visualization/`** |
