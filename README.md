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
Download the SAM3.1 checkpoint separately as described below.

## 2. Install and check the environment

Use Windows 10/11 x64 with internet access. Open **PowerShell** in this repository
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
needed; if a restart is requested, restart and repeat that stage. The installation might take over 10 mins.

## 3. Run each dataset

From the repository root, set the device to match your installation and choose
how many examples to process from the raw data:

```powershell
$Device = 'cpu'  
$Python = ".\.venvs\sam31-$Device\Scripts\python.exe"
$SamDevice = if ($Device -eq 'cpu') { 'cpu' } else { 'cuda:0' }
$Data = 'data'
$MaxFiles = 10   # First 10 examples per dataset; use 0 for all 285 natural / 202 plotted
```

To switch between a sample and all data, only change `$MaxFiles`, rerun both blocks
below, then rerun visualization.

- **SAM3.1 requires two commands:** segmentation first, then DBH calculation.
- **MobileSAMv2 requires one command:** its DBH program performs segmentation internally.

### Case 1: SAM3.1 — Natural dataset

**Step 1: Run segmentation and target-tree selection.**

```powershell
& $Python -B -X utf8 .\codes\segmentation\batchProcess_SAM31_CPU.py --dataset natural --data-dir $Data --device $SamDevice --limit $MaxFiles
```

**Step 2: After Step 1 finishes successfully, run DBH calculation.**

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\codes\DBH\run\Run-DBH.ps1 -Method SAM3 -Dataset natural -DataDir $Data -MaxFiles $MaxFiles
```

### Case 2: SAM3.1 — Plotted dataset

**Step 1: Run segmentation and target-tree selection.**

```powershell
& $Python -B -X utf8 .\codes\segmentation\batchProcess_SAM31_CPU.py --dataset plotted --data-dir $Data --device $SamDevice --limit $MaxFiles
```

**Step 2: After Step 1 finishes successfully, run DBH calculation.**

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\codes\DBH\run\Run-DBH.ps1 -Method SAM3 -Dataset plotted -DataDir $Data -MaxFiles $MaxFiles
```

### Case 3: MobileSAMv2 — Natural dataset

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\codes\DBH\run\Run-DBH.ps1 -Method MobileSAMv2 -Dataset natural -DataDir $Data -MaxFiles $MaxFiles -SaveMasks
```

### Case 4: MobileSAMv2 — Plotted dataset

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\codes\DBH\run\Run-DBH.ps1 -Method MobileSAMv2 -Dataset plotted -DataDir $Data -MaxFiles $MaxFiles -SaveMasks
```

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
