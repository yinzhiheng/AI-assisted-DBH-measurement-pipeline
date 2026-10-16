# AI-assisted diameter-at-breast-height (DBH) measurement pipeline

SAM3.1 and MobileSAMv2 tree DBH measurement on the natural and plotted datasets.

## 1. SAM3.1 Model download

Please download the checkpoint from dropbox link here: https://www.dropbox.com/scl/fo/x5h4o6eyw9db6huogjgnj/AAT_fnW2s2rpvOSpKBBYExg?rlkey=00hno0ktgnu3dqwesi9s0iab1&st=ooxojppt&dl=0
Once downloaded, please move sam3.1_multiplex.pt to the folder ./model/SAM31/checkpoints

## 2. Install and check the environment

Use Windows 10/11 x64 with internet access. Open **PowerShell** in this repository
root and run these commands in order. 

```powershell
$Device = 'cpu' 

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

## 3. Generate SAM3.1 masks on Colab GPU (optional)

For generating tree mask using SAM3.1 via GPU, follow this section. CPU segmentation commands are in Step 4.

Upload files below to **My Drive**; for simplicity, you could directly upload the whole repo AI-assisted-DBH-measurement pipeline:

```text
My Drive/                              # /content/drive/MyDrive/
└── AI-assisted-DBH-measurement/
    ├── codes/segmentation/
    │   ├── batchProcess_SAM31_GPU.ipynb
    │   ├── batchProcess_SAM31_CPU.py
    │   └── selectTargetTree_SAM31.py
    ├── environment/python/sam31_gpu/
    │   └── requirements.txt
    ├── model/SAM31/
    │   ├── source/                   
    │   └── checkpoints/sam3.1_multiplex.pt
    ├── data/
    │   ├── rawdata/
    │   │   ├── rgb_naturalField/     
    │   │   └── rgb_plottedField/     
    │   ├── basePoints_natural.csv
    │   └── basePoints_plotted.csv
    └── output/SAM3SegResults/         
```

The checkpoint in ./model/SAM31/checkpoints is excluded from GitHub; please download the checkpoint first as described in step1 and then upload it from your local copy.
Then, you could generate masks on GPU by executing steps below in order:
1. Open [batchProcess_SAM31_GPU.ipynb](codes/segmentation/batchProcess_SAM31_GPU.ipynb)
   in Colab and select a **GPU runtime**.
2. In the first cell, `PROJECT_ROOT` must point to
   **`/content/drive/MyDrive/DBH-pipeline-whole-repo-clean`**. Run the mount and
   installation cells. If NumPy changes, restart the session and rerun the mount cell.
3. Set `DATASET` to `'natural'`, `'plotted'`, or `'all'`, and set
   `LIMIT_PER_DATASET` to match `$MaxFiles`.
4. Run every cell through the notebook.

**Bring results back to local computer:** copy the entire Drive `output/SAM3SegResults` folder
to your local computer and replace the folder at `AI-assisted-DBH-measurement/output/SAM3SegResults`, then continue to Step 4.

## 4. Run the four pipelines locally

From the repository root, set the device to match your installation and choose
how many examples to process from the raw data:

```powershell
$Device = 'cpu'  
$Python = ".\.venvs\sam31-$Device\Scripts\python.exe"
$SamDevice = if ($Device -eq 'cpu') { 'cpu' } else { 'cuda:0' }
$Data = 'data'
$MaxFiles = 10   # First 10 examples per dataset; use 0 for all 285 natural / 202 plotted
```

To switch between a sample and all data, only change `$MaxFiles`.

**For SAM3.1 pipeline, skip the first command if you copied that dataset's GPU masks
from Colab.** Otherwise, run both commands in order.

**1. Natural dataset — SAM3.1**

```powershell
& $Python -B -X utf8 .\codes\segmentation\batchProcess_SAM31_CPU.py --dataset natural --device cpu --limit $MaxFiles

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\codes\DBH\run\Run-DBH.ps1 -Method SAM3 -Dataset natural -MaxFiles $MaxFiles
```

**2. Plotted dataset — SAM3.1**

```powershell
& $Python -B -X utf8 .\codes\segmentation\batchProcess_SAM31_CPU.py --dataset plotted --device cpu --limit $MaxFiles

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\codes\DBH\run\Run-DBH.ps1 -Method SAM3 -Dataset plotted -MaxFiles $MaxFiles
```

**3. Natural dataset — MobileSAMv2**

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\codes\DBH\run\Run-DBH.ps1 -Method MobileSAMv2 -Dataset natural -MaxFiles $MaxFiles -SaveMasks
```

**4. Plotted dataset — MobileSAMv2**

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\codes\DBH\run\Run-DBH.ps1 -Method MobileSAMv2 -Dataset plotted -MaxFiles $MaxFiles -SaveMasks
```

## 5. Visualization and output locations

After four pipelines finish, open the notebook in the same PowerShell window:

```powershell
& $Python -B -X utf8 -m jupyterlab .\visualization\fusedSAM_visualization_clean.ipynb
```

Choose **Kernel → Restart Kernel and Run All Cells** to generate all statistical analysis plots.

All generated files are under `output/`.

| Output | Location |
| --- | --- |
| SAM3.1 selected masks and overlays | `output/SAM3SegResults/targetTreeMasks/` and `targetTreeOverlays/` |
| SAM3.1 segmentation processing time per image (CPU/GPU, seconds) | `output/SAM3SegResults/reports/timings_<run_id>.csv` |
| Four DBH result CSVs (cm) | `output/DBHResults/SAM3/` and `output/DBHResults/MobileSAMv2/` |
| Final plots (PNG/PDF) and statistics (CSV) | **`output/visualization/`** |
