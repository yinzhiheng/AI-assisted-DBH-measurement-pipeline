# AI-assisted diameter-at-breast-height (DBH) measurement pipeline

SAM3.1 and MobileSAMv2 tree DBH measurement on the natural and plotted datasets.

## 1. SAM3.1 Model download

Please download the checkpoint from dropbox link here: https://www.dropbox.com/scl/fo/x5h4o6eyw9db6huogjgnj/AAT_fnW2s2rpvOSpKBBYExg?rlkey=00hno0ktgnu3dqwesi9s0iab1&st=ooxojppt&dl=0
Once downloaded, please move sam3.1_multiplex.pt to the folder ./model/SAM31/checkpoints

## 2. Install and check the environment

Use Windows 10/11 x64 with internet access. Open **PowerShell** in `AI-assisted-DBH-measurement-pipeline-main`, the project root folder containing README.md and the codes, data, and environment folders. Then run these commands in order. 

```powershell
$Device = 'cpu' 
```
```
# Microsoft C++ runtime, compiler and Windows SDK
.\environment\Setup-Windows.cmd -Stage System -Device $Device -InstallSystemPrerequisites
```
```
# Python environment
.\environment\Setup-Windows.cmd -Stage Python -Device $Device
```
```
# PyTorch for SAM3.1, using the selected CPU/GPU
.\environment\Setup-Windows.cmd -Stage Torch -Device $Device
```
```
# SAM3.1 and visualization packages
.\environment\Setup-Windows.cmd -Stage Packages -Device $Device
```
```
# Build the four C++ measurement programs
.\environment\Setup-Windows.cmd -Stage Cpp
```

Each stage must report `PASS` before continuing. Accept administrator prompts if
needed; if a restart is requested, restart and repeat that stage. The installation might take over 10 mins.

## 3. Generate SAM3.1 masks on Colab GPU (optional)

CPU users can continue to Step 4.

**Upload the whole project.** Open [Google Drive → My Drive](https://drive.google.com/drive/my-drive)
using the same Google account as Colab, then upload the entire folder **`AI-assisted-DBH-measurement-pipeline-main`**. Before uploading, make sure the downloaded checkpoint is saved at `model/SAM31/checkpoints/sam3.1_multiplex.pt` inside the project folder. The project is approximately 4 GB, so uploading may take around an hour, depending on your internet upload speed.

**Run the notebook.** Open
[batchProcess_SAM31_GPU.ipynb](codes/segmentation/batchProcess_SAM31_GPU.ipynb)
in Colab, select a **GPU runtime**, and check these settings:

| Setting | Value |
| --- | --- |
| `PROJECT_ROOT` | `/content/drive/MyDrive/AI-assisted-DBH-measurement-pipeline-main` |
| `DATASET` | `'natural'`, `'plotted'`, or `'all'` |
| `LIMIT_PER_DATASET` | Match local `$MaxFiles`: `10`, or `0` for all images |
| `GPU_PRECISION` | `'float32'` |

Choose **Runtime → Run all**. Allow Drive access when prompted. If installation
requires a session restart, restart and choose **Run all** again.

**Continue locally.** Download the entire Drive project's
`output/SAM3SegResults/` folder and replace the local folder at **`<local-path>/AI-assisted-DBH-measurement-pipeline-main/output/SAM3SegResults/`**,
then continue to Step 4.

## 4. Run the four pipelines locally

Open PowerShell in the repository root and set the following options. By default, the pipelines use the CPU and process the first 10 images per dataset.

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
```
```
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\codes\DBH\run\Run-DBH.ps1 -Method SAM3 -Dataset natural -MaxFiles $MaxFiles
```

**2. Plotted dataset — SAM3.1**

```powershell
& $Python -B -X utf8 .\codes\segmentation\batchProcess_SAM31_CPU.py --dataset plotted --device cpu --limit $MaxFiles
```
```
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

## 5. Running Time Test

**SAM3.1 on CPU/GPU:** Run the CPU segmentation script (Section 4) or the Colab GPU notebook (Section 3). Per-image running times in seconds are automatically saved to `output/SAM3SegResults/reports/timings_<run_id>.csv`. 

**MobileSAMv2 on iPhone:** Follow [the setup instructions](codes/runTimeTestingCodesForiPhone/README.md) to install the test app. Run **Batch all 487**, then select **Export latest CSV**.

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
