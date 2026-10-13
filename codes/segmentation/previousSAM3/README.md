# SAM3 preprocessing

## Purpose

Generate tree-mask candidates with [SAM3](https://github.com/facebookresearch/sam3), then select the target tree using its base point. **Final segmented masks are already provided in `data/Segmentation`; skip preprocessing unless preparing new data or regenerating masks.**

## Run

1. Copy `preprocessing/` and the required `data/rawdata/rgb_*Field/` folders to `MyDrive/DBH-pipeline/` in Google Drive.
2. Open [sam3_batch_segment.ipynb](sam3_batch_segment.ipynb) in **Google Colab**, choose a **GPU** runtime, set `PROJECT_ROOT` and `DATASET`, and run the cells. The notebook installs the official SAM3 package, handles authentication, and runs `sam3_batch_segment.py`.
3. Copy `output/sam3_preprocessing/` from Drive into your local `DBH-pipeline/output/`. From the local project folder, run:

   ```sh
   python -m pip install -r preprocessing/requirements.txt
   python preprocessing/select_target_tree.py --dataset all --save-overlays
   ```

Selection uses `data/basePoints_natural.csv` and `data/basePoints_plotted.csv`. Use `--dataset natural` or `--dataset plot` for one dataset. Existing masks are kept; add `--overwrite` to reselect and replace them. Keep the same RGB files in both locations.

## Outputs

- **Candidates, reports, and optional overlays:** `output/sam3_preprocessing/`.
- **Natural masks:** `data/Segmentation/MaskForNaturalDataset/`.
- **Plotted masks:** `data/Segmentation/MaskForPlottedDataset/`.

Final files are `<uniqueName>_selected_mask.png`: original RGB dimensions, 8-bit grayscale, tree **255**, background **0**. Disconnected regions are preserved. The C++ SAM3 batch programs read these masks directly.
