# Segmentation entry points

- `batchProcess_SAM31_CPU.py`: shared CPU/GPU runner; target selection follows automatically.
- `batchProcess_SAM31_GPU.ipynb`: Colab notebook using the same implementation.
- `selectTargetTree_SAM31.py`: repeat target selection from saved candidates.
- `MobileSAMv2_export/export_dbh_onnx.py`: optional PyTorch-to-ONNX export.
- `supporting_scripts`: earlier CPU runner and backend.
- `previousSAM3`: historical SAM3 preprocessing, requiring its original environment/checkpoint.

Use the root README for the current workflow. Paths are relative to the repository.
`--data-dir` chooses a folder containing `rawdata` and base-point CSVs;
`--output-dir` chooses the candidates/masks/report destination. Both options are
available on the batch and selection runners. `--limit` limits images per dataset.
