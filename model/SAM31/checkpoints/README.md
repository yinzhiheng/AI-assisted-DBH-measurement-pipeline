# SAM3.1 checkpoint

The checkpoint is downloaded separately from Meta's
[official Hugging Face repository](https://huggingface.co/facebook/sam3.1).

1. Sign in to Hugging Face and request access on the model page.
2. Once access is granted, open **Files and versions** and download
   [`sam3.1_multiplex.pt`](https://huggingface.co/facebook/sam3.1/blob/main/sam3.1_multiplex.pt).
3. Place the downloaded file here:

   ```text
   model/SAM31/checkpoints/sam3.1_multiplex.pt
   ```

The version used by this pipeline is **3,502,755,717 bytes** with SHA-256:

```text
0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6
```

To check the SHA-256 of sam3.1_multiplex.pt from the repository root, you could run the following line:

```powershell
Get-FileHash .\model\SAM31\checkpoints\sam3.1_multiplex.pt -Algorithm SHA256
```

MobileSAMv2 models are already included under `model/MobileSAMv2/`.
