"""Simple SAM 3.1 processing for the two tree datasets.

Use the documented SAM31 environment. Run with --dataset all or --limit 1.
Paths are resolved from this repository, independent of the working directory. No model runs when this is imported.
The Colab notebook imports these same functions to keep the comparison fair.

Naming: functions use action-based camelCase; variables use camelCase;
editable constants use UPPER_SNAKE_CASE. CSV/JSON fields retain their existing
names so previous logs and new logs can still be compared.
"""
from contextlib import nullcontext
import colorsys
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import importlib
import json
import os
from pathlib import Path
import platform
import re
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from PIL.PngImagePlugin import PngInfo
import psutil

# Project paths: all are relative to the codes folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAM_SOURCE_DIRECTORY = PROJECT_ROOT / "model/SAM31/source"
SAM31_CHECKPOINT_PATH = PROJECT_ROOT / "model/SAM31/checkpoints/sam3.1_multiplex.pt"
RAW_IMAGE_DIRECTORY = PROJECT_ROOT / "data/rawdata"
NATURAL_FIELD_IMAGE_DIRECTORY = RAW_IMAGE_DIRECTORY / "rgb_naturalField"
PLOTTED_FIELD_IMAGE_DIRECTORY = RAW_IMAGE_DIRECTORY / "rgb_plottedField"
OUTPUT_DIRECTORY = PROJECT_ROOT / "output/SAM3SegResults"

# Segmentation settings: the target object, confidence cutoff and CPU workload.
TEXT_PROMPT = "tree trunk"
CONFIDENCE_THRESHOLD = 0.3
CPU_THREAD_COUNT = 8
WARMUP_PASS_COUNT = 1  # One extra, untimed prediction per run; set 0 for no warm-up.
MODEL_INPUT_SIZE = 1008


def readCpuHardwareConfiguration():
    """Return a dictionary describing this computer's CPU, memory and software."""

    # 1. Read the processor name from Windows locally, or from Linux on Colab.
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as registryKey:
            cpuName = winreg.QueryValueEx(registryKey, "ProcessorNameString")[0].strip()
    else:  # Colab uses Linux.
        cpuName = next(line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name"))

    # 2. Collect core counts and memory sizes for the configuration JSON.
    # RAM values use GiB (bytes divided by 1024 cubed).
    systemMemory = psutil.virtual_memory()
    return {"cpu_name": cpuName, "physical_cores_reported": psutil.cpu_count(logical=False),
            "logical_processors": os.cpu_count(), "ram_gib": round(systemMemory.total / 2**30, 3),
            "available_ram_gib_at_start": round(systemMemory.available / 2**30, 3),
            "operating_system": platform.platform(), "python_version": platform.python_version()}


def locateSam31SourceDirectory():
    """Return the source folder containing sam3/model_builder.py, or explain what is missing."""

    # 1. Resolve the configured relative path from this script's codes folder.
    # This also works when a notebook cell has changed its working directory.
    codesDirectory = Path(__file__).resolve().parent
    sourceDirectory = (codesDirectory / SAM_SOURCE_DIRECTORY).resolve()
    modelBuilderPath = sourceDirectory / "sam3" / "model_builder.py"
    packageInitPath = sourceDirectory / "sam3" / "__init__.py"

    # 2. Require the extracted source code, not just a ZIP file or checkpoint.
    # This check explains the observed Colab import error before model loading.
    if not modelBuilderPath.is_file() or not packageInitPath.is_file():
        raise FileNotFoundError(
            "The SAM source folder is missing or incomplete. Expected files:\n"
            f"{modelBuilderPath}\n{packageInitPath}\n\n"
            "Copy/extract the full sam3-main source folder into your project's codes folder. "
            "If it is elsewhere, set SAM_SOURCE_DIRECTORY to the folder that directly "
            "contains the sam3 subfolder. This path is relative to the codes folder. "
            "Installing the supporting packages does not copy SAM's source code."
        )

    # 3. Return the exact folder Python needs on its import search path.
    return sourceDirectory


def loadSam31ImageSegmentationModel(device):
    """Return the loaded SAM 3.1 image detector on the requested CPU or GPU.

    device is 'cpu' or a CUDA device such as 'cuda:0'. Model loading happens
    once per batch and is excluded from every image's segmentation time.
    """

    # 1. Import the SAM implementation from this project's source directory.
    import torch
    sourceDirectory = locateSam31SourceDirectory()
    sys.path.insert(0, str(sourceDirectory))
    # Refresh earlier failed lookups when the source was uploaded into Drive
    # after this Colab session had already tried to import the sam3 package.
    importlib.invalidate_caches()
    from sam3 import model_builder as sam31ModelBuilder
    from sam3.model import vitdet
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.vl_combiner import SAM3VLBackboneTri

    # 2. Use the chosen calculation precision consistently across devices.
    # SAM's fused helper forces BF16 on GPU. Ordinary PyTorch operations work
    # on CPU, T4 and A100, respect our precision setting, and keep the weights.
    # This changes this Python session only; no SAM source files are edited.
    vitdet.addmm_act = lambda activationClass, linearLayer, inputTensor: (
        activationClass()(linearLayer(inputTensor))
    )
    # The SAM import enables TF32 on newer GPUs. Turn it off for an FP32 baseline.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    # 3. Build the image detector: image/text encoders, object detector and masks.
    # Independent images do not need the video tracker.
    segmentationModel = Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=SAM3VLBackboneTri(
            scalp=0,
            visual=sam31ModelBuilder._create_multiplex_tri_backbone(
                compile_mode=None, use_fa3=False, use_rope_real=True),
            text=sam31ModelBuilder._create_text_encoder(
                str(sourceDirectory / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
            ),
        ),
        transformer=sam31ModelBuilder._create_sam3_transformer(use_fa3=False),
        segmentation_head=sam31ModelBuilder._create_segmentation_head(use_fa3=False),
        semantic_segmentation_head=None,
        input_geometry_encoder=sam31ModelBuilder._create_geometry_encoder(),
        use_early_fusion=True, use_dot_prod_scoring=True,
        dot_prod_scoring=sam31ModelBuilder._create_dot_product_scoring(),
        supervise_joint_box_scores=True, is_multiplex=True,
    )

    # 4. Read the checkpoint and select only its detector weights.
    # mmap avoids holding an extra full copy of the checkpoint in RAM.
    checkpointPath = (Path(__file__).resolve().parent / SAM31_CHECKPOINT_PATH).resolve()
    checkpointContents = torch.load(
        str(checkpointPath), map_location="cpu", weights_only=True, mmap=True
    )
    checkpointContents = checkpointContents.get("model", checkpointContents)
    detectorWeights = {
        key.removeprefix("detector."): value
        for key, value in checkpointContents.items() if key.startswith("detector.")
    }

    # 5. Reconstruct the position tables needed by this implementation.
    # The real/imaginary position tables come from the checkpoint's complex
    # tables. They are calculated constants, not missing learned weights.
    for bufferName, _ in segmentationModel.named_buffers():
        if bufferName not in detectorWeights and bufferName.endswith((".freqs_cis_real", ".freqs_cis_imag")):
            complexBufferName, complexComponent = bufferName.rsplit("_", 1)
            detectorWeights[bufferName] = getattr(
                detectorWeights[complexBufferName], complexComponent
            ).contiguous()

    # 6. Load all learned weights, move to the device and disable training.
    segmentationModel.load_state_dict(detectorWeights, strict=True)
    return segmentationModel.to(device).eval().requires_grad_(False)


def resizeAndNormalizeImageForSam31(rgbImage):
    """Return a CPU tensor with shape [1, 3, 1008, 1008] and pixels in [-1, 1]."""
    import torch

    # 1. Resize the RGB image to the model's fixed input size.
    resizedImage = rgbImage.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR)

    # 2. Convert pixel values from 0..255 to 0..1 and put channels before rows.
    # The leading dimension of 1 means we process one image at a time.
    normalizedPixels = np.asarray(resizedImage, dtype=np.float32) / np.float32(255.0)
    imageTensor = torch.from_numpy(normalizedPixels).permute(2, 0, 1).contiguous().unsqueeze(0)

    # 3. Shift and scale to the pixel range used when SAM was trained.
    return imageTensor.sub_(0.5).div_(0.5)


def predictObjectMasksWithSam31(segmentationModel, imageTensor, textPrompt, precision):
    """Return raw mask predictions, confidence values and boxes on the device.

    imageTensor is already prepared and on the selected device. This function
    performs model work only; image reading, resizing and saving happen elsewhere.
    """
    import torch
    from sam3.model.data_misc import FindStage

    # 1. Choose full precision or the requested GPU mixed precision.
    device = imageTensor.device
    precisionContext = (torch.autocast("cuda", dtype=getattr(torch, precision))
                       if device.type == "cuda" and precision != "float32" else nullcontext())
    with torch.inference_mode(), precisionContext:

        # 2. Encode the image and text anew for this image's timing measurement.
        encodedFeatures = segmentationModel.backbone.forward_image(
            imageTensor, need_sam3_out=True, need_interactive_out=False, need_propagation_out=False)
        encodedFeatures.update(segmentationModel.backbone.forward_text([textPrompt], device=device))

        # 3. Pair the single image with the single text prompt.
        # No click points or user-drawn boxes are supplied.
        imageTextPair = FindStage(
            img_ids=torch.tensor([0], dtype=torch.long, device=device),
            text_ids=torch.tensor([0], dtype=torch.long, device=device),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None)

        # 4. Find objects matching the text and predict their masks and boxes.
        modelOutputs = segmentationModel.forward_grounding(
            backbone_out=encodedFeatures, find_input=imageTextPair, find_target=None,
            geometric_prompt=segmentationModel._get_dummy_prompt())

        # 5. Return the three outputs used to create the saved image products.
        return {key: modelOutputs[key] for key in ("pred_masks", "pred_logits", "pred_boxes")}


def convertPredictionsToBinaryMaskAndObjectIds(modelPredictions, originalImageSize, confidenceThreshold):
    """Return a combined mask, object IDs, metadata and packed individual masks.

    Predictions must be on CPU. originalImageSize is (width, height).
    Retain all candidates above confidenceThreshold; higher scores win overlaps.
    """
    import torch
    import torch.nn.functional as torchFunctional
    imageWidth, imageHeight = originalImageSize

    # 1. Convert raw scores to probabilities and reject unusable numeric results.
    # SAM 3.1 already includes presence in these scores. Do not multiply twice.
    confidenceScores = modelPredictions["pred_logits"].sigmoid().reshape(-1)
    if not torch.isfinite(confidenceScores).all() or not torch.isfinite(modelPredictions["pred_masks"]).all():
        raise RuntimeError("Non-finite predictions. If using mixed precision, retry with float32.")

    # 2. Keep candidates above the cutoff, ordered from highest to lowest score.
    # This order determines the object IDs and which object owns an overlap.
    retainedQueryIndices = torch.where(confidenceScores > confidenceThreshold)[0]
    retainedQueryIndices = retainedQueryIndices[
        torch.argsort(confidenceScores[retainedQueryIndices], descending=True, stable=True)
    ]

    # 3. Resize each mask to the original image and fill its unclaimed ID pixels.
    # A raw mask value above zero is equivalent to a probability above 0.5.
    objectIdImage = np.zeros((imageHeight, imageWidth), dtype=np.uint16)
    candidateAreas = []
    packedCandidateMasks = []
    for objectId, queryIndex in enumerate(retainedQueryIndices.tolist(), start=1):
        candidateMask = torchFunctional.interpolate(
            modelPredictions["pred_masks"][0, queryIndex][None, None],
            size=(imageHeight, imageWidth), mode="bilinear", align_corners=False,
        )[0, 0] > 0
        candidateMask = candidateMask.numpy()
        candidateAreas.append(int(candidateMask.sum()))
        packedCandidateMasks.append(np.packbits(candidateMask.reshape(-1), bitorder="little"))
        objectIdImage[candidateMask & (objectIdImage == 0)] = objectId

    # 4. Convert normalized center/size boxes into image-pixel corner coordinates.
    boxCenterX, boxCenterY, boxWidth, boxHeight = (
        modelPredictions["pred_boxes"][0, retainedQueryIndices].unbind(-1)
    )
    pixelBoxes = torch.stack(
        (boxCenterX - boxWidth / 2, boxCenterY - boxHeight / 2,
         boxCenterX + boxWidth / 2, boxCenterY + boxHeight / 2), -1,
    )
    pixelBoxes *= torch.tensor([imageWidth, imageHeight, imageWidth, imageHeight])

    # 5. Keep the IDs, scores, boxes and areas to embed inside each saved PNG.
    candidateMetadata = {
        "object_ids": list(range(1, len(retainedQueryIndices) + 1)),
        "query_indices": retainedQueryIndices.tolist(),
        "scores": confidenceScores[retainedQueryIndices].tolist(),
        "boxes_xyxy": pixelBoxes.tolist(), "areas_pixels": candidateAreas,
        "image_shape": [imageHeight, imageWidth],
        "all_query_scores": confidenceScores.tolist(),
    }

    # 6. Combine all object pixels: black background (0), white foreground (255).
    packed = np.stack(packedCandidateMasks) if packedCandidateMasks else np.empty((0, (imageHeight * imageWidth + 7) // 8), dtype=np.uint8)
    return (objectIdImage > 0).astype(np.uint8) * 255, objectIdImage, candidateMetadata, packed


def drawObjectIdsAndScoresOnRgbImage(rgbImage, objectIdImage, candidateMetadata):
    """Return an RGB visualization with colored objects and ID/confidence labels."""

    # 1. Assign each object ID a repeatable color; ID 0 is the background.
    objectColors = np.zeros((len(candidateMetadata["object_ids"]) + 1, 3), dtype=np.uint8)
    for objectId in candidateMetadata["object_ids"]:
        objectColors[objectId] = np.round(np.array(colorsys.hsv_to_rgb(
            (objectId * 0.61803398875) % 1, 0.8, 1.0)) * 255).astype(np.uint8)

    # 2. Blend object colors into a copy of the original RGB image.
    overlayPixels = np.array(rgbImage, copy=True)
    foregroundPixels = objectIdImage > 0
    overlayPixels[foregroundPixels] = np.round(
        0.55 * overlayPixels[foregroundPixels]
        + 0.45 * objectColors[objectIdImage[foregroundPixels]]
    )

    # 3. Prepare the drawing surface and choose a readable label size.
    overlayImage = Image.fromarray(overlayPixels)
    imageDrawing = ImageDraw.Draw(overlayImage)
    labelFont = ImageFont.load_default(size=max(14, round(min(rgbImage.size) / 70)))

    # 4. Place each ID/confidence label on a visible pixel of its object.
    for objectId, confidenceScore in zip(candidateMetadata["object_ids"], candidateMetadata["scores"]):
        pixelRows, pixelColumns = np.nonzero(objectIdImage == objectId)
        if len(pixelColumns):  # A candidate fully covered by another has no visible pixels.
            labelPixelIndex = np.argmin(
                (pixelColumns - np.median(pixelColumns))**2
                + (pixelRows - np.median(pixelRows))**2
            )
            labelText = f"{objectId} ({confidenceScore:.2f})"
            # Keep the full label inside the image, including its black outline.
            textLeft, textTop, textRight, textBottom = imageDrawing.textbbox(
                (0, 0), labelText, font=labelFont, anchor="mm", stroke_width=2
            )
            labelX = int(np.clip(
                pixelColumns[labelPixelIndex], -textLeft, max(-textLeft, rgbImage.width - textRight)
            ))
            labelY = int(np.clip(
                pixelRows[labelPixelIndex], -textTop, max(-textTop, rgbImage.height - textBottom)
            ))
            imageDrawing.text((labelX, labelY), labelText, font=labelFont, fill="white", anchor="mm",
                      stroke_width=2, stroke_fill="black")
    return overlayImage


def saveMaskObjectIdsAndOverlayPngs(
    imagePath, rgbImage, combinedBinaryMask, objectIdImage, candidateMetadata, packedCandidateMasks, runConfiguration
):
    """Save PNGs and lossless candidate archive; return paths and save times."""

    # 1. Build output filenames from the image stem, device and current run.
    relativeImagePath = imagePath.relative_to(RAW_IMAGE_DIRECTORY)
    # Each run gets its own folder. The original .jpeg never enters the name.
    # The run folder and PNG metadata already identify the hardware. Repeating
    # a long GPU name here can exceed Windows' path-length limit after relocation.
    outputFilenameStem = imagePath.stem
    outputPaths = {
        outputKind: (
            OUTPUT_DIRECTORY / "candidates" / runConfiguration["run_id"]
            / {"rgb_naturalField": "natural", "rgb_plottedField": "plotted"}[relativeImagePath.parts[0]] / outputKind / f"{outputFilenameStem}{filenameSuffix}"
        )
        for outputKind, filenameSuffix in (
            ("masks", "_masks.png"), ("object_id", "_object_ids.png"), ("overlay", "_overlay.png")
        )
    }

    # 2. Create the three output directories before starting any save timer.
    for outputPath in outputPaths.values():
        outputPath.parent.mkdir(parents=True, exist_ok=True)

    # 3. Embed candidate details and run settings in the PNGs and archive.
    candidateMetadata.update(
        run_id=runConfiguration["run_id"], device_tag=runConfiguration["device_tag"],
        model="SAM 3.1", checkpoint=str(SAM31_CHECKPOINT_PATH), prompt=runConfiguration["prompt"],
        confidence_threshold=runConfiguration["confidence_threshold"], mask_values=[0, 255],
    )
    pngMetadata = PngInfo()
    pngMetadata.add_text("sam31_metadata", json.dumps(candidateMetadata, separators=(",", ":")))

    # 4. Draw the visualization and record drawing time separately from saving.
    startTime = perf_counter()
    overlayImage = drawObjectIdsAndScoresOnRgbImage(rgbImage, objectIdImage, candidateMetadata)
    saveTimings = {"overlay_render_seconds": perf_counter() - startTime}

    # 5. Encode and write each PNG using its own timer, then return all results.
    # These timers do not wait for background Drive or OneDrive synchronization.
    outputImages = {
        "masks": Image.fromarray(combinedBinaryMask),
        "object_id": Image.fromarray(objectIdImage), "overlay": overlayImage,
    }
    for outputKind, outputImage in outputImages.items():
        startTime = perf_counter()
        outputImage.save(outputPaths[outputKind], format="PNG", pnginfo=pngMetadata)
        saveTimings[f"{outputKind}_save_seconds"] = perf_counter() - startTime
        saveTimings[f"{outputKind}_path"] = outputPaths[outputKind].as_posix()
    archivePath = outputPaths["masks"].parent.parent / "individual_masks" / f"{outputFilenameStem}_candidates.npz"
    archivePath.parent.mkdir(parents=True, exist_ok=True)
    startTime = perf_counter()
    np.savez_compressed(archivePath, packed_masks=packedCandidateMasks,
                        image_shape=np.array(candidateMetadata["image_shape"], dtype=np.int64),
                        metadata_json=np.array(json.dumps(candidateMetadata)))
    saveTimings["candidate_archive_save_seconds"] = perf_counter() - startTime
    saveTimings["candidate_archive_path"] = archivePath.as_posix()
    return saveTimings


def segmentSingleImageAndSaveMasks(segmentationModel, imagePath, runConfiguration, warmupPassCount=0):
    """Segment one image, save PNGs plus an archive and return its timing record.

    segmentationModel is already loaded. runConfiguration holds the device,
    prompt, confidence and output naming settings. Optional warm-up passes are
    excluded from the measured prediction. Every timed stage is marked below.
    """
    import torch
    device, precision = runConfiguration["device"], runConfiguration["precision"]
    # Timing helper: GPU calls can return before the calculations finish.
    # Wait at timing boundaries to measure completed work; CPU needs no wait.
    def waitForGpuToFinish():
        """Wait for outstanding work on the selected GPU, if a GPU is used."""
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

    # 1. Read/decode the image, apply camera orientation and convert to RGB.
    startTime = perf_counter()
    with Image.open(imagePath) as openedImage:
        rgbImage = ImageOps.exif_transpose(openedImage).convert("RGB")
        rgbImage.load()
    imageTimingRecord = {"image_read_seconds": perf_counter() - startTime}

    # 2. Resize and normalize on CPU; exclude this from model inference time.
    startTime = perf_counter()
    imageTensor = resizeAndNormalizeImageForSam31(rgbImage)
    imageTimingRecord["preprocess_seconds"] = perf_counter() - startTime

    # 3. Transfer the prepared pixels to the device, using a separate timer.
    waitForGpuToFinish()
    startTime = perf_counter()
    imageTensor = imageTensor.to(device)
    waitForGpuToFinish()
    imageTimingRecord["to_device_seconds"] = perf_counter() - startTime

    # 4. Optionally warm up the model before starting the measured prediction.
    for _ in range(warmupPassCount):
        print("  Untimed warm-up...", flush=True)
        predictObjectMasksWithSam31(segmentationModel, imageTensor, runConfiguration["prompt"], precision)

    # 5. Time image/text encoding, object detection and raw mask prediction only.
    waitForGpuToFinish()
    startTime = perf_counter()
    modelPredictions = predictObjectMasksWithSam31(
        segmentationModel, imageTensor, runConfiguration["prompt"], precision
    )
    waitForGpuToFinish()
    imageTimingRecord["model_inference_seconds"] = perf_counter() - startTime

    # 6. Bring predictions back to CPU as float32 for the shared mask finishing.
    startTime = perf_counter()
    modelPredictions = {key: value.float().cpu() for key, value in modelPredictions.items()}
    waitForGpuToFinish()
    imageTimingRecord["to_cpu_seconds"] = perf_counter() - startTime

    # 7. Apply the confidence cutoff and create original-size binary/ID masks.
    startTime = perf_counter()
    combinedBinaryMask, objectIdImage, candidateMetadata, packedCandidateMasks = convertPredictionsToBinaryMaskAndObjectIds(
        modelPredictions, rgbImage.size, runConfiguration["confidence_threshold"]
    )
    imageTimingRecord["postprocess_seconds"] = perf_counter() - startTime

    # 8. Add the segmentation stages and identify the image/settings in its record.
    # File reading, drawing and saving are deliberately outside this total.
    imageTimingRecord["segmentation_seconds"] = sum(imageTimingRecord[key] for key in (
        "preprocess_seconds", "to_device_seconds", "model_inference_seconds",
        "to_cpu_seconds", "postprocess_seconds"))
    imageTimingRecord.update(
        run_id=runConfiguration["run_id"], device_tag=runConfiguration["device_tag"], precision=precision,
        cpu_threads=runConfiguration["pytorch_threads"], prompt=runConfiguration["prompt"],
        confidence_threshold=runConfiguration["confidence_threshold"],
        image=imagePath.relative_to(RAW_IMAGE_DIRECTORY).as_posix(), dataset=imagePath.parent.name,
        width=rgbImage.width, height=rgbImage.height,
        candidate_count=len(candidateMetadata["object_ids"]),
    )

    # 9. Save PNGs and the archive, recording each output's path and save time.
    imageTimingRecord.update(saveMaskObjectIdsAndOverlayPngs(
        imagePath, rgbImage, combinedBinaryMask, objectIdImage, candidateMetadata, packedCandidateMasks, runConfiguration
    ))
    return imageTimingRecord


def segmentImageBatchAndRecordTimings(
    imagePaths, *, device="cpu", precision="float32", cpuThreadCount=CPU_THREAD_COUNT,
    textPrompt=TEXT_PROMPT, confidenceThreshold=CONFIDENCE_THRESHOLD, warmupPassCount=WARMUP_PASS_COUNT
):
    """Segment a list of image paths, save PNGs/configuration and return the CSV path.

    Load the model once for the whole batch. device and precision choose where
    and how it calculates; cpuThreadCount sets the CPU workload. textPrompt and
    confidenceThreshold select objects. warmupPassCount applies only to the
    first image. Image paths are relative to the codes folder.
    """

    # 1. Configure CPU/GPU visibility before importing PyTorch or SAM.
    os.environ["USE_PERFLIB"] = "0"
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch

    # 2. Set CPU calculation threads, including the CPU work supporting GPU runs.
    torch.set_num_threads(cpuThreadCount)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)

    # 3. Materialize the image list and check the few settings needed for a valid run.
    imagePaths = list(imagePaths)
    if not imagePaths:
        raise ValueError("No .jpeg images selected. Run from the codes folder.")
    if precision not in ("float32", "float16", "bfloat16") or (device == "cpu" and precision != "float32"):
        raise ValueError("CPU uses float32; GPU accepts float32, float16 or bfloat16.")

    # 4. Read the host CPU/RAM details; on GPU, also record the actual GPU supplied.
    runConfiguration = readCpuHardwareConfiguration()
    deviceNameForFiles = "CPU"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Select a GPU in Colab: Runtime > Change runtime type.")
        torch.cuda.set_device(device)
        gpuProperties = torch.cuda.get_device_properties(device)
        if precision == "bfloat16" and gpuProperties.major < 8:
            raise ValueError("Choose float32 or float16 on T4; bfloat16 requires a newer GPU.")
        runConfiguration.update(
            gpu_name=gpuProperties.name,
            gpu_memory_gib=round(gpuProperties.total_memory / 2**30, 3),
            gpu_compute_capability=f"{gpuProperties.major}.{gpuProperties.minor}",
            gpu_multiprocessors=gpuProperties.multi_processor_count,
            cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        )
        deviceNameForFiles = "GPU_" + re.sub(r"[^A-Za-z0-9]+", "-", gpuProperties.name).strip("-")

    # 5. Build the device tag and a timestamp so every run has separate outputs.
    # The GPU name comes from the hardware rather than a manually typed label.
    deviceTag = f"{deviceNameForFiles}_{precision}_t{torch.get_num_threads()}"
    runTimestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    # 6. Record model settings, software versions and the meaning of each timer.
    runConfiguration.update(
        # Run identity and source/checkpoint locations.
        run_id=f"{'CPU' if device == 'cpu' else 'GPU'}_{precision}_{runTimestamp}",
        device_tag=deviceTag, device=device, precision=precision,
        model="SAM 3.1", checkpoint=SAM31_CHECKPOINT_PATH.as_posix(),
        source=SAM_SOURCE_DIRECTORY.as_posix(),
        # Software versions and actual CPU thread settings.
        torch_version=str(torch.__version__), pytorch_threads=torch.get_num_threads(),
        library_versions={
            packageName: version(packageName) for packageName in ("torchvision", "timm", "numpy", "Pillow")
        },
        pytorch_interop_threads=torch.get_num_interop_threads(),
        # Segmentation choices shared by all images in this run.
        prompt=textPrompt, confidence_threshold=confidenceThreshold,
        input_size=[MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        batch_size=1, warmup_passes=warmupPassCount, tf32=False, compile=False, use_fa3=False,
        mlp_implementation="ordinary PyTorch linear + activation on CPU and GPU",
        # Progress and first-run timing information.
        postprocessing_device="cpu", selected_images=len(imagePaths), completed_images=0,
        first_inference_without_warmup=(warmupPassCount == 0), status="running",
        # Definitions saved alongside the measurements for later interpretation.
        timing_notes={
            "model_inference_seconds": (
                "Image/text encoding and mask prediction; synchronized on GPU. Excludes file I/O, "
                "loading, warm-up, pixel preparation, device transfers, mask resizing and saving."
            ),
            "segmentation_seconds": (
                "Preparation + device transfers + model inference + finishing masks; "
                "excludes reading, drawing and saving."
            ),
            "save_seconds": (
                "PNG encoding, write and close for each file. Excludes drawing, folder creation "
                "and background Drive/OneDrive synchronization."
            ),
        },
    )

    # 7. Save the configuration before loading the model; reserve the CSV path.
    configurationDirectory = OUTPUT_DIRECTORY / "reports/configurations"
    configurationDirectory.mkdir(parents=True, exist_ok=True)
    deviceCategory = "CPU" if device == "cpu" else "GPU"
    configurationPath = (
        configurationDirectory / f"{deviceCategory}_configuration_{runConfiguration['run_id']}.json"
    )
    timingCsvPath = OUTPUT_DIRECTORY / "reports" / f"timings_{runConfiguration['run_id']}.csv"
    configurationPath.write_text(json.dumps(runConfiguration, indent=2), encoding="utf-8")
    print(f"{len(imagePaths)} images; {deviceTag}\nConfiguration: {configurationPath}", flush=True)
    try:

        # 8. Load the checkpoint once, outside all per-image timers.
        print("Loading SAM 3.1 (outside image timing)...", flush=True)
        segmentationModel = loadSam31ImageSegmentationModel(device)

        # 9. Process images in order and keep the timing CSV open for this batch.
        with timingCsvPath.open("w", newline="", encoding="utf-8-sig") as timingCsvFile:
            for imageIndex, imagePath in enumerate(imagePaths):
                print(f"[{imageIndex + 1}/{len(imagePaths)}] {imagePath}", flush=True)

                # 9a. Segment/save this image; warm up only on the first image.
                imageTimingRecord = segmentSingleImageAndSaveMasks(
                    segmentationModel, imagePath, runConfiguration,
                    warmupPassCount=warmupPassCount if imageIndex == 0 else 0,
                )
                imageTimingRecord["first_inference_without_warmup"] = int(
                    imageIndex == 0 and warmupPassCount == 0
                )

                # 9b. Write the header once, then immediately save this image's row.
                if imageIndex == 0:
                    timingCsvWriter = csv.DictWriter(timingCsvFile, fieldnames=list(imageTimingRecord))
                    timingCsvWriter.writeheader()
                timingCsvWriter.writerow({
                    fieldName: round(fieldValue, 6) if isinstance(fieldValue, float) else fieldValue
                    for fieldName, fieldValue in imageTimingRecord.items()
                })
                timingCsvFile.flush()

                # 9c. Count completed images and show a short progress message.
                runConfiguration["completed_images"] += 1
                print(f"  Model: {imageTimingRecord['model_inference_seconds']:.3f} s; "
                      f"{imageTimingRecord['candidate_count']} candidates; saved three PNGs and individual candidate archive.", flush=True)

        # 10. Mark success only after every selected image has been saved.
        runConfiguration["status"] = "complete"
    finally:

        # 11. Save the final status/count even if an error or Ctrl+C stops the run.
        if runConfiguration["status"] != "complete":
            runConfiguration["status"] = "stopped_before_completion"
        runConfiguration["ended_utc"] = datetime.now(timezone.utc).isoformat()
        configurationPath.write_text(json.dumps(runConfiguration, indent=2), encoding="utf-8")

    # 12. Return the CSV path so a notebook can use it for later analysis.
    print(f"Finished. Timing log: {timingCsvPath}")
    return timingCsvPath



def segmentNaturalFieldImagesAndSaveMasks(**segmentationSettings):
    """Process all 285 current natural-field JPEG images."""
    # Select natural-field JPEGs in filename order, then segment/save the batch.
    return segmentImageBatchAndRecordTimings(
        sorted(NATURAL_FIELD_IMAGE_DIRECTORY.glob("*.jpeg")), **segmentationSettings
    )


def segmentPlottedFieldImagesAndSaveMasks(**segmentationSettings):
    """Process all 202 current plotted-field JPEG images."""
    # Select plotted-field JPEGs in filename order, then segment/save the batch.
    return segmentImageBatchAndRecordTimings(
        sorted(PLOTTED_FIELD_IMAGE_DIRECTORY.glob("*.jpeg")), **segmentationSettings
    )


def segmentBothFieldDatasetsAndSaveMasks(**segmentationSettings):
    """Process both folders, loading the model only once (487 current images)."""
    # Combine both filename-ordered lists into one batch, natural field first.
    return segmentImageBatchAndRecordTimings(
        sorted(NATURAL_FIELD_IMAGE_DIRECTORY.glob("*.jpeg"))
        + sorted(PLOTTED_FIELD_IMAGE_DIRECTORY.glob("*.jpeg")), **segmentationSettings
    )


def main():
    """Process either dataset from any working directory; select targets by default."""
    global RAW_IMAGE_DIRECTORY, NATURAL_FIELD_IMAGE_DIRECTORY, PLOTTED_FIELD_IMAGE_DIRECTORY, OUTPUT_DIRECTORY
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "natural", "plotted"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="Images per dataset; 0 means all")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda:0"])
    parser.add_argument("--precision", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--threads", type=int, default=CPU_THREAD_COUNT)
    parser.add_argument("--warmup", type=int, default=WARMUP_PASS_COUNT)
    parser.add_argument("--skip-selection", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="Folder containing rawdata and base-point CSVs")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    if args.limit < 0 or args.threads < 1 or args.warmup < 0:
        parser.error("limit/warmup must be nonnegative and threads must be positive")
    dataDirectory = (PROJECT_ROOT / args.data_dir).resolve()
    RAW_IMAGE_DIRECTORY = dataDirectory / "rawdata"
    NATURAL_FIELD_IMAGE_DIRECTORY = RAW_IMAGE_DIRECTORY / "rgb_naturalField"
    PLOTTED_FIELD_IMAGE_DIRECTORY = RAW_IMAGE_DIRECTORY / "rgb_plottedField"
    OUTPUT_DIRECTORY = (PROJECT_ROOT / args.output_dir).resolve()
    paths = []
    for name, directory in [("natural", NATURAL_FIELD_IMAGE_DIRECTORY), ("plotted", PLOTTED_FIELD_IMAGE_DIRECTORY)]:
        if args.dataset in ("all", name):
            selected = sorted(directory.glob("*.jpeg"))
            if args.limit: selected = selected[:args.limit]
            if not selected: raise FileNotFoundError(directory)
            paths.extend(selected)
    print(f"Selected {len(paths)} images. Model: {SAM31_CHECKPOINT_PATH}", flush=True)
    timingPath = segmentImageBatchAndRecordTimings(paths, device=args.device, precision=args.precision,
        cpuThreadCount=args.threads, warmupPassCount=args.warmup)
    if not args.skip_selection:
        import selectTargetTree_SAM31 as selection
        selection.DATA_DIRECTORY = dataDirectory
        selection.SEG_RESULTS = OUTPUT_DIRECTORY
        selection.selectRun(timingPath.stem.removeprefix("timings_"), args.dataset)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"): stream.reconfigure(encoding="utf-8")
    main()
