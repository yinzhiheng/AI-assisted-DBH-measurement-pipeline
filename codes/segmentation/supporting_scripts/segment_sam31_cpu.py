"""Batch SAM 3.1 tree-image segmentation on CPU.

Examples (run with the SAM_3 Python environment):
    python segment_sam31_cpu.py --dry-run
    python segment_sam31_cpu.py --limit 3 --warmup 0
    python segment_sam31_cpu.py

The main CSV metric, model_inference_seconds, times image encoding, text
encoding, detection and mask prediction only. All disk I/O, model loading,
preprocessing, output resizing, drawing and file writing are excluded.
segmentation_seconds additionally includes preprocessing and postprocessing.
Each output's encoding/writing time is logged separately from segmentation.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import uuid

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# These settings precede ANY torch/SAM import, including indirect imports.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["USE_PERFLIB"] = "0"

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from PIL.PngImagePlugin import PngInfo

from sam31_cpu_backend import SAM31_SHA256, build_detector, file_sha256, forward_image

CODES = Path(__file__).resolve().parent
PIPELINE = CODES.parents[2]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OUTPUT_FORMAT_VERSION = 3
PNG_METADATA_KEY = "sam31_metadata"
SAVE_TIME_FIELDS = ("masks_save_seconds", "object_id_save_seconds", "overlay_save_seconds")
CSV_FIELDS = [
    "run_id", "dataset", "relative_image", "status", "width", "height",
    "prompt", "confidence_threshold", "candidate_count", "nonempty_count",
    "model_inference_seconds", "preprocess_seconds", "postprocess_seconds",
    "segmentation_seconds", *SAVE_TIME_FIELDS, "overlay_render_seconds",
    "first_inference_without_warmup", "cpu_threads", "precision",
    "checkpoint_sha256", "masks_path", "object_id_path", "overlay_path", "error",
]


def default_inputs() -> list[Path]:
    raw = PIPELINE / "data" / "rawdata"
    roots = []
    for category in ("naturalField", "plottedField"):
        # Accept both the actual on-disk names and the paths in the request.
        candidates = [raw / f"rgb_{category}", raw / "rgb" / f"_{category}"]
        roots.append(next((p for p in candidates if p.is_dir()), candidates[0]))
    return roots


def find_images(roots: list[Path], limit: int | None):
    groups = []
    labels = [root.name for root in roots]
    if len(set(labels)) != len(labels):
        raise ValueError("Input directories must have distinct names for unambiguous output paths.")
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {root}")
        images = sorted(
            (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS),
            key=lambda p: str(p.relative_to(root)).casefold(),
        )
        if not images:
            raise ValueError(f"No supported images found in {root}")
        names = {}
        for path in images:
            key = str(path.relative_to(root).with_suffix("")).casefold()
            if key in names:
                raise ValueError(
                    f"Images {names[key]} and {path} have the same output base name. "
                    "Give them different base names or put them in separate subfolders."
                )
            names[key] = path
        groups.append([(root, p) for p in images])
    # Round-robin means --limit 3 samples both datasets rather than just one.
    selected = []
    for index in range(max(map(len, groups))):
        for group in groups:
            if index < len(group):
                selected.append(group[index])
                if limit is not None and len(selected) >= limit:
                    return selected
    return selected


def preprocess(image: Image.Image):
    import torch
    # PIL bilinear resize matches the SAM 3.1 single-image resource loader.
    # Keep FP32 for CPU; no FP16 storage conversion / CUDA autocast.
    resized = image.resize((1008, 1008), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().unsqueeze(0)
    return tensor.sub_(0.5).div_(0.5)


def postprocess(predictions, width: int, height: int, threshold: float):
    """Retain every candidate above threshold; NO NMS or top-object selection.

    The combined original-resolution mask contains every candidate's foreground
    as 255, with background 0. It does not retain separate overlapping masks.
    The object-ID image resolves overlaps by confidence.
    """
    import torch
    import torch.nn.functional as F
    with torch.inference_mode():
        # Unlike build_sam3_image_model(), the SAM 3.1 detector is constructed
        # with supervise_joint_box_scores=True: presence is ALREADY included
        # in pred_logits. Multiplying it again would suppress valid objects.
        scores = predictions["pred_logits"].sigmoid().reshape(-1).float().cpu()
        logits = predictions["pred_masks"][0].float().cpu()
        boxes = predictions["pred_boxes"][0].float().cpu()
        if not (torch.isfinite(scores).all() and torch.isfinite(logits).all()
                and torch.isfinite(boxes).all()):
            raise ValueError("Non-finite model output; refusing to save invalid masks.")
        selected = torch.where(scores > threshold)[0]
        selected = selected[torch.argsort(scores[selected], descending=True, stable=True)]
        count = len(selected)
        if count > 65535:
            raise ValueError("Too many candidates for a uint16 object-ID image.")
        ids = np.arange(1, count + 1, dtype=np.uint16)
        label_image = np.zeros((height, width), dtype=np.uint16)
        areas = np.zeros(count, dtype=np.int64)
        # One full-resolution float mask at a time avoids a large N*H*W allocation.
        for index, query in enumerate(selected.tolist()):
            resized = F.interpolate(
                logits[query][None, None], size=(height, width),
                mode="bilinear", align_corners=False,
            )[0, 0]
            mask = (resized > 0).numpy()  # sigmoid(logit)>0.5, without an extra array
            areas[index] = int(mask.sum())
            label_image[mask & (label_image == 0)] = ids[index]
        cx, cy, bw, bh = boxes[selected].unbind(-1)
        xyxy = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), -1)
        xyxy *= torch.tensor([width, height, width, height])
        candidates = {
            "mask_union": (label_image > 0).astype(np.uint8) * np.uint8(255),
            "object_ids": ids,
            "query_indices": selected.numpy().astype(np.int32),
            "scores": scores[selected].numpy(),
            "boxes_xyxy": xyxy.numpy(),
            "areas_pixels": areas,
            "image_shape": np.array([height, width], dtype=np.int32),
            "all_query_scores": scores.numpy(),
        }
    return candidates, label_image


def load_candidate_masks(path: str | Path, object_id: int | None = None):
    """Read LEGACY NPZ archives as uint8 0/255 individual candidate masks.

    Also reads older packed archives, converting their bits to 0/255 values.
    New combined PNGs are read with PIL.Image.open; they contain one HxW mask.
    """
    with np.load(path, allow_pickle=False) as data:
        legacy = "masks" not in data
        masks = data["masks_packed" if legacy else "masks"]
        if object_id is not None:
            positions = np.flatnonzero(data["object_ids"] == object_id)
            if not len(positions):
                raise KeyError(f"Object ID {object_id} is not in {path}")
            masks = masks[int(positions[0])]
        if legacy:
            masks = np.unpackbits(masks, axis=-1, count=int(data["image_shape"][1]),
                                  bitorder="little") * np.uint8(255)
        return masks


def make_overlay(image: Image.Image, labels: np.ndarray, candidates) -> Image.Image:
    ids = candidates["object_ids"]
    palette = np.zeros((len(ids) + 1, 3), dtype=np.uint8)
    for object_id in ids:
        color = colorsys.hsv_to_rgb((int(object_id) * 0.61803398875) % 1, 0.8, 1.0)
        palette[int(object_id)] = np.round(np.array(color) * 255).astype(np.uint8)
    rgb = np.array(image, copy=True)
    # Work by rows to avoid several full-size floating-point RGB arrays.
    for start in range(0, len(labels), 256):
        lab = labels[start:start + 256]
        tile = rgb[start:start + 256]
        active = lab > 0
        tile[active] = np.round(tile[active] * 0.55 + palette[lab[active]] * 0.45).astype(np.uint8)
    result = Image.fromarray(rgb)
    draw = ImageDraw.Draw(result)
    size = max(14, round(min(image.size) / 70))
    try:
        font = ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        font = ImageFont.load_default(size=size)
    for index, object_id in enumerate(ids):
        ys, xs = np.nonzero(labels == object_id)
        if len(xs) == 0:
            continue  # a completely covered candidate has no visible ID pixels
        # Choose an actual labeled pixel near the median position.
        anchor = np.argmin((xs - np.median(xs)) ** 2 + (ys - np.median(ys)) ** 2)
        text = f"{int(object_id)} ({candidates['scores'][index]:.2f})"
        # Keep labels readable even when a candidate touches an image edge.
        left, top, right, bottom = draw.textbbox(
            (0, 0), text, font=font, stroke_width=2, anchor="mm"
        )
        cx = int(np.clip(xs[anchor], -left, max(-left, image.width - right)))
        cy = int(np.clip(ys[anchor], -top, max(-top, image.height - bottom)))
        draw.text((cx, cy), text, fill="white", font=font,
                  stroke_width=2, stroke_fill="black", anchor="mm")
    return result


def output_files(output: Path, root: Path, image_path: Path):
    relative = image_path.relative_to(root)
    # Strip only the final input extension; find_images rejects stem collisions.
    return [output / folder / root.name / relative.parent / (relative.stem + suffix)
            for folder, suffix in (("masks", "_masks.png"),
                                   ("object_id", "_object_ids.png"),
                                   ("overlay", "_overlay.png"))]


def write_outputs(paths, image, candidates, labels, prompt, threshold, digest, run_id):
    """Return separate encoding/write times; overlay drawing has its own timer.

    Save times include compression/encoding, file writing/closing and the final
    rename. They exclude folder creation and overlay rendering, and do not wait
    for operating-system cache flushing or background OneDrive synchronization.
    """
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = [path.with_name(path.name + f".{run_id}.tmp") for path in paths]
    timings = {}
    try:
        start = perf_counter()
        # Embed candidate/settings metadata in the PNG, keeping three files.
        metadata = {key: value.tolist() for key, value in candidates.items()
                    if key != "mask_union"}
        metadata.update(format_version=OUTPUT_FORMAT_VERSION, mask_values=[0, 255],
                        mask_semantics="union of all retained candidate masks",
                        prompt=prompt, confidence_threshold=threshold,
                        checkpoint_sha256=digest, run_id=run_id)
        png_info = PngInfo()
        png_info.add_text(PNG_METADATA_KEY, json.dumps(metadata, separators=(",", ":")))
        Image.fromarray(candidates["mask_union"]).save(temporary[0], format="PNG", pnginfo=png_info)
        timings["masks_save_seconds"] = perf_counter() - start

        id_image = Image.fromarray(labels)
        start = perf_counter()
        id_image.save(temporary[1], format="PNG", pnginfo=png_info)
        timings["object_id_save_seconds"] = perf_counter() - start

        start = perf_counter()
        overlay = make_overlay(image, labels, candidates)
        timings["overlay_render_seconds"] = perf_counter() - start
        start = perf_counter()
        overlay.save(temporary[2], format="PNG", pnginfo=png_info)
        timings["overlay_save_seconds"] = perf_counter() - start

        for field, temp, final in zip(SAVE_TIME_FIELDS, temporary, paths):
            start = perf_counter()
            os.replace(temp, final)
            timings[field] += perf_counter() - start
        return timings
    finally:
        for temp in temporary:
            temp.unlink(missing_ok=True)


def read_output_metadata(path: str | Path):
    """Read embedded settings/candidate metadata without decoding image pixels."""
    with Image.open(path) as image:
        return json.loads(image.info[PNG_METADATA_KEY])


def append_csv(path, row):
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=PIPELINE / "model/SAM31/source")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--input-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=PIPELINE / "output/SAM3SegResults/legacy_runner")
    parser.add_argument("--prompt", default="tree trunk", help="All candidates matching this text; default: tree trunk")
    parser.add_argument("--threshold", type=float, default=0.3, help="Retain scores above this value; 0 exports all positive-score queries")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None, help="Maximum images, sampled across both datasets")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed model passes on the first image; default 1; use 0 for exactly --limit passes")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs; otherwise existing files are protected")
    parser.add_argument("--dry-run", action="store_true", help="List paths and first images without loading a model")
    parser.add_argument("--check-env", action="store_true", help="Check imports/CPU compatibility without loading weights or running images")
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1 or args.threads < 1 or args.warmup < 0:
        parser.error("Require threshold in [0,1], positive threads, and nonnegative warmup.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if not args.prompt.strip():
        parser.error("--prompt cannot be empty.")
    args.checkpoint = (args.checkpoint or PIPELINE / "model/SAM31/checkpoints/sam3.1_multiplex.pt").resolve()
    args.input_dirs = [p.resolve() for p in (args.input_dirs or default_inputs())]
    args.source, args.output = args.source.resolve(), args.output.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.check_env:
        from sam31_cpu_backend import import_sam31
        torch, _ = import_sam31(args.source)
        print(f"Python: {sys.executable}\nPyTorch: {torch.__version__}\nCUDA visible: {torch.cuda.is_available()}")
        print(f"SAM 3.1 imports OK. Checkpoint exists: {args.checkpoint.is_file()}")
        return 0
    images = find_images(args.input_dirs, args.limit)
    print(f"Selected {len(images)} image(s). Prompt: {args.prompt!r}. Device: CPU / FP32.")
    for root in args.input_dirs:
        print(f"Input: {root}")
    print(f"Output: {args.output}\nCheckpoint: {args.checkpoint}")
    if args.dry_run:
        for root, path in images[:12]:
            print(f"  {root.name}/{path.relative_to(root)}")
        print(f"Checkpoint exists: {args.checkpoint.is_file()}. No inference performed.")
        return 0
    if not args.checkpoint.is_file():
        print(
            "\nSAM 3.1 checkpoint is missing. Download sam3.1_multiplex.pt from\n"
            "https://huggingface.co/facebook/sam3.1/tree/main\n"
            f"Save it to: {args.checkpoint}\nNo images were processed.", file=sys.stderr,
        )
        return 2

    pending = []
    for root, path in images:
        files = output_files(args.output, root, path)
        if any(p.exists() for p in files) and not args.overwrite:
            if all(p.exists() for p in files):
                # Only resume outputs generated with the same model and prompt/settings.
                saved_outputs = [read_output_metadata(p) for p in files]
                matches = all(
                    saved.get("format_version") == OUTPUT_FORMAT_VERSION
                    and saved["checkpoint_sha256"] == SAM31_SHA256
                    and saved["prompt"] == args.prompt
                    and saved["confidence_threshold"] == args.threshold
                    and saved["run_id"] == saved_outputs[0]["run_id"]
                    for saved in saved_outputs
                )
                if not matches:
                    raise ValueError(f"Existing outputs use different settings: {files[0]}. Choose another --output or --overwrite.")
                print(f"Already complete, skipping: {path.name}")
                continue
            raise FileExistsError(f"Partial output files for {path.name}; use --overwrite to repair them.")
        pending.append((root, path))
    if not pending:
        print("All selected images already have matching outputs.")
        return 0

    model, info = build_detector(args.source, args.checkpoint, args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    info.update({
        "run_id": run_id, "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "cpu": platform.processor(),
        "logical_cpu_count": os.cpu_count(), "prompt": args.prompt,
        "confidence_threshold": args.threshold, "input_resolution": [1008, 1008],
        "batch_size": 1, "warmup_passes": args.warmup,
        "input_directories": list(map(str, args.input_dirs)),
        "selected_images": len(images), "pending_images": len(pending),
        "completed": 0, "failed": 0, "status": "running",
        "timing_definition": "model_inference_seconds: image encoder + text encoder + grounding + mask heads; excludes preprocessing, postprocessing, all I/O, load, warmup, and visualization",
        "save_timing_definition": "masks_save_seconds, object_id_save_seconds, overlay_save_seconds: each file's compression/encoding, write, close and final rename; excludes folder creation, overlay drawing, OS cache flush and background OneDrive sync",
        "overlay_render_timing_definition": "overlay_render_seconds: RGB blending and ID/score drawing, separate from PNG saving",
        "output_format_version": OUTPUT_FORMAT_VERSION,
        "output_layout": "output/{masks,object_id,overlay}/dataset/relative_path",
        "output_filename_pattern": "input stem (without final extension) + output suffix + .png",
        "mask_encoding": "One combined grayscale PNG per image: uint8 [H,W], union foreground=255, background=0",
        "candidate_policy": "All queries with presence-adjusted score above threshold; no NMS, top-k or target-tree selection; PNG combines foreground, individual overlaps are not separately recoverable",
        "id_policy": "IDs restart per image, descending confidence; background=0; highest-confidence wins overlap in ID PNG only",
    })
    try:
        import psutil
        info.update({"physical_cpu_count": psutil.cpu_count(logical=False),
                     "ram_bytes": psutil.virtual_memory().total,
                     "available_ram_bytes_at_start": psutil.virtual_memory().available})
    except ImportError:
        pass
    info["source_file_sha256"] = {
        str(path.relative_to(args.source)): file_sha256(path)
        for path in sorted((args.source / "sam3").rglob("*.py"))
    }
    metadata_path = args.output / f"run_{run_id}.json"
    csv_path = args.output / f"timings_{run_id}.csv"
    metadata_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    first_call = True
    times = []
    try:
        for index, (root, path) in enumerate(pending, 1):
            row = {"run_id": run_id, "dataset": root.name,
                   "relative_image": str(path.relative_to(root)), "prompt": args.prompt,
                   "confidence_threshold": args.threshold, "cpu_threads": args.threads,
                   "precision": "float32", "checkpoint_sha256": info["checkpoint_sha256"]}
            print(f"[{index}/{len(pending)}] {root.name}/{path.name}", flush=True)
            predictions = tensor = candidates = labels = image = None
            try:
                # File decoding and EXIF orientation are intentionally outside timing.
                with Image.open(path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    image.load()
                width, height = image.size
                row.update(width=width, height=height)
                start = perf_counter()
                tensor = preprocess(image)
                pre_seconds = perf_counter() - start
                if first_call and args.warmup:
                    for warm in range(args.warmup):
                        print(f"  Untimed warm-up {warm + 1}/{args.warmup} (may take several minutes)...", flush=True)
                        warm_outputs = forward_image(model, tensor, args.prompt)
                        del warm_outputs
                row["first_inference_without_warmup"] = int(first_call and args.warmup == 0)
                print("  Segmenting on CPU...", flush=True)
                start = perf_counter()
                predictions = forward_image(model, tensor, args.prompt)
                inference_seconds = perf_counter() - start
                first_call = False
                start = perf_counter()
                candidates, labels = postprocess(predictions, width, height, args.threshold)
                post_seconds = perf_counter() - start
                del predictions
                predictions = None
                files = output_files(args.output, root, path)
                save_timings = write_outputs(files, image, candidates, labels, args.prompt,
                                             args.threshold, info["checkpoint_sha256"], run_id)
                row.update({key: f"{value:.6f}" for key, value in save_timings.items()})
                row.update(
                    status="ok", candidate_count=len(candidates["object_ids"]),
                    nonempty_count=int(np.count_nonzero(candidates["areas_pixels"])),
                    model_inference_seconds=f"{inference_seconds:.6f}",
                    preprocess_seconds=f"{pre_seconds:.6f}",
                    postprocess_seconds=f"{post_seconds:.6f}",
                    segmentation_seconds=f"{pre_seconds + inference_seconds + post_seconds:.6f}",
                    masks_path=str(files[0]), object_id_path=str(files[1]),
                    overlay_path=str(files[2]), error="",
                )
                times.append(inference_seconds)
                info["completed"] += 1
                print(f"  {inference_seconds:.3f} s model inference; {row['candidate_count']} candidates. Saved 3 files.", flush=True)
                print("  Save times: " + ", ".join(
                    f"{field.removesuffix('_save_seconds')}={save_timings[field]:.3f} s"
                    for field in SAVE_TIME_FIELDS), flush=True)
            except Exception as error:
                row.update(status="error", error=f"{type(error).__name__}: {error}")
                info["failed"] += 1
                append_csv(csv_path, row)
                # Fail early instead of repeating a model/runtime error on the whole dataset.
                raise
            finally:
                predictions = tensor = candidates = labels = image = None
                gc.collect()  # excluded from all timings
            append_csv(csv_path, row)
        info["status"] = "complete"
    except BaseException:
        info["status"] = "interrupted_or_failed"
        raise
    finally:
        info["ended_utc"] = datetime.now(timezone.utc).isoformat()
        if times:
            info["mean_model_inference_seconds"] = float(np.mean(times))
            info["median_model_inference_seconds"] = float(np.median(times))
        metadata_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Finished. Per-image timing log: {csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user. Finished image outputs and timing rows remain saved.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        import traceback
        traceback.print_exc()
        print(f"\nERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
