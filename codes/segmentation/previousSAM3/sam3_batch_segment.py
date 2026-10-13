"""Step 1: batch text-prompt SAM3 segmentation on a CUDA/cloud machine.

Replaces the batch section of sam3_image_predictor_example.ipynb. Each image
produces one compressed candidate archive, not a final target-tree mask.
Run select_target_tree.py next. See README.md in this directory for commands.

NumPy/Pillow utilities are shared with step 2. PyTorch and SAM3 are imported
only when inference is needed, so selection and --dry-run work without them.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image, ImageOps


DEFAULT_ROOT = Path(__file__).resolve().parents[3]
# Paths and point coordinates match the C++ batch programs.
DATASETS = {
    "natural": ("rgb_naturalField", "basePoints_natural.csv", "natural"),
    "plot": ("rgb_plottedField", "basePoints_plotted.csv", "plotted"),
}
FORMAT_VERSION = 1


def configure_console() -> None:
    """Keep Windows paths in consoles and redirected logs."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT,
                        help="DBH-pipeline-whole repository root")
    parser.add_argument("--dataset", choices=["natural", "plot", "all"], default="all")
    parser.add_argument("--candidates-dir", type=Path,
                        help="Intermediate parent folder; default: output/SAM3SegResults/previousSAM3Candidates")
    parser.add_argument("--limit", type=int, default=0, help="First N images per dataset; 0 means all")
    parser.add_argument("--overwrite", action="store_true", help="Replace this step's existing outputs")
    return parser


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, list[str]]:
    root = args.project_root.expanduser().resolve()
    work = args.candidates_dir or Path("output/SAM3SegResults/previousSAM3Candidates")
    work = work.expanduser()
    work = (work if work.is_absolute() else root / work).resolve()
    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    return root, work, list(DATASETS) if args.dataset == "all" else [args.dataset]


def list_images(root: Path, dataset: str, limit: int = 0) -> list[tuple[str, Path]]:
    folder = root / "data/rawdata" / DATASETS[dataset][0]
    found = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or not path.name.startswith("rgb-"):
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        unique_name = path.stem[4:]
        if not unique_name or unique_name in found:
            raise ValueError(f"Empty or duplicate image ID: {path.name}")
        found[unique_name] = path
    if not found:
        raise ValueError(f"No rgb-*.jpeg/jpg/png images found in {folder}")
    items = sorted(found.items())
    return items[:limit] if limit else items


def load_rgb(path: Path) -> Image.Image:
    """Use displayed RGB coordinates, including EXIF rotation, as in the notebooks."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_identity(path: Path, dataset: str, unique_name: str, size: tuple[int, int]) -> dict:
    return {"format_version": FORMAT_VERSION, "dataset": dataset, "uniqueName": unique_name,
            "rgb_sha256": file_sha256(path), "rgb_size": list(size),
            "coordinates": "EXIF-oriented full-resolution RGB pixels"}


def atomic_save(path: Path, writer) -> None:
    """Replace only a completed file; an interrupted write cannot create a valid-looking output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".writing-", delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_report(path: Path, rows: list[dict], columns: list[str]) -> None:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_save(path, lambda stream: stream.write(text.getvalue().encode("utf-8")))


def load_candidates(path: Path, expected: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Validate the archive against its RGB image before using any candidate pixels."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Candidate {key} mismatch in {path}; regenerate step 1")
        masks, scores, boxes = archive["masks"], archive["scores"], archive["boxes"]
    width, height = expected["rgb_size"]
    if masks.ndim != 3 or masks.shape[1:] != (height, width) or masks.dtype != np.bool_:
        raise ValueError(f"Expected boolean masks [N,{height},{width}] in {path}")
    count = len(masks)
    if scores.shape != (count,) or boxes.shape != (count, 4):
        raise ValueError(f"Inconsistent candidate counts in {path}")
    if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
        raise ValueError(f"Non-finite scores/boxes in {path}")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"Scores must be probabilities in {path}")
    return masks, scores, boxes, metadata


def restore_predictions(masks: np.ndarray, scores: np.ndarray, boxes: np.ndarray,
                        input_size: tuple[int, int], rgb_size: tuple[int, int]):
    """Nearest-neighbor masks and scaled XYXY boxes return to original RGB coordinates.

    Keep the full mask of every detection, including disconnected parts. Separate
    masks also preserve overlap pixels that a single instance-ID image would lose.
    """
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[1:] != (input_size[1], input_size[0]):
        raise ValueError(f"Unexpected SAM3 mask shape: {masks.shape}")
    if scores.shape != (len(masks),) or boxes.shape != (len(masks), 4):
        raise ValueError("SAM3 mask, score and box counts differ")
    if not np.isfinite(masks).all() or not np.isfinite(scores).all() or not np.isfinite(boxes).all():
        raise ValueError("SAM3 returned non-finite predictions")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("SAM3 scores must be probabilities in [0,1]")
    restored = np.empty((len(masks), rgb_size[1], rgb_size[0]), dtype=bool)
    for i, mask in enumerate(masks):
        binary = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
        restored[i] = np.asarray(binary.resize(rgb_size, Image.Resampling.NEAREST)) > 0
    boxes = boxes.astype(np.float32, copy=True)
    boxes[:, [0, 2]] *= rgb_size[0] / input_size[0]
    boxes[:, [1, 3]] *= rgb_size[1] / input_size[1]
    # Empty detections cannot identify a tree. Keep arrays aligned after filtering.
    keep = restored.any(axis=(1, 2))
    return restored[keep], scores[keep].astype(np.float32), boxes[keep]


class Sam3Runner:
    """One model/processor for the run, with a fresh inference state for each RGB."""

    def __init__(self, args: argparse.Namespace):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if not torch.cuda.is_available():
            raise RuntimeError("Step 1 requires the configured CUDA/cloud SAM3 environment")
        self.torch = torch
        precision = args.precision
        if precision == "auto":
            precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float32"
        if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bfloat16; use --precision float32")
        self.dtype = None if precision == "float32" else getattr(torch, precision)
        self.model = build_sam3_image_model(
            device="cuda", eval_mode=True,
            checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
            load_from_HF=args.checkpoint is None,
        )
        self.processor = Sam3Processor(self.model, device="cuda", confidence_threshold=args.confidence)
        print(f"SAM3 loaded; precision={precision}")

    def predict(self, image: Image.Image, prompt: str):
        state = None
        try:
            amp = self.torch.autocast("cuda", dtype=self.dtype) if self.dtype else nullcontext()
            with self.torch.inference_mode(), amp:
                state = self.processor.set_image(image)
                state = self.processor.set_text_prompt(state=state, prompt=prompt)
            # Processor masks are already boolean, not raw mask logits.
            return (state["masks"].detach().cpu().numpy(),
                    state["scores"].detach().float().cpu().numpy(),
                    state["boxes"].detach().float().cpu().numpy())
        finally:
            del state
            self.torch.cuda.empty_cache()


def main(argv=None) -> int:
    parser = common_parser("Step 1: save SAM3 tree-trunk candidates for each RGB image.")
    parser.add_argument("--prompt", default="tree trunk")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--max-side", type=int, default=1024, help="Maximum pre-inference image side in pixels")
    parser.add_argument("--checkpoint", type=Path, help="Local SAM3 .pt file; otherwise use authenticated Hugging Face access")
    parser.add_argument("--precision", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Check input discovery only; no model loading or file writes")
    args = parser.parse_args(argv)
    root, work, datasets = resolve_paths(args)
    if not 0 <= args.confidence <= 1 or args.max_side < 1 or not args.prompt.strip():
        raise ValueError("Use confidence in [0,1], a positive --max-side, and a nonempty prompt")
    images = {d: list_images(root, d, args.limit) for d in datasets}
    for dataset, items in images.items():
        print(f"{dataset}: {len(items)} images; candidates -> {work / dataset / 'candidates'}")
    if args.dry_run:
        return 0
    if args.checkpoint:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_id = file_sha256(args.checkpoint) if args.checkpoint else "facebook/sam3 (Hugging Face)"
    settings = {"prompt": args.prompt, "confidence": args.confidence, "max_side": args.max_side,
                "precision": args.precision, "checkpoint": checkpoint_id}
    runner = None
    failures = 0
    for dataset, items in images.items():
        report = []
        for index, (unique_name, path) in enumerate(items, 1):
            record = {"uniqueName": unique_name, "status": "error", "candidate_count": "", "message": ""}
            setting_up_model = False
            try:
                rgb = load_rgb(path)
                identity = image_identity(path, dataset, unique_name, rgb.size)
                expected = {**identity, "settings": settings}
                output = work / dataset / "candidates" / f"{unique_name}_candidates.npz"
                if output.exists() and not args.overwrite:
                    masks, _, _, _ = load_candidates(output, expected)
                    record.update(status="kept_existing", candidate_count=len(masks))
                else:
                    # A model setup failure stops the run; image-specific errors below
                    # are reported and skipped so the remaining images can proceed.
                    if runner is None:
                        setting_up_model = True
                        runner = Sam3Runner(args)
                        setting_up_model = False
                    scale = min(1.0, args.max_side / max(rgb.size))
                    size = (max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale)))
                    input_image = rgb.resize(size, Image.Resampling.LANCZOS) if size != rgb.size else rgb
                    predictions = runner.predict(input_image, args.prompt)
                    masks, scores, boxes = restore_predictions(*predictions, input_image.size, rgb.size)
                    atomic_save(output, lambda stream: np.savez_compressed(
                        stream, masks=masks, scores=scores, boxes=boxes,
                        metadata=np.asarray(json.dumps(expected, sort_keys=True))))
                    record.update(status="saved", candidate_count=len(masks))
                if not len(masks):
                    record.update(status="no_candidates", message="No tree-trunk masks; review prompt/image")
                    failures += 1
            except Exception as exc:
                record["message"] = str(exc)
                failures += 1
                # A missing package, checkpoint or model setup error affects every image.
                if setting_up_model:
                    report.append(record)
                    save_report(work / dataset / "segmentation_report.csv", report, list(record))
                    raise
            report.append(record)
            print(f"[{dataset} {index}/{len(items)}] {unique_name}: {record['status']} {record['message']}")
        save_report(work / dataset / "segmentation_report.csv", report,
                    ["uniqueName", "status", "candidate_count", "message"])
    print(f"Step 1 finished: {failures} images need attention. Run select_target_tree.py next.")
    return 1 if failures else 0


if __name__ == "__main__":
    configure_console()
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
