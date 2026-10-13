"""Step 2: select a target tree from step 1 candidates using its RGB base point.

Replaces newPostProcess.ipynb. This step needs only NumPy and Pillow and may run
locally after downloading the candidate archives from the cloud. Output is an
RGB-aligned 8-bit PNG (background=0, tree=255) read directly by the C++ pipeline.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

# Shared CPU-only I/O helpers; importing this module does not import SAM3/Torch.
from sam3_batch_segment import (
    DATASETS, atomic_save, common_parser, configure_console, image_identity, list_images,
    load_candidates, load_rgb, resolve_paths, save_report,
)


def load_points(path: Path, dataset: str) -> dict[str, tuple[int, int]]:
    """Match the C++ duplicate policy: natural last-wins; plot duplicates are errors."""
    points = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"uniqueName", "baseCol", "baseRow"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Required columns: uniqueName,baseCol,baseRow in {path}")
        for row in reader:
            unique_name = row["uniqueName"].strip()
            point = (int(row["baseCol"]), int(row["baseRow"]))
            if not unique_name or min(point) < 0:
                raise ValueError(f"Invalid base point at {path}:{reader.line_num}")
            if unique_name in points:
                if dataset == "plot":
                    raise ValueError(f"Duplicate point ID: {unique_name}")
                print(f"Warning: duplicate natural ID {unique_name}; using the last row", file=sys.stderr)
            points[unique_name] = point
    return points


def choose_candidate(masks: np.ndarray, scores: np.ndarray, point: tuple[int, int]):
    """Inside a mask wins; otherwise minimize Euclidean distance to any mask pixel.

    Use pixels, not bounding-box centers. Overlaps/equal distances are resolved
    by higher SAM3 confidence, then lower candidate index, deterministically.
    Return the zero-based candidate index, distance in RGB pixels, and rule.
    """
    col, row = point
    if masks.ndim != 3 or scores.shape != (len(masks),):
        raise ValueError("Invalid candidate arrays")
    _, height, width = masks.shape
    if not (0 <= col < width and 0 <= row < height):
        raise ValueError(f"Base point {point} is outside RGB size {(width, height)}")
    inside = np.flatnonzero(masks[:, row, col])
    if len(inside):
        winner = min(inside, key=lambda i: (-float(scores[i]), int(i)))
        return int(winner), 0.0, "point_inside_mask"
    ranked = []
    for i, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if len(xs):
            distance2 = int(np.min((xs - col) ** 2 + (ys - row) ** 2))
            ranked.append((distance2, -float(scores[i]), i))
    if not ranked:
        raise ValueError("No nonempty candidate masks; no selected mask was written")
    distance2, _, winner = min(ranked)
    return winner, float(np.sqrt(distance2)), "minimum_mask_distance"


def read_existing_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image)
        if image.size != size or mask.ndim != 2 or mask.dtype != np.uint8:
            raise ValueError(f"Existing mask has wrong dimensions/type: {path}; use --overwrite")
        if not np.isin(mask, [0, 255]).all() or not np.any(mask):
            raise ValueError(f"Existing mask must be nonempty and binary 0/255: {path}; use --overwrite")
        return mask > 0


def save_overlay(path: Path, rgb: Image.Image, mask: np.ndarray, point: tuple[int, int]):
    pixels = np.array(rgb)
    pixels[mask] = (0.55 * pixels[mask] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    col, row = point
    radius = max(3, min(image.size) // 100)
    draw.ellipse((col - radius, row - radius, col + radius, row + radius), fill="yellow")
    atomic_save(path, lambda stream: image.save(stream, format="PNG"))


def main(argv=None) -> int:
    parser = common_parser("Step 2: select one SAM3 tree mask per RGB using the base-point CSV.")
    parser.add_argument("--points-csv", type=Path, help="Override points for a single --dataset")
    parser.add_argument("--max-distance", type=float,
                        help="Reject nearest-mask matches farther than this many RGB pixels; default: no limit")
    parser.add_argument("--save-overlays", action="store_true", help="Save red mask overlays with yellow base-point dots")
    args = parser.parse_args(argv)
    root, work, datasets = resolve_paths(args)
    if args.points_csv and args.dataset == "all":
        raise ValueError("--points-csv requires --dataset natural or --dataset plot")
    if args.max_distance is not None and (not np.isfinite(args.max_distance) or args.max_distance < 0):
        raise ValueError("--max-distance must be finite and nonnegative")
    inputs = {}
    for dataset in datasets:
        csv_path = args.points_csv or Path("data") / DATASETS[dataset][1]
        csv_path = csv_path.expanduser()
        if not csv_path.is_absolute():
            csv_path = root / csv_path
        inputs[dataset] = (list_images(root, dataset, args.limit), load_points(csv_path, dataset))

    failures = 0
    columns = ["uniqueName", "status", "baseCol", "baseRow", "candidate_id", "selection_rule",
               "distance_px", "confidence", "foreground_pixels", "message"]
    for dataset, (images, points) in inputs.items():
        report = []
        output_dir = root / "output/SAM3SegResults/previousSAM3Results" / DATASETS[dataset][2]
        for index, (unique_name, path) in enumerate(images, 1):
            record = dict.fromkeys(columns, "")
            record.update(uniqueName=unique_name, status="error")
            output = output_dir / f"{unique_name}_selected_mask.png"
            try:
                if unique_name not in points:
                    raise ValueError("Missing base point")
                point = points[unique_name]
                record.update(baseCol=point[0], baseRow=point[1])
                rgb = load_rgb(path)
                if not (0 <= point[0] < rgb.width and 0 <= point[1] < rgb.height):
                    raise ValueError(f"Base point {point} is outside RGB size {rgb.size}")
                keep_existing = output.exists() and not args.overwrite
                if keep_existing:
                    selected = read_existing_mask(output, rgb.size)
                    record.update(status="kept_existing", message="Selection not recomputed; use --overwrite to regenerate")
                else:
                    candidate_file = work / dataset / "candidates" / f"{unique_name}_candidates.npz"
                    identity = image_identity(path, dataset, unique_name, rgb.size)
                    masks, scores, _, _ = load_candidates(candidate_file, identity)
                    winner, distance, rule = choose_candidate(masks, scores, point)
                    if args.max_distance is not None and distance > args.max_distance:
                        raise ValueError(f"Nearest mask is {distance:.2f} px away, exceeding --max-distance")
                    selected = masks[winner]
                    record.update(status="selected", candidate_id=winner + 1, selection_rule=rule,
                                  distance_px=f"{distance:.6f}", confidence=f"{float(scores[winner]):.6f}")
                record["foreground_pixels"] = int(np.count_nonzero(selected))
                if args.save_overlays:
                    save_overlay(work / dataset / "selected_overlays" / f"{unique_name}_overlay.png", rgb, selected, point)
                if not keep_existing:
                    # Preserve EVERY region of this candidate. No largest-component
                    # filtering, gap filling, erosion, or union with neighboring trees.
                    image = Image.fromarray(selected.astype(np.uint8) * 255)
                    atomic_save(output, lambda stream: image.save(stream, format="PNG"))
            except Exception as exc:
                record.update(status="error", message=str(exc))
                if output.exists():
                    record["message"] += "; existing final mask retained"
                failures += 1
            report.append(record)
            print(f"[{dataset} {index}/{len(images)}] {unique_name}: {record['status']} {record['message']}")
        save_report(work / dataset / "selection_report.csv", report, columns)
        print(f"{dataset} final masks: {output_dir}")
    print(f"Step 2 finished: {failures} images need attention.")
    return 1 if failures else 0


if __name__ == "__main__":
    configure_console()
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
