"""Select SAM31 target trees and publish masks directly to the C++ input folder.

Selection uses the saved object-ID map and preserves the existing base-point rule.
No model inference or manual transfer is required. Candidate archives additionally
retain all individual overlapping masks for future analysis.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data"
SEG_RESULTS = PROJECT_ROOT / "output/SAM3SegResults"
DATASETS = {"natural": ("rgb_naturalField", "basePoints_natural.csv"),
            "plotted": ("rgb_plottedField", "basePoints_plotted.csv")}
OVERLAY_COLOR = (255, 0, 0)
OVERLAY_OPACITY = 0.45
SHOW_BASE_POINT = True


def readTappedBasePointsForDataset(dataset):
    points = {}
    with (DATA_DIRECTORY / DATASETS[dataset][1]).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["uniqueName"].strip()
            if key in points and dataset == "plotted":
                raise ValueError(f"Duplicate plotted-field base point: {key}")
            points[key] = (int(row["baseCol"]), int(row["baseRow"]))
    return points


def selectTreeObjectNearestToTappedBasePoint(objectIdImage, basePoint, scoresByObjectId):
    """Return the selected object ID, distance in RGB pixels and selection rule.

    A point inside a tree selects that tree. Otherwise, measure the straight-line
    distance to actual mask pixels, not to a bounding box or the tree's center.
    Equal distances use higher confidence, then lower object ID, as in the
    original select_target_tree.py. Confidence is never drawn on the overlay.
    """

    # 1. Interpret the CSV coordinates as (column, row), measured from the
    # top-left RGB pixel. Reject an outside tap instead of silently moving it.
    baseColumn, baseRow = basePoint
    imageHeight, imageWidth = objectIdImage.shape
    if not (0 <= baseColumn < imageWidth and 0 <= baseRow < imageHeight):
        raise ValueError(f"Base point {basePoint} is outside RGB size {(imageWidth, imageHeight)}.")

    # 2. If the tap lies on an object, that saved object is the target.
    objectIdAtPoint = int(objectIdImage[baseRow, baseColumn])
    if objectIdAtPoint > 0:
        return objectIdAtPoint, 0.0, "point_inside_mask"

    # 3. For a tap on the background, find the nearest foreground pixel.
    # Searching all foreground pixels once also considers every disconnected
    # piece of each tree. Squared distances give the same nearest ordering.
    pixelRows, pixelColumns = np.nonzero(objectIdImage)
    if len(pixelColumns) == 0:
        raise ValueError("No tree pixels in the saved mask; no target mask was written.")
    squaredDistances = (pixelColumns - baseColumn) ** 2 + (pixelRows - baseRow) ** 2
    minimumSquaredDistance = int(squaredDistances.min())

    # 4. Several objects can be equally close. Apply the original deterministic
    # tie rule, using the scores embedded in the saved SAM object-ID PNG.
    nearestPixels = squaredDistances == minimumSquaredDistance
    tiedObjectIds = np.unique(objectIdImage[pixelRows[nearestPixels], pixelColumns[nearestPixels]])
    selectedObjectId = min(
        (int(objectId) for objectId in tiedObjectIds),
        key=lambda objectId: (-float(scoresByObjectId[objectId]), objectId),
    )
    return selectedObjectId, float(np.sqrt(minimumSquaredDistance)), "minimum_mask_distance"


def createSelectedTreeRgbOverlayWithoutText(rgbImage, selectedTreeMask, basePoint):
    """Highlight the selected tree, optionally mark the tap, and draw no text."""

    # 1. Blend color into selected pixels only; keep the surrounding RGB intact.
    overlayPixels = np.array(rgbImage, copy=True)
    overlayPixels[selectedTreeMask] = np.round(
        (1 - OVERLAY_OPACITY) * overlayPixels[selectedTreeMask]
        + OVERLAY_OPACITY * np.array(OVERLAY_COLOR)
    ).astype(np.uint8)
    overlayImage = Image.fromarray(overlayPixels)

    # 2. Optionally show where the user tapped. The dot belongs only to the
    # visualization and never changes the binary tree mask.
    if SHOW_BASE_POINT:
        baseColumn, baseRow = basePoint
        dotRadius = max(3, min(rgbImage.size) // 100)
        imageDrawing = ImageDraw.Draw(overlayImage)
        imageDrawing.ellipse(
            (baseColumn - dotRadius, baseRow - dotRadius,
             baseColumn + dotRadius, baseRow + dotRadius),
            fill="yellow",
        )
    return overlayImage


def latestRun(dataset="all"):
    # Require a completed run covering the requested datasets. The per-image
    # provenance report records partial updates from limited runs.
    configs = list((SEG_RESULTS / "reports").rglob("*_configuration_*.json"))
    completed = []
    for path in configs:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        runId = config.get("run_id", "")
        if config.get("status") == "complete" and all(
            any((SEG_RESULTS / "candidates" / runId / name / "masks").glob("*.png"))
            for name in DATASETS if dataset in ("all", name)
        ):
            completed.append(runId)
    if not completed:
        raise FileNotFoundError("No completed run found; pass --run-id explicitly.")
    return max(completed, key=lambda value: value.rsplit("_", 1)[-1])


def selectRun(runId, dataset="all", limit=0):
    if Path(runId).name != runId or runId in (".", ".."):
        raise ValueError("run-id must be a single folder name")
    reportPath = SEG_RESULTS / "reports/target_selection.json"
    report = json.loads(reportPath.read_text(encoding="utf-8")) if reportPath.exists() else {"records": {}}
    errors, selected = [], []
    for name, (rgbFolder, _) in DATASETS.items():
        if dataset not in ("all", name):
            continue
        candidateDir = SEG_RESULTS / "candidates" / runId / name
        maskPaths = sorted((candidateDir / "masks").glob("*_masks.png"))
        if limit: maskPaths = maskPaths[:limit]
        if not maskPaths:
            raise FileNotFoundError(f"No candidates: {candidateDir}")
        points = readTappedBasePointsForDataset(name)
        records = report["records"].setdefault(name, {})
        for index, maskPath in enumerate(maskPaths, 1):
            stem = maskPath.name.removesuffix("_masks.png")
            rgbStem = stem.rsplit("__", 1)[0]
            uniqueName = rgbStem.removeprefix("rgb-")
            record = {"uniqueName": uniqueName, "dataset": name, "source_run_id": runId,
                      "source_mask": str(maskPath.relative_to(PROJECT_ROOT))}
            try:
                with Image.open(maskPath) as image: combined = np.asarray(image)
                idPath = candidateDir / "object_id" / f"{stem}_object_ids.png"
                with Image.open(idPath) as image:
                    ids = np.asarray(image)
                    metadata = json.loads(image.info["sam31_metadata"])
                with Image.open(DATA_DIRECTORY / "rawdata" / rgbFolder / f"{rgbStem}.jpeg") as image:
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                if ids.shape != (rgb.height, rgb.width) or combined.shape != ids.shape:
                    raise ValueError("RGB and mask dimensions do not match")
                if not np.array_equal(combined > 0, ids > 0):
                    raise ValueError("Combined mask and object IDs disagree")
                scores = dict(zip(metadata["object_ids"], metadata["scores"]))
                base = points[uniqueName]
                objectId, distance, rule = selectTreeObjectNearestToTappedBasePoint(ids, base, scores)
                target = ids == objectId
                maskOut = SEG_RESULTS / "targetTreeMasks" / name / f"{uniqueName}_selected_mask.png"
                overlayOut = SEG_RESULTS / "targetTreeOverlays" / name / f"{uniqueName}_selected_overlay.png"
                maskOut.parent.mkdir(parents=True, exist_ok=True)
                overlayOut.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(target.astype(np.uint8) * 255).save(maskOut)
                createSelectedTreeRgbOverlayWithoutText(rgb, target, base).save(overlayOut)
                record.update(status="selected", object_id=int(objectId), distance_pixels=distance,
                              selection_rule=rule, foreground_pixels=int(target.sum()),
                              target_mask=str(maskOut.relative_to(PROJECT_ROOT)))
                selected.append(record)
            except (ValueError, KeyError, OSError) as error:
                record.update(status="error", error=str(error))
                errors.append(record)
            records[uniqueName] = record
            print(f"[{name} {index}/{len(maskPaths)}] {uniqueName}: {record['status']}", flush=True)
        report["updated_utc"] = datetime.now(timezone.utc).isoformat()
        reportPath.parent.mkdir(parents=True, exist_ok=True)
        reportPath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected {len(selected)} target trees; {len(errors)} errors.", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} target selections failed; see {reportPath}")
    return selected


def main():
    global DATA_DIRECTORY, SEG_RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Candidate run folder; defaults to latest completed run")
    parser.add_argument("--dataset", choices=["all", "natural", "plotted"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="Images per dataset; 0 means all")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=SEG_RESULTS)
    args = parser.parse_args()
    DATA_DIRECTORY = (PROJECT_ROOT / args.data_dir).resolve()
    SEG_RESULTS = (PROJECT_ROOT / args.output_dir).resolve()
    if args.limit < 0: parser.error("limit must be nonnegative")
    selectRun(args.run_id or latestRun(args.dataset), args.dataset, args.limit)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"): stream.reconfigure(encoding="utf-8")
    main()
