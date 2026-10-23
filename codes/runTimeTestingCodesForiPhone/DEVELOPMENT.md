# Runtime Test for iPhone

A separate native iOS app for benchmarking the MobileSAMv2 segmentation path in
iForester. This published folder builds independently of the original iForester checkout.
This folder is self-contained: Swift UI, copied C++ pipeline with the GitHub
breast-height/edge-seed recovery update, both ONNX models,
OpenCV 4.8.0, ONNX Runtime 1.21.0, and the complete dataset are included. No Flutter,
CocoaPods installation, Python runtime, server, or network connection is needed
on the phone.

## Install and run on your iPhone

1. Open **RuntimeTest.xcodeproj** in Xcode 16.4 or a compatible newer Xcode.
2. Connect and unlock your iPhone, trust the Mac if prompted, and enable
   **Settings → Privacy & Security → Developer Mode** if Xcode requires it.
   The app supports iOS 15.6 or newer; use an Xcode version that supports your
   phone's installed iOS version.
3. Select the **RuntimeTest** scheme and your **physical iPhone** as the run
   destination. This project intentionally targets `iphoneos` only. The copied
   frameworks contain the same device code as iForester; simulator measurements
   are not supported.
4. Under **RuntimeTest target → Signing & Capabilities**, select your Apple
   development team and leave **Automatically manage signing** enabled. The
   development team is intentionally unset. The initial bundle ID is
   `com.yzhiheng.iForesterRuntimeTest`; change it if your signing account requires it. This installs a separate app alongside iForester.
5. The shared scheme already uses **Release** with **Debug executable disabled**.
   Press **Run**. All images, depth maps and metadata are copied to the phone as
   app resources during installation. The built app is approximately 0.9 GB;
   allow several GB of free phone storage for installation and working space.
   The first installation can take a few minutes over USB.
6. After installation, launch **Runtime Test** directly from the iPhone Home
   Screen. You can unplug the phone. The screen should say **487 images ready
   offline · 285 natural + 202 plotted**. No Photos import, Files import, download
   button or storage permission is necessary.
7. **Select image → Segment selected** runs the full saved-base pipeline for
   one image. The image preview shows the box and segmented region. **Batch all 487**
   processes every image in deterministic natural-then-plotted order, with a
   **200 ms pause between images**, excluded from segmentation timing.
8. Keep the app visible. It prevents automatic screen sleep while processing.
   **Stop after current image** preserves completed rows. Leaving the app also
   requests a stop after the current image; iOS can suspend execution in the
   background. An interrupted current image is not represented as a completed
   result. Starting again creates a new run and CSV; it does not resume or
   overwrite a previous run.
9. Tap **Export latest CSV** and choose AirDrop, Save to Files, or another share
   destination. Older runs are under **Saved CSV files** and **Files → On My
   iPhone → Runtime Test → Results**. You can also copy the Results folder via
   **Finder → your iPhone → Files → Runtime Test**. Export before deleting the app.

For comparable measurements, use the same power conditions and a cool device,
and avoid Low Power Mode. No performance results are claimed until this runs on
your physical iPhone.

## What runs and what is timed

Each image uses its saved base point, paired 16-bit combined depth PNG and the
camera intrinsics encoded in the PNG's first four rows. Those are the inputs
that replace a live AR capture and the user's base tap.

1. Decode the RGB image to BGR and load depth/intrinsics with iForester's existing
   `Load/load.cpp` functions. All images are 1440 × 1920, with 192 × 256 depth.
2. Construct the same `Frame` and `CameraModel`.
3. Run the GitHub MobileSAM `BreastHeightLocalizer::localize` at **1.4 m**.
   Keep the original BH seed when it yields depth edges. Otherwise retry with
   depth segments that do not touch the image border; accept that alternative
   only if it yields a depth-edge pair. Use GitHub's minimum valid depth of
   **0.01 m**, rounded/clamped projected seed, and annotated-base fallback when
   the localizer returns no point, as in its MobileSAM batch runner.
4. Use the same depth-gradient edge detection, edge cleaning, growth direction,
   centroid and pixel-width helpers as `DBHMeasurer::measure`.
5. Rotate the image and generate the original box with explicit parameters
   **extension 0.15, top lift 300 px, base lift 150 px**.
6. Run the original `SAMBoxSegmentation` path. Inside `Sam`, time only the
   synchronous encoder and decoder ONNX `Session::Run` calls, separately,
   with `std::chrono::steady_clock`, then sum those durations.
   Every pass uses `needPreprocess=true`, so even repeating an image recomputes
   its embedding. CPU execution, one intra-op thread, graph optimization, direct
   1024 × 720 resize, RGB packing, box labels, mask ranking and threshold are the
   original inference implementation.
7. Apply the original inverse rotation, trunk/edge filtering, and breast-height
   check outside the timer. If the check requests a corrected box, preserve
   iForester's **one retry**, including a fresh encoder pass. Add only that retry's
   encoder and decoder inference durations, then repeat the original check.
8. Stop before diameter estimation. Save the CSV row and, for single mode,
   render the preview outside the timer.

**`segmentation_ms` is encoder ONNX inference time + decoder ONNX inference time,
summed over the first pass and any retry.** The clock surrounds only the two
synchronous `Session::Run` calls. OpenCV input/output resizing, RGB packing,
tensor and prompt preparation, embedding copies between calls, mask selection,
thresholding, model loading, breast-height finding, rotation, box preparation,
mask checks, rendering and CSV I/O are excluded. Only the total is exported.

The original ONNX graphs and their outputs are unchanged. Operations embedded
inside those graphs (such as encoder normalization/padding and decoder mask
upsampling) remain part of model inference; the excluded resizing and mask
selection are the operations performed by the surrounding app code.

Models stay loaded between images. There is no hidden warm-up or discarded
first run. Relaunch to measure a new session. Model initialization is excluded
even on the first image.

Batch processing requests a 200 ms asynchronous sleep after saving each image's
result and before starting the next image. There is no pause before the first
image or after the last image, and single-image runs have no added pause.
The actual gap may be longer if iOS delays resuming the task. This pause is
outside the encoder/decoder timers and adds no time to `segmentation_ms`.
The original retry within an image is unchanged. The UI remains responsive;
a stop requested during the pause prevents the next image from starting.
This pacing is an experiment to reduce sustained load, not a guarantee that the
phone will remain cool. Record the 200 ms setting when reporting these results.

The source's intermediate depth-plane/diameter calculations are omitted: they
do not feed the box and the requested app does not calculate DBH. This also
avoids blocking a segmentation benchmark on a diameter-only fitting failure.
All box-building and SAM helper implementations remain the copied originals.
The depth-edge, edge-cleaning, growth-direction and box-building functions match
the GitHub pipeline. Its localizer recovery fixes the missing-edge failures for
`03BB002E` and `50AAA4FD` without inserting arbitrary boxes or skipping those
images. The full bundled dataset has been checked through the encoder entry
point using the production C++ prompt code. A poor mask can still fail the
post-inference trunk/BH check; its mask and measured inference time are retained.

## CSV and preview interpretation

One CSV per single/batch run, one row per attempted image; each completed row is
flushed to storage immediately. Numeric times use a decimal point regardless
of the phone's locale. Both modes export exactly these two columns:

| Field | Meaning |
| --- | --- |
| `segmentation_ms` | Sum of encoder and decoder ONNX inference durations, including any retry; blank if neither model was invoked |
| `image_name` | Original RGB filename, including the `rgb-` prefix and `.jpeg` extension |

If an image fails before model inference, its time is blank rather than zero.
Failures during/after inference retain the attempted model time. Status and
diagnostic messages remain in the app UI and are not exported. Existing CSVs
from older app versions retain their previous format and timing definition;
new runs use two columns and encoder/decoder-only timing.

Single preview: the final blue box and a translucent green segmented region
mapped back into the original image. Base/breast-height markers stay hidden. The same base-point,
breast-height and segmentation calculations still run. Batch mode does not
render previews; the displayed selected image remains the single-mode preview.
No result images or DBH measurements are saved.

## Dataset and reproducibility

Source: [AI-assisted-DBH-measurement-pipeline data](https://github.com/yinzhiheng/AI-assisted-DBH-measurement-pipeline/tree/fc6e3c0fcea7b4664a0747390c080c795d571078/data).
The actual folder names use underscores: `rgb_naturalField`, `rgb_plottedField`,
`depth_naturalField`, `depth_plottedField`.

All **978 original data files / 849,912,078 bytes** are included: 487 JPEGs,
487 combined depth PNGs, two base-point CSVs and two label CSVs. Labels are kept
for provenance and are not used for segmentation or timing. Extra annotation
rows without RGB files do not create extra test images. Every original file is
verified against its Git blob hash at commit
`fc6e3c0fcea7b4664a0747390c080c795d571078`.

The natural-field CSV contains two different base points for
`1FFBD4EC-92E6-44C4-A134-A1E22CC4D504-240502113802`. The manifest deterministically
uses the **last CSV row, (664, 1576)**, rather than the earlier (658, 1552).
Both original rows are preserved; `manifest.json → duplicateAnnotations`
records the selection and the base points used by the app.

To restore missing/corrupt dataset files on the Mac, run from this folder:

```sh
python3 scripts/prepare_dataset.py
```

This resumes verified downloads from the pinned revision, checks all 978 files,
then writes the manifest. Rebuild/reinstall to transfer the recovered data to
the phone. It never downloads during a measured phone run.

## Source fidelity and local edits

`Vendor/SegmentAnything/sam_prompts.hpp`, both ONNX models,
and the original `DBHMeasurer.cpp` are byte-for-byte copies of iForester. The
device frameworks are the exact existing local dependencies. See
`Resources/source-provenance.json` and `validation/integrity.json`.

The copied core files have narrowly scoped changes:

- `DBHMeasurer.h` declares `segmentForRuntime` so the new entry point can call
  the original private helpers. Its implementation is in
  `Sources/RuntimePipeline.cpp` and stops before DBH calculation.
- `BreastHeightLocalizer.cpp/.h` now use the MobileSAM branch of the GitHub
  implementation at commit `fc6e3c0fcea7b4664a0747390c080c795d571078`, including
  its bounded-depth-segment recovery, default projected seed, depth validity
  checks and safe row-lambda capture. Only the iOS `PixelLoc_cpp` type name and
  placement of `Params` are adapted; unrelated SAM3 methods are omitted.
  Exact upstream files and hashes are in `Reference/GitHub/`. The adaptation is
  checked by `scripts/upstream_localizer.py`. `source-provenance.json` preserves
  the original iForester baseline, and validation still checks that the original
  app's files have not changed. `RuntimeBridge.mm` also matches the GitHub batch
  runner's annotated-base fallback when localization returns no point.
- `sam.cpp` wraps the active encoder and decoder `Session::Run` calls with timing
  instrumentation. `sam.hpp` exposes timing reset/read access; the new
  `sam_inference_timing.hpp` supplies the timer. Inference inputs, outputs and
  mask processing are unchanged. Integrity validation removes exactly these
  instrumentation additions before comparing against the original source
  hashes, and still verifies that the original iForester files are untouched.

The exact original ONNX models are:

```text
encoder: 62dd325618b7e49f0a5013059ebf3ee83ff82dafa81a7b83713b732692e15658
decoder: 318f77e3b0c100e9e070f04d1789a966a7f4ab8c74dc05f650e5a75e1b24f1f9
```

## Verification and project maintenance

Run the reproducible checks from this folder:

```sh
bash scripts/verify.sh
```

If your Xcode has another name/location, set `DEVELOPER_DIR` for that command.
Checks cover the actual copied prompt serializer, encoder/decoder timing
boundaries and reset/exception behavior, CSV durability/escaping and
failure-row blank times, an unsigned arm64 iPhone Release build, all dataset
hashes, PNG metadata/intrinsics, image/depth geometry, base-point bounds, source
fidelity, and the final bundled models/data. Logs are in `validation/`.

The additional dataset regression runs the production C++ prompt pipeline for
all 487 images and stops at a test encoder boundary. It requires macOS Python
with OpenCV 4.8 (matching the app), whose exported C++ symbols provide the host
OpenCV library. The test stub is never included in the iPhone target:

```sh
python scripts/check_dataset_prompts.py
```

To run `03BB002E` and `50AAA4FD` through the actual C++ pipeline and both ONNX
models, use `python scripts/check_dataset_prompts.py --real-model`. This also
requires a macOS ONNX Runtime 1.21.0 framework, supplied with
`--macos-runtime /path/to/onnxruntime.framework`. This optional desktop check is
not needed to build or run the iPhone app. It verifies nonempty masks, saves mask/preview images under
`validation/segmentation-regression/`, and does not export benchmark timings.
These are desktop functional regressions, not iPhone performance measurements.

An unsigned build confirms compilation/linking and resource packaging; signing,
installation, UI interaction and actual timing must still be verified on the
physical iPhone. No simulator or desktop runtime is presented as an iPhone
benchmark.

`scripts/generate_project.rb` can regenerate the checked-in Xcode project if
needed (requires the Ruby `xcodeproj` gem already available on this Mac).
Normal use only requires opening the existing project. Regeneration clears
the development team; select your team again in Xcode.
