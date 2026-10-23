# MobileSAMv2 runtime test for iPhone

Build on a **Mac with Xcode 16.4 or newer** and run on a **physical iPhone
(iOS 15.6+)**. Use an Xcode version that supports your phone's iOS version.
The folder includes the Xcode project, C++/Swift sources, both ONNX models,
OpenCV/ONNX Runtime iOS libraries, and all **487 RGB/depth image pairs** with
base-point annotations. No Flutter, CocoaPods, or on-phone download is required.

## 1. Download on your Mac

Install [Git LFS](https://git-lfs.com/) first (`brew install git-lfs` if you use
Homebrew). Open Terminal and run:

```sh
git lfs install
git clone --depth 1 --filter=blob:none --sparse https://github.com/yinzhiheng/AI-assisted-DBH-measurement-pipeline.git
cd AI-assisted-DBH-measurement-pipeline
git sparse-checkout set codes/runTimeTestingCodesForiPhone
git lfs pull --include="codes/runTimeTestingCodesForiPhone/**"
cd codes/runTimeTestingCodesForiPhone
open RuntimeTest.xcodeproj
```

The folder is approximately **1.4 GB**. Git LFS downloads the large OpenCV
library; use the commands above rather than downloading individual files.

## 2. Build and install with Xcode

1. Connect and unlock your iPhone, trust the Mac, and enable **Developer Mode**
   on the phone if requested.
2. Select the **RuntimeTest** scheme and your **iPhone** as the destination.
3. In **RuntimeTest target → Signing & Capabilities**, choose your Apple
   development team and enable automatic signing. Change the bundle identifier
   if Xcode requests a unique one.
4. Press **Run**. The shared scheme uses **Release** with the debugger disabled.
   The first install transfers the bundled dataset; allow several GB of free
   phone storage. Simulator benchmarking is unsupported.
5. Launch **Runtime Test** from the phone's Home Screen. All images are available
   offline; the Mac can be disconnected.

## 3. Run and export

- **Select image → Segment selected:** process one image and show its box/mask.
- **Batch all 487:** process all images with a **200 ms pause between images**.
- Keep the app visible. **Stop after current image** preserves saved rows.
- **Export latest CSV**, or use **Files → On My iPhone → Runtime Test → Results**.

CSV columns are exactly `segmentation_ms,image_name`. Timing includes only
encoder and decoder inference, including any automatic retry. Prompt generation,
mask selection, CSV writing, and the 200 ms pause are excluded. Device heating
can affect results; keep test conditions consistent.

## Optional build verification

Select Xcode under **Xcode → Settings → Locations → Command Line Tools**, then:

```sh
bash scripts/verify.sh
```

This checks data/model integrity, tests timing/CSV logic, and builds an unsigned
iPhone Release app. For a custom Xcode installation, set `DEVELOPER_DIR` when
running the script. Actual performance must be measured on your iPhone.
See [DEVELOPMENT.md](DEVELOPMENT.md) for implementation and provenance details.
