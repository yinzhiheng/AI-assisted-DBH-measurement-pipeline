# MobileSAMv2 runtime test for iPhone

Build on a **Mac with Xcode 16.4 or newer** and run on a **physical iPhone
(iOS 15.6+)**. Use an Xcode version that supports your phone's iOS version.

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

The folder is approximately **1.4 GB**. 

## 2. Build and install with Xcode

1. Connect and unlock your iPhone, trust the Mac, and enable **Developer Mode**
   on the phone if requested.
2. Select the **RuntimeTest** scheme and your **iPhone** as the destination.
3. In **RuntimeTest target → Signing & Capabilities**, choose your Apple
   development team and enable automatic signing. Change the bundle identifier
   if Xcode requests a unique one.
4. Press **Run**. 
5. Launch the app **Runtime Test** from the phone's Home Screen. All images are available
   offline; the Mac can be disconnected.

## 3. Run and export

- **Select image → Segment selected:** process one image and show the inference time.
- **Batch all 487:** process all images with a **200 ms pause between images**.
- **Export latest CSV**, or use **Files → On My iPhone → Runtime Test → Results**.
