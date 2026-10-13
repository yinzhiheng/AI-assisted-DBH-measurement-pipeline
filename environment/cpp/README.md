# C++ dependencies

This folder contains OpenCV 4.11.0 and ONNX Runtime 1.24.1 headers, import
libraries, runtime DLLs and their license/notice files.

All four projects import `Dependencies.props`. It resolves shared source under
`codes/DBH/src`, places executables under `codes/DBH/build/Release`, and copies
the required DLLs beside them. Global vcpkg integration is disabled so bundled
headers and binaries remain consistent.

Build `codes/DBH/DBHCalculationBatch.sln` with Visual Studio 2022, MSVC v143,
an installed Windows SDK, and Release/x64. C++17 and `/MD` are selected by the
projects. Standalone execution needs the Visual C++ v14 x64 runtime.

MobileSAMv2 reads `model/MobileSAMv2/onnx`. The SAM3 measurement branch reads
`output/SAM3SegResults/targetTreeMasks`. Both use normalized CSV labels and
require neither Python nor Excel at measurement time. Shared C++ code still
links ONNX Runtime for both branches.

See the [main README](../../README.md#2-install-and-check-the-environment) for setup, build
and run commands. When replacing a bundled dependency, use matching
headers and binaries and update the versioned filenames in `Dependencies.props`.
